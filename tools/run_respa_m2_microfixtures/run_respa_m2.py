from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.pcff_fixture_bridge.common import (  # noqa: E402
    ANGSTROM_TO_NM,
    build_typed_ir,
    parse_lammps_data,
    render_gromacs_topology,
)


DEFAULT_FIXTURES = ("coulomb_toy", "dense_oligomer")
DEFAULT_DT_VALUES = (0.0005, 0.00025)
DEFAULT_TOTAL_TIME_PS = 0.004
LEGACY_MTS_FACTOR = 2
EXACT_LEVEL2_FACTOR = 2
EXACT_LEVEL3_FACTOR = 4
OUTPUT_INTERVAL = 4
FOURIER_SPACING_NM = 0.08
FORCE_BOOKKEEPING_TOL = 5.0e-3
POTENTIAL_BOOKKEEPING_TOL = 5.0e-3
LEGACY_FORCE_GROUPS = "nonbonded longrange-nonbonded"
NUMERIC_FIELD_TOL = 1.0e-9
PME_LEGACY_SIDE_REFERENCE_MODE = "pme_legacy_side_reference"
ARCHIVED_M1_SCRIPT = (
    REPO_ROOT / "tools" / "run_respa_m1_microfixtures" / "run_respa_m1.py"
)
ARCHIVED_M1_SUMMARY = (
    REPO_ROOT / "tests" / "reference_results" / "r_respa_m1_microfixtures" / "summary.json"
)
DIAGNOSTIC_ENERGY_TERMS = (
    "LJ-14",
    "Coulomb-14",
    "Class2-Bond",
    "Class2-Angle",
    "Class2-Dih",
    "Coul.-recip.",
    "Coulomb-(SR)",
    "LJ-(SR)",
    "Potential",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exact 3-level R-RESPA M2 reconnection on validated PCFF microfixtures."
    )
    parser.add_argument(
        "--gmx-bin",
        default=str(REPO_ROOT / "build" / "bin" / "gmx"),
        help="Path to the GROMACS gmx binary.",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        dest="fixtures",
        help="Fixture id to run. Repeat to select multiple fixtures. Default: coulomb_toy, dense_oligomer.",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "tests" / "reference_results" / "r_respa_m2_microfixtures"),
        help="Output directory for run artifacts.",
    )
    parser.add_argument(
        "--total-time-ps",
        type=float,
        default=DEFAULT_TOTAL_TIME_PS,
        help="Physical trajectory length to compare at each timestep.",
    )
    parser.add_argument(
        "--milestone-name",
        default="R-RESPA M2",
        help="Milestone label to persist into the output summary.",
    )
    parser.add_argument(
        "--dense-bookkeeping-isolation",
        action="store_true",
        help="Collect extra dense_oligomer step-0 bookkeeping diagnostics for M2b-style isolation.",
    )
    parser.add_argument(
        "--dense-force-ownership-isolation",
        action="store_true",
        help="Collect dense_oligomer step-0 force-buffer ownership diagnostics for M2c-style isolation.",
    )
    parser.add_argument(
        "--dense-merge-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 merge-stage diagnostics for M2d-style localization.",
    )
    parser.add_argument(
        "--dense-early-accumulation-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 early outer-accumulation diagnostics for M2e-style localization.",
    )
    parser.add_argument(
        "--exact-pair-write-ownership-proof",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 exact pair-write ownership diagnostics for M2f-style proof.",
    )
    return parser.parse_args()


def run_command(
    cmd: list[str],
    cwd: Path,
    label: str,
    commands_log: list[str],
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env_prefix = ""
    if extra_env:
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(extra_env.items())) + " "
    commands_log.append(f"(cd {shlex.quote(str(cwd))} && {env_prefix}{' '.join(shlex.quote(part) for part in cmd)})")
    env = None
    if extra_env:
        env = dict(os.environ)
        env.update(extra_env)
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True, errors="replace", env=env)
    (cwd / f"{label}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (cwd / f"{label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    return result


def molecule_local_indices(data: dict[str, Any]) -> dict[int, int]:
    local_indices: dict[int, int] = {}
    current_counts: dict[int, int] = {}
    for atom in data["atoms"]:
        molecule_id = atom["molecule_id"]
        current_counts[molecule_id] = current_counts.get(molecule_id, 0) + 1
        local_indices[atom["id"]] = current_counts[molecule_id]
    return local_indices


def create_gro_from_lammps(lammps_data_path: Path, gro_path: Path) -> dict[str, Any]:
    data = parse_lammps_data(lammps_data_path)
    box_x = (data["box"]["x"]["hi"] - data["box"]["x"]["lo"]) * ANGSTROM_TO_NM
    box_y = (data["box"]["y"]["hi"] - data["box"]["y"]["lo"]) * ANGSTROM_TO_NM
    box_z = (data["box"]["z"]["hi"] - data["box"]["z"]["lo"]) * ANGSTROM_TO_NM
    local_indices = molecule_local_indices(data)

    with gro_path.open("w", encoding="utf-8") as handle:
        handle.write("Generated from LAMMPS data with high precision\n")
        handle.write(f"{len(data['atoms']):5d}\n")
        for atom in data["atoms"]:
            x = atom["x_angstrom"] * ANGSTROM_TO_NM
            y = atom["y_angstrom"] * ANGSTROM_TO_NM
            z = atom["z_angstrom"] * ANGSTROM_TO_NM
            residue_number = atom["molecule_id"] % 100000
            atom_name = f"A{local_indices[atom['id']]}"
            handle.write(
                f"{residue_number:5d}{'MOL':<5s}{atom_name:>5s}{atom['id']:5d}"
                f"{x:15.7f}{y:15.7f}{z:15.7f}\n"
            )
        handle.write(f"{box_x:15.7f}{box_y:15.7f}{box_z:15.7f}\n")

    return {
        "num_atoms": len(data["atoms"]),
        "box_nm": [box_x, box_y, box_z],
    }


def section_line_counts(topology_text: str) -> dict[str, int]:
    counts = {"pairs": 0, "bonds": 0, "angles": 0, "dihedrals": 0, "impropers": 0}
    current_section = None
    for raw_line in topology_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line.strip("[]").strip().lower()
            continue
        if current_section in counts:
            counts[current_section] += 1
    return counts


def inner_terms_from_topology(topology_text: str) -> list[str]:
    counts = section_line_counts(topology_text)
    ordered_terms = [
        ("pairs", "pair14"),
        ("bonds", "bond"),
        ("angles", "angle"),
        ("dihedrals", "dihedral"),
        ("impropers", "improper"),
    ]
    return [label for section, label in ordered_terms if counts[section] > 0]


def common_mdp_lines(dt_ps: float, total_time_ps: float, integrator: str) -> tuple[list[str], int]:
    nsteps = int(round(total_time_ps / dt_ps))
    if nsteps <= 0 or nsteps % OUTPUT_INTERVAL != 0:
        raise ValueError(f"nsteps={nsteps} must be positive and divisible by {OUTPUT_INTERVAL}")

    lines = [
        f"integrator               = {integrator}",
        f"dt                       = {dt_ps:.6f}",
        f"nsteps                   = {nsteps}",
        "constraints              = none",
        "cutoff-scheme            = Verlet",
        f"nstlist                  = {OUTPUT_INTERVAL}",
        "rlist                    = 0.99",
        "rvdw                     = 0.9",
        "rcoulomb                 = 0.9",
        "vdwtype                  = Cut-off",
        "vdw-modifier             = none",
        "coulombtype              = PME",
        "coulomb-modifier         = none",
        "ewald-rtol               = 1e-6",
        "pme-order                = 4",
        f"fourierspacing           = {FOURIER_SPACING_NM}",
        "epsilon-r                = 1",
        "pbc                      = xyz",
        "tcoupl                   = no",
        "pcoupl                   = no",
        "comm-mode                = none",
        "verlet-buffer-tolerance  = -1",
        "gen-vel                  = no",
        f"nstcalcenergy            = {OUTPUT_INTERVAL}",
        f"nstenergy                = {OUTPUT_INTERVAL}",
        f"nstlog                   = {OUTPUT_INTERVAL}",
        f"nstxout                  = {OUTPUT_INTERVAL}",
        f"nstvout                  = {OUTPUT_INTERVAL}",
        f"nstfout                  = {OUTPUT_INTERVAL}",
        "nstxout-compressed       = 0",
    ]
    return lines, nsteps


def write_plain_mdp(path: Path, dt_ps: float, total_time_ps: float) -> int:
    lines, nsteps = common_mdp_lines(dt_ps, total_time_ps, "md-vv")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return nsteps


def write_legacy_mdp(path: Path, dt_ps: float, total_time_ps: float) -> int:
    lines, nsteps = common_mdp_lines(dt_ps, total_time_ps, "md")
    lines.extend(
        [
            "mts                      = yes",
            "mts-mode                 = legacy",
            "mts-levels               = 2",
            f"mts-level2-forces        = {LEGACY_FORCE_GROUPS}",
            f"mts-level2-factor        = {LEGACY_MTS_FACTOR}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return nsteps


def write_exact_mdp(path: Path, dt_ps: float, total_time_ps: float) -> int:
    lines, nsteps = common_mdp_lines(dt_ps, total_time_ps, "md-vv")
    lines.extend(
        [
            "mts                      = yes",
            "mts-mode                 = lammps-respa",
            "mts-levels               = 3",
            f"mts-level2-factor        = {EXACT_LEVEL2_FACTOR}",
            f"mts-level3-factor        = {EXACT_LEVEL3_FACTOR}",
            "mts-respa-bond-level     = 1",
            "mts-respa-angle-level    = 1",
            "mts-respa-dihedral-level = 1",
            "mts-respa-improper-level = 1",
            "mts-respa-pair14-level   = 1",
            "mts-respa-kspace-level   = 3",
            "mts-respa-inner-level    = 1",
            "mts-respa-middle-level   = 2",
            "mts-respa-outer-level    = 3",
            "mts-respa-inner-off      = 0.30",
            "mts-respa-inner-on       = 0.45",
            "mts-respa-outer-on       = 0.60",
            "mts-respa-outer-off      = 0.80",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return nsteps


def parse_tpr_mts_fields(dump_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in dump_text.splitlines():
        if "=" not in line:
            continue
        stripped = line.strip()
        if stripped.startswith("mts"):
            key, value = [part.strip() for part in stripped.split("=", 1)]
            fields[key] = value
    return fields


def parse_trr_dump(dump_text: str) -> dict[int, dict[str, dict[int, tuple[float, float, float]]]]:
    frames: dict[int, dict[str, dict[int, tuple[float, float, float]]]] = {}
    current_step = None
    for line in dump_text.splitlines():
        if "natoms=" in line and "step=" in line:
            match = re.search(r"step=\s*(\d+)", line)
            if match is None:
                continue
            current_step = int(match.group(1))
            frames[current_step] = {"x": {}, "v": {}, "f": {}}
            continue
        match = re.match(
            r"\s*([xvf])\[\s*(\d+)\]=\{\s*([^,]+),\s*([^,]+),\s*([^}]+)\}",
            line,
        )
        if match is None or current_step is None:
            continue
        field = match.group(1)
        index = int(match.group(2))
        vector = tuple(float(match.group(i)) for i in range(3, 6))
        frames[current_step][field][index] = vector
    return frames


def parse_force_dump(path: Path) -> dict[int, dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        columns = stripped.split("\t")
        if len(columns) != 7:
            raise ValueError(f"Unexpected force dump line in {path}: {stripped}")
        step = int(columns[0])
        time_ps = float(columns[1])
        highest_active_level = int(columns[2])
        atom = int(columns[3])
        frame = frames.setdefault(
            step,
            {
                "time_ps": time_ps,
                "highest_active_level": highest_active_level,
                "forces": {},
            },
        )
        frame["forces"][atom] = (float(columns[4]), float(columns[5]), float(columns[6]))
    return frames


def parse_vector_dump(path: Path) -> dict[str, Any]:
    metadata: dict[str, str] = {}
    forces: dict[int, tuple[float, float, float]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            for token in stripped[1:].split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    metadata[key] = value
            continue
        atom, fx, fy, fz = stripped.split("\t")
        forces[int(atom)] = (float(fx), float(fy), float(fz))
    return {"metadata": metadata, "forces": forces}


def parse_key_value_text(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        row: dict[str, str] = {}
        for token in stripped.split():
            if "=" in token:
                key, value = token.split("=", 1)
                row[key] = value
        if row:
            rows.append(row)
    return rows


def expand_sparse_vector(
    sparse: dict[int, tuple[float, float, float]],
    template: dict[int, tuple[float, float, float]],
) -> dict[int, tuple[float, float, float]]:
    expanded = {atom_index: (0.0, 0.0, 0.0) for atom_index in sorted(template)}
    for atom_index, vector in sparse.items():
        expanded[atom_index] = vector
    return expanded


def add_pair_source(
    mapping: dict[tuple[int, int], list[str]],
    atom_i_1based: int,
    atom_j_1based: int,
    label: str,
) -> None:
    key = tuple(sorted((atom_i_1based - 1, atom_j_1based - 1)))
    mapping.setdefault(key, [])
    if label not in mapping[key]:
        mapping[key].append(label)


def parse_topology_pair_sources(topology_path: Path) -> dict[tuple[int, int], list[str]]:
    pair_sources: dict[tuple[int, int], list[str]] = {}
    current_section = None
    for raw_line in topology_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line.strip("[]").strip().lower()
            continue
        fields = line.split()
        if current_section == "bonds" and len(fields) >= 2:
            add_pair_source(pair_sources, int(fields[0]), int(fields[1]), "bond")
        elif current_section == "pairs" and len(fields) >= 2:
            add_pair_source(pair_sources, int(fields[0]), int(fields[1]), "pair14")
        elif current_section == "angles" and len(fields) >= 3:
            add_pair_source(pair_sources, int(fields[0]), int(fields[2]), "angle_endpoint")
        elif current_section == "dihedrals" and len(fields) >= 4:
            add_pair_source(pair_sources, int(fields[0]), int(fields[3]), "dihedral_endpoint")
    return pair_sources


def parse_excluded_force_dump(path: Path) -> dict[str, Any]:
    return parse_vector_dump(path)


def read_potential_xvg(path: Path) -> list[dict[str, float]]:
    series = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", "@")):
            continue
        time_value, potential_value = stripped.split()[:2]
        series.append({"time_ps": float(time_value), "potential_kj_per_mol": float(potential_value)})
    return series


def maybe_minimum_image(diff: float, box_length: float | None) -> float:
    if box_length is None or box_length <= 0:
        return diff
    return diff - round(diff / box_length) * box_length


def vector_metrics(
    left: dict[int, tuple[float, float, float]],
    right: dict[int, tuple[float, float, float]],
    box_lengths: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    sum_sq = 0.0
    max_abs = 0.0
    count = 0
    for atom_index in sorted(left):
        for component_index, (left_value, right_value) in enumerate(zip(left[atom_index], right[atom_index])):
            box_length = None if box_lengths is None else box_lengths[component_index]
            diff = maybe_minimum_image(left_value - right_value, box_length)
            sum_sq += diff * diff
            max_abs = max(max_abs, abs(diff))
            count += 1
    rms = math.sqrt(sum_sq / max(count, 1))
    return {
        "l2_norm": math.sqrt(sum_sq),
        "rms": rms,
        "max_abs": max_abs,
    }


def subtract_vectors(
    left: dict[int, tuple[float, float, float]],
    right: dict[int, tuple[float, float, float]],
) -> dict[int, tuple[float, float, float]]:
    return {
        atom_index: tuple(left_value - right_value for left_value, right_value in zip(left[atom_index], right[atom_index]))
        for atom_index in sorted(left)
    }


def add_vectors(
    left: dict[int, tuple[float, float, float]],
    right: dict[int, tuple[float, float, float]],
) -> dict[int, tuple[float, float, float]]:
    return {
        atom_index: tuple(left_value + right_value for left_value, right_value in zip(left[atom_index], right[atom_index]))
        for atom_index in sorted(left)
    }


def sum_vectors(*vectors: dict[int, tuple[float, float, float]]) -> dict[int, tuple[float, float, float]]:
    atom_indices = sorted(vectors[0])
    return {
        atom_index: tuple(sum(vector[atom_index][dim] for vector in vectors) for dim in range(3))
        for atom_index in atom_indices
    }


def vector_alignment(
    left: dict[int, tuple[float, float, float]],
    right: dict[int, tuple[float, float, float]],
) -> float | None:
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for atom_index in sorted(left):
        for left_value, right_value in zip(left[atom_index], right[atom_index]):
            dot += left_value * right_value
            left_sq += left_value * left_value
            right_sq += right_value * right_value
    if left_sq == 0.0 or right_sq == 0.0:
        return None
    return dot / math.sqrt(left_sq * right_sq)


def extract_potential_series(
    gmx_bin: str, work_dir: Path, deffnm: str, commands_log: list[str], label_prefix: str
) -> list[dict[str, float]]:
    process = subprocess.run(
        [gmx_bin, "energy", "-f", f"{deffnm}.edr", "-o", f"{deffnm}_potential.xvg", "-xvg", "none"],
        cwd=work_dir,
        input="Potential\n0\n",
        capture_output=True,
        text=True,
        check=True,
        errors="replace",
    )
    commands_log.append(
        f"(cd {shlex.quote(str(work_dir))} && {' '.join(shlex.quote(part) for part in [gmx_bin, 'energy', '-f', f'{deffnm}.edr', '-o', f'{deffnm}_potential.xvg', '-xvg', 'none'])} <<'EOF'\nPotential\n0\nEOF)"
    )
    (work_dir / f"{label_prefix}_energy_select.stdout.txt").write_text(process.stdout, encoding="utf-8")
    (work_dir / f"{label_prefix}_energy_select.stderr.txt").write_text(process.stderr, encoding="utf-8")

    series = []
    for raw_line in (work_dir / f"{deffnm}_potential.xvg").read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", "@")):
            continue
        time_value, potential_value = stripped.split()[:2]
        series.append({"time_ps": float(time_value), "potential_kj_per_mol": float(potential_value)})
    return series


def extract_named_energy_series(
    gmx_bin: str,
    work_dir: Path,
    deffnm: str,
    term_names: tuple[str, ...],
    output_stem: str,
    commands_log: list[str],
    label_prefix: str,
) -> list[dict[str, float]]:
    process = subprocess.run(
        [gmx_bin, "energy", "-f", f"{deffnm}.edr", "-o", f"{output_stem}.xvg", "-xvg", "none"],
        cwd=work_dir,
        input="".join(f"{term}\n" for term in term_names) + "0\n",
        capture_output=True,
        text=True,
        check=True,
        errors="replace",
    )
    commands_log.append(
        f"(cd {shlex.quote(str(work_dir))} && {' '.join(shlex.quote(part) for part in [gmx_bin, 'energy', '-f', f'{deffnm}.edr', '-o', f'{output_stem}.xvg', '-xvg', 'none'])} <<'EOF'\n"
        + "\n".join(term_names)
        + "\n0\nEOF)"
    )
    (work_dir / f"{label_prefix}_energy_terms.stdout.txt").write_text(process.stdout, encoding="utf-8")
    (work_dir / f"{label_prefix}_energy_terms.stderr.txt").write_text(process.stderr, encoding="utf-8")

    rows = []
    for raw_line in (work_dir / f"{output_stem}.xvg").read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", "@")):
            continue
        values = [float(token) for token in stripped.split()]
        row = {"time_ps": values[0]}
        for index, term_name in enumerate(term_names, start=1):
            row[term_name] = values[index]
        rows.append(row)
    return rows


def parse_debug_stats(stderr_text: str) -> dict[str, Any]:
    overlap_pattern = re.compile(
        r"GMX_PCFF_RESPA_DEBUG listedPairOverlaps pairs=(?P<pairs>-?\d+) excludedPairs=(?P<excluded>-?\d+) duplicatePairs=(?P<duplicates>-?\d+) duplicateExcludedPairs=(?P<duplicate_excluded>-?\d+)"
    )
    stats_pattern = re.compile(
        r"GMX_PCFF_RESPA_DEBUG (?P<label>pairs|excludedPairs) count=(?P<count>-?\d+) lj=\s*(?P<lj>[-+0-9.eE]+) coul=\s*(?P<coul>[-+0-9.eE]+) qq=\s*(?P<qq>[-+0-9.eE]+) self=\s*(?P<self>[-+0-9.eE]+)"
    )

    overlaps = []
    stats: dict[str, list[dict[str, float]]] = {"pairs": [], "excludedPairs": []}
    for line in stderr_text.splitlines():
        overlap_match = overlap_pattern.search(line)
        if overlap_match is not None:
            overlaps.append(
                {
                    "pairs": int(overlap_match.group("pairs")),
                    "excluded_pairs": int(overlap_match.group("excluded")),
                    "duplicate_pairs": int(overlap_match.group("duplicates")),
                    "duplicate_excluded_pairs": int(overlap_match.group("duplicate_excluded")),
                }
            )
            continue
        stats_match = stats_pattern.search(line)
        if stats_match is not None:
            stats[stats_match.group("label")].append(
                {
                    "count": int(stats_match.group("count")),
                    "lj_kj_per_mol": float(stats_match.group("lj")),
                    "coulomb_kj_per_mol": float(stats_match.group("coul")),
                    "qq_sum": float(stats_match.group("qq")),
                    "self_kj_per_mol": float(stats_match.group("self")),
                }
            )

    return {
        "first_overlap": overlaps[0] if overlaps else None,
        "pairs": stats["pairs"],
        "excluded_pairs": stats["excludedPairs"],
        "step0_pairs": stats["pairs"][0] if stats["pairs"] else None,
        "step0_excluded_pairs": stats["excludedPairs"][0] if stats["excludedPairs"] else None,
    }


def archived_m1_continuity_record() -> dict[str, Any]:
    return {
        "archived_m1_summary": str(ARCHIVED_M1_SUMMARY),
        "archived_m1_script": str(ARCHIVED_M1_SCRIPT),
        "direct_archived_m1_comparison_done": False,
        "direct_archived_m1_comparison_feasible": False,
        "why_not_feasible": (
            "Archived M1 uses integrator = md with coulombtype = Cut-off and mts-level2-forces = nonbonded; "
            "the current exact-3-level harness requires PME-side md-vv settings, so the current simpler split can only be a PME-side legacy side-reference."
        ),
        "archived_m1_settings": {
            "integrator": "md",
            "coulombtype": "Cut-off",
            "mts_mode": "legacy",
            "mts_levels": 2,
            "mts_level2_forces": "nonbonded",
        },
        "current_side_reference_settings": {
            "label": PME_LEGACY_SIDE_REFERENCE_MODE,
            "integrator": "md",
            "coulombtype": "PME",
            "mts_mode": "legacy",
            "mts_levels": 2,
            "mts_level2_forces": LEGACY_FORCE_GROUPS,
        },
        "wording_requirement": (
            "The current simpler split must be reported as a PME-side legacy side-reference, not as direct archived-M1 continuity."
        ),
    }


def dense_bookkeeping_isolation(
    gmx_bin: str,
    dense_fixture_summary: dict[str, Any],
    commands_log: list[str],
) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    exact_dir = Path(coarse["exact_work_dir"])
    side_ref_dir = Path(coarse["side_reference_work_dir"])

    plain_terms = extract_named_energy_series(
        gmx_bin, plain_dir, "plain", DIAGNOSTIC_ENERGY_TERMS, "plain_terms", commands_log, "plain"
    )[0]
    side_ref_terms = extract_named_energy_series(
        gmx_bin,
        side_ref_dir,
        "legacy",
        DIAGNOSTIC_ENERGY_TERMS,
        "side_reference_terms",
        commands_log,
        "side_reference",
    )[0]
    exact_terms = extract_named_energy_series(
        gmx_bin, exact_dir, "exact", DIAGNOSTIC_ENERGY_TERMS, "exact_terms", commands_log, "exact"
    )[0]
    debug_stats = parse_debug_stats((exact_dir / "exact_mdrun.stderr.txt").read_text(encoding="utf-8"))

    term_deltas: dict[str, dict[str, float]] = {}
    for term_name in DIAGNOSTIC_ENERGY_TERMS:
        if term_name == "time_ps":
            continue
        term_deltas[term_name] = {
            "plain_kj_per_mol": plain_terms[term_name],
            "side_reference_kj_per_mol": side_ref_terms[term_name],
            "exact_kj_per_mol": exact_terms[term_name],
            "exact_minus_plain_kj_per_mol": exact_terms[term_name] - plain_terms[term_name],
            "side_reference_minus_plain_kj_per_mol": side_ref_terms[term_name] - plain_terms[term_name],
        }

    step0_pairs = debug_stats["step0_pairs"]
    step0_excluded = debug_stats["step0_excluded_pairs"]
    coulomb_delta = term_deltas["Coulomb-(SR)"]["exact_minus_plain_kj_per_mol"]
    excluded_coulomb = None if step0_excluded is None else step0_excluded["coulomb_kj_per_mol"]
    localized_to_excluded = excluded_coulomb is not None and abs(coulomb_delta - excluded_coulomb) <= 2.0e-2

    localization = {
        "classification": (
            "mis-owned excluded-pair Coulomb correction contribution"
            if localized_to_excluded
            else "still unresolved"
        ),
        "supports_missing_term": False,
        "supports_duplicated_term": False,
        "supports_wrong_level_ownership": localized_to_excluded,
        "supports_wrong_reference_side_accounting": False,
        "step0_force_diff_l2_norm": coarse["exact_vs_plain"]["step0_force_diff"]["l2_norm"],
        "step0_force_diff_max_abs": coarse["exact_vs_plain"]["step0_force_diff"]["max_abs"],
        "step0_potential_abs_diff_kj_per_mol": coarse["exact_vs_plain"]["step0_potential_abs_diff_kj_per_mol"],
        "coulomb_sr_exact_minus_plain_kj_per_mol": coulomb_delta,
        "excluded_pairs_coulomb_debug_kj_per_mol": excluded_coulomb,
        "exact_coulomb_sr_reconstructed_from_debug_kj_per_mol": (
            None
            if step0_pairs is None or step0_excluded is None
            else step0_pairs["coulomb_kj_per_mol"] + step0_excluded["coulomb_kj_per_mol"] + step0_pairs["self_kj_per_mol"]
        ),
        "why_not_fully_closed": (
            "The step-0 energy defect localizes cleanly to excluded-pair Coulomb correction accounting, but this artifact alone does not prove whether the large step-0 force mismatch comes from the same ownership bug or from a separate force-buffer issue."
        ),
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "plain_terms_step0": plain_terms,
        "side_reference_terms_step0": side_ref_terms,
        "exact_terms_step0": exact_terms,
        "term_deltas_step0": term_deltas,
        "debug_stats": debug_stats,
        "localization": localization,
    }


def dense_force_ownership_isolation(dense_fixture_summary: dict[str, Any]) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    exact_dir = Path(coarse["exact_work_dir"])
    side_ref_dir = Path(coarse["side_reference_work_dir"])

    plain_force_frames = parse_force_dump(plain_dir / "plain_total_force.tsv")
    exact_force_frames = parse_force_dump(exact_dir / "exact_total_force.tsv")
    side_reference_force_frames = parse_trr_dump(
        (side_ref_dir / "legacy_trr_dump.stdout.txt").read_text(encoding="utf-8")
    )
    excluded_force_dump = parse_excluded_force_dump(exact_dir / "exact_excluded_pairs_correction_force.tsv")

    step0 = min(set(plain_force_frames) & set(exact_force_frames) & set(side_reference_force_frames))
    plain_force = plain_force_frames[step0]["forces"]
    exact_force = exact_force_frames[step0]["forces"]
    side_reference_force = side_reference_force_frames[step0]["f"]
    excluded_force = excluded_force_dump["forces"]

    exact_minus_plain = subtract_vectors(exact_force, plain_force)
    plain_minus_exact = subtract_vectors(plain_force, exact_force)
    exact_minus_excluded = subtract_vectors(exact_force, excluded_force)
    exact_plus_excluded = add_vectors(exact_force, excluded_force)

    exact_vs_plain = vector_metrics(exact_force, plain_force)
    side_reference_vs_plain = vector_metrics(side_reference_force, plain_force)
    excluded_vs_exact_minus_plain = vector_metrics(excluded_force, exact_minus_plain)
    excluded_vs_plain_minus_exact = vector_metrics(excluded_force, plain_minus_exact)
    exact_minus_excluded_vs_plain = vector_metrics(exact_minus_excluded, plain_force)
    exact_plus_excluded_vs_plain = vector_metrics(exact_plus_excluded, plain_force)

    baseline_l2 = max(exact_vs_plain["l2_norm"], 1.0)
    subtract_ratio = exact_minus_excluded_vs_plain["l2_norm"] / baseline_l2
    add_ratio = exact_plus_excluded_vs_plain["l2_norm"] / baseline_l2
    alignment_with_exact_minus_plain = vector_alignment(excluded_force, exact_minus_plain)
    alignment_with_plain_minus_exact = vector_alignment(excluded_force, plain_minus_exact)

    supports_duplicated = (
        subtract_ratio <= 1.0e-4
        and exact_minus_excluded_vs_plain["max_abs"] <= FORCE_BOOKKEEPING_TOL
        and alignment_with_exact_minus_plain is not None
        and alignment_with_exact_minus_plain >= 0.9999
    )
    supports_missing = (
        add_ratio <= 1.0e-4
        and exact_plus_excluded_vs_plain["max_abs"] <= FORCE_BOOKKEEPING_TOL
        and alignment_with_plain_minus_exact is not None
        and alignment_with_plain_minus_exact >= 0.9999
    )

    localization = {
        "classification": (
            "duplicated excluded-pair Coulomb correction force contribution"
            if supports_duplicated
            else (
                "missing excluded-pair Coulomb correction force contribution"
                if supports_missing
                else "still unresolved"
            )
        ),
        "supports_missing_term": supports_missing,
        "supports_duplicated_term": supports_duplicated,
        "supports_wrong_buffer_level_ownership": False,
        "exact_step0_total_force_path": str(exact_dir / "exact_total_force.tsv"),
        "plain_step0_total_force_path": str(plain_dir / "plain_total_force.tsv"),
        "side_reference_step0_total_force_path": str(side_ref_dir / "legacy_total_force.tsv"),
        "excluded_correction_force_path": str(exact_dir / "exact_excluded_pairs_correction_force.tsv"),
        "step0": step0,
        "exact_vs_plain_step0_force_diff": exact_vs_plain,
        "side_reference_vs_plain_step0_force_diff": side_reference_vs_plain,
        "excluded_vs_exact_minus_plain_step0_force_diff": excluded_vs_exact_minus_plain,
        "excluded_vs_plain_minus_exact_step0_force_diff": excluded_vs_plain_minus_exact,
        "exact_minus_excluded_vs_plain_step0_force_diff": exact_minus_excluded_vs_plain,
        "exact_plus_excluded_vs_plain_step0_force_diff": exact_plus_excluded_vs_plain,
        "alignment_with_exact_minus_plain": alignment_with_exact_minus_plain,
        "alignment_with_plain_minus_exact": alignment_with_plain_minus_exact,
        "excluded_force_dump_metadata": excluded_force_dump["metadata"],
        "why_not_fully_closed": (
            "The excluded-pair Coulomb correction force dump localizes the correction vector directly, "
            "but buffer ownership should only be called closed if subtracting or adding that vector collapses "
            "the dense step-0 exact-vs-plain force mismatch."
        ),
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "localization": localization,
    }


def dense_merge_trace_localization(dense_fixture_summary: dict[str, Any]) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    exact_dir = Path(coarse["exact_work_dir"])

    plain_force_frames = parse_force_dump(plain_dir / "plain_total_force.tsv")
    exact_force_frames = parse_force_dump(exact_dir / "exact_total_force.tsv")
    excluded_force_dump = parse_vector_dump(exact_dir / "exact_excluded_pairs_correction_force.tsv")
    level0_post = parse_vector_dump(exact_dir / "step0_level0_post_postprocess_shift.tsv")
    level1_post = parse_vector_dump(exact_dir / "step0_level1_post_postprocess_shift.tsv")
    level2_pre_virial = parse_vector_dump(exact_dir / "step0_level2_pre_postprocess_virial.tsv")
    level2_post_shift = parse_vector_dump(exact_dir / "step0_level2_post_postprocess_shift.tsv")
    physical_postcombine = parse_vector_dump(exact_dir / "step0_physical_postcombine.tsv")
    impulse_postcombine = parse_vector_dump(exact_dir / "step0_impulse_postcombine.tsv")

    step0 = min(set(plain_force_frames) & set(exact_force_frames))
    plain_force = plain_force_frames[step0]["forces"]
    exact_force = exact_force_frames[step0]["forces"]
    excluded_force = excluded_force_dump["forces"]
    level0_post_force = level0_post["forces"]
    level1_post_force = level1_post["forces"]
    level2_pre_virial_force = level2_pre_virial["forces"]
    level2_post_shift_force = level2_post_shift["forces"]
    physical_postcombine_force = physical_postcombine["forces"]
    impulse_postcombine_force = impulse_postcombine["forces"]

    reconstructed_physical = sum_vectors(level0_post_force, level1_post_force, level2_post_shift_force)
    reconstructed_without_excluded = sum_vectors(
        level0_post_force, level1_post_force, subtract_vectors(level2_post_shift_force, excluded_force)
    )
    postprocess_delta = subtract_vectors(level2_post_shift_force, level2_pre_virial_force)
    combine_delta = subtract_vectors(physical_postcombine_force, reconstructed_physical)

    outer_post_vs_pre = vector_metrics(level2_post_shift_force, level2_pre_virial_force)
    combine_reconstruction = vector_metrics(physical_postcombine_force, reconstructed_physical)
    corrected_reconstruction_vs_plain = vector_metrics(reconstructed_without_excluded, plain_force)
    postprocess_delta_vs_excluded = vector_metrics(postprocess_delta, excluded_force)
    combine_delta_vs_excluded = vector_metrics(combine_delta, excluded_force)
    physical_postcombine_vs_exact = vector_metrics(physical_postcombine_force, exact_force)
    impulse_postcombine_vs_exact = vector_metrics(impulse_postcombine_force, exact_force)

    supports_before_postprocess = (
        outer_post_vs_pre["max_abs"] <= FORCE_BOOKKEEPING_TOL
        and combine_reconstruction["max_abs"] <= FORCE_BOOKKEEPING_TOL
        and corrected_reconstruction_vs_plain["max_abs"] <= FORCE_BOOKKEEPING_TOL
    )
    supports_postprocess = (
        postprocess_delta_vs_excluded["max_abs"] <= FORCE_BOOKKEEPING_TOL
        and combine_reconstruction["max_abs"] <= FORCE_BOOKKEEPING_TOL
    )
    supports_combine = combine_delta_vs_excluded["max_abs"] <= FORCE_BOOKKEEPING_TOL

    localization = {
        "classification": (
            "duplicated correction already present before postProcessForces"
            if supports_before_postprocess
            else (
                "duplication first appears during postProcessForces"
                if supports_postprocess
                else (
                    "duplication first appears during combineMtsForces"
                    if supports_combine
                    else "still unresolved"
                )
            )
        ),
        "step0": step0,
        "exact_vs_plain_step0_force_diff": vector_metrics(exact_force, plain_force),
        "outer_post_vs_pre_postprocess_diff": outer_post_vs_pre,
        "combine_reconstruction_diff": combine_reconstruction,
        "corrected_reconstruction_vs_plain_diff": corrected_reconstruction_vs_plain,
        "postprocess_delta_vs_excluded_diff": postprocess_delta_vs_excluded,
        "combine_delta_vs_excluded_diff": combine_delta_vs_excluded,
        "physical_postcombine_vs_exact_diff": physical_postcombine_vs_exact,
        "impulse_postcombine_vs_exact_diff": impulse_postcombine_vs_exact,
        "alignment_postprocess_delta_with_excluded": vector_alignment(postprocess_delta, excluded_force),
        "alignment_combine_delta_with_excluded": vector_alignment(combine_delta, excluded_force),
        "supports_before_postprocess": supports_before_postprocess,
        "supports_postprocess": supports_postprocess,
        "supports_combine": supports_combine,
        "excluded_force_dump_metadata": excluded_force_dump["metadata"],
        "level2_pre_postprocess_virial_metadata": level2_pre_virial["metadata"],
        "level2_post_postprocess_shift_metadata": level2_post_shift["metadata"],
        "physical_postcombine_metadata": physical_postcombine["metadata"],
        "impulse_postcombine_metadata": impulse_postcombine["metadata"],
        "level0_post_postprocess_shift_path": str(exact_dir / "step0_level0_post_postprocess_shift.tsv"),
        "level1_post_postprocess_shift_path": str(exact_dir / "step0_level1_post_postprocess_shift.tsv"),
        "level2_pre_postprocess_virial_path": str(exact_dir / "step0_level2_pre_postprocess_virial.tsv"),
        "level2_post_postprocess_shift_path": str(exact_dir / "step0_level2_post_postprocess_shift.tsv"),
        "physical_postcombine_path": str(exact_dir / "step0_physical_postcombine.tsv"),
        "impulse_postcombine_path": str(exact_dir / "step0_impulse_postcombine.tsv"),
        "why_not_fully_closed": (
            "This localization only closes the merge stage on dense_oligomer coarse step 0. "
            "It does not yet patch the defect or generalize beyond this harness."
        ),
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "localization": localization,
    }


def dense_early_accumulation_localization(dense_fixture_summary: dict[str, Any]) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    exact_dir = Path(coarse["exact_work_dir"])

    plain_force_frames = parse_force_dump(plain_dir / "plain_total_force.tsv")
    exact_force_frames = parse_force_dump(exact_dir / "exact_total_force.tsv")
    excluded_force_dump = parse_vector_dump(exact_dir / "exact_excluded_pairs_correction_force.tsv")
    initial_outer = parse_vector_dump(exact_dir / "step0_level2_initial_outer_virial.tsv")
    after_pairs = parse_vector_dump(exact_dir / "step0_level2_after_pairs_virial.tsv")
    after_excluded = parse_vector_dump(exact_dir / "step0_level2_after_excluded_pairs_virial.tsv")
    before_longrange = parse_vector_dump(exact_dir / "step0_level2_before_longrange_virial.tsv")
    after_longrange = parse_vector_dump(exact_dir / "step0_level2_after_longrange_virial.tsv")
    first_excluded_write = parse_vector_dump(exact_dir / "step0_outer_first_excluded_write.tsv")
    pre_postprocess_outer = parse_vector_dump(exact_dir / "step0_level2_pre_postprocess_virial.tsv")
    level0_post = parse_vector_dump(exact_dir / "step0_level0_post_postprocess_shift.tsv")
    level1_post = parse_vector_dump(exact_dir / "step0_level1_post_postprocess_shift.tsv")

    step0 = min(set(plain_force_frames) & set(exact_force_frames))
    plain_force = plain_force_frames[step0]["forces"]
    exact_force = exact_force_frames[step0]["forces"]
    excluded_force = excluded_force_dump["forces"]
    initial_outer_force = initial_outer["forces"]
    after_pairs_force = after_pairs["forces"]
    after_excluded_force = after_excluded["forces"]
    before_longrange_force = before_longrange["forces"]
    after_longrange_force = after_longrange["forces"]
    pre_postprocess_outer_force = pre_postprocess_outer["forces"]
    level0_post_force = level0_post["forces"]
    level1_post_force = level1_post["forces"]

    zero_outer_force = {atom_index: (0.0, 0.0, 0.0) for atom_index in sorted(initial_outer_force)}
    after_pairs_delta = subtract_vectors(after_pairs_force, initial_outer_force)
    after_excluded_delta = subtract_vectors(after_excluded_force, after_pairs_force)
    longrange_delta = subtract_vectors(after_longrange_force, before_longrange_force)
    reconstructed_without_excluded = sum_vectors(
        level0_post_force, level1_post_force, subtract_vectors(after_longrange_force, excluded_force)
    )

    initial_outer_vs_zero = vector_metrics(initial_outer_force, zero_outer_force)
    after_excluded_delta_vs_excluded = vector_metrics(after_excluded_delta, excluded_force)
    before_longrange_vs_after_excluded = vector_metrics(before_longrange_force, after_excluded_force)
    after_longrange_vs_pre_postprocess = vector_metrics(after_longrange_force, pre_postprocess_outer_force)
    reconstructed_without_excluded_vs_plain = vector_metrics(reconstructed_without_excluded, plain_force)
    exact_vs_plain_step0_force_diff = vector_metrics(exact_force, plain_force)

    supports_first_illegal_site = (
        initial_outer_vs_zero["max_abs"] <= FORCE_BOOKKEEPING_TOL
        and after_excluded_delta_vs_excluded["max_abs"] <= FORCE_BOOKKEEPING_TOL
        and before_longrange_vs_after_excluded["max_abs"] <= FORCE_BOOKKEEPING_TOL
        and after_longrange_vs_pre_postprocess["max_abs"] <= FORCE_BOOKKEEPING_TOL
        and reconstructed_without_excluded_vs_plain["max_abs"] <= FORCE_BOOKKEEPING_TOL
    )
    alias_with_shift = first_excluded_write["metadata"].get("alias_with_shift")

    localization = {
        "classification": (
            "first illegal accumulation occurs in excludedPairs outer dispatch"
            if supports_first_illegal_site
            else "still unresolved"
        ),
        "step0": step0,
        "initial_outer_vs_zero_diff": initial_outer_vs_zero,
        "after_excluded_delta_vs_excluded_diff": after_excluded_delta_vs_excluded,
        "after_excluded_delta_alignment_with_excluded": vector_alignment(after_excluded_delta, excluded_force),
        "before_longrange_vs_after_excluded_diff": before_longrange_vs_after_excluded,
        "after_longrange_vs_pre_postprocess_diff": after_longrange_vs_pre_postprocess,
        "reconstructed_without_excluded_vs_plain_diff": reconstructed_without_excluded_vs_plain,
        "longrange_delta_norm": vector_metrics(longrange_delta, zero_outer_force),
        "exact_vs_plain_step0_force_diff": exact_vs_plain_step0_force_diff,
        "first_excluded_write_metadata": first_excluded_write["metadata"],
        "excluded_force_dump_metadata": excluded_force_dump["metadata"],
        "supports_first_illegal_site": supports_first_illegal_site,
        "supports_aliasing": alias_with_shift == "true",
        "supports_refolding_before_postprocess": False,
        "ordered_outer_accumulation_trace": [
            {
                "order": 0,
                "source": "initial_outer_zero",
                "path": str(exact_dir / "step0_level2_initial_outer_virial.tsv"),
                "diff_vs_zero": initial_outer_vs_zero,
            },
            {
                "order": 1,
                "source": "after_plain_pairs_dispatch",
                "path": str(exact_dir / "step0_level2_after_pairs_virial.tsv"),
                "delta_from_previous": vector_metrics(after_pairs_delta, zero_outer_force),
            },
            {
                "order": 2,
                "source": "after_excluded_pairs_dispatch",
                "path": str(exact_dir / "step0_level2_after_excluded_pairs_virial.tsv"),
                "delta_from_previous": vector_metrics(after_excluded_delta, zero_outer_force),
                "delta_vs_excluded_correction": after_excluded_delta_vs_excluded,
            },
            {
                "order": 3,
                "source": "before_longrange_nonbonded_reconciliation",
                "path": str(exact_dir / "step0_level2_before_longrange_virial.tsv"),
                "diff_vs_previous": before_longrange_vs_after_excluded,
            },
            {
                "order": 4,
                "source": "after_longrange_nonbonded_dispatch",
                "path": str(exact_dir / "step0_level2_after_longrange_virial.tsv"),
                "delta_from_previous": vector_metrics(longrange_delta, zero_outer_force),
                "reconstructed_total_vs_plain_if_excluded_removed_here": reconstructed_without_excluded_vs_plain,
            },
            {
                "order": 5,
                "source": "pre_postprocess_outer_reconciliation",
                "path": str(exact_dir / "step0_level2_pre_postprocess_virial.tsv"),
                "diff_vs_previous": after_longrange_vs_pre_postprocess,
            },
        ],
        "why_not_fully_closed": (
            "This closes the first illegal outer-force accumulation site on dense_oligomer coarse step 0, "
            "but it does not yet identify the earlier conceptual ownership rule that should have prevented "
            "the excludedPairs outer correction write."
        ),
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "localization": localization,
    }


def dense_pair_write_ownership_proof(dense_fixture_summary: dict[str, Any]) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    exact_dir = Path(coarse["exact_work_dir"])
    trace_off_dir = Path(coarse["exact_trace_off_work_dir"])

    plain_force_frames = parse_force_dump(plain_dir / "plain_total_force.tsv")
    exact_force_frames = parse_force_dump(exact_dir / "exact_total_force.tsv")
    trace_off_force_frames = parse_force_dump(trace_off_dir / "exact_total_force.tsv")
    topology_sources = parse_topology_pair_sources(exact_dir / "system.top")

    step0 = min(set(plain_force_frames) & set(exact_force_frames) & set(trace_off_force_frames))
    plain_force = plain_force_frames[step0]["forces"]
    exact_force = exact_force_frames[step0]["forces"]
    trace_off_force = trace_off_force_frames[step0]["forces"]

    excluded_before = parse_vector_dump(exact_dir / "step0_outer_excluded_write_ord000_before.tsv")
    excluded_event = parse_vector_dump(exact_dir / "step0_outer_excluded_write_ord000_event.tsv")
    excluded_after = parse_vector_dump(exact_dir / "step0_outer_excluded_write_ord000_after.tsv")
    excluded_delta = subtract_vectors(excluded_after["forces"], excluded_before["forces"])
    excluded_event_expanded = expand_sparse_vector(excluded_event["forces"], excluded_before["forces"])
    excluded_delta_vs_event = vector_metrics(excluded_delta, excluded_event_expanded)

    storage_rows = parse_key_value_text(exact_dir / "step0_force_storage_identity.txt")
    builder_rows = parse_key_value_text(exact_dir / "step0_pairlist_builder_append_trace.txt")
    preview_rows = parse_key_value_text(exact_dir / "step0_plain_pairlist_preview.txt")
    membership_rows = parse_key_value_text(exact_dir / "step0_pair_key_membership_scan.txt")

    top_level_storage: dict[str, str] = {}
    level_storage: dict[str, dict[str, str]] = {}
    for row in storage_rows:
        if "level" in row:
            level_storage.setdefault(row["level"], {}).update(row)
        else:
            top_level_storage.update(row)

    excluded_builder = next(
        (row for row in builder_rows if row.get("kind") == "excludedPairs" and row.get("ordinal") == "0"),
        None,
    )
    excluded_preview = next(
        (row for row in preview_rows if row.get("kind") == "excludedPairs" and row.get("ordinal") == "0"),
        None,
    )
    excluded_membership = next(
        (row for row in membership_rows if row.get("kind") == "excluded" and row.get("ordinal") == "0"),
        None,
    )

    excluded_metadata = excluded_event["metadata"]
    excluded_pair = (int(excluded_metadata["ai"]), int(excluded_metadata["aj"]))
    excluded_key = tuple(sorted(excluded_pair))
    excluded_topology_sources = topology_sources.get(excluded_key, [])

    excluded_builder_matches_event = (
        excluded_builder is not None
        and excluded_builder.get("ai") == excluded_metadata.get("ai")
        and excluded_builder.get("aj") == excluded_metadata.get("aj")
        and excluded_builder.get("shift_index") == excluded_metadata.get("shift_index")
    )
    excluded_preview_matches_event = (
        excluded_preview is not None
        and excluded_preview.get("ai") == excluded_metadata.get("ai")
        and excluded_preview.get("aj") == excluded_metadata.get("aj")
        and excluded_preview.get("shift_index") == excluded_metadata.get("shift_index")
    )

    def choose_control() -> dict[str, Any] | None:
        for ordinal in range(8):
            event_path = exact_dir / f"step0_outer_pairs_write_ord{ordinal}_event.tsv"
            before_path = exact_dir / f"step0_outer_pairs_write_ord{ordinal}_before.tsv"
            after_path = exact_dir / f"step0_outer_pairs_write_ord{ordinal}_after.tsv"
            if not event_path.exists() or not before_path.exists() or not after_path.exists():
                continue
            event_dump = parse_vector_dump(event_path)
            event_meta = event_dump["metadata"]
            pair = (int(event_meta["ai"]), int(event_meta["aj"]))
            pair_key = tuple(sorted(pair))
            topology_labels = topology_sources.get(pair_key, [])
            membership_row = next(
                (
                    row
                    for row in membership_rows
                    if row.get("kind") == "pairs" and row.get("ordinal") == str(ordinal)
                ),
                None,
            )
            builder_row = next(
                (
                    row
                    for row in builder_rows
                    if row.get("kind") == "pairs" and row.get("ordinal") == str(ordinal)
                ),
                None,
            )
            if membership_row is None or builder_row is None:
                continue
            is_clean = (
                not topology_labels
                and membership_row.get("in_plain_excluded") == "0"
                and membership_row.get("in_debug_listed_pair_keys") == "false"
            )
            if not is_clean:
                continue
            before_dump = parse_vector_dump(before_path)
            after_dump = parse_vector_dump(after_path)
            control_delta = subtract_vectors(after_dump["forces"], before_dump["forces"])
            control_event_expanded = expand_sparse_vector(event_dump["forces"], before_dump["forces"])
            control_minus_plain = vector_metrics(subtract_vectors(exact_force, control_event_expanded), plain_force)
            return {
                "ordinal": ordinal,
                "pair": pair,
                "topology_sources": topology_labels,
                "builder_row": builder_row,
                "membership_row": membership_row,
                "before_dump": before_dump,
                "event_dump": event_dump,
                "after_dump": after_dump,
                "delta_vs_event": vector_metrics(control_delta, control_event_expanded),
                "alignment_with_exact_minus_plain": vector_alignment(
                    control_event_expanded, subtract_vectors(exact_force, plain_force)
                ),
                "exact_minus_control_vs_plain": control_minus_plain,
            }
        return None

    control = choose_control()
    trace_off_vs_trace_on_force = vector_metrics(exact_force, trace_off_force)
    trace_on_potential = read_potential_xvg(exact_dir / "exact_potential.xvg")
    trace_off_potential = read_potential_xvg(trace_off_dir / "exact_potential.xvg")
    trace_off_vs_trace_on_potential = abs(
        trace_on_potential[0]["potential_kj_per_mol"] - trace_off_potential[0]["potential_kj_per_mol"]
    )

    exact_minus_excluded_vs_plain = vector_metrics(
        subtract_vectors(exact_force, excluded_event_expanded), plain_force
    )

    storage_disjointness = {
        "outer_vs_shift_disjoint": top_level_storage.get("outer_accumulator_force_ptr")
        != top_level_storage.get("outer_outputs_shift_force_ptr"),
        "outer_vs_level0_shift_disjoint": top_level_storage.get("outer_accumulator_force_ptr")
        != level_storage.get("0", {}).get("shift_force_ptr"),
        "outer_vs_level1_shift_disjoint": top_level_storage.get("outer_accumulator_force_ptr")
        != level_storage.get("1", {}).get("shift_force_ptr"),
        "outer_vs_level2_shift_disjoint": top_level_storage.get("outer_accumulator_force_ptr")
        != level_storage.get("2", {}).get("shift_force_ptr"),
        "outer_force_equals_virial_buffer": top_level_storage.get("outer_accumulator_force_ptr")
        == top_level_storage.get("outer_accumulator_virial_ptr"),
        "outer_aliases_shift": top_level_storage.get("outer_aliases_shift") == "true",
    }

    earlier_ownership_fault_alive = bool(excluded_topology_sources)
    control_clean = control is not None
    trace_semantics_unchanged = (
        trace_off_vs_trace_on_force["max_abs"] <= FORCE_BOOKKEEPING_TOL
        and trace_off_vs_trace_on_potential <= POTENTIAL_BOOKKEEPING_TOL
    )
    supports_exact_first_illegal_write = (
        excluded_delta_vs_event["max_abs"] <= FORCE_BOOKKEEPING_TOL
        and excluded_builder_matches_event
        and excluded_preview_matches_event
        and not earlier_ownership_fault_alive
        and control_clean
        and control["delta_vs_event"]["max_abs"] <= FORCE_BOOKKEEPING_TOL
        and storage_disjointness["outer_vs_shift_disjoint"]
        and storage_disjointness["outer_vs_level0_shift_disjoint"]
        and storage_disjointness["outer_vs_level1_shift_disjoint"]
        and storage_disjointness["outer_force_equals_virial_buffer"]
        and not storage_disjointness["outer_aliases_shift"]
        and trace_semantics_unchanged
    )

    localization = {
        "classification": (
            "exact first illegal pair-write ownership proven"
            if supports_exact_first_illegal_write
            else "first visible consumer proven; earlier ownership/spec fault still alive"
            if earlier_ownership_fault_alive
            else "still unresolved"
        ),
        "step0": step0,
        "excluded_pair_write": {
            "metadata": excluded_metadata,
            "delta_vs_event": excluded_delta_vs_event,
            "builder_matches_event": excluded_builder_matches_event,
            "preview_matches_event": excluded_preview_matches_event,
            "builder_row": excluded_builder,
            "preview_row": excluded_preview,
            "membership_row": excluded_membership,
            "topology_sources": excluded_topology_sources,
            "exact_minus_excluded_vs_plain": exact_minus_excluded_vs_plain,
        },
        "storage_identity": {
            "top_level": top_level_storage,
            "levels": level_storage,
            "disjointness": storage_disjointness,
        },
        "ownership_lineage": {
            "builder_trace_path": str(exact_dir / "step0_pairlist_builder_append_trace.txt"),
            "pairlist_preview_path": str(exact_dir / "step0_plain_pairlist_preview.txt"),
            "membership_scan_path": str(exact_dir / "step0_pair_key_membership_scan.txt"),
        },
        "known_good_control": None
        if control is None
        else {
            "ordinal": control["ordinal"],
            "pair": control["pair"],
            "builder_row": control["builder_row"],
            "membership_row": control["membership_row"],
            "topology_sources": control["topology_sources"],
            "delta_vs_event": control["delta_vs_event"],
            "alignment_with_exact_minus_plain": control["alignment_with_exact_minus_plain"],
            "exact_minus_control_vs_plain": control["exact_minus_control_vs_plain"],
        },
        "trace_on_vs_trace_off": {
            "step0_force_diff": trace_off_vs_trace_on_force,
            "step0_potential_abs_diff_kj_per_mol": trace_off_vs_trace_on_potential,
        },
        "supports_exact_first_illegal_write": supports_exact_first_illegal_write,
        "earlier_ownership_fault_alive": earlier_ownership_fault_alive,
        "why_not_fully_closed": (
            "The exact pair-write boundary is captured, but the traced pair already belongs to earlier "
            "topology ownership buckets."
            if earlier_ownership_fault_alive
            else "The exact pair-write boundary is captured within this dense step-0 scope only."
        ),
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "localization": localization,
    }


def summarize_dense_trace_fixture(
    fixture_id: str,
    topology_terms: list[str],
    box_nm: tuple[float, float, float],
    coarse_plain: dict[str, Any],
    coarse_side_reference: dict[str, Any],
    coarse_exact: dict[str, Any],
) -> dict[str, Any]:
    coarse_exact_vs_plain = compare_series(coarse_exact, coarse_plain, box_nm)
    coarse_side_reference_vs_plain = compare_series(coarse_side_reference, coarse_plain, box_nm)
    coarse_exact_vs_side_reference = compare_series(coarse_exact, coarse_side_reference, box_nm)

    return {
        "fixture_id": fixture_id,
        "listed_terms_present": topology_terms,
        "exact_split": {
            "inner_terms": topology_terms + ["nonbonded_inner"],
            "middle_terms": ["nonbonded_middle"],
            "outer_terms": ["pair", "nonbonded_outer", "kspace"],
        },
        "exact_schedule_active": exact_scheduler_active(coarse_exact["schedule"]),
        "pme_side_reference_active": legacy_scheduler_active(coarse_side_reference["schedule"]),
        "bookkeeping_ok": (
            coarse_exact_vs_plain["step0_force_diff"]["max_abs"] <= FORCE_BOOKKEEPING_TOL
            and coarse_exact_vs_plain["step0_potential_abs_diff_kj_per_mol"] <= POTENTIAL_BOOKKEEPING_TOL
        ),
        "coarse": {
            "dt_ps": coarse_exact["dt_ps"],
            "nsteps": coarse_exact["nsteps"],
            "plain_work_dir": coarse_plain["work_dir"],
            "side_reference_work_dir": coarse_side_reference["work_dir"],
            "exact_work_dir": coarse_exact["work_dir"],
            "exact_vs_plain": coarse_exact_vs_plain,
            "side_reference_vs_plain": coarse_side_reference_vs_plain,
            "exact_vs_side_reference": coarse_exact_vs_side_reference,
        },
        "exact_schedule_dump": coarse_exact["schedule"],
        "pme_side_reference_dump": coarse_side_reference["schedule"],
        "comparison_notes": [
            "Plain Verlet uses integrator = md-vv with the same PME/Cut-off settings as the exact 3-level path.",
            "The simpler split here is a PME-side legacy side-reference on the same harness; it is not direct archived-M1 continuity because archived M1 used md + Cut-off settings.",
        ],
    }


def load_build_provenance(gmx_bin: str) -> dict[str, Any]:
    version_output = subprocess.run(
        [gmx_bin, "--version"], capture_output=True, text=True, check=True, errors="replace"
    ).stdout
    source_tree = None
    git_sha = None
    for line in version_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Data prefix:"):
            source_tree = stripped.split(":", 1)[1].strip().split(" ")[0]
        elif stripped.startswith("GIT SHA1 hash:"):
            git_sha = stripped.split(":", 1)[1].strip()
    return {
        "gmx_bin": gmx_bin,
        "version_text": version_output,
        "source_tree": source_tree,
        "git_sha": git_sha,
    }


def fixture_source_record(system_id: str) -> dict[str, str]:
    return {
        "system_json": str(REPO_ROOT / "testdata" / "lammps_golden" / "systems" / system_id / "system.json"),
        "lammps_data": str(REPO_ROOT / "testdata" / "lammps_golden" / "systems" / system_id / "lammps" / "system.data"),
        "lammps_input": str(REPO_ROOT / "testdata" / "lammps_golden" / "systems" / system_id / "lammps" / "system.in"),
    }


def run_case(
    gmx_bin: str,
    fixture_root: Path,
    work_dir: Path,
    mode: str,
    dt_ps: float,
    total_time_ps: float,
    commands_log: list[str],
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)

    if mode == "plain_verlet":
        deffnm = "plain"
        mdp_path = work_dir / "plain.mdp"
        nsteps = write_plain_mdp(mdp_path, dt_ps, total_time_ps)
    elif mode == PME_LEGACY_SIDE_REFERENCE_MODE:
        deffnm = "legacy"
        mdp_path = work_dir / "legacy.mdp"
        nsteps = write_legacy_mdp(mdp_path, dt_ps, total_time_ps)
    elif mode in {"exact_three_level", "exact_three_level_trace_off"}:
        deffnm = "exact"
        mdp_path = work_dir / "exact.mdp"
        nsteps = write_exact_mdp(mdp_path, dt_ps, total_time_ps)
    else:
        raise ValueError(f"unknown mode {mode}")

    (work_dir / "system.top").write_bytes((fixture_root / "system.top").read_bytes())
    (work_dir / "system.gro").write_bytes((fixture_root / "system.gro").read_bytes())

    run_command(
        [
            gmx_bin,
            "grompp",
            "-f",
            str(mdp_path),
            "-c",
            "system.gro",
            "-p",
            "system.top",
            "-o",
            f"{deffnm}.tpr",
            "-maxwarn",
            "10",
        ],
        work_dir,
        f"{deffnm}_grompp",
        commands_log,
    )

    run_command(
        [
            gmx_bin,
            "mdrun",
            "-s",
            f"{deffnm}.tpr",
            "-deffnm",
            deffnm,
            "-ntmpi",
            "1",
            "-ntomp",
            "1",
            "-pin",
            "off",
            "-nb",
            "cpu",
            "-bonded",
            "cpu",
            "-pme",
            "cpu",
            "-update",
            "cpu",
            "-reprod",
        ],
        work_dir,
        f"{deffnm}_mdrun",
        commands_log,
        extra_env=extra_env,
    )

    tpr_dump = run_command([gmx_bin, "dump", "-s", f"{deffnm}.tpr"], work_dir, f"{deffnm}_tpr_dump", commands_log)
    trr_dump = run_command([gmx_bin, "dump", "-f", f"{deffnm}.trr"], work_dir, f"{deffnm}_trr_dump", commands_log)
    trr_frames = parse_trr_dump(trr_dump.stdout)
    potential_series = extract_potential_series(gmx_bin, work_dir, deffnm, commands_log, deffnm)

    return {
        "mode": mode,
        "dt_ps": dt_ps,
        "nsteps": nsteps,
        "work_dir": str(work_dir),
        "schedule": parse_tpr_mts_fields(tpr_dump.stdout),
        "frames": trr_frames,
        "potential_series": potential_series,
    }


def compare_series(
    left: dict[str, Any], right: dict[str, Any], box_nm: tuple[float, float, float] | None = None
) -> dict[str, Any]:
    final_step = left["nsteps"]
    return {
        "step0_force_diff": vector_metrics(left["frames"][0]["f"], right["frames"][0]["f"]),
        "final_force_diff": vector_metrics(left["frames"][final_step]["f"], right["frames"][final_step]["f"]),
        "final_coord_diff": vector_metrics(
            left["frames"][final_step]["x"], right["frames"][final_step]["x"], box_nm
        ),
        "step0_potential_abs_diff_kj_per_mol": abs(
            left["potential_series"][0]["potential_kj_per_mol"] - right["potential_series"][0]["potential_kj_per_mol"]
        ),
        "final_potential_abs_diff_kj_per_mol": abs(
            left["potential_series"][-1]["potential_kj_per_mol"] - right["potential_series"][-1]["potential_kj_per_mol"]
        ),
    }


def numeric_field_matches(schedule: dict[str, str], key: str, expected: float) -> bool:
    actual = schedule.get(key)
    if actual is None:
        return False
    try:
        return abs(float(actual) - expected) <= NUMERIC_FIELD_TOL
    except ValueError:
        return False


def normalized_force_groups(value: str | None) -> set[str]:
    if value is None:
        return set()
    return {token for token in value.split() if token}


def exact_scheduler_active(schedule: dict[str, str]) -> bool:
    if schedule.get("mts") != "true" or schedule.get("mts-mode") != "lammps-respa":
        return False
    if "mts-levels" in schedule and schedule.get("mts-levels") != "3":
        return False
    if normalized_force_groups(schedule.get("mts-level2-forces")) != {"nonbonded-middle"}:
        return False
    if normalized_force_groups(schedule.get("mts-level3-forces")) != {
        "longrange-nonbonded",
        "nonbonded-outer",
    }:
        return False
    numeric_expectations = {
        "mts-level2-factor": float(EXACT_LEVEL2_FACTOR),
        "mts-level3-factor": float(EXACT_LEVEL3_FACTOR),
        "mts-respa-bond-level": 1.0,
        "mts-respa-angle-level": 1.0,
        "mts-respa-dihedral-level": 1.0,
        "mts-respa-improper-level": 1.0,
        "mts-respa-pair14-level": 1.0,
        "mts-respa-pair-level": 3.0,
        "mts-respa-kspace-level": 3.0,
        "mts-respa-inner-level": 1.0,
        "mts-respa-middle-level": 2.0,
        "mts-respa-outer-level": 3.0,
        "mts-respa-inner-off": 0.30,
        "mts-respa-inner-on": 0.45,
        "mts-respa-outer-on": 0.60,
        "mts-respa-outer-off": 0.80,
    }
    return all(numeric_field_matches(schedule, key, expected) for key, expected in numeric_expectations.items())


def legacy_scheduler_active(schedule: dict[str, str]) -> bool:
    return (
        schedule.get("mts") == "true"
        and schedule.get("mts-mode") == "legacy"
        and ("mts-levels" not in schedule or schedule.get("mts-levels") == "2")
        and numeric_field_matches(schedule, "mts-level2-factor", float(LEGACY_MTS_FACTOR))
        and normalized_force_groups(schedule.get("mts-level2-forces")) == normalized_force_groups(LEGACY_FORCE_GROUPS)
    )


def summarize_fixture(
    fixture_id: str,
    topology_terms: list[str],
    box_nm: tuple[float, float, float],
    coarse_plain: dict[str, Any],
    coarse_side_reference: dict[str, Any],
    coarse_exact: dict[str, Any],
    fine_plain: dict[str, Any],
    fine_side_reference: dict[str, Any],
    fine_exact: dict[str, Any],
) -> dict[str, Any]:
    coarse_exact_vs_plain = compare_series(coarse_exact, coarse_plain, box_nm)
    fine_exact_vs_plain = compare_series(fine_exact, fine_plain, box_nm)
    coarse_side_reference_vs_plain = compare_series(coarse_side_reference, coarse_plain, box_nm)
    fine_side_reference_vs_plain = compare_series(fine_side_reference, fine_plain, box_nm)
    coarse_exact_vs_side_reference = compare_series(coarse_exact, coarse_side_reference, box_nm)
    fine_exact_vs_side_reference = compare_series(fine_exact, fine_side_reference, box_nm)

    exact_bookkeeping_ok = (
        coarse_exact_vs_plain["step0_force_diff"]["max_abs"] <= FORCE_BOOKKEEPING_TOL
        and fine_exact_vs_plain["step0_force_diff"]["max_abs"] <= FORCE_BOOKKEEPING_TOL
        and coarse_exact_vs_plain["step0_potential_abs_diff_kj_per_mol"] <= POTENTIAL_BOOKKEEPING_TOL
        and fine_exact_vs_plain["step0_potential_abs_diff_kj_per_mol"] <= POTENTIAL_BOOKKEEPING_TOL
    )
    convergence_ok = (
        fine_exact_vs_plain["final_coord_diff"]["l2_norm"] < coarse_exact_vs_plain["final_coord_diff"]["l2_norm"]
        and fine_exact_vs_plain["final_force_diff"]["l2_norm"] <= coarse_exact_vs_plain["final_force_diff"]["l2_norm"]
        and fine_exact_vs_plain["final_potential_abs_diff_kj_per_mol"]
        <= coarse_exact_vs_plain["final_potential_abs_diff_kj_per_mol"]
    )
    no_worse_than_side_reference_fine = (
        fine_exact_vs_plain["final_coord_diff"]["l2_norm"] <= fine_side_reference_vs_plain["final_coord_diff"]["l2_norm"]
        and fine_exact_vs_plain["final_force_diff"]["l2_norm"] <= fine_side_reference_vs_plain["final_force_diff"]["l2_norm"]
        and fine_exact_vs_plain["final_potential_abs_diff_kj_per_mol"]
        <= fine_side_reference_vs_plain["final_potential_abs_diff_kj_per_mol"]
    )

    return {
        "fixture_id": fixture_id,
        "listed_terms_present": topology_terms,
        "exact_split": {
            "inner_terms": topology_terms + ["nonbonded_inner"],
            "middle_terms": ["nonbonded_middle"],
            "outer_terms": ["pair", "nonbonded_outer", "kspace"],
        },
        "exact_schedule_active": exact_scheduler_active(coarse_exact["schedule"]),
        "pme_side_reference_active": legacy_scheduler_active(coarse_side_reference["schedule"]),
        "bookkeeping_ok": exact_bookkeeping_ok,
        "convergence_ok": convergence_ok,
        "fine_exact_no_worse_than_pme_side_reference": no_worse_than_side_reference_fine,
        "coarse": {
            "dt_ps": coarse_exact["dt_ps"],
            "nsteps": coarse_exact["nsteps"],
            "plain_work_dir": coarse_plain["work_dir"],
            "side_reference_work_dir": coarse_side_reference["work_dir"],
            "exact_work_dir": coarse_exact["work_dir"],
            "exact_vs_plain": coarse_exact_vs_plain,
            "side_reference_vs_plain": coarse_side_reference_vs_plain,
            "exact_vs_side_reference": coarse_exact_vs_side_reference,
        },
        "fine": {
            "dt_ps": fine_exact["dt_ps"],
            "nsteps": fine_exact["nsteps"],
            "plain_work_dir": fine_plain["work_dir"],
            "side_reference_work_dir": fine_side_reference["work_dir"],
            "exact_work_dir": fine_exact["work_dir"],
            "exact_vs_plain": fine_exact_vs_plain,
            "side_reference_vs_plain": fine_side_reference_vs_plain,
            "exact_vs_side_reference": fine_exact_vs_side_reference,
        },
        "exact_schedule_dump": coarse_exact["schedule"],
        "pme_side_reference_dump": coarse_side_reference["schedule"],
        "comparison_notes": [
            "Plain Verlet uses integrator = md-vv with the same PME/Cut-off settings as the exact 3-level path.",
            "The simpler split here is a PME-side legacy side-reference on the same harness; it is not direct archived-M1 continuity because archived M1 used md + Cut-off settings.",
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    gmx_bin = str(Path(args.gmx_bin).resolve())
    output_root = Path(args.out).resolve()
    fixtures = args.fixtures or list(DEFAULT_FIXTURES)
    corpus_root = REPO_ROOT / "testdata" / "lammps_golden"
    output_root.mkdir(parents=True, exist_ok=True)

    commands_log: list[str] = []
    build_provenance = load_build_provenance(gmx_bin)

    fixture_results = []
    for fixture_id in fixtures:
        typed_ir = build_typed_ir({"id": fixture_id, "path": f"systems/{fixture_id}"}, corpus_root)
        topology_text = render_gromacs_topology(typed_ir)
        system_root = output_root / fixture_id
        system_root.mkdir(parents=True, exist_ok=True)
        (system_root / "system.top").write_text(topology_text, encoding="utf-8")
        gro_meta = create_gro_from_lammps(
            corpus_root / "systems" / fixture_id / "lammps" / "system.data",
            system_root / "system.gro",
        )

        coarse_plain_env = None
        coarse_side_reference_env = None
        coarse_exact_env = None
        if (
            args.dense_force_ownership_isolation
            or args.dense_merge_trace
            or args.dense_early_accumulation_trace
            or args.exact_pair_write_ownership_proof
        ) and fixture_id == "dense_oligomer":
            coarse_plain_env = {
                "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                "GMX_TOTAL_FORCE_DUMP_FILE": str(system_root / "dt_0p0005" / "plain_verlet" / "plain_total_force.tsv"),
            }
            coarse_exact_env = {
                "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                "GMX_TOTAL_FORCE_DUMP_FILE": str(
                    system_root / "dt_0p0005" / "exact_three_level" / "exact_total_force.tsv"
                ),
                "GMX_PCFF_RESPA_EXCLUDED_FORCE_DUMP_FILE": str(
                    system_root
                    / "dt_0p0005"
                    / "exact_three_level"
                    / "exact_excluded_pairs_correction_force.tsv"
                ),
            }
        if (args.dense_merge_trace or args.dense_early_accumulation_trace) and fixture_id == "dense_oligomer":
            coarse_exact_env = dict(coarse_exact_env or {})
            coarse_exact_env["GMX_PCFF_RESPA_MERGE_TRACE_DIR"] = str(system_root / "dt_0p0005" / "exact_three_level")
        if args.dense_early_accumulation_trace and fixture_id == "dense_oligomer":
            coarse_exact_env = dict(coarse_exact_env or {})
            coarse_exact_env["GMX_PCFF_RESPA_EARLY_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "exact_three_level"
            )
        if args.exact_pair_write_ownership_proof and fixture_id == "dense_oligomer":
            coarse_exact_env = dict(coarse_exact_env or {})
            coarse_exact_env["GMX_PCFF_RESPA_PAIR_WRITE_PROOF_DIR"] = str(
                system_root / "dt_0p0005" / "exact_three_level"
            )
        if args.dense_bookkeeping_isolation and fixture_id == "dense_oligomer":
            coarse_exact_env = dict(coarse_exact_env or {})
            coarse_exact_env["GMX_PCFF_RESPA_DEBUG"] = "1"

        coarse_plain = run_case(
            gmx_bin,
            system_root,
            system_root / "dt_0p0005" / "plain_verlet",
            "plain_verlet",
            DEFAULT_DT_VALUES[0],
            args.total_time_ps,
            commands_log,
            extra_env=coarse_plain_env,
        )
        coarse_side_reference = run_case(
            gmx_bin,
            system_root,
            system_root / "dt_0p0005" / PME_LEGACY_SIDE_REFERENCE_MODE,
            PME_LEGACY_SIDE_REFERENCE_MODE,
            DEFAULT_DT_VALUES[0],
            args.total_time_ps,
            commands_log,
            extra_env=coarse_side_reference_env,
        )
        coarse_exact = run_case(
            gmx_bin,
            system_root,
            system_root / "dt_0p0005" / "exact_three_level",
            "exact_three_level",
            DEFAULT_DT_VALUES[0],
            args.total_time_ps,
            commands_log,
            extra_env=coarse_exact_env,
        )
        coarse_exact_trace_off = None
        if args.exact_pair_write_ownership_proof and fixture_id == "dense_oligomer":
            coarse_exact_trace_off = run_case(
                gmx_bin,
                system_root,
                system_root / "dt_0p0005" / "exact_three_level_trace_off",
                "exact_three_level_trace_off",
                DEFAULT_DT_VALUES[0],
                args.total_time_ps,
                commands_log,
                extra_env={
                    "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                    "GMX_TOTAL_FORCE_DUMP_FILE": str(
                        system_root / "dt_0p0005" / "exact_three_level_trace_off" / "exact_total_force.tsv"
                    ),
                },
            )
        if (
            args.dense_merge_trace
            or args.dense_early_accumulation_trace
            or args.exact_pair_write_ownership_proof
        ) and fixture_id == "dense_oligomer":
            fixture_summary = summarize_dense_trace_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                tuple(gro_meta["box_nm"]),
                coarse_plain,
                coarse_side_reference,
                coarse_exact,
            )
            if coarse_exact_trace_off is not None:
                fixture_summary["coarse"]["exact_trace_off_work_dir"] = coarse_exact_trace_off["work_dir"]
        else:
            fine_plain = run_case(
                gmx_bin,
                system_root,
                system_root / "dt_0p00025" / "plain_verlet",
                "plain_verlet",
                DEFAULT_DT_VALUES[1],
                args.total_time_ps,
                commands_log,
            )
            fine_side_reference = run_case(
                gmx_bin,
                system_root,
                system_root / "dt_0p00025" / PME_LEGACY_SIDE_REFERENCE_MODE,
                PME_LEGACY_SIDE_REFERENCE_MODE,
                DEFAULT_DT_VALUES[1],
                args.total_time_ps,
                commands_log,
            )
            fine_exact = run_case(
                gmx_bin,
                system_root,
                system_root / "dt_0p00025" / "exact_three_level",
                "exact_three_level",
                DEFAULT_DT_VALUES[1],
                args.total_time_ps,
                commands_log,
            )

            fixture_summary = summarize_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                tuple(gro_meta["box_nm"]),
                coarse_plain,
                coarse_side_reference,
                coarse_exact,
                fine_plain,
                fine_side_reference,
                fine_exact,
            )
        fixture_summary["fixture_sources"] = fixture_source_record(fixture_id)
        fixture_summary["gro_metadata"] = gro_meta
        fixture_summary["m1_continuity"] = archived_m1_continuity_record()
        if args.dense_bookkeeping_isolation and fixture_id == "dense_oligomer":
            fixture_summary["dense_exact_bookkeeping_isolation"] = dense_bookkeeping_isolation(
                gmx_bin, fixture_summary, commands_log
            )
        if args.dense_force_ownership_isolation and fixture_id == "dense_oligomer":
            fixture_summary["dense_exact_force_ownership_isolation"] = dense_force_ownership_isolation(
                fixture_summary
            )
        if args.dense_merge_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_exact_merge_trace_localization"] = dense_merge_trace_localization(
                fixture_summary
            )
        if args.dense_early_accumulation_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_exact_early_accumulation_localization"] = dense_early_accumulation_localization(
                fixture_summary
            )
        if args.exact_pair_write_ownership_proof and fixture_id == "dense_oligomer":
            fixture_summary["dense_exact_pair_write_ownership_proof"] = dense_pair_write_ownership_proof(
                fixture_summary
            )
        fixture_results.append(fixture_summary)
        write_json(system_root / "fixture_summary.json", fixture_summary)

    if args.exact_pair_write_ownership_proof:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_exact_pair_write_ownership_proof", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_exact_first_illegal_write"):
            verdict = "EXACT FIRST ILLEGAL PAIR-WRITE OWNERSHIP PROVEN"
        elif dense_localization and dense_localization.get("earlier_ownership_fault_alive"):
            verdict = "FIRST VISIBLE CONSUMER PROVEN; EARLIER OWNERSHIP STILL ALIVE"
        else:
            verdict = "PAIR-WRITE OWNERSHIP STILL PARTIAL"
    elif args.dense_early_accumulation_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_exact_early_accumulation_localization", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_first_illegal_site"):
            verdict = "EARLY ACCUMULATION SITE LOCALIZED"
        else:
            verdict = "EARLY ACCUMULATION TRACE STILL PARTIAL"
    elif args.dense_merge_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_exact_merge_trace_localization", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_before_postprocess"):
            verdict = "DUPLICATION LOCALIZED BEFORE POSTPROCESS"
        elif dense_localization and dense_localization.get("supports_postprocess"):
            verdict = "DUPLICATION LOCALIZED TO POSTPROCESSFORCES"
        elif dense_localization and dense_localization.get("supports_combine"):
            verdict = "DUPLICATION LOCALIZED TO COMBINEMTSFORCES"
        else:
            verdict = "DUPLICATION NARROWED BUT STILL PARTIAL"
    elif args.dense_force_ownership_isolation:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_exact_force_ownership_isolation", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_duplicated_term"):
            verdict = "FORCE DEFECT LOCALIZED TO DUPLICATED EXCLUDED-PAIR COULOMB FORCE OWNERSHIP"
        elif dense_localization and dense_localization.get("supports_missing_term"):
            verdict = "FORCE DEFECT LOCALIZED TO MISSING EXCLUDED-PAIR COULOMB FORCE OWNERSHIP"
        else:
            verdict = "FORCE-SIDE DEFECT NARROWED BUT STILL PARTIAL"
    elif args.dense_bookkeeping_isolation:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_exact_bookkeeping_isolation", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_wrong_level_ownership"):
            verdict = "DENSE BOOKKEEPING STILL PARTIAL BUT NARROWED"
        else:
            verdict = "BASELINE CONTINUITY CORRECTED BUT DEFECT STILL UNRESOLVED"
    else:
        verdict = (
            "EXACT 3-LEVEL PATH RECONNECTED AND VALIDATED ON MICROFIXTURES"
            if all(
                item["exact_schedule_active"] and item["bookkeeping_ok"] and item["convergence_ok"]
                for item in fixture_results
            )
            else "EXACT 3-LEVEL PATH RUNS BUT BOOKKEEPING/CONVERGENCE IS STILL PARTIAL"
        )

    summary = {
        "milestone": args.milestone_name,
        "worktree": str(REPO_ROOT),
        "branch": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
            errors="replace",
        ).stdout.strip(),
        "head_commit": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            errors="replace",
        ).stdout.strip(),
        "git_status": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--short"],
            capture_output=True,
            text=True,
            check=True,
            errors="replace",
        ).stdout.splitlines(),
        "build_provenance": build_provenance,
        "exact_three_level_path": {
            "mode": "lammps-respa",
            "integrator": "md-vv",
            "mts_levels": 3,
            "level_factors": {"level2": EXACT_LEVEL2_FACTOR, "level3": EXACT_LEVEL3_FACTOR},
            "ownership": {
                "bond": 1,
                "angle": 1,
                "dihedral": 1,
                "improper": 1,
                "pair14": 1,
                "pair": 3,
                "kspace": 3,
                "nonbonded_inner": 1,
                "nonbonded_middle": 2,
                "nonbonded_outer": 3,
            },
            "switching_nm": {
                "inner_off": 0.30,
                "inner_on": 0.45,
                "outer_on": 0.60,
                "outer_off": 0.80,
            },
            "notes": [
                "The frozen exact 3-level path already exists in the engine; this harness exercises it directly instead of silently falling back to the simpler 2-level path.",
                "The simpler split reference on this harness is a PME-side legacy side-reference, not direct archived-M1 continuity.",
            ],
        },
        "m1_continuity": archived_m1_continuity_record(),
        "fixtures": fixture_results,
        "verdict": verdict,
    }
    write_json(output_root / "summary.json", summary)
    (output_root / "raw_commands.txt").write_text("\n".join(commands_log) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
