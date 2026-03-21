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
    return parser.parse_args()


def run_command(
    cmd: list[str],
    cwd: Path,
    label: str,
    commands_log: list[str],
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    commands_log.append(f"(cd {shlex.quote(str(cwd))} && {' '.join(shlex.quote(part) for part in cmd)})")
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
    elif mode == "exact_three_level":
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

        coarse_plain = run_case(
            gmx_bin,
            system_root,
            system_root / "dt_0p0005" / "plain_verlet",
            "plain_verlet",
            DEFAULT_DT_VALUES[0],
            args.total_time_ps,
            commands_log,
        )
        coarse_side_reference = run_case(
            gmx_bin,
            system_root,
            system_root / "dt_0p0005" / PME_LEGACY_SIDE_REFERENCE_MODE,
            PME_LEGACY_SIDE_REFERENCE_MODE,
            DEFAULT_DT_VALUES[0],
            args.total_time_ps,
            commands_log,
        )
        coarse_exact = run_case(
            gmx_bin,
            system_root,
            system_root / "dt_0p0005" / "exact_three_level",
            "exact_three_level",
            DEFAULT_DT_VALUES[0],
            args.total_time_ps,
            commands_log,
            extra_env={"GMX_PCFF_RESPA_DEBUG": "1"} if args.dense_bookkeeping_isolation and fixture_id == "dense_oligomer" else None,
        )
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
        fixture_results.append(fixture_summary)
        write_json(system_root / "fixture_summary.json", fixture_summary)

    if args.dense_bookkeeping_isolation:
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
