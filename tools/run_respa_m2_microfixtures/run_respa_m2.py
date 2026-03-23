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
LEDGER_TRACE_TOL = 1.0e-5
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
M2J_PROBE_SPECS = (
    {
        "candidate": "includePair policy",
        "key": "includepair_policy",
        "probe_mode": "includepair_restricted",
        "work_dir_name": "probe_includepair_restricted",
    },
    {
        "candidate": "activeContributions configuration",
        "key": "active_contributions",
        "probe_mode": "active_outer_narrowed",
        "work_dir_name": "probe_active_outer_narrowed",
    },
    {
        "candidate": "outer routing / forceWithVirial selection",
        "key": "outer_routing",
        "probe_mode": "outer_routing_suppressed",
        "work_dir_name": "probe_outer_routing_suppressed",
    },
    {
        "candidate": "excluded correction -> outer contribution selection",
        "key": "correction_outer_selection",
        "probe_mode": "correction_outer_suppressed",
        "work_dir_name": "probe_correction_outer_suppressed",
    },
)
M2K_PATCH_SPECS = (
    {
        "candidate": "Patch-shape A",
        "key": "patch_shape_a",
        "patch_mode": "patch_shape_a",
        "work_dir_name": "patch_shape_a",
    },
    {
        "candidate": "Patch-shape B",
        "key": "patch_shape_b",
        "patch_mode": "patch_shape_b",
        "work_dir_name": "patch_shape_b",
    },
)
M2L_DIAGNOSTIC_PROBE = {
    "candidate": "Patch-B bookkeeping-suppressed micro-probe",
    "key": "patch_b_bookkeeping_suppressed",
    "probe_mode": "patch_shape_b_bookkeeping_suppressed",
    "work_dir_name": "probe_patch_b_bookkeeping_suppressed",
}
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
    parser.add_argument(
        "--upstream-ownership-handoff-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 upstream ownership/spec handoff diagnostics for M2g-style tracing.",
    )
    parser.add_argument(
        "--pair-rule-derivation-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 pair-specific generate_excl rule-derivation diagnostics for M2h-style tracing.",
    )
    parser.add_argument(
        "--downstream-misconsumption-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 downstream runtime contract diagnostics for M2i-style tracing.",
    )
    parser.add_argument(
        "--dispatch-minimal-fix-isolation",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 exact excludedPairs dispatch-internal diagnostics for M2j-style minimal-fix isolation.",
    )
    parser.add_argument(
        "--narrow-patch-proof",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 narrow patch-proof diagnostics for M2k-style excluded-correction outer-promotion validation.",
    )
    parser.add_argument(
        "--locked-scope-bookkeeping-residual-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 Patch-shape-B bookkeeping residual diagnostics for M2l-style classification.",
    )
    parser.add_argument(
        "--reciprocal-internal-delta-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 equal-depth reciprocal internal diagnostics for M2m-style origin tracing.",
    )
    parser.add_argument(
        "--post-final-ledger-mutation-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 post-FINAL-LEDGER mutation/export diagnostics for M2n-style tracing.",
    )
    parser.add_argument(
        "--lj-sr-first-sink-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 LJ-(SR) first sink/origin diagnostics for M2p-style tracing.",
    )
    parser.add_argument(
        "--lj-sr-true-first-raw-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 contract-matched earliest raw LJ diagnostics for M2q-style tracing.",
    )
    parser.add_argument(
        "--lj-sr-first-amplification-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 contract-matched LJ amplification diagnostics for M2r-style tracing.",
    )
    parser.add_argument(
        "--raw-sr-formation-internal-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 RAW_SR_FORMATION internal mutation/read diagnostics for M2s-style tracing.",
    )
    parser.add_argument(
        "--raw-sr-write-ordinal-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 RAW_FIRST_WRITE->RAW_POST_WRITE write-ordinal running-total diagnostics for M2u-style tracing.",
    )
    parser.add_argument(
        "--aligned-write-contract-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 cross-side aligned-event running-total diagnostics for M2v-style contract alignment.",
    )
    parser.add_argument(
        "--aligned-event-669-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 aligned-event 669 identity/arithmetic diagnostics for M2w-style localization.",
    )
    parser.add_argument(
        "--event-669-geometry-producer-trace",
        action="store_true",
        help="Collect dense_oligomer coarse step-0 upstream geometry-producer diagnostics for aligned event 669 / pair (18,0).",
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


def parse_topology_nrexcl(topology_path: Path) -> int | None:
    current_section = None
    for raw_line in topology_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line.strip("[]").strip().lower()
            continue
        if current_section == "moleculetype":
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1])
    return None


def parse_bool_text(value: str | None) -> bool:
    return value == "true"


def parse_index_csv(value: str | None) -> list[int]:
    if value is None or value == "none":
        return []
    return [int(token) for token in value.split(",") if token]


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


def canonical_energy_term_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def parse_energy_stdout_term_order(stdout_text: str) -> list[str]:
    term_order: list[str] = []
    in_energy_table = False
    energy_row_pattern = re.compile(
        r"^(?P<label>.+?)\s{2,}(?P<value>[-+0-9.]+(?:[eE][-+]?\d+)?|nan|inf|--)\b"
    )

    for raw_line in stdout_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if not in_energy_table:
            if set(stripped) == {"-"}:
                in_energy_table = True
            continue

        match = energy_row_pattern.match(line)
        if match is None:
            continue
        term_order.append(match.group("label").strip())
    return term_order


def resolve_energy_output_order(stdout_text: str, requested_term_names: tuple[str, ...]) -> tuple[str, ...]:
    stdout_term_order = parse_energy_stdout_term_order(stdout_text)
    requested_by_canonical = {
        canonical_energy_term_label(term_name): term_name for term_name in requested_term_names
    }
    resolved = []
    for stdout_label in stdout_term_order:
        requested_name = requested_by_canonical.get(canonical_energy_term_label(stdout_label))
        if requested_name is not None:
            resolved.append(requested_name)
    if len(resolved) == len(requested_term_names) and len(set(resolved)) == len(requested_term_names):
        return tuple(resolved)
    return requested_term_names


def extract_named_energy_series_detail(
    gmx_bin: str,
    work_dir: Path,
    deffnm: str,
    term_names: tuple[str, ...],
    output_stem: str,
    commands_log: list[str],
    label_prefix: str,
) -> dict[str, Any]:
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
    stdout_path = work_dir / f"{label_prefix}_energy_terms.stdout.txt"
    stderr_path = work_dir / f"{label_prefix}_energy_terms.stderr.txt"
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")

    resolved_output_order = resolve_energy_output_order(process.stdout, term_names)
    rows = []
    legacy_rows = []
    raw_rows = []
    for raw_line in (work_dir / f"{output_stem}.xvg").read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", "@")):
            continue
        values = [float(token) for token in stripped.split()]
        raw_rows.append(values)
        legacy_row = {"time_ps": values[0]}
        corrected_row = {"time_ps": values[0]}
        for index, term_name in enumerate(term_names, start=1):
            legacy_row[term_name] = values[index]
        for index, term_name in enumerate(resolved_output_order, start=1):
            corrected_row[term_name] = values[index]
        rows.append(corrected_row)
        legacy_rows.append(legacy_row)

    return {
        "rows": rows,
        "legacy_rows": legacy_rows,
        "raw_rows": raw_rows,
        "resolved_output_order": resolved_output_order,
        "stdout_term_order": parse_energy_stdout_term_order(process.stdout),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


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
    return extract_named_energy_series_detail(
        gmx_bin, work_dir, deffnm, term_names, output_stem, commands_log, label_prefix
    )["rows"]


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


def dense_upstream_ownership_handoff_trace(dense_fixture_summary: dict[str, Any]) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    exact_dir = Path(coarse["exact_work_dir"])

    topology_path = exact_dir / "system.top"
    topology_sources = parse_topology_pair_sources(topology_path)
    topology_nrexcl = parse_topology_nrexcl(topology_path)

    generate_rows = parse_key_value_text(exact_dir / "step0_grompp_generate_excl_trace.txt")
    runtime_rows = parse_key_value_text(exact_dir / "step0_runtime_exclusions_input.txt")
    bit_clear_rows = parse_key_value_text(exact_dir / "step0_exclusion_bit_clear_trace.txt")
    append_rows = parse_key_value_text(exact_dir / "step0_append_branch_trace.txt")
    builder_rows = parse_key_value_text(exact_dir / "step0_pairlist_builder_append_trace.txt")
    preview_rows = parse_key_value_text(exact_dir / "step0_plain_pairlist_preview.txt")
    membership_rows = parse_key_value_text(exact_dir / "step0_pair_key_membership_scan.txt")

    target_pair = (0, 1)
    control_pair = (0, 4)
    target_sources = topology_sources.get(target_pair, [])
    control_sources = topology_sources.get(control_pair, [])

    generate_row = next((row for row in generate_rows if row.get("stage") == "generate_excl_output"), None)
    runtime_row = next((row for row in runtime_rows if row.get("stage") == "runtime_exclusions_input"), None)
    bit_clear_row = next(
        (
            row
            for row in bit_clear_rows
            if row.get("stage") == "runtime_clear_exclusion_bit"
            and row.get("atom_i") == "0"
            and row.get("atom_j") == "1"
        ),
        None,
    )
    target_append_row = next((row for row in append_rows if row.get("role") == "target_pair_0_1"), None)
    control_append_row = next((row for row in append_rows if row.get("role") == "control_pair_0_4"), None)
    target_builder_row = next(
        (row for row in builder_rows if row.get("kind") == "excludedPairs" and row.get("ordinal") == "0"),
        None,
    )
    control_builder_row = next(
        (row for row in builder_rows if row.get("kind") == "pairs" and row.get("ordinal") == "0"),
        None,
    )
    target_preview_row = next(
        (row for row in preview_rows if row.get("kind") == "excludedPairs" and row.get("ordinal") == "0"),
        None,
    )
    control_preview_row = next(
        (row for row in preview_rows if row.get("kind") == "pairs" and row.get("ordinal") == "0"),
        None,
    )
    target_membership_row = next(
        (row for row in membership_rows if row.get("kind") == "excluded" and row.get("ordinal") == "0"),
        None,
    )
    control_membership_row = next(
        (row for row in membership_rows if row.get("kind") == "pairs" and row.get("ordinal") == "0"),
        None,
    )

    generate_contains_target = parse_bool_text(None if generate_row is None else generate_row.get("contains_target"))
    generate_contains_control = parse_bool_text(None if generate_row is None else generate_row.get("contains_control"))
    runtime_contains_target = parse_bool_text(None if runtime_row is None else runtime_row.get("contains_target"))
    runtime_contains_control = parse_bool_text(None if runtime_row is None else runtime_row.get("contains_control"))

    bit_clear_ok = (
        bit_clear_row is not None
        and bit_clear_row.get("rule") == "jAtom_in_topology_exclusions"
        and int(bit_clear_row.get("masked_before", "0")) != 0
        and int(bit_clear_row.get("masked_after", "0")) == 0
    )
    target_append_ok = (
        target_append_row is not None
        and target_append_row.get("branch") == "excludedPairs"
        and target_append_row.get("predicate_mask_nonzero") == "false"
        and target_append_row.get("predicate_excluded_branch") == "true"
        and target_append_row.get("masked_value") == "0"
    )
    control_append_ok = (
        control_append_row is not None
        and control_append_row.get("branch") == "pairs"
        and control_append_row.get("predicate_mask_nonzero") == "true"
        and control_append_row.get("predicate_excluded_branch") == "false"
        and control_append_row.get("masked_value") != "0"
    )
    target_builder_ok = (
        target_builder_row is not None
        and target_builder_row.get("ai") == "0"
        and target_builder_row.get("aj") == "1"
        and target_builder_row.get("excl_bit") == "0"
    )
    control_builder_ok = (
        control_builder_row is not None
        and control_builder_row.get("ai") == "0"
        and control_builder_row.get("aj") == "4"
        and control_builder_row.get("excl_bit") == "1"
    )
    target_membership_ok = (
        target_membership_row is not None
        and target_membership_row.get("in_plain_pairs") == "0"
        and target_membership_row.get("in_plain_excluded") == "1"
    )
    control_membership_ok = (
        control_membership_row is not None
        and control_membership_row.get("in_plain_pairs") == "1"
        and control_membership_row.get("in_plain_excluded") == "0"
        and control_membership_row.get("in_debug_listed_pair_keys") == "false"
    )
    target_preview_ok = (
        target_preview_row is not None
        and target_preview_row.get("ai") == "0"
        and target_preview_row.get("aj") == "1"
    )
    control_preview_ok = (
        control_preview_row is not None
        and control_preview_row.get("ai") == "0"
        and control_preview_row.get("aj") == "4"
    )

    dual_membership_at_generate = bool(target_sources) and generate_contains_target
    runtime_copy_matches_generate = (
        generate_row is not None
        and runtime_row is not None
        and set(parse_index_csv(runtime_row.get("exclusions"))) == set(parse_index_csv(generate_row.get("exclusions")))
    )
    earliest_bad_handoff_identified = (
        "bond" in target_sources
        and topology_nrexcl == 3
        and dual_membership_at_generate
        and not generate_contains_control
        and runtime_contains_target
        and not runtime_contains_control
        and runtime_copy_matches_generate
        and bit_clear_ok
        and target_append_ok
        and control_append_ok
        and target_builder_ok
        and control_builder_ok
        and target_preview_ok
        and control_preview_ok
        and target_membership_ok
        and control_membership_ok
        and not control_sources
    )

    localization = {
        "classification": (
            "earliest bad handoff identified at generate_excl output from bonded topology + nrexcl overlap policy"
            if earliest_bad_handoff_identified
            else "still unresolved"
        ),
        "target_pair": list(target_pair),
        "control_pair": list(control_pair),
        "topology": {
            "nrexcl": topology_nrexcl,
            "target_topology_sources": target_sources,
            "control_topology_sources": control_sources,
            "topology_path": str(topology_path),
        },
        "lineage_trace": [
            {
                "order": 0,
                "stage": "topology_inputs",
                "state": {
                    "target_pair_0_1_sources": target_sources,
                    "control_pair_0_4_sources": control_sources,
                    "nrexcl": topology_nrexcl,
                },
            },
            {
                "order": 1,
                "stage": "generate_excl_output",
                "state": generate_row,
            },
            {
                "order": 2,
                "stage": "runtime_exclusions_input",
                "state": runtime_row,
            },
            {
                "order": 3,
                "stage": "runtime_clear_exclusion_bit",
                "state": bit_clear_row,
            },
            {
                "order": 4,
                "stage": "append_plain_pairlist_branch_target",
                "state": target_append_row,
            },
            {
                "order": 5,
                "stage": "append_plain_pairlist_branch_control",
                "state": control_append_row,
            },
            {
                "order": 6,
                "stage": "plain_pairlist_append_target",
                "state": target_builder_row,
            },
            {
                "order": 7,
                "stage": "plain_pairlist_membership_target",
                "state": target_membership_row,
            },
        ],
        "earliest_divergence": {
            "stage": "generate_excl_output" if dual_membership_at_generate else None,
            "fault_class": (
                "upstream_spec_overlap_policy_defect"
                if earliest_bad_handoff_identified
                else None
            ),
            "why": (
                "Pair (0,1) is already bonded in topology, and generate_excl materializes it into the atom-0 exclusion list under nrexcl=3 before any runtime bit clearing or excludedPairs packing happens."
                if dual_membership_at_generate
                else "Target pair did not prove dual-membership at generate_excl output."
            ),
        },
        "append_branch_proof": {
            "target_row": target_append_row,
            "control_row": control_append_row,
            "target_branch_provenance": {
                "mask_bit_cleared_by_runtime_exclusions": bit_clear_ok,
                "upstream_runtime_exclusions_contains_target": runtime_contains_target,
                "predicate_mask_nonzero": None if target_append_row is None else target_append_row.get("predicate_mask_nonzero"),
                "predicate_excluded_branch": None if target_append_row is None else target_append_row.get("predicate_excluded_branch"),
            },
        },
        "dual_membership": {
            "target_pair_has_bonded_topology_source": bool(target_sources),
            "target_pair_in_generate_excl_output": generate_contains_target,
            "dual_membership_first_materialized_at_generate_excl": dual_membership_at_generate,
            "target_membership_row": target_membership_row,
        },
        "known_good_control": {
            "pair": list(control_pair),
            "generate_contains_control": generate_contains_control,
            "runtime_contains_control": runtime_contains_control,
            "append_branch_row": control_append_row,
            "builder_row": control_builder_row,
            "preview_row": control_preview_row,
            "membership_row": control_membership_row,
            "clean_lineage": (
                not control_sources
                and not generate_contains_control
                and not runtime_contains_control
                and control_append_ok
                and control_builder_ok
                and control_preview_ok
                and control_membership_ok
            ),
        },
        "artifact_paths": {
            "generate_excl_trace": str(exact_dir / "step0_grompp_generate_excl_trace.txt"),
            "runtime_exclusions_input": str(exact_dir / "step0_runtime_exclusions_input.txt"),
            "exclusion_bit_clear_trace": str(exact_dir / "step0_exclusion_bit_clear_trace.txt"),
            "append_branch_trace": str(exact_dir / "step0_append_branch_trace.txt"),
            "pairlist_builder_trace": str(exact_dir / "step0_pairlist_builder_append_trace.txt"),
            "plain_pairlist_preview": str(exact_dir / "step0_plain_pairlist_preview.txt"),
            "pair_key_membership_scan": str(exact_dir / "step0_pair_key_membership_scan.txt"),
        },
        "supports_exact_earliest_handoff": earliest_bad_handoff_identified,
        "why_not_fully_closed": (
            "This closes the earliest bad handoff only for dense_oligomer, exact 3-level, coarse dt=0.0005, step 0."
            if earliest_bad_handoff_identified
            else "The lineage does not yet prove an exact earliest upstream handoff."
        ),
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "localization": localization,
    }


def dense_pair_rule_derivation_trace(dense_fixture_summary: dict[str, Any]) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    exact_dir = Path(coarse["exact_work_dir"])

    topology_path = exact_dir / "system.top"
    topology_sources = parse_topology_pair_sources(topology_path)
    topology_nrexcl = parse_topology_nrexcl(topology_path)

    internal_rows = parse_key_value_text(exact_dir / "step0_generate_excl_internal_trace.txt")
    generate_rows = parse_key_value_text(exact_dir / "step0_grompp_generate_excl_trace.txt")
    runtime_rows = parse_key_value_text(exact_dir / "step0_runtime_exclusions_input.txt")

    target_pair = (0, 1)
    control_pair = (0, 4)
    target_sources = topology_sources.get(target_pair, [])
    control_sources = topology_sources.get(control_pair, [])

    gen_nnb_row = next((row for row in internal_rows if row.get("stage") == "gen_nnb_bond_membership"), None)
    direct_rule_row = next(
        (
            row
            for row in internal_rows
            if row.get("stage") == "do_gen_direct_bond_rule_fire"
            and row.get("ai") == "0"
            and row.get("aj") == "1"
        ),
        None,
    )
    summary_row = next((row for row in internal_rows if row.get("stage") == "do_gen_summary"), None)
    sort_row = next(
        (
            row
            for row in internal_rows
            if row.get("stage") == "sort_and_purge_output" and row.get("atom") == "0"
        ),
        None,
    )
    flattened_row = next(
        (
            row
            for row in internal_rows
            if row.get("stage") == "nnb2excl_flattened" and row.get("atom") == "0"
        ),
        None,
    )
    control_check_row = next(
        (
            row
            for row in internal_rows
            if row.get("stage") == "nnb2excl_control_check" and row.get("atom") == "0"
        ),
        None,
    )
    emit_row = next(
        (
            row
            for row in internal_rows
            if row.get("stage") == "nnb2excl_emit" and row.get("atom") == "0"
        ),
        None,
    )

    generate_row = next((row for row in generate_rows if row.get("stage") == "generate_excl_output"), None)
    runtime_row = next((row for row in runtime_rows if row.get("stage") == "runtime_exclusions_input"), None)

    direct_rule_fire_proven = (
        direct_rule_row is not None
        and direct_rule_row.get("source") == "exclude_all_bonded_atoms"
        and direct_rule_row.get("condition_nrex_positive") == "true"
        and direct_rule_row.get("nre_bucket") == "1"
        and direct_rule_row.get("add_nnb_called") == "true"
    )
    target_higher_order_rule_fired = parse_bool_text(
        None if summary_row is None else summary_row.get("target_higher_order_rule_fired")
    )
    control_higher_order_rule_fired = parse_bool_text(
        None if summary_row is None else summary_row.get("control_higher_order_rule_fired")
    )
    target_direct_count = 0 if gen_nnb_row is None else int(gen_nnb_row.get("target_count", "0"))
    control_direct_count = 0 if gen_nnb_row is None else int(gen_nnb_row.get("control_count", "0"))

    sort_level1 = [] if sort_row is None else parse_index_csv(sort_row.get("level1"))
    sort_level2 = [] if sort_row is None else parse_index_csv(sort_row.get("level2"))
    sort_level3 = [] if sort_row is None else parse_index_csv(sort_row.get("level3"))
    emitted_indices = [] if emit_row is None else parse_index_csv(emit_row.get("emitted"))
    generated_indices = [] if generate_row is None else parse_index_csv(generate_row.get("exclusions"))
    runtime_indices = [] if runtime_row is None else parse_index_csv(runtime_row.get("exclusions"))

    target_in_sort_level1 = sort_row is not None and sort_row.get("target_in_level1") == "true"
    control_in_sort_any = sort_row is not None and sort_row.get("control_in_any_level") == "true"
    target_in_emit = emit_row is not None and emit_row.get("target_present") == "true"
    control_in_emit = emit_row is not None and emit_row.get("control_present") == "true"
    target_in_generate = generate_row is not None and generate_row.get("contains_target") == "true"
    control_in_generate = generate_row is not None and generate_row.get("contains_control") == "true"
    target_in_runtime = runtime_row is not None and runtime_row.get("contains_target") == "true"
    control_in_runtime = runtime_row is not None and runtime_row.get("contains_control") == "true"

    continuity_ok = (
        target_in_emit
        and target_in_generate
        and target_in_runtime
        and set(emitted_indices) == set(generated_indices)
        and set(generated_indices) == set(runtime_indices)
    )
    control_clean = (
        not control_sources
        and control_direct_count == 0
        and not control_higher_order_rule_fired
        and not control_in_sort_any
        and not control_in_emit
        and not control_in_generate
        and not control_in_runtime
    )

    baseline_intended = (
        "bond" in target_sources
        and topology_nrexcl == 3
        and direct_rule_fire_proven
        and target_direct_count > 0
        and not target_higher_order_rule_fired
        and target_in_sort_level1
        and target_pair[1] in sort_level1
        and target_pair[1] not in sort_level2
        and target_pair[1] not in sort_level3
        and continuity_ok
        and not control_in_generate
        and control_clean
    )
    semantically_inconsistent = (
        direct_rule_fire_proven
        and (
            "bond" not in target_sources
            or topology_nrexcl is None
            or topology_nrexcl <= 0
        )
    ) or (
        not direct_rule_fire_proven
        and target_in_emit
        and target_in_generate
    )

    policy_interpretation_verdict = (
        "BASELINE-INTENDED"
        if baseline_intended
        else ("SEMANTICALLY-INCONSISTENT" if semantically_inconsistent else "PARTIAL")
    )
    earliest_bad_handoff_verdict = (
        "NOT-HERE"
        if baseline_intended
        else ("PASS-CANDIDATE" if semantically_inconsistent else "PARTIAL")
    )

    localization = {
        "policy_interpretation_verdict": policy_interpretation_verdict,
        "earliest_bad_handoff_verdict": earliest_bad_handoff_verdict,
        "target_pair": list(target_pair),
        "control_pair": list(control_pair),
        "topology": {
            "nrexcl": topology_nrexcl,
            "target_topology_sources": target_sources,
            "control_topology_sources": control_sources,
            "topology_path": str(topology_path),
        },
        "internal_rule_trace": [
            {
                "order": 0,
                "stage": "gen_nnb_bond_membership",
                "state": gen_nnb_row,
            },
            {
                "order": 1,
                "stage": "do_gen_direct_bond_rule_fire",
                "state": direct_rule_row,
            },
            {
                "order": 2,
                "stage": "do_gen_summary",
                "state": summary_row,
            },
            {
                "order": 3,
                "stage": "sort_and_purge_output",
                "state": sort_row,
            },
            {
                "order": 4,
                "stage": "nnb2excl_flattened",
                "state": flattened_row,
            },
            {
                "order": 5,
                "stage": "nnb2excl_control_check",
                "state": control_check_row,
            },
            {
                "order": 6,
                "stage": "nnb2excl_emit",
                "state": emit_row,
            },
        ],
        "exact_rule_fire_proof": {
            "gen_nnb_bond_membership_row": gen_nnb_row,
            "direct_bond_rule_row": direct_rule_row,
            "sort_and_purge_row": sort_row,
            "nnb2excl_emit_row": emit_row,
            "direct_bond_rule_fire_proven": direct_rule_fire_proven,
            "target_higher_order_rule_fired": target_higher_order_rule_fired,
            "target_becomes_level1_neighbor": target_in_sort_level1,
            "target_emitted_into_exclusions": target_in_emit,
        },
        "policy_interpretation": {
            "target_is_bonded_in_topology": "bond" in target_sources,
            "nrexcl": topology_nrexcl,
            "rule_source": None if direct_rule_row is None else direct_rule_row.get("source"),
            "condition_nrex_positive": None
            if direct_rule_row is None
            else direct_rule_row.get("condition_nrex_positive"),
            "interpretation": (
                "Bonded-neighbor exclusion generation is baseline-intended here because the traced rule is "
                "'exclude_all_bonded_atoms' under nrexcl > 0, the target appears as a direct bond, and it is "
                "carried through level-1 exclusions without requiring a higher-order propagation rule."
                if baseline_intended
                else (
                    "The traced rule firing is semantically inconsistent for this exact path."
                    if semantically_inconsistent
                    else "The generate_excl rule meaning is still not exact."
                )
            ),
        },
        "continuity_with_m2g": {
            "generate_excl_output_row": generate_row,
            "runtime_exclusions_input_row": runtime_row,
            "continuity_proven": continuity_ok,
            "generated_indices": generated_indices,
            "runtime_indices": runtime_indices,
            "emitted_indices": emitted_indices,
        },
        "known_good_control": {
            "pair": list(control_pair),
            "gen_nnb_bond_membership_row": gen_nnb_row,
            "do_gen_summary_row": summary_row,
            "nnb2excl_control_check_row": control_check_row,
            "generate_excl_output_row": generate_row,
            "runtime_exclusions_input_row": runtime_row,
            "control_clean": control_clean,
        },
        "supports_exact_rule_fire_point": direct_rule_fire_proven and target_in_emit,
        "supports_generate_excl_not_earliest_bad_handoff": baseline_intended,
        "supports_generate_excl_earliest_bad_handoff": semantically_inconsistent,
        "why_not_fully_closed": (
            "This closes only the pair-specific rule meaning inside generate_excl for dense_oligomer step 0."
            if baseline_intended or semantically_inconsistent
            else "The pair-specific rule derivation is still not exact enough to decide whether generate_excl is the first bad handoff."
        ),
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "localization": localization,
    }


def dense_downstream_misconsumption_trace(dense_fixture_summary: dict[str, Any]) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    exact_dir = Path(coarse["exact_work_dir"])

    topology_path = exact_dir / "system.top"
    topology_sources = parse_topology_pair_sources(topology_path)

    runtime_rows = parse_key_value_text(exact_dir / "step0_runtime_exclusions_input.txt")
    append_rows = parse_key_value_text(exact_dir / "step0_append_branch_trace.txt")
    membership_rows = parse_key_value_text(exact_dir / "step0_pair_key_membership_scan.txt")
    downstream_rows = parse_key_value_text(exact_dir / "step0_downstream_contract_trace.txt")

    target_pair = (0, 1)
    control_pair = (0, 4)
    target_sources = topology_sources.get(target_pair, [])
    control_sources = topology_sources.get(control_pair, [])

    runtime_row = next((row for row in runtime_rows if row.get("stage") == "runtime_exclusions_input"), None)
    target_append_row = next((row for row in append_rows if row.get("role") == "target_pair_0_1"), None)
    control_append_row = next((row for row in append_rows if row.get("role") == "control_pair_0_4"), None)
    target_membership_row = next(
        (row for row in membership_rows if row.get("kind") == "excluded" and row.get("ordinal") == "0"),
        None,
    )
    control_membership_row = next(
        (row for row in membership_rows if row.get("kind") == "pairs" and row.get("ordinal") == "0"),
        None,
    )
    excluded_dispatch_row = next(
        (row for row in downstream_rows if row.get("stage") == "excluded_pairs_dispatch_contract"),
        None,
    )
    pairs_dispatch_row = next(
        (row for row in downstream_rows if row.get("stage") == "pairs_dispatch_contract"),
        None,
    )
    target_consumer_eval_row = next(
        (
            row
            for row in downstream_rows
            if row.get("stage") == "consumer_pair_eval"
            and row.get("pair_list") == "excludedPairs"
            and row.get("ai") == "0"
            and row.get("aj") == "1"
        ),
        None,
    )
    control_consumer_eval_row = next(
        (
            row
            for row in downstream_rows
            if row.get("stage") == "consumer_pair_eval"
            and row.get("pair_list") == "pairs"
            and row.get("ai") == "0"
            and row.get("aj") == "4"
        ),
        None,
    )

    target_runtime_excluded = runtime_row is not None and runtime_row.get("contains_target") == "true"
    target_reference_consumer_eligible = (
        target_append_row is not None
        and target_append_row.get("predicate_mask_nonzero") == "true"
        and target_append_row.get("branch") == "pairs"
    )
    target_membership_legal = (
        target_append_row is not None
        and target_append_row.get("branch") == "excludedPairs"
        and target_append_row.get("masked_value") == "0"
        and target_membership_row is not None
        and target_membership_row.get("in_plain_excluded") == "1"
        and target_membership_row.get("in_plain_pairs") == "0"
    )
    target_excluded_dispatch_admits = (
        excluded_dispatch_row is not None
        and excluded_dispatch_row.get("target_in_list") == "true"
        and excluded_dispatch_row.get("target_ordinal") == "0"
        and excluded_dispatch_row.get("include_rule") == "always_true"
    )
    target_outer_write_eligible = (
        target_consumer_eval_row is not None
        and target_consumer_eval_row.get("include_pair") == "true"
        and target_consumer_eval_row.get("outer_force_write_eligible") == "true"
        and abs(float(target_consumer_eval_row.get("outer_scalar", "0"))) > 0.0
        and abs(float(target_consumer_eval_row.get("correction_scalar", "0"))) > 0.0
    )

    control_reference_consumer_eligible = (
        control_append_row is not None
        and control_append_row.get("predicate_mask_nonzero") == "true"
        and control_append_row.get("branch") == "pairs"
    )
    control_pairs_dispatch_admits = (
        pairs_dispatch_row is not None
        and pairs_dispatch_row.get("control_in_list") == "true"
        and pairs_dispatch_row.get("control_ordinal") == "0"
        and pairs_dispatch_row.get("include_rule") == "always_true"
    )
    control_outer_write_eligible = (
        control_consumer_eval_row is not None
        and control_consumer_eval_row.get("include_pair") == "true"
        and control_consumer_eval_row.get("outer_force_write_eligible") == "true"
        and control_consumer_eval_row.get("pair_list") == "pairs"
    )
    control_clean = (
        not control_sources
        and control_reference_consumer_eligible
        and control_pairs_dispatch_admits
        and control_outer_write_eligible
        and control_membership_row is not None
        and control_membership_row.get("in_plain_pairs") == "1"
        and control_membership_row.get("in_plain_excluded") == "0"
    )

    first_bad_site_found = (
        target_runtime_excluded
        and not target_reference_consumer_eligible
        and target_membership_legal
        and target_excluded_dispatch_admits
        and target_outer_write_eligible
        and control_clean
    )

    contract_trace = [
        {
            "order": 0,
            "stage": "runtime_exclusions_input",
            "exclusion_membership_state": "target_in_runtime_exclusions=true",
            "bonded_listed_ownership_state": f"topology_sources={','.join(target_sources) if target_sources else 'none'}; listed_pair_key=false",
            "physical_force_consumer_eligibility": {
                "reference_semantics": False,
                "exact_semantics": False,
            },
            "verdict": "still_legal",
            "evidence": runtime_row,
        },
        {
            "order": 1,
            "stage": "append_plain_pairlist_branch",
            "exclusion_membership_state": "branch=excludedPairs; masked_value=0",
            "bonded_listed_ownership_state": f"topology_sources={','.join(target_sources) if target_sources else 'none'}; listed_pair_key=false",
            "physical_force_consumer_eligibility": {
                "reference_semantics": False,
                "exact_semantics": False,
            },
            "verdict": "still_legal",
            "evidence": target_append_row,
        },
        {
            "order": 2,
            "stage": "plain_pairlist_membership",
            "exclusion_membership_state": "in_plain_excluded=1; in_plain_pairs=0",
            "bonded_listed_ownership_state": f"topology_sources={','.join(target_sources) if target_sources else 'none'}; listed_pair_key="
            + ("true" if target_membership_row is not None and target_membership_row.get("in_debug_listed_pair_keys") == "true" else "false"),
            "physical_force_consumer_eligibility": {
                "reference_semantics": False,
                "exact_semantics": False,
            },
            "verdict": "still_legal",
            "evidence": target_membership_row,
        },
        {
            "order": 3,
            "stage": "exact_excludedPairs_dispatch_contract",
            "exclusion_membership_state": "excludedPairs consumer dispatch target_in_list=true include_rule=always_true",
            "bonded_listed_ownership_state": f"topology_sources={','.join(target_sources) if target_sources else 'none'}; listed_pair_key=false",
            "physical_force_consumer_eligibility": {
                "reference_semantics": False,
                "exact_semantics": True,
            },
            "verdict": "first_bad_interpretation" if first_bad_site_found else "still_suspect",
            "evidence": excluded_dispatch_row,
        },
        {
            "order": 4,
            "stage": "excludedPairs_outer_consumer_eval",
            "exclusion_membership_state": "excludedPairs pair-specific evaluation",
            "bonded_listed_ownership_state": f"topology_sources={','.join(target_sources) if target_sources else 'none'}; listed_pair_key=false",
            "physical_force_consumer_eligibility": {
                "reference_semantics": False,
                "exact_semantics": target_outer_write_eligible,
            },
            "verdict": "confirms_first_bad_site" if first_bad_site_found else "still_suspect",
            "evidence": target_consumer_eval_row,
        },
    ]

    reference_reconciliation = {
        "target_pair": list(target_pair),
        "reference_semantics": {
            "source": "runtime exclusion mask + append branch contract on the same run",
            "masked_value": None if target_append_row is None else target_append_row.get("masked_value"),
            "branch": None if target_append_row is None else target_append_row.get("branch"),
            "physical_nonbonded_consumer_eligible": target_reference_consumer_eligible,
        },
        "exact_semantics": {
            "dispatch_stage": None if excluded_dispatch_row is None else excluded_dispatch_row.get("stage"),
            "dispatch_include_rule": None if excluded_dispatch_row is None else excluded_dispatch_row.get("include_rule"),
            "dispatch_semantic_role": None if excluded_dispatch_row is None else excluded_dispatch_row.get("semantic_role"),
            "pair_eval_stage": None if target_consumer_eval_row is None else target_consumer_eval_row.get("stage"),
            "outer_force_write_eligible": None
            if target_consumer_eval_row is None
            else target_consumer_eval_row.get("outer_force_write_eligible"),
            "outer_scalar": None if target_consumer_eval_row is None else target_consumer_eval_row.get("outer_scalar"),
            "correction_scalar": None
            if target_consumer_eval_row is None
            else target_consumer_eval_row.get("correction_scalar"),
        },
        "divergence": (
            "Reference semantics keeps pair (0,1) out of the standard physical nonbonded consumer because the exclusion mask clears the interaction bit, "
            "but the exact 3-level path re-admits the same pair via excludedPairs with include_rule=always_true and a non-zero outer correction force."
            if first_bad_site_found
            else "The exact downstream divergence is still not exact."
        ),
    }

    localization = {
        "earliest_semantic_misuse_verdict": "FIRST-BAD-SITE-FOUND" if first_bad_site_found else "NOT-YET",
        "first_bad_site": None
        if not first_bad_site_found
        else {
            "stage": "exact_excludedPairs_dispatch_contract",
            "code_path": str(REPO_ROOT / "src" / "gromacs" / "mdlib" / "sim_util.cpp"),
            "why": (
                "This is the first downstream stage where a valid exclusion-membership container is admitted into the exact nonbonded consumer path with include_rule=always_true, "
                "before the pair-specific outer correction write makes the physical misuse concrete."
            ),
        },
        "exact_bad_handoff_proof": {
            "dispatch_row": excluded_dispatch_row,
            "pair_eval_row": target_consumer_eval_row,
            "target_membership_row": target_membership_row,
            "target_append_row": target_append_row,
        },
        "pair_contract_trace": contract_trace,
        "reference_reconciliation": reference_reconciliation,
        "control_result": {
            "pair": list(control_pair),
            "append_row": control_append_row,
            "membership_row": control_membership_row,
            "dispatch_row": pairs_dispatch_row,
            "pair_eval_row": control_consumer_eval_row,
            "control_clean": control_clean,
            "why_clean": (
                "Control pair (0,4) remains in the standard pairs consumer path with mask-preserved eligibility and does not cross from exclusion bookkeeping into a separate correction consumer."
            ),
        },
        "artifact_paths": {
            "runtime_exclusions_input": str(exact_dir / "step0_runtime_exclusions_input.txt"),
            "append_branch_trace": str(exact_dir / "step0_append_branch_trace.txt"),
            "pair_key_membership_scan": str(exact_dir / "step0_pair_key_membership_scan.txt"),
            "downstream_contract_trace": str(exact_dir / "step0_downstream_contract_trace.txt"),
            "topology_path": str(topology_path),
        },
        "supports_first_bad_site": first_bad_site_found,
        "why_not_fully_closed": (
            "This identifies only the first downstream semantic misuse site for pair (0,1) on dense_oligomer coarse step 0."
            if first_bad_site_found
            else "The downstream runtime stages are narrowed, but the first semantic misuse site is still not exact."
        ),
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "localization": localization,
    }


def load_dispatch_probe_state(work_dir: Path, role: str) -> dict[str, Any]:
    rows = parse_key_value_text(work_dir / "step0_dispatch_internal_trace.txt")
    include_row = next(
        (row for row in rows if row.get("stage") == "dispatch_internal_include_pair" and row.get("role") == role),
        None,
    )
    active_row = next(
        (
            row
            for row in rows
            if row.get("stage") == "dispatch_internal_active_contributions" and row.get("role") == role
        ),
        None,
    )
    routing_row = next(
        (
            row
            for row in rows
            if row.get("stage") == "dispatch_internal_outer_routing" and row.get("role") == role
        ),
        None,
    )
    admitted = include_row is not None and include_row.get("include_pair_effective") == "true"
    effective_outer_active = active_row is not None and active_row.get("effective_outer_active") == "true"
    effective_outer_scalar = 0.0 if active_row is None else float(active_row.get("outer_scalar_effective", "0"))
    first_bad_semantics_occurs = admitted and effective_outer_active and abs(effective_outer_scalar) > 0.0
    physical_outer_realization = (
        routing_row is not None and routing_row.get("actual_outer_write_executed") == "true"
    )
    return {
        "work_dir": str(work_dir),
        "include_row": include_row,
        "active_row": active_row,
        "routing_row": routing_row,
        "admitted": admitted,
        "effective_outer_active": effective_outer_active,
        "effective_outer_scalar": effective_outer_scalar,
        "first_bad_semantics_occurs": first_bad_semantics_occurs,
        "physical_outer_realization": physical_outer_realization,
    }


def load_bookkeeping_trace_state(work_dir: Path, role: str) -> dict[str, Any]:
    rows = parse_key_value_text(work_dir / "step0_patch_b_bookkeeping_trace.txt")
    raw_row = next(
        (row for row in rows if row.get("stage") == "bookkeeping_raw_state" and row.get("role") == role),
        None,
    )
    force_row = next(
        (row for row in rows if row.get("stage") == "bookkeeping_force_state" and row.get("role") == role),
        None,
    )
    energy_row = next(
        (row for row in rows if row.get("stage") == "bookkeeping_energy_sink" and row.get("role") == role),
        None,
    )
    raw_scalar_present = raw_row is not None and parse_bool_text(raw_row.get("raw_scalar_present"))
    effective_outer_scalar = 0.0 if force_row is None else float(force_row.get("effective_outer_scalar", "0"))
    outer_force_write = force_row is not None and parse_bool_text(force_row.get("actual_outer_write_executed"))
    bookkeeping_sink_active = (
        energy_row is not None and parse_bool_text(energy_row.get("accumulate_energy_effective"))
    )
    residual_visible = energy_row is not None and parse_bool_text(energy_row.get("residual_visible"))
    return {
        "work_dir": str(work_dir),
        "raw_row": raw_row,
        "force_row": force_row,
        "energy_row": energy_row,
        "raw_scalar_present": raw_scalar_present,
        "effective_outer_scalar": effective_outer_scalar,
        "outer_force_write": outer_force_write,
        "bookkeeping_sink_active": bookkeeping_sink_active,
        "residual_visible": residual_visible,
    }


def load_bookkeeping_reciprocal_row(work_dir: Path) -> dict[str, str] | None:
    rows = parse_key_value_text(work_dir / "step0_patch_b_bookkeeping_trace.txt")
    return next((row for row in rows if row.get("stage") == "bookkeeping_reciprocal_sink"), None)


def load_reciprocal_internal_trace_rows(work_dir: Path) -> dict[str, dict[str, str]]:
    rows = parse_key_value_text(work_dir / "step0_reciprocal_internal_trace.txt")
    return {row["stage"]: row for row in rows if "stage" in row}


def load_post_final_ledger_trace_rows(work_dir: Path) -> list[dict[str, str]]:
    return [row for row in parse_key_value_text(work_dir / "step0_post_final_ledger_trace.txt") if "stage" in row]


def load_lj_sr_internal_trace_rows(work_dir: Path) -> list[dict[str, str]]:
    return [row for row in parse_key_value_text(work_dir / "step0_lj_sr_internal_trace.txt") if "stage" in row]


def load_aligned_event_identity_rows(work_dir: Path) -> dict[str, dict[str, str]]:
    path = work_dir / "step0_aligned_event_identity_trace.txt"
    if not path.exists():
        return {}
    rows = [row for row in parse_key_value_text(path) if "stage" in row]
    first_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        stage = row["stage"]
        if stage not in first_rows:
            first_rows[stage] = row
    return first_rows


def load_event_669_geometry_rows(work_dir: Path) -> dict[str, dict[str, str]]:
    path = work_dir / "step0_event_669_geometry_trace.txt"
    if not path.exists():
        return {}
    rows = [row for row in parse_key_value_text(path) if "stage" in row]
    first_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        stage = row["stage"]
        if stage not in first_rows:
            first_rows[stage] = row
    return first_rows


def load_coulomb_source_truth_rows(work_dir: Path) -> dict[str, dict[str, str]]:
    path = work_dir / "step0_coulomb_source_truth_trace.txt"
    if not path.exists():
        return {}
    rows = [row for row in parse_key_value_text(path) if "variable" in row]
    first_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        variable = row["variable"]
        if variable not in first_rows:
            first_rows[variable] = row
    return first_rows


def load_lj_source_truth_rows(work_dir: Path) -> dict[str, dict[str, str]]:
    path = work_dir / "step0_lj_source_truth_trace.txt"
    if not path.exists():
        return {}
    rows = [row for row in parse_key_value_text(path) if "variable" in row]
    first_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        variable = row["variable"]
        if variable not in first_rows:
            first_rows[variable] = row
    return first_rows


def load_potential_ledger_trace_row(work_dir: Path) -> dict[str, str] | None:
    path = work_dir / "step0_potential_ledger_trace.txt"
    if not path.exists():
        return None
    rows = [row for row in parse_key_value_text(path) if row.get("stage") == "FINAL_INTERNAL_LEDGER"]
    return rows[0] if rows else None


def trim_lj_sr_trace_to_first_cycle(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    trimmed_rows: list[dict[str, str]] = []
    for row in rows:
        trimmed_rows.append(row)
        if row.get("stage") == "FINAL_INTERNAL_LEDGER":
            break
    return trimmed_rows


def parse_float_field(row: dict[str, str] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def aggregate_lj_sr_stage(rows: list[dict[str, str]], stage: str) -> dict[str, Any]:
    stage_rows = [row for row in rows if row.get("stage") == stage]
    if not stage_rows:
        return {
            "stage": stage,
            "rows": [],
            "lj_sr": None,
            "coulomb_sr": None,
            "code_locations": [],
            "execution_paths": [],
        }

    lj_values = [float(row["lj_sr"]) for row in stage_rows if row.get("lj_sr") is not None]
    coul_values = [float(row["coulomb_sr"]) for row in stage_rows if row.get("coulomb_sr") is not None]
    lj_total = None if len(lj_values) != len(stage_rows) else sum(lj_values)
    coul_total = None if len(coul_values) != len(stage_rows) else sum(coul_values)

    return {
        "stage": stage,
        "rows": stage_rows,
        "lj_sr": lj_total,
        "coulomb_sr": coul_total,
        "code_locations": sorted({row.get("code_location", "") for row in stage_rows if row.get("code_location")}),
        "execution_paths": sorted({row.get("execution_path", "") for row in stage_rows if row.get("execution_path")}),
    }


def aggregate_lj_sr_stage_with_paths(
    rows: list[dict[str, str]], stage: str, execution_path_prefixes: tuple[str, ...]
) -> dict[str, Any]:
    filtered_rows = [
        row
        for row in rows
        if row.get("stage") == stage
        and (
            not execution_path_prefixes
            or any((row.get("execution_path") or "").startswith(prefix) for prefix in execution_path_prefixes)
        )
    ]
    return aggregate_lj_sr_stage(filtered_rows, stage)


def write_ordinal_from_row(row: dict[str, str]) -> int | None:
    value = row.get("write_ordinal")
    if value is not None:
        return int(value)
    stage = row.get("stage", "")
    if stage == "RAW_FIRST_WRITE":
        return 1
    match = re.fullmatch(r"AFTER_WRITE_ORDINAL_(\d+)", stage)
    if match is None:
        return None
    return int(match.group(1))


def collect_write_ordinals(
    rows: list[dict[str, str]], execution_path_prefixes: tuple[str, ...]
) -> list[int]:
    ordinals: set[int] = set()
    for row in rows:
        execution_path = row.get("execution_path") or ""
        if execution_path_prefixes and not any(execution_path.startswith(prefix) for prefix in execution_path_prefixes):
            continue
        write_ordinal = write_ordinal_from_row(row)
        if write_ordinal is not None:
            ordinals.add(write_ordinal)
    return sorted(ordinals)


def aligned_event_ordinal_from_row(row: dict[str, str]) -> int | None:
    value = row.get("aligned_event_ordinal")
    if value is not None:
        return int(value)
    stage = row.get("stage", "")
    match = re.fullmatch(r"ALIGNED_WRITE_EVENT_(\d+)", stage)
    if match is None:
        return None
    return int(match.group(1))


def collect_aligned_event_ordinals(
    rows: list[dict[str, str]], execution_path_prefixes: tuple[str, ...]
) -> list[int]:
    ordinals: set[int] = set()
    for row in rows:
        execution_path = row.get("execution_path") or ""
        if execution_path_prefixes and not any(execution_path.startswith(prefix) for prefix in execution_path_prefixes):
            continue
        aligned_event_ordinal = aligned_event_ordinal_from_row(row)
        if aligned_event_ordinal is not None:
            ordinals.add(aligned_event_ordinal)
    return sorted(ordinals)


def trim_post_final_trace_to_first_export_cycle(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    trimmed_rows: list[dict[str, str]] = []
    for row in rows:
        trimmed_rows.append(row)
        if row.get("stage") == "PRINTSTEP_DO_ENX_INPUT":
            break
    return trimmed_rows


def with_stage_occurrences(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    occurrence_counts: dict[str, int] = {}
    labeled_rows: list[dict[str, Any]] = []
    for row in rows:
        stage = row["stage"]
        occurrence_counts[stage] = occurrence_counts.get(stage, 0) + 1
        labeled_row = dict(row)
        labeled_row["occurrence"] = occurrence_counts[stage]
        labeled_row["stage_label"] = f"{stage}#{occurrence_counts[stage]}"
        labeled_rows.append(labeled_row)
    return labeled_rows


def find_stage_value(rows: list[dict[str, Any]], stage_label: str) -> float | None:
    row = next((candidate for candidate in rows if candidate["stage_label"] == stage_label), None)
    return None if row is None else parse_float_field(row, "value")


def dense_patch_b_post_final_ledger_trace(
    gmx_bin: str,
    dense_fixture_summary: dict[str, Any],
    commands_log: list[str],
) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    patch_b_dir = Path(coarse["patch_b_work_dir"])

    plain_rows = with_stage_occurrences(
        trim_post_final_trace_to_first_export_cycle(load_post_final_ledger_trace_rows(plain_dir))
    )
    patch_b_rows = with_stage_occurrences(
        trim_post_final_trace_to_first_export_cycle(load_post_final_ledger_trace_rows(patch_b_dir))
    )
    plain_terms_detail = extract_named_energy_series_detail(
        gmx_bin, plain_dir, "plain", DIAGNOSTIC_ENERGY_TERMS, "m2n_plain_diag_terms", commands_log, "m2n_plain"
    )
    patch_b_terms_detail = extract_named_energy_series_detail(
        gmx_bin,
        patch_b_dir,
        "exact",
        DIAGNOSTIC_ENERGY_TERMS,
        "m2n_patch_b_diag_terms",
        commands_log,
        "m2n_patch_b",
    )
    plain_terms = plain_terms_detail["rows"][0]
    patch_b_terms = patch_b_terms_detail["rows"][0]

    def aggregate_stage_with_paths(
        rows: list[dict[str, str]], stage: str, execution_path_prefixes: tuple[str, ...]
    ) -> dict[str, Any]:
        filtered_rows = [
            row
            for row in rows
            if row.get("stage") == stage
            and (
                not execution_path_prefixes
                or any((row.get("execution_path") or "").startswith(prefix) for prefix in execution_path_prefixes)
            )
        ]
        return aggregate_lj_sr_stage(filtered_rows, stage)
    plain_legacy_terms = plain_terms_detail["legacy_rows"][0]
    patch_b_legacy_terms = patch_b_terms_detail["legacy_rows"][0]

    stage_inventory = [
        {
            "stage": "FORCECPP_FINAL_LEDGER_WRITE",
            "code_location": "src/gromacs/mdlib/force.cpp:455",
            "candidate_type": "direct_write",
            "contract_identity": "direct_energy_field",
        },
        {
            "stage": "SIM_UTIL_PME_RECEIVE_ADD",
            "code_location": "src/gromacs/mdlib/sim_util.cpp:327",
            "candidate_type": "direct_write",
            "contract_identity": "direct_energy_field",
        },
        {
            "stage": "PME_GPU_REDUCE_ADD",
            "code_location": "src/gromacs/ewald/pme_gpu.cpp:295",
            "candidate_type": "direct_write",
            "contract_identity": "direct_energy_field",
        },
        {
            "stage": "ENERGYOUTPUT_ADDDATA_INPUT",
            "code_location": "src/gromacs/mdlib/energyoutput.cpp:933",
            "candidate_type": "copy_input",
            "contract_identity": "direct_energy_field",
        },
        {
            "stage": "ENERGYOUTPUT_AFTER_ADDVALUES",
            "code_location": "src/gromacs/mdlib/energyoutput.cpp:944",
            "candidate_type": "copy_output",
            "contract_identity": "aliased_container",
        },
        {
            "stage": "PRINTSTEP_DO_ENX_INPUT",
            "code_location": "src/gromacs/mdlib/energyoutput.cpp:1306",
            "candidate_type": "export_input",
            "contract_identity": "exported_field",
        },
        {
            "stage": "FINAL_EDR_EXPORT",
            "code_location": "gmx energy xvg output",
            "candidate_type": "export_output",
            "contract_identity": "exported_field",
        },
        {
            "stage": "ANALYSIS_XVG_COLUMN_MAPPING",
            "code_location": "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series",
            "candidate_type": "export_mapping",
            "contract_identity": "reported_field",
        },
    ]

    plain_stage_set = {row["stage"] for row in plain_rows}
    patch_b_stage_set = {row["stage"] for row in patch_b_rows}
    for candidate in stage_inventory:
        if candidate["stage"] in {"FINAL_EDR_EXPORT", "ANALYSIS_XVG_COLUMN_MAPPING"}:
            candidate["executed_plain"] = True
            candidate["executed_patch_b"] = True
        else:
            candidate["executed_plain"] = candidate["stage"] in plain_stage_set
            candidate["executed_patch_b"] = candidate["stage"] in patch_b_stage_set
        candidate["executed_in_locked_scope"] = candidate["executed_plain"] or candidate["executed_patch_b"]

    executed_branch_plain = next(
        (row.get("reciprocal_branch") for row in plain_rows if row["stage"] == "FORCECPP_FINAL_LEDGER_WRITE"),
        None,
    )
    executed_branch_patch_b = next(
        (row.get("reciprocal_branch") for row in patch_b_rows if row["stage"] == "FORCECPP_FINAL_LEDGER_WRITE"),
        None,
    )

    stage_order = [
        "FORCECPP_FINAL_LEDGER_WRITE",
        "SIM_UTIL_PME_RECEIVE_ADD",
        "PME_GPU_REDUCE_ADD",
        "ENERGYOUTPUT_ADDDATA_INPUT",
        "ENERGYOUTPUT_AFTER_ADDVALUES",
        "PRINTSTEP_DO_ENX_INPUT",
    ]
    occurrence_limits: dict[str, int] = {}
    for stage_name in stage_order:
        plain_count = sum(1 for row in plain_rows if row["stage"] == stage_name)
        patch_b_count = sum(1 for row in patch_b_rows if row["stage"] == stage_name)
        occurrence_limits[stage_name] = max(plain_count, patch_b_count)

    comparison_table: list[dict[str, Any]] = []
    plain_initial_trace = find_stage_value(plain_rows, "FORCECPP_FINAL_LEDGER_WRITE#1")
    patch_b_initial_trace = find_stage_value(patch_b_rows, "FORCECPP_FINAL_LEDGER_WRITE#1")
    final_edr_plain = plain_terms["Coul.-recip."]
    final_edr_patch_b = patch_b_terms["Coul.-recip."]
    legacy_edr_plain = plain_legacy_terms["Coul.-recip."]
    legacy_edr_patch_b = patch_b_legacy_terms["Coul.-recip."]

    first_divergence_stage = None
    first_divergence_reason = None
    earlier_exonerated: list[str] = []
    divergence_seen = False

    for stage_name in stage_order:
        for occurrence in range(1, occurrence_limits[stage_name] + 1):
            stage_label = f"{stage_name}#{occurrence}"
            plain_row = next((row for row in plain_rows if row["stage_label"] == stage_label), None)
            patch_b_row = next((row for row in patch_b_rows if row["stage_label"] == stage_label), None)
            plain_value = None if plain_row is None else parse_float_field(plain_row, "value")
            patch_b_value = None if patch_b_row is None else parse_float_field(patch_b_row, "value")
            delta = (
                None
                if plain_value is None or patch_b_value is None
                else patch_b_value - plain_value
            )

            plain_breaks_trace = (
                plain_value is not None
                and plain_initial_trace is not None
                and stage_label != "FORCECPP_FINAL_LEDGER_WRITE#1"
                and abs(plain_value - plain_initial_trace) > LEDGER_TRACE_TOL
            )
            patch_b_breaks_trace = (
                patch_b_value is not None
                and patch_b_initial_trace is not None
                and stage_label != "FORCECPP_FINAL_LEDGER_WRITE#1"
                and abs(patch_b_value - patch_b_initial_trace) > LEDGER_TRACE_TOL
            )
            if stage_name == "PRINTSTEP_DO_ENX_INPUT":
                traced_equals_edr_contract = (
                    plain_value is not None
                    and patch_b_value is not None
                    and abs(plain_value - final_edr_plain) <= LEDGER_TRACE_TOL
                    and abs(patch_b_value - final_edr_patch_b) <= LEDGER_TRACE_TOL
                )
            else:
                traced_equals_edr_contract = None

            divergence_here = False
            divergence_reason = None
            if (
                stage_label != "FORCECPP_FINAL_LEDGER_WRITE#1"
                and delta is not None
                and abs(delta) > LEDGER_TRACE_TOL
            ):
                divergence_here = True
                divergence_reason = "plain_vs_patch_b_delta_nonzero"
            elif plain_breaks_trace or patch_b_breaks_trace:
                divergence_here = True
                divergence_reason = "traced_field_ceases_to_equal_later_contract"

            if not divergence_seen and divergence_here:
                divergence_seen = True
                first_divergence_stage = stage_label
                first_divergence_reason = divergence_reason
            elif not divergence_seen:
                earlier_exonerated.append(stage_label)

            comparison_table.append(
                {
                    "stage": stage_label,
                    "code_location": (
                        plain_row.get("code_location")
                        if plain_row is not None
                        else (None if patch_b_row is None else patch_b_row.get("code_location"))
                    ),
                    "plain": plain_value,
                    "patch_b": patch_b_value,
                    "delta_patch_b_minus_plain": delta,
                    "traced_equals_edr_contract": traced_equals_edr_contract,
                    "plain_breaks_initial_trace": plain_breaks_trace,
                    "patch_b_breaks_initial_trace": patch_b_breaks_trace,
                    "first_divergence_here": divergence_here and first_divergence_stage == stage_label,
                }
            )

    final_export_delta = final_edr_patch_b - final_edr_plain
    printstep_plain = find_stage_value(plain_rows, "PRINTSTEP_DO_ENX_INPUT#1")
    printstep_patch_b = find_stage_value(patch_b_rows, "PRINTSTEP_DO_ENX_INPUT#1")
    final_export_matches_runtime = (
        printstep_plain is not None
        and printstep_patch_b is not None
        and abs(printstep_plain - final_edr_plain) <= LEDGER_TRACE_TOL
        and abs(printstep_patch_b - final_edr_patch_b) <= LEDGER_TRACE_TOL
    )
    final_export_divergence = not final_export_matches_runtime
    if not divergence_seen and final_export_divergence:
        first_divergence_stage = "FINAL_EDR_EXPORT"
        first_divergence_reason = "runtime_export_input_differs_from_readback"
        divergence_seen = True
    elif not divergence_seen:
        earlier_exonerated.append("FINAL_EDR_EXPORT")
    comparison_table.append(
        {
            "stage": "FINAL_EDR_EXPORT",
            "code_location": "gmx energy xvg output",
            "plain": final_edr_plain,
            "patch_b": final_edr_patch_b,
            "delta_patch_b_minus_plain": final_export_delta,
            "traced_equals_edr_contract": final_export_matches_runtime,
            "plain_breaks_initial_trace": (
                plain_initial_trace is not None and abs(final_edr_plain - plain_initial_trace) > LEDGER_TRACE_TOL
            ),
            "patch_b_breaks_initial_trace": (
                patch_b_initial_trace is not None
                and abs(final_edr_patch_b - patch_b_initial_trace) > LEDGER_TRACE_TOL
            ),
            "first_divergence_here": first_divergence_stage == "FINAL_EDR_EXPORT",
        }
    )

    legacy_contract_matches_runtime = (
        printstep_plain is not None
        and printstep_patch_b is not None
        and abs(printstep_plain - legacy_edr_plain) <= LEDGER_TRACE_TOL
        and abs(printstep_patch_b - legacy_edr_patch_b) <= LEDGER_TRACE_TOL
    )
    if not divergence_seen and not legacy_contract_matches_runtime:
        first_divergence_stage = "ANALYSIS_XVG_COLUMN_MAPPING"
        first_divergence_reason = "legacy_requested_order_assumption_mislabels_xvg_columns"
        divergence_seen = True
    elif not divergence_seen:
        earlier_exonerated.append("ANALYSIS_XVG_COLUMN_MAPPING")
    comparison_table.append(
        {
            "stage": "ANALYSIS_XVG_COLUMN_MAPPING",
            "code_location": "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series",
            "plain": legacy_edr_plain,
            "patch_b": legacy_edr_patch_b,
            "delta_patch_b_minus_plain": legacy_edr_patch_b - legacy_edr_plain,
            "traced_equals_edr_contract": legacy_contract_matches_runtime,
            "plain_breaks_initial_trace": (
                plain_initial_trace is not None and abs(legacy_edr_plain - plain_initial_trace) > LEDGER_TRACE_TOL
            ),
            "patch_b_breaks_initial_trace": (
                patch_b_initial_trace is not None
                and abs(legacy_edr_patch_b - patch_b_initial_trace) > LEDGER_TRACE_TOL
            ),
            "first_divergence_here": first_divergence_stage == "ANALYSIS_XVG_COLUMN_MAPPING",
        }
    )

    if first_divergence_stage is None:
        classification = "NOT-YET-RESOLVED"
    elif first_divergence_stage.startswith("FORCECPP_FINAL_LEDGER_WRITE") or first_divergence_stage.startswith(
        "SIM_UTIL_PME_RECEIVE_ADD"
    ):
        classification = "DIRECT_MUTATION_OF_LEDGER"
    elif first_divergence_stage.startswith("ENERGYOUTPUT_ADDDATA_INPUT") or first_divergence_stage.startswith(
        "ENERGYOUTPUT_AFTER_ADDVALUES"
    ):
        classification = "ALIAS_OR_COPY_DIVERGENCE"
    elif first_divergence_stage.startswith("PRINTSTEP_DO_ENX_INPUT"):
        classification = "POSTPROCESS_AGGREGATION_MISMATCH"
    elif first_divergence_stage in {"FINAL_EDR_EXPORT", "ANALYSIS_XVG_COLUMN_MAPPING"}:
        classification = "EXPORT_CONTRACT_MISMATCH"
    else:
        classification = "NOT-YET-RESOLVED"

    supports_post_final_origin = (
        first_divergence_stage is not None
        and classification != "NOT-YET-RESOLVED"
        and executed_branch_plain is not None
        and executed_branch_patch_b is not None
    )

    localization = {
        "post_final_ledger_write_export_inventory": stage_inventory,
        "runtime_post_final_ledger_trace_dossier": {
            "plain_rows": plain_rows,
            "patch_b_rows": patch_b_rows,
            "plain_energy_terms_step0": plain_terms,
            "patch_b_energy_terms_step0": patch_b_terms,
            "plain_legacy_energy_terms_step0": plain_legacy_terms,
            "patch_b_legacy_energy_terms_step0": patch_b_legacy_terms,
        },
        "first_divergence_proof": {
            "first_stage": first_divergence_stage,
            "reason": first_divergence_reason,
            "earlier_exonerated": earlier_exonerated,
            "printstep_matches_final_edr": final_export_matches_runtime,
            "legacy_mapping_matches_final_edr": legacy_contract_matches_runtime,
        },
        "classification_verdict": classification,
        "comparison_table": comparison_table,
        "supports_post_final_divergence": supports_post_final_origin,
        "provenance": {
            "executed_branch_plain": executed_branch_plain,
            "executed_branch_patch_b": executed_branch_patch_b,
            "instrumented_code_locations": [
                "src/gromacs/mdlib/force.cpp:455",
                "src/gromacs/mdlib/sim_util.cpp:327",
                "src/gromacs/mdlib/energyoutput.cpp:933",
                "src/gromacs/mdlib/energyoutput.cpp:944",
                "src/gromacs/mdlib/energyoutput.cpp:1306",
                "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series",
            ],
            "final_edr_plain": final_edr_plain,
            "final_edr_patch_b": final_edr_patch_b,
            "final_edr_delta_patch_b_minus_plain": final_export_delta,
            "legacy_edr_plain": legacy_edr_plain,
            "legacy_edr_patch_b": legacy_edr_patch_b,
            "plain_stdout_term_order": plain_terms_detail["stdout_term_order"],
            "patch_b_stdout_term_order": patch_b_terms_detail["stdout_term_order"],
            "plain_resolved_output_order": plain_terms_detail["resolved_output_order"],
            "patch_b_resolved_output_order": patch_b_terms_detail["resolved_output_order"],
        },
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "localization": localization,
    }


def dense_patch_b_lj_sr_first_sink_trace(
    gmx_bin: str,
    dense_fixture_summary: dict[str, Any],
    commands_log: list[str],
) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    patch_b_dir = Path(coarse["patch_b_work_dir"])

    plain_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(plain_dir))
    patch_b_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(patch_b_dir))
    plain_lj_source_truth_rows = load_lj_source_truth_rows(plain_dir)
    plain_coulomb_source_truth_rows = load_coulomb_source_truth_rows(plain_dir)
    plain_terms_detail = extract_named_energy_series_detail(
        gmx_bin, plain_dir, "plain", DIAGNOSTIC_ENERGY_TERMS, "m2p_plain_diag_terms", commands_log, "m2p_plain"
    )
    patch_b_terms_detail = extract_named_energy_series_detail(
        gmx_bin,
        patch_b_dir,
        "exact",
        DIAGNOSTIC_ENERGY_TERMS,
        "m2p_patch_b_diag_terms",
        commands_log,
        "m2p_patch_b",
    )
    plain_terms = plain_terms_detail["rows"][0]
    patch_b_terms = patch_b_terms_detail["rows"][0]
    plain_native_coulomb_sr = parse_float_field(
        plain_coulomb_source_truth_rows.get("rawCoulReadTrace.finalTotal"), "after"
    )
    plain_patch_contract_replay_coulomb_sr = parse_float_field(
        plain_coulomb_source_truth_rows.get("plainPatchContractReplay.finalTotal"), "after"
    )
    plain_native_lj_sr = parse_float_field(plain_lj_source_truth_rows.get("rawLjReadTrace.finalTotal"), "after")
    plain_patch_contract_replay_lj_sr = parse_float_field(
        plain_lj_source_truth_rows.get("plainPatchLjContractReplay.finalTotal"), "after"
    )

    def contract_matched_plain_lj_reference(stage: str, native_value: float | None) -> float | None:
        if plain_patch_contract_replay_lj_sr is None:
            return native_value
        if stage in ("SR_ACCUMULATION", "FINAL_INTERNAL_LEDGER"):
            return plain_patch_contract_replay_lj_sr
        if stage == "CORRECTED_EXPORT":
            return float(f"{plain_patch_contract_replay_lj_sr:.6f}")
        return native_value

    def contract_matched_plain_coulomb_reference(stage: str, native_value: float | None) -> float | None:
        if plain_patch_contract_replay_coulomb_sr is None:
            return native_value
        if stage in ("SR_ACCUMULATION", "FINAL_INTERNAL_LEDGER"):
            return plain_patch_contract_replay_coulomb_sr
        if stage == "CORRECTED_EXPORT":
            return float(f"{plain_patch_contract_replay_coulomb_sr:.6f}")
        return native_value

    stage_order = ["RAW_SR_FORMATION", "SR_ACCUMULATION", "FINAL_INTERNAL_LEDGER", "CORRECTED_EXPORT"]
    stage_data_plain = {
        stage: aggregate_lj_sr_stage(plain_rows, stage) for stage in ("RAW_SR_FORMATION", "SR_ACCUMULATION", "FINAL_INTERNAL_LEDGER")
    }
    stage_data_patch_b = {
        stage: aggregate_lj_sr_stage(patch_b_rows, stage) for stage in ("RAW_SR_FORMATION", "SR_ACCUMULATION", "FINAL_INTERNAL_LEDGER")
    }
    stage_data_plain["CORRECTED_EXPORT"] = {
        "stage": "CORRECTED_EXPORT",
        "rows": [],
        "lj_sr": plain_terms["LJ-(SR)"],
        "coulomb_sr": plain_terms["Coulomb-(SR)"],
        "code_locations": ["gmx energy corrected extractor"],
        "execution_paths": ["corrected_export"],
    }
    stage_data_patch_b["CORRECTED_EXPORT"] = {
        "stage": "CORRECTED_EXPORT",
        "rows": [],
        "lj_sr": patch_b_terms["LJ-(SR)"],
        "coulomb_sr": patch_b_terms["Coulomb-(SR)"],
        "code_locations": ["gmx energy corrected extractor"],
        "execution_paths": ["corrected_export"],
    }

    comparison_table: list[dict[str, Any]] = []
    first_nonzero_lj_stage = None
    earlier_exonerated: list[str] = []
    first_nonzero_coul_stage = None

    for stage in stage_order:
        plain_stage = stage_data_plain[stage]
        patch_stage = stage_data_patch_b[stage]
        plain_lj = plain_stage["lj_sr"]
        patch_lj = patch_stage["lj_sr"]
        plain_coul = plain_stage["coulomb_sr"]
        patch_coul = patch_stage["coulomb_sr"]
        comparator_plain_lj = contract_matched_plain_lj_reference(stage, plain_lj)
        comparator_plain_coul = contract_matched_plain_coulomb_reference(stage, plain_coul)
        delta_lj = None if comparator_plain_lj is None or patch_lj is None else patch_lj - comparator_plain_lj
        delta_coul = (
            None if comparator_plain_coul is None or patch_coul is None else patch_coul - comparator_plain_coul
        )
        first_nonzero_here = False
        if first_nonzero_lj_stage is None and delta_lj is not None and abs(delta_lj) > NUMERIC_FIELD_TOL:
            first_nonzero_lj_stage = stage
            first_nonzero_here = True
        elif first_nonzero_lj_stage is None:
            earlier_exonerated.append(stage)
        plain_locations = plain_stage["code_locations"]
        patch_locations = patch_stage["code_locations"]
        if plain_locations == patch_locations:
            code_location = "; ".join(plain_locations)
        else:
            code_location = f"plain={'; '.join(plain_locations)} | patch_b={'; '.join(patch_locations)}"

        comparison_table.append(
            {
                "stage": stage,
                "code_location": code_location,
                "plain_LJ_SR": comparator_plain_lj,
                "plain_native_LJ_SR": plain_lj,
                "plain_patch_contract_replay_LJ_SR": plain_patch_contract_replay_lj_sr,
                "patch_b_LJ_SR": patch_lj,
                "delta_LJ_SR": delta_lj,
                "plain_Coulomb_SR": comparator_plain_coul,
                "plain_native_Coulomb_SR": plain_coul,
                "plain_patch_contract_replay_Coulomb_SR": plain_patch_contract_replay_coulomb_sr,
                "patch_b_Coulomb_SR": patch_coul,
                "delta_Coulomb_SR": delta_coul,
                "first_nonzero_LJ_here": first_nonzero_here,
            }
        )

    for stage in ("SR_ACCUMULATION", "FINAL_INTERNAL_LEDGER", "CORRECTED_EXPORT"):
        row = next(candidate for candidate in comparison_table if candidate["stage"] == stage)
        delta_coul = row["delta_Coulomb_SR"]
        if first_nonzero_coul_stage is None and delta_coul is not None and abs(delta_coul) > NUMERIC_FIELD_TOL:
            first_nonzero_coul_stage = stage

    final_internal_plain = stage_data_plain["FINAL_INTERNAL_LEDGER"]
    final_internal_patch_b = stage_data_patch_b["FINAL_INTERNAL_LEDGER"]
    corrected_export_matches_internal = (
        final_internal_plain["lj_sr"] is not None
        and final_internal_patch_b["lj_sr"] is not None
        and abs(final_internal_plain["lj_sr"] - plain_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
        and abs(final_internal_patch_b["lj_sr"] - patch_b_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
        and final_internal_plain["coulomb_sr"] is not None
        and final_internal_patch_b["coulomb_sr"] is not None
        and abs(final_internal_plain["coulomb_sr"] - plain_terms["Coulomb-(SR)"]) <= LEDGER_TRACE_TOL
        and abs(final_internal_patch_b["coulomb_sr"] - patch_b_terms["Coulomb-(SR)"]) <= LEDGER_TRACE_TOL
    )

    if first_nonzero_coul_stage is None:
        coulomb_coupling = "NOT-YET-RESOLVED"
    elif first_nonzero_lj_stage is None:
        coulomb_coupling = "NOT-YET-RESOLVED"
    elif first_nonzero_coul_stage == first_nonzero_lj_stage:
        coulomb_coupling = "SAME_STAGE_COUPLED"
    elif stage_order.index(first_nonzero_coul_stage) > stage_order.index(first_nonzero_lj_stage):
        coulomb_coupling = "LATER_STAGE_REFLECTION"
    else:
        coulomb_coupling = "INDEPENDENT_SECONDARY"

    if first_nonzero_lj_stage is None:
        classification = "NOT-YET-RESOLVED"
    elif first_nonzero_lj_stage == "RAW_SR_FORMATION":
        classification = "LJ_SR_EVALUATION_ORIGIN"
    elif first_nonzero_lj_stage == "SR_ACCUMULATION":
        classification = "SR_ACCUMULATION_ORIGIN"
    elif first_nonzero_lj_stage == "FINAL_INTERNAL_LEDGER":
        classification = "LEDGER_AGGREGATION_ORIGIN"
    elif first_nonzero_lj_stage == "CORRECTED_EXPORT":
        classification = "EXPORT_CONTRACT_MISMATCH"
    else:
        classification = "NOT-YET-RESOLVED"

    patch_b_raw_rows = stage_data_patch_b["RAW_SR_FORMATION"]["rows"]
    inventory = [
        {
            "candidate": "plain_pairwise_lj_sr_evaluation",
            "code_location": "src/gromacs/nbnxm/kerneldispatch.cpp:430",
            "candidate_type": "pairwise_lj_sr_evaluation_path",
            "executed_plain": any(row.get("execution_path") == "plain_nbnxm_cpu" for row in plain_rows),
            "executed_patch_b": False,
        },
        {
            "candidate": "exact_pairwise_lj_sr_evaluation",
            "code_location": "src/gromacs/mdlib/sim_util.cpp:1754",
            "candidate_type": "pairwise_lj_sr_evaluation_path",
            "executed_plain": False,
            "executed_patch_b": any(row.get("execution_path") == "exact_respa_pairs" for row in patch_b_raw_rows),
        },
        {
            "candidate": "exact_excluded_pair_sr_handling",
            "code_location": "src/gromacs/mdlib/sim_util.cpp:1769",
            "candidate_type": "excluded_pair_sr_handling",
            "executed_plain": False,
            "executed_patch_b": any(
                row.get("execution_path") == "exact_respa_excluded_pairs" for row in patch_b_raw_rows
            ),
        },
        {
            "candidate": "sr_accumulation_grpp",
            "code_location": "src/gromacs/mdlib/sim_util.cpp:4284",
            "candidate_type": "bookkeeping_ledger_accumulation",
            "executed_plain": stage_data_plain["SR_ACCUMULATION"]["lj_sr"] is not None,
            "executed_patch_b": stage_data_patch_b["SR_ACCUMULATION"]["lj_sr"] is not None,
        },
        {
            "candidate": "final_internal_ledger",
            "code_location": "src/gromacs/mdlib/sim_util.cpp:4298",
            "candidate_type": "ledger_aggregation",
            "executed_plain": stage_data_plain["FINAL_INTERNAL_LEDGER"]["lj_sr"] is not None,
            "executed_patch_b": stage_data_patch_b["FINAL_INTERNAL_LEDGER"]["lj_sr"] is not None,
        },
        {
            "candidate": "corrected_export",
            "code_location": "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            "candidate_type": "output_export_mapping_path",
            "executed_plain": True,
            "executed_patch_b": True,
        },
    ]

    supports_origin = False

    localization = {
        "lj_sr_sink_origin_inventory": inventory,
        "runtime_trace_dossier": {
            "plain_rows": plain_rows,
            "patch_b_rows": patch_b_rows,
            "plain_energy_terms_step0": plain_terms,
            "patch_b_energy_terms_step0": patch_b_terms,
            "plain_lj_source_truth_rows": plain_lj_source_truth_rows,
            "plain_native_lj_sr": plain_native_lj_sr,
            "plain_patch_contract_replay_lj_sr": plain_patch_contract_replay_lj_sr,
            "plain_coulomb_source_truth_rows": plain_coulomb_source_truth_rows,
            "plain_native_coulomb_sr": plain_native_coulomb_sr,
            "plain_patch_contract_replay_coulomb_sr": plain_patch_contract_replay_coulomb_sr,
        },
        "lj_comparator_contract": {
            "plain_native_lj_sr": plain_native_lj_sr,
            "plain_patch_contract_replay_lj_sr": plain_patch_contract_replay_lj_sr,
            "comparator_rule": (
                "Use plain patch-contract replay LJ total as the plain reference for "
                "SR_ACCUMULATION, FINAL_INTERNAL_LEDGER, and CORRECTED_EXPORT; keep plain native total reported separately."
            ),
        },
        "first_nonzero_lj_sr_delta_proof": {
            "first_stage": first_nonzero_lj_stage,
            "earlier_exonerated": earlier_exonerated,
            "corrected_export_matches_internal": corrected_export_matches_internal,
        },
        "coulomb_sr_coupling_result": coulomb_coupling,
        "coulomb_comparator_contract": {
            "plain_native_coulomb_sr": plain_native_coulomb_sr,
            "plain_patch_contract_replay_coulomb_sr": plain_patch_contract_replay_coulomb_sr,
            "comparator_rule": (
                "Use plain patch-contract replay Coulomb total as the plain reference for "
                "SR_ACCUMULATION, FINAL_INTERNAL_LEDGER, and CORRECTED_EXPORT; keep plain native total reported separately."
            ),
        },
        "origin_classification_verdict": classification,
        "comparison_table": comparison_table,
        "supports_lj_sr_origin": supports_origin,
        "provenance": {
            "instrumented_code_locations": [
                "src/gromacs/nbnxm/kerneldispatch.cpp:430",
                "src/gromacs/mdlib/sim_util.cpp:1754",
                "src/gromacs/mdlib/sim_util.cpp:1769",
                "src/gromacs/mdlib/sim_util.cpp:4284",
                "src/gromacs/mdlib/sim_util.cpp:4298",
                "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            ],
            "corrected_extractor_code_path": "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            "plain_resolved_output_order": plain_terms_detail["resolved_output_order"],
            "patch_b_resolved_output_order": patch_b_terms_detail["resolved_output_order"],
            "plain_stdout_term_order": plain_terms_detail["stdout_term_order"],
            "patch_b_stdout_term_order": patch_b_terms_detail["stdout_term_order"],
        },
        "why_not_fully_closed": (
            "This closes only the first LJ-(SR) residual origin for dense_oligomer coarse step 0 under Patch-shape B; it does not establish all-term bookkeeping closure."
        ),
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "step0": 0,
        "localization": localization,
    }


def dense_patch_b_potential_ledger_trace(
    gmx_bin: str,
    dense_fixture_summary: dict[str, Any],
    commands_log: list[str],
) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    patch_b_dir = Path(coarse["patch_b_work_dir"])

    plain_ledger_row = load_potential_ledger_trace_row(plain_dir)
    patch_ledger_row = load_potential_ledger_trace_row(patch_b_dir)
    plain_lj_source_truth_rows = load_lj_source_truth_rows(plain_dir)
    plain_coulomb_source_truth_rows = load_coulomb_source_truth_rows(plain_dir)
    plain_terms_detail = extract_named_energy_series_detail(
        gmx_bin, plain_dir, "plain", DIAGNOSTIC_ENERGY_TERMS, "m2pot_plain_diag_terms", commands_log, "m2pot_plain"
    )
    patch_terms_detail = extract_named_energy_series_detail(
        gmx_bin,
        patch_b_dir,
        "exact",
        DIAGNOSTIC_ENERGY_TERMS,
        "m2pot_patch_b_diag_terms",
        commands_log,
        "m2pot_patch_b",
    )
    plain_terms = plain_terms_detail["rows"][0]
    patch_terms = patch_terms_detail["rows"][0]

    plain_patch_contract_replay_coulomb_sr = parse_float_field(
        plain_coulomb_source_truth_rows.get("plainPatchContractReplay.finalTotal"), "after"
    )
    plain_patch_contract_replay_lj_sr = parse_float_field(
        plain_lj_source_truth_rows.get("plainPatchLjContractReplay.finalTotal"), "after"
    )

    def ledger_value(row: dict[str, str] | None, key: str) -> float | None:
        return parse_float_field(row, key)

    def adjusted_plain_ledger_value(key: str) -> float | None:
        native_value = ledger_value(plain_ledger_row, key)
        native_coulomb = ledger_value(plain_ledger_row, "coul_sr")
        native_lj = ledger_value(plain_ledger_row, "lj_sr")
        adjusted_value = native_value
        if adjusted_value is None:
            return None
        if key == "coul_sr" and plain_patch_contract_replay_coulomb_sr is not None:
            adjusted_value = plain_patch_contract_replay_coulomb_sr
        elif key == "lj_sr" and plain_patch_contract_replay_lj_sr is not None:
            adjusted_value = plain_patch_contract_replay_lj_sr
        else:
            if native_coulomb is not None and plain_patch_contract_replay_coulomb_sr is not None and key in (
                "component_sum",
                "potential",
            ):
                adjusted_value = adjusted_value - native_coulomb + plain_patch_contract_replay_coulomb_sr
            if native_lj is not None and plain_patch_contract_replay_lj_sr is not None and key in (
                "component_sum",
                "potential",
            ):
                adjusted_value = adjusted_value - native_lj + plain_patch_contract_replay_lj_sr
        return adjusted_value

    def adjusted_plain_export_potential() -> float | None:
        native_potential = plain_terms.get("Potential")
        native_coulomb = plain_terms.get("Coulomb-(SR)")
        native_lj = plain_terms.get("LJ-(SR)")
        if native_potential is None:
            return native_potential
        adjusted_value = native_potential
        if native_coulomb is not None and plain_patch_contract_replay_coulomb_sr is not None:
            replay_export_coulomb = float(f"{plain_patch_contract_replay_coulomb_sr:.6f}")
            adjusted_value = adjusted_value - native_coulomb + replay_export_coulomb
        if native_lj is not None and plain_patch_contract_replay_lj_sr is not None:
            replay_export_lj = float(f"{plain_patch_contract_replay_lj_sr:.6f}")
            adjusted_value = adjusted_value - native_lj + replay_export_lj
        return adjusted_value

    component_rows = [
        ("bond", "Class2-Bond"),
        ("angle", "Class2-Angle"),
        ("proper_dih", "Class2-Dih"),
        ("improper_dih", "Improper-Dih"),
        ("lj14", "LJ-14"),
        ("coul14", "Coulomb-14"),
        ("lj_sr", "LJ-(SR)"),
        ("coul_sr", "Coulomb-(SR)"),
        ("coul_recip", "Coul.-recip."),
        ("buckingham_sr", "Buckingham-(SR)"),
        ("other_terms", "Other-Terms"),
    ]
    component_table: list[dict[str, Any]] = []
    for key, label in component_rows:
        plain_native_value = ledger_value(plain_ledger_row, key)
        plain_contract_value = adjusted_plain_ledger_value(key)
        patch_value = ledger_value(patch_ledger_row, key)
        delta_value = (
            None
            if plain_contract_value is None or patch_value is None
            else patch_value - plain_contract_value
        )
        component_table.append(
            {
                "component": label,
                "plain_native": plain_native_value,
                "plain_coulomb_contract_baseline": plain_contract_value,
                "patch_b": patch_value,
                "delta_patch_minus_plain_baseline": delta_value,
            }
        )

    plain_component_sum_baseline = adjusted_plain_ledger_value("component_sum")
    patch_component_sum = ledger_value(patch_ledger_row, "component_sum")
    plain_final_internal_potential_baseline = adjusted_plain_ledger_value("potential")
    patch_final_internal_potential = ledger_value(patch_ledger_row, "potential")
    plain_corrected_export_potential_baseline = adjusted_plain_export_potential()
    patch_corrected_export_potential = patch_terms.get("Potential")

    component_sum_delta = (
        None
        if plain_component_sum_baseline is None or patch_component_sum is None
        else patch_component_sum - plain_component_sum_baseline
    )
    final_internal_delta = (
        None
        if plain_final_internal_potential_baseline is None or patch_final_internal_potential is None
        else patch_final_internal_potential - plain_final_internal_potential_baseline
    )
    corrected_export_delta = (
        None
        if plain_corrected_export_potential_baseline is None or patch_corrected_export_potential is None
        else patch_corrected_export_potential - plain_corrected_export_potential_baseline
    )

    plain_ledger_sum_matches_potential = (
        plain_component_sum_baseline is not None
        and plain_final_internal_potential_baseline is not None
        and abs(plain_component_sum_baseline - plain_final_internal_potential_baseline) <= LEDGER_TRACE_TOL
    )
    patch_ledger_sum_matches_potential = (
        patch_component_sum is not None
        and patch_final_internal_potential is not None
        and abs(patch_component_sum - patch_final_internal_potential) <= LEDGER_TRACE_TOL
    )

    if (
        (component_sum_delta is not None and abs(component_sum_delta) > LEDGER_TRACE_TOL)
        or (final_internal_delta is not None and abs(final_internal_delta) > LEDGER_TRACE_TOL)
    ):
        first_stage = "FINAL_INTERNAL_LEDGER"
    elif corrected_export_delta is not None and abs(corrected_export_delta) > LEDGER_TRACE_TOL:
        first_stage = "CORRECTED_EXPORT"
    else:
        first_stage = None

    if (
        first_stage == "FINAL_INTERNAL_LEDGER"
        and plain_ledger_sum_matches_potential
        and patch_ledger_sum_matches_potential
    ):
        classification = "POTENTIAL_LEDGER_AGGREGATION_MISMATCH"
    elif (
        first_stage == "CORRECTED_EXPORT"
        and (component_sum_delta is None or abs(component_sum_delta) <= LEDGER_TRACE_TOL)
        and (final_internal_delta is None or abs(final_internal_delta) <= LEDGER_TRACE_TOL)
    ):
        classification = "POTENTIAL_EXPORT_CONTRACT_MISMATCH"
    else:
        classification = "NOT_YET_RESOLVED"

    localization = {
        "potential_component_table": component_table,
        "potential_ledger_export_table": {
            "plain_component_sum_at_ledger": plain_component_sum_baseline,
            "patch_component_sum_at_ledger": patch_component_sum,
            "plain_final_internal_potential": plain_final_internal_potential_baseline,
            "patch_final_internal_potential": patch_final_internal_potential,
            "plain_corrected_export_potential": plain_corrected_export_potential_baseline,
            "patch_corrected_export_potential": patch_corrected_export_potential,
            "component_sum_delta_patch_minus_plain": component_sum_delta,
            "final_internal_delta_patch_minus_plain": final_internal_delta,
            "corrected_export_delta_patch_minus_plain": corrected_export_delta,
        },
        "coulomb_comparator_contract": {
            "plain_patch_contract_replay_coulomb_sr": plain_patch_contract_replay_coulomb_sr,
            "plain_native_coulomb_sr_at_ledger": ledger_value(plain_ledger_row, "coul_sr"),
            "plain_native_coulomb_sr_at_export": plain_terms.get("Coulomb-(SR)"),
            "rule": "Replace plain Coulomb-(SR) with the plain patch-contract replay value when comparing Potential under the locked-scope Coulomb baseline.",
        },
        "lj_comparator_contract": {
            "plain_patch_contract_replay_lj_sr": plain_patch_contract_replay_lj_sr,
            "plain_native_lj_sr_at_ledger": ledger_value(plain_ledger_row, "lj_sr"),
            "plain_native_lj_sr_at_export": plain_terms.get("LJ-(SR)"),
            "rule": "Replace plain LJ-(SR) with the plain patch-contract replay value when comparing Potential under the locked-scope LJ baseline.",
        },
        "first_stage_where_potential_diverges": first_stage,
        "classification": classification,
        "provenance": {
            "instrumented_code_locations": [
                "src/gromacs/mdlib/sim_util.cpp:5410",
                "src/gromacs/mdlib/enerdata_utils.cpp:179",
                "src/gromacs/mdlib/enerdata_utils.cpp:322",
                "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            ],
            "trace_row_plain": plain_ledger_row,
            "trace_row_patch_b": patch_ledger_row,
            "plain_resolved_output_order": plain_terms_detail["resolved_output_order"],
            "patch_b_resolved_output_order": patch_terms_detail["resolved_output_order"],
        },
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "step0": 0,
        "localization": localization,
    }


def dense_patch_b_lj_sr_true_first_raw_trace(
    gmx_bin: str,
    dense_fixture_summary: dict[str, Any],
    commands_log: list[str],
) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    patch_b_dir = Path(coarse["patch_b_work_dir"])

    plain_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(plain_dir))
    patch_b_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(patch_b_dir))
    plain_terms_detail = extract_named_energy_series_detail(
        gmx_bin, plain_dir, "plain", DIAGNOSTIC_ENERGY_TERMS, "m2q_plain_diag_terms", commands_log, "m2q_plain"
    )
    patch_b_terms_detail = extract_named_energy_series_detail(
        gmx_bin,
        patch_b_dir,
        "exact",
        DIAGNOSTIC_ENERGY_TERMS,
        "m2q_patch_b_diag_terms",
        commands_log,
        "m2q_patch_b",
    )
    plain_terms = plain_terms_detail["rows"][0]
    patch_b_terms = patch_b_terms_detail["rows"][0]

    def aggregate_stage_with_paths(
        rows: list[dict[str, str]], stage: str, execution_path_prefixes: tuple[str, ...]
    ) -> dict[str, Any]:
        filtered_rows = [
            row
            for row in rows
            if row.get("stage") == stage
            and (
                not execution_path_prefixes
                or any((row.get("execution_path") or "").startswith(prefix) for prefix in execution_path_prefixes)
            )
        ]
        return aggregate_lj_sr_stage(filtered_rows, stage)

    stage_order = [
        "EARLIEST_RAW_STAGE",
        "RAW_SR_FORMATION",
        "SR_ACCUMULATION",
        "FINAL_INTERNAL_LEDGER",
        "CORRECTED_EXPORT",
    ]
    stage_data_plain = {
        stage: aggregate_lj_sr_stage(plain_rows, stage)
        for stage in ("EARLIEST_RAW_STAGE", "RAW_SR_FORMATION", "SR_ACCUMULATION", "FINAL_INTERNAL_LEDGER")
    }
    stage_data_patch_b = {
        stage: aggregate_lj_sr_stage(patch_b_rows, stage)
        for stage in ("EARLIEST_RAW_STAGE", "RAW_SR_FORMATION", "SR_ACCUMULATION", "FINAL_INTERNAL_LEDGER")
    }
    stage_data_plain["CORRECTED_EXPORT"] = {
        "stage": "CORRECTED_EXPORT",
        "rows": [],
        "lj_sr": plain_terms["LJ-(SR)"],
        "coulomb_sr": plain_terms["Coulomb-(SR)"],
        "code_locations": ["gmx energy corrected extractor"],
        "execution_paths": ["corrected_export"],
    }
    stage_data_patch_b["CORRECTED_EXPORT"] = {
        "stage": "CORRECTED_EXPORT",
        "rows": [],
        "lj_sr": patch_b_terms["LJ-(SR)"],
        "coulomb_sr": patch_b_terms["Coulomb-(SR)"],
        "code_locations": ["gmx energy corrected extractor"],
        "execution_paths": ["corrected_export"],
    }

    comparison_table: list[dict[str, Any]] = []
    first_nonzero_lj_stage = None
    earlier_exonerated: list[str] = []
    first_nonzero_coul_stage = None

    for stage in stage_order:
        plain_stage = stage_data_plain[stage]
        patch_stage = stage_data_patch_b[stage]
        plain_lj = plain_stage["lj_sr"]
        patch_lj = patch_stage["lj_sr"]
        plain_coul = plain_stage["coulomb_sr"]
        patch_coul = patch_stage["coulomb_sr"]
        delta_lj = None if plain_lj is None or patch_lj is None else patch_lj - plain_lj
        delta_coul = None if plain_coul is None or patch_coul is None else patch_coul - plain_coul
        first_nonzero_here = False
        if first_nonzero_lj_stage is None and delta_lj is not None and abs(delta_lj) > NUMERIC_FIELD_TOL:
            first_nonzero_lj_stage = stage
            first_nonzero_here = True
        elif first_nonzero_lj_stage is None:
            earlier_exonerated.append(stage)
        plain_locations = plain_stage["code_locations"]
        patch_locations = patch_stage["code_locations"]
        if plain_locations == patch_locations:
            code_location = "; ".join(plain_locations)
        else:
            code_location = f"plain={'; '.join(plain_locations)} | patch_b={'; '.join(patch_locations)}"

        comparison_table.append(
            {
                "stage": stage,
                "code_location": code_location,
                "plain_LJ_SR": plain_lj,
                "patch_b_LJ_SR": patch_lj,
                "delta_LJ_SR": delta_lj,
                "plain_Coulomb_SR": plain_coul,
                "patch_b_Coulomb_SR": patch_coul,
                "delta_Coulomb_SR": delta_coul,
                "first_nonzero_LJ_here": first_nonzero_here,
            }
        )

    for stage in ("SR_ACCUMULATION", "FINAL_INTERNAL_LEDGER", "CORRECTED_EXPORT"):
        row = next(candidate for candidate in comparison_table if candidate["stage"] == stage)
        delta_coul = row["delta_Coulomb_SR"]
        if first_nonzero_coul_stage is None and delta_coul is not None and abs(delta_coul) > NUMERIC_FIELD_TOL:
            first_nonzero_coul_stage = stage

    final_internal_plain = stage_data_plain["FINAL_INTERNAL_LEDGER"]
    final_internal_patch_b = stage_data_patch_b["FINAL_INTERNAL_LEDGER"]
    corrected_export_matches_internal = (
        final_internal_plain["lj_sr"] is not None
        and final_internal_patch_b["lj_sr"] is not None
        and abs(final_internal_plain["lj_sr"] - plain_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
        and abs(final_internal_patch_b["lj_sr"] - patch_b_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
        and final_internal_plain["coulomb_sr"] is not None
        and final_internal_patch_b["coulomb_sr"] is not None
        and abs(final_internal_plain["coulomb_sr"] - plain_terms["Coulomb-(SR)"]) <= LEDGER_TRACE_TOL
        and abs(final_internal_patch_b["coulomb_sr"] - patch_b_terms["Coulomb-(SR)"]) <= LEDGER_TRACE_TOL
    )

    if first_nonzero_lj_stage == "EARLIEST_RAW_STAGE":
        exact_first_verdict = "EARLIEST_RAW_STAGE_CONFIRMED"
    elif first_nonzero_lj_stage == "RAW_SR_FORMATION":
        exact_first_verdict = "RAW_SR_FORMATION_STILL_FIRST"
    else:
        exact_first_verdict = "NOT-YET-RESOLVED"

    if first_nonzero_coul_stage is None or first_nonzero_lj_stage is None:
        coulomb_coupling = "NOT-YET-RESOLVED"
    elif first_nonzero_coul_stage == first_nonzero_lj_stage:
        coulomb_coupling = "SAME_STAGE_COUPLED"
    elif stage_order.index(first_nonzero_coul_stage) > stage_order.index(first_nonzero_lj_stage):
        coulomb_coupling = "LATER_STAGE_REFLECTION"
    else:
        coulomb_coupling = "INDEPENDENT_SECONDARY"

    if first_nonzero_lj_stage in ("EARLIEST_RAW_STAGE", "RAW_SR_FORMATION"):
        classification = "LJ_SR_EVALUATION_ORIGIN"
    elif first_nonzero_lj_stage == "SR_ACCUMULATION":
        classification = "SR_ACCUMULATION_ORIGIN"
    elif first_nonzero_lj_stage == "FINAL_INTERNAL_LEDGER":
        classification = "LEDGER_AGGREGATION_ORIGIN"
    elif first_nonzero_lj_stage == "CORRECTED_EXPORT":
        classification = "EXPORT_CONTRACT_MISMATCH"
    else:
        classification = "NOT-YET-RESOLVED"

    plain_earliest_rows = stage_data_plain["EARLIEST_RAW_STAGE"]["rows"]
    patch_earliest_rows = stage_data_patch_b["EARLIEST_RAW_STAGE"]["rows"]
    plain_raw_rows = stage_data_plain["RAW_SR_FORMATION"]["rows"]
    patch_raw_rows = stage_data_patch_b["RAW_SR_FORMATION"]["rows"]
    plain_earliest_row = plain_earliest_rows[0] if plain_earliest_rows else {}
    patch_earliest_row = patch_earliest_rows[0] if patch_earliest_rows else {}
    inventory = [
        {
            "candidate": "plain_earliest_lj_raw_site",
            "code_location": "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:pre_Vvdw_accumulation",
            "candidate_type": "contract_matched_raw_lj_formation",
            "executed_plain": bool(plain_earliest_rows),
            "executed_patch_b": False,
            "execution_path": plain_earliest_row.get("execution_path"),
            "kernel_type": plain_earliest_row.get("kernel_type"),
        },
        {
            "candidate": "patch_b_earliest_lj_raw_site",
            "code_location": "src/gromacs/mdlib/sim_util.cpp:per_pair_rawLjEnergy_before_pairStats_aggregate",
            "candidate_type": "contract_matched_raw_lj_formation",
            "executed_plain": False,
            "executed_patch_b": bool(patch_earliest_rows),
            "execution_path": patch_earliest_row.get("execution_path"),
        },
        {
            "candidate": "plain_raw_sr_formation_aggregate",
            "code_location": "src/gromacs/nbnxm/kerneldispatch.cpp:thread_output_pre_reduce",
            "candidate_type": "post_kernel_thread_output_aggregate",
            "executed_plain": any(row.get("execution_path") == "plain_nbnxm_cpu" for row in plain_raw_rows),
            "executed_patch_b": False,
        },
        {
            "candidate": "patch_b_raw_sr_formation_aggregate",
            "code_location": "src/gromacs/mdlib/sim_util.cpp:pair_loop_raw_energy_delta",
            "candidate_type": "post_pairloop_aggregate",
            "executed_plain": False,
            "executed_patch_b": any(row.get("execution_path") == "exact_respa_pairs" for row in patch_raw_rows),
        },
    ]

    supports_origin = (
        exact_first_verdict != "NOT-YET-RESOLVED"
        and corrected_export_matches_internal
        and stage_data_plain["EARLIEST_RAW_STAGE"]["lj_sr"] is not None
        and stage_data_patch_b["EARLIEST_RAW_STAGE"]["lj_sr"] is not None
    )
    overclaim_reason = (
        "Harness now ties origin support to the contract-matched EARLIEST_RAW_STAGE/RAW_SR_FORMATION proof instead of the old aggregate-only RAW_SR_FORMATION claim."
        if supports_origin
        else "Harness origin support stays false because the exact contract-matched first raw LJ stage is still not proven."
    )

    localization = {
        "executed_earliest_site_inventory": inventory,
        "runtime_trace_dossier": {
            "plain_rows": plain_rows,
            "patch_b_rows": patch_b_rows,
            "plain_energy_terms_step0": plain_terms,
            "patch_b_energy_terms_step0": patch_b_terms,
            "plain_execution_path": plain_earliest_row.get("execution_path"),
            "plain_kernel_type": plain_earliest_row.get("kernel_type"),
            "patch_b_execution_path": patch_earliest_row.get("execution_path"),
        },
        "first_nonzero_lj_sr_delta_proof": {
            "first_stage": first_nonzero_lj_stage,
            "earlier_exonerated": earlier_exonerated,
            "corrected_export_matches_internal": corrected_export_matches_internal,
        },
        "exact_first_nonzero_lj_verdict": exact_first_verdict,
        "coulomb_sr_coupling_result": coulomb_coupling,
        "origin_classification_verdict": classification,
        "comparison_table": comparison_table,
        "harness_claim_audit": {
            "supports_lj_sr_origin": supports_origin,
            "old_overclaim_removed": True,
            "reason": overclaim_reason,
        },
        "supports_lj_sr_origin": supports_origin,
        "provenance": {
            "instrumented_code_locations": [
                "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:pre_Vvdw_accumulation",
                "src/gromacs/nbnxm/kerneldispatch.cpp:thread_output_pre_reduce",
                "src/gromacs/mdlib/sim_util.cpp:per_pair_rawLjEnergy_before_pairStats_aggregate",
                "src/gromacs/mdlib/sim_util.cpp:pair_loop_raw_energy_delta",
                "src/gromacs/mdlib/sim_util.cpp:4284",
                "src/gromacs/mdlib/sim_util.cpp:4298",
                "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            ],
            "corrected_extractor_code_path": "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            "plain_resolved_output_order": plain_terms_detail["resolved_output_order"],
            "patch_b_resolved_output_order": patch_b_terms_detail["resolved_output_order"],
            "plain_stdout_term_order": plain_terms_detail["stdout_term_order"],
            "patch_b_stdout_term_order": patch_b_terms_detail["stdout_term_order"],
        },
        "why_not_fully_closed": (
            "This milestone closes only the exact first contract-matched LJ-(SR) raw stage for dense_oligomer coarse step 0 under Patch-shape B."
        ),
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "step0": 0,
        "localization": localization,
    }


def dense_patch_b_lj_sr_first_amplification_trace(
    gmx_bin: str,
    dense_fixture_summary: dict[str, Any],
    commands_log: list[str],
) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    patch_b_dir = Path(coarse["patch_b_work_dir"])

    plain_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(plain_dir))
    patch_b_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(patch_b_dir))
    plain_terms_detail = extract_named_energy_series_detail(
        gmx_bin, plain_dir, "plain", DIAGNOSTIC_ENERGY_TERMS, "m2r_plain_diag_terms", commands_log, "m2r_plain"
    )
    patch_b_terms_detail = extract_named_energy_series_detail(
        gmx_bin,
        patch_b_dir,
        "exact",
        DIAGNOSTIC_ENERGY_TERMS,
        "m2r_patch_b_diag_terms",
        commands_log,
        "m2r_patch_b",
    )
    plain_terms = plain_terms_detail["rows"][0]
    patch_b_terms = patch_b_terms_detail["rows"][0]

    def aggregate_stage_with_paths(
        rows: list[dict[str, str]], stage: str, execution_path_prefixes: tuple[str, ...]
    ) -> dict[str, Any]:
        filtered_rows = [
            row
            for row in rows
            if row.get("stage") == stage
            and (
                not execution_path_prefixes
                or any((row.get("execution_path") or "").startswith(prefix) for prefix in execution_path_prefixes)
            )
        ]
        return aggregate_lj_sr_stage(filtered_rows, stage)

    stage_order = [
        "EARLIEST_RAW_STAGE",
        "INTERMEDIATE_LOCAL_STAGE",
        "RAW_SR_FORMATION",
        "SR_ACCUMULATION",
        "FINAL_INTERNAL_LEDGER",
        "CORRECTED_EXPORT",
    ]
    traced_internal_stages = (
        "EARLIEST_RAW_STAGE",
        "INTERMEDIATE_LOCAL_STAGE",
        "RAW_SR_FORMATION",
        "SR_ACCUMULATION",
        "FINAL_INTERNAL_LEDGER",
    )
    stage_data_plain = {stage: aggregate_lj_sr_stage(plain_rows, stage) for stage in traced_internal_stages}
    stage_data_patch_b = {stage: aggregate_lj_sr_stage(patch_b_rows, stage) for stage in traced_internal_stages}
    stage_data_plain["CORRECTED_EXPORT"] = {
        "stage": "CORRECTED_EXPORT",
        "rows": [],
        "lj_sr": plain_terms["LJ-(SR)"],
        "coulomb_sr": plain_terms["Coulomb-(SR)"],
        "code_locations": ["gmx energy corrected extractor"],
        "execution_paths": ["corrected_export"],
    }
    stage_data_patch_b["CORRECTED_EXPORT"] = {
        "stage": "CORRECTED_EXPORT",
        "rows": [],
        "lj_sr": patch_b_terms["LJ-(SR)"],
        "coulomb_sr": patch_b_terms["Coulomb-(SR)"],
        "code_locations": ["gmx energy corrected extractor"],
        "execution_paths": ["corrected_export"],
    }

    # Raw/intermediate Coulomb rows are not contract-matched across plain and exact Patch-B paths.
    for stage in ("EARLIEST_RAW_STAGE", "INTERMEDIATE_LOCAL_STAGE", "RAW_SR_FORMATION"):
        stage_data_plain[stage]["coulomb_sr"] = None
        stage_data_patch_b[stage]["coulomb_sr"] = None

    comparison_table: list[dict[str, Any]] = []
    first_amplification_stage = None
    first_amplification_increment = None
    non_amplifying_stages_before_first: list[str] = ["EARLIEST_RAW_STAGE"]
    first_comparable_coulomb_stage = None
    material_amplification_tol = NUMERIC_FIELD_TOL
    prev_abs_delta_lj = None

    for index, stage in enumerate(stage_order):
        plain_stage = stage_data_plain[stage]
        patch_stage = stage_data_patch_b[stage]
        plain_lj = plain_stage["lj_sr"]
        patch_lj = patch_stage["lj_sr"]
        plain_coul = plain_stage["coulomb_sr"]
        patch_coul = patch_stage["coulomb_sr"]
        delta_lj = None if plain_lj is None or patch_lj is None else patch_lj - plain_lj
        abs_delta_lj = None if delta_lj is None else abs(delta_lj)
        delta_increment = None
        first_amplification_here = False
        if prev_abs_delta_lj is not None and abs_delta_lj is not None:
            delta_increment = abs_delta_lj - prev_abs_delta_lj
            if first_amplification_stage is None and delta_increment > material_amplification_tol:
                first_amplification_stage = stage
                first_amplification_increment = delta_increment
                first_amplification_here = True
            elif first_amplification_stage is None:
                non_amplifying_stages_before_first.append(stage)
        delta_coul = None if plain_coul is None or patch_coul is None else patch_coul - plain_coul
        if (
            first_comparable_coulomb_stage is None
            and delta_coul is not None
            and abs(delta_coul) > NUMERIC_FIELD_TOL
        ):
            first_comparable_coulomb_stage = stage

        plain_locations = plain_stage["code_locations"]
        patch_locations = patch_stage["code_locations"]
        if plain_locations == patch_locations:
            code_location = "; ".join(plain_locations)
        else:
            code_location = f"plain={'; '.join(plain_locations)} | patch_b={'; '.join(patch_locations)}"

        comparison_table.append(
            {
                "stage": stage,
                "code_location": code_location,
                "plain_LJ_SR": plain_lj,
                "patch_b_LJ_SR": patch_lj,
                "delta_LJ_SR": delta_lj,
                "delta_increment_vs_prev": delta_increment,
                "plain_Coulomb_SR": plain_coul,
                "patch_b_Coulomb_SR": patch_coul,
                "delta_Coulomb_SR": delta_coul,
                "first_amplification_here": first_amplification_here,
            }
        )
        if abs_delta_lj is not None:
            prev_abs_delta_lj = abs_delta_lj

    final_internal_plain = stage_data_plain["FINAL_INTERNAL_LEDGER"]
    final_internal_patch_b = stage_data_patch_b["FINAL_INTERNAL_LEDGER"]
    corrected_export_matches_internal = (
        final_internal_plain["lj_sr"] is not None
        and final_internal_patch_b["lj_sr"] is not None
        and abs(final_internal_plain["lj_sr"] - plain_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
        and abs(final_internal_patch_b["lj_sr"] - patch_b_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
        and final_internal_plain["coulomb_sr"] is not None
        and final_internal_patch_b["coulomb_sr"] is not None
        and abs(final_internal_plain["coulomb_sr"] - plain_terms["Coulomb-(SR)"]) <= LEDGER_TRACE_TOL
        and abs(final_internal_patch_b["coulomb_sr"] - patch_b_terms["Coulomb-(SR)"]) <= LEDGER_TRACE_TOL
    )

    if first_amplification_stage == "INTERMEDIATE_LOCAL_STAGE":
        exact_first_amplification_verdict = "INTERMEDIATE_STAGE_CONFIRMED"
    elif first_amplification_stage == "RAW_SR_FORMATION":
        exact_first_amplification_verdict = "RAW_SR_FORMATION_IS_FIRST_AMPLIFICATION"
    elif first_amplification_stage == "SR_ACCUMULATION":
        exact_first_amplification_verdict = "SR_ACCUMULATION_IS_FIRST_AMPLIFICATION"
    else:
        exact_first_amplification_verdict = "NOT-YET-RESOLVED"

    if first_amplification_stage is None or first_comparable_coulomb_stage is None:
        coulomb_verdict = "NOT-YET-RESOLVED"
    elif first_comparable_coulomb_stage == first_amplification_stage:
        coulomb_verdict = "SAME_STAGE_COUPLED_AMPLIFICATION"
    elif stage_order.index(first_comparable_coulomb_stage) > stage_order.index(first_amplification_stage):
        coulomb_verdict = "LATER_STAGE_REFLECTION"
    else:
        coulomb_verdict = "INDEPENDENT_SECONDARY"

    if first_amplification_stage == "INTERMEDIATE_LOCAL_STAGE":
        amplification_classification = "KERNEL_LOCAL_ACCUMULATION_AMPLIFICATION"
    elif first_amplification_stage == "RAW_SR_FORMATION":
        amplification_classification = "REDUCTION_TRANSFER_AMPLIFICATION"
    elif first_amplification_stage == "SR_ACCUMULATION":
        amplification_classification = "SR_LEDGER_ACCUMULATION_AMPLIFICATION"
    else:
        amplification_classification = "NOT-YET-RESOLVED"

    plain_earliest_row = stage_data_plain["EARLIEST_RAW_STAGE"]["rows"][0] if stage_data_plain["EARLIEST_RAW_STAGE"]["rows"] else {}
    patch_earliest_row = stage_data_patch_b["EARLIEST_RAW_STAGE"]["rows"][0] if stage_data_patch_b["EARLIEST_RAW_STAGE"]["rows"] else {}
    plain_intermediate_row = (
        stage_data_plain["INTERMEDIATE_LOCAL_STAGE"]["rows"][0]
        if stage_data_plain["INTERMEDIATE_LOCAL_STAGE"]["rows"]
        else {}
    )
    patch_intermediate_row = (
        stage_data_patch_b["INTERMEDIATE_LOCAL_STAGE"]["rows"][0]
        if stage_data_patch_b["INTERMEDIATE_LOCAL_STAGE"]["rows"]
        else {}
    )
    inventory = {
        "plain_executed_stages": [
            {
                "stage": "EARLIEST_RAW_STAGE",
                "code_location": "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:pre_Vvdw_accumulation",
                "executed": bool(stage_data_plain["EARLIEST_RAW_STAGE"]["rows"]),
                "execution_path": plain_earliest_row.get("execution_path"),
                "kernel_type": plain_earliest_row.get("kernel_type"),
            },
            {
                "stage": "INTERMEDIATE_LOCAL_STAGE",
                "code_location": "src/gromacs/nbnxm/kernels_reference/kernel_ref_outer.h:post_kernel_local_energy_buffer_before_dispatch_transfer",
                "executed": bool(stage_data_plain["INTERMEDIATE_LOCAL_STAGE"]["rows"]),
                "execution_path": plain_intermediate_row.get("execution_path"),
                "kernel_type": plain_intermediate_row.get("kernel_type"),
            },
            {
                "stage": "RAW_SR_FORMATION",
                "code_location": "src/gromacs/nbnxm/kerneldispatch.cpp:thread_output_pre_reduce",
                "executed": bool(stage_data_plain["RAW_SR_FORMATION"]["rows"]),
                "execution_path": (
                    stage_data_plain["RAW_SR_FORMATION"]["rows"][0].get("execution_path")
                    if stage_data_plain["RAW_SR_FORMATION"]["rows"]
                    else None
                ),
            },
            {
                "stage": "SR_ACCUMULATION",
                "code_location": "src/gromacs/mdlib/sim_util.cpp:4284",
                "executed": bool(stage_data_plain["SR_ACCUMULATION"]["rows"]),
                "execution_path": (
                    stage_data_plain["SR_ACCUMULATION"]["rows"][0].get("execution_path")
                    if stage_data_plain["SR_ACCUMULATION"]["rows"]
                    else None
                ),
            },
        ],
        "patch_b_executed_stages": [
            {
                "stage": "EARLIEST_RAW_STAGE",
                "code_location": "src/gromacs/mdlib/sim_util.cpp:per_pair_rawLjEnergy_before_pairStats_aggregate",
                "executed": bool(stage_data_patch_b["EARLIEST_RAW_STAGE"]["rows"]),
                "execution_path": patch_earliest_row.get("execution_path"),
            },
            {
                "stage": "INTERMEDIATE_LOCAL_STAGE",
                "code_location": "src/gromacs/mdlib/sim_util.cpp:after_pairs_pairStats_before_excluded_transfer",
                "executed": bool(stage_data_patch_b["INTERMEDIATE_LOCAL_STAGE"]["rows"]),
                "execution_path": patch_intermediate_row.get("execution_path"),
            },
            {
                "stage": "RAW_SR_FORMATION",
                "code_location": "src/gromacs/mdlib/sim_util.cpp:1754; src/gromacs/mdlib/sim_util.cpp:1769",
                "executed": bool(stage_data_patch_b["RAW_SR_FORMATION"]["rows"]),
                "execution_path": sorted(
                    {
                        row.get("execution_path")
                        for row in stage_data_patch_b["RAW_SR_FORMATION"]["rows"]
                        if row.get("execution_path")
                    }
                ),
            },
            {
                "stage": "SR_ACCUMULATION",
                "code_location": "src/gromacs/mdlib/sim_util.cpp:4284",
                "executed": bool(stage_data_patch_b["SR_ACCUMULATION"]["rows"]),
                "execution_path": (
                    stage_data_patch_b["SR_ACCUMULATION"]["rows"][0].get("execution_path")
                    if stage_data_patch_b["SR_ACCUMULATION"]["rows"]
                    else None
                ),
            },
        ],
        "contract_match_notes": [
            "EARLIEST_RAW_STAGE compares the aggregate of pairwise final LJ energy contributions before later local buffering on both sides.",
            "INTERMEDIATE_LOCAL_STAGE compares the first local LJ aggregate that survives into later transfer/reduction on both sides.",
            "RAW_SR_FORMATION remains a later visible aggregate; it is not used for raw Coulomb companion claims.",
            "Coulomb-(SR) is considered contract-matched only from SR_ACCUMULATION onward because Patch-B raw/local Coulomb rows split across pairs and excludedPairs paths.",
        ],
    }

    supports_amplification_localization = (
        first_amplification_stage is not None
        and corrected_export_matches_internal
        and stage_data_plain["INTERMEDIATE_LOCAL_STAGE"]["lj_sr"] is not None
        and stage_data_patch_b["INTERMEDIATE_LOCAL_STAGE"]["lj_sr"] is not None
    )

    localization = {
        "executed_intermediate_stage_inventory": inventory,
        "runtime_trace_dossier": {
            "plain_rows": plain_rows,
            "patch_b_rows": patch_b_rows,
            "plain_energy_terms_step0": plain_terms,
            "patch_b_energy_terms_step0": patch_b_terms,
            "material_amplification_definition": f"abs(delta_LJ_SR_stage_n) - abs(delta_LJ_SR_stage_n-1) > {material_amplification_tol}",
            "material_amplification_tol": material_amplification_tol,
        },
        "first_amplification_proof": {
            "first_stage": first_amplification_stage,
            "first_increment": first_amplification_increment,
            "earlier_non_amplifying_stages": non_amplifying_stages_before_first,
            "corrected_export_matches_internal": corrected_export_matches_internal,
        },
        "exact_first_amplification_verdict": exact_first_amplification_verdict,
        "coulomb_companion_verdict": coulomb_verdict,
        "amplification_classification_verdict": amplification_classification,
        "comparison_table": comparison_table,
        "supports_lj_sr_amplification": supports_amplification_localization,
        "provenance": {
            "instrumented_code_locations": [
                "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:pre_Vvdw_accumulation",
                "src/gromacs/nbnxm/kernels_reference/kernel_ref_outer.h:post_kernel_local_energy_buffer_before_dispatch_transfer",
                "src/gromacs/nbnxm/kerneldispatch.cpp:thread_output_pre_reduce",
                "src/gromacs/mdlib/sim_util.cpp:per_pair_rawLjEnergy_before_pairStats_aggregate",
                "src/gromacs/mdlib/sim_util.cpp:after_pairs_pairStats_before_excluded_transfer",
                "src/gromacs/mdlib/sim_util.cpp:1754",
                "src/gromacs/mdlib/sim_util.cpp:1769",
                "src/gromacs/mdlib/sim_util.cpp:4284",
                "src/gromacs/mdlib/sim_util.cpp:4298",
                "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            ],
            "corrected_extractor_code_path": "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            "plain_execution_path": plain_earliest_row.get("execution_path"),
            "plain_kernel_type": plain_earliest_row.get("kernel_type"),
            "patch_b_execution_path": patch_earliest_row.get("execution_path"),
            "plain_resolved_output_order": plain_terms_detail["resolved_output_order"],
            "patch_b_resolved_output_order": patch_b_terms_detail["resolved_output_order"],
        },
        "why_not_fully_closed": (
            "This milestone closes only the first amplification stage for dense_oligomer coarse step 0 under Patch-shape B."
        ),
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "step0": 0,
        "localization": localization,
    }


def dense_patch_b_raw_sr_formation_internal_trace(
    gmx_bin: str,
    dense_fixture_summary: dict[str, Any],
    commands_log: list[str],
) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    patch_b_dir = Path(coarse["patch_b_work_dir"])

    plain_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(plain_dir))
    patch_b_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(patch_b_dir))
    plain_terms_detail = extract_named_energy_series_detail(
        gmx_bin, plain_dir, "plain", DIAGNOSTIC_ENERGY_TERMS, "m2s_plain_diag_terms", commands_log, "m2s_plain"
    )
    patch_b_terms_detail = extract_named_energy_series_detail(
        gmx_bin,
        patch_b_dir,
        "exact",
        DIAGNOSTIC_ENERGY_TERMS,
        "m2s_patch_b_diag_terms",
        commands_log,
        "m2s_patch_b",
    )
    plain_terms = plain_terms_detail["rows"][0]
    patch_b_terms = patch_b_terms_detail["rows"][0]

    stage_order = [
        "INTERMEDIATE_LOCAL_STAGE",
        "RAW_PRE_TRANSFER",
        "RAW_FIRST_WRITE",
        "RAW_POST_WRITE",
        "RAW_FIRST_READ_OR_REDUCE",
        "RAW_POST_READ_OR_REDUCE",
        "RAW_SR_FORMATION",
        "SR_ACCUMULATION",
        "FINAL_INTERNAL_LEDGER",
        "CORRECTED_EXPORT",
    ]
    traced_internal_stages = tuple(stage for stage in stage_order if stage != "CORRECTED_EXPORT")
    plain_path_filters = {
        "INTERMEDIATE_LOCAL_STAGE": ("plain_",),
        "RAW_PRE_TRANSFER": ("plain_",),
        "RAW_FIRST_WRITE": ("plain_",),
        "RAW_POST_WRITE": ("plain_",),
        "RAW_FIRST_READ_OR_REDUCE": ("plain_",),
        "RAW_POST_READ_OR_REDUCE": ("plain_",),
        "RAW_SR_FORMATION": ("plain_",),
        "SR_ACCUMULATION": (),
        "FINAL_INTERNAL_LEDGER": (),
    }
    patch_path_filters = {
        "INTERMEDIATE_LOCAL_STAGE": ("exact_",),
        "RAW_PRE_TRANSFER": ("exact_",),
        "RAW_FIRST_WRITE": ("exact_",),
        "RAW_POST_WRITE": ("exact_",),
        "RAW_FIRST_READ_OR_REDUCE": ("exact_",),
        "RAW_POST_READ_OR_REDUCE": ("exact_",),
        "RAW_SR_FORMATION": ("exact_",),
        "SR_ACCUMULATION": (),
        "FINAL_INTERNAL_LEDGER": (),
    }
    stage_data_plain = {
        stage: aggregate_lj_sr_stage_with_paths(plain_rows, stage, plain_path_filters[stage])
        for stage in traced_internal_stages
    }
    stage_data_patch_b = {
        stage: aggregate_lj_sr_stage_with_paths(patch_b_rows, stage, patch_path_filters[stage])
        for stage in traced_internal_stages
    }
    stage_data_plain["CORRECTED_EXPORT"] = {
        "stage": "CORRECTED_EXPORT",
        "rows": [],
        "lj_sr": plain_terms["LJ-(SR)"],
        "coulomb_sr": plain_terms["Coulomb-(SR)"],
        "code_locations": ["gmx energy corrected extractor"],
        "execution_paths": ["corrected_export"],
    }
    stage_data_patch_b["CORRECTED_EXPORT"] = {
        "stage": "CORRECTED_EXPORT",
        "rows": [],
        "lj_sr": patch_b_terms["LJ-(SR)"],
        "coulomb_sr": patch_b_terms["Coulomb-(SR)"],
        "code_locations": ["gmx energy corrected extractor"],
        "execution_paths": ["corrected_export"],
    }

    # Contract-matched Coulomb evidence starts only at the shared ledger stages.
    for stage in (
        "INTERMEDIATE_LOCAL_STAGE",
        "RAW_PRE_TRANSFER",
        "RAW_FIRST_WRITE",
        "RAW_POST_WRITE",
        "RAW_FIRST_READ_OR_REDUCE",
        "RAW_POST_READ_OR_REDUCE",
        "RAW_SR_FORMATION",
    ):
        stage_data_plain[stage]["coulomb_sr"] = None
        stage_data_patch_b[stage]["coulomb_sr"] = None

    comparison_table: list[dict[str, Any]] = []
    first_internal_stage = None
    first_internal_increment = None
    earlier_non_amplifying_stages: list[str] = ["INTERMEDIATE_LOCAL_STAGE"]
    first_comparable_coulomb_stage = None
    material_amplification_tol = NUMERIC_FIELD_TOL
    prev_abs_delta_lj = None

    for stage in stage_order:
        plain_stage = stage_data_plain[stage]
        patch_stage = stage_data_patch_b[stage]
        plain_lj = plain_stage["lj_sr"]
        patch_lj = patch_stage["lj_sr"]
        plain_coul = plain_stage["coulomb_sr"]
        patch_coul = patch_stage["coulomb_sr"]
        delta_lj = None if plain_lj is None or patch_lj is None else patch_lj - plain_lj
        abs_delta_lj = None if delta_lj is None else abs(delta_lj)
        delta_increment = None
        first_amplification_here = False
        if prev_abs_delta_lj is not None and abs_delta_lj is not None:
            delta_increment = abs_delta_lj - prev_abs_delta_lj
            if first_internal_stage is None and delta_increment > material_amplification_tol:
                first_internal_stage = stage
                first_internal_increment = delta_increment
                first_amplification_here = True
            elif first_internal_stage is None:
                earlier_non_amplifying_stages.append(stage)
        delta_coul = None if plain_coul is None or patch_coul is None else patch_coul - plain_coul
        if (
            first_comparable_coulomb_stage is None
            and delta_coul is not None
            and abs(delta_coul) > NUMERIC_FIELD_TOL
        ):
            first_comparable_coulomb_stage = stage

        plain_locations = plain_stage["code_locations"]
        patch_locations = patch_stage["code_locations"]
        if plain_locations == patch_locations:
            code_location = "; ".join(plain_locations)
        else:
            code_location = f"plain={'; '.join(plain_locations)} | patch_b={'; '.join(patch_locations)}"

        comparison_table.append(
            {
                "stage": stage,
                "code_location": code_location,
                "plain_LJ_SR": plain_lj,
                "patch_b_LJ_SR": patch_lj,
                "delta_LJ_SR": delta_lj,
                "delta_increment_vs_prev": delta_increment,
                "plain_Coulomb_SR": plain_coul,
                "patch_b_Coulomb_SR": patch_coul,
                "delta_Coulomb_SR": delta_coul,
                "first_internal_amplification_here": first_amplification_here,
            }
        )
        if abs_delta_lj is not None:
            prev_abs_delta_lj = abs_delta_lj

    final_internal_plain = stage_data_plain["FINAL_INTERNAL_LEDGER"]
    final_internal_patch_b = stage_data_patch_b["FINAL_INTERNAL_LEDGER"]
    corrected_export_matches_internal = (
        final_internal_plain["lj_sr"] is not None
        and final_internal_patch_b["lj_sr"] is not None
        and abs(final_internal_plain["lj_sr"] - plain_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
        and abs(final_internal_patch_b["lj_sr"] - patch_b_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
        and final_internal_plain["coulomb_sr"] is not None
        and final_internal_patch_b["coulomb_sr"] is not None
        and abs(final_internal_plain["coulomb_sr"] - plain_terms["Coulomb-(SR)"]) <= LEDGER_TRACE_TOL
        and abs(final_internal_patch_b["coulomb_sr"] - patch_b_terms["Coulomb-(SR)"]) <= LEDGER_TRACE_TOL
    )

    if first_internal_stage == "RAW_FIRST_WRITE":
        exact_internal_verdict = "RAW_FIRST_WRITE_CONFIRMED"
    elif first_internal_stage == "RAW_FIRST_READ_OR_REDUCE":
        exact_internal_verdict = "RAW_FIRST_READ_OR_REDUCE_CONFIRMED"
    elif first_internal_stage in ("RAW_POST_WRITE", "RAW_POST_READ_OR_REDUCE"):
        exact_internal_verdict = "LATER_INTERNAL_SUBSTEP_CONFIRMED"
    else:
        exact_internal_verdict = "NOT-YET-RESOLVED"

    asymmetry_verdict = "NOT-YET-RESOLVED"
    if first_internal_stage is not None:
        stage_index = stage_order.index(first_internal_stage)
        if stage_index > 0:
            prev_stage_name = stage_order[stage_index - 1]
            prev_plain_lj = stage_data_plain[prev_stage_name]["lj_sr"]
            prev_patch_lj = stage_data_patch_b[prev_stage_name]["lj_sr"]
            curr_plain_lj = stage_data_plain[first_internal_stage]["lj_sr"]
            curr_patch_lj = stage_data_patch_b[first_internal_stage]["lj_sr"]
            plain_change = (
                None if prev_plain_lj is None or curr_plain_lj is None else curr_plain_lj - prev_plain_lj
            )
            patch_change = (
                None if prev_patch_lj is None or curr_patch_lj is None else curr_patch_lj - prev_patch_lj
            )
            plain_changed = plain_change is not None and abs(plain_change) > NUMERIC_FIELD_TOL
            patch_changed = patch_change is not None and abs(patch_change) > NUMERIC_FIELD_TOL
            if first_internal_stage in ("RAW_FIRST_READ_OR_REDUCE", "RAW_POST_READ_OR_REDUCE"):
                asymmetry_verdict = "READ_REDUCE_INTERPRETATION_DIFFERENCE"
            elif plain_changed and not patch_changed:
                asymmetry_verdict = "PLAIN_SIDE_MUTATION"
            elif patch_changed and not plain_changed:
                asymmetry_verdict = "PATCH_SIDE_MUTATION"
            elif plain_changed and patch_changed:
                asymmetry_verdict = "SYMMETRIC_PROCESS_ASYMMETRIC_INPUT"

    if first_internal_stage is None or first_comparable_coulomb_stage is None:
        coulomb_verdict = "NOT-YET-RESOLVED"
    elif stage_order.index(first_comparable_coulomb_stage) == stage_order.index(first_internal_stage):
        coulomb_verdict = "SAME_INTERNAL_STAGE_COUPLED"
    elif stage_order.index(first_comparable_coulomb_stage) > stage_order.index(first_internal_stage):
        coulomb_verdict = "LATER_STAGE_REFLECTION"
    else:
        coulomb_verdict = "INDEPENDENT_SECONDARY"

    supports_internal_culprit = False
    harness_reason = (
        "RAW_POST_WRITE and RAW_POST_READ_OR_REDUCE remain composite boundaries. The harness now keeps culprit support disabled until a write-ordinal proof isolates the exact internal write."
    )

    plain_exec = {
        stage: (
            {
                "code_location": "; ".join(stage_data_plain[stage]["code_locations"]),
                "executed": bool(stage_data_plain[stage]["rows"]),
                "execution_paths": stage_data_plain[stage]["execution_paths"],
            }
        )
        for stage in (
            "INTERMEDIATE_LOCAL_STAGE",
            "RAW_PRE_TRANSFER",
            "RAW_FIRST_WRITE",
            "RAW_POST_WRITE",
            "RAW_FIRST_READ_OR_REDUCE",
            "RAW_POST_READ_OR_REDUCE",
            "RAW_SR_FORMATION",
        )
    }
    patch_exec = {
        stage: (
            {
                "code_location": "; ".join(stage_data_patch_b[stage]["code_locations"]),
                "executed": bool(stage_data_patch_b[stage]["rows"]),
                "execution_paths": stage_data_patch_b[stage]["execution_paths"],
            }
        )
        for stage in (
            "INTERMEDIATE_LOCAL_STAGE",
            "RAW_PRE_TRANSFER",
            "RAW_FIRST_WRITE",
            "RAW_POST_WRITE",
            "RAW_FIRST_READ_OR_REDUCE",
            "RAW_POST_READ_OR_REDUCE",
            "RAW_SR_FORMATION",
        )
    }

    localization = {
        "raw_sr_internal_substep_inventory": {
            "plain_executed_internal_substeps": plain_exec,
            "patch_b_executed_internal_substeps": patch_exec,
            "contract_match_notes": [
                "INTERMEDIATE_LOCAL_STAGE and RAW_PRE_TRANSFER compare the local source aggregate immediately before crossing the RAW_SR_FORMATION boundary on both sides.",
                "RAW_FIRST_WRITE compares the first mutation of the target energy container on both sides: plain outputBuffer.Vvdw versus exact vdwEnergyTerms.",
                "RAW_POST_WRITE compares the target-container state after writes and before the first later consumer read.",
                "RAW_FIRST_READ_OR_REDUCE and RAW_POST_READ_OR_REDUCE compare the first and final consumer reads of those target containers.",
                "Coulomb-(SR) is treated as contract-matched only from SR_ACCUMULATION onward.",
            ],
        },
        "runtime_trace_dossier": {
            "plain_rows": plain_rows,
            "patch_b_rows": patch_b_rows,
            "plain_energy_terms_step0": plain_terms,
            "patch_b_energy_terms_step0": patch_b_terms,
            "material_amplification_definition": (
                f"abs(delta_LJ_SR_stage_n) - abs(delta_LJ_SR_stage_n-1) > {material_amplification_tol}"
            ),
            "material_amplification_tol": material_amplification_tol,
        },
        "first_internal_amplification_proof": {
            "first_stage": first_internal_stage,
            "first_increment": first_internal_increment,
            "earlier_non_amplifying_stages": earlier_non_amplifying_stages,
            "corrected_export_matches_internal": corrected_export_matches_internal,
        },
        "exact_first_internal_amplification_verdict": exact_internal_verdict,
        "asymmetry_classification_verdict": asymmetry_verdict,
        "coulomb_companion_verdict": coulomb_verdict,
        "comparison_table": comparison_table,
        "supports_raw_sr_internal_culprit": supports_internal_culprit,
        "harness_claim_audit": {
            "supports_raw_sr_internal_culprit": supports_internal_culprit,
            "old_stage_level_only_claim_preserved_correctly": True,
            "reason": harness_reason,
        },
        "provenance": {
            "instrumented_code_locations": [
                "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:first_output_buffer_mutation",
                "src/gromacs/nbnxm/kerneldispatch.cpp:plain_pre_output_buffer_transfer",
                "src/gromacs/nbnxm/kerneldispatch.cpp:plain_output_buffer_post_kernel",
                "src/gromacs/nbnxm/kerneldispatch.cpp:sumKernelEnergyOutputs_first_read",
                "src/gromacs/nbnxm/kerneldispatch.cpp:sumKernelEnergyOutputs_final_total",
                "src/gromacs/mdlib/sim_util.cpp:1700",
                "src/gromacs/mdlib/sim_util.cpp:before_vdwEnergyTerms_transfer",
                "src/gromacs/mdlib/sim_util.cpp:after_pair_loop_vdwEnergyTerms",
                "src/gromacs/mdlib/sim_util.cpp:sumEnergyTerms_first_read",
                "src/gromacs/mdlib/sim_util.cpp:sumEnergyTerms_final_total",
                "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            ],
            "corrected_extractor_code_path": "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            "plain_execution_path": stage_data_plain["INTERMEDIATE_LOCAL_STAGE"]["execution_paths"],
            "patch_b_execution_path": stage_data_patch_b["INTERMEDIATE_LOCAL_STAGE"]["execution_paths"],
            "plain_resolved_output_order": plain_terms_detail["resolved_output_order"],
            "patch_b_resolved_output_order": patch_b_terms_detail["resolved_output_order"],
        },
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "step0": 0,
        "localization": localization,
    }


def dense_patch_b_raw_sr_write_ordinal_trace(
    gmx_bin: str,
    dense_fixture_summary: dict[str, Any],
    commands_log: list[str],
) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    patch_b_dir = Path(coarse["patch_b_work_dir"])

    plain_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(plain_dir))
    patch_b_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(patch_b_dir))
    plain_terms_detail = extract_named_energy_series_detail(
        gmx_bin, plain_dir, "plain", DIAGNOSTIC_ENERGY_TERMS, "m2u_plain_diag_terms", commands_log, "m2u_plain"
    )
    patch_b_terms_detail = extract_named_energy_series_detail(
        gmx_bin,
        patch_b_dir,
        "exact",
        DIAGNOSTIC_ENERGY_TERMS,
        "m2u_patch_b_diag_terms",
        commands_log,
        "m2u_patch_b",
    )
    plain_terms = plain_terms_detail["rows"][0]
    patch_b_terms = patch_b_terms_detail["rows"][0]

    def aggregate_stage_with_paths(
        rows: list[dict[str, str]], stage: str, execution_path_prefixes: tuple[str, ...]
    ) -> dict[str, Any]:
        return aggregate_lj_sr_stage_with_paths(rows, stage, execution_path_prefixes)

    plain_write_ordinals = collect_write_ordinals(plain_rows, ("plain_outputBuffer_after_write_ordinal",))
    patch_write_ordinals = collect_write_ordinals(patch_b_rows, ("exact_vdwEnergyTerms_after_write_ordinal",))
    all_write_ordinals = sorted(set(plain_write_ordinals) | set(patch_write_ordinals))
    write_stage_order = ["RAW_FIRST_WRITE"] + [
        f"AFTER_WRITE_ORDINAL_{ordinal}" for ordinal in all_write_ordinals if ordinal >= 2
    ]
    stage_order = list(write_stage_order)
    stage_order.extend(
        [
            "AFTER_LAST_WRITE_BEFORE_RAW_POST_WRITE",
            "RAW_POST_WRITE",
            "RAW_FIRST_READ_OR_REDUCE",
            "RAW_POST_READ_OR_REDUCE",
            "RAW_SR_FORMATION",
            "SR_ACCUMULATION",
            "FINAL_INTERNAL_LEDGER",
            "CORRECTED_EXPORT",
        ]
    )

    def plain_stage_prefixes(stage: str) -> tuple[str, ...]:
        if stage == "RAW_PRE_TRANSFER":
            return ("plain_cpu4x4_ref_kernel_pre_transfer",)
        if stage == "RAW_FIRST_WRITE" or stage.startswith("AFTER_WRITE_ORDINAL_"):
            return ("plain_outputBuffer_after_write_ordinal",)
        if stage == "AFTER_LAST_WRITE_BEFORE_RAW_POST_WRITE":
            return ("plain_outputBuffer_after_last_write",)
        if stage == "RAW_POST_WRITE":
            return ("plain_output_buffer_after_kernel_write",)
        if stage in ("RAW_FIRST_READ_OR_REDUCE", "RAW_POST_READ_OR_REDUCE"):
            return ("plain_sumKernelEnergyOutputs_",)
        if stage == "RAW_SR_FORMATION":
            return ("plain_",)
        return ()

    def patch_stage_prefixes(stage: str) -> tuple[str, ...]:
        if stage == "RAW_PRE_TRANSFER":
            return ("exact_pairs_local_aggregate_pre_transfer",)
        if stage == "RAW_FIRST_WRITE" or stage.startswith("AFTER_WRITE_ORDINAL_"):
            return ("exact_vdwEnergyTerms_after_write_ordinal",)
        if stage == "AFTER_LAST_WRITE_BEFORE_RAW_POST_WRITE":
            return ("exact_vdwEnergyTerms_after_last_write",)
        if stage == "RAW_POST_WRITE":
            return ("exact_vdwEnergyTerms_post_write",)
        if stage in ("RAW_FIRST_READ_OR_REDUCE", "RAW_POST_READ_OR_REDUCE"):
            return ("exact_sumEnergyTerms_",)
        if stage == "RAW_SR_FORMATION":
            return ("exact_",)
        return ()

    traced_stages = set(stage_order)
    traced_stages.add("RAW_PRE_TRANSFER")
    stage_data_plain = {
        stage: aggregate_stage_with_paths(plain_rows, stage, plain_stage_prefixes(stage))
        for stage in traced_stages
        if stage != "CORRECTED_EXPORT"
    }
    stage_data_patch_b = {
        stage: aggregate_stage_with_paths(patch_b_rows, stage, patch_stage_prefixes(stage))
        for stage in traced_stages
        if stage != "CORRECTED_EXPORT"
    }
    stage_data_plain["CORRECTED_EXPORT"] = {
        "stage": "CORRECTED_EXPORT",
        "rows": [],
        "lj_sr": plain_terms["LJ-(SR)"],
        "coulomb_sr": None,
        "code_locations": ["gmx energy corrected extractor"],
        "execution_paths": ["corrected_export"],
    }
    stage_data_patch_b["CORRECTED_EXPORT"] = {
        "stage": "CORRECTED_EXPORT",
        "rows": [],
        "lj_sr": patch_b_terms["LJ-(SR)"],
        "coulomb_sr": None,
        "code_locations": ["gmx energy corrected extractor"],
        "execution_paths": ["corrected_export"],
    }

    plain_first_write_row = stage_data_plain["RAW_FIRST_WRITE"]["rows"][0] if stage_data_plain["RAW_FIRST_WRITE"]["rows"] else {}
    patch_first_write_row = stage_data_patch_b["RAW_FIRST_WRITE"]["rows"][0] if stage_data_patch_b["RAW_FIRST_WRITE"]["rows"] else {}
    plain_post_write_row = stage_data_plain["RAW_POST_WRITE"]["rows"][0] if stage_data_plain["RAW_POST_WRITE"]["rows"] else {}
    patch_post_write_row = stage_data_patch_b["RAW_POST_WRITE"]["rows"][0] if stage_data_patch_b["RAW_POST_WRITE"]["rows"] else {}

    plain_write_count = int(plain_post_write_row.get("write_count", str(len(plain_write_ordinals) or 0)))
    patch_write_count = int(patch_post_write_row.get("write_count", str(len(patch_write_ordinals) or 0)))
    plain_output_buffer_count = (
        None
        if plain_post_write_row.get("output_buffer_count") is None
        else int(plain_post_write_row["output_buffer_count"])
    )
    plain_target_container = plain_first_write_row.get("target_container")
    patch_target_container = patch_first_write_row.get("target_container")

    plain_expected_ordinals = list(range(1, plain_write_count + 1)) if plain_write_count > 0 else []
    patch_expected_ordinals = list(range(1, patch_write_count + 1)) if patch_write_count > 0 else []
    plain_has_full_ordinal_trace = plain_write_ordinals == plain_expected_ordinals and plain_write_count > 0
    patch_has_full_ordinal_trace = patch_write_ordinals == patch_expected_ordinals and patch_write_count > 0
    actual_runtime_checkpoints_exist = plain_has_full_ordinal_trace and patch_has_full_ordinal_trace
    same_write_ordinal_depth = plain_write_count == patch_write_count
    contract_match = (
        actual_runtime_checkpoints_exist
        and plain_target_container == "outputBuffer.Vvdw"
        and patch_target_container == "vdwEnergyTerms"
        and plain_output_buffer_count == 1
        and same_write_ordinal_depth
    )

    comparison_table: list[dict[str, Any]] = []
    first_delta_write_stage = None
    first_delta_write_increment = None
    earlier_write_ordinals_exonerated: list[str] = []
    materiality_tol = NUMERIC_FIELD_TOL
    prev_delta_lj = None

    write_stage_set = set(write_stage_order)
    for stage in stage_order:
        plain_stage = stage_data_plain[stage]
        patch_stage = stage_data_patch_b[stage]
        plain_lj = plain_stage["lj_sr"]
        patch_lj = patch_stage["lj_sr"]
        delta_lj = None if plain_lj is None or patch_lj is None else patch_lj - plain_lj
        delta_increment = None
        first_delta_changing_write_here = False
        if prev_delta_lj is not None and delta_lj is not None:
            delta_increment = delta_lj - prev_delta_lj
            if stage in write_stage_set and first_delta_write_stage is None and abs(delta_increment) > materiality_tol:
                first_delta_write_stage = stage
                first_delta_write_increment = delta_increment
                first_delta_changing_write_here = True
            elif stage in write_stage_set and first_delta_write_stage is None:
                earlier_write_ordinals_exonerated.append(stage)
        elif stage == "RAW_FIRST_WRITE" and delta_lj is not None:
            earlier_write_ordinals_exonerated.append(stage)

        plain_locations = plain_stage["code_locations"]
        patch_locations = patch_stage["code_locations"]
        if plain_locations == patch_locations:
            code_location = "; ".join(plain_locations)
        else:
            code_location = f"plain={'; '.join(plain_locations)} | patch_b={'; '.join(patch_locations)}"

        comparison_table.append(
            {
                "stage": stage,
                "code_location": code_location,
                "plain_LJ_SR_running_total": plain_lj,
                "patch_b_LJ_SR_running_total": patch_lj,
                "delta_LJ_SR": delta_lj,
                "delta_increment_vs_prev": delta_increment,
                "first_delta_changing_write_here": first_delta_changing_write_here,
            }
        )
        if delta_lj is not None:
            prev_delta_lj = delta_lj

    final_internal_plain = stage_data_plain["FINAL_INTERNAL_LEDGER"]
    final_internal_patch_b = stage_data_patch_b["FINAL_INTERNAL_LEDGER"]
    corrected_export_matches_internal = (
        final_internal_plain["lj_sr"] is not None
        and final_internal_patch_b["lj_sr"] is not None
        and abs(final_internal_plain["lj_sr"] - plain_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
        and abs(final_internal_patch_b["lj_sr"] - patch_b_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
    )

    last_write_stage_name = (
        f"AFTER_WRITE_ORDINAL_{all_write_ordinals[-1]}"
        if all_write_ordinals and all_write_ordinals[-1] >= 2
        else "RAW_FIRST_WRITE"
    )
    last_write_row = next((row for row in comparison_table if row["stage"] == last_write_stage_name), None)
    after_last_row = next(
        (row for row in comparison_table if row["stage"] == "AFTER_LAST_WRITE_BEFORE_RAW_POST_WRITE"), None
    )
    raw_post_write_row_cmp = next((row for row in comparison_table if row["stage"] == "RAW_POST_WRITE"), None)
    if first_delta_write_stage is not None and contract_match:
        exact_first_write_verdict = "WRITE_ORDINAL_CONFIRMED"
    elif (
        last_write_row is not None
        and after_last_row is not None
        and last_write_row["delta_LJ_SR"] is not None
        and after_last_row["delta_LJ_SR"] is not None
        and abs(after_last_row["delta_LJ_SR"] - last_write_row["delta_LJ_SR"]) > materiality_tol
    ):
        exact_first_write_verdict = "WRITE_ORDINAL_TRACE_EXISTS_BUT_FIRST_NOT_ISOLATED"
    elif (
        after_last_row is not None
        and raw_post_write_row_cmp is not None
        and after_last_row["delta_LJ_SR"] is not None
        and raw_post_write_row_cmp["delta_LJ_SR"] is not None
        and abs(raw_post_write_row_cmp["delta_LJ_SR"] - after_last_row["delta_LJ_SR"]) > materiality_tol
    ):
        exact_first_write_verdict = "WRITE_ORDINAL_TRACE_EXISTS_BUT_FIRST_NOT_ISOLATED"
    elif actual_runtime_checkpoints_exist:
        exact_first_write_verdict = "WRITE_ORDINAL_TRACE_EXISTS_BUT_FIRST_NOT_ISOLATED"
    else:
        exact_first_write_verdict = "NOT-YET-RESOLVED"

    side_responsibility_verdict = "NOT-YET-RESOLVED"
    if first_delta_write_stage is not None and contract_match:
        stage_index = stage_order.index(first_delta_write_stage)
        prev_stage_name = stage_order[stage_index - 1]
        prev_plain = stage_data_plain[prev_stage_name]["lj_sr"]
        prev_patch = stage_data_patch_b[prev_stage_name]["lj_sr"]
        curr_plain = stage_data_plain[first_delta_write_stage]["lj_sr"]
        curr_patch = stage_data_patch_b[first_delta_write_stage]["lj_sr"]
        plain_change = None if prev_plain is None or curr_plain is None else curr_plain - prev_plain
        patch_change = None if prev_patch is None or curr_patch is None else curr_patch - prev_patch
        plain_changed = plain_change is not None and abs(plain_change) > materiality_tol
        patch_changed = patch_change is not None and abs(patch_change) > materiality_tol
        if plain_changed and not patch_changed:
            side_responsibility_verdict = "PLAIN_SIDE_LATER_WRITE"
        elif patch_changed and not plain_changed:
            side_responsibility_verdict = "PATCH_SIDE_LATER_WRITE"
        elif plain_changed and patch_changed:
            if abs(plain_change) > abs(patch_change) + materiality_tol:
                side_responsibility_verdict = "BOTH_SIDES_WRITE_BUT_PLAIN_DOMINATES"
            elif abs(patch_change) > abs(plain_change) + materiality_tol:
                side_responsibility_verdict = "BOTH_SIDES_WRITE_BUT_PATCH_DOMINATES"
            else:
                side_responsibility_verdict = "SYMMETRIC_WRITES_ASYMMETRIC_INPUT"

    old_stage_order = [
        "RAW_PRE_TRANSFER",
        "RAW_FIRST_WRITE",
        "RAW_POST_WRITE",
        "RAW_FIRST_READ_OR_REDUCE",
        "RAW_POST_READ_OR_REDUCE",
        "RAW_SR_FORMATION",
        "SR_ACCUMULATION",
        "FINAL_INTERNAL_LEDGER",
        "CORRECTED_EXPORT",
    ]
    old_first_stage = None
    prev_old_abs_delta = None
    for stage in old_stage_order:
        plain_lj = stage_data_plain[stage]["lj_sr"]
        patch_lj = stage_data_patch_b[stage]["lj_sr"]
        delta_lj = None if plain_lj is None or patch_lj is None else patch_lj - plain_lj
        abs_delta_lj = None if delta_lj is None else abs(delta_lj)
        if prev_old_abs_delta is not None and abs_delta_lj is not None:
            if old_first_stage is None and (abs_delta_lj - prev_old_abs_delta) > materiality_tol:
                old_first_stage = stage
                break
        if abs_delta_lj is not None:
            prev_old_abs_delta = abs_delta_lj

    if old_first_stage == "RAW_FIRST_WRITE":
        old_exact_internal_verdict = "RAW_FIRST_WRITE_CONFIRMED"
    elif old_first_stage == "RAW_FIRST_READ_OR_REDUCE":
        old_exact_internal_verdict = "RAW_FIRST_READ_OR_REDUCE_CONFIRMED"
    elif old_first_stage in ("RAW_POST_WRITE", "RAW_POST_READ_OR_REDUCE"):
        old_exact_internal_verdict = "LATER_INTERNAL_SUBSTEP_CONFIRMED"
    else:
        old_exact_internal_verdict = "NOT-YET-RESOLVED"
    supports_before = old_exact_internal_verdict != "NOT-YET-RESOLVED" and corrected_export_matches_internal
    verdict_before = (
        "RAW_SR_FORMATION INTERNAL CULPRIT IDENTIFIED"
        if supports_before
        else (
            "RAW_SR_FORMATION INTERNAL TRACE STILL PARTIAL"
            if old_first_stage is not None
            else "RAW_SR_FORMATION INTERNAL TRACE FAILED"
        )
    )
    supports_after = (
        contract_match
        and exact_first_write_verdict == "WRITE_ORDINAL_CONFIRMED"
        and corrected_export_matches_internal
    )
    verdict_after = (
        "RAW_SR_FORMATION INTERNAL CULPRIT IDENTIFIED"
        if supports_after
        else (
            "RAW_SR_FORMATION WRITE-ORDINAL TRACE STILL PARTIAL"
            if actual_runtime_checkpoints_exist
            else "RAW_SR_FORMATION WRITE-ORDINAL TRACE FAILED"
        )
    )

    old_overclaim_removed = (
        not supports_after if exact_first_write_verdict != "WRITE_ORDINAL_CONFIRMED" else supports_after
    )
    final_verdict = (
        "PASS"
        if (
            actual_runtime_checkpoints_exist
            and contract_match
            and corrected_export_matches_internal
            and old_overclaim_removed
        )
        else ("PARTIAL" if actual_runtime_checkpoints_exist and old_overclaim_removed else "FAIL")
    )

    localization = {
        "write_ordinal_instrumentation_inventory": {
            "plain_side_instrumented_checkpoints": [
                "RAW_FIRST_WRITE",
                *[f"AFTER_WRITE_ORDINAL_{ordinal}" for ordinal in plain_write_ordinals if ordinal >= 2],
                "AFTER_LAST_WRITE_BEFORE_RAW_POST_WRITE",
                "RAW_POST_WRITE",
            ],
            "patch_b_side_instrumented_checkpoints": [
                "RAW_FIRST_WRITE",
                *[f"AFTER_WRITE_ORDINAL_{ordinal}" for ordinal in patch_write_ordinals if ordinal >= 2],
                "AFTER_LAST_WRITE_BEFORE_RAW_POST_WRITE",
                "RAW_POST_WRITE",
            ],
            "plain_actual_executed_write_count": plain_write_count,
            "patch_b_actual_executed_write_count": patch_write_count,
            "same_write_ordinal_depth": same_write_ordinal_depth,
            "plain_target_container": plain_target_container,
            "patch_b_target_container": patch_target_container,
            "plain_output_buffer_count": plain_output_buffer_count,
            "contract_match_note": (
                "Compared checkpoints use the same contract: target container running total immediately after write ordinal N."
                if contract_match
                else "Contract match is limited. Plain target container must resolve to a single outputBuffer.Vvdw, Patch-B must resolve to vdwEnergyTerms, and both sides must expose the same write-ordinal depth."
            ),
        },
        "runtime_write_ordinal_trace_dossier": {
            "plain_rows": plain_rows,
            "patch_b_rows": patch_b_rows,
            "plain_energy_terms_step0": plain_terms,
            "patch_b_energy_terms_step0": patch_b_terms,
            "material_delta_change_definition": f"abs(delta_LJ_SR_checkpoint_n - delta_LJ_SR_checkpoint_n-1) > {materiality_tol}",
            "material_delta_change_tol": materiality_tol,
        },
        "comparison_table": comparison_table,
        "exact_first_delta_changing_write_proof": {
            "first_stage": first_delta_write_stage,
            "first_increment": first_delta_write_increment,
            "earlier_write_ordinals_exonerated": earlier_write_ordinals_exonerated,
            "corrected_export_matches_internal": corrected_export_matches_internal,
        },
        "exact_first_write_verdict": exact_first_write_verdict,
        "side_responsibility_verdict": side_responsibility_verdict,
        "coulomb_companion_verdict": "OMITTED_NO_CONTRACT_MATCH",
        "overclaim_shutdown_proof": {
            "supports_raw_sr_internal_culprit_before": supports_before,
            "supports_raw_sr_internal_culprit_after": supports_after,
            "culprit_identified_verdict_before": verdict_before,
            "culprit_identified_verdict_after": verdict_after,
            "exact_rule_change_summary": (
                "Composite RAW_POST_WRITE/RAW_POST_READ_OR_REDUCE evidence is no longer sufficient. Culprit support now requires exact write-ordinal confirmation plus final-ledger/export reconciliation."
            ),
        },
        "harness_claim_audit": {
            "supports_exact_raw_internal_culprit": supports_after,
            "old_overclaim_removed": old_overclaim_removed,
            "reason": (
                "Exact culprit support stays disabled until WRITE_ORDINAL_CONFIRMED on a contract-matched write-ordinal depth."
                if not supports_after
                else "Exact culprit support is re-enabled only because a contract-matched write ordinal is isolated."
            ),
        },
        "write_ordinal_instrumentation_ready": actual_runtime_checkpoints_exist,
        "contract_match": contract_match,
        "final_internal_reconciled": corrected_export_matches_internal,
        "final_verdict": final_verdict,
        "provenance": {
            "instrumented_code_locations": [
                "src/gromacs/nbnxm/kernels_reference/kernel_ref_outer.h:after_outputBuffer_Vvdw_write_ordinal",
                "src/gromacs/nbnxm/kerneldispatch.cpp:plain_output_buffer_post_kernel",
                "src/gromacs/nbnxm/kerneldispatch.cpp:sumKernelEnergyOutputs_first_read",
                "src/gromacs/nbnxm/kerneldispatch.cpp:sumKernelEnergyOutputs_final_total",
                "src/gromacs/mdlib/sim_util.cpp:after_vdwEnergyTerms_write_ordinal",
                "src/gromacs/mdlib/sim_util.cpp:after_pair_loop_vdwEnergyTerms",
                "src/gromacs/mdlib/sim_util.cpp:sumEnergyTerms_first_read",
                "src/gromacs/mdlib/sim_util.cpp:sumEnergyTerms_final_total",
                "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            ],
            "plain_execution_path": stage_data_plain["RAW_FIRST_WRITE"]["execution_paths"],
            "patch_b_execution_path": stage_data_patch_b["RAW_FIRST_WRITE"]["execution_paths"],
            "corrected_extractor_code_path": "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            "plain_resolved_output_order": plain_terms_detail["resolved_output_order"],
            "patch_b_resolved_output_order": patch_b_terms_detail["resolved_output_order"],
        },
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "step0": 0,
        "localization": localization,
    }


def dense_patch_b_aligned_write_contract_trace(
    gmx_bin: str,
    dense_fixture_summary: dict[str, Any],
    commands_log: list[str],
) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    patch_b_dir = Path(coarse["patch_b_work_dir"])

    plain_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(plain_dir))
    patch_b_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(patch_b_dir))
    plain_terms_detail = extract_named_energy_series_detail(
        gmx_bin, plain_dir, "plain", DIAGNOSTIC_ENERGY_TERMS, "m2v_plain_diag_terms", commands_log, "m2v_plain"
    )
    patch_b_terms_detail = extract_named_energy_series_detail(
        gmx_bin,
        patch_b_dir,
        "exact",
        DIAGNOSTIC_ENERGY_TERMS,
        "m2v_patch_b_diag_terms",
        commands_log,
        "m2v_patch_b",
    )
    plain_terms = plain_terms_detail["rows"][0]
    patch_b_terms = patch_b_terms_detail["rows"][0]

    repo_root = Path(__file__).resolve().parents[2]
    prior_m2u_path = (
        repo_root
        / "tests/reference_results/r_respa_m2u_write_ordinal_trace_rerun2/dense_oligomer/fixture_summary.json"
    )
    prior_m2u_audit = {
        "plain_target_container": "outputBuffer.Vvdw",
        "patch_b_target_container": "vdwEnergyTerms",
        "plain_actual_executed_write_count": 255,
        "patch_b_actual_executed_write_count": 20352,
        "semantic_depth_mismatch": (
            "Plain RAW_FIRST_WRITE->RAW_POST_WRITE rows tracked outputBuffer.Vvdw target writes, "
            "while Patch-B tracked per-pair vdwEnergyTerms ledger writes."
        ),
        "why_after_write_ordinal_n_was_not_equivalent": (
            "AFTER_WRITE_ORDINAL_N did not mean the same event depth across sides because plain advanced once per "
            "kernel-visible outputBuffer write, while Patch-B advanced once per admitted pair-energy ledger write."
        ),
    }
    if prior_m2u_path.exists():
        prior_fixture = json.loads(prior_m2u_path.read_text())
        prior_loc = prior_fixture.get("dense_patch_b_raw_sr_write_ordinal_trace", {}).get("localization", {})
        prior_inv = prior_loc.get("write_ordinal_instrumentation_inventory", {})
        prior_m2u_audit.update(
            {
                "plain_target_container": prior_inv.get("plain_target_container", prior_m2u_audit["plain_target_container"]),
                "patch_b_target_container": prior_inv.get("patch_b_target_container", prior_m2u_audit["patch_b_target_container"]),
                "plain_actual_executed_write_count": prior_inv.get(
                    "plain_actual_executed_write_count", prior_m2u_audit["plain_actual_executed_write_count"]
                ),
                "patch_b_actual_executed_write_count": prior_inv.get(
                    "patch_b_actual_executed_write_count", prior_m2u_audit["patch_b_actual_executed_write_count"]
                ),
            }
        )

    chosen_contract = "running_total_after_admitted_pair_energy_event_K"
    rejected_alternative = "running_total_after_reduced_target_container_write_batch_K"
    rejected_reason = (
        "Patch-B pair-energy ledger writes cannot be collapsed upward into the plain outputBuffer write sequence with "
        "a runtime-explicit one-to-one mapping, so that batching would stay heuristic."
    )

    aligned_stage_prefixes_plain = ("plain_aligned_pair_energy_event",)
    aligned_stage_prefixes_patch = ("exact_aligned_pair_energy_event",)
    plain_aligned_ordinals = collect_aligned_event_ordinals(plain_rows, aligned_stage_prefixes_plain)
    patch_aligned_ordinals = collect_aligned_event_ordinals(patch_b_rows, aligned_stage_prefixes_patch)
    all_aligned_ordinals = sorted(set(plain_aligned_ordinals) | set(patch_aligned_ordinals))
    aligned_stage_order = [f"ALIGNED_WRITE_EVENT_{ordinal}" for ordinal in all_aligned_ordinals]
    stage_order = list(aligned_stage_order)
    stage_order.extend(
        [
            "ALIGNED_LAST_EVENT_BEFORE_RAW_POST_WRITE",
            "RAW_POST_WRITE_EQUIVALENT",
            "RAW_FIRST_READ_OR_REDUCE",
            "RAW_POST_READ_OR_REDUCE",
            "RAW_SR_FORMATION",
            "SR_ACCUMULATION",
            "FINAL_INTERNAL_LEDGER",
            "CORRECTED_EXPORT",
        ]
    )

    def plain_stage_prefixes(stage: str) -> tuple[str, ...]:
        if stage.startswith("ALIGNED_WRITE_EVENT_"):
            return ("plain_aligned_pair_energy_event",)
        if stage == "ALIGNED_LAST_EVENT_BEFORE_RAW_POST_WRITE":
            return ("plain_aligned_pair_energy_after_last_event",)
        if stage == "RAW_POST_WRITE_EQUIVALENT":
            return ("plain_aligned_post_write_equivalent",)
        if stage in ("RAW_FIRST_READ_OR_REDUCE", "RAW_POST_READ_OR_REDUCE"):
            return ("plain_sumKernelEnergyOutputs_",)
        if stage == "RAW_SR_FORMATION":
            return ("plain_",)
        return ()

    def patch_stage_prefixes(stage: str) -> tuple[str, ...]:
        if stage.startswith("ALIGNED_WRITE_EVENT_"):
            return ("exact_aligned_pair_energy_event",)
        if stage == "ALIGNED_LAST_EVENT_BEFORE_RAW_POST_WRITE":
            return ("exact_aligned_pair_energy_after_last_event",)
        if stage == "RAW_POST_WRITE_EQUIVALENT":
            return ("exact_aligned_post_write_equivalent",)
        if stage in ("RAW_FIRST_READ_OR_REDUCE", "RAW_POST_READ_OR_REDUCE"):
            return ("exact_sumEnergyTerms_",)
        if stage == "RAW_SR_FORMATION":
            return ("exact_",)
        return ()

    stage_data_plain = {
        stage: aggregate_lj_sr_stage_with_paths(plain_rows, stage, plain_stage_prefixes(stage))
        for stage in stage_order
        if stage != "CORRECTED_EXPORT"
    }
    stage_data_patch_b = {
        stage: aggregate_lj_sr_stage_with_paths(patch_b_rows, stage, patch_stage_prefixes(stage))
        for stage in stage_order
        if stage != "CORRECTED_EXPORT"
    }
    stage_data_plain["CORRECTED_EXPORT"] = {
        "stage": "CORRECTED_EXPORT",
        "rows": [],
        "lj_sr": plain_terms["LJ-(SR)"],
        "coulomb_sr": None,
        "code_locations": ["gmx energy corrected extractor"],
        "execution_paths": ["corrected_export"],
    }
    stage_data_patch_b["CORRECTED_EXPORT"] = {
        "stage": "CORRECTED_EXPORT",
        "rows": [],
        "lj_sr": patch_b_terms["LJ-(SR)"],
        "coulomb_sr": None,
        "code_locations": ["gmx energy corrected extractor"],
        "execution_paths": ["corrected_export"],
    }

    plain_expected_ordinals = list(range(1, len(plain_aligned_ordinals) + 1)) if plain_aligned_ordinals else []
    patch_expected_ordinals = list(range(1, len(patch_aligned_ordinals) + 1)) if patch_aligned_ordinals else []
    plain_has_full_aligned_trace = plain_aligned_ordinals == plain_expected_ordinals and bool(plain_aligned_ordinals)
    patch_has_full_aligned_trace = patch_aligned_ordinals == patch_expected_ordinals and bool(patch_aligned_ordinals)
    actual_runtime_checkpoints_exist = plain_has_full_aligned_trace and patch_has_full_aligned_trace
    same_write_ordinal_depth = len(plain_aligned_ordinals) == len(patch_aligned_ordinals) and bool(plain_aligned_ordinals)
    contract_match = actual_runtime_checkpoints_exist and same_write_ordinal_depth

    comparison_table: list[dict[str, Any]] = []
    first_aligned_stage = None
    first_aligned_increment = None
    earlier_aligned_events_exonerated: list[str] = []
    materiality_tol = NUMERIC_FIELD_TOL
    prev_delta_lj = None
    aligned_stage_set = set(aligned_stage_order)
    for stage in stage_order:
        plain_stage = stage_data_plain[stage]
        patch_stage = stage_data_patch_b[stage]
        plain_lj = plain_stage["lj_sr"]
        patch_lj = patch_stage["lj_sr"]
        delta_lj = None if plain_lj is None or patch_lj is None else patch_lj - plain_lj
        delta_increment = None
        first_delta_changing_here = False
        if prev_delta_lj is not None and delta_lj is not None:
            delta_increment = delta_lj - prev_delta_lj
            if stage in aligned_stage_set and first_aligned_stage is None and abs(delta_increment) > materiality_tol:
                first_aligned_stage = stage
                first_aligned_increment = delta_increment
                first_delta_changing_here = True
            elif stage in aligned_stage_set and first_aligned_stage is None:
                earlier_aligned_events_exonerated.append(stage)
        elif stage == "ALIGNED_WRITE_EVENT_1" and delta_lj is not None:
            earlier_aligned_events_exonerated.append(stage)

        plain_locations = plain_stage["code_locations"]
        patch_locations = patch_stage["code_locations"]
        if plain_locations == patch_locations:
            code_location = "; ".join(plain_locations)
        else:
            code_location = f"plain={'; '.join(plain_locations)} | patch_b={'; '.join(patch_locations)}"

        comparison_table.append(
            {
                "stage": stage,
                "code_location": code_location,
                "plain_LJ_SR_running_total": plain_lj,
                "patch_b_LJ_SR_running_total": patch_lj,
                "delta_LJ_SR": delta_lj,
                "delta_increment_vs_prev": delta_increment,
                "first_delta_changing_aligned_event_here": first_delta_changing_here,
            }
        )
        if delta_lj is not None:
            prev_delta_lj = delta_lj

    final_internal_plain = stage_data_plain["FINAL_INTERNAL_LEDGER"]
    final_internal_patch_b = stage_data_patch_b["FINAL_INTERNAL_LEDGER"]
    corrected_export_matches_internal = (
        final_internal_plain["lj_sr"] is not None
        and final_internal_patch_b["lj_sr"] is not None
        and abs(final_internal_plain["lj_sr"] - plain_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
        and abs(final_internal_patch_b["lj_sr"] - patch_b_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
    )

    admissibility_verdict = (
        "CONTRACT_ALIGNED_AND_ADMISSIBLE"
        if contract_match
        else ("ALIGNMENT_IMPROVED_BUT_STILL_INADMISSIBLE" if actual_runtime_checkpoints_exist else "NOT-YET-RESOLVED")
    )
    exact_first_write_admissibility_verdict = (
        "EXACT_FIRST_WRITE_NOW_ADMISSIBLE"
        if contract_match and same_write_ordinal_depth and first_aligned_stage is not None
        else ("CULPRIT_STILL_BLOCKED_BY_CONTRACT" if not contract_match or not same_write_ordinal_depth else "NOT-YET-RESOLVED")
    )
    culprit_support_enabled = False
    culprit_reinstatement_blocked = True
    overclaim_reason = (
        "Scope remains pre-reinstatement. Even with admissible alignment, culprit support stays disabled until a dedicated culprit milestone chooses to re-enable it."
        if contract_match
        else "Culprit support remains blocked because the aligned contract is not yet admissible."
    )
    final_verdict = (
        "PASS"
        if (
            contract_match
            and same_write_ordinal_depth
            and actual_runtime_checkpoints_exist
            and corrected_export_matches_internal
            and exact_first_write_admissibility_verdict == "EXACT_FIRST_WRITE_NOW_ADMISSIBLE"
        )
        else ("PARTIAL" if actual_runtime_checkpoints_exist and corrected_export_matches_internal else "FAIL")
    )

    localization = {
        "contract_gap_audit": prior_m2u_audit,
        "alignment_design_note": {
            "chosen_contract": chosen_contract,
            "rejected_alternative": rejected_alternative,
            "admissibility_rationale": (
                "Lowering plain to the admitted pair-energy event depth matches Patch-B's natural per-pair ledger event semantics one-for-one."
            ),
            "rejected_alternative_reason": rejected_reason,
        },
        "aligned_runtime_trace_dossier": comparison_table,
        "aligned_event_proof": {
            "first_stage": first_aligned_stage,
            "first_increment": first_aligned_increment,
            "earlier_aligned_events_exonerated": earlier_aligned_events_exonerated,
        },
        "aligned_event_inventory": {
            "plain_aligned_event_count": len(plain_aligned_ordinals),
            "patch_b_aligned_event_count": len(patch_aligned_ordinals),
            "same_write_ordinal_depth": same_write_ordinal_depth,
            "plain_execution_path": "plain_aligned_pair_energy_event",
            "patch_b_execution_path": "exact_aligned_pair_energy_event",
        },
        "admissibility_verdict": admissibility_verdict,
        "exact_first_write_admissibility_verdict": exact_first_write_admissibility_verdict,
        "overclaim_status": {
            "culprit_support_enabled": culprit_support_enabled,
            "culprit_reinstatement_blocked": culprit_reinstatement_blocked,
            "reason": overclaim_reason,
        },
        "contract_match": contract_match,
        "same_write_ordinal_depth": same_write_ordinal_depth,
        "final_internal_reconciled": corrected_export_matches_internal,
        "final_verdict": final_verdict,
        "provenance": {
            "instrumented_code_locations": [
                "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:after_plain_pair_energy_event",
                "src/gromacs/nbnxm/kerneldispatch.cpp:plain_aligned_post_write_equivalent",
                "src/gromacs/mdlib/sim_util.cpp:after_patch_pair_energy_event",
                "src/gromacs/mdlib/sim_util.cpp:after_pair_loop_vdwEnergyTerms",
                "src/gromacs/mdlib/sim_util.cpp:sumEnergyTerms_first_read",
                "src/gromacs/mdlib/sim_util.cpp:sumEnergyTerms_final_total",
                "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            ],
            "chosen_contract_definition": chosen_contract,
            "rejected_alternative": rejected_alternative,
            "corrected_extractor_code_path": "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            "plain_execution_path": stage_data_plain.get("ALIGNED_WRITE_EVENT_1", {}).get("execution_paths", []),
            "patch_b_execution_path": stage_data_patch_b.get("ALIGNED_WRITE_EVENT_1", {}).get("execution_paths", []),
            "plain_resolved_output_order": plain_terms_detail["resolved_output_order"],
            "patch_b_resolved_output_order": patch_b_terms_detail["resolved_output_order"],
        },
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "step0": 0,
        "localization": localization,
    }


def dense_patch_b_aligned_event_669_trace(
    gmx_bin: str,
    dense_fixture_summary: dict[str, Any],
    commands_log: list[str],
) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    patch_b_dir = Path(coarse["patch_b_work_dir"])

    plain_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(plain_dir))
    patch_b_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(patch_b_dir))
    plain_event_rows = load_aligned_event_identity_rows(plain_dir)
    patch_event_rows = load_aligned_event_identity_rows(patch_b_dir)
    plain_terms_detail = extract_named_energy_series_detail(
        gmx_bin, plain_dir, "plain", DIAGNOSTIC_ENERGY_TERMS, "m2w_plain_diag_terms", commands_log, "m2w_plain"
    )
    patch_b_terms_detail = extract_named_energy_series_detail(
        gmx_bin,
        patch_b_dir,
        "exact",
        DIAGNOSTIC_ENERGY_TERMS,
        "m2w_patch_b_diag_terms",
        commands_log,
        "m2w_patch_b",
    )
    plain_terms = plain_terms_detail["rows"][0]
    patch_b_terms = patch_b_terms_detail["rows"][0]

    event_stages = ["ALIGNED_WRITE_EVENT_668", "ALIGNED_WRITE_EVENT_669", "ALIGNED_WRITE_EVENT_670"]

    def compare_numeric(left: str | None, right: str | None) -> bool:
        if left is None or right is None:
            return left == right
        return abs(float(left) - float(right)) <= NUMERIC_FIELD_TOL

    def stage_lj(rows: list[dict[str, str]], stage: str, prefixes: tuple[str, ...]) -> float | None:
        return aggregate_lj_sr_stage_with_paths(rows, stage, prefixes)["lj_sr"]

    def stage_code_location(rows: list[dict[str, str]], stage: str, prefixes: tuple[str, ...]) -> str:
        stage_data = aggregate_lj_sr_stage_with_paths(rows, stage, prefixes)
        return "; ".join(stage_data["code_locations"])

    event_identity_dossier: list[dict[str, Any]] = []
    event_local_delta_table: list[dict[str, Any]] = []
    first_differing_event = None
    identity_match_669 = False

    for stage in event_stages:
        plain_row = plain_event_rows.get(stage)
        patch_row = patch_event_rows.get(stage)
        plain_identity = (
            None
            if plain_row is None
            else f"pair=({plain_row.get('pair_i')},{plain_row.get('pair_j')}) "
                 f"types=({plain_row.get('type_i')},{plain_row.get('type_j')}) "
                 f"key={plain_row.get('event_ordering_key')}"
        )
        patch_identity = (
            None
            if patch_row is None
            else f"pair=({patch_row.get('pair_i')},{patch_row.get('pair_j')}) "
                 f"types=({patch_row.get('type_i')},{patch_row.get('type_j')}) "
                 f"key={patch_row.get('event_ordering_key')}"
        )
        identities_match = (
            plain_row is not None
            and patch_row is not None
            and plain_row.get("pair_i") == patch_row.get("pair_i")
            and plain_row.get("pair_j") == patch_row.get("pair_j")
            and plain_row.get("type_i") == patch_row.get("type_i")
            and plain_row.get("type_j") == patch_row.get("type_j")
            and plain_row.get("event_ordering_key") == patch_row.get("event_ordering_key")
        )
        if stage == "ALIGNED_WRITE_EVENT_669":
            identity_match_669 = identities_match

        plain_before = parse_float_field(plain_row, "running_total_before")
        patch_before = parse_float_field(patch_row, "running_total_before")
        plain_after = parse_float_field(plain_row, "running_total_after")
        patch_after = parse_float_field(patch_row, "running_total_after")
        plain_event_lj = parse_float_field(plain_row, "final_event_lj_contribution")
        patch_event_lj = parse_float_field(patch_row, "final_event_lj_contribution")
        delta_event_lj = (
            None if plain_event_lj is None or patch_event_lj is None else patch_event_lj - plain_event_lj
        )
        delta_after = None if plain_after is None or patch_after is None else patch_after - plain_after
        is_first = False
        if first_differing_event is None and delta_event_lj is not None and abs(delta_event_lj) > NUMERIC_FIELD_TOL:
            first_differing_event = stage
            is_first = True

        event_identity_dossier.append(
            {
                "stage": stage,
                "plain_event_identity": plain_identity,
                "patch_b_event_identity": patch_identity,
                "identities_match": identities_match,
                "plain_running_total_before": plain_before,
                "patch_b_running_total_before": patch_before,
                "plain_running_total_after": plain_after,
                "patch_b_running_total_after": patch_after,
            }
        )
        event_local_delta_table.append(
            {
                "stage": stage,
                "plain_event_LJ": plain_event_lj,
                "patch_b_event_LJ": patch_event_lj,
                "delta_event_LJ": delta_event_lj,
                "plain_running_total_before": plain_before,
                "patch_b_running_total_before": patch_before,
                "plain_running_total_after": plain_after,
                "patch_b_running_total_after": patch_after,
                "delta_running_total_after": delta_after,
                "is_first_differing_event": is_first,
            }
        )

    plain_669 = plain_event_rows.get("ALIGNED_WRITE_EVENT_669")
    patch_669 = patch_event_rows.get("ALIGNED_WRITE_EVENT_669")
    arithmetic_fields = [
        "pair_i",
        "pair_j",
        "type_i",
        "type_j",
        "c6",
        "c12",
        "rsq",
        "r",
        "scaling_factor",
        "raw_lj_term",
        "final_event_lj_contribution",
    ]
    arithmetic_input_dossier: list[dict[str, Any]] = []
    differing_fields: list[str] = []
    for field_name in arithmetic_fields:
        plain_value = None if plain_669 is None else plain_669.get(field_name)
        patch_value = None if patch_669 is None else patch_669.get(field_name)
        if field_name in {"pair_i", "pair_j", "type_i", "type_j"}:
            values_match = plain_value == patch_value
        else:
            values_match = compare_numeric(plain_value, patch_value)
        if not values_match:
            differing_fields.append(field_name)
        arithmetic_input_dossier.append(
            {
                "field_name": field_name,
                "plain_value": None if plain_value is None else (int(plain_value) if field_name in {"pair_i", "pair_j", "type_i", "type_j"} else float(plain_value)),
                "patch_b_value": None if patch_value is None else (int(patch_value) if field_name in {"pair_i", "pair_j", "type_i", "type_j"} else float(patch_value)),
                "values_match": values_match,
            }
        )

    arithmetic_source_verdict = "NOT-YET-RESOLVED"
    if not identity_match_669:
        arithmetic_source_verdict = "PAIR_IDENTITY_MISMATCH"
    elif {"c6", "c12"} & set(differing_fields):
        arithmetic_source_verdict = "PARAMETER_INPUT_MISMATCH"
    elif {"rsq", "r"} & set(differing_fields):
        arithmetic_source_verdict = "DISTANCE_OR_GEOMETRY_INPUT_MISMATCH"
    elif {"scaling_factor"} & set(differing_fields):
        arithmetic_source_verdict = "SCALING_OR_SWITCH_INPUT_MISMATCH"
    elif {"raw_lj_term", "final_event_lj_contribution"} & set(differing_fields):
        arithmetic_source_verdict = "RAW_LJ_FORMULA_INPUT_MISMATCH"
    else:
        row_669 = next((row for row in event_local_delta_table if row["stage"] == "ALIGNED_WRITE_EVENT_669"), None)
        if row_669 is not None:
            delta_event = row_669["delta_event_LJ"]
            delta_after = row_669["delta_running_total_after"]
            if (
                delta_event is not None
                and abs(delta_event) <= NUMERIC_FIELD_TOL
                and delta_after is not None
                and abs(delta_after) > NUMERIC_FIELD_TOL
            ):
                arithmetic_source_verdict = "RUNNING_TOTAL_ONLY_NO_EVENT_LOCAL_DIFF"

    final_internal_plain = aggregate_lj_sr_stage_with_paths(
        plain_rows, "FINAL_INTERNAL_LEDGER", ("post_sum_epot_enerd_term",)
    )["lj_sr"]
    final_internal_patch = aggregate_lj_sr_stage_with_paths(
        patch_b_rows, "FINAL_INTERNAL_LEDGER", ("post_sum_epot_enerd_term",)
    )["lj_sr"]
    corrected_export_matches_internal = (
        final_internal_plain is not None
        and final_internal_patch is not None
        and abs(final_internal_plain - plain_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
        and abs(final_internal_patch - patch_b_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
    )

    culprit_support_enabled = False
    culprit_reinstatement_blocked = True
    if identity_match_669 and arithmetic_source_verdict != "NOT-YET-RESOLVED":
        culprit_reason = (
            "Scope remains pre-reinstatement. Event 669 arithmetic divergence is localized, but culprit support stays disabled until a dedicated reinstatement milestone opts in."
        )
    else:
        culprit_reason = (
            "Culprit support remains blocked because event 669 semantic identity and arithmetic divergence are not both proven."
        )

    final_verdict = (
        "PASS"
        if (
            identity_match_669
            and all(item["identities_match"] for item in event_identity_dossier)
            and first_differing_event == "ALIGNED_WRITE_EVENT_669"
            and arithmetic_source_verdict != "NOT-YET-RESOLVED"
            and corrected_export_matches_internal
        )
        else ("PARTIAL" if plain_669 is not None and patch_669 is not None and corrected_export_matches_internal else "FAIL")
    )

    localization = {
        "chosen_contract": "running_total_after_admitted_pair_energy_event_K",
        "event_identity_dossier": event_identity_dossier,
        "arithmetic_input_dossier_event_669": arithmetic_input_dossier,
        "event_local_delta_table": event_local_delta_table,
        "first_differing_event": first_differing_event,
        "arithmetic_source_verdict": arithmetic_source_verdict,
        "culprit_admissibility_status": {
            "culprit_support_enabled": culprit_support_enabled,
            "culprit_reinstatement_blocked": culprit_reinstatement_blocked,
            "reason": culprit_reason,
        },
        "final_internal_reconciled": corrected_export_matches_internal,
        "final_verdict": final_verdict,
        "provenance": {
            "instrumented_code_locations": [
                "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:after_plain_pair_energy_event",
                "src/gromacs/nbnxm/kerneldispatch.cpp:plain_aligned_post_write_equivalent",
                "src/gromacs/mdlib/sim_util.cpp:after_patch_pair_energy_event",
                "src/gromacs/mdlib/sim_util.cpp:after_pair_loop_vdwEnergyTerms",
                "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            ],
            "aligned_event_contract_source": "running_total_after_admitted_pair_energy_event_K",
            "event_identity_extraction_logic": [
                "pair_i",
                "pair_j",
                "type_i",
                "type_j",
                "event_ordering_key",
                "running_total_before",
                "final_event_lj_contribution",
                "running_total_after",
            ],
            "corrected_extractor_code_path": "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
        },
    }
    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "step0": 0,
        "localization": localization,
    }


def dense_patch_b_event_669_geometry_producer_trace(
    gmx_bin: str,
    dense_fixture_summary: dict[str, Any],
    commands_log: list[str],
) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    patch_b_dir = Path(coarse["patch_b_work_dir"])

    plain_lj_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(plain_dir))
    patch_lj_rows = trim_lj_sr_trace_to_first_cycle(load_lj_sr_internal_trace_rows(patch_b_dir))
    plain_identity_rows = load_aligned_event_identity_rows(plain_dir)
    patch_identity_rows = load_aligned_event_identity_rows(patch_b_dir)
    plain_geometry_rows = load_event_669_geometry_rows(plain_dir)
    patch_geometry_rows = load_event_669_geometry_rows(patch_b_dir)
    plain_terms_detail = extract_named_energy_series_detail(
        gmx_bin, plain_dir, "plain", DIAGNOSTIC_ENERGY_TERMS, "m2x_plain_diag_terms", commands_log, "m2x_plain"
    )
    patch_b_terms_detail = extract_named_energy_series_detail(
        gmx_bin,
        patch_b_dir,
        "exact",
        DIAGNOSTIC_ENERGY_TERMS,
        "m2x_patch_b_diag_terms",
        commands_log,
        "m2x_patch_b",
    )
    plain_terms = plain_terms_detail["rows"][0]
    patch_b_terms = patch_b_terms_detail["rows"][0]

    identity_stage = "ALIGNED_WRITE_EVENT_669"
    plain_identity_row = plain_identity_rows.get(identity_stage)
    patch_identity_row = patch_identity_rows.get(identity_stage)
    expected_pair_i = None if plain_identity_row is None else plain_identity_row.get("pair_i")
    expected_pair_j = None if plain_identity_row is None else plain_identity_row.get("pair_j")
    expected_key = None if plain_identity_row is None else plain_identity_row.get("event_ordering_key")

    stage_order = [
        "GEOM_COORD_SOURCE",
        "GEOM_SHIFT_OR_PBC_APPLY",
        "GEOM_DXDYDZ_CONSTRUCTION",
        "GEOM_RSQ_FORMATION",
        "EVENT_669_LJ_INPUT",
    ]
    stage_field_map: dict[str, list[str]] = {
        "GEOM_COORD_SOURCE": [
            "coord_i_x",
            "coord_i_y",
            "coord_i_z",
            "coord_j_x",
            "coord_j_y",
            "coord_j_z",
        ],
        "GEOM_SHIFT_OR_PBC_APPLY": [
            "shift_index",
            "shift_x",
            "shift_y",
            "shift_z",
            "coord_i_shifted_x",
            "coord_i_shifted_y",
            "coord_i_shifted_z",
        ],
        "GEOM_DXDYDZ_CONSTRUCTION": ["dx", "dy", "dz"],
        "GEOM_RSQ_FORMATION": ["rsq"],
        "EVENT_669_LJ_INPUT": ["rsq", "r"],
    }

    def float_matches(left: str | None, right: str | None) -> bool:
        if left is None or right is None:
            return left == right
        return abs(float(left) - float(right)) <= NUMERIC_FIELD_TOL

    geometry_inventory: list[dict[str, Any]] = [
        {
            "stage": "GEOM_COORD_SOURCE",
            "plain_executed": "GEOM_COORD_SOURCE" in plain_geometry_rows,
            "patch_b_executed": "GEOM_COORD_SOURCE" in patch_geometry_rows,
            "semantic_note": "pair-local coordinate source components for pair (18,0) before shift/PBC application",
        },
        {
            "stage": "GEOM_SHIFT_OR_PBC_APPLY",
            "plain_executed": "GEOM_SHIFT_OR_PBC_APPLY" in plain_geometry_rows,
            "patch_b_executed": "GEOM_SHIFT_OR_PBC_APPLY" in patch_geometry_rows,
            "semantic_note": "shift index/vector and shifted i-coordinate after PBC image application",
        },
        {
            "stage": "GEOM_DXDYDZ_CONSTRUCTION",
            "plain_executed": "GEOM_DXDYDZ_CONSTRUCTION" in plain_geometry_rows,
            "patch_b_executed": "GEOM_DXDYDZ_CONSTRUCTION" in patch_geometry_rows,
            "semantic_note": "pair-local dx/dy/dz components used to build rsq",
        },
        {
            "stage": "GEOM_RSQ_FORMATION",
            "plain_executed": "GEOM_RSQ_FORMATION" in plain_geometry_rows,
            "patch_b_executed": "GEOM_RSQ_FORMATION" in patch_geometry_rows,
            "semantic_note": "rsq formed from dx/dy/dz before LJ evaluation",
        },
        {
            "stage": "EVENT_669_LJ_INPUT",
            "plain_executed": "EVENT_669_LJ_INPUT" in plain_geometry_rows,
            "patch_b_executed": "EVENT_669_LJ_INPUT" in patch_geometry_rows,
            "semantic_note": "event-local LJ input stage carrying rsq/r into raw LJ evaluation",
        },
    ]

    geometry_dossier: list[dict[str, Any]] = []
    first_divergence_stage = None
    earlier_exonerated_stages: list[str] = []
    producer_classification_verdict = "NOT-YET-RESOLVED"

    for stage in stage_order:
        plain_row = plain_geometry_rows.get(stage)
        patch_row = patch_geometry_rows.get(stage)
        plain_location = None if plain_row is None else plain_row.get("code_location")
        patch_location = None if patch_row is None else patch_row.get("code_location")
        if plain_location == patch_location:
            code_location = plain_location
        else:
            code_location = f"plain={plain_location} | patch_b={patch_location}"

        pair_identity_matches = (
            plain_row is not None
            and patch_row is not None
            and plain_row.get("pair_i") == patch_row.get("pair_i")
            and plain_row.get("pair_j") == patch_row.get("pair_j")
            and plain_row.get("event_ordering_key") == patch_row.get("event_ordering_key")
            and plain_row.get("pair_i") == expected_pair_i
            and plain_row.get("pair_j") == expected_pair_j
            and plain_row.get("event_ordering_key") == expected_key
        )

        stage_differs = False
        if first_divergence_stage is None:
            if not pair_identity_matches:
                stage_differs = True
                first_divergence_stage = stage
                producer_classification_verdict = "PAIR_TRAVERSAL_STATE_MISMATCH"
            elif plain_row is None or patch_row is None:
                stage_differs = False
            else:
                relevant_fields = stage_field_map[stage]
                stage_differs = any(
                    not float_matches(plain_row.get(field_name), patch_row.get(field_name))
                    for field_name in relevant_fields
                )
                if stage_differs:
                    first_divergence_stage = stage
                    if stage == "GEOM_COORD_SOURCE":
                        producer_classification_verdict = "COORD_SOURCE_MISMATCH"
                    elif stage == "GEOM_SHIFT_OR_PBC_APPLY":
                        producer_classification_verdict = "SHIFT_OR_PBC_APPLICATION_MISMATCH"
                    elif stage == "GEOM_DXDYDZ_CONSTRUCTION":
                        producer_classification_verdict = "DXDYDZ_CONSTRUCTION_MISMATCH"
                    elif stage == "GEOM_RSQ_FORMATION":
                        producer_classification_verdict = "RSQ_FORMATION_MISMATCH"
                    else:
                        producer_classification_verdict = "NOT-YET-RESOLVED"
                else:
                    earlier_exonerated_stages.append(stage)

        geometry_dossier.append(
            {
                "stage": stage,
                "code_location": code_location,
                "plain_pair_identity": None
                if plain_row is None
                else f"pair=({plain_row.get('pair_i')},{plain_row.get('pair_j')}) key={plain_row.get('event_ordering_key')}",
                "patch_b_pair_identity": None
                if patch_row is None
                else f"pair=({patch_row.get('pair_i')},{patch_row.get('pair_j')}) key={patch_row.get('event_ordering_key')}",
                "plain_dx": parse_float_field(plain_row, "dx"),
                "patch_b_dx": parse_float_field(patch_row, "dx"),
                "plain_dy": parse_float_field(plain_row, "dy"),
                "patch_b_dy": parse_float_field(patch_row, "dy"),
                "plain_dz": parse_float_field(plain_row, "dz"),
                "patch_b_dz": parse_float_field(patch_row, "dz"),
                "plain_rsq": parse_float_field(plain_row, "rsq"),
                "patch_b_rsq": parse_float_field(patch_row, "rsq"),
                "plain_r": parse_float_field(plain_row, "r"),
                "patch_b_r": parse_float_field(patch_row, "r"),
                "first_geometry_divergence_here": first_divergence_stage == stage,
            }
        )

    final_internal_plain = aggregate_lj_sr_stage_with_paths(
        plain_lj_rows, "FINAL_INTERNAL_LEDGER", ("post_sum_epot_enerd_term",)
    )["lj_sr"]
    final_internal_patch = aggregate_lj_sr_stage_with_paths(
        patch_lj_rows, "FINAL_INTERNAL_LEDGER", ("post_sum_epot_enerd_term",)
    )["lj_sr"]
    corrected_export_matches_internal = (
        final_internal_plain is not None
        and final_internal_patch is not None
        and abs(final_internal_plain - plain_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
        and abs(final_internal_patch - patch_b_terms["LJ-(SR)"]) <= LEDGER_TRACE_TOL
    )

    culprit_support_enabled = False
    culprit_reinstatement_blocked = True
    culprit_reason = (
        "Scope remains pre-reinstatement. Even with a concrete upstream geometry producer mismatch, culprit support stays disabled until a dedicated reinstatement milestone opts in."
        if first_divergence_stage is not None and producer_classification_verdict != "NOT-YET-RESOLVED"
        else "Culprit support remains blocked because the first upstream geometry producer mismatch is not yet concrete."
    )

    final_verdict = (
        "PASS"
        if (
            expected_pair_i == "18"
            and expected_pair_j == "0"
            and first_divergence_stage is not None
            and producer_classification_verdict != "NOT-YET-RESOLVED"
            and corrected_export_matches_internal
        )
        else (
            "PARTIAL"
            if (
                plain_geometry_rows
                and patch_geometry_rows
                and corrected_export_matches_internal
            )
            else "FAIL"
        )
    )

    localization = {
        "geometry_producer_inventory": geometry_inventory,
        "event_669_upstream_geometry_dossier": geometry_dossier,
        "producer_classification_verdict": producer_classification_verdict,
        "first_divergence_proof": {
            "first_upstream_stage": first_divergence_stage,
            "earlier_traced_stages_exonerated": earlier_exonerated_stages,
            "event_669_identity_binding": {
                "plain_event_identity": None
                if plain_identity_row is None
                else {
                    "pair_i": int(plain_identity_row["pair_i"]),
                    "pair_j": int(plain_identity_row["pair_j"]),
                    "event_ordering_key": plain_identity_row["event_ordering_key"],
                },
                "patch_b_event_identity": None
                if patch_identity_row is None
                else {
                    "pair_i": int(patch_identity_row["pair_i"]),
                    "pair_j": int(patch_identity_row["pair_j"]),
                    "event_ordering_key": patch_identity_row["event_ordering_key"],
                },
            },
        },
        "culprit_admissibility_status": {
            "culprit_support_enabled": culprit_support_enabled,
            "culprit_reinstatement_blocked": culprit_reinstatement_blocked,
            "reason": culprit_reason,
        },
        "final_internal_reconciled": corrected_export_matches_internal,
        "final_verdict": final_verdict,
        "provenance": {
            "instrumented_code_locations": [
                "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h:plain_event_669_geometry_trace_capture",
                "src/gromacs/nbnxm/kernels_reference/kernel_ref_outer.h:plain_coord_source_and_shift_application",
                "src/gromacs/mdlib/sim_util.cpp:coordinates_fetch_before_shift",
                "src/gromacs/mdlib/sim_util.cpp:shift_vec_application_before_dx",
                "src/gromacs/mdlib/sim_util.cpp:dx_vector_construction",
                "src/gromacs/mdlib/sim_util.cpp:iprod_dx_dx_before_lj",
                "src/gromacs/mdlib/sim_util.cpp:rawLjEnergy_factorLj_event_input",
                "tools/run_respa_m2_microfixtures/run_respa_m2.py:load_event_669_geometry_rows",
                "tools/run_respa_m2_microfixtures/run_respa_m2.py:dense_patch_b_event_669_geometry_producer_trace",
                "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
            ],
            "aligned_event_contract_source": "running_total_after_admitted_pair_energy_event_K",
            "event_identity_extraction_logic": "load_aligned_event_identity_rows(stage=ALIGNED_WRITE_EVENT_669)",
            "geometry_producer_trace_logic": stage_order,
            "corrected_extractor_code_path": "tools/run_respa_m2_microfixtures/run_respa_m2.py:extract_named_energy_series_detail",
        },
    }
    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "step0": 0,
        "localization": localization,
    }


def dense_patch_b_reciprocal_internal_delta_trace(
    gmx_bin: str,
    dense_fixture_summary: dict[str, Any],
    commands_log: list[str],
) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    patch_b_dir = Path(coarse["patch_b_work_dir"])

    plain_rows = load_reciprocal_internal_trace_rows(plain_dir)
    patch_b_rows = load_reciprocal_internal_trace_rows(patch_b_dir)
    plain_terms = extract_named_energy_series(
        gmx_bin, plain_dir, "plain", DIAGNOSTIC_ENERGY_TERMS, "m2m_plain_diag_terms", commands_log, "m2m_plain"
    )[0]
    patch_b_terms = extract_named_energy_series(
        gmx_bin,
        patch_b_dir,
        "exact",
        DIAGNOSTIC_ENERGY_TERMS,
        "m2m_patch_b_diag_terms",
        commands_log,
        "m2m_patch_b",
    )[0]

    stage_specs = (
        ("VCORR", "vcorr_q"),
        ("VLR", "vlr_q"),
        ("MERGE_PRE_LEDGER", "merge_pre_ledger"),
        ("FINAL_LEDGER", "final_ledger"),
        ("PME_RECEIVE_EQ", "eq_received"),
        ("PME_RECEIVE_POST_LEDGER", "ledger_after_receive"),
    )

    comparison_table = []
    stage_order = [stage_name for stage_name, _ in stage_specs]
    first_nonzero_stage = None
    earlier_stages_exonerated = True
    for stage_name, field_name in stage_specs:
        plain_row = plain_rows.get(stage_name)
        patch_b_row = patch_b_rows.get(stage_name)
        plain_value = parse_float_field(plain_row, field_name)
        patch_b_value = parse_float_field(patch_b_row, field_name)
        delta = None
        if plain_value is not None and patch_b_value is not None:
            delta = patch_b_value - plain_value
        first_here = (
            earlier_stages_exonerated
            and delta is not None
            and abs(delta) > NUMERIC_FIELD_TOL
        )
        comparison_table.append(
            {
                "stage": stage_name,
                "plain": plain_value,
                "patch_b": patch_b_value,
                "delta_patch_b_minus_plain": delta,
                "first_nonzero_delta_here": first_here,
            }
        )
        if first_here:
            first_nonzero_stage = stage_name
            earlier_stages_exonerated = False
        elif delta is None or abs(delta) > NUMERIC_FIELD_TOL:
            earlier_stages_exonerated = False

    delta_by_stage = {
        row["stage"]: row["delta_patch_b_minus_plain"] for row in comparison_table
    }
    plain_branch = None if plain_rows.get("VLR") is None else plain_rows["VLR"].get("reciprocal_branch")
    patch_b_branch = None if patch_b_rows.get("VLR") is None else patch_b_rows["VLR"].get("reciprocal_branch")

    if first_nonzero_stage == "VCORR":
        classification = "VCORR_Q_ORIGIN"
    elif (
        first_nonzero_stage == "VLR"
        and delta_by_stage.get("VCORR") is not None
        and abs(delta_by_stage["VCORR"]) <= NUMERIC_FIELD_TOL
    ):
        classification = "VLR_Q_ORIGIN"
    elif (
        first_nonzero_stage in {"MERGE_PRE_LEDGER", "FINAL_LEDGER"}
        and delta_by_stage.get("VCORR") is not None
        and delta_by_stage.get("VLR") is not None
        and abs(delta_by_stage["VCORR"]) <= NUMERIC_FIELD_TOL
        and abs(delta_by_stage["VLR"]) <= NUMERIC_FIELD_TOL
    ):
        classification = "MERGE_REPRESENTATION_MISMATCH"
    elif (
        first_nonzero_stage in {"PME_RECEIVE_EQ", "PME_RECEIVE_POST_LEDGER"}
        and delta_by_stage.get("VCORR") is not None
        and delta_by_stage.get("VLR") is not None
        and delta_by_stage.get("MERGE_PRE_LEDGER") is not None
        and delta_by_stage.get("FINAL_LEDGER") is not None
        and abs(delta_by_stage["VCORR"]) <= NUMERIC_FIELD_TOL
        and abs(delta_by_stage["VLR"]) <= NUMERIC_FIELD_TOL
        and abs(delta_by_stage["MERGE_PRE_LEDGER"]) <= NUMERIC_FIELD_TOL
        and abs(delta_by_stage["FINAL_LEDGER"]) <= NUMERIC_FIELD_TOL
    ):
        classification = "RECIPROCAL_SEMANTIC_CONSUMER_MISMATCH"
    else:
        classification = "NOT-YET-RESOLVED"

    final_trace_row_plain = plain_rows.get("PME_RECEIVE_POST_LEDGER") or plain_rows.get("FINAL_LEDGER")
    final_trace_row_patch_b = patch_b_rows.get("PME_RECEIVE_POST_LEDGER") or patch_b_rows.get("FINAL_LEDGER")
    final_trace_key = "ledger_after_receive" if plain_rows.get("PME_RECEIVE_POST_LEDGER") else "final_ledger"
    final_trace_plain = parse_float_field(final_trace_row_plain, final_trace_key)
    final_trace_patch_b = parse_float_field(final_trace_row_patch_b, final_trace_key)
    final_trace_matches_energy_term = (
        final_trace_plain is not None
        and final_trace_patch_b is not None
        and abs(final_trace_plain - plain_terms["Coul.-recip."]) <= LEDGER_TRACE_TOL
        and abs(final_trace_patch_b - patch_b_terms["Coul.-recip."]) <= LEDGER_TRACE_TOL
    )

    supports_origin = (
        first_nonzero_stage is not None
        and classification != "NOT-YET-RESOLVED"
        and plain_branch is not None
        and patch_b_branch is not None
        and plain_branch == patch_b_branch
        and final_trace_matches_energy_term
    )

    localization = {
        "reciprocal_trace_dossier": {
            "plain_rows": plain_rows,
            "patch_b_rows": patch_b_rows,
            "plain_energy_terms_step0": plain_terms,
            "patch_b_energy_terms_step0": patch_b_terms,
        },
        "first_nonzero_delta_stage": first_nonzero_stage,
        "first_nonzero_delta_proof": {
            "first_stage": first_nonzero_stage,
            "earlier_stages_exonerated": [
                row["stage"]
                for row in comparison_table
                if first_nonzero_stage is not None
                and stage_order.index(row["stage"]) < stage_order.index(first_nonzero_stage)
                and row["delta_patch_b_minus_plain"] is not None
                and abs(row["delta_patch_b_minus_plain"]) <= NUMERIC_FIELD_TOL
            ],
        },
        "residual_origin_classification_verdict": classification,
        "comparison_table": comparison_table,
        "provenance_note": {
            "instrumented_code_locations": [
                "src/gromacs/mdlib/force.cpp:ewald_charge_correction",
                "src/gromacs/mdlib/force.cpp:gmx_pme_do/do_ewald Vlr_q assignment",
                "src/gromacs/mdlib/force.cpp:CoulombReciprocalSpace pre-ledger merge",
                "src/gromacs/mdlib/force.cpp:CoulombReciprocalSpace final ledger assignment",
                "src/gromacs/mdlib/sim_util.cpp:pme_receive_force_ener reciprocal receive accumulation",
            ],
            "executed_reciprocal_branch": {
                "plain": plain_branch,
                "patch_b": patch_b_branch,
            },
            "final_trace_matches_energy_term": final_trace_matches_energy_term,
            "final_trace_vs_energy_term": {
                "plain_trace_final_ledger": final_trace_plain,
                "plain_energy_term_coul_recip": plain_terms["Coul.-recip."],
                "patch_b_trace_final_ledger": final_trace_patch_b,
                "patch_b_energy_term_coul_recip": patch_b_terms["Coul.-recip."],
                "energy_term_delta_patch_b_minus_plain": patch_b_terms["Coul.-recip."] - plain_terms["Coul.-recip."],
            },
        },
        "supports_reciprocal_internal_origin": supports_origin,
        "why_not_fully_closed": (
            "This closes only the first equal-depth reciprocal delta stage for dense_oligomer coarse step 0 under Patch-shape B; it does not establish full bookkeeping closure."
        ),
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "step0": 0,
        "localization": localization,
    }


def dense_dispatch_minimal_fix_isolation(dense_fixture_summary: dict[str, Any]) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    exact_dir = Path(coarse["exact_work_dir"])
    probe_work_dirs = coarse.get("dispatch_probe_work_dirs", {})

    runtime_rows = parse_key_value_text(exact_dir / "step0_runtime_exclusions_input.txt")
    append_rows = parse_key_value_text(exact_dir / "step0_append_branch_trace.txt")
    membership_rows = parse_key_value_text(exact_dir / "step0_pair_key_membership_scan.txt")
    downstream_rows = parse_key_value_text(exact_dir / "step0_downstream_contract_trace.txt")

    target_runtime_row = next((row for row in runtime_rows if row.get("stage") == "runtime_exclusions_input"), None)
    target_append_row = next((row for row in append_rows if row.get("role") == "target_pair_0_1"), None)
    target_membership_row = next(
        (row for row in membership_rows if row.get("kind") == "excluded" and row.get("ordinal") == "0"),
        None,
    )
    target_dispatch_row = next(
        (row for row in downstream_rows if row.get("stage") == "excluded_pairs_dispatch_contract"),
        None,
    )

    baseline_target = load_dispatch_probe_state(exact_dir, "target_pair_0_1")
    baseline_control = load_dispatch_probe_state(exact_dir, "control_pair_0_4")

    probe_states = {
        spec["key"]: load_dispatch_probe_state(Path(probe_work_dirs[spec["key"]]), "target_pair_0_1")
        for spec in M2J_PROBE_SPECS
        if spec["key"] in probe_work_dirs
    }
    control_probe_states = {
        spec["key"]: load_dispatch_probe_state(Path(probe_work_dirs[spec["key"]]), "control_pair_0_4")
        for spec in M2J_PROBE_SPECS
        if spec["key"] in probe_work_dirs
    }

    include_probe = probe_states.get("includepair_policy")
    active_probe = probe_states.get("active_contributions")
    routing_probe = probe_states.get("outer_routing")
    correction_probe = probe_states.get("correction_outer_selection")

    include_verdict = (
        "SUFFICIENT-BUT-NOT-MINIMAL"
        if include_probe is not None
        and not include_probe["first_bad_semantics_occurs"]
        and baseline_target["first_bad_semantics_occurs"]
        else "NOT-CAUSAL"
    )
    active_verdict = (
        "SUFFICIENT-BUT-NOT-MINIMAL"
        if active_probe is not None
        and not active_probe["first_bad_semantics_occurs"]
        and baseline_target["first_bad_semantics_occurs"]
        else "NOT-CAUSAL"
    )
    routing_verdict = (
        "NOT-CAUSAL"
        if routing_probe is not None
        and routing_probe["first_bad_semantics_occurs"]
        and not routing_probe["physical_outer_realization"]
        else "NECESSARY-BUT-NOT-SUFFICIENT"
        if routing_probe is not None and not routing_probe["first_bad_semantics_occurs"]
        else "NOT-CAUSAL"
    )

    minimal_fix_candidate_isolated = (
        correction_probe is not None
        and baseline_target["first_bad_semantics_occurs"]
        and baseline_target["physical_outer_realization"]
        and correction_probe["admitted"]
        and correction_probe["effective_outer_active"]
        and not correction_probe["first_bad_semantics_occurs"]
        and not correction_probe["physical_outer_realization"]
    )

    control_clean = (
        baseline_control["admitted"]
        and baseline_control["first_bad_semantics_occurs"]
        and baseline_control["physical_outer_realization"]
        and all(
            state["admitted"] and state["first_bad_semantics_occurs"] and state["physical_outer_realization"]
            for state in control_probe_states.values()
        )
    )

    minimal_causality_table = [
        {
            "candidate": "includePair policy",
            "baseline_exact_behavior": (
                "includePair admits target into excludedPairs dispatch; downstream bad semantics persists."
            ),
            "diagnostic_perturbation": "Restrict includePair for target (0,1) inside excludedPairs dispatch only.",
            "semantic_result": (
                "Target is blocked before consumer evaluation; first-bad semantics disappears."
                if include_probe is not None and not include_probe["first_bad_semantics_occurs"]
                else "Target still reaches bad semantics."
            ),
            "first_bad_semantics_still_occurs": None
            if include_probe is None
            else include_probe["first_bad_semantics_occurs"],
            "verdict": include_verdict,
            "probe_state": include_probe,
        },
        {
            "candidate": "activeContributions configuration",
            "baseline_exact_behavior": "Target is admitted with outer contribution active and non-zero outer scalar.",
            "diagnostic_perturbation": "Keep admission, but narrow effective active contributions by removing outer.",
            "semantic_result": (
                "Target remains admitted, but no non-zero outer contribution remains; first-bad semantics disappears."
                if active_probe is not None and not active_probe["first_bad_semantics_occurs"]
                else "Target still has non-zero outer contribution."
            ),
            "first_bad_semantics_still_occurs": None
            if active_probe is None
            else active_probe["first_bad_semantics_occurs"],
            "verdict": active_verdict,
            "probe_state": active_probe,
        },
        {
            "candidate": "outer routing / forceWithVirial selection",
            "baseline_exact_behavior": "Non-zero outer contribution is routed into forceWithVirial and physically written.",
            "diagnostic_perturbation": "Keep admission and outer contribution live, but suppress outer physical write.",
            "semantic_result": (
                "Bad semantics still exists before routing, but physical outer realization is suppressed."
                if routing_probe is not None
                and routing_probe["first_bad_semantics_occurs"]
                and not routing_probe["physical_outer_realization"]
                else "Routing change removes the earlier bad semantics."
            ),
            "first_bad_semantics_still_occurs": None
            if routing_probe is None
            else routing_probe["first_bad_semantics_occurs"],
            "verdict": routing_verdict,
            "probe_state": routing_probe,
        },
        {
            "candidate": "excluded correction -> outer contribution selection",
            "baseline_exact_behavior": "Admitted target promotes correction_scalar into a non-zero outer contribution.",
            "diagnostic_perturbation": (
                "Keep admission, active contributions, and routing, but suppress correction promotion into effective outer scalar."
            ),
            "semantic_result": (
                "Target remains admitted with outer contribution active, but effective_outer_scalar becomes zero; first-bad semantics disappears."
                if correction_probe is not None and not correction_probe["first_bad_semantics_occurs"]
                else "Target still reaches bad semantics."
            ),
            "first_bad_semantics_still_occurs": None
            if correction_probe is None
            else correction_probe["first_bad_semantics_occurs"],
            "verdict": "MINIMAL-FIX-CANDIDATE" if minimal_fix_candidate_isolated else "NOT-CAUSAL",
            "probe_state": correction_probe,
        },
    ]

    localization = {
        "pair_dispatch_internal_trace": {
            "runtime_exclusions_input": target_runtime_row,
            "append_branch": target_append_row,
            "plain_pairlist_membership": target_membership_row,
            "excluded_dispatch_contract": target_dispatch_row,
            "baseline_dispatch_internal_state": baseline_target,
        },
        "candidate_causality_table": minimal_causality_table,
        "candidate_verdicts": {
            "includePair policy": include_verdict,
            "activeContributions configuration": active_verdict,
            "outer routing / forceWithVirial selection": routing_verdict,
            "excluded correction -> outer contribution selection": (
                "MINIMAL-FIX-CANDIDATE" if minimal_fix_candidate_isolated else "NOT-CAUSAL"
            ),
        },
        "minimal_fix_candidate": None
        if not minimal_fix_candidate_isolated
        else {
            "classification": "excluded correction -> outer contribution selection",
            "code_path": str(REPO_ROOT / "src" / "gromacs" / "mdlib" / "sim_util.cpp"),
            "why": (
                "Blocking only correction promotion into the effective outer scalar removes the bad semantics while preserving admission, "
                "baseline activeContributions, and baseline outer routing."
            ),
        },
        "reference_reconciliation": {
            "target_pair": [0, 1],
            "includePair_layer": {
                "reference_semantics": "The cleared exclusion contract should keep the pair out of the physical nonbonded consumer.",
                "baseline_exact": baseline_target["include_row"],
                "restricted_probe": None if include_probe is None else include_probe["include_row"],
            },
            "activeContributions_layer": {
                "reference_semantics": "A valid exclusion pair should not have a live non-zero outer physical contribution.",
                "baseline_exact": baseline_target["active_row"],
                "narrowed_probe": None if active_probe is None else active_probe["active_row"],
                "minimal_probe": None if correction_probe is None else correction_probe["active_row"],
            },
            "outerRouting_layer": {
                "reference_semantics": "No physical outer forceWithVirial write should occur for the excluded bookkeeping pair.",
                "baseline_exact": baseline_target["routing_row"],
                "suppressed_probe": None if routing_probe is None else routing_probe["routing_row"],
            },
            "divergence": (
                "The exact path diverges from reference semantics only after it admits the excluded bookkeeping pair into dispatch-internal evaluation and then promotes correction_scalar into a live outer contribution."
                if minimal_fix_candidate_isolated
                else "The sub-decision divergence is still not exact."
            ),
        },
        "control_result": {
            "pair": [0, 4],
            "baseline_state": baseline_control,
            "probe_states": control_probe_states,
            "control_clean": control_clean,
            "why_clean": (
                "The clean control remains admitted through the standard pairs path and keeps the same active contribution and routing semantics across all dispatch probes."
            ),
        },
        "supports_minimal_fix_candidate": minimal_fix_candidate_isolated,
        "why_not_fully_closed": (
            "This isolates only the narrowest sub-decision for pair (0,1) on dense_oligomer coarse step 0 inside the excludedPairs dispatch site."
        ),
    }
    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "localization": localization,
    }


def dense_narrow_patch_proof(dense_fixture_summary: dict[str, Any]) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    exact_dir = Path(coarse["exact_work_dir"])
    patch_work_dirs = coarse.get("patch_work_dirs", {})

    plain_force_frames = parse_force_dump(plain_dir / "plain_total_force.tsv")
    baseline_force_frames = parse_force_dump(exact_dir / "exact_total_force.tsv")
    step0 = min(set(plain_force_frames) & set(baseline_force_frames))
    plain_force = plain_force_frames[step0]["forces"]

    def trace_signature(state: dict[str, Any]) -> dict[str, Any]:
        include_row = state["include_row"]
        active_row = state["active_row"]
        routing_row = state["routing_row"]
        return {
            "include_pair_effective": None
            if include_row is None
            else parse_bool_text(include_row.get("include_pair_effective")),
            "effective_outer_active": None
            if active_row is None
            else parse_bool_text(active_row.get("effective_outer_active")),
            "outer_scalar_baseline": None
            if active_row is None
            else float(active_row.get("outer_scalar_baseline", "0")),
            "outer_scalar_effective": None
            if active_row is None
            else float(active_row.get("outer_scalar_effective", "0")),
            "actual_outer_write_executed": None
            if routing_row is None
            else parse_bool_text(routing_row.get("actual_outer_write_executed")),
        }

    def same_signature(left: dict[str, Any], right: dict[str, Any]) -> bool:
        keys = (
            "include_pair_effective",
            "effective_outer_active",
            "actual_outer_write_executed",
        )
        if any(left[key] != right[key] for key in keys):
            return False
        scalar_keys = ("outer_scalar_baseline", "outer_scalar_effective")
        return all(
            left[key] is not None
            and right[key] is not None
            and abs(float(left[key]) - float(right[key])) <= NUMERIC_FIELD_TOL
            for key in scalar_keys
        )

    def load_patch_record(work_dir: Path) -> dict[str, Any]:
        force_frames = parse_force_dump(work_dir / "exact_total_force.tsv")
        exact_force = force_frames[step0]["forces"]
        exact_vs_plain = vector_metrics(exact_force, plain_force)
        target_state = load_dispatch_probe_state(work_dir, "target_pair_0_1")
        control_state = load_dispatch_probe_state(work_dir, "control_pair_0_4")
        target_signature = trace_signature(target_state)
        control_signature = trace_signature(control_state)
        return {
            "work_dir": str(work_dir),
            "step0_force_diff": exact_vs_plain,
            "target_state": target_state,
            "control_state": control_state,
            "target_signature": target_signature,
            "control_signature": control_signature,
        }

    baseline_record = load_patch_record(exact_dir)
    patch_records = {
        spec["key"]: load_patch_record(Path(patch_work_dirs[spec["key"]]))
        for spec in M2K_PATCH_SPECS
        if spec["key"] in patch_work_dirs
    }

    baseline_control_signature = baseline_record["control_signature"]

    proof_rows = [
        {
            "variant": "baseline",
            "pair": [0, 1],
            "include_pair_effective": baseline_record["target_signature"]["include_pair_effective"],
            "effective_outer_active": baseline_record["target_signature"]["effective_outer_active"],
            "outer_scalar_effective": baseline_record["target_signature"]["outer_scalar_effective"],
            "actual_outer_write_executed": baseline_record["target_signature"]["actual_outer_write_executed"],
            "first_bad_semantics_occurs": baseline_record["target_state"]["first_bad_semantics_occurs"],
            "exact_total_vs_plain_verdict": (
                "CLOSED"
                if baseline_record["step0_force_diff"]["max_abs"] <= FORCE_BOOKKEEPING_TOL
                else "NOT-CLOSED"
            ),
            "control_unchanged_verdict": "BASELINE",
        },
        {
            "variant": "baseline",
            "pair": [0, 4],
            "include_pair_effective": baseline_record["control_signature"]["include_pair_effective"],
            "effective_outer_active": baseline_record["control_signature"]["effective_outer_active"],
            "outer_scalar_effective": baseline_record["control_signature"]["outer_scalar_effective"],
            "actual_outer_write_executed": baseline_record["control_signature"]["actual_outer_write_executed"],
            "first_bad_semantics_occurs": baseline_record["control_state"]["first_bad_semantics_occurs"],
            "exact_total_vs_plain_verdict": "BASELINE",
            "control_unchanged_verdict": "BASELINE",
        },
    ]

    patch_dossiers = {}
    target_closure_verdicts = {}
    control_preservation_verdicts = {}
    for spec in M2K_PATCH_SPECS:
        record = patch_records.get(spec["key"])
        if record is None:
            continue
        target_closed = (
            not record["target_state"]["first_bad_semantics_occurs"]
            and record["step0_force_diff"]["max_abs"] <= FORCE_BOOKKEEPING_TOL
        )
        control_preserved = same_signature(record["control_signature"], baseline_control_signature)
        target_closure_verdicts[spec["candidate"]] = (
            "TARGET-CLOSED" if target_closed else "TARGET-NOT-CLOSED"
        )
        control_preservation_verdicts[spec["candidate"]] = (
            "CONTROL-PRESERVED" if control_preserved else "CONTROL-REGRESSED"
        )
        patch_dossiers[spec["candidate"]] = {
            "semantic_intent": (
                "Cut excluded-pair correction at raw outerScalar formation inside the excludedPairs outer path."
                if spec["key"] == "patch_shape_a"
                else "Preserve raw outerScalar formation but block excluded-pair correction from becoming effective outer physical contribution."
            ),
            "touched_region": str(REPO_ROOT / "src" / "gromacs" / "mdlib" / "sim_util.cpp"),
            "why_narrow": (
                "Touches only the excludedPairs outer-scalar formation site."
                if spec["key"] == "patch_shape_a"
                else "Touches only the effectiveOuterScalar selection guard for excludedPairs outer routing."
            ),
            "target_state": record["target_state"],
            "control_state": record["control_state"],
            "step0_force_diff": record["step0_force_diff"],
        }
        proof_rows.extend(
            [
                {
                    "variant": spec["candidate"],
                    "pair": [0, 1],
                    "include_pair_effective": record["target_signature"]["include_pair_effective"],
                    "effective_outer_active": record["target_signature"]["effective_outer_active"],
                    "outer_scalar_effective": record["target_signature"]["outer_scalar_effective"],
                    "actual_outer_write_executed": record["target_signature"]["actual_outer_write_executed"],
                    "first_bad_semantics_occurs": record["target_state"]["first_bad_semantics_occurs"],
                    "exact_total_vs_plain_verdict": "CLOSED" if target_closed else "NOT-CLOSED",
                    "control_unchanged_verdict": (
                        "CONTROL-PRESERVED" if control_preserved else "CONTROL-REGRESSED"
                    ),
                },
                {
                    "variant": spec["candidate"],
                    "pair": [0, 4],
                    "include_pair_effective": record["control_signature"]["include_pair_effective"],
                    "effective_outer_active": record["control_signature"]["effective_outer_active"],
                    "outer_scalar_effective": record["control_signature"]["outer_scalar_effective"],
                    "actual_outer_write_executed": record["control_signature"]["actual_outer_write_executed"],
                    "first_bad_semantics_occurs": record["control_state"]["first_bad_semantics_occurs"],
                    "exact_total_vs_plain_verdict": "N/A",
                    "control_unchanged_verdict": (
                        "CONTROL-PRESERVED" if control_preserved else "CONTROL-REGRESSED"
                    ),
                },
            ]
        )

    patch_a = patch_records.get("patch_shape_a")
    patch_b = patch_records.get("patch_shape_b")
    patch_a_closed = target_closure_verdicts.get("Patch-shape A") == "TARGET-CLOSED"
    patch_b_closed = target_closure_verdicts.get("Patch-shape B") == "TARGET-CLOSED"
    patch_a_control = control_preservation_verdicts.get("Patch-shape A") == "CONTROL-PRESERVED"
    patch_b_control = control_preservation_verdicts.get("Patch-shape B") == "CONTROL-PRESERVED"
    patch_a_changes_raw_outer = (
        patch_a is not None
        and abs(
            patch_a["target_signature"]["outer_scalar_baseline"]
            - baseline_record["target_signature"]["outer_scalar_baseline"]
        )
        > NUMERIC_FIELD_TOL
    )
    patch_b_changes_raw_outer = (
        patch_b is not None
        and abs(
            patch_b["target_signature"]["outer_scalar_baseline"]
            - baseline_record["target_signature"]["outer_scalar_baseline"]
        )
        > NUMERIC_FIELD_TOL
    )

    if patch_a_closed and patch_a_control and patch_b_closed and patch_b_control:
        preferred_patch = "PATCH-SHAPE-B preferred"
        narrow_patch_verdict = "PASS"
        minimality_reason = (
            "Patch-shape B preserves raw outerScalar formation and only guards effectiveOuterScalar promotion, "
            "whereas Patch-shape A cuts the raw outerScalar earlier."
        )
    elif patch_b_closed and patch_b_control:
        preferred_patch = "PATCH-SHAPE-B preferred"
        narrow_patch_verdict = "PASS"
        minimality_reason = "Patch-shape B is the only candidate that closes the target while preserving control."
    elif patch_a_closed and patch_a_control:
        preferred_patch = "PATCH-SHAPE-A preferred"
        narrow_patch_verdict = "PASS"
        minimality_reason = "Patch-shape A is the only candidate that closes the target while preserving control."
    elif (patch_a_closed and not patch_a_control) or (patch_b_closed and not patch_b_control):
        preferred_patch = "neither proven yet"
        narrow_patch_verdict = "PARTIAL"
        minimality_reason = "At least one target-closing patch changes control semantics."
    else:
        preferred_patch = "neither proven yet"
        narrow_patch_verdict = "FAIL"
        minimality_reason = "Neither narrow patch closes the target within the locked-scope force tolerance."

    localization = {
        "patch_dossier": patch_dossiers,
        "before_after_proof_table": proof_rows,
        "target_closure_verdict": target_closure_verdicts,
        "control_preservation_verdict": control_preservation_verdicts,
        "minimality_comparison": {
            "Patch-shape A": {
                "target_closed": patch_a_closed,
                "control_preserved": patch_a_control,
                "changes_raw_outer_scalar": patch_a_changes_raw_outer,
                "step0_force_diff": None if patch_a is None else patch_a["step0_force_diff"],
                "blast_radius_note": (
                    "Cuts raw outerScalar formation earlier than the physical-promotion boundary."
                    if patch_a is not None
                    else "Patch not run."
                ),
            },
            "Patch-shape B": {
                "target_closed": patch_b_closed,
                "control_preserved": patch_b_control,
                "changes_raw_outer_scalar": patch_b_changes_raw_outer,
                "step0_force_diff": None if patch_b is None else patch_b["step0_force_diff"],
                "blast_radius_note": (
                    "Preserves raw outerScalar formation and only removes excluded correction from effective outer physical promotion."
                    if patch_b is not None
                    else "Patch not run."
                ),
            },
            "preferred_patch": preferred_patch,
            "why": minimality_reason,
        },
        "reference_reconciliation": {
            "target_pair": [0, 1],
            "reference_semantics": (
                "A valid exclusion bookkeeping pair should not produce a live non-zero outer physical contribution and the exact step-0 total force should re-close to plain."
            ),
            "baseline_exact": baseline_record["target_state"],
            "patch_shape_a": None if patch_a is None else patch_a["target_state"],
            "patch_shape_b": None if patch_b is None else patch_b["target_state"],
        },
        "control_result": {
            "pair": [0, 4],
            "baseline": baseline_record["control_state"],
            "patch_shape_a": None if patch_a is None else patch_a["control_state"],
            "patch_shape_b": None if patch_b is None else patch_b["control_state"],
            "control_preserved": patch_a_control and patch_b_control,
        },
        "supports_narrow_patch_proof": narrow_patch_verdict == "PASS",
        "narrow_patch_verdict": narrow_patch_verdict,
        "final_recommendation": preferred_patch,
        "why_not_fully_closed": (
            "This proves only the locked-scope patch behavior for dense_oligomer coarse step 0; it does not establish full-system correctness."
        ),
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "step0": step0,
        "localization": localization,
    }


def dense_patch_b_bookkeeping_residual_trace(
    gmx_bin: str,
    dense_fixture_summary: dict[str, Any],
    commands_log: list[str],
) -> dict[str, Any]:
    coarse = dense_fixture_summary["coarse"]
    plain_dir = Path(coarse["plain_work_dir"])
    exact_dir = Path(coarse["exact_work_dir"])
    patch_b_dir = Path(coarse["patch_b_work_dir"])
    diagnostic_probe_dir = Path(coarse["bookkeeping_probe_work_dir"])

    plain_terms = extract_named_energy_series(
        gmx_bin, plain_dir, "plain", DIAGNOSTIC_ENERGY_TERMS, "plain_terms", commands_log, "plain"
    )[0]
    baseline_terms = extract_named_energy_series(
        gmx_bin, exact_dir, "exact", DIAGNOSTIC_ENERGY_TERMS, "exact_terms", commands_log, "exact"
    )[0]
    patch_b_terms = extract_named_energy_series(
        gmx_bin, patch_b_dir, "exact", DIAGNOSTIC_ENERGY_TERMS, "patch_b_terms", commands_log, "patch_b"
    )[0]
    diagnostic_probe_terms = extract_named_energy_series(
        gmx_bin,
        diagnostic_probe_dir,
        "exact",
        DIAGNOSTIC_ENERGY_TERMS,
        "patch_b_bookkeeping_probe_terms",
        commands_log,
        "patch_b_bookkeeping_probe",
    )[0]

    baseline_target = load_bookkeeping_trace_state(exact_dir, "target_pair_0_1")
    patch_b_target = load_bookkeeping_trace_state(patch_b_dir, "target_pair_0_1")
    diagnostic_probe_target = load_bookkeeping_trace_state(diagnostic_probe_dir, "target_pair_0_1")
    baseline_control = load_bookkeeping_trace_state(exact_dir, "control_pair_0_4")
    patch_b_control = load_bookkeeping_trace_state(patch_b_dir, "control_pair_0_4")
    diagnostic_probe_control = load_bookkeeping_trace_state(diagnostic_probe_dir, "control_pair_0_4")
    baseline_reciprocal_row = load_bookkeeping_reciprocal_row(exact_dir)
    patch_b_reciprocal_row = load_bookkeeping_reciprocal_row(patch_b_dir)
    diagnostic_probe_reciprocal_row = load_bookkeeping_reciprocal_row(diagnostic_probe_dir)

    def compare_terms(row: dict[str, float]) -> dict[str, float]:
        return {
            "LJ-(SR)_minus_plain_kj_per_mol": row["LJ-(SR)"] - plain_terms["LJ-(SR)"],
            "Coul.-recip._minus_plain_kj_per_mol": row["Coul.-recip."] - plain_terms["Coul.-recip."],
            "Coulomb-(SR)_minus_plain_kj_per_mol": row["Coulomb-(SR)"] - plain_terms["Coulomb-(SR)"],
            "Potential_minus_plain_kj_per_mol": row["Potential"] - plain_terms["Potential"],
        }

    baseline_term_delta = compare_terms(baseline_terms)
    patch_b_term_delta = compare_terms(patch_b_terms)
    diagnostic_probe_term_delta = compare_terms(diagnostic_probe_terms)

    baseline_vs_plain = coarse["exact_vs_plain"]
    patch_b_vs_plain = coarse["patch_b_vs_plain"]
    diagnostic_probe_vs_plain = coarse["bookkeeping_probe_vs_plain"]

    def bookkeeping_ok(series_diff: dict[str, Any]) -> bool:
        return (
            series_diff["step0_force_diff"]["max_abs"] <= FORCE_BOOKKEEPING_TOL
            and series_diff["step0_potential_abs_diff_kj_per_mol"] <= POTENTIAL_BOOKKEEPING_TOL
        )

    baseline_bookkeeping_ok = bookkeeping_ok(baseline_vs_plain)
    patch_b_bookkeeping_ok = bookkeeping_ok(patch_b_vs_plain)
    diagnostic_probe_bookkeeping_ok = bookkeeping_ok(diagnostic_probe_vs_plain)
    patch_b_nonaggregate_deltas = {
        "LJ-(SR)": patch_b_term_delta["LJ-(SR)_minus_plain_kj_per_mol"],
        "Coulomb-(SR)": patch_b_term_delta["Coulomb-(SR)_minus_plain_kj_per_mol"],
        "Coul.-recip.": patch_b_term_delta["Coul.-recip._minus_plain_kj_per_mol"],
    }
    dominant_patch_b_term = max(
        patch_b_nonaggregate_deltas, key=lambda term_name: abs(patch_b_nonaggregate_deltas[term_name])
    )

    control_clean = (
        baseline_control["raw_row"] is not None
        and patch_b_control["raw_row"] is not None
        and diagnostic_probe_control["raw_row"] is not None
        and abs(patch_b_control["effective_outer_scalar"] - baseline_control["effective_outer_scalar"])
        <= NUMERIC_FIELD_TOL
        and abs(
            diagnostic_probe_control["effective_outer_scalar"] - baseline_control["effective_outer_scalar"]
        )
        <= NUMERIC_FIELD_TOL
        and patch_b_control["outer_force_write"] == baseline_control["outer_force_write"]
        and diagnostic_probe_control["outer_force_write"] == baseline_control["outer_force_write"]
        and patch_b_control["bookkeeping_sink_active"] == baseline_control["bookkeeping_sink_active"]
        and diagnostic_probe_control["bookkeeping_sink_active"] == baseline_control["bookkeeping_sink_active"]
    )

    target_pair_sink_exonerated = (
        patch_b_target["raw_scalar_present"]
        and abs(patch_b_target["effective_outer_scalar"]) <= NUMERIC_FIELD_TOL
        and not patch_b_target["outer_force_write"]
        and not patch_b_target["bookkeeping_sink_active"]
        and patch_b_target["energy_row"] is None
    )
    diagnostic_probe_matches_patch_b = (
        diagnostic_probe_target["raw_scalar_present"] == patch_b_target["raw_scalar_present"]
        and abs(diagnostic_probe_target["effective_outer_scalar"] - patch_b_target["effective_outer_scalar"])
        <= NUMERIC_FIELD_TOL
        and diagnostic_probe_target["outer_force_write"] == patch_b_target["outer_force_write"]
        and diagnostic_probe_target["bookkeeping_sink_active"] == patch_b_target["bookkeeping_sink_active"]
    )
    reciprocal_sink_found = (
        patch_b_reciprocal_row is not None
        and diagnostic_probe_reciprocal_row is not None
        and parse_bool_text(patch_b_reciprocal_row.get("residual_visible"))
        and abs(patch_b_term_delta["Coul.-recip._minus_plain_kj_per_mol"])
        > abs(patch_b_term_delta["Coulomb-(SR)_minus_plain_kj_per_mol"])
        and abs(
            float(diagnostic_probe_reciprocal_row.get("received_coulomb_reciprocal_energy", "0"))
            - float(patch_b_reciprocal_row.get("received_coulomb_reciprocal_energy", "0"))
        )
        <= NUMERIC_FIELD_TOL
    )

    first_sink_found = target_pair_sink_exonerated and diagnostic_probe_matches_patch_b and reciprocal_sink_found

    if first_sink_found:
        residual_classification = "BOOKKEEPING-ONLY"
    elif patch_b_target["outer_force_write"]:
        residual_classification = "RESIDUAL-CONSUMER-STILL-LIVE"
    else:
        residual_classification = "NOT-YET-RESOLVED"

    comparison_table = [
        {
            "variant": "baseline",
            "raw_scalar_present": baseline_target["raw_scalar_present"],
            "effective_outer_scalar": baseline_target["effective_outer_scalar"],
            "outer_force_write": baseline_target["outer_force_write"],
            "bookkeeping_sink_active": baseline_target["bookkeeping_sink_active"]
            or parse_bool_text(None if baseline_reciprocal_row is None else baseline_reciprocal_row.get("residual_visible")),
            "residual_visible": baseline_vs_plain["step0_potential_abs_diff_kj_per_mol"] > NUMERIC_FIELD_TOL,
            "bookkeeping_ok": baseline_bookkeeping_ok,
            "step0_potential_abs_diff_kj_per_mol": baseline_vs_plain["step0_potential_abs_diff_kj_per_mol"],
            "lj_sr_minus_plain_kj_per_mol": baseline_term_delta["LJ-(SR)_minus_plain_kj_per_mol"],
            "coul_recip_minus_plain_kj_per_mol": baseline_term_delta["Coul.-recip._minus_plain_kj_per_mol"],
            "coulomb_sr_minus_plain_kj_per_mol": baseline_term_delta["Coulomb-(SR)_minus_plain_kj_per_mol"],
        },
        {
            "variant": "patch_b",
            "raw_scalar_present": patch_b_target["raw_scalar_present"],
            "effective_outer_scalar": patch_b_target["effective_outer_scalar"],
            "outer_force_write": patch_b_target["outer_force_write"],
            "bookkeeping_sink_active": patch_b_target["bookkeeping_sink_active"]
            or parse_bool_text(None if patch_b_reciprocal_row is None else patch_b_reciprocal_row.get("residual_visible")),
            "residual_visible": patch_b_vs_plain["step0_potential_abs_diff_kj_per_mol"] > NUMERIC_FIELD_TOL,
            "bookkeeping_ok": patch_b_bookkeeping_ok,
            "step0_potential_abs_diff_kj_per_mol": patch_b_vs_plain["step0_potential_abs_diff_kj_per_mol"],
            "lj_sr_minus_plain_kj_per_mol": patch_b_term_delta["LJ-(SR)_minus_plain_kj_per_mol"],
            "coul_recip_minus_plain_kj_per_mol": patch_b_term_delta["Coul.-recip._minus_plain_kj_per_mol"],
            "coulomb_sr_minus_plain_kj_per_mol": patch_b_term_delta["Coulomb-(SR)_minus_plain_kj_per_mol"],
        },
        {
            "variant": "patch_b_bookkeeping_suppressed_probe",
            "raw_scalar_present": diagnostic_probe_target["raw_scalar_present"],
            "effective_outer_scalar": diagnostic_probe_target["effective_outer_scalar"],
            "outer_force_write": diagnostic_probe_target["outer_force_write"],
            "bookkeeping_sink_active": diagnostic_probe_target["bookkeeping_sink_active"]
            or parse_bool_text(
                None if diagnostic_probe_reciprocal_row is None else diagnostic_probe_reciprocal_row.get("residual_visible")
            ),
            "residual_visible": diagnostic_probe_vs_plain["step0_potential_abs_diff_kj_per_mol"] > NUMERIC_FIELD_TOL,
            "bookkeeping_ok": diagnostic_probe_bookkeeping_ok,
            "step0_potential_abs_diff_kj_per_mol": diagnostic_probe_vs_plain[
                "step0_potential_abs_diff_kj_per_mol"
            ],
            "lj_sr_minus_plain_kj_per_mol": diagnostic_probe_term_delta["LJ-(SR)_minus_plain_kj_per_mol"],
            "coul_recip_minus_plain_kj_per_mol": diagnostic_probe_term_delta[
                "Coul.-recip._minus_plain_kj_per_mol"
            ],
            "coulomb_sr_minus_plain_kj_per_mol": diagnostic_probe_term_delta[
                "Coulomb-(SR)_minus_plain_kj_per_mol"
            ],
        },
    ]

    localization = {
        "patch_b_bookkeeping_trace_dossier": {
            "target_pair": {
                "pair": [0, 1],
                "baseline": baseline_target,
                "patch_b": patch_b_target,
                "diagnostic_probe": diagnostic_probe_target,
            },
            "control_pair": {
                "pair": [0, 4],
                "baseline": baseline_control,
                "patch_b": patch_b_control,
                "diagnostic_probe": diagnostic_probe_control,
            },
            "bookkeeping_sink_map": [
                {
                    "sink_name": "forceWithVirial",
                    "sink_class": "physical_force_sink",
                    "patch_b_target_receives_contribution": patch_b_target["outer_force_write"],
                },
                {
                    "sink_name": "coulEnergyTerms",
                    "sink_class": "energy_potential_sink",
                    "patch_b_target_receives_contribution": patch_b_target["bookkeeping_sink_active"],
                    "patch_b_target_energy_delta_kj_per_mol": None
                    if patch_b_target["energy_row"] is None
                    else float(patch_b_target["energy_row"].get("coul_energy_delta", "0")),
                },
                {
                    "sink_name": "enerd.term[CoulombReciprocalSpace]",
                    "sink_class": "deferred_bookkeeping_sink",
                    "patch_b_target_receives_contribution": parse_bool_text(
                        None if patch_b_reciprocal_row is None else patch_b_reciprocal_row.get("residual_visible")
                    ),
                    "patch_b_target_energy_delta_kj_per_mol": patch_b_term_delta[
                        "Coul.-recip._minus_plain_kj_per_mol"
                    ],
                },
            ],
        },
        "first_residual_sink": {
            "found": first_sink_found,
            "stage": None if patch_b_reciprocal_row is None else patch_b_reciprocal_row.get("stage"),
            "sink_name": None if patch_b_reciprocal_row is None else patch_b_reciprocal_row.get("sink_name"),
            "sink_class": None if patch_b_reciprocal_row is None else patch_b_reciprocal_row.get("sink_class"),
            "why_earlier_stages_are_exonerated": (
                "Patch B still forms correction_scalar, but the target pair has effectiveOuterScalar = 0, no physical outer force sink, and no target-pair bookkeeping_energy_sink row."
            ),
            "why_later_stages_are_exonerated": (
                "The target-only bookkeeping-suppressed probe does not change the residual, so the remaining step-0 delta lies beyond the target-pair excludedPairs bookkeeping sink."
            ),
        },
        "residual_classification_verdict": residual_classification,
        "baseline_vs_patch_b_table": comparison_table,
        "reference_reconciliation": {
            "plain_terms_step0": plain_terms,
            "baseline_terms_step0": baseline_terms,
            "patch_b_terms_step0": patch_b_terms,
            "diagnostic_probe_terms_step0": diagnostic_probe_terms,
            "interpretation": (
                f"Patch B removes the target pair's outer-force misuse, but the remaining corrected step-0 residual is now led by {dominant_patch_b_term}; Coul.-recip. matches plain at the exported-term level."
            ),
        },
        "control_result": {
            "pair": [0, 4],
            "control_clean": control_clean,
            "baseline": baseline_control,
            "patch_b": patch_b_control,
            "diagnostic_probe": diagnostic_probe_control,
        },
        "supports_patch_b_bookkeeping_trace": first_sink_found and control_clean,
        "why_not_fully_closed": (
            "This classifies only the first locked-scope bookkeeping sink for Patch-shape B on pair (0,1); it does not prove a production-ready combined force/energy fix."
        ),
    }

    return {
        "coarse_dt_ps": coarse["dt_ps"],
        "step0": 0,
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


def summarize_dense_patch_b_bookkeeping_fixture(
    fixture_id: str,
    topology_terms: list[str],
    box_nm: tuple[float, float, float],
    coarse_plain: dict[str, Any],
    coarse_exact: dict[str, Any],
    patch_b_run: dict[str, Any],
    bookkeeping_probe_run: dict[str, Any],
) -> dict[str, Any]:
    coarse_exact_vs_plain = compare_series(coarse_exact, coarse_plain, box_nm)
    patch_b_vs_plain = compare_series(patch_b_run, coarse_plain, box_nm)
    bookkeeping_probe_vs_plain = compare_series(bookkeeping_probe_run, coarse_plain, box_nm)

    return {
        "fixture_id": fixture_id,
        "listed_terms_present": topology_terms,
        "exact_split": {
            "inner_terms": topology_terms + ["nonbonded_inner"],
            "middle_terms": ["nonbonded_middle"],
            "outer_terms": ["pair", "nonbonded_outer", "kspace"],
        },
        "exact_schedule_active": exact_scheduler_active(coarse_exact["schedule"]),
        "bookkeeping_ok": (
            patch_b_vs_plain["step0_force_diff"]["max_abs"] <= FORCE_BOOKKEEPING_TOL
            and patch_b_vs_plain["step0_potential_abs_diff_kj_per_mol"] <= POTENTIAL_BOOKKEEPING_TOL
        ),
        "coarse": {
            "dt_ps": coarse_exact["dt_ps"],
            "nsteps": coarse_exact["nsteps"],
            "plain_work_dir": coarse_plain["work_dir"],
            "exact_work_dir": coarse_exact["work_dir"],
            "patch_b_work_dir": patch_b_run["work_dir"],
            "bookkeeping_probe_work_dir": bookkeeping_probe_run["work_dir"],
            "exact_vs_plain": coarse_exact_vs_plain,
            "patch_b_vs_plain": patch_b_vs_plain,
            "bookkeeping_probe_vs_plain": bookkeeping_probe_vs_plain,
        },
        "comparison_notes": [
            "This milestone audits only Patch-shape B and one target-only bookkeeping-suppressed diagnostic micro-probe.",
            "The bookkeeping reference contract remains the same plain Verlet step-0 energy ledger on the same dense_oligomer harness.",
        ],
    }


def summarize_dense_patch_b_reciprocal_fixture(
    fixture_id: str,
    topology_terms: list[str],
    box_nm: tuple[float, float, float],
    coarse_plain: dict[str, Any],
    coarse_exact: dict[str, Any],
    patch_b_run: dict[str, Any],
) -> dict[str, Any]:
    coarse_exact_vs_plain = compare_series(coarse_exact, coarse_plain, box_nm)
    patch_b_vs_plain = compare_series(patch_b_run, coarse_plain, box_nm)

    return {
        "fixture_id": fixture_id,
        "listed_terms_present": topology_terms,
        "exact_split": {
            "inner_terms": topology_terms + ["nonbonded_inner"],
            "middle_terms": ["nonbonded_middle"],
            "outer_terms": ["pair", "nonbonded_outer", "kspace"],
        },
        "exact_schedule_active": exact_scheduler_active(patch_b_run["schedule"]),
        "bookkeeping_ok": (
            patch_b_vs_plain["step0_force_diff"]["max_abs"] <= FORCE_BOOKKEEPING_TOL
            and patch_b_vs_plain["step0_potential_abs_diff_kj_per_mol"] <= POTENTIAL_BOOKKEEPING_TOL
        ),
        "coarse": {
            "dt_ps": patch_b_run["dt_ps"],
            "nsteps": patch_b_run["nsteps"],
            "plain_work_dir": coarse_plain["work_dir"],
            "exact_work_dir": coarse_exact["work_dir"],
            "patch_b_work_dir": patch_b_run["work_dir"],
            "exact_vs_plain": coarse_exact_vs_plain,
            "patch_b_vs_plain": patch_b_vs_plain,
        },
        "comparison_notes": [
            "This milestone traces only the reciprocal bookkeeping path for plain Verlet versus exact three-level Patch-shape B.",
            "The plain/reference contract remains the step-0 Coulomb reciprocal ledger from the same dense_oligomer harness.",
        ],
    }


def summarize_dense_patch_b_post_final_fixture(
    fixture_id: str,
    topology_terms: list[str],
    box_nm: tuple[float, float, float],
    coarse_plain: dict[str, Any],
    coarse_exact: dict[str, Any],
    patch_b_run: dict[str, Any],
) -> dict[str, Any]:
    coarse_exact_vs_plain = compare_series(coarse_exact, coarse_plain, box_nm)
    patch_b_vs_plain = compare_series(patch_b_run, coarse_plain, box_nm)

    return {
        "fixture_id": fixture_id,
        "listed_terms_present": topology_terms,
        "exact_split": {
            "inner_terms": topology_terms + ["nonbonded_inner"],
            "middle_terms": ["nonbonded_middle"],
            "outer_terms": ["pair", "nonbonded_outer", "kspace"],
        },
        "exact_schedule_active": exact_scheduler_active(patch_b_run["schedule"]),
        "bookkeeping_ok": (
            patch_b_vs_plain["step0_force_diff"]["max_abs"] <= FORCE_BOOKKEEPING_TOL
            and patch_b_vs_plain["step0_potential_abs_diff_kj_per_mol"] <= POTENTIAL_BOOKKEEPING_TOL
        ),
        "coarse": {
            "dt_ps": patch_b_run["dt_ps"],
            "nsteps": patch_b_run["nsteps"],
            "plain_work_dir": coarse_plain["work_dir"],
            "exact_work_dir": coarse_exact["work_dir"],
            "patch_b_work_dir": patch_b_run["work_dir"],
            "exact_vs_plain": coarse_exact_vs_plain,
            "patch_b_vs_plain": patch_b_vs_plain,
        },
        "comparison_notes": [
            "This milestone traces only the post-FINAL-LEDGER mutation/export path for CoulombReciprocalSpace.",
            "The comparison remains plain/reference versus exact three-level Patch-shape B on dense_oligomer coarse step 0.",
        ],
    }


def summarize_dense_patch_b_lj_sr_fixture(
    fixture_id: str,
    topology_terms: list[str],
    box_nm: tuple[float, float, float],
    coarse_plain: dict[str, Any],
    coarse_exact: dict[str, Any],
    patch_b_run: dict[str, Any],
) -> dict[str, Any]:
    coarse_exact_vs_plain = compare_series(coarse_exact, coarse_plain, box_nm)
    patch_b_vs_plain = compare_series(patch_b_run, coarse_plain, box_nm)

    return {
        "fixture_id": fixture_id,
        "listed_terms_present": topology_terms,
        "exact_split": {
            "inner_terms": topology_terms + ["nonbonded_inner"],
            "middle_terms": ["nonbonded_middle"],
            "outer_terms": ["pair", "nonbonded_outer", "kspace"],
        },
        "exact_schedule_active": exact_scheduler_active(patch_b_run["schedule"]),
        "bookkeeping_ok": (
            patch_b_vs_plain["step0_force_diff"]["max_abs"] <= FORCE_BOOKKEEPING_TOL
            and patch_b_vs_plain["step0_potential_abs_diff_kj_per_mol"] <= POTENTIAL_BOOKKEEPING_TOL
        ),
        "coarse": {
            "dt_ps": patch_b_run["dt_ps"],
            "nsteps": patch_b_run["nsteps"],
            "plain_work_dir": coarse_plain["work_dir"],
            "exact_work_dir": coarse_exact["work_dir"],
            "patch_b_work_dir": patch_b_run["work_dir"],
            "exact_vs_plain": coarse_exact_vs_plain,
            "patch_b_vs_plain": patch_b_vs_plain,
        },
        "comparison_notes": [
            "This milestone traces only the LJ-(SR) primary residual path under corrected exported-term mapping.",
            "Coulomb-(SR) is treated only as a secondary companion term on the same dense_oligomer step-0 harness.",
        ],
    }


def summarize_dense_patch_fixture(
    fixture_id: str,
    topology_terms: list[str],
    box_nm: tuple[float, float, float],
    coarse_plain: dict[str, Any],
    coarse_exact: dict[str, Any],
    patch_runs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    coarse_exact_vs_plain = compare_series(coarse_exact, coarse_plain, box_nm)
    patch_vs_plain = {
        key: compare_series(patch_run, coarse_plain, box_nm) for key, patch_run in patch_runs.items()
    }

    return {
        "fixture_id": fixture_id,
        "listed_terms_present": topology_terms,
        "exact_split": {
            "inner_terms": topology_terms + ["nonbonded_inner"],
            "middle_terms": ["nonbonded_middle"],
            "outer_terms": ["pair", "nonbonded_outer", "kspace"],
        },
        "exact_schedule_active": exact_scheduler_active(coarse_exact["schedule"]),
        "bookkeeping_ok": (
            coarse_exact_vs_plain["step0_force_diff"]["max_abs"] <= FORCE_BOOKKEEPING_TOL
            and coarse_exact_vs_plain["step0_potential_abs_diff_kj_per_mol"] <= POTENTIAL_BOOKKEEPING_TOL
        ),
        "coarse": {
            "dt_ps": coarse_exact["dt_ps"],
            "nsteps": coarse_exact["nsteps"],
            "plain_work_dir": coarse_plain["work_dir"],
            "exact_work_dir": coarse_exact["work_dir"],
            "patch_work_dirs": {key: value["work_dir"] for key, value in patch_runs.items()},
            "exact_vs_plain": coarse_exact_vs_plain,
            "patch_vs_plain": patch_vs_plain,
        },
        "exact_schedule_dump": coarse_exact["schedule"],
        "comparison_notes": [
            "Plain Verlet uses integrator = md-vv with the same PME/Cut-off settings as the exact 3-level path.",
            "This milestone tests only two narrow patch shapes inside the already-isolated excluded-correction outer-promotion path.",
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
    grompp_env: dict[str, str] | None = None,
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
        extra_env=grompp_env,
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


def summarize_exact_only_trace_fixture(
    fixture_id: str,
    topology_terms: list[str],
    coarse_exact: dict[str, Any],
    comparison_notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "listed_terms_present": topology_terms,
        "exact_split": {
            "inner_terms": topology_terms + ["nonbonded_inner"],
            "middle_terms": ["nonbonded_middle"],
            "outer_terms": ["pair", "nonbonded_outer", "kspace"],
        },
        "exact_schedule_active": exact_scheduler_active(coarse_exact["schedule"]),
        "coarse": {
            "dt_ps": coarse_exact["dt_ps"],
            "nsteps": coarse_exact["nsteps"],
            "exact_work_dir": coarse_exact["work_dir"],
        },
        "exact_schedule_dump": coarse_exact["schedule"],
        "comparison_notes": comparison_notes
        or [
            "This milestone traces only the upstream ownership/spec lineage that leads into excludedPairs.",
            "No plain-Verlet or PME-side legacy side-reference comparison is needed for this exact handoff proof.",
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    gmx_bin = str(Path(args.gmx_bin).resolve())
    output_root = Path(args.out).resolve()
    if (
        args.upstream_ownership_handoff_trace
        or args.pair_rule_derivation_trace
        or args.downstream_misconsumption_trace
        or args.dispatch_minimal_fix_isolation
        or args.narrow_patch_proof
        or args.locked_scope_bookkeeping_residual_trace
        or args.reciprocal_internal_delta_trace
        or args.post_final_ledger_mutation_trace
        or args.lj_sr_first_amplification_trace
        or args.raw_sr_formation_internal_trace
        or args.raw_sr_write_ordinal_trace
        or args.event_669_geometry_producer_trace
        or args.aligned_event_669_trace
        or args.lj_sr_true_first_raw_trace
        or args.lj_sr_first_sink_trace
    ) and not args.fixtures:
        fixtures = ["dense_oligomer"]
    else:
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

        coarse_plain = None
        coarse_side_reference = None
        coarse_plain_env = None
        coarse_side_reference_env = None
        coarse_exact_env = None
        coarse_exact_grompp_env = None
        if (
            args.dense_force_ownership_isolation
            or args.dense_merge_trace
            or args.dense_early_accumulation_trace
            or args.exact_pair_write_ownership_proof
            or args.upstream_ownership_handoff_trace
            or args.pair_rule_derivation_trace
            or args.downstream_misconsumption_trace
            or args.dispatch_minimal_fix_isolation
            or args.narrow_patch_proof
            or args.locked_scope_bookkeeping_residual_trace
            or args.reciprocal_internal_delta_trace
            or args.post_final_ledger_mutation_trace
            or args.lj_sr_first_amplification_trace
            or args.raw_sr_formation_internal_trace
            or args.raw_sr_write_ordinal_trace
            or args.event_669_geometry_producer_trace
            or args.aligned_event_669_trace
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
        if (
            args.upstream_ownership_handoff_trace
            or args.pair_rule_derivation_trace
            or args.downstream_misconsumption_trace
            or args.dispatch_minimal_fix_isolation
        ) and fixture_id == "dense_oligomer":
            coarse_exact_env = dict(coarse_exact_env or {})
            coarse_exact_env["GMX_DISABLE_MODULAR_SIMULATOR"] = "ON"
            coarse_exact_env["GMX_PCFF_RESPA_OWNERSHIP_HANDOFF_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "exact_three_level"
            )
        if args.downstream_misconsumption_trace and fixture_id == "dense_oligomer":
            coarse_exact_env = dict(coarse_exact_env or {})
            coarse_exact_env["GMX_PCFF_RESPA_PAIR_WRITE_PROOF_DIR"] = str(
                system_root / "dt_0p0005" / "exact_three_level"
            )
            coarse_exact_env["GMX_PCFF_RESPA_DOWNSTREAM_CONTRACT_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "exact_three_level"
            )
        if args.dispatch_minimal_fix_isolation and fixture_id == "dense_oligomer":
            coarse_exact_env = dict(coarse_exact_env or {})
            coarse_exact_env["GMX_PCFF_RESPA_PAIR_WRITE_PROOF_DIR"] = str(
                system_root / "dt_0p0005" / "exact_three_level"
            )
            coarse_exact_env["GMX_PCFF_RESPA_DOWNSTREAM_CONTRACT_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "exact_three_level"
            )
            coarse_exact_env["GMX_PCFF_RESPA_M2J_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "exact_three_level"
            )
            coarse_exact_env["GMX_PCFF_RESPA_M2J_PROBE_MODE"] = "baseline"
        if args.narrow_patch_proof and fixture_id == "dense_oligomer":
            coarse_exact_env = dict(coarse_exact_env or {})
            coarse_exact_env["GMX_PCFF_RESPA_M2K_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "exact_three_level"
            )
            coarse_exact_env["GMX_PCFF_RESPA_M2K_PATCH_MODE"] = "baseline"
        if args.locked_scope_bookkeeping_residual_trace and fixture_id == "dense_oligomer":
            coarse_exact_env = dict(coarse_exact_env or {})
            coarse_exact_env["GMX_PCFF_RESPA_M2L_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "exact_three_level"
            )
            coarse_exact_env["GMX_PCFF_RESPA_M2L_PROBE_MODE"] = "baseline"
        if args.post_final_ledger_mutation_trace and fixture_id == "dense_oligomer":
            coarse_plain_env = dict(coarse_plain_env or {})
            coarse_plain_env["GMX_PCFF_RESPA_M2N_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "plain_verlet"
            )
            coarse_plain_env["GMX_PCFF_RESPA_M2N_MODE"] = "plain"
        if args.lj_sr_first_amplification_trace and fixture_id == "dense_oligomer":
            coarse_plain_env = dict(coarse_plain_env or {})
            coarse_plain_env["GMX_PCFF_RESPA_M2R_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "plain_verlet"
            )
            coarse_plain_env["GMX_PCFF_RESPA_M2R_CASE_LABEL"] = "plain_verlet"
        if args.raw_sr_formation_internal_trace and fixture_id == "dense_oligomer":
            coarse_plain_env = dict(coarse_plain_env or {})
            coarse_plain_env["GMX_PCFF_RESPA_M2S_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "plain_verlet"
            )
            coarse_plain_env["GMX_PCFF_RESPA_M2S_CASE_LABEL"] = "plain_verlet"
        if args.raw_sr_write_ordinal_trace and fixture_id == "dense_oligomer":
            coarse_plain_env = dict(coarse_plain_env or {})
            coarse_plain_env["GMX_PCFF_RESPA_M2U_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "plain_verlet"
            )
            coarse_plain_env["GMX_PCFF_RESPA_M2U_CASE_LABEL"] = "plain_verlet"
        if args.aligned_write_contract_trace and fixture_id == "dense_oligomer":
            coarse_plain_env = dict(coarse_plain_env or {})
            coarse_plain_env["GMX_PCFF_RESPA_M2V_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "plain_verlet"
            )
            coarse_plain_env["GMX_PCFF_RESPA_M2V_CASE_LABEL"] = "plain_verlet"
        if args.aligned_event_669_trace and fixture_id == "dense_oligomer":
            coarse_plain_env = dict(coarse_plain_env or {})
            coarse_plain_env["GMX_PCFF_RESPA_M2W_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "plain_verlet"
            )
            coarse_plain_env["GMX_PCFF_RESPA_M2W_CASE_LABEL"] = "plain_verlet"
        if args.event_669_geometry_producer_trace and fixture_id == "dense_oligomer":
            coarse_plain_env = dict(coarse_plain_env or {})
            coarse_plain_env["GMX_PCFF_RESPA_M2W_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "plain_verlet"
            )
            coarse_plain_env["GMX_PCFF_RESPA_M2W_CASE_LABEL"] = "plain_verlet"
            coarse_plain_env["GMX_PCFF_RESPA_M2X_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "plain_verlet"
            )
            coarse_plain_env["GMX_PCFF_RESPA_M2X_CASE_LABEL"] = "plain_verlet"
        if args.lj_sr_true_first_raw_trace and fixture_id == "dense_oligomer":
            coarse_plain_env = dict(coarse_plain_env or {})
            coarse_plain_env["GMX_PCFF_RESPA_M2Q_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "plain_verlet"
            )
            coarse_plain_env["GMX_PCFF_RESPA_M2Q_CASE_LABEL"] = "plain_verlet"
        if args.lj_sr_first_sink_trace and fixture_id == "dense_oligomer":
            coarse_plain_env = dict(coarse_plain_env or {})
            coarse_plain_env["GMX_PCFF_RESPA_M2P_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "plain_verlet"
            )
            coarse_plain_env["GMX_PCFF_RESPA_M2P_CASE_LABEL"] = "plain_verlet"
        if args.reciprocal_internal_delta_trace and fixture_id == "dense_oligomer":
            coarse_plain_env = dict(coarse_plain_env or {})
            coarse_plain_env["GMX_PCFF_RESPA_M2M_TRACE_DIR"] = str(
                system_root / "dt_0p0005" / "plain_verlet"
            )
            coarse_plain_env["GMX_PCFF_RESPA_M2M_MODE"] = "plain"
        if (args.upstream_ownership_handoff_trace or args.pair_rule_derivation_trace) and fixture_id == "dense_oligomer":
            coarse_exact_grompp_env = {
                "GMX_PCFF_RESPA_OWNERSHIP_HANDOFF_TRACE_DIR": str(
                    system_root / "dt_0p0005" / "exact_three_level"
                )
            }
        if args.dense_bookkeeping_isolation and fixture_id == "dense_oligomer":
            coarse_exact_env = dict(coarse_exact_env or {})
            coarse_exact_env["GMX_PCFF_RESPA_DEBUG"] = "1"

        if (
            args.narrow_patch_proof
            or args.locked_scope_bookkeeping_residual_trace
            or args.reciprocal_internal_delta_trace
            or args.post_final_ledger_mutation_trace
            or args.lj_sr_first_amplification_trace
            or args.raw_sr_formation_internal_trace
            or args.raw_sr_write_ordinal_trace
            or args.event_669_geometry_producer_trace
            or args.aligned_write_contract_trace
            or args.aligned_event_669_trace
            or args.lj_sr_true_first_raw_trace
            or args.lj_sr_first_sink_trace
        ) and fixture_id == "dense_oligomer":
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

        coarse_exact = run_case(
            gmx_bin,
            system_root,
            system_root / "dt_0p0005" / "exact_three_level",
            "exact_three_level",
            DEFAULT_DT_VALUES[0],
            args.total_time_ps,
            commands_log,
            extra_env=coarse_exact_env,
            grompp_env=coarse_exact_grompp_env,
        )
        dispatch_probe_runs: dict[str, dict[str, Any]] = {}
        if args.dispatch_minimal_fix_isolation and fixture_id == "dense_oligomer":
            for probe_spec in M2J_PROBE_SPECS:
                probe_work_dir = system_root / "dt_0p0005" / probe_spec["work_dir_name"]
                probe_env = {
                    "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                    "GMX_PCFF_RESPA_M2J_TRACE_DIR": str(probe_work_dir),
                    "GMX_PCFF_RESPA_M2J_PROBE_MODE": probe_spec["probe_mode"],
                }
                dispatch_probe_runs[probe_spec["key"]] = run_case(
                    gmx_bin,
                    system_root,
                    probe_work_dir,
                    "exact_three_level",
                    DEFAULT_DT_VALUES[0],
                    args.total_time_ps,
                    commands_log,
                    extra_env=probe_env,
                )
        patch_runs: dict[str, dict[str, Any]] = {}
        if args.narrow_patch_proof and fixture_id == "dense_oligomer":
            for patch_spec in M2K_PATCH_SPECS:
                patch_work_dir = system_root / "dt_0p0005" / patch_spec["work_dir_name"]
                patch_env = {
                    "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                    "GMX_TOTAL_FORCE_DUMP_FILE": str(patch_work_dir / "exact_total_force.tsv"),
                    "GMX_PCFF_RESPA_M2K_TRACE_DIR": str(patch_work_dir),
                    "GMX_PCFF_RESPA_M2K_PATCH_MODE": patch_spec["patch_mode"],
                }
                patch_runs[patch_spec["key"]] = run_case(
                    gmx_bin,
                    system_root,
                    patch_work_dir,
                    "exact_three_level",
                    DEFAULT_DT_VALUES[0],
                    args.total_time_ps,
                    commands_log,
                    extra_env=patch_env,
                )
        bookkeeping_probe_run = None
        patch_b_run = None
        if args.locked_scope_bookkeeping_residual_trace and fixture_id == "dense_oligomer":
            patch_b_work_dir = system_root / "dt_0p0005" / "patch_shape_b"
            patch_b_env = {
                "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                "GMX_PCFF_RESPA_M2L_TRACE_DIR": str(patch_b_work_dir),
                "GMX_PCFF_RESPA_M2L_PROBE_MODE": "patch_shape_b",
            }
            patch_b_run = run_case(
                gmx_bin,
                system_root,
                patch_b_work_dir,
                "exact_three_level",
                DEFAULT_DT_VALUES[0],
                args.total_time_ps,
                commands_log,
                extra_env=patch_b_env,
            )
            bookkeeping_probe_work_dir = system_root / "dt_0p0005" / M2L_DIAGNOSTIC_PROBE["work_dir_name"]
            bookkeeping_probe_env = {
                "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                "GMX_PCFF_RESPA_M2L_TRACE_DIR": str(bookkeeping_probe_work_dir),
                "GMX_PCFF_RESPA_M2L_PROBE_MODE": M2L_DIAGNOSTIC_PROBE["probe_mode"],
            }
            bookkeeping_probe_run = run_case(
                gmx_bin,
                system_root,
                bookkeeping_probe_work_dir,
                "exact_three_level",
                DEFAULT_DT_VALUES[0],
                args.total_time_ps,
                commands_log,
                extra_env=bookkeeping_probe_env,
            )
        elif args.post_final_ledger_mutation_trace and fixture_id == "dense_oligomer":
            patch_b_work_dir = system_root / "dt_0p0005" / "patch_shape_b"
            patch_b_env = {
                "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                "GMX_PCFF_RESPA_M2N_TRACE_DIR": str(patch_b_work_dir),
                "GMX_PCFF_RESPA_M2N_MODE": "patch_b",
                "GMX_PCFF_RESPA_M2K_PATCH_MODE": "patch_shape_b",
            }
            patch_b_run = run_case(
                gmx_bin,
                system_root,
                patch_b_work_dir,
                "exact_three_level",
                DEFAULT_DT_VALUES[0],
                args.total_time_ps,
                commands_log,
                extra_env=patch_b_env,
            )
        elif args.reciprocal_internal_delta_trace and fixture_id == "dense_oligomer":
            patch_b_work_dir = system_root / "dt_0p0005" / "patch_shape_b"
            patch_b_env = {
                "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                "GMX_PCFF_RESPA_M2M_TRACE_DIR": str(patch_b_work_dir),
                "GMX_PCFF_RESPA_M2M_MODE": "patch_b",
                "GMX_PCFF_RESPA_M2K_PATCH_MODE": "patch_shape_b",
            }
            patch_b_run = run_case(
                gmx_bin,
                system_root,
                patch_b_work_dir,
                "exact_three_level",
                DEFAULT_DT_VALUES[0],
                args.total_time_ps,
                commands_log,
                extra_env=patch_b_env,
            )
        elif args.lj_sr_first_amplification_trace and fixture_id == "dense_oligomer":
            patch_b_work_dir = system_root / "dt_0p0005" / "patch_shape_b"
            patch_b_env = {
                "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                "GMX_PCFF_RESPA_M2K_PATCH_MODE": "patch_shape_b",
                "GMX_PCFF_RESPA_M2R_TRACE_DIR": str(patch_b_work_dir),
                "GMX_PCFF_RESPA_M2R_CASE_LABEL": "patch_shape_b",
            }
            patch_b_run = run_case(
                gmx_bin,
                system_root,
                patch_b_work_dir,
                "exact_three_level",
                DEFAULT_DT_VALUES[0],
                args.total_time_ps,
                commands_log,
                extra_env=patch_b_env,
            )
        elif args.raw_sr_formation_internal_trace and fixture_id == "dense_oligomer":
            patch_b_work_dir = system_root / "dt_0p0005" / "patch_shape_b"
            patch_b_env = {
                "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                "GMX_PCFF_RESPA_M2K_PATCH_MODE": "patch_shape_b",
                "GMX_PCFF_RESPA_M2S_TRACE_DIR": str(patch_b_work_dir),
                "GMX_PCFF_RESPA_M2S_CASE_LABEL": "patch_shape_b",
            }
            patch_b_run = run_case(
                gmx_bin,
                system_root,
                patch_b_work_dir,
                "exact_three_level",
                DEFAULT_DT_VALUES[0],
                args.total_time_ps,
                commands_log,
                extra_env=patch_b_env,
            )
        elif args.raw_sr_write_ordinal_trace and fixture_id == "dense_oligomer":
            patch_b_work_dir = system_root / "dt_0p0005" / "patch_shape_b"
            patch_b_env = {
                "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                "GMX_PCFF_RESPA_M2K_PATCH_MODE": "patch_shape_b",
                "GMX_PCFF_RESPA_M2U_TRACE_DIR": str(patch_b_work_dir),
                "GMX_PCFF_RESPA_M2U_CASE_LABEL": "patch_shape_b",
            }
            patch_b_run = run_case(
                gmx_bin,
                system_root,
                patch_b_work_dir,
                "exact_three_level",
                DEFAULT_DT_VALUES[0],
                args.total_time_ps,
                commands_log,
                extra_env=patch_b_env,
            )
        elif args.aligned_write_contract_trace and fixture_id == "dense_oligomer":
            patch_b_work_dir = system_root / "dt_0p0005" / "patch_shape_b"
            patch_b_env = {
                "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                "GMX_PCFF_RESPA_M2K_PATCH_MODE": "patch_shape_b",
                "GMX_PCFF_RESPA_M2V_TRACE_DIR": str(patch_b_work_dir),
                "GMX_PCFF_RESPA_M2V_CASE_LABEL": "patch_shape_b",
            }
            patch_b_run = run_case(
                gmx_bin,
                system_root,
                patch_b_work_dir,
                "exact_three_level",
                DEFAULT_DT_VALUES[0],
                args.total_time_ps,
                commands_log,
                extra_env=patch_b_env,
            )
        elif args.aligned_event_669_trace and fixture_id == "dense_oligomer":
            patch_b_work_dir = system_root / "dt_0p0005" / "patch_shape_b"
            patch_b_env = {
                "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                "GMX_PCFF_RESPA_M2K_PATCH_MODE": "patch_shape_b",
                "GMX_PCFF_RESPA_M2W_TRACE_DIR": str(patch_b_work_dir),
                "GMX_PCFF_RESPA_M2W_CASE_LABEL": "patch_shape_b",
            }
            patch_b_run = run_case(
                gmx_bin,
                system_root,
                patch_b_work_dir,
                "exact_three_level",
                DEFAULT_DT_VALUES[0],
                args.total_time_ps,
                commands_log,
                extra_env=patch_b_env,
            )
        elif args.event_669_geometry_producer_trace and fixture_id == "dense_oligomer":
            patch_b_work_dir = system_root / "dt_0p0005" / "patch_shape_b"
            patch_b_env = {
                "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                "GMX_PCFF_RESPA_M2K_PATCH_MODE": "patch_shape_b",
                "GMX_PCFF_RESPA_M2W_TRACE_DIR": str(patch_b_work_dir),
                "GMX_PCFF_RESPA_M2W_CASE_LABEL": "patch_shape_b",
                "GMX_PCFF_RESPA_M2X_TRACE_DIR": str(patch_b_work_dir),
                "GMX_PCFF_RESPA_M2X_CASE_LABEL": "patch_shape_b",
            }
            patch_b_run = run_case(
                gmx_bin,
                system_root,
                patch_b_work_dir,
                "exact_three_level",
                DEFAULT_DT_VALUES[0],
                args.total_time_ps,
                commands_log,
                extra_env=patch_b_env,
            )
        elif args.lj_sr_true_first_raw_trace and fixture_id == "dense_oligomer":
            patch_b_work_dir = system_root / "dt_0p0005" / "patch_shape_b"
            patch_b_env = {
                "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                "GMX_PCFF_RESPA_M2K_PATCH_MODE": "patch_shape_b",
                "GMX_PCFF_RESPA_M2Q_TRACE_DIR": str(patch_b_work_dir),
                "GMX_PCFF_RESPA_M2Q_CASE_LABEL": "patch_shape_b",
            }
            patch_b_run = run_case(
                gmx_bin,
                system_root,
                patch_b_work_dir,
                "exact_three_level",
                DEFAULT_DT_VALUES[0],
                args.total_time_ps,
                commands_log,
                extra_env=patch_b_env,
            )
        elif args.lj_sr_first_sink_trace and fixture_id == "dense_oligomer":
            patch_b_work_dir = system_root / "dt_0p0005" / "patch_shape_b"
            patch_b_env = {
                "GMX_DISABLE_MODULAR_SIMULATOR": "ON",
                "GMX_PCFF_RESPA_M2K_PATCH_MODE": "patch_shape_b",
                "GMX_PCFF_RESPA_M2P_TRACE_DIR": str(patch_b_work_dir),
                "GMX_PCFF_RESPA_M2P_CASE_LABEL": "patch_shape_b",
            }
            patch_b_run = run_case(
                gmx_bin,
                system_root,
                patch_b_work_dir,
                "exact_three_level",
                DEFAULT_DT_VALUES[0],
                args.total_time_ps,
                commands_log,
                extra_env=patch_b_env,
            )
        coarse_exact_trace_off = None
        if args.locked_scope_bookkeeping_residual_trace and fixture_id == "dense_oligomer":
            fixture_summary = summarize_dense_patch_b_bookkeeping_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                tuple(gro_meta["box_nm"]),
                coarse_plain,
                coarse_exact,
                patch_b_run,
                bookkeeping_probe_run,
            )
        elif args.post_final_ledger_mutation_trace and fixture_id == "dense_oligomer":
            fixture_summary = summarize_dense_patch_b_post_final_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                tuple(gro_meta["box_nm"]),
                coarse_plain,
                coarse_exact,
                patch_b_run,
            )
        elif args.reciprocal_internal_delta_trace and fixture_id == "dense_oligomer":
            fixture_summary = summarize_dense_patch_b_reciprocal_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                tuple(gro_meta["box_nm"]),
                coarse_plain,
                coarse_exact,
                patch_b_run,
            )
        elif args.lj_sr_first_amplification_trace and fixture_id == "dense_oligomer":
            fixture_summary = summarize_dense_patch_b_lj_sr_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                tuple(gro_meta["box_nm"]),
                coarse_plain,
                coarse_exact,
                patch_b_run,
            )
        elif args.raw_sr_formation_internal_trace and fixture_id == "dense_oligomer":
            fixture_summary = summarize_dense_patch_b_lj_sr_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                tuple(gro_meta["box_nm"]),
                coarse_plain,
                coarse_exact,
                patch_b_run,
            )
        elif args.raw_sr_write_ordinal_trace and fixture_id == "dense_oligomer":
            fixture_summary = summarize_dense_patch_b_lj_sr_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                tuple(gro_meta["box_nm"]),
                coarse_plain,
                coarse_exact,
                patch_b_run,
            )
        elif args.aligned_write_contract_trace and fixture_id == "dense_oligomer":
            fixture_summary = summarize_dense_patch_b_lj_sr_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                tuple(gro_meta["box_nm"]),
                coarse_plain,
                coarse_exact,
                patch_b_run,
            )
        elif args.aligned_event_669_trace and fixture_id == "dense_oligomer":
            fixture_summary = summarize_dense_patch_b_lj_sr_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                tuple(gro_meta["box_nm"]),
                coarse_plain,
                coarse_exact,
                patch_b_run,
            )
        elif args.event_669_geometry_producer_trace and fixture_id == "dense_oligomer":
            fixture_summary = summarize_dense_patch_b_lj_sr_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                tuple(gro_meta["box_nm"]),
                coarse_plain,
                coarse_exact,
                patch_b_run,
            )
        elif args.lj_sr_true_first_raw_trace and fixture_id == "dense_oligomer":
            fixture_summary = summarize_dense_patch_b_lj_sr_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                tuple(gro_meta["box_nm"]),
                coarse_plain,
                coarse_exact,
                patch_b_run,
            )
        elif args.lj_sr_first_sink_trace and fixture_id == "dense_oligomer":
            fixture_summary = summarize_dense_patch_b_lj_sr_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                tuple(gro_meta["box_nm"]),
                coarse_plain,
                coarse_exact,
                patch_b_run,
            )
        elif args.narrow_patch_proof and fixture_id == "dense_oligomer":
            fixture_summary = summarize_dense_patch_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                tuple(gro_meta["box_nm"]),
                coarse_plain,
                coarse_exact,
                patch_runs,
            )
        elif (
            args.upstream_ownership_handoff_trace
            or args.pair_rule_derivation_trace
            or args.downstream_misconsumption_trace
            or args.dispatch_minimal_fix_isolation
        ) and fixture_id == "dense_oligomer":
            fixture_summary = summarize_exact_only_trace_fixture(
                fixture_id,
                inner_terms_from_topology(topology_text),
                coarse_exact,
                comparison_notes=(
                    [
                        "This milestone traces only the dispatch-internal sub-decisions inside exact_excludedPairs_dispatch_contract.",
                        "No plain-Verlet or PME-side legacy side-reference trajectory comparison is needed; the reference contract is the same-run cleared exclusion bookkeeping path."
                    ]
                    if args.dispatch_minimal_fix_isolation
                    else None
                ),
            )
            if dispatch_probe_runs:
                fixture_summary["coarse"]["dispatch_probe_work_dirs"] = {
                    key: value["work_dir"] for key, value in dispatch_probe_runs.items()
                }
        else:
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
        if args.upstream_ownership_handoff_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_upstream_ownership_handoff_trace"] = dense_upstream_ownership_handoff_trace(
                fixture_summary
            )
        if args.pair_rule_derivation_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_pair_rule_derivation_trace"] = dense_pair_rule_derivation_trace(fixture_summary)
        if args.downstream_misconsumption_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_downstream_misconsumption_trace"] = dense_downstream_misconsumption_trace(
                fixture_summary
            )
        if args.dispatch_minimal_fix_isolation and fixture_id == "dense_oligomer":
            fixture_summary["dense_dispatch_minimal_fix_isolation"] = dense_dispatch_minimal_fix_isolation(
                fixture_summary
            )
        if args.narrow_patch_proof and fixture_id == "dense_oligomer":
            fixture_summary["dense_narrow_patch_proof"] = dense_narrow_patch_proof(fixture_summary)
        if args.locked_scope_bookkeeping_residual_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_patch_b_bookkeeping_residual_trace"] = dense_patch_b_bookkeeping_residual_trace(
                gmx_bin, fixture_summary, commands_log
            )
        if args.reciprocal_internal_delta_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_patch_b_reciprocal_internal_delta_trace"] = (
                dense_patch_b_reciprocal_internal_delta_trace(gmx_bin, fixture_summary, commands_log)
            )
        if args.lj_sr_first_amplification_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_patch_b_lj_sr_first_amplification_trace"] = (
                dense_patch_b_lj_sr_first_amplification_trace(gmx_bin, fixture_summary, commands_log)
            )
        if args.raw_sr_formation_internal_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_patch_b_raw_sr_formation_internal_trace"] = (
                dense_patch_b_raw_sr_formation_internal_trace(gmx_bin, fixture_summary, commands_log)
            )
        if args.raw_sr_write_ordinal_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_patch_b_raw_sr_write_ordinal_trace"] = (
                dense_patch_b_raw_sr_write_ordinal_trace(gmx_bin, fixture_summary, commands_log)
            )
        if args.aligned_write_contract_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_patch_b_aligned_write_contract_trace"] = (
                dense_patch_b_aligned_write_contract_trace(gmx_bin, fixture_summary, commands_log)
            )
        if args.aligned_event_669_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_patch_b_aligned_event_669_trace"] = (
                dense_patch_b_aligned_event_669_trace(gmx_bin, fixture_summary, commands_log)
            )
        if args.event_669_geometry_producer_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_patch_b_event_669_geometry_producer_trace"] = (
                dense_patch_b_event_669_geometry_producer_trace(gmx_bin, fixture_summary, commands_log)
            )
        if args.lj_sr_true_first_raw_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_patch_b_lj_sr_true_first_raw_trace"] = (
                dense_patch_b_lj_sr_true_first_raw_trace(gmx_bin, fixture_summary, commands_log)
            )
        if args.post_final_ledger_mutation_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_patch_b_post_final_ledger_trace"] = dense_patch_b_post_final_ledger_trace(
                gmx_bin, fixture_summary, commands_log
            )
        if args.lj_sr_first_sink_trace and fixture_id == "dense_oligomer":
            fixture_summary["dense_patch_b_lj_sr_first_sink_trace"] = dense_patch_b_lj_sr_first_sink_trace(
                gmx_bin, fixture_summary, commands_log
            )
            fixture_summary["dense_patch_b_potential_ledger_trace"] = dense_patch_b_potential_ledger_trace(
                gmx_bin, fixture_summary, commands_log
            )
        fixture_results.append(fixture_summary)
        write_json(system_root / "fixture_summary.json", fixture_summary)

    if args.event_669_geometry_producer_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_patch_b_event_669_geometry_producer_trace", {}).get("localization")
        )
        if dense_localization and dense_localization.get("final_verdict") == "PASS":
            verdict = "EVENT 669 GEOMETRY PRODUCER IDENTIFIED"
        elif dense_localization and dense_localization.get("final_verdict") == "PARTIAL":
            verdict = "EVENT 669 GEOMETRY PRODUCER TRACE STILL PARTIAL"
        else:
            verdict = "EVENT 669 GEOMETRY PRODUCER TRACE FAILED"
    elif args.aligned_event_669_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_patch_b_aligned_event_669_trace", {}).get("localization")
        )
        if dense_localization and dense_localization.get("final_verdict") == "PASS":
            verdict = "ALIGNED EVENT 669 ARITHMETIC SOURCE IDENTIFIED"
        elif dense_localization and dense_localization.get("final_verdict") == "PARTIAL":
            verdict = "ALIGNED EVENT 669 TRACE STILL PARTIAL"
        else:
            verdict = "ALIGNED EVENT 669 TRACE FAILED"
    elif args.aligned_write_contract_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_patch_b_aligned_write_contract_trace", {}).get("localization")
        )
        if dense_localization and dense_localization.get("final_verdict") == "PASS":
            verdict = "CROSS-SIDE ALIGNED WRITE CONTRACT READY"
        elif dense_localization and dense_localization.get("final_verdict") == "PARTIAL":
            verdict = "CROSS-SIDE ALIGNED WRITE CONTRACT STILL PARTIAL"
        else:
            verdict = "CROSS-SIDE ALIGNED WRITE CONTRACT FAILED"
    elif args.raw_sr_write_ordinal_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_patch_b_raw_sr_write_ordinal_trace", {}).get("localization")
        )
        if dense_localization and dense_localization.get("final_verdict") == "PASS":
            verdict = "RAW_SR_FORMATION WRITE-ORDINAL TRACE READY"
        elif dense_localization and dense_localization.get("final_verdict") == "PARTIAL":
            verdict = "RAW_SR_FORMATION WRITE-ORDINAL TRACE STILL PARTIAL"
        else:
            verdict = "RAW_SR_FORMATION WRITE-ORDINAL TRACE FAILED"
    elif args.raw_sr_formation_internal_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_patch_b_raw_sr_formation_internal_trace", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_raw_sr_internal_culprit"):
            verdict = "RAW_SR_FORMATION INTERNAL CULPRIT IDENTIFIED"
        elif dense_localization and dense_localization.get("first_internal_amplification_proof", {}).get("first_stage"):
            verdict = "RAW_SR_FORMATION INTERNAL TRACE STILL PARTIAL"
        else:
            verdict = "RAW_SR_FORMATION INTERNAL TRACE FAILED"
    elif args.lj_sr_first_amplification_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_patch_b_lj_sr_first_amplification_trace", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_lj_sr_amplification"):
            verdict = "FIRST LJ-SR AMPLIFICATION STAGE IDENTIFIED"
        elif dense_localization and dense_localization.get("first_amplification_proof", {}).get("first_stage"):
            verdict = "FIRST LJ-SR AMPLIFICATION TRACE STILL PARTIAL"
        else:
            verdict = "FIRST LJ-SR AMPLIFICATION TRACE FAILED"
    elif args.lj_sr_true_first_raw_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_patch_b_lj_sr_true_first_raw_trace", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_lj_sr_origin"):
            verdict = "TRUE FIRST LJ-SR RAW STAGE IDENTIFIED"
        elif dense_localization and dense_localization.get("first_nonzero_lj_sr_delta_proof", {}).get("first_stage"):
            verdict = "TRUE FIRST LJ-SR RAW TRACE STILL PARTIAL"
        else:
            verdict = "TRUE FIRST LJ-SR RAW TRACE FAILED"
    elif args.lj_sr_first_sink_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_patch_b_lj_sr_first_sink_trace", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_lj_sr_origin"):
            verdict = "LJ-SR FIRST ORIGIN IDENTIFIED"
        elif dense_localization and dense_localization.get("first_nonzero_lj_sr_delta_proof", {}).get("first_stage"):
            verdict = "LJ-SR ORIGIN NARROWED BUT STILL PARTIAL"
        else:
            verdict = "LJ-SR FIRST SINK TRACE STILL PARTIAL"
    elif args.post_final_ledger_mutation_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_patch_b_post_final_ledger_trace", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_post_final_divergence"):
            verdict = "POST-FINAL-LEDGER DIVERGENCE IDENTIFIED"
        else:
            verdict = "POST-FINAL-LEDGER TRACE STILL PARTIAL"
    elif args.reciprocal_internal_delta_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_patch_b_reciprocal_internal_delta_trace", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_reciprocal_internal_origin"):
            verdict = "RECIPROCAL INTERNAL DELTA ORIGIN IDENTIFIED"
        elif dense_localization and dense_localization.get("first_nonzero_delta_stage") is not None:
            verdict = "RECIPROCAL INTERNAL DELTA NARROWED BUT STILL PARTIAL"
        else:
            verdict = "RECIPROCAL INTERNAL DELTA TRACE STILL PARTIAL"
    elif args.locked_scope_bookkeeping_residual_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_patch_b_bookkeeping_residual_trace", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_patch_b_bookkeeping_trace"):
            verdict = "PATCH-B RESIDUAL BOOKKEEPING SOURCE IDENTIFIED"
        else:
            verdict = "PATCH-B RESIDUAL BOOKKEEPING TRACE STILL PARTIAL"
    elif args.narrow_patch_proof:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None if dense_fixture is None else dense_fixture.get("dense_narrow_patch_proof", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_narrow_patch_proof"):
            verdict = "NARROW PATCH PROOF COMPLETE"
        else:
            verdict = "NARROW PATCH PROOF STILL PARTIAL"
    elif args.dispatch_minimal_fix_isolation:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_dispatch_minimal_fix_isolation", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_minimal_fix_candidate"):
            verdict = "DISPATCH-INTERNAL MINIMAL FIX CANDIDATE ISOLATED"
        else:
            verdict = "DISPATCH-INTERNAL MINIMAL FIX ISOLATION STILL PARTIAL"
    elif args.downstream_misconsumption_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_downstream_misconsumption_trace", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_first_bad_site"):
            verdict = "FIRST DOWNSTREAM MIS-CONSUMPTION SITE IDENTIFIED"
        else:
            verdict = "DOWNSTREAM MIS-CONSUMPTION TRACE STILL PARTIAL"
    elif args.pair_rule_derivation_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_pair_rule_derivation_trace", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_generate_excl_not_earliest_bad_handoff"):
            verdict = "BASELINE-INTENDED GENERATE_EXCL RULE FIRE PROVEN; EARLIEST BAD HANDOFF NOT HERE"
        elif dense_localization and dense_localization.get("supports_generate_excl_earliest_bad_handoff"):
            verdict = "GENERATE_EXCL RULE DERIVATION IS THE EARLIEST BAD HANDOFF"
        else:
            verdict = "PAIR-RULE DERIVATION NARROWED BUT STILL PARTIAL"
    elif args.upstream_ownership_handoff_trace:
        dense_fixture = next((item for item in fixture_results if item["fixture_id"] == "dense_oligomer"), None)
        dense_localization = (
            None
            if dense_fixture is None
            else dense_fixture.get("dense_upstream_ownership_handoff_trace", {}).get("localization")
        )
        if dense_localization and dense_localization.get("supports_exact_earliest_handoff"):
            verdict = "EARLIEST OWNERSHIP/SPEC HANDOFF DEFECT IDENTIFIED"
        elif dense_localization:
            verdict = "OWNERSHIP/SPEC HANDOFF NARROWED BUT STILL PARTIAL"
        else:
            verdict = "UPSTREAM OWNERSHIP HANDOFF STILL UNRESOLVED"
    elif args.exact_pair_write_ownership_proof:
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
