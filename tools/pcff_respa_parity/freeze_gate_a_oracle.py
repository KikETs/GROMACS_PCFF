from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
FIXTURE_ROOT = REPO_ROOT / "tests" / "reference_results" / "m6_respa"
DEFAULT_GMX = BUILD_DIR / "bin" / "gmx"
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_a_cpu_oracle"
SYSTEMS = ("small_oligomer", "small_salt_polymer_box")
LEVEL_FACTORS = (1, 2, 4)
RESPA_ENERGY_INTERVAL = 4
FOURIER_SPACING_NM = 0.08

EVENT_LINE_RE = re.compile(r"^(\d+)\t([a-z_]+)\t(-?\d+)$")
ENERGY_FRAME_RE = re.compile(r"time:\s*([\-+0-9.eE]+)\s+step:\s*(\d+)")
ENERGY_TERM_RE = re.compile(r"^\s{2,}(.+?)\s+([\-+0-9.]+(?:e[\-+0-9]+)?)\s*$", re.IGNORECASE)
TRR_META_RE = re.compile(r"natoms=\s*(\d+)\s+step=\s*(\d+)\s+time=([\-+0-9.eE]+)")
TRR_VECTOR_RE = re.compile(
    r"^\s*[xv]\[\s*(\d+)\]=\{\s*([\-+0-9.eE]+),\s*([\-+0-9.eE]+),\s*([\-+0-9.eE]+)\}"
)
CLASS2_SUBTERM_TRACE_TERM_ORDER = (
    "bond_class2_main",
    "angle_class2_main",
    "angle_class2_bond_bond",
    "angle_class2_bond_angle_1",
    "angle_class2_bond_angle_2",
    "dihedral_class2_main",
    "dihedral_class2_middle_bond_torsion",
    "dihedral_class2_end_bond_torsion_1",
    "dihedral_class2_end_bond_torsion_2",
    "dihedral_class2_angle_torsion_1",
    "dihedral_class2_angle_torsion_2",
    "dihedral_class2_angle_angle_torsion",
    "dihedral_class2_bond_bond_13_torsion",
    "improper_class2_main",
    "improper_class2_angle_angle_1",
    "improper_class2_angle_angle_2",
    "improper_class2_angle_angle_3",
)
CPU_CORRECTION_TRACE_TERM_ORDER = (
    "coulomb_pairs_short_range",
    "coulomb_excluded_correction",
    "coulomb_self",
    "coulomb_short_range_total",
    "coulomb_reciprocal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a CPU-only Gate A oracle for standalone exact r-RESPA.")
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--build-target", default="gmx", help="CMake target to build.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks for mdrun.")
    parser.add_argument("--ntomp", type=int, default=1, help="OpenMP threads for mdrun.")
    parser.add_argument("--outer-steps", type=int, default=5, help="Number of exact r-RESPA outer steps.")
    parser.add_argument("--pair14-level", type=int, default=1, help="Exact r-RESPA pair14 level.")
    return parser.parse_args()


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> None:
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("w", encoding="utf-8") if stdout_path is not None else None
    stderr_handle = stderr_path.open("w", encoding="utf-8") if stderr_path is not None else None
    try:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    finally:
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()


def capture_output(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(command, cwd=cwd, env=env, check=True, text=True, capture_output=True)
    return completed.stdout


def write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def fixture_dir(system_id: str) -> Path:
    return FIXTURE_ROOT / system_id


def read_gro_atom_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Unexpected GRO file, missing atom-count line: {path}")
    return int(lines[1].strip())


def build_trace_atom_indices(atom_count: int) -> str:
    if atom_count <= 0:
        raise ValueError(f"Atom count must be positive, got {atom_count}")
    return ",".join(str(atom_index) for atom_index in range(atom_count))


def make_exact_respa_mdp(outer_steps: int, pair14_level: int) -> str:
    nsteps = outer_steps * RESPA_ENERGY_INTERVAL
    return (
        "title                   = gate a standalone exact respa oracle\n"
        "integrator              = md-vv\n"
        "dt                      = 0.0005\n"
        f"nsteps                  = {nsteps}\n"
        "constraints             = none\n"
        "cutoff-scheme           = Verlet\n"
        f"nstlist                 = {RESPA_ENERGY_INTERVAL}\n"
        "rlist                   = 0.99\n"
        "rvdw                    = 0.9\n"
        "rcoulomb                = 0.9\n"
        "vdwtype                 = Cut-off\n"
        "vdw-modifier            = none\n"
        "coulombtype             = PME\n"
        "coulomb-modifier        = none\n"
        "ewald-rtol              = 1e-6\n"
        "pme-order               = 4\n"
        f"fourierspacing          = {FOURIER_SPACING_NM}\n"
        "epsilon-r               = 1\n"
        "pbc                     = xyz\n"
        "tcoupl                  = no\n"
        "pcoupl                  = no\n"
        "comm-mode               = none\n"
        "verlet-buffer-tolerance = -1\n"
        "gen-vel                 = no\n"
        "exact-respa             = yes\n"
        "exact-respa-levels      = 3\n"
        "exact-respa-level2-factor = 2\n"
        "exact-respa-level3-factor = 4\n"
        "exact-respa-bond-level  = 1\n"
        "exact-respa-angle-level = 1\n"
        "exact-respa-dihedral-level = 1\n"
        "exact-respa-improper-level = 1\n"
        f"exact-respa-pair14-level = {pair14_level}\n"
        "exact-respa-pair-level  = 3\n"
        "exact-respa-kspace-level = 3\n"
        "exact-respa-inner-level = 1\n"
        "exact-respa-middle-level = 2\n"
        "exact-respa-outer-level = 3\n"
        "exact-respa-inner-off   = 0.30\n"
        "exact-respa-inner-on    = 0.45\n"
        "exact-respa-outer-on    = 0.60\n"
        "exact-respa-outer-off   = 0.80\n"
        f"nstcalcenergy           = {RESPA_ENERGY_INTERVAL}\n"
        f"nstenergy               = {RESPA_ENERGY_INTERVAL}\n"
        f"nstlog                  = {RESPA_ENERGY_INTERVAL}\n"
        f"nstxout                 = {RESPA_ENERGY_INTERVAL}\n"
        f"nstvout                 = {RESPA_ENERGY_INTERVAL}\n"
        "nstfout                 = 0\n"
        "nstxout-compressed      = 0\n"
    )


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.ntomp)
    env["GMX_DISABLE_MODULAR_SIMULATOR"] = "1"
    return env


def full_run_env(args: argparse.Namespace, system_root: Path, atom_count: int | None = None) -> dict[str, str]:
    env = base_env(args)
    all_steps = ",".join(str(step) for step in range(args.outer_steps * LEVEL_FACTORS[-1] + 1))
    env["GMX_EXACT_RESPA_RUNTIME_EVENT_TRACE_FILE"] = str(system_root / "full" / "event_trace.tsv")
    env["GMX_PCFF_RESPA_MERGE_TRACE_DIR"] = str(system_root / "full" / "merge_trace")
    env["GMX_EXACT_RESPA_TOTAL_FORCE_DUMP_FILE"] = str(system_root / "full" / "total_force.tsv")
    if atom_count is not None:
        env["GMX_PCFF_RESPA_TRACE_ATOMS"] = build_trace_atom_indices(atom_count)
    env["GMX_PCFF_RESPA_TRACE_FORCE_COMPONENTS"] = "1"
    env["GMX_PCFF_RESPA_TRACE_FORCE_COMPONENTS_STEPS"] = all_steps
    env["GMX_PCFF_RESPA_TRACE_REALSPACE_FORCE_SUBCOMPONENTS"] = "1"
    env["GMX_PCFF_RESPA_TRACE_REALSPACE_FORCE_SUBCOMPONENTS_STEPS"] = "0,2"
    env["GMX_PCFF_RESPA_TRACE_STEP1_SUBSET01_FORCEGROUP_AUDIT"] = "1"
    env["GMX_PCFF_RESPA_TRACE_STEP1_SUBSET01_FORCEGROUP_AUDIT_STEPS"] = "2"
    env["GMX_PCFF_RESPA_TRACE_CLASS2_SUBTERM_ENERGIES"] = "1"
    env["GMX_PCFF_RESPA_TRACE_CLASS2_SUBTERM_ENERGIES_STEPS"] = all_steps
    env["GMX_PCFF_RESPA_TRACE_CPU_CORRECTION_ENERGIES"] = "1"
    env["GMX_PCFF_RESPA_TRACE_CPU_CORRECTION_ENERGIES_STEPS"] = all_steps
    env["GMX_PCFF_RESPA_TRACE_MULTI_STEP_COULOMB_STEPS"] = all_steps
    env["GMX_PCFF_RESPA_M2P_TRACE_DIR"] = str(system_root / "full" / "m2p_trace")
    env["GMX_PCFF_RESPA_M2P_CASE_LABEL"] = system_root.name
    return env


def mdrun_args(args: argparse.Namespace, deffnm: Path) -> list[str]:
    return [
        "-s",
        str(deffnm.with_suffix(".tpr")),
        "-deffnm",
        str(deffnm),
        "-ntmpi",
        str(args.ntmpi),
        "-ntomp",
        str(args.ntomp),
        "-dlb",
        "no",
        "-nb",
        "cpu",
        "-pme",
        "cpu",
        "-bonded",
        "cpu",
        "-update",
        "cpu",
        "-pin",
        "off",
        "-reprod",
    ]


def parse_event_trace(path: Path) -> list[dict[str, int | str]]:
    events: list[dict[str, int | str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = EVENT_LINE_RE.match(line)
        if match is None:
            raise ValueError(f"Unexpected event trace line in {path}: {line}")
        events.append(
            {
                "base_step": int(match.group(1)),
                "event": match.group(2),
                "level": int(match.group(3)),
            }
        )
    return events


def parse_class2_subterm_energy_trace(path: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    entries_by_step_level: dict[tuple[int, int], dict[str, object]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 7:
            raise ValueError(f"Unexpected class2 subterm trace line in {path}: {raw_line}")
        step = int(parts[0])
        level = int(parts[1])
        row = {
            "step": step,
            "level": level,
            "actual_backend": parts[2],
            "term": parts[3],
            "energy_kj_mol": float(parts[4]),
            "interaction_count": int(parts[5]),
            "diagnostic_origin": parts[6],
        }
        rows.append(row)
        entry = entries_by_step_level.setdefault(
            (step, level),
            {
                "step": step,
                "level": level,
                "actual_backend": parts[2],
                "diagnostic_origin": parts[6],
                "terms_kj_mol": {},
                "interaction_counts": {},
            },
        )
        entry["terms_kj_mol"][parts[3]] = float(parts[4])
        entry["interaction_counts"][parts[3]] = int(parts[5])

    entries = [entries_by_step_level[key] for key in sorted(entries_by_step_level.keys())]
    return {
        "schema_version": 1,
        "term_order": list(CLASS2_SUBTERM_TRACE_TERM_ORDER),
        "rows": rows,
        "entries": entries,
    }


def parse_cpu_correction_energy_trace(path: Path) -> dict[str, object]:
    rows = []
    entries_by_step_level: dict[tuple[int, int], dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if len(parts) != 7:
            raise ValueError(f"Unexpected cpu correction trace line in {path}: {stripped}")
        row = {
            "step": int(parts[0]),
            "level": int(parts[1]),
            "actual_backend": parts[2],
            "term": parts[3],
            "energy_kj_mol": float(parts[4]),
            "interaction_count": int(parts[5]),
            "diagnostic_origin": parts[6],
        }
        rows.append(row)
        entry = entries_by_step_level.setdefault(
            (row["step"], row["level"]),
            {
                "step": row["step"],
                "level": row["level"],
                "actual_backend": row["actual_backend"],
                "diagnostic_origin": row["diagnostic_origin"],
                "terms_kj_mol": {},
                "interaction_counts": {},
            },
        )
        entry["terms_kj_mol"][row["term"]] = row["energy_kj_mol"]
        entry["interaction_counts"][row["term"]] = row["interaction_count"]

    entries = [entries_by_step_level[key] for key in sorted(entries_by_step_level.keys())]
    return {
        "schema_version": 1,
        "term_order": list(CPU_CORRECTION_TRACE_TERM_ORDER),
        "rows": rows,
        "entries": entries,
    }


def append_reference_events(level: int, output: list[tuple[str, int]]) -> None:
    loops = 1 if level + 1 == len(LEVEL_FACTORS) else LEVEL_FACTORS[level + 1] // LEVEL_FACTORS[level]
    for _ in range(loops):
        output.append(("kick", level))
        if level == 0:
            output.append(("drift", 0))
        else:
            append_reference_events(level - 1, output)
        output.append(("force", level))
        output.append(("final_kick", level))


def reference_base_step_traces(outer_cycles: int) -> list[dict[str, list[int]]]:
    placeholder_events: list[tuple[str, int]] = []
    for _ in range(outer_cycles):
        append_reference_events(len(LEVEL_FACTORS) - 1, placeholder_events)

    traces: list[dict[str, list[int]]] = []
    index = 0
    while index < len(placeholder_events):
        trace = {"initial_kick_levels": [], "refreshed_force_levels": [], "final_kick_levels": []}
        while index < len(placeholder_events) and placeholder_events[index][0] == "kick":
            trace["initial_kick_levels"].append(placeholder_events[index][1])
            index += 1

        if index >= len(placeholder_events) or placeholder_events[index][0] != "drift":
            raise ValueError("Reference trace generation failed before drift")
        index += 1

        while index < len(placeholder_events) and placeholder_events[index][0] == "force":
            trace["refreshed_force_levels"].append(placeholder_events[index][1])
            index += 1
            if index >= len(placeholder_events) or placeholder_events[index][0] != "final_kick":
                raise ValueError("Reference trace generation failed before final kick")
            trace["final_kick_levels"].append(placeholder_events[index][1])
            index += 1
        traces.append(trace)
    return traces


def highest_active_level(base_step: int) -> int:
    highest = 0
    for level, factor in enumerate(LEVEL_FACTORS[1:], start=1):
        if base_step % factor == 0:
            highest = level
        else:
            break
    return highest


def reference_event_trace(outer_steps: int) -> list[dict[str, int | str]]:
    executed_base_steps = outer_steps * LEVEL_FACTORS[-1] + 1
    events: list[dict[str, int | str]] = []
    traces = [
        {
            "initial_kick_levels": [0],
            "refreshed_force_levels": [0],
            "final_kick_levels": [0],
        }
    ]
    for base_step in range(1, executed_base_steps):
        highest_initial = highest_active_level(base_step)
        highest_final = highest_active_level(base_step + 1)
        traces.append(
            {
                "initial_kick_levels": list(range(highest_initial, -1, -1)),
                "refreshed_force_levels": list(range(0, highest_final + 1)),
                "final_kick_levels": list(range(0, highest_final + 1)),
            }
        )

    for base_step, trace in enumerate(traces):
        for level in trace["initial_kick_levels"]:
            events.append({"base_step": base_step, "event": "kick", "level": level})
        events.append({"base_step": base_step, "event": "drift", "level": 0})
        for level in trace["refreshed_force_levels"]:
            events.append({"base_step": base_step, "event": "force", "level": level})
        for level in trace["final_kick_levels"]:
            events.append({"base_step": base_step, "event": "final_kick", "level": level})
    return events


def parse_energy_dump(dump_text: str) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in dump_text.splitlines():
        frame_match = ENERGY_FRAME_RE.search(line)
        if frame_match is not None:
            if current is not None:
                frames.append(current)
            current = {"time_ps": float(frame_match.group(1)), "step": int(frame_match.group(2)), "terms": {}}
            continue

        if current is None or ":" in line:
            continue

        term_match = ENERGY_TERM_RE.match(line)
        if term_match is None:
            continue
        name = term_match.group(1).strip()
        value = float(term_match.group(2))
        current["terms"][name] = value

    if current is not None:
        frames.append(current)
    if not frames:
        raise ValueError("No energy frames parsed from gmx dump output")
    return frames


def parse_trr_dump(dump_text: str) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    section: str | None = None
    for line in dump_text.splitlines():
        if " frame " in line and line.rstrip().endswith(":"):
            if current is not None:
                frames.append(current)
            current = {"coordinates": [], "velocities": []}
            section = None
            continue

        if current is None:
            continue

        meta_match = TRR_META_RE.search(line)
        if meta_match is not None:
            current["natoms"] = int(meta_match.group(1))
            current["step"] = int(meta_match.group(2))
            current["time_ps"] = float(meta_match.group(3))
            continue

        stripped = line.strip()
        if stripped.startswith("x ("):
            section = "coordinates"
            continue
        if stripped.startswith("v ("):
            section = "velocities"
            continue
        if stripped.startswith("box (") or stripped.startswith("f ("):
            section = None
            continue

        vector_match = TRR_VECTOR_RE.match(line)
        if section is not None and vector_match is not None:
            current[section].append(
                [float(vector_match.group(2)), float(vector_match.group(3)), float(vector_match.group(4))]
            )

    if current is not None:
        frames.append(current)
    if not frames:
        raise ValueError("No trajectory frames parsed from gmx dump output")
    return frames


def parse_total_force_dump(path: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    grouped: dict[tuple[int, int], dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 7:
            step_str, time_str, highest_level_str, atom_str, fx_str, fy_str, fz_str = parts
            local_atom = int(atom_str)
            atom = local_atom
        elif len(parts) == 8:
            step_str, time_str, highest_level_str, local_atom_str, atom_str, fx_str, fy_str, fz_str = parts
            local_atom = int(local_atom_str)
            atom = int(atom_str)
        else:
            raise ValueError(f"Unexpected force dump line in {path}: {line}")
        step = int(step_str)
        highest_level = int(highest_level_str)
        fx = float(fx_str)
        fy = float(fy_str)
        fz = float(fz_str)
        entries.append(
            {
                "step": step,
                "time_ps": float(time_str),
                "highest_active_level": highest_level,
                "local_atom": local_atom,
                "atom": atom,
                "force": [fx, fy, fz],
            }
        )

        key = (step, highest_level)
        bucket = grouped.setdefault(
            key,
            {
                "step": step,
                "time_ps": float(time_str),
                "highest_active_level": highest_level,
                "atom_count": 0,
                "vector_sum": [0.0, 0.0, 0.0],
            },
        )
        bucket["atom_count"] += 1
        bucket["vector_sum"][0] += fx
        bucket["vector_sum"][1] += fy
        bucket["vector_sum"][2] += fz

    return {
        "schema_version": 1,
        "entries": entries,
        "per_step_totals": list(grouped.values()),
    }


def parse_merge_trace_dir(path: Path) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    for trace_path in sorted(path.glob("*.tsv")):
        header = ""
        vector_sum = [0.0, 0.0, 0.0]
        atom_count = 0
        with trace_path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    header = stripped[1:].strip()
                    continue
                parts = stripped.split("\t")
                if len(parts) != 4:
                    raise ValueError(f"Unexpected merge trace line in {trace_path}: {stripped}")
                vector_sum[0] += float(parts[1])
                vector_sum[1] += float(parts[2])
                vector_sum[2] += float(parts[3])
                atom_count += 1

        step_match = re.search(r"step(\d+)", trace_path.name)
        level_match = re.search(r"level(\d+)", trace_path.name)
        summaries.append(
            {
                "relative_path": str(trace_path.relative_to(path.parent)),
                "step": int(step_match.group(1)) if step_match is not None else None,
                "level": int(level_match.group(1)) if level_match is not None else None,
                "header": header,
                "atom_count": atom_count,
                "vector_sum": vector_sum,
            }
        )
    return {"schema_version": 1, "entries": summaries}


def energy_summary(system_id: str, outer_steps: int, pair14_level: int, frames: list[dict[str, object]]) -> dict[str, object]:
    def term(frame: dict[str, object], name: str) -> float:
        return float(frame["terms"][name])

    step0 = frames[0]
    final = frames[-1]
    total_series = [term(frame, "Total Energy") for frame in frames]
    summary = {
        "schema_version": 1,
        "system_id": system_id,
        "mode": "standalone_exact_respa",
        "schedule": {
            "outer_steps": outer_steps,
            "pair14_level": pair14_level,
            "level_step_factors": list(LEVEL_FACTORS),
        },
        "step0_terms_kj_mol": step0["terms"],
        "final_terms_kj_mol": final["terms"],
        "derived_terms_step0_kj_mol": {
            "bonded_total": term(step0, "Class2 Bond") + term(step0, "Class2 Angle") + term(step0, "Class2 Dih."),
            "vdw_total": term(step0, "LJ-14") + term(step0, "LJ (SR)"),
            "electro_total": term(step0, "Coulomb-14") + term(step0, "Coulomb (SR)") + term(step0, "Coul. recip."),
        },
        "total_energy_span_kj_mol": max(total_series) - min(total_series),
        "total_energy_drift_abs_kj_mol": abs(total_series[-1] - total_series[0]),
        "frames": frames,
    }
    return summary


def restart_summary(
    system_id: str,
    split_outer_steps: int,
    full_energy_frames: list[dict[str, object]],
    split_energy_frames: list[dict[str, object]],
    full_trr_frames: list[dict[str, object]],
    split_trr_frames: list[dict[str, object]],
) -> dict[str, object]:
    full_final_terms = full_energy_frames[-1]["terms"]
    split_final_terms = split_energy_frames[-1]["terms"]
    full_final_frame = full_trr_frames[-1]
    split_final_frame = split_trr_frames[-1]

    max_coordinate_delta = 0.0
    max_velocity_delta = 0.0
    for full_coord, split_coord in zip(full_final_frame["coordinates"], split_final_frame["coordinates"]):
        for full_value, split_value in zip(full_coord, split_coord):
            max_coordinate_delta = max(max_coordinate_delta, abs(full_value - split_value))
    for full_vel, split_vel in zip(full_final_frame["velocities"], split_final_frame["velocities"]):
        for full_value, split_value in zip(full_vel, split_vel):
            max_velocity_delta = max(max_velocity_delta, abs(full_value - split_value))

    return {
        "schema_version": 1,
        "system_id": system_id,
        "mode": "standalone_exact_respa",
        "split_outer_steps": split_outer_steps,
        "final_step": full_final_frame["step"],
        "final_time_ps": full_final_frame["time_ps"],
        "full_final_potential_kj_mol": full_final_terms["Potential"],
        "restart_final_potential_kj_mol": split_final_terms["Potential"],
        "full_final_total_kj_mol": full_final_terms["Total Energy"],
        "restart_final_total_kj_mol": split_final_terms["Total Energy"],
        "potential_abs_delta_kj_mol": abs(full_final_terms["Potential"] - split_final_terms["Potential"]),
        "total_abs_delta_kj_mol": abs(full_final_terms["Total Energy"] - split_final_terms["Total Energy"]),
        "max_coordinate_abs_delta_nm": max_coordinate_delta,
        "max_velocity_abs_delta_nm_ps": max_velocity_delta,
    }


def ensure_gate_a_invariants(
    actual_events: list[dict[str, int | str]],
    expected_events: list[dict[str, int | str]],
    energy_frames: list[dict[str, object]],
    total_force_summary: dict[str, object],
    per_level_force_totals: dict[str, object],
    restart_summary_data: dict[str, object],
) -> None:
    if actual_events != expected_events:
        raise AssertionError("Actual exact r-RESPA runtime events do not match the reference schedule")

    required_terms = {
        "Class2 Bond",
        "Class2 Angle",
        "Class2 Dih.",
        "LJ-14",
        "Coulomb-14",
        "LJ (SR)",
        "Coulomb (SR)",
        "Coul. recip.",
        "Potential",
        "Total Energy",
        "Pressure",
        "Vir-XX",
        "Vir-YY",
        "Vir-ZZ",
        "Pres-XX",
        "Pres-YY",
        "Pres-ZZ",
    }
    step0_terms = set(energy_frames[0]["terms"].keys())
    missing_terms = sorted(required_terms - step0_terms)
    if missing_terms:
        raise AssertionError(f"Missing required exact r-RESPA energy terms: {missing_terms}")

    if not total_force_summary["entries"]:
        raise AssertionError("Total-force dump is empty")
    if not per_level_force_totals["entries"]:
        raise AssertionError("Per-level merge trace totals are empty")

    if restart_summary_data["potential_abs_delta_kj_mol"] > 1e-6:
        raise AssertionError("Restart potential-energy continuity exceeded 1e-6 kJ/mol")
    if restart_summary_data["total_abs_delta_kj_mol"] > 1e-6:
        raise AssertionError("Restart total-energy continuity exceeded 1e-6 kJ/mol")
    if restart_summary_data["max_coordinate_abs_delta_nm"] > 1e-7:
        raise AssertionError("Restart coordinate continuity exceeded 1e-7 nm")
    if restart_summary_data["max_velocity_abs_delta_nm_ps"] > 1e-7:
        raise AssertionError("Restart velocity continuity exceeded 1e-7 nm/ps")


def command_record(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    env_overrides: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, object]:
    return {
        "name": name,
        "cwd": str(cwd),
        "argv": command,
        "env_overrides": env_overrides,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def env_delta(env: dict[str, str], baseline: dict[str, str]) -> dict[str, str]:
    keys = {
        "OMP_NUM_THREADS",
        "GMX_DISABLE_MODULAR_SIMULATOR",
        "GMX_EXACT_RESPA_RUNTIME_EVENT_TRACE_FILE",
        "GMX_EXACT_RESPA_STATE_TRACE_FILE",
        "GMX_EXACT_RESPA_STATE_TRACE_ATOMS",
        "GMX_EXACT_RESPA_STATE_TRACE_MAX_BASE_STEP",
        "GMX_PCFF_RESPA_MERGE_TRACE_DIR",
        "GMX_EXACT_RESPA_TOTAL_FORCE_DUMP_FILE",
        "GMX_PCFF_RESPA_TRACE_FORCE_COMPONENTS",
        "GMX_PCFF_RESPA_TRACE_FORCE_COMPONENTS_STEPS",
        "GMX_PCFF_RESPA_TRACE_REALSPACE_FORCE_SUBCOMPONENTS",
        "GMX_PCFF_RESPA_TRACE_REALSPACE_FORCE_SUBCOMPONENTS_STEPS",
        "GMX_PCFF_RESPA_TRACE_STEP1_SUBSET01_FORCEGROUP_AUDIT",
        "GMX_PCFF_RESPA_TRACE_STEP1_SUBSET01_FORCEGROUP_AUDIT_STEPS",
        "GMX_PCFF_RESPA_TRACE_CLASS2_SUBTERM_ENERGIES",
        "GMX_PCFF_RESPA_TRACE_CLASS2_SUBTERM_ENERGIES_STEPS",
        "GMX_PCFF_RESPA_TRACE_CPU_CORRECTION_ENERGIES",
        "GMX_PCFF_RESPA_TRACE_CPU_CORRECTION_ENERGIES_STEPS",
        "GMX_PCFF_RESPA_TRACE_MULTI_STEP_COULOMB_STEPS",
        "GMX_PCFF_RESPA_M2P_TRACE_DIR",
        "GMX_PCFF_RESPA_M2P_CASE_LABEL",
        "GMX_PCFF_RESPA_EXACT_GPU_BONDED_SEQUENTIAL_FTYPES",
    }
    return {key: env[key] for key in sorted(keys) if env.get(key) != baseline.get(key)}


def write_commands_script(path: Path, records: list[dict[str, object]]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for record in records:
        lines.append(f"# {record['name']}")
        env_prefix = "env"
        for key, value in record["env_overrides"].items():
            env_prefix += f" {shlex.quote(f'{key}={value}')}"
        command = " ".join(shlex.quote(arg) for arg in record["argv"])
        lines.append(
            f"(cd {shlex.quote(record['cwd'])} && {env_prefix} {command} > {shlex.quote(record['stdout'])} 2> {shlex.quote(record['stderr'])})"
        )
        lines.append("")
    write_text(path, "\n".join(lines))
    path.chmod(0o755)


def collect_system_artifacts(args: argparse.Namespace, gmx: Path, out_root: Path, system_id: str) -> dict[str, object]:
    system_root = out_root / system_id
    inputs_dir = system_root / "inputs"
    full_dir = system_root / "full"
    restart_full_dir = system_root / "restart_full"
    restart_split_dir = system_root / "restart_split"
    logs_dir = system_root / "logs"
    summaries_dir = system_root / "summaries"
    for directory in (inputs_dir, full_dir, restart_full_dir, restart_split_dir, logs_dir, summaries_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mdp_path = inputs_dir / "exact_respa.mdp"
    write_text(mdp_path, make_exact_respa_mdp(args.outer_steps, args.pair14_level))

    fixture = fixture_dir(system_id)
    commands: list[dict[str, object]] = []
    base_environment = base_env(args)
    full_environment = full_run_env(args, system_root, atom_count=read_gro_atom_count(fixture / "initial_nve.gro"))
    full_env_delta = env_delta(full_environment, os.environ)
    base_env_delta = env_delta(base_environment, os.environ)

    full_deffnm = full_dir / "exact_full"
    full_grompp = [
        str(gmx),
        "grompp",
        "-f",
        str(mdp_path),
        "-c",
        str(fixture / "initial_nve.gro"),
        "-p",
        str(fixture / "topol.top"),
        "-o",
        str(full_deffnm.with_suffix(".tpr")),
        "-po",
        str(inputs_dir / "exact_respa_full_mdout.mdp"),
        "-maxwarn",
        "1",
    ]
    full_grompp_stdout = logs_dir / "grompp_full.stdout"
    full_grompp_stderr = logs_dir / "grompp_full.stderr"
    run_command(full_grompp, cwd=REPO_ROOT, env=base_environment, stdout_path=full_grompp_stdout, stderr_path=full_grompp_stderr)
    commands.append(
        command_record(
            "grompp_full",
            full_grompp,
            cwd=REPO_ROOT,
            env_overrides=base_env_delta,
            stdout_path=full_grompp_stdout,
            stderr_path=full_grompp_stderr,
        )
    )

    full_mdrun = [str(gmx), "mdrun", *mdrun_args(args, full_deffnm)]
    full_mdrun_stdout = logs_dir / "mdrun_full.stdout"
    full_mdrun_stderr = logs_dir / "mdrun_full.stderr"
    run_command(full_mdrun, cwd=REPO_ROOT, env=full_environment, stdout_path=full_mdrun_stdout, stderr_path=full_mdrun_stderr)
    commands.append(
        command_record(
            "mdrun_full",
            full_mdrun,
            cwd=REPO_ROOT,
            env_overrides=full_env_delta,
            stdout_path=full_mdrun_stdout,
            stderr_path=full_mdrun_stderr,
        )
    )

    split_outer_steps = max(1, args.outer_steps // 2)
    split_steps = split_outer_steps * RESPA_ENERGY_INTERVAL

    restart_full_deffnm = restart_full_dir / "exact_full"
    restart_full_grompp = [
        str(gmx),
        "grompp",
        "-f",
        str(mdp_path),
        "-c",
        str(fixture / "initial_nve.gro"),
        "-p",
        str(fixture / "topol.top"),
        "-o",
        str(restart_full_deffnm.with_suffix(".tpr")),
        "-po",
        str(inputs_dir / "exact_respa_restart_full_mdout.mdp"),
        "-maxwarn",
        "1",
    ]
    restart_full_grompp_stdout = logs_dir / "grompp_restart_full.stdout"
    restart_full_grompp_stderr = logs_dir / "grompp_restart_full.stderr"
    run_command(
        restart_full_grompp,
        cwd=REPO_ROOT,
        env=base_environment,
        stdout_path=restart_full_grompp_stdout,
        stderr_path=restart_full_grompp_stderr,
    )
    commands.append(
        command_record(
            "grompp_restart_full",
            restart_full_grompp,
            cwd=REPO_ROOT,
            env_overrides=base_env_delta,
            stdout_path=restart_full_grompp_stdout,
            stderr_path=restart_full_grompp_stderr,
        )
    )

    restart_full_mdrun = [str(gmx), "mdrun", *mdrun_args(args, restart_full_deffnm)]
    restart_full_mdrun_stdout = logs_dir / "mdrun_restart_full.stdout"
    restart_full_mdrun_stderr = logs_dir / "mdrun_restart_full.stderr"
    run_command(
        restart_full_mdrun,
        cwd=REPO_ROOT,
        env=base_environment,
        stdout_path=restart_full_mdrun_stdout,
        stderr_path=restart_full_mdrun_stderr,
    )
    commands.append(
        command_record(
            "mdrun_restart_full",
            restart_full_mdrun,
            cwd=REPO_ROOT,
            env_overrides=base_env_delta,
            stdout_path=restart_full_mdrun_stdout,
            stderr_path=restart_full_mdrun_stderr,
        )
    )

    restart_split_deffnm = restart_split_dir / "exact_split"
    restart_split_grompp = [
        str(gmx),
        "grompp",
        "-f",
        str(mdp_path),
        "-c",
        str(fixture / "initial_nve.gro"),
        "-p",
        str(fixture / "topol.top"),
        "-o",
        str(restart_split_deffnm.with_suffix(".tpr")),
        "-po",
        str(inputs_dir / "exact_respa_restart_split_mdout.mdp"),
        "-maxwarn",
        "1",
    ]
    restart_split_grompp_stdout = logs_dir / "grompp_restart_split.stdout"
    restart_split_grompp_stderr = logs_dir / "grompp_restart_split.stderr"
    run_command(
        restart_split_grompp,
        cwd=REPO_ROOT,
        env=base_environment,
        stdout_path=restart_split_grompp_stdout,
        stderr_path=restart_split_grompp_stderr,
    )
    commands.append(
        command_record(
            "grompp_restart_split",
            restart_split_grompp,
            cwd=REPO_ROOT,
            env_overrides=base_env_delta,
            stdout_path=restart_split_grompp_stdout,
            stderr_path=restart_split_grompp_stderr,
        )
    )

    restart_split_first_mdrun = [str(gmx), "mdrun", *mdrun_args(args, restart_split_deffnm), "-nsteps", str(split_steps)]
    restart_split_first_stdout = logs_dir / "mdrun_restart_split_first.stdout"
    restart_split_first_stderr = logs_dir / "mdrun_restart_split_first.stderr"
    run_command(
        restart_split_first_mdrun,
        cwd=REPO_ROOT,
        env=base_environment,
        stdout_path=restart_split_first_stdout,
        stderr_path=restart_split_first_stderr,
    )
    commands.append(
        command_record(
            "mdrun_restart_split_first",
            restart_split_first_mdrun,
            cwd=REPO_ROOT,
            env_overrides=base_env_delta,
            stdout_path=restart_split_first_stdout,
            stderr_path=restart_split_first_stderr,
        )
    )

    restart_checkpoint = restart_split_deffnm.with_suffix(".cpt")
    restart_split_second_mdrun = [str(gmx), "mdrun", *mdrun_args(args, restart_split_deffnm), "-cpi", str(restart_checkpoint)]
    restart_split_second_stdout = logs_dir / "mdrun_restart_split_second.stdout"
    restart_split_second_stderr = logs_dir / "mdrun_restart_split_second.stderr"
    run_command(
        restart_split_second_mdrun,
        cwd=REPO_ROOT,
        env=base_environment,
        stdout_path=restart_split_second_stdout,
        stderr_path=restart_split_second_stderr,
    )
    commands.append(
        command_record(
            "mdrun_restart_split_second",
            restart_split_second_mdrun,
            cwd=REPO_ROOT,
            env_overrides=base_env_delta,
            stdout_path=restart_split_second_stdout,
            stderr_path=restart_split_second_stderr,
        )
    )

    full_energy_dump = capture_output([str(gmx), "dump", "-e", str(full_deffnm.with_suffix(".edr"))], cwd=REPO_ROOT)
    full_trr_dump = capture_output([str(gmx), "dump", "-f", str(full_deffnm.with_suffix(".trr"))], cwd=REPO_ROOT)
    restart_full_energy_dump = capture_output(
        [str(gmx), "dump", "-e", str(restart_full_deffnm.with_suffix(".edr"))], cwd=REPO_ROOT
    )
    restart_full_trr_dump = capture_output(
        [str(gmx), "dump", "-f", str(restart_full_deffnm.with_suffix(".trr"))], cwd=REPO_ROOT
    )
    restart_split_energy_dump = capture_output(
        [str(gmx), "dump", "-e", str(restart_split_deffnm.with_suffix(".edr"))], cwd=REPO_ROOT
    )
    restart_split_trr_dump = capture_output(
        [str(gmx), "dump", "-f", str(restart_split_deffnm.with_suffix(".trr"))], cwd=REPO_ROOT
    )

    write_text(summaries_dir / "full_energy_dump.txt", full_energy_dump)
    write_text(summaries_dir / "full_trr_dump.txt", full_trr_dump)
    write_text(summaries_dir / "restart_full_energy_dump.txt", restart_full_energy_dump)
    write_text(summaries_dir / "restart_full_trr_dump.txt", restart_full_trr_dump)
    write_text(summaries_dir / "restart_split_energy_dump.txt", restart_split_energy_dump)
    write_text(summaries_dir / "restart_split_trr_dump.txt", restart_split_trr_dump)

    actual_events = parse_event_trace(full_dir / "event_trace.tsv")
    expected_events = reference_event_trace(args.outer_steps)
    full_energy_frames = parse_energy_dump(full_energy_dump)
    restart_full_energy_frames = parse_energy_dump(restart_full_energy_dump)
    restart_split_energy_frames = parse_energy_dump(restart_split_energy_dump)
    restart_full_trr_frames = parse_trr_dump(restart_full_trr_dump)
    restart_split_trr_frames = parse_trr_dump(restart_split_trr_dump)
    total_force_summary = parse_total_force_dump(full_dir / "total_force.tsv")
    per_level_force_totals = parse_merge_trace_dir(full_dir / "merge_trace")
    class2_subterm_energy_trace = parse_class2_subterm_energy_trace(
        full_dir / "m2p_trace" / "class2_subterm_energy_trace.tsv"
    )
    cpu_correction_energy_trace = parse_cpu_correction_energy_trace(
        full_dir / "m2p_trace" / "cpu_correction_energy_trace.tsv"
    )
    restart_summary_data = restart_summary(
        system_id,
        split_outer_steps,
        restart_full_energy_frames,
        restart_split_energy_frames,
        restart_full_trr_frames,
        restart_split_trr_frames,
    )

    ensure_gate_a_invariants(
        actual_events,
        expected_events,
        full_energy_frames,
        total_force_summary,
        per_level_force_totals,
        restart_summary_data,
    )

    energy_summary_data = energy_summary(system_id, args.outer_steps, args.pair14_level, full_energy_frames)
    event_trace_json = {
        "schema_version": 1,
        "system_id": system_id,
        "mode": "standalone_exact_respa",
        "schedule": {
            "outer_steps": args.outer_steps,
            "level_step_factors": list(LEVEL_FACTORS),
            "pair14_level": args.pair14_level,
        },
        "notes": [
            "The direct CLI standalone exact-r-RESPA path starts from a bootstrap base step that applies only the level-0 initial kick before the first drift.",
            "Subsequent base steps follow the standalone exact-r-RESPA schedule frozen here for later GPU-gate comparisons.",
        ],
        "actual_event_trace": actual_events,
        "reference_event_trace": expected_events,
    }

    write_text(summaries_dir / "event_trace.json", json.dumps(event_trace_json, indent=2, sort_keys=True) + "\n")
    write_text(summaries_dir / "energy_terms.json", json.dumps(energy_summary_data, indent=2, sort_keys=True) + "\n")
    write_text(
        summaries_dir / "total_force_summary.json",
        json.dumps(total_force_summary, indent=2, sort_keys=True) + "\n",
    )
    write_text(
        summaries_dir / "per_level_force_totals.json",
        json.dumps(per_level_force_totals, indent=2, sort_keys=True) + "\n",
    )
    write_text(
        summaries_dir / "class2_subterm_energy_trace.json",
        json.dumps(class2_subterm_energy_trace, indent=2, sort_keys=True) + "\n",
    )
    write_text(
        summaries_dir / "cpu_correction_energy_trace.json",
        json.dumps(cpu_correction_energy_trace, indent=2, sort_keys=True) + "\n",
    )
    write_text(
        summaries_dir / "restart_summary.json",
        json.dumps(restart_summary_data, indent=2, sort_keys=True) + "\n",
    )
    write_text(summaries_dir / "commands.json", json.dumps(commands, indent=2, sort_keys=True) + "\n")
    write_commands_script(system_root / "run_commands.sh", commands)

    return {
        "system_id": system_id,
        "artifact_root": str(system_root),
        "mdp": str(mdp_path),
        "commands_json": str(summaries_dir / "commands.json"),
        "commands_sh": str(system_root / "run_commands.sh"),
        "event_trace": str(summaries_dir / "event_trace.json"),
        "energy_terms": str(summaries_dir / "energy_terms.json"),
        "total_force_dump": str(full_dir / "total_force.tsv"),
        "total_force_summary": str(summaries_dir / "total_force_summary.json"),
        "per_level_force_totals": str(summaries_dir / "per_level_force_totals.json"),
        "class2_subterm_energy_trace": str(summaries_dir / "class2_subterm_energy_trace.json"),
        "cpu_correction_energy_trace": str(summaries_dir / "cpu_correction_energy_trace.json"),
        "merge_trace_dir": str(full_dir / "merge_trace"),
        "m2p_trace_dir": str(full_dir / "m2p_trace"),
        "restart_summary": str(summaries_dir / "restart_summary.json"),
        "full_run_outputs": {
            "tpr": str(full_deffnm.with_suffix(".tpr")),
            "edr": str(full_deffnm.with_suffix(".edr")),
            "trr": str(full_deffnm.with_suffix(".trr")),
            "cpt": str(full_deffnm.with_suffix(".cpt")),
            "gro": str(full_deffnm.with_suffix(".gro")),
            "log": str(full_deffnm.with_suffix(".log")),
            "event_trace_tsv": str(full_dir / "event_trace.tsv"),
        },
        "restart_outputs": {
            "full": {
                "edr": str(restart_full_deffnm.with_suffix(".edr")),
                "trr": str(restart_full_deffnm.with_suffix(".trr")),
                "cpt": str(restart_full_deffnm.with_suffix(".cpt")),
            },
            "split": {
                "edr": str(restart_split_deffnm.with_suffix(".edr")),
                "trr": str(restart_split_deffnm.with_suffix(".trr")),
                "cpt": str(restart_split_deffnm.with_suffix(".cpt")),
            },
        },
        "logs": {
            "grompp_full_stdout": str(full_grompp_stdout),
            "grompp_full_stderr": str(full_grompp_stderr),
            "mdrun_full_stdout": str(full_mdrun_stdout),
            "mdrun_full_stderr": str(full_mdrun_stderr),
            "grompp_restart_full_stdout": str(restart_full_grompp_stdout),
            "grompp_restart_full_stderr": str(restart_full_grompp_stderr),
            "mdrun_restart_full_stdout": str(restart_full_mdrun_stdout),
            "mdrun_restart_full_stderr": str(restart_full_mdrun_stderr),
            "grompp_restart_split_stdout": str(restart_split_grompp_stdout),
            "grompp_restart_split_stderr": str(restart_split_grompp_stderr),
            "mdrun_restart_split_first_stdout": str(restart_split_first_stdout),
            "mdrun_restart_split_first_stderr": str(restart_split_first_stderr),
            "mdrun_restart_split_second_stdout": str(restart_split_second_stdout),
            "mdrun_restart_split_second_stderr": str(restart_split_second_stderr),
        },
    }


def build_manifest(args: argparse.Namespace, out_root: Path, gmx: Path, gmx_version: str, systems: list[dict[str, object]]) -> dict[str, object]:
    precision_match = re.search(r"Precision:\s*(.+)", gmx_version)
    precision_mode = precision_match.group(1).strip() if precision_match is not None else "unknown"
    return {
        "schema_version": 1,
        "gate": "Gate A",
        "status": "PASS",
        "objective": "Freeze a CPU-only oracle for the standalone exact r-RESPA path before GPU validation.",
        "artifact_root": str(out_root),
        "gmx": str(gmx.resolve()),
        "gmx_version": gmx_version.strip(),
        "precision_mode": precision_mode,
        "ntmpi": args.ntmpi,
        "ntomp": args.ntomp,
        "dlb": "no",
        "pme_rank_count": 0,
        "reproducibility_flags": [
            "-reprod",
            "-dlb no",
            "-pin off",
            "-nb cpu",
            "-pme cpu",
            "-bonded cpu",
            "-update cpu",
            "GMX_DISABLE_MODULAR_SIMULATOR=1",
        ],
        "rerun_used": False,
        "normal_md_used": True,
        "schedule": {
            "outer_steps": args.outer_steps,
            "pair14_level": args.pair14_level,
            "level_step_factors": list(LEVEL_FACTORS),
        },
        "systems": systems,
        "known_limitations": [
            "The standalone exact r-RESPA path is frozen from the direct CLI mdrun path, but it still executes inside the legacy simulator container; the trace itself comes from exactrespastepper.cpp standalone exact-r-RESPA entrypoints.",
            "The direct CLI path has a bootstrap step-0 event pattern that differs from the older LAMMPS-style recursive test harness; Gate A freezes the actual CLI behavior rather than forcing the older wrapper-derived reference.",
            "Per-term virial contributors are not frozen individually; the oracle freezes EDR virial and pressure tensor components plus raw per-level merge-trace virial-related buffers.",
            "PCFF class2 subterm visibility is frozen from a host-side diagnostic rescan of the exact-r-RESPA level interaction lists and coordinates; it is an ownership/debug oracle, not a raw GPU accumulator dump.",
            "CPU reciprocal/self/exclusion electrostatics are frozen from runtime split traces written by the exact standalone path; they are later comparison oracles, not standalone user-facing EDR terms.",
            "No GPU-path comparison is performed here; Gate A only freezes the CPU oracle and validates its internal consistency.",
        ],
        "recommended_comparison_fields_gates_b_h": [
            "event_trace.actual_event_trace[].base_step",
            "event_trace.actual_event_trace[].event",
            "event_trace.actual_event_trace[].level",
            "per_level_force_totals.entries[].relative_path",
            "per_level_force_totals.entries[].vector_sum",
            "class2_subterm_energy_trace.entries[].step",
            "class2_subterm_energy_trace.entries[].level",
            "class2_subterm_energy_trace.entries[].terms_kj_mol.*",
            "class2_subterm_energy_trace.entries[].interaction_counts.*",
            "cpu_correction_energy_trace.entries[].step",
            "cpu_correction_energy_trace.entries[].level",
            "cpu_correction_energy_trace.entries[].terms_kj_mol.*",
            "cpu_correction_energy_trace.entries[].interaction_counts.*",
            "total_force_summary.entries[].step",
            "total_force_summary.entries[].highest_active_level",
            "total_force_summary.entries[].force",
            "energy_terms.step0_terms_kj_mol.*",
            "energy_terms.derived_terms_step0_kj_mol.*",
            "energy_terms.frames[].terms[Potential]",
            "energy_terms.frames[].terms[Total Energy]",
            "energy_terms.frames[].terms[Vir-XX]",
            "energy_terms.frames[].terms[Vir-YY]",
            "energy_terms.frames[].terms[Vir-ZZ]",
            "energy_terms.frames[].terms[Pres-XX]",
            "energy_terms.frames[].terms[Pres-YY]",
            "energy_terms.frames[].terms[Pres-ZZ]",
            "m2p_trace/step0_force_component_trace.txt",
            "m2p_trace/step0_potential_ledger_trace.txt",
            "m2p_trace/step0_virial_pressure_ledger_trace.txt",
            "m2p_trace/step0_realspace_force_subcomponent_trace.txt",
            "restart_summary.potential_abs_delta_kj_mol",
            "restart_summary.total_abs_delta_kj_mol",
            "restart_summary.max_coordinate_abs_delta_nm",
            "restart_summary.max_velocity_abs_delta_nm_ps",
        ],
    }


def write_manifest_markdown(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Gate A Oracle Manifest",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Artifact root: `{manifest['artifact_root']}`",
        f"- Precision mode: `{manifest['precision_mode']}`",
        f"- ntmpi / ntomp: `{manifest['ntmpi']}` / `{manifest['ntomp']}`",
        "- DLB: `no`",
        "- PME rank count: `0`",
        "- Reproducibility flags: `-reprod -dlb no -pin off -nb cpu -pme cpu -bonded cpu -update cpu`",
        "- Simulator pin: `GMX_DISABLE_MODULAR_SIMULATOR=1`",
        "- Rerun used: `no`",
        "- Normal MD used: `yes`",
        "",
        "## Systems",
    ]
    for system in manifest["systems"]:
        lines.extend(
            [
                f"- `{system['system_id']}` event trace: `{system['event_trace']}`",
                f"- `{system['system_id']}` per-level totals: `{system['per_level_force_totals']}`",
                f"- `{system['system_id']}` class2 subterms: `{system['class2_subterm_energy_trace']}`",
                f"- `{system['system_id']}` cpu corrections: `{system['cpu_correction_energy_trace']}`",
                f"- `{system['system_id']}` energy terms: `{system['energy_terms']}`",
                f"- `{system['system_id']}` restart summary: `{system['restart_summary']}`",
            ]
        )
    lines.extend(["", "## Known Limitations"])
    lines.extend(f"- {item}" for item in manifest["known_limitations"])
    lines.extend(["", "## Recommended Comparison Fields"])
    lines.extend(f"- {item}" for item in manifest["recommended_comparison_fields_gates_b_h"])
    write_text(path, "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    gmx = Path(args.gmx).resolve()
    out_root = Path(args.out).resolve()

    if not args.skip_build:
        run_command(
            ["cmake", "--build", str(BUILD_DIR), "--target", args.build_target, "-j4"],
            cwd=REPO_ROOT,
        )

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    system_manifests = [collect_system_artifacts(args, gmx, out_root, system_id) for system_id in SYSTEMS]
    gmx_version = capture_output([str(gmx), "--version"], cwd=REPO_ROOT)
    manifest = build_manifest(args, out_root, gmx, gmx_version, system_manifests)
    write_text(out_root / "oracle_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_manifest_markdown(out_root / "oracle_manifest.md", manifest)


if __name__ == "__main__":
    main()
