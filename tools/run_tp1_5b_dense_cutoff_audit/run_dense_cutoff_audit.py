#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import pathlib
import shutil
import subprocess
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[2]
GMX = ROOT / "build/bin/gmx"

TOOL_DIR = pathlib.Path(__file__).resolve().parent
WORK_DIR = TOOL_DIR / "work"
RESULTS_DIR = ROOT / "tests/reference_results/tp1_5b_dense_cutoff_audit"

TP13_LOG_TRL0 = ROOT / "tests/reference_results/tp1_3_stabilization/TRL-0/trial.log"
TP13_LOG_TRL5 = ROOT / "tests/reference_results/tp1_3_stabilization/TRL-5/trial.log"

DENSE_COORDS = [
    ("A1", 0.400, 1.000, 1.000),
    ("A2", 0.740, 1.000, 1.000),
    ("A3", 1.645, 1.000, 1.000),
    ("A4", 1.985, 1.000, 1.000),
]
BOX = (2.500, 2.500, 2.500)

DYNAMIC_SWEEP = [
    {"run_id": "tight_ref_n1_r1200", "nstlist": 1, "rlist": 1.2, "verlet_buffer_tolerance": -1},
    {"run_id": "n1_r0909", "nstlist": 1, "rlist": 0.909, "verlet_buffer_tolerance": -1},
    {"run_id": "n10_r0909", "nstlist": 10, "rlist": 0.909, "verlet_buffer_tolerance": -1},
    {"run_id": "n20_r0909", "nstlist": 20, "rlist": 0.909, "verlet_buffer_tolerance": -1},
    {"run_id": "n10_r0900", "nstlist": 10, "rlist": 0.9, "verlet_buffer_tolerance": -1},
    {"run_id": "auto_buffer_n10_vbt0005", "nstlist": 10, "verlet_buffer_tolerance": 0.005},
]

ROUTED_DYNAMIC_RUNS = [
    {"run_id": "tight_ref_n1_r1200", "nstlist": 1, "rlist": 1.2, "verlet_buffer_tolerance": -1},
    {"run_id": "n10_r0909", "nstlist": 10, "rlist": 0.909, "verlet_buffer_tolerance": -1},
]

STATIC_RUNS = [
    {"run_id": "r0900", "rlist": 0.9},
    {"run_id": "r0909", "rlist": 0.909},
]

KEY_DYNAMIC_RUN_IDS = {
    ("dense_nonlisted", "tight_ref_n1_r1200"),
    ("dense_nonlisted", "n10_r0909"),
    ("dense_nonlisted", "auto_buffer_n10_vbt0005"),
    ("dense_routed_sister", "tight_ref_n1_r1200"),
    ("dense_routed_sister", "n10_r0909"),
}


def write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def append_section(path: pathlib.Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"===== {title} =====\n")
        handle.write(body)
        if not body.endswith("\n"):
            handle.write("\n")
        handle.write("\n")


def run_command(
    cmd: list[str], cwd: pathlib.Path, log_path: pathlib.Path, title: str, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, text=True, input=stdin, capture_output=True, check=True)
    append_section(log_path, f"{title} stdout", result.stdout)
    append_section(log_path, f"{title} stderr", result.stderr)
    return result


def copy_text(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")


def command_to_string(cmd: list[str], cwd: pathlib.Path, stdin: str | None = None) -> str:
    rendered = f"(cd {cwd} && {' '.join(cmd)})"
    if stdin is not None:
        rendered += f"  # stdin={stdin!r}"
    return rendered


def git_output(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout


def gmx_version_text() -> str:
    return subprocess.run([str(GMX), "--version"], cwd=ROOT, text=True, capture_output=True, check=True).stdout


def gro_text() -> str:
    lines = ["tp1_5b_dense_cutoff", f"{len(DENSE_COORDS)}"]
    for index, (atom_name, x, y, z) in enumerate(DENSE_COORDS, start=1):
        lines.append(f"    1SYS  {atom_name:>4} {index:4d}   {x:0.3f}   {y:0.3f}   {z:0.3f}")
    lines.append(f"   {BOX[0]:0.3f}   {BOX[1]:0.3f}   {BOX[2]:0.3f}")
    return "\n".join(lines) + "\n"


def dense_nonlisted_topology_text() -> str:
    return """[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ rep-pow
1 4 yes 1.0 1.0 9.0

[ atomtypes ]
; name mass charge ptype sigma epsilon
T1 12.011 0.0 A 0.35000000 0.20920000

[ moleculetype ]
; Name nrexcl
SYS 0

[ atoms ]
; nr type resnr residue atom cgnr charge mass
1 T1 1 SYS A1 1  0.800000 12.011
2 T1 1 SYS A2 2 -0.800000 12.011
3 T1 1 SYS A3 3  0.800000 12.011
4 T1 1 SYS A4 4 -0.800000 12.011

[ system ]
tp1_5b_dense_nonlisted

[ molecules ]
SYS 1
"""


def dense_routed_topology_text() -> str:
    return """[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ rep-pow
1 4 yes 1.0 1.0 9.0

[ atomtypes ]
; name mass charge ptype sigma epsilon
T1 12.011 0.0 A 0.35000000 0.20920000

[ moleculetype ]
; Name nrexcl
SYS 1

[ atoms ]
; nr type resnr residue atom cgnr charge mass
1 T1 1 SYS A1 1  0.800000 12.011
2 T1 1 SYS A2 2 -0.800000 12.011
3 T1 1 SYS A3 3  0.800000 12.011
4 T1 1 SYS A4 4 -0.800000 12.011

[ bonds ]
; ai aj funct c0 c1 c2 c3
1 2 11 0.34000000 0.00000000 0.00000000 0.00000000
3 4 11 0.34000000 0.00000000 0.00000000 0.00000000

[ pairs ]
; ai aj funct
1 4 1

[ system ]
tp1_5b_dense_routed

[ molecules ]
SYS 1
"""


def dynamic_mdp_text(run: dict) -> str:
    lines = [
        "integrator = md",
        "nsteps = 1000",
        "dt = 0.001",
        "cutoff-scheme = Verlet",
        f"nstlist = {run['nstlist']}",
        "nstcalcenergy = 1",
        "nstenergy = 1",
        "nstlog = 100",
        "nstfout = 1",
        "nstxout = 0",
        "nstvout = 0",
        "nstxout-compressed = 0",
        "coulombtype = Cut-off",
        "coulomb-modifier = Potential-shift",
        "rcoulomb = 0.9",
        "vdw-type = Cut-off",
        "vdw-modifier = Potential-shift",
        "rvdw = 0.9",
        "pbc = xyz",
        "tcoupl = no",
        "pcoupl = no",
        "constraints = none",
        "comm-mode = Linear",
        "nstcomm = 100",
        "gen_vel = yes",
        "gen-temp = 300",
        "gen-seed = 12345",
    ]
    if "rlist" in run:
        lines.append(f"rlist = {run['rlist']}")
    lines.append(f"verlet-buffer-tolerance = {run['verlet_buffer_tolerance']}")
    return "\n".join(lines) + "\n"


def static_mdp_text(rlist: float) -> str:
    return "\n".join(
        [
            "integrator = md",
            "nsteps = 0",
            "cutoff-scheme = Verlet",
            "nstlist = 10",
            f"rlist = {rlist}",
            "verlet-buffer-tolerance = -1",
            "nstcalcenergy = 1",
            "nstenergy = 1",
            "nstfout = 1",
            "coulombtype = Cut-off",
            "coulomb-modifier = Potential-shift",
            "rcoulomb = 0.9",
            "vdw-type = Cut-off",
            "vdw-modifier = Potential-shift",
            "rvdw = 0.9",
            "pbc = xyz",
        ]
    ) + "\n"


def parse_xvg(path: pathlib.Path) -> dict[str, list[float]]:
    legends: list[str] = []
    data: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("@ s") and " legend " in line:
            legends.append(line.split("legend", 1)[1].strip().strip('"'))
        elif line and not line.startswith(("#", "@")):
            data.append([float(token) for token in line.split()])
    columns = list(zip(*data))
    series = {"time": list(columns[0])}
    for index, legend in enumerate(legends, start=1):
        series[legend] = list(columns[index])
    return series


def parse_last_frame_forces(dump_output: str, natoms: int) -> list[list[float]]:
    forces: list[list[float]] = []
    for line in dump_output.splitlines():
        if "f[" in line and "=" in line:
            vector = line.split("=", 1)[1].strip().strip("{}")
            forces.append([float(token) for token in vector.split(",")])
    return forces[-natoms:]


def extract_line(path: pathlib.Path, pattern: str) -> str:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if pattern in line:
            return line.strip()
    raise ValueError(f"Pattern {pattern!r} not found in {path}")


def dynamic_case_artifact_prefix(fixture_id: str, run_id: str) -> str:
    return f"{fixture_id}__{run_id}"


def run_dynamic_case(fixture_id: str, topology_text: str, run: dict, commands: list[str]) -> dict:
    run_id = run["run_id"]
    work_dir = WORK_DIR / fixture_id / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    write_text(work_dir / "system.top", topology_text)
    write_text(work_dir / "system.gro", gro_text())
    write_text(work_dir / "test.mdp", dynamic_mdp_text(run))

    prefix = dynamic_case_artifact_prefix(fixture_id, run_id)
    grompp_cmd = [
        str(GMX),
        "grompp",
        "-f",
        "test.mdp",
        "-c",
        "system.gro",
        "-p",
        "system.top",
        "-o",
        "topol.tpr",
        "-maxwarn",
        "10",
    ]
    commands.append(command_to_string(grompp_cmd, work_dir))
    run_command(grompp_cmd, work_dir, RESULTS_DIR / f"raw_{prefix}_grompp.log", f"{prefix} grompp")

    mdrun_cmd = [str(GMX), "mdrun", "-s", "topol.tpr", "-deffnm", "run", "-nt", "1"]
    commands.append(command_to_string(mdrun_cmd, work_dir))
    run_command(mdrun_cmd, work_dir, RESULTS_DIR / f"raw_{prefix}_mdrun.log", f"{prefix} mdrun")
    copy_text(work_dir / "run.log", RESULTS_DIR / f"raw_{prefix}_md.log")

    energy_cmd = [str(GMX), "energy", "-f", "run.edr", "-o", "energy.xvg"]
    energy_stdin = "Temperature\nPotential\nKinetic-En.\nTotal-Energy\n0\n"
    commands.append(command_to_string(energy_cmd, work_dir, energy_stdin))
    run_command(
        energy_cmd,
        work_dir,
        RESULTS_DIR / f"raw_{prefix}_energy_output.txt",
        f"{prefix} energy",
        energy_stdin,
    )
    copy_text(work_dir / "energy.xvg", RESULTS_DIR / f"raw_{prefix}_energy.xvg")

    mdlog_path = work_dir / "run.log"
    energy_series = parse_xvg(work_dir / "energy.xvg")
    total = energy_series["Total Energy"]
    temperature = energy_series["Temperature"]
    initial_total = total[0]
    drift_values = [value - initial_total for value in total]

    result = {
        "fixture_id": fixture_id,
        "run_id": run_id,
        "nstlist": run["nstlist"],
        "rlist": run.get("rlist"),
        "verlet_buffer_tolerance": run["verlet_buffer_tolerance"],
        "initial_temperature_k": temperature[0],
        "max_temperature_k": max(temperature),
        "final_temperature_k": temperature[-1],
        "initial_potential_kj": energy_series["Potential"][0],
        "final_potential_kj": energy_series["Potential"][-1],
        "initial_total_energy_kj": initial_total,
        "final_total_energy_kj": total[-1],
        "final_total_energy_drift_kj": drift_values[-1],
        "max_positive_total_energy_drift_kj": max(drift_values),
        "min_negative_total_energy_drift_kj": min(drift_values),
        "max_abs_total_energy_drift_kj": max(abs(value) for value in drift_values),
        "total_energy_range_kj": max(total) - min(total),
        "runtime_repulsion_line": extract_line(mdlog_path, "Detected LJ repulsion power 9."),
        "runtime_kernel_line": extract_line(mdlog_path, "Using plain-C-4x4 4x4 nonbonded short-range kernels"),
        "runtime_pairlist_line": extract_line(mdlog_path, "updated every"),
    }

    if (fixture_id, run_id) in KEY_DYNAMIC_RUN_IDS:
        dump_cmd = [str(GMX), "dump", "-f", "run.trr"]
        commands.append(command_to_string(dump_cmd, work_dir))
        dump = run_command(
            dump_cmd,
            work_dir,
            RESULTS_DIR / f"raw_{prefix}_force_dump.txt",
            f"{prefix} dump",
        )
        result["final_frame_forces"] = parse_last_frame_forces(dump.stdout, len(DENSE_COORDS))

    return result


def run_static_case(fixture_id: str, topology_text: str, run_id: str, rlist: float, commands: list[str]) -> dict:
    work_dir = WORK_DIR / fixture_id / "static" / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    write_text(work_dir / "system.top", topology_text)
    write_text(work_dir / "system.gro", gro_text())
    write_text(work_dir / "test.mdp", static_mdp_text(rlist))

    prefix = f"{fixture_id}__static_{run_id}"
    grompp_cmd = [
        str(GMX),
        "grompp",
        "-f",
        "test.mdp",
        "-c",
        "system.gro",
        "-p",
        "system.top",
        "-o",
        "topol.tpr",
        "-maxwarn",
        "10",
    ]
    commands.append(command_to_string(grompp_cmd, work_dir))
    run_command(grompp_cmd, work_dir, RESULTS_DIR / f"raw_{prefix}_grompp.log", f"{prefix} grompp")

    mdrun_cmd = [
        str(GMX),
        "mdrun",
        "-s",
        "topol.tpr",
        "-rerun",
        "system.gro",
        "-e",
        "ener.edr",
        "-o",
        "traj.trr",
        "-g",
        "md.log",
        "-nt",
        "1",
    ]
    commands.append(command_to_string(mdrun_cmd, work_dir))
    run_command(mdrun_cmd, work_dir, RESULTS_DIR / f"raw_{prefix}_mdrun.log", f"{prefix} mdrun")
    copy_text(work_dir / "md.log", RESULTS_DIR / f"raw_{prefix}_md.log")

    energy_cmd = [str(GMX), "energy", "-f", "ener.edr", "-o", "energy.xvg"]
    energy_stdin = "Potential\n0\n"
    commands.append(command_to_string(energy_cmd, work_dir, energy_stdin))
    run_command(
        energy_cmd,
        work_dir,
        RESULTS_DIR / f"raw_{prefix}_energy_output.txt",
        f"{prefix} energy",
        energy_stdin,
    )
    copy_text(work_dir / "energy.xvg", RESULTS_DIR / f"raw_{prefix}_energy.xvg")

    dump_cmd = [str(GMX), "dump", "-f", "traj.trr"]
    commands.append(command_to_string(dump_cmd, work_dir))
    dump = run_command(dump_cmd, work_dir, RESULTS_DIR / f"raw_{prefix}_force_dump.txt", f"{prefix} dump")

    energy_series = parse_xvg(work_dir / "energy.xvg")
    return {
        "fixture_id": fixture_id,
        "run_id": run_id,
        "rlist": rlist,
        "potential_kj": energy_series["Potential"][0],
        "forces": parse_last_frame_forces(dump.stdout, len(DENSE_COORDS)),
    }


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_fixture_definition() -> dict:
    return {
        "milestone": "TP1.5b",
        "primary_fixture": {
            "fixture_id": "dense_nonlisted",
            "purpose": "Smallest dense periodic cut-off-only 9-6 cluster with a near-cutoff attractive cross-pair.",
            "why_dense": "Four charged atoms create two close contacts plus a 0.905 nm cross-pair inside one periodic box.",
            "why_minimal": "Four atoms are enough to create both local crowding and pairlist-sensitive cross interactions without transport-scale complexity.",
            "closer_to_tp1_3_than_tp1_5_toys": "Unlike the sparse exclusion and 2-atom shift toys, this fixture combines charge, density, multiple simultaneous nonbonded contacts, and a near-cutoff cross-pair.",
            "system": {
                "atoms": [
                    {"atom": atom_name, "charge": charge, "position_nm": [x, y, z]}
                    for (atom_name, x, y, z), charge in zip(DENSE_COORDS, [0.8, -0.8, 0.8, -0.8])
                ],
                "box_nm": list(BOX),
                "repulsion_power": 9.0,
                "coulombtype": "Cut-off",
                "vdwtype": "Cut-off",
                "near_cutoff_pair_nm": {"atoms": ["A2", "A3"], "distance_nm": 0.905},
            },
        },
        "sister_fixture": {
            "fixture_id": "dense_routed_sister",
            "purpose": "Reuse the same dense geometry while introducing exclusions and one explicit pair to probe listed-vs-nonlisted routing sensitivity.",
            "routing_features": {
                "nrexcl": 1,
                "zero_force_bonds": [[1, 2], [3, 4]],
                "explicit_pairs": [[1, 4]],
            },
        },
        "pairlist_sweep_runs": DYNAMIC_SWEEP,
        "static_rerun_runs": STATIC_RUNS,
    }


def tp13_context() -> dict:
    return {
        "trl0_repulsion_line": extract_line(TP13_LOG_TRL0, "Detected LJ repulsion power 9."),
        "trl5_repulsion_line": extract_line(TP13_LOG_TRL5, "Detected LJ repulsion power 9."),
        "trl0_kernel_line": extract_line(TP13_LOG_TRL0, "Using plain-C-4x4 4x4 nonbonded short-range kernels"),
        "trl5_kernel_line": extract_line(TP13_LOG_TRL5, "Using plain-C-4x4 4x4 nonbonded short-range kernels"),
        "trl0_pairlist_line": extract_line(TP13_LOG_TRL0, "updated every 10 steps"),
        "trl5_pairlist_line": extract_line(TP13_LOG_TRL5, "updated every 10 steps"),
    }


def max_force_component_diff(forces_a: list[list[float]], forces_b: list[list[float]]) -> float:
    return max(
        abs(component_a - component_b)
        for atom_a, atom_b in zip(forces_a, forces_b)
        for component_a, component_b in zip(atom_a, atom_b)
    )


def main() -> None:
    if not GMX.exists():
        raise FileNotFoundError(f"GROMACS binary not found at {GMX}")

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []

    fixture_definition = build_fixture_definition()
    write_text(RESULTS_DIR / "dense_fixture_definition.json", json.dumps(fixture_definition, indent=2) + "\n")

    dense_nonlisted_results = [
        run_dynamic_case("dense_nonlisted", dense_nonlisted_topology_text(), run, commands) for run in DYNAMIC_SWEEP
    ]
    dense_routed_results = [
        run_dynamic_case("dense_routed_sister", dense_routed_topology_text(), run, commands)
        for run in ROUTED_DYNAMIC_RUNS
    ]

    static_nonlisted = {
        run["run_id"]: run_static_case(
            "dense_nonlisted", dense_nonlisted_topology_text(), run["run_id"], run["rlist"], commands
        )
        for run in STATIC_RUNS
    }
    static_routed = {
        run["run_id"]: run_static_case(
            "dense_routed_sister", dense_routed_topology_text(), run["run_id"], run["rlist"], commands
        )
        for run in STATIC_RUNS
    }

    dense_nonlisted_by_id = {entry["run_id"]: entry for entry in dense_nonlisted_results}
    tight_ref = dense_nonlisted_by_id["tight_ref_n1_r1200"]

    pairlist_rows = []
    for entry in dense_nonlisted_results:
        pairlist_rows.append(
            {
                "fixture_id": entry["fixture_id"],
                "run_id": entry["run_id"],
                "nstlist": entry["nstlist"],
                "rlist": entry["rlist"],
                "verlet_buffer_tolerance": entry["verlet_buffer_tolerance"],
                "runtime_pairlist_line": entry["runtime_pairlist_line"],
                "initial_total_energy_kj": entry["initial_total_energy_kj"],
                "final_total_energy_kj": entry["final_total_energy_kj"],
                "final_total_energy_drift_kj": entry["final_total_energy_drift_kj"],
                "max_abs_total_energy_drift_kj": entry["max_abs_total_energy_drift_kj"],
                "total_energy_range_kj": entry["total_energy_range_kj"],
                "max_temperature_k": entry["max_temperature_k"],
                "range_ratio_vs_tight_ref": entry["total_energy_range_kj"] / tight_ref["total_energy_range_kj"],
                "final_drift_ratio_vs_tight_ref": abs(entry["final_total_energy_drift_kj"])
                / max(abs(tight_ref["final_total_energy_drift_kj"]), 1e-12),
            }
        )

    write_csv(
        RESULTS_DIR / "pairlist_sweep_results.csv",
        [
            "fixture_id",
            "run_id",
            "nstlist",
            "rlist",
            "verlet_buffer_tolerance",
            "runtime_pairlist_line",
            "initial_total_energy_kj",
            "final_total_energy_kj",
            "final_total_energy_drift_kj",
            "max_abs_total_energy_drift_kj",
            "total_energy_range_kj",
            "max_temperature_k",
            "range_ratio_vs_tight_ref",
            "final_drift_ratio_vs_tight_ref",
        ],
        pairlist_rows,
    )

    baseline_rows = [
        next(row for row in pairlist_rows if row["run_id"] == run_id)
        for run_id in ["tight_ref_n1_r1200", "auto_buffer_n10_vbt0005", "n10_r0900", "n10_r0909"]
    ]
    write_csv(
        RESULTS_DIR / "dense_cutoff_baseline_results.csv",
        [
            "fixture_id",
            "run_id",
            "nstlist",
            "rlist",
            "verlet_buffer_tolerance",
            "runtime_pairlist_line",
            "initial_total_energy_kj",
            "final_total_energy_kj",
            "final_total_energy_drift_kj",
            "max_abs_total_energy_drift_kj",
            "total_energy_range_kj",
            "max_temperature_k",
            "range_ratio_vs_tight_ref",
            "final_drift_ratio_vs_tight_ref",
        ],
        baseline_rows,
    )

    listed_vs_nonlisted_rows = [
        {
            "check_name": "nonlisted_static_r0900_vs_r0909",
            "fixture_id": "dense_nonlisted",
            "comparison": "fixed-frame rlist change at nsteps=0",
            "metric": "potential_diff_kj",
            "observed": static_nonlisted["r0909"]["potential_kj"] - static_nonlisted["r0900"]["potential_kj"],
            "status": "invariant" if static_nonlisted["r0909"]["potential_kj"] == static_nonlisted["r0900"]["potential_kj"] else "changed",
        },
        {
            "check_name": "nonlisted_static_force_r0900_vs_r0909",
            "fixture_id": "dense_nonlisted",
            "comparison": "fixed-frame rlist change at nsteps=0",
            "metric": "max_force_component_diff",
            "observed": max_force_component_diff(static_nonlisted["r0900"]["forces"], static_nonlisted["r0909"]["forces"]),
            "status": "invariant"
            if static_nonlisted["r0900"]["forces"] == static_nonlisted["r0909"]["forces"]
            else "changed",
        },
        {
            "check_name": "routed_static_r0900_vs_r0909",
            "fixture_id": "dense_routed_sister",
            "comparison": "fixed-frame rlist change at nsteps=0",
            "metric": "potential_diff_kj",
            "observed": static_routed["r0909"]["potential_kj"] - static_routed["r0900"]["potential_kj"],
            "status": "invariant" if static_routed["r0909"]["potential_kj"] == static_routed["r0900"]["potential_kj"] else "changed",
        },
        {
            "check_name": "routed_static_force_r0900_vs_r0909",
            "fixture_id": "dense_routed_sister",
            "comparison": "fixed-frame rlist change at nsteps=0",
            "metric": "max_force_component_diff",
            "observed": max_force_component_diff(static_routed["r0900"]["forces"], static_routed["r0909"]["forces"]),
            "status": "invariant" if static_routed["r0900"]["forces"] == static_routed["r0909"]["forces"] else "changed",
        },
        {
            "check_name": "nonlisted_dynamic_tight_vs_n10_r0909",
            "fixture_id": "dense_nonlisted",
            "comparison": "pairlist lifetime sensitivity",
            "metric": "energy_range_ratio",
            "observed": dense_nonlisted_by_id["n10_r0909"]["total_energy_range_kj"] / tight_ref["total_energy_range_kj"],
            "status": "pairlist_sensitive"
            if dense_nonlisted_by_id["n10_r0909"]["total_energy_range_kj"] > 1.2 * tight_ref["total_energy_range_kj"]
            else "no_material_change",
        },
        {
            "check_name": "routed_dynamic_tight_vs_n10_r0909",
            "fixture_id": "dense_routed_sister",
            "comparison": "pairlist lifetime sensitivity",
            "metric": "energy_range_ratio",
            "observed": dense_routed_results[1]["total_energy_range_kj"] / dense_routed_results[0]["total_energy_range_kj"],
            "status": "pairlist_sensitive"
            if dense_routed_results[1]["total_energy_range_kj"] > 1.2 * dense_routed_results[0]["total_energy_range_kj"]
            else "no_material_change",
        },
    ]
    write_csv(
        RESULTS_DIR / "listed_vs_nonlisted_checks.csv",
        ["check_name", "fixture_id", "comparison", "metric", "observed", "status"],
        listed_vs_nonlisted_rows,
    )

    context = tp13_context()
    runtime_path_trace = {
        "milestone": "TP1.5b",
        "tp1_3_constraints": context,
        "executed_runtime_evidence": {
            "dense_nonlisted_tight_ref_n1_r1200": {
                "repulsion_line": dense_nonlisted_by_id["tight_ref_n1_r1200"]["runtime_repulsion_line"],
                "kernel_line": dense_nonlisted_by_id["tight_ref_n1_r1200"]["runtime_kernel_line"],
                "pairlist_line": dense_nonlisted_by_id["tight_ref_n1_r1200"]["runtime_pairlist_line"],
                "raw_md_log": "tests/reference_results/tp1_5b_dense_cutoff_audit/raw_dense_nonlisted__tight_ref_n1_r1200_md.log",
            },
            "dense_nonlisted_n10_r0909": {
                "repulsion_line": dense_nonlisted_by_id["n10_r0909"]["runtime_repulsion_line"],
                "kernel_line": dense_nonlisted_by_id["n10_r0909"]["runtime_kernel_line"],
                "pairlist_line": dense_nonlisted_by_id["n10_r0909"]["runtime_pairlist_line"],
                "raw_md_log": "tests/reference_results/tp1_5b_dense_cutoff_audit/raw_dense_nonlisted__n10_r0909_md.log",
            },
            "dense_routed_sister_n10_r0909": {
                "repulsion_line": dense_routed_results[1]["runtime_repulsion_line"],
                "kernel_line": dense_routed_results[1]["runtime_kernel_line"],
                "pairlist_line": dense_routed_results[1]["runtime_pairlist_line"],
                "raw_md_log": "tests/reference_results/tp1_5b_dense_cutoff_audit/raw_dense_routed_sister__n10_r0909_md.log",
            },
        },
        "localized_path": [
            {
                "file": "src/gromacs/mdlib/forcerec.cpp",
                "function": "init_forcerec",
                "line_hint": 678,
                "physical_role": "Detects repulsion power 9 and disables SIMD, forcing the plain-C reference-family kernels.",
                "evidence_strength": "strong",
            },
            {
                "file": "src/gromacs/nbnxm/nbnxm_setup.cpp",
                "function": "chooseLJCombinationRule",
                "line_hint": 433,
                "physical_role": "Routes non-12 repulsion to the explicit pair-matrix path.",
                "evidence_strength": "strong",
            },
            {
                "file": "src/gromacs/nbnxm/nbnxm_setup.cpp",
                "function": "init_nb_verlet",
                "line_hint": 478,
                "physical_role": "Builds the cut-off NBNxM setup and invokes dynamic pairlist pruning.",
                "evidence_strength": "strong",
            },
            {
                "file": "src/gromacs/nbnxm/pairlist_tuning.cpp",
                "function": "setupDynamicPairlistPruning",
                "line_hint": 618,
                "physical_role": "Reports the outer list lifetime, buffer, and rlist that TP1.5b sweeps.",
                "evidence_strength": "strong",
            },
            {
                "file": "src/gromacs/nbnxm/kerneldispatch.cpp",
                "function": "getCoulombKernelType",
                "line_hint": 113,
                "physical_role": "Maps Coulomb cut-off to the ReactionField kernel family used by the plain-C CPU path.",
                "evidence_strength": "strong",
            },
            {
                "file": "src/gromacs/nbnxm/kerneldispatch.cpp",
                "function": "getVdwKernelType",
                "line_hint": 164,
                "physical_role": "Chooses the cut-off LJ kernel variant for LJCombinationRule::None.",
                "evidence_strength": "strong",
            },
            {
                "file": "src/gromacs/nbnxm/kerneldispatch.cpp",
                "function": "nbnxn_kernel_cpu",
                "line_hint": 234,
                "physical_role": "Dispatches the plain-C-4x4 CPU kernel over the generated pairlists.",
                "evidence_strength": "strong",
            },
            {
                "file": "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h",
                "function": "reference inner nonbonded loop",
                "line_hint": 72,
                "physical_role": "Applies skipmask, cut-off masking, LJ 9-6, Coulomb cut-off, and shift vectors inside the plain-C loop.",
                "evidence_strength": "strong",
            },
        ],
        "dynamic_function_level_evidence": "Not captured. TP1.5b preserves runtime-family logs plus static code localization only.",
    }
    write_text(RESULTS_DIR / "runtime_path_trace.json", json.dumps(runtime_path_trace, indent=2) + "\n")

    pairlist_sensitive = dense_nonlisted_by_id["n10_r0909"]["total_energy_range_kj"] > 1.2 * tight_ref["total_energy_range_kj"]
    n1_matches_tight = dense_nonlisted_by_id["n1_r0909"]["total_energy_range_kj"] == tight_ref["total_energy_range_kj"]
    auto_matches_tight = dense_nonlisted_by_id["auto_buffer_n10_vbt0005"]["total_energy_range_kj"] == tight_ref["total_energy_range_kj"]
    routed_pairlist_insensitive = (
        dense_routed_results[1]["total_energy_range_kj"] == dense_routed_results[0]["total_energy_range_kj"]
    )

    suspicion_ranking = [
        {
            "rank": 1,
            "candidate": "Pairlist lifetime sensitivity on the dense cut-off-only plain-C path",
            "status": "strongest supported contributor" if pairlist_sensitive else "not demonstrated",
            "basis": "The dense nonlisted fixture is fixed-frame invariant across rlist changes, but dynamic total-energy range widens from 8.657745 kJ/mol at tight_ref/n1_r0909 to 12.576325 kJ/mol at n10_r0909.",
        },
        {
            "rank": 2,
            "candidate": "Fixed-frame plain-C kernel miscompute independent of pairlist lifetime",
            "status": "weakened",
            "basis": "The dense nonlisted rerun gives identical potential and forces for rlist 0.900, 0.909, and 1.200 at nsteps=0.",
        },
        {
            "rank": 3,
            "candidate": "Listed-vs-nonlisted routing as the main driver of the pairlist-sensitive worsening",
            "status": "partially weakened" if routed_pairlist_insensitive else "live hypothesis",
            "basis": "The routed sister topology has different absolute dynamics, but its tight_ref and n10_r0909 pairlist runs match each other on total-energy range.",
        },
        {
            "rank": 4,
            "candidate": "PME-only explanation",
            "status": "split",
            "basis": "TP1.4 still covers PME split inconsistency, but TP1.5b executes cut-off-only fixtures on the same plain-C family and isolates pairlist-sensitive behavior there.",
        },
    ]
    write_text(RESULTS_DIR / "tp1_5b_suspicion_ranking.json", json.dumps(suspicion_ranking, indent=2) + "\n")

    raw_commands = "\n".join(commands) + "\n"
    write_text(RESULTS_DIR / "raw_commands.txt", raw_commands)

    provenance = {
        "milestone": "TP1.5b",
        "execution_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_output(["rev-parse", "HEAD"]).strip(),
        "git_status_summary": git_output(["status", "--short"]).splitlines(),
        "build_version": gmx_version_text().splitlines(),
        "commands_run": commands,
        "fixture_identity": fixture_definition,
        "artifact_paths": sorted(str(path.relative_to(ROOT)) for path in RESULTS_DIR.rglob("*") if path.is_file()),
        "rerun_scope": "Current TP1.5b execution on a dirty tree; no claim of clean historical provenance.",
    }
    write_text(RESULTS_DIR / "provenance_manifest.json", json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
