#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import pathlib
import shutil
import subprocess
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[2]
GMX = ROOT / "build/bin/gmx"

TOOL_DIR = pathlib.Path(__file__).resolve().parent
WORK_DIR = TOOL_DIR / "work"
RESULTS_DIR = ROOT / "tests/reference_results/tp1_5c_pairlist_trace_audit"

TP15B_FIXTURE = ROOT / "tests/reference_results/tp1_5b_dense_cutoff_audit/dense_fixture_definition.json"
TP15B_SWEEP = ROOT / "tests/reference_results/tp1_5b_dense_cutoff_audit/pairlist_sweep_results.csv"

DENSE_COORDS = [
    ("A1", 0.400, 1.000, 1.000),
    ("A2", 0.740, 1.000, 1.000),
    ("A3", 1.645, 1.000, 1.000),
    ("A4", 1.985, 1.000, 1.000),
]
BOX = (2.500, 2.500, 2.500)

RUNS = [
    {"run_id": "n1_r0909", "nstlist": 1, "rlist": 0.909, "verlet_buffer_tolerance": -1, "trace_suffix": "n1"},
    {"run_id": "n10_r0909", "nstlist": 10, "rlist": 0.909, "verlet_buffer_tolerance": -1, "trace_suffix": "n10"},
]

SENSITIVE_PAIR = (2, 3)


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
    cmd: list[str],
    cwd: pathlib.Path,
    log_path: pathlib.Path,
    title: str,
    stdin: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    effective_env = None
    if env is not None:
        effective_env = {**os.environ, **env}
    result = subprocess.run(
        cmd, cwd=cwd, text=True, input=stdin, capture_output=True, check=True, env=effective_env
    )
    append_section(log_path, f"{title} stdout", result.stdout)
    append_section(log_path, f"{title} stderr", result.stderr)
    return result


def copy_text(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")


def git_output(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout


def gmx_version_text() -> str:
    return subprocess.run([str(GMX), "--version"], cwd=ROOT, text=True, capture_output=True, check=True).stdout


def gro_text() -> str:
    lines = ["tp1_5c_dense_nonlisted", f"{len(DENSE_COORDS)}"]
    for index, (atom_name, x, y, z) in enumerate(DENSE_COORDS, start=1):
        lines.append(f"    1SYS  {atom_name:>4} {index:4d}   {x:0.3f}   {y:0.3f}   {z:0.3f}")
    lines.append(f"   {BOX[0]:0.3f}   {BOX[1]:0.3f}   {BOX[2]:0.3f}")
    return "\n".join(lines) + "\n"


def topology_text() -> str:
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
tp1_5c_dense_nonlisted

[ molecules ]
SYS 1
"""


def dynamic_mdp_text(run: dict) -> str:
    return "\n".join(
        [
            "integrator = md",
            "nsteps = 1000",
            "dt = 0.001",
            "cutoff-scheme = Verlet",
            f"nstlist = {run['nstlist']}",
            f"rlist = {run['rlist']}",
            f"verlet-buffer-tolerance = {run['verlet_buffer_tolerance']}",
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
    ) + "\n"


def command_to_string(
    cmd: list[str], cwd: pathlib.Path, stdin: str | None = None, env: dict[str, str] | None = None
) -> str:
    env_prefix = ""
    if env:
        env_prefix = " ".join(f"{key}={value}" for key, value in sorted(env.items())) + " "
    rendered = f"(cd {cwd} && {env_prefix}{' '.join(cmd)})"
    if stdin is not None:
        rendered += f"  # stdin={stdin!r}"
    return rendered


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


def read_tp15b_reference() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with TP15B_SWEEP.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["run_id"]] = {
                "total_energy_range_kj": float(row["total_energy_range_kj"]),
                "max_abs_total_energy_drift_kj": float(row["max_abs_total_energy_drift_kj"]),
            }
    return rows


def parse_membership(path: pathlib.Path) -> dict[int, dict[str, object]]:
    per_step: dict[int, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["locality"] != "local":
                continue
            step = int(row["step"])
            bucket = per_step.setdefault(
                step,
                {
                    "prune_step": bool(int(row["prune_step"])),
                    "outer": set(),
                    "inner": set(),
                    "outer_excluded": set(),
                    "inner_excluded": set(),
                },
            )
            atom_i = int(row["atom_i"])
            atom_j = int(row["atom_j"])
            key = (min(atom_i, atom_j), max(atom_i, atom_j), int(row["shift_index"]))
            list_key = f"{row['list_kind']}_excluded" if row["pair_kind"] == "excluded" else row["list_kind"]
            bucket[list_key].add(key)
    return per_step


def parse_force_trace(path: pathlib.Path) -> dict[int, dict[int, tuple[float, float, float]]]:
    per_step: dict[int, dict[int, tuple[float, float, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = int(row["step"])
            atom_index = int(row["atom_index"])
            per_step.setdefault(step, {})[atom_index] = (
                float(row["fx"]),
                float(row["fy"]),
                float(row["fz"]),
            )
    return per_step


def summarize_membership(per_step: dict[int, dict[str, object]]) -> dict[str, object]:
    steps = sorted(per_step)
    step_rows = []
    steps_with_difference = 0
    sensitive_outer_steps = 0
    sensitive_inner_steps = 0
    first_difference_step = None
    for step in steps:
        outer = per_step[step]["outer"]
        inner = per_step[step]["inner"]
        outer_only = sorted(outer - inner)
        inner_only = sorted(inner - outer)
        if outer_only or inner_only:
            steps_with_difference += 1
            if first_difference_step is None:
                first_difference_step = step
        sensitive_outer = any(tuple(sorted(pair[:2])) == SENSITIVE_PAIR for pair in outer)
        sensitive_inner = any(tuple(sorted(pair[:2])) == SENSITIVE_PAIR for pair in inner)
        sensitive_outer_steps += int(sensitive_outer)
        sensitive_inner_steps += int(sensitive_inner)
        step_rows.append(
            {
                "step": step,
                "prune_step": per_step[step]["prune_step"],
                "outer_count": len(outer),
                "inner_count": len(inner),
                "outer_only_count": len(outer_only),
                "inner_only_count": len(inner_only),
                "sensitive_pair_in_outer": sensitive_outer,
                "sensitive_pair_in_inner": sensitive_inner,
            }
        )
    return {
        "num_steps_traced": len(steps),
        "num_prune_steps": sum(int(per_step[step]["prune_step"]) for step in steps),
        "steps_with_outer_inner_difference": steps_with_difference,
        "first_outer_inner_difference_step": first_difference_step,
        "sensitive_pair_outer_steps": sensitive_outer_steps,
        "sensitive_pair_inner_steps": sensitive_inner_steps,
        "step_rows": step_rows,
    }


def max_component_diff(vec_a: tuple[float, float, float], vec_b: tuple[float, float, float]) -> float:
    return max(abs(a - b) for a, b in zip(vec_a, vec_b))


def summarize_cross_run_membership(
    n1_membership: dict[int, dict[str, object]],
    n10_membership: dict[int, dict[str, object]],
    n1_force: dict[int, dict[int, tuple[float, float, float]]],
    n10_force: dict[int, dict[int, tuple[float, float, float]]],
) -> dict[str, object]:
    common_steps = sorted(set(n1_membership) & set(n10_membership))
    first_inner_difference_step = None
    first_inner_difference_detail = None
    first_force_difference_step = None
    first_force_difference_detail = None
    max_sensitive_force_component_gap = 0.0
    force_gap_by_step = []

    for step in common_steps:
        n1_inner = n1_membership[step]["inner"]
        n10_inner = n10_membership[step]["inner"]
        if n1_inner != n10_inner and first_inner_difference_step is None:
            first_inner_difference_step = step
            first_inner_difference_detail = {
                "n1_only": sorted(n1_inner - n10_inner),
                "n10_only": sorted(n10_inner - n1_inner),
            }

        n1_atom_forces = n1_force.get(step, {})
        n10_atom_forces = n10_force.get(step, {})
        atom_gaps = []
        for atom_index in SENSITIVE_PAIR:
            if atom_index in n1_atom_forces and atom_index in n10_atom_forces:
                atom_gaps.append(max_component_diff(n1_atom_forces[atom_index], n10_atom_forces[atom_index]))
        gap = max(atom_gaps) if atom_gaps else 0.0
        force_gap_by_step.append({"step": step, "max_sensitive_force_component_gap": gap})
        max_sensitive_force_component_gap = max(max_sensitive_force_component_gap, gap)
        if gap > 1e-12 and first_force_difference_step is None:
            first_force_difference_step = step
            first_force_difference_detail = {
                "atom_2": {
                    "n1": n1_atom_forces.get(2),
                    "n10": n10_atom_forces.get(2),
                },
                "atom_3": {
                    "n1": n1_atom_forces.get(3),
                    "n10": n10_atom_forces.get(3),
                },
            }

    return {
        "common_steps": len(common_steps),
        "first_cross_run_inner_difference_step": first_inner_difference_step,
        "first_cross_run_inner_difference_detail": first_inner_difference_detail,
        "first_cross_run_sensitive_force_difference_step": first_force_difference_step,
        "first_cross_run_sensitive_force_difference_detail": first_force_difference_detail,
        "max_cross_run_sensitive_force_component_gap": max_sensitive_force_component_gap,
        "force_gap_by_step": force_gap_by_step,
    }


def run_case(run: dict, commands: list[str]) -> dict[str, object]:
    run_id = run["run_id"]
    suffix = run["trace_suffix"]
    work_dir = WORK_DIR / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    write_text(work_dir / "system.top", topology_text())
    write_text(work_dir / "system.gro", gro_text())
    write_text(work_dir / "test.mdp", dynamic_mdp_text(run))

    membership_trace = RESULTS_DIR / f"per_step_pair_membership_{suffix}.csv"
    force_trace = RESULTS_DIR / f"per_step_force_trace_{suffix}.csv"
    for trace_path in (membership_trace, force_trace):
        if trace_path.exists():
            trace_path.unlink()

    trace_env = {
        "GMX_TP15C_PAIRLIST_TRACE_PATH": str(membership_trace),
        "GMX_TP15C_FORCE_TRACE_PATH": str(force_trace),
        "GMX_TP15C_TRACE_RANGE": f"{run['rlist']:.6f}",
    }

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
    run_command(grompp_cmd, work_dir, RESULTS_DIR / f"raw_{suffix}_grompp.log", f"{suffix} grompp")

    mdrun_cmd = [str(GMX), "mdrun", "-s", "topol.tpr", "-deffnm", "run", "-nt", "1"]
    commands.append(command_to_string(mdrun_cmd, work_dir, env=trace_env))
    run_command(mdrun_cmd, work_dir, RESULTS_DIR / f"raw_{suffix}_mdrun.log", f"{suffix} mdrun", env=trace_env)
    copy_text(work_dir / "run.log", RESULTS_DIR / f"raw_{suffix}_md.log")

    energy_cmd = [str(GMX), "energy", "-f", "run.edr", "-o", "energy.xvg"]
    energy_stdin = "Temperature\nPotential\nKinetic-En.\nTotal-Energy\n0\n"
    commands.append(command_to_string(energy_cmd, work_dir, stdin=energy_stdin))
    run_command(
        energy_cmd,
        work_dir,
        RESULTS_DIR / f"raw_{suffix}_energy_output.txt",
        f"{suffix} energy",
        stdin=energy_stdin,
    )
    copy_text(work_dir / "energy.xvg", RESULTS_DIR / f"raw_{suffix}_energy.xvg")

    energy_series = parse_xvg(work_dir / "energy.xvg")
    total = energy_series["Total Energy"]
    initial_total = total[0]
    drift_values = [value - initial_total for value in total]

    return {
        "run_id": run_id,
        "trace_suffix": suffix,
        "nstlist": run["nstlist"],
        "rlist": run["rlist"],
        "verlet_buffer_tolerance": run["verlet_buffer_tolerance"],
        "initial_potential_kj": energy_series["Potential"][0],
        "final_potential_kj": energy_series["Potential"][-1],
        "initial_total_energy_kj": initial_total,
        "final_total_energy_kj": total[-1],
        "final_total_energy_drift_kj": drift_values[-1],
        "max_abs_total_energy_drift_kj": max(abs(value) for value in drift_values),
        "total_energy_range_kj": max(total) - min(total),
        "membership_trace_path": str(membership_trace.relative_to(ROOT)),
        "force_trace_path": str(force_trace.relative_to(ROOT)),
        "runtime_kernel_line": next(
            (
                line.strip()
                for line in (work_dir / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()
                if "Using plain-C-4x4 4x4 nonbonded short-range kernels" in line
            ),
            None,
        ),
        "runtime_pairlist_line": next(
            (
                line.strip()
                for line in (work_dir / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()
                if "updated every" in line
            ),
            None,
        ),
    }


def main() -> None:
    if not GMX.exists():
        raise FileNotFoundError(f"GROMACS binary not found at {GMX}")

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fixture_reference = json.loads(TP15B_FIXTURE.read_text(encoding="utf-8"))
    write_text(
        RESULTS_DIR / "dense_fixture_reference.json",
        json.dumps(
            {
                "milestone": "TP1.5c",
                "source_fixture_artifact": str(TP15B_FIXTURE.relative_to(ROOT)),
                "fixture_id": fixture_reference["primary_fixture"]["fixture_id"],
                "fixture": fixture_reference["primary_fixture"],
                "executed_runs": RUNS,
                "trace_note": "TP1.5c uses env-var-gated inner/outer pairlist and final-force tracing on the TP1.5b dense_nonlisted fixture.",
            },
            indent=2,
        )
        + "\n",
    )

    commands = [
        "(cd /home/kiket/바탕화면/test/GROMACS_PCFF && git status --short)",
        "(cd /home/kiket/바탕화면/test/GROMACS_PCFF && git rev-parse HEAD)",
        "(cd /home/kiket/바탕화면/test/GROMACS_PCFF && build/bin/gmx --version)",
        "(cd /home/kiket/바탕화면/test/GROMACS_PCFF && cmake --build build --target gmx -j4)",
    ]
    reference_metrics = read_tp15b_reference()

    run_results = [run_case(run, commands) for run in RUNS]
    results_by_id = {entry["run_id"]: entry for entry in run_results}

    n1_membership = parse_membership(RESULTS_DIR / "per_step_pair_membership_n1.csv")
    n10_membership = parse_membership(RESULTS_DIR / "per_step_pair_membership_n10.csv")
    n1_force = parse_force_trace(RESULTS_DIR / "per_step_force_trace_n1.csv")
    n10_force = parse_force_trace(RESULTS_DIR / "per_step_force_trace_n10.csv")

    n1_membership_summary = summarize_membership(n1_membership)
    n10_membership_summary = summarize_membership(n10_membership)
    cross_run_summary = summarize_cross_run_membership(n1_membership, n10_membership, n1_force, n10_force)

    pairlist_lifetime_supported = cross_run_summary["first_cross_run_inner_difference_step"] is not None
    pruning_supported = (
        n1_membership_summary["steps_with_outer_inner_difference"] > 0
        or n10_membership_summary["steps_with_outer_inner_difference"] > 0
        or n1_membership_summary["num_prune_steps"] > 0
        or n10_membership_summary["num_prune_steps"] > 0
    )
    downstream_mechanism_supported = (
        cross_run_summary["first_cross_run_inner_difference_step"] is None
        and cross_run_summary["first_cross_run_sensitive_force_difference_step"] is not None
    )

    summary = {
        "milestone": "TP1.5c",
        "tp1_5b_reference_metrics": reference_metrics,
        "rerun_metrics": {
            run_id: {
                "total_energy_range_kj": results_by_id[run_id]["total_energy_range_kj"],
                "max_abs_total_energy_drift_kj": results_by_id[run_id]["max_abs_total_energy_drift_kj"],
                "range_ratio_vs_tp1_5b": results_by_id[run_id]["total_energy_range_kj"]
                / reference_metrics[run_id]["total_energy_range_kj"],
                "drift_ratio_vs_tp1_5b": results_by_id[run_id]["max_abs_total_energy_drift_kj"]
                / reference_metrics[run_id]["max_abs_total_energy_drift_kj"],
            }
            for run_id in results_by_id
        },
        "n1_membership_summary": n1_membership_summary,
        "n10_membership_summary": n10_membership_summary,
        "cross_run_summary": cross_run_summary,
        "mechanism_assessment": {
            "pairlist_lifetime_or_stale_membership_supported": pairlist_lifetime_supported,
            "pruning_supported": pruning_supported,
            "downstream_kernel_accumulation_supported": downstream_mechanism_supported,
            "interpretation": (
                "Pairlist lifetime / stale membership strengthened; pruning weakened"
                if pairlist_lifetime_supported and not pruning_supported and not downstream_mechanism_supported
                else "Pairlist-side mechanism strengthened"
                if pairlist_lifetime_supported and not downstream_mechanism_supported
                else "Deeper kernel accumulation remains live"
                if downstream_mechanism_supported and not pairlist_lifetime_supported
                else "Still unresolved"
            ),
        },
    }
    write_text(RESULTS_DIR / "membership_vs_force_summary.json", json.dumps(summary, indent=2) + "\n")

    suspicion_ranking = {
        "milestone": "TP1.5c",
        "ranked_suspicions": [
            {
                "rank": 1,
                "topic": "pairlist_lifetime_or_stale_membership",
                "status": "confirmed_path_issue" if pairlist_lifetime_supported else "not_confirmed",
                "evidence": {
                    "first_cross_run_inner_difference_step": cross_run_summary["first_cross_run_inner_difference_step"],
                    "n1_sensitive_pair_inner_steps": n1_membership_summary["sensitive_pair_inner_steps"],
                    "n10_sensitive_pair_inner_steps": n10_membership_summary["sensitive_pair_inner_steps"],
                },
            },
            {
                "rank": 2,
                "topic": "pruning_inside_existing_outer_list",
                "status": "weakened" if not pruning_supported else "live_alternative",
                "evidence": {
                    "n1_steps_with_outer_inner_difference": n1_membership_summary["steps_with_outer_inner_difference"],
                    "n10_steps_with_outer_inner_difference": n10_membership_summary["steps_with_outer_inner_difference"],
                    "n1_num_prune_steps": n1_membership_summary["num_prune_steps"],
                    "n10_num_prune_steps": n10_membership_summary["num_prune_steps"],
                },
            },
            {
                "rank": 3,
                "topic": "downstream_kernel_accumulation_with_identical_membership",
                "status": "live_alternative" if not pairlist_lifetime_supported else "weakened",
                "evidence": {
                    "first_cross_run_sensitive_force_difference_step": cross_run_summary[
                        "first_cross_run_sensitive_force_difference_step"
                    ],
                    "first_cross_run_inner_difference_step": cross_run_summary["first_cross_run_inner_difference_step"],
                },
            },
            {
                "rank": 4,
                "topic": "listed_vs_nonlisted_routing",
                "status": "deferred_from_tp1_5b",
                "evidence": "TP1.5c stays on dense_nonlisted only and does not reopen routed-sister closure work.",
            },
        ],
    }
    write_text(RESULTS_DIR / "tp1_5c_suspicion_ranking.json", json.dumps(suspicion_ranking, indent=2) + "\n")

    write_text(RESULTS_DIR / "raw_commands.txt", "\n".join(commands) + "\n")
    write_text(
        RESULTS_DIR / "provenance_manifest.json",
        json.dumps(
            {
                "milestone": "TP1.5c",
                "execution_time_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit_hash": git_output(["rev-parse", "HEAD"]).strip(),
                "git_status_summary": git_output(["status", "--short"]).splitlines(),
                "build_version": gmx_version_text().splitlines(),
                "commands_run": commands,
                "fixture_reference": str((RESULTS_DIR / "dense_fixture_reference.json").relative_to(ROOT)),
                "output_artifacts": [
                    "tests/reference_results/tp1_5c_pairlist_trace_audit/dense_fixture_reference.json",
                    "tests/reference_results/tp1_5c_pairlist_trace_audit/per_step_pair_membership_n1.csv",
                    "tests/reference_results/tp1_5c_pairlist_trace_audit/per_step_pair_membership_n10.csv",
                    "tests/reference_results/tp1_5c_pairlist_trace_audit/per_step_force_trace_n1.csv",
                    "tests/reference_results/tp1_5c_pairlist_trace_audit/per_step_force_trace_n10.csv",
                    "tests/reference_results/tp1_5c_pairlist_trace_audit/membership_vs_force_summary.json",
                    "tests/reference_results/tp1_5c_pairlist_trace_audit/tp1_5c_suspicion_ranking.json",
                ],
                "run_regime": "Current dirty-tree rerun with env-var-gated TP1.5c tracing",
            },
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
