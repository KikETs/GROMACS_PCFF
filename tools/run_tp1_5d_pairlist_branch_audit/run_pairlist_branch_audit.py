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
RESULTS_DIR = ROOT / "tests/reference_results/tp1_5d_pairlist_branch_audit"

TP15B_FIXTURE = ROOT / "tests/reference_results/tp1_5b_dense_cutoff_audit/dense_fixture_definition.json"
TP15C_SUMMARY = ROOT / "tests/reference_results/tp1_5c_pairlist_trace_audit/membership_vs_force_summary.json"
TP15C_RUNTIME_PATH = ROOT / "tests/reference_results/tp1_5_cutoff_audit/cutoff_path_trace.json"

DENSE_COORDS = [
    ("A1", 0.400, 1.000, 1.000),
    ("A2", 0.740, 1.000, 1.000),
    ("A3", 1.645, 1.000, 1.000),
    ("A4", 1.985, 1.000, 1.000),
]
BOX = (2.500, 2.500, 2.500)

RUNS = [
    {"run_id": "n1_r0909", "nstlist": 1, "rlist": 0.909, "verlet_buffer_tolerance": -1, "suffix": "n1"},
    {"run_id": "n10_r0909", "nstlist": 10, "rlist": 0.909, "verlet_buffer_tolerance": -1, "suffix": "n10"},
]

TRACE_PAIR = {"atom_i": 1, "atom_j": 4, "shift_index": 21}
TRACE_STEP_RANGE = {"start": 168, "end": 173}


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
    effective_env = None if env is None else {**os.environ, **env}
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
    lines = ["tp1_5d_dense_nonlisted", f"{len(DENSE_COORDS)}"]
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
tp1_5d_dense_nonlisted

[ molecules ]
SYS 1
"""


def dynamic_mdp_text(run: dict[str, object]) -> str:
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
            "nstfout = 0",
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


def parse_branch_trace(path: pathlib.Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "step": int(row["step"]),
                    "locality": row["locality"],
                    "rebuild_this_step": bool(int(row["rebuild_this_step"])),
                    "pairlist_age": int(row["pairlist_age"]),
                    "dynamic_pruning_enabled": bool(int(row["dynamic_pruning_enabled"])),
                    "prune_step": bool(int(row["prune_step"])),
                    "rlist_outer_nm": float(row["rlist_outer_nm"]),
                    "rlist_inner_nm": float(row["rlist_inner_nm"]),
                    "pair_atom_i": int(row["pair_atom_i"]),
                    "pair_atom_j": int(row["pair_atom_j"]),
                    "target_shift_index": int(row["target_shift_index"]),
                    "geometry_atom_i": int(row["geometry_atom_i"]),
                    "geometry_atom_j": int(row["geometry_atom_j"]),
                    "target_shift_dx": float(row["target_shift_dx"]),
                    "target_shift_dy": float(row["target_shift_dy"]),
                    "target_shift_dz": float(row["target_shift_dz"]),
                    "target_shift_distance_nm": float(row["target_shift_distance_nm"]),
                    "min_shift_index": int(row["min_shift_index"]),
                    "min_distance_nm": float(row["min_distance_nm"]),
                    "target_shift_is_minimum": bool(int(row["target_shift_is_minimum"])),
                    "pair_in_outer_active": bool(int(row["pair_in_outer_active"])),
                    "pair_in_outer_excluded": bool(int(row["pair_in_outer_excluded"])),
                    "pair_in_inner_active": bool(int(row["pair_in_inner_active"])),
                    "pair_in_inner_excluded": bool(int(row["pair_in_inner_excluded"])),
                }
            )
    return rows


def write_branch_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "step",
        "locality",
        "rebuild_this_step",
        "pairlist_age",
        "dynamic_pruning_enabled",
        "prune_step",
        "rlist_outer_nm",
        "rlist_inner_nm",
        "pair_atom_i",
        "pair_atom_j",
        "target_shift_index",
        "geometry_atom_i",
        "geometry_atom_j",
        "target_shift_dx",
        "target_shift_dy",
        "target_shift_dz",
        "target_shift_distance_nm",
        "min_shift_index",
        "min_distance_nm",
        "target_shift_is_minimum",
        "pair_in_outer_active",
        "pair_in_outer_excluded",
        "pair_in_inner_active",
        "pair_in_inner_excluded",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rows_by_step(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    return {int(row["step"]): row for row in rows}


def runtime_lines(run_log: pathlib.Path) -> dict[str, str | None]:
    text = run_log.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "kernel_line": next(
            (line.strip() for line in text if "Using plain-C-4x4 4x4 nonbonded short-range kernels" in line), None
        ),
        "repulsion_line": next((line.strip() for line in text if "Detected LJ repulsion power 9." in line), None),
        "pairlist_line": next((line.strip() for line in text if "updated every" in line), None),
    }


def summarize_decision(
    n1_rows: list[dict[str, object]], n10_rows: list[dict[str, object]], run_info: dict[str, dict[str, object]]
) -> dict[str, object]:
    n1_by_step = rows_by_step(n1_rows)
    n10_by_step = rows_by_step(n10_rows)
    common_steps = sorted(set(n1_by_step) & set(n10_by_step))

    first_presence_divergence_step = None
    first_presence_divergence_detail = None
    first_target_distance_below_cutoff_step = None

    for step in common_steps:
        n1_row = n1_by_step[step]
        n10_row = n10_by_step[step]
        if (
            n1_row["pair_in_inner_active"] != n10_row["pair_in_inner_active"]
            or n1_row["pair_in_outer_active"] != n10_row["pair_in_outer_active"]
        ) and first_presence_divergence_step is None:
            first_presence_divergence_step = step
            first_presence_divergence_detail = {"n1": n1_row, "n10": n10_row}
        if n1_row["target_shift_distance_nm"] < n1_row["rlist_outer_nm"] and first_target_distance_below_cutoff_step is None:
            first_target_distance_below_cutoff_step = step

    any_dynamic_pruning = any(bool(row["dynamic_pruning_enabled"]) for row in n1_rows + n10_rows)
    any_prune_steps = any(bool(row["prune_step"]) for row in n1_rows + n10_rows)

    rebuild_cadence_supported = (
        first_presence_divergence_step is not None
        and bool(n1_by_step[first_presence_divergence_step]["rebuild_this_step"])
        and not bool(n10_by_step[first_presence_divergence_step]["rebuild_this_step"])
    )
    shift_selection_supported = (
        first_presence_divergence_step is not None
        and bool(n1_by_step[first_presence_divergence_step]["target_shift_is_minimum"])
        and bool(n10_by_step[first_presence_divergence_step]["target_shift_is_minimum"])
    )
    previous_step = None if first_presence_divergence_step is None else first_presence_divergence_step - 1
    cutoff_decision_supported = (
        first_presence_divergence_step is not None
        and previous_step in n1_by_step
        and first_target_distance_below_cutoff_step == first_presence_divergence_step
        and n1_by_step[previous_step]["target_shift_distance_nm"] >= n1_by_step[previous_step]["rlist_outer_nm"]
        and n1_by_step[first_presence_divergence_step]["target_shift_distance_nm"]
        < n1_by_step[first_presence_divergence_step]["rlist_outer_nm"]
    )

    interpretation = (
        "Rebuild cadence is the confirmed branch-level difference; refresh/pruning is not exercised here."
        if rebuild_cadence_supported and not any_dynamic_pruning and not any_prune_steps
        else "Branch-level difference narrowed, but multiple branches remain live."
    )

    return {
        "milestone": "TP1.5d",
        "trace_pair": TRACE_PAIR,
        "trace_step_range": TRACE_STEP_RANGE,
        "tp1_5c_reference": {
            "source_fixture": str(TP15B_FIXTURE.relative_to(ROOT)),
            "source_membership_summary": str(TP15C_SUMMARY.relative_to(ROOT)),
            "source_runtime_path_trace": str(TP15C_RUNTIME_PATH.relative_to(ROOT)),
        },
        "run_runtime_info": run_info,
        "common_steps": common_steps,
        "n1_rows": n1_rows,
        "n10_rows": n10_rows,
        "first_presence_divergence_step": first_presence_divergence_step,
        "first_presence_divergence_detail": first_presence_divergence_detail,
        "first_target_distance_below_cutoff_step": first_target_distance_below_cutoff_step,
        "mechanism_assessment": {
            "rebuild_cadence_supported": rebuild_cadence_supported,
            "refresh_or_pruning_supported": any_dynamic_pruning or any_prune_steps,
            "shift_selection_supported": shift_selection_supported,
            "cutoff_threshold_crossing_supported": cutoff_decision_supported,
            "interpretation": interpretation,
        },
    }


def run_case(run: dict[str, object], commands: list[str]) -> dict[str, object]:
    run_id = str(run["run_id"])
    suffix = str(run["suffix"])
    work_dir = WORK_DIR / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    write_text(work_dir / "system.top", topology_text())
    write_text(work_dir / "system.gro", gro_text())
    write_text(work_dir / "test.mdp", dynamic_mdp_text(run))

    raw_debug_path = RESULTS_DIR / f"raw_debug_{suffix}.log"
    if raw_debug_path.exists():
        raw_debug_path.unlink()

    trace_env = {
        "GMX_TP15D_BRANCH_TRACE_PATH": str(raw_debug_path),
        "GMX_TP15D_PAIR_I": str(TRACE_PAIR["atom_i"]),
        "GMX_TP15D_PAIR_J": str(TRACE_PAIR["atom_j"]),
        "GMX_TP15D_SHIFT_INDEX": str(TRACE_PAIR["shift_index"]),
        "GMX_TP15D_STEP_START": str(TRACE_STEP_RANGE["start"]),
        "GMX_TP15D_STEP_END": str(TRACE_STEP_RANGE["end"]),
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

    if not raw_debug_path.exists():
        raise FileNotFoundError(f"Expected TP1.5d debug trace at {raw_debug_path}")

    branch_rows = parse_branch_trace(raw_debug_path)
    branch_csv = RESULTS_DIR / f"branch_trace_{suffix}.csv"
    write_branch_csv(branch_csv, branch_rows)

    run_runtime = runtime_lines(work_dir / "run.log")
    return {
        "run_id": run_id,
        "suffix": suffix,
        "nstlist": int(run["nstlist"]),
        "rlist": float(run["rlist"]),
        "verlet_buffer_tolerance": float(run["verlet_buffer_tolerance"]),
        "raw_debug_path": str(raw_debug_path.relative_to(ROOT)),
        "branch_trace_path": str(branch_csv.relative_to(ROOT)),
        **run_runtime,
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
    tp15c_reference = json.loads(TP15C_SUMMARY.read_text(encoding="utf-8"))

    commands = [
        "(cd /home/kiket/바탕화면/test/GROMACS_PCFF && git status --short)",
        "(cd /home/kiket/바탕화면/test/GROMACS_PCFF && git rev-parse HEAD)",
        "(cd /home/kiket/바탕화면/test/GROMACS_PCFF && build/bin/gmx --version)",
    ]

    run_results = [run_case(run, commands) for run in RUNS]
    run_info = {entry["run_id"]: entry for entry in run_results}

    n1_rows = parse_branch_trace(RESULTS_DIR / "raw_debug_n1.log")
    n10_rows = parse_branch_trace(RESULTS_DIR / "raw_debug_n10.log")
    decision_summary = summarize_decision(n1_rows, n10_rows, run_info)

    write_text(RESULTS_DIR / "pair_1_4_decision_summary.json", json.dumps(decision_summary, indent=2) + "\n")

    source_path_map = {
        "milestone": "TP1.5d",
        "path_map": [
            {
                "file": "src/gromacs/mdlib/sim_util.cpp",
                "function": "doPairSearch",
                "branch_decision_role": "Schedules pairlist rebuilds on search steps and calls constructPairlist with the current step.",
                "why_suspected": "This is where n1_r0909 rebuilds every step while n10_r0909 reuses an older list.",
                "evidence_strength": "confirmed_runtime_entry_point",
            },
            {
                "file": "src/gromacs/nbnxm/pairlistsets.h",
                "function": "PairlistSets::numStepsWithPairlist / isDynamicPruningStepCpu",
                "branch_decision_role": "Defines pairlist age and whether a CPU prune step should occur.",
                "why_suspected": "TP1.5d needs the exact rebuild-age versus prune-step distinction around steps 170-173.",
                "evidence_strength": "confirmed_runtime_state_accessor",
            },
            {
                "file": "src/gromacs/nbnxm/pairlist.cpp",
                "function": "PairlistSets::construct",
                "branch_decision_role": "Stores outerListCreationStep_ after list construction.",
                "why_suspected": "This is the narrowest rebuild-timing state updated by constructPairlist.",
                "evidence_strength": "confirmed_static_localization",
            },
            {
                "file": "src/gromacs/nbnxm/prunekerneldispatch.cpp",
                "function": "PairlistSet::dispatchPruneKernel",
                "branch_decision_role": "Runs refresh/pruning from ciOuter/cjOuter to ci/cj using rlistInner.",
                "why_suspected": "Needed to rule pruning in or out as a live mechanism on the dense_nonlisted rerun.",
                "evidence_strength": "confirmed_static_localization",
            },
            {
                "file": "src/gromacs/nbnxm/pairlist.cpp",
                "function": "nbnxn_make_pairlist_part",
                "branch_decision_role": "Chooses shift indices, half-list skipping, cell ranges, and near-cutoff admission into the fresh outer list.",
                "why_suspected": "Pair (1,4, shift 21) can only appear if this construction path admits that shifted image.",
                "evidence_strength": "confirmed_static_localization",
            },
            {
                "file": "src/gromacs/nbnxm/pairlist.cpp",
                "function": "prepareListsForDynamicPruning",
                "branch_decision_role": "Separates fresh outer lists from active inner lists when dynamic pruning is enabled.",
                "why_suspected": "Needed to interpret whether TP1.5d is seeing rebuild reuse or inner/outer refresh differences.",
                "evidence_strength": "confirmed_static_localization",
            },
        ],
    }
    write_text(RESULTS_DIR / "source_path_map.json", json.dumps(source_path_map, indent=2) + "\n")

    mechanism = decision_summary["mechanism_assessment"]
    suspicion_ranking = {
        "milestone": "TP1.5d",
        "ranked_suspicions": [
            {
                "rank": 1,
                "topic": "rebuild_cadence_with_near_cutoff_crossing",
                "status": "confirmed_branch_difference" if mechanism["rebuild_cadence_supported"] else "not_confirmed",
                "evidence": {
                    "first_presence_divergence_step": decision_summary["first_presence_divergence_step"],
                    "first_target_distance_below_cutoff_step": decision_summary["first_target_distance_below_cutoff_step"],
                },
            },
            {
                "rank": 2,
                "topic": "shift_21_image_selection_during_rebuild",
                "status": "supported_contributor" if mechanism["shift_selection_supported"] else "still_unclear",
                "evidence": {
                    "target_shift_index": TRACE_PAIR["shift_index"],
                    "n1_min_shift_indices": [row["min_shift_index"] for row in n1_rows],
                    "n10_min_shift_indices": [row["min_shift_index"] for row in n10_rows],
                },
            },
            {
                "rank": 3,
                "topic": "refresh_or_pruning_logic",
                "status": "weakened" if not mechanism["refresh_or_pruning_supported"] else "live_alternative",
                "evidence": {
                    "n1_dynamic_pruning_enabled": any(row["dynamic_pruning_enabled"] for row in n1_rows),
                    "n10_dynamic_pruning_enabled": any(row["dynamic_pruning_enabled"] for row in n10_rows),
                    "n1_prune_steps": [row["step"] for row in n1_rows if row["prune_step"]],
                    "n10_prune_steps": [row["step"] for row in n10_rows if row["prune_step"]],
                },
            },
            {
                "rank": 4,
                "topic": "source_level_bug_requiring_patch",
                "status": "deferred" if mechanism["rebuild_cadence_supported"] else "still_unclear",
                "evidence": "TP1.5d traces the branch-level difference but does not yet prove an incorrect source-level decision.",
            },
        ],
    }
    write_text(RESULTS_DIR / "tp1_5d_suspicion_ranking.json", json.dumps(suspicion_ranking, indent=2) + "\n")

    write_text(RESULTS_DIR / "raw_commands.txt", "\n".join(commands) + "\n")
    write_text(
        RESULTS_DIR / "provenance_manifest.json",
        json.dumps(
            {
                "milestone": "TP1.5d",
                "execution_time_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit_hash": git_output(["rev-parse", "HEAD"]).strip(),
                "git_status_summary": git_output(["status", "--short"]).splitlines(),
                "build_version": gmx_version_text().splitlines(),
                "commands_run": commands,
                "fixture_reference": {
                    "tp1_5b_fixture_definition": str(TP15B_FIXTURE.relative_to(ROOT)),
                    "tp1_5c_membership_summary": str(TP15C_SUMMARY.relative_to(ROOT)),
                    "primary_fixture_id": fixture_reference["primary_fixture"]["fixture_id"],
                },
                "tp1_5c_reference_excerpt": {
                    "first_cross_run_inner_difference_step": tp15c_reference["cross_run_summary"][
                        "first_cross_run_inner_difference_step"
                    ],
                    "first_cross_run_inner_difference_detail": tp15c_reference["cross_run_summary"][
                        "first_cross_run_inner_difference_detail"
                    ],
                },
                "output_artifacts": [
                    "tests/reference_results/tp1_5d_pairlist_branch_audit/branch_trace_n1.csv",
                    "tests/reference_results/tp1_5d_pairlist_branch_audit/branch_trace_n10.csv",
                    "tests/reference_results/tp1_5d_pairlist_branch_audit/pair_1_4_decision_summary.json",
                    "tests/reference_results/tp1_5d_pairlist_branch_audit/source_path_map.json",
                    "tests/reference_results/tp1_5d_pairlist_branch_audit/tp1_5d_suspicion_ranking.json",
                    "tests/reference_results/tp1_5d_pairlist_branch_audit/raw_debug_n1.log",
                    "tests/reference_results/tp1_5d_pairlist_branch_audit/raw_debug_n10.log",
                ],
                "run_regime": "Current dirty-tree rerun with env-var-gated TP1.5d branch tracing",
            },
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
