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
RESULTS_DIR = ROOT / "tests/reference_results/tp1_5e_pairlist_contract_audit"

TP15B_FIXTURE = ROOT / "tests/reference_results/tp1_5b_dense_cutoff_audit/dense_fixture_definition.json"
TP15B_SWEEP = ROOT / "tests/reference_results/tp1_5b_dense_cutoff_audit/pairlist_sweep_results.csv"
TP15C_SUMMARY = ROOT / "tests/reference_results/tp1_5c_pairlist_trace_audit/membership_vs_force_summary.json"
TP15D_SUMMARY = ROOT / "tests/reference_results/tp1_5d_pairlist_branch_audit/pair_1_4_decision_summary.json"

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
TRACE_STEP_RANGE = {"start": 160, "end": 180}
CUTOFF_NM = 0.9

OFFICIAL_SOURCES = [
    {
        "site": "GROMACS Manual 2024.4",
        "title": "Mdp options",
        "url": "https://manual.gromacs.org/documentation/2024.4/user-guide/mdp-options.html",
        "checked_date": "2026-03-19",
        "relevance": "Default Verlet buffering is controlled by verlet-buffer-tolerance; setting it to -1 means rlist must be set manually.",
    },
    {
        "site": "GROMACS Manual 2024.4",
        "title": "2024.4 Release Notes",
        "url": "https://manual.gromacs.org/2024.4/release-notes/2024/2024.4.html",
        "checked_date": "2026-03-19",
        "relevance": "Manual Verlet buffer settings disable dual-list dynamic pruning.",
    },
]


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


def git_output(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout


def gmx_version_text() -> str:
    return subprocess.run([str(GMX), "--version"], cwd=ROOT, text=True, capture_output=True, check=True).stdout


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


def gro_text() -> str:
    lines = ["tp1_5e_dense_nonlisted", f"{len(DENSE_COORDS)}"]
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
tp1_5e_dense_nonlisted

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


def parse_trace_csv(path: pathlib.Path) -> list[dict[str, object]]:
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


def copy_text(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_tp15b_sweep() -> dict[str, dict[str, str]]:
    rows = {}
    with TP15B_SWEEP.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["run_id"]] = row
    return rows


def run_case(run: dict[str, object], raw_commands: list[str]) -> dict[str, object]:
    run_dir = WORK_DIR / str(run["run_id"])
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    write_text(run_dir / "system.gro", gro_text())
    write_text(run_dir / "system.top", topology_text())
    write_text(run_dir / "test.mdp", dynamic_mdp_text(run))

    grompp_log = RESULTS_DIR / f"raw_{run['suffix']}_grompp.log"
    mdrun_log = RESULTS_DIR / f"raw_{run['suffix']}_mdrun.log"
    md_log_copy = RESULTS_DIR / f"raw_{run['suffix']}_md.log"
    debug_log = RESULTS_DIR / f"raw_debug_{run['suffix']}.log"

    for path in [grompp_log, mdrun_log, md_log_copy, debug_log]:
        if path.exists():
            path.unlink()

    grompp_cmd = [str(GMX), "grompp", "-f", "test.mdp", "-c", "system.gro", "-p", "system.top", "-o", "topol.tpr", "-maxwarn", "10"]
    raw_commands.append(command_to_string(grompp_cmd, run_dir))
    run_command(grompp_cmd, run_dir, grompp_log, f"{run['run_id']} grompp")

    trace_env = {
        "GMX_TP15D_BRANCH_TRACE_PATH": str(debug_log),
        "GMX_TP15D_PAIR_I": str(TRACE_PAIR["atom_i"]),
        "GMX_TP15D_PAIR_J": str(TRACE_PAIR["atom_j"]),
        "GMX_TP15D_SHIFT_INDEX": str(TRACE_PAIR["shift_index"]),
        "GMX_TP15D_STEP_START": str(TRACE_STEP_RANGE["start"]),
        "GMX_TP15D_STEP_END": str(TRACE_STEP_RANGE["end"]),
    }
    mdrun_cmd = [str(GMX), "mdrun", "-s", "topol.tpr", "-deffnm", "run", "-nt", "1"]
    raw_commands.append(command_to_string(mdrun_cmd, run_dir, env=trace_env))
    run_command(mdrun_cmd, run_dir, mdrun_log, f"{run['run_id']} mdrun", env=trace_env)
    copy_text(run_dir / "run.log", md_log_copy)

    rows = parse_trace_csv(debug_log)
    runtime_text = md_log_copy.read_text(encoding="utf-8", errors="replace").splitlines()
    pairlist_line = next((line.strip() for line in runtime_text if "updated every" in line), None)
    kernel_line = next((line.strip() for line in runtime_text if "Using plain-C-4x4 4x4 nonbonded short-range kernels" in line), None)
    repulsion_line = next((line.strip() for line in runtime_text if "Detected LJ repulsion power 9." in line), None)

    return {
        "run_id": run["run_id"],
        "suffix": run["suffix"],
        "nstlist": int(run["nstlist"]),
        "rlist": float(run["rlist"]),
        "verlet_buffer_tolerance": float(run["verlet_buffer_tolerance"]),
        "rows": rows,
        "raw_debug_path": str(debug_log.relative_to(ROOT)),
        "raw_mdrun_log_path": str(mdrun_log.relative_to(ROOT)),
        "raw_md_log_path": str(md_log_copy.relative_to(ROOT)),
        "runtime_pairlist_line": pairlist_line,
        "runtime_kernel_line": kernel_line,
        "runtime_repulsion_line": repulsion_line,
    }


def make_rebuild_history_rows(run_result: dict[str, object]) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for row in run_result["rows"]:
        distance = float(row["target_shift_distance_nm"])
        rlist_outer = float(row["rlist_outer_nm"])
        rows_out.append(
            {
                "run_id": run_result["run_id"],
                "step": int(row["step"]),
                "rebuild_this_step": row["rebuild_this_step"],
                "pairlist_age": int(row["pairlist_age"]),
                "dynamic_pruning_enabled": row["dynamic_pruning_enabled"],
                "prune_step": row["prune_step"],
                "rlist_outer_nm": rlist_outer,
                "buffer_nm": rlist_outer - CUTOFF_NM,
                "target_shift_index": int(row["target_shift_index"]),
                "geometry_atom_i": int(row["geometry_atom_i"]),
                "geometry_atom_j": int(row["geometry_atom_j"]),
                "target_shift_distance_nm": distance,
                "distance_minus_cutoff_nm": distance - CUTOFF_NM,
                "distance_minus_rlist_nm": distance - rlist_outer,
                "pair_in_outer_active": row["pair_in_outer_active"],
                "pair_in_inner_active": row["pair_in_inner_active"],
            }
        )
    return rows_out


def build_margin_analysis(
    n1_result: dict[str, object], n10_result: dict[str, object], tp15b_sweep: dict[str, dict[str, str]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    key_steps = [160, 170, 171, 172, 173, 180]
    for run_result in [n1_result, n10_result]:
        by_step = {int(row["step"]): row for row in run_result["rows"]}
        for step in key_steps:
            row = by_step[step]
            distance = float(row["target_shift_distance_nm"])
            rlist_outer = float(row["rlist_outer_nm"])
            rows.append(
                {
                    "source": "tp1_5e_rerun",
                    "run_id": run_result["run_id"],
                    "step": step,
                    "nstlist": run_result["nstlist"],
                    "rlist_outer_nm": rlist_outer,
                    "cutoff_nm": CUTOFF_NM,
                    "buffer_nm": rlist_outer - CUTOFF_NM,
                    "target_shift_distance_nm": distance,
                    "distance_minus_cutoff_nm": distance - CUTOFF_NM,
                    "distance_minus_rlist_nm": distance - rlist_outer,
                    "rebuild_this_step": row["rebuild_this_step"],
                    "pair_in_outer_active": row["pair_in_outer_active"],
                    "pair_in_inner_active": row["pair_in_inner_active"],
                }
            )

    auto_row = tp15b_sweep["auto_buffer_n10_vbt0005"]
    auto_rlist = float(auto_row["runtime_pairlist_line"].split("rlist ")[1].split(" nm")[0])
    baseline_step = next(row for row in n10_result["rows"] if int(row["step"]) == 170)
    baseline_distance = float(baseline_step["target_shift_distance_nm"])
    rows.append(
        {
            "source": "tp1_5b_reference",
            "run_id": "auto_buffer_n10_vbt0005",
            "step": 170,
            "nstlist": 10,
            "rlist_outer_nm": auto_rlist,
            "cutoff_nm": CUTOFF_NM,
            "buffer_nm": auto_rlist - CUTOFF_NM,
            "target_shift_distance_nm": baseline_distance,
            "distance_minus_cutoff_nm": baseline_distance - CUTOFF_NM,
            "distance_minus_rlist_nm": baseline_distance - auto_rlist,
            "rebuild_this_step": True,
            "pair_in_outer_active": baseline_distance <= auto_rlist,
            "pair_in_inner_active": baseline_distance <= auto_rlist,
        }
    )
    return rows


def build_contract_path_map() -> dict[str, object]:
    return {
        "milestone": "TP1.5e",
        "path_map": [
            {
                "source_type": "code",
                "file": "src/gromacs/mdlib/sim_util.cpp",
                "function": "doPairSearch",
                "contract_role": "Pairlist rebuild scheduling on search steps.",
                "why_relevant": "Determines when a reused outer list is rebuilt versus carried forward.",
                "evidence_strength": "confirmed_runtime_entry_point",
            },
            {
                "source_type": "code",
                "file": "src/gromacs/nbnxm/pairlistsets.h",
                "function": "PairlistSets::numStepsWithPairlist / isDynamicPruningStepCpu",
                "contract_role": "Defines pairlist age and CPU dynamic-pruning cadence.",
                "why_relevant": "Separates plain list reuse from pruning behavior.",
                "evidence_strength": "confirmed_runtime_state_accessor",
            },
            {
                "source_type": "code",
                "file": "src/gromacs/nbnxm/pairlist_tuning.cpp",
                "function": "supportsDynamicPairlistGenerationInterval",
                "contract_role": "Enables dynamic pairlist tuning only when verletbuf_tol > 0.",
                "why_relevant": "With verlet-buffer-tolerance = -1, dynamic pruning contract is disabled.",
                "evidence_strength": "confirmed_static_contract",
            },
            {
                "source_type": "code",
                "file": "src/gromacs/nbnxm/pairlist_tuning.cpp",
                "function": "setupDynamicPairlistPruning",
                "contract_role": "Configures dual-list pruning and reports outer/inner buffers.",
                "why_relevant": "Manual buffer mode bypasses this dynamic pruning path.",
                "evidence_strength": "confirmed_static_contract",
            },
            {
                "source_type": "code",
                "file": "src/gromacs/mdlib/calc_verletbuf.h",
                "function": "calcVerletBufferSize",
                "contract_role": "Defines positive verlet-buffer-tolerance as a bound on average energy jump over list lifetime.",
                "why_relevant": "Explains what guarantee exists in auto-buffer mode and what is absent in manual mode.",
                "evidence_strength": "confirmed_static_contract",
            },
            {
                "source_type": "code",
                "file": "src/gromacs/nbnxm/nbnxm_setup.cpp",
                "function": "nonbonded_verlet_t setup",
                "contract_role": "Calls setupDynamicPairlistPruning only for non-PlainC1x1 kernels.",
                "why_relevant": "Shows the audited plain-C cut-off path still goes through pairlist-tuning contract setup.",
                "evidence_strength": "confirmed_static_localization",
            },
            {
                "source_type": "doc",
                "file": "docs/release-notes/2024/2024.4.rst",
                "section": "Fix missing non-bonded interactions close to cut-off with GPUs",
                "contract_role": "Manual Verlet buffer settings disable dual-list dynamic pruning.",
                "why_relevant": "Supports that manually set buffers fall outside the automatic dual-list safeguard path.",
                "evidence_strength": "supporting_doc",
            },
        ],
        "official_sources": OFFICIAL_SOURCES,
    }


def build_verdict(
    n1_result: dict[str, object], n10_result: dict[str, object], tp15b_sweep: dict[str, dict[str, str]]
) -> dict[str, object]:
    n10_by_step = {int(row["step"]): row for row in n10_result["rows"]}
    n1_by_step = {int(row["step"]): row for row in n1_result["rows"]}

    step_170 = n10_by_step[170]
    step_171 = n10_by_step[171]
    step_172 = n10_by_step[172]
    step_180 = n10_by_step[180]

    distance_170 = float(step_170["target_shift_distance_nm"])
    distance_171 = float(step_171["target_shift_distance_nm"])
    distance_172 = float(step_172["target_shift_distance_nm"])
    rlist_outer = float(step_170["rlist_outer_nm"])
    buffer_nm = rlist_outer - CUTOFF_NM
    auto_rlist = float(tp15b_sweep["auto_buffer_n10_vbt0005"]["runtime_pairlist_line"].split("rlist ")[1].split(" nm")[0])

    first_below_rlist = next(int(row["step"]) for row in n10_result["rows"] if float(row["target_shift_distance_nm"]) <= rlist_outer)
    first_below_cutoff = next(int(row["step"]) for row in n10_result["rows"] if float(row["target_shift_distance_nm"]) <= CUTOFF_NM)

    classification = "ALLOWED-UNSAFE"
    justification = (
        "The pair is outside the manually requested outer list at the last rebuild (step 170) by "
        f"{distance_170 - rlist_outer:.9f} nm, so omission on reused steps is allowed by the current manual-rlist contract. "
        "Auto-buffered TP1.5b settings choose rlist ≈ 0.911 nm and remove the instability on the same fixture."
    )

    return {
        "milestone": "TP1.5e",
        "classification": classification,
        "patching_now_justified": False,
        "strongest_confirmed_contract_fact": {
            "n10_last_rebuild_before_step_171": 170,
            "n10_next_rebuild_after_step_171": 180,
            "distance_at_last_rebuild_nm": distance_170,
            "manual_rlist_outer_nm": rlist_outer,
            "manual_buffer_nm": buffer_nm,
            "distance_minus_rlist_at_last_rebuild_nm": distance_170 - rlist_outer,
            "first_step_below_rlist": first_below_rlist,
            "distance_at_first_step_below_rlist_nm": distance_171,
            "first_step_below_cutoff": first_below_cutoff,
            "distance_at_first_step_below_cutoff_nm": distance_172,
        },
        "observed_branch_facts": {
            "n1_rebuild_step_170": bool(n1_by_step[170]["rebuild_this_step"]),
            "n1_rebuild_step_171": bool(n1_by_step[171]["rebuild_this_step"]),
            "n10_rebuild_step_170": bool(step_170["rebuild_this_step"]),
            "n10_rebuild_step_171": bool(step_171["rebuild_this_step"]),
            "n10_pair_present_step_170": bool(step_170["pair_in_outer_active"]),
            "n10_pair_present_step_171": bool(step_171["pair_in_outer_active"]),
            "n10_pair_present_step_172": bool(step_172["pair_in_outer_active"]),
            "n10_pair_present_step_180": bool(step_180["pair_in_outer_active"]),
            "dynamic_pruning_enabled_n1": any(bool(row["dynamic_pruning_enabled"]) for row in n1_result["rows"]),
            "dynamic_pruning_enabled_n10": any(bool(row["dynamic_pruning_enabled"]) for row in n10_result["rows"]),
        },
        "contract_interpretation": {
            "manual_rlist_mode": "With verlet-buffer-tolerance = -1, rlist is user-set rather than auto-sized from the buffer-tolerance contract.",
            "dynamic_pruning_status": "Dynamic pruning is not active in this manual-buffer run.",
            "guarantee_assessment": "The current settings do not guarantee that pairs outside rlist at the last rebuild remain captured before the next rebuild.",
            "justification": justification,
        },
        "comparison_to_reference_runs": {
            "tp1_5b_n10_r0909_total_energy_range_kj": float(tp15b_sweep["n10_r0909"]["total_energy_range_kj"]),
            "tp1_5b_n1_r0909_total_energy_range_kj": float(tp15b_sweep["n1_r0909"]["total_energy_range_kj"]),
            "tp1_5b_auto_buffer_n10_vbt0005_rlist_nm": auto_rlist,
            "tp1_5b_auto_buffer_n10_vbt0005_total_energy_range_kj": float(tp15b_sweep["auto_buffer_n10_vbt0005"]["total_energy_range_kj"]),
            "auto_buffer_exceeds_step_170_distance_nm": auto_rlist - distance_170,
        },
        "official_sources_checked": OFFICIAL_SOURCES,
        "remaining_uncertainty": "TP1.5e does not prove whether the chosen manual buffer is physically wise for the larger TP1.3 system; it only shows this omission is contract-compliant under the current manual settings.",
        "exact_next_step_recommendation": "Do not patch production pairlist logic yet. For TP1.6, either move this fixture to auto-buffered settings when testing physics, or only patch if a later milestone proves that manual-rlist mode is required and should still guarantee stronger capture semantics.",
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    raw_commands: list[str] = []
    raw_commands.append(command_to_string(["git", "status", "--short"], ROOT))
    raw_commands.append(command_to_string(["git", "rev-parse", "HEAD"], ROOT))
    raw_commands.append(command_to_string([str(GMX), "--version"], ROOT))

    git_status_summary = git_output(["status", "--short"]).splitlines()
    git_commit_hash = git_output(["rev-parse", "HEAD"]).strip()
    build_version = gmx_version_text().splitlines()

    tp15b_sweep = read_tp15b_sweep()

    run_results = []
    for run in RUNS:
        run_results.append(run_case(run, raw_commands))

    n1_result = next(result for result in run_results if result["run_id"] == "n1_r0909")
    n10_result = next(result for result in run_results if result["run_id"] == "n10_r0909")

    rebuild_history_n1 = make_rebuild_history_rows(n1_result)
    rebuild_history_n10 = make_rebuild_history_rows(n10_result)
    margin_analysis = build_margin_analysis(n1_result, n10_result, tp15b_sweep)
    contract_path_map = build_contract_path_map()
    verdict = build_verdict(n1_result, n10_result, tp15b_sweep)

    write_csv(
        RESULTS_DIR / "rebuild_history_n1.csv",
        list(rebuild_history_n1[0].keys()),
        rebuild_history_n1,
    )
    write_csv(
        RESULTS_DIR / "rebuild_history_n10.csv",
        list(rebuild_history_n10[0].keys()),
        rebuild_history_n10,
    )
    write_csv(
        RESULTS_DIR / "pair_1_4_margin_analysis.csv",
        list(margin_analysis[0].keys()),
        margin_analysis,
    )
    write_text(RESULTS_DIR / "contract_path_map.json", json.dumps(contract_path_map, indent=2) + "\n")
    write_text(RESULTS_DIR / "tp1_5e_contract_verdict.json", json.dumps(verdict, indent=2) + "\n")
    write_text(RESULTS_DIR / "raw_commands.txt", "\n".join(raw_commands) + "\n")

    provenance_manifest = {
        "milestone": "TP1.5e",
        "execution_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_commit_hash,
        "git_status_summary": git_status_summary,
        "build_version": build_version,
        "fixture_reference": {
            "tp1_5b_fixture_definition": str(TP15B_FIXTURE.relative_to(ROOT)),
            "tp1_5c_membership_summary": str(TP15C_SUMMARY.relative_to(ROOT)),
            "tp1_5d_decision_summary": str(TP15D_SUMMARY.relative_to(ROOT)),
            "primary_fixture_id": "dense_nonlisted",
        },
        "trace_pair": TRACE_PAIR,
        "trace_step_range": TRACE_STEP_RANGE,
        "commands_run": raw_commands,
        "output_artifacts": [
            "tests/reference_results/tp1_5e_pairlist_contract_audit/rebuild_history_n1.csv",
            "tests/reference_results/tp1_5e_pairlist_contract_audit/rebuild_history_n10.csv",
            "tests/reference_results/tp1_5e_pairlist_contract_audit/pair_1_4_margin_analysis.csv",
            "tests/reference_results/tp1_5e_pairlist_contract_audit/contract_path_map.json",
            "tests/reference_results/tp1_5e_pairlist_contract_audit/tp1_5e_contract_verdict.json",
            "tests/reference_results/tp1_5e_pairlist_contract_audit/raw_commands.txt",
        ],
        "official_sources_checked": OFFICIAL_SOURCES,
    }
    write_text(RESULTS_DIR / "provenance_manifest.json", json.dumps(provenance_manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
