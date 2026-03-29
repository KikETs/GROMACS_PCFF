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
RESULTS_DIR = ROOT / "tests/reference_results/tp1_6_rebaseline_audit"

TP15B_FIXTURE = ROOT / "tests/reference_results/tp1_5b_dense_cutoff_audit/dense_fixture_definition.json"
TP15B_SWEEP = ROOT / "tests/reference_results/tp1_5b_dense_cutoff_audit/pairlist_sweep_results.csv"
TP15C_SUMMARY = ROOT / "tests/reference_results/tp1_5c_pairlist_trace_audit/membership_vs_force_summary.json"
TP15D_SUMMARY = ROOT / "tests/reference_results/tp1_5d_pairlist_branch_audit/pair_1_4_decision_summary.json"
TP15E_VERDICT = ROOT / "tests/reference_results/tp1_5e_pairlist_contract_audit/tp1_5e_contract_verdict.json"

DENSE_COORDS = [
    ("A1", 0.400, 1.000, 1.000),
    ("A2", 0.740, 1.000, 1.000),
    ("A3", 1.645, 1.000, 1.000),
    ("A4", 1.985, 1.000, 1.000),
]
BOX = (2.500, 2.500, 2.500)

RUNS = [
    {
        "run_id": "tight_ref_n1_r1200",
        "role": "tight_reference",
        "nstlist": 1,
        "rlist": 1.2,
        "verlet_buffer_tolerance": -1,
        "why_safer": "Large manual list margin removes near-cutoff reuse sensitivity and serves as the TP1.5b tight reference.",
        "intended_use": "diagnostic_reference_only",
    },
    {
        "run_id": "n10_r0909",
        "role": "unsafe_reference",
        "nstlist": 10,
        "rlist": 0.909,
        "verlet_buffer_tolerance": -1,
        "why_safer": "Not safe; rerun as the TP1.5b/TP1.5e unsafe reference.",
        "intended_use": "unsafe_reference",
    },
    {
        "run_id": "n1_r0909",
        "role": "safe_candidate",
        "nstlist": 1,
        "rlist": 0.909,
        "verlet_buffer_tolerance": -1,
        "why_safer": "Eliminates list reuse entirely on the same manual rlist, so the critical near-cutoff pair cannot be missed between rebuilds.",
        "intended_use": "diagnostic_control_not_preferred_baseline",
    },
    {
        "run_id": "n10_r0911",
        "role": "safe_candidate",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "why_safer": "Manual rlist exceeds the TP1.5e step-170 critical pair distance and should avoid the specific allowed-unsafe omission while keeping nstlist=10.",
        "intended_use": "secondary_manual_safe_candidate",
    },
    {
        "run_id": "auto_buffer_n10_vbt0005",
        "role": "safe_candidate",
        "nstlist": 10,
        "verlet_buffer_tolerance": 0.005,
        "why_safer": "Positive verlet-buffer-tolerance re-enables automatic safe list sizing and was already known in TP1.5b to recover the tight reference.",
        "intended_use": "preferred_validation_baseline_candidate",
    },
]

KEY_DUMP_RUN_IDS = {"tight_ref_n1_r1200", "n10_r0909", "n10_r0911", "auto_buffer_n10_vbt0005"}


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
    lines = ["tp1_6_dense_nonlisted", f"{len(DENSE_COORDS)}"]
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
tp1_6_dense_nonlisted

[ molecules ]
SYS 1
"""


def dynamic_mdp_text(run: dict[str, object]) -> str:
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


def run_dynamic_case(run: dict[str, object], commands: list[str]) -> dict[str, object]:
    run_id = str(run["run_id"])
    work_dir = WORK_DIR / run_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    write_text(work_dir / "system.top", topology_text())
    write_text(work_dir / "system.gro", gro_text())
    write_text(work_dir / "test.mdp", dynamic_mdp_text(run))

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
    run_command(grompp_cmd, work_dir, RESULTS_DIR / f"raw_{run_id}_grompp.log", f"{run_id} grompp")

    mdrun_cmd = [str(GMX), "mdrun", "-s", "topol.tpr", "-deffnm", "run", "-nt", "1"]
    commands.append(command_to_string(mdrun_cmd, work_dir))
    run_command(mdrun_cmd, work_dir, RESULTS_DIR / f"raw_{run_id}_mdrun.log", f"{run_id} mdrun")
    copy_text(work_dir / "run.log", RESULTS_DIR / f"raw_{run_id}_md.log")

    energy_cmd = [str(GMX), "energy", "-f", "run.edr", "-o", "energy.xvg"]
    energy_stdin = "Temperature\nPotential\nKinetic-En.\nTotal-Energy\n0\n"
    commands.append(command_to_string(energy_cmd, work_dir, energy_stdin))
    run_command(
        energy_cmd,
        work_dir,
        RESULTS_DIR / f"raw_{run_id}_energy_output.txt",
        f"{run_id} energy",
        energy_stdin,
    )
    copy_text(work_dir / "energy.xvg", RESULTS_DIR / f"raw_{run_id}_energy.xvg")

    mdlog_path = work_dir / "run.log"
    energy_series = parse_xvg(work_dir / "energy.xvg")
    total = energy_series["Total Energy"]
    temperature = energy_series["Temperature"]
    initial_total = total[0]
    drift_values = [value - initial_total for value in total]

    result = {
        "run_id": run_id,
        "role": run["role"],
        "nstlist": int(run["nstlist"]),
        "rlist": run.get("rlist"),
        "verlet_buffer_tolerance": run["verlet_buffer_tolerance"],
        "why_safer": run["why_safer"],
        "intended_use": run["intended_use"],
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

    if run_id in KEY_DUMP_RUN_IDS:
        dump_cmd = [str(GMX), "dump", "-f", "run.trr"]
        commands.append(command_to_string(dump_cmd, work_dir))
        dump = run_command(dump_cmd, work_dir, RESULTS_DIR / f"raw_{run_id}_force_dump.txt", f"{run_id} dump")
        result["final_frame_forces"] = parse_last_frame_forces(dump.stdout, len(DENSE_COORDS))

    return result


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_by_run(path: pathlib.Path) -> dict[str, dict[str, str]]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out[row["run_id"]] = row
    return out


def candidate_status(range_ratio_vs_tight_ref: float, range_ratio_vs_unsafe: float) -> str:
    if range_ratio_vs_tight_ref <= 1.02:
        return "worsening_removed"
    if range_ratio_vs_unsafe <= 0.85:
        return "worsening_materially_weakened"
    return "worsening_persists"


def build_safe_regime_candidates() -> dict[str, object]:
    return {
        "milestone": "TP1.6",
        "fixture_reference": str(TP15B_FIXTURE.relative_to(ROOT)),
        "constraining_inputs": {
            "tp1_5b_pairlist_sweep": str(TP15B_SWEEP.relative_to(ROOT)),
            "tp1_5c_membership_summary": str(TP15C_SUMMARY.relative_to(ROOT)),
            "tp1_5d_decision_summary": str(TP15D_SUMMARY.relative_to(ROOT)),
            "tp1_5e_contract_verdict": str(TP15E_VERDICT.relative_to(ROOT)),
        },
        "unsafe_regime_identified_in_tp1_5e": {
            "run_id": "n10_r0909",
            "nstlist": 10,
            "rlist": 0.909,
            "verlet_buffer_tolerance": -1,
            "why_unsafe": "Manual rlist leaves the critical pair outside the list at the last rebuild and reuses the stale list until the next rebuild.",
        },
        "candidates": RUNS,
    }


def build_summary(results: list[dict[str, object]], tp15b_rows: dict[str, dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    by_id = {entry["run_id"]: entry for entry in results}
    tight_ref = by_id["tight_ref_n1_r1200"]
    unsafe_ref = by_id["n10_r0909"]

    comparison_rows: list[dict[str, object]] = []
    candidate_findings: list[dict[str, object]] = []

    for entry in results:
        range_ratio_vs_tight = entry["total_energy_range_kj"] / tight_ref["total_energy_range_kj"]
        range_ratio_vs_unsafe = entry["total_energy_range_kj"] / unsafe_ref["total_energy_range_kj"]
        drift_delta_vs_tight = entry["final_total_energy_drift_kj"] - tight_ref["final_total_energy_drift_kj"]
        drift_delta_vs_unsafe = entry["final_total_energy_drift_kj"] - unsafe_ref["final_total_energy_drift_kj"]
        status = candidate_status(range_ratio_vs_tight, range_ratio_vs_unsafe)

        comparison_rows.append(
            {
                "run_id": entry["run_id"],
                "role": entry["role"],
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
                "range_ratio_vs_tight_ref": range_ratio_vs_tight,
                "range_ratio_vs_unsafe_reference": range_ratio_vs_unsafe,
                "final_drift_delta_vs_tight_ref_kj": drift_delta_vs_tight,
                "final_drift_delta_vs_unsafe_reference_kj": drift_delta_vs_unsafe,
                "rebaseline_status": status,
            }
        )

        if entry["role"] == "safe_candidate":
            candidate_findings.append(
                {
                    "run_id": entry["run_id"],
                    "settings": {
                        "nstlist": entry["nstlist"],
                        "rlist": entry["rlist"],
                        "verlet_buffer_tolerance": entry["verlet_buffer_tolerance"],
                    },
                    "runtime_pairlist_line": entry["runtime_pairlist_line"],
                    "total_energy_range_kj": entry["total_energy_range_kj"],
                    "range_ratio_vs_tight_ref": range_ratio_vs_tight,
                    "range_ratio_vs_unsafe_reference": range_ratio_vs_unsafe,
                    "final_total_energy_drift_kj": entry["final_total_energy_drift_kj"],
                    "status": status,
                    "why_safer": entry["why_safer"],
                    "intended_use": entry["intended_use"],
                }
            )

    safe_removed = [item for item in candidate_findings if item["status"] == "worsening_removed"]
    safe_weakened = [item for item in candidate_findings if item["status"] == "worsening_materially_weakened"]

    summary = {
        "milestone": "TP1.6",
        "constraining_summary": {
            "unsafe_regime": "TP1.5e classified n10_r0909 with manual rlist=0.909 and verlet-buffer-tolerance=-1 as ALLOWED-UNSAFE rather than a confirmed source-code bug.",
            "why_source_patching_was_deferred": "The critical pair was outside the list at the last rebuild, so TP1.5e found no contract violation to patch.",
            "what_tp1_6_must_establish": "Whether rerunning the same fixture under safer pairlist/buffer settings removes or materially weakens the reproduced worsening.",
        },
        "unsafe_reference": {
            "run_id": unsafe_ref["run_id"],
            "total_energy_range_kj": unsafe_ref["total_energy_range_kj"],
            "final_total_energy_drift_kj": unsafe_ref["final_total_energy_drift_kj"],
            "runtime_pairlist_line": unsafe_ref["runtime_pairlist_line"],
        },
        "tight_reference": {
            "run_id": tight_ref["run_id"],
            "total_energy_range_kj": tight_ref["total_energy_range_kj"],
            "final_total_energy_drift_kj": tight_ref["final_total_energy_drift_kj"],
            "runtime_pairlist_line": tight_ref["runtime_pairlist_line"],
        },
        "safe_candidates": candidate_findings,
        "tp1_5b_reference_rows": {
            run_id: {
                "total_energy_range_kj": float(tp15b_rows[run_id]["total_energy_range_kj"]),
                "final_total_energy_drift_kj": float(tp15b_rows[run_id]["final_total_energy_drift_kj"]),
            }
            for run_id in ["tight_ref_n1_r1200", "n10_r0909", "n1_r0909", "auto_buffer_n10_vbt0005"]
            if run_id in tp15b_rows
        },
    }

    recommendation = {
        "milestone": "TP1.6",
        "source_patching_now_justified": False,
        "safe_baseline_acceptable_for_later_validation": "PARTIAL",
        "unsafe_regime_reference": "n10_r0909",
        "preferred_safe_validation_baseline": "auto_buffer_n10_vbt0005" if any(item["run_id"] == "auto_buffer_n10_vbt0005" and item["status"] == "worsening_removed" for item in candidate_findings) else None,
        "secondary_safe_control": "n1_r0909" if any(item["run_id"] == "n1_r0909" and item["status"] == "worsening_removed" for item in candidate_findings) else None,
        "manual_safe_candidate": "n10_r0911" if any(item["run_id"] == "n10_r0911" and item["status"] != "worsening_persists" for item in candidate_findings) else None,
        "interpretation": "",
        "remaining_short_range_concern": "",
        "next_step_recommendation": "",
    }

    if len(safe_removed) >= 2:
        recommendation["interpretation"] = (
            "On the dense_nonlisted fixture, the previously reproduced worsening is removed by safer pairlist/buffer regimes, "
            "so the strongest supported interpretation is unsafe-regime behavior rather than a surviving short-range implementation defect."
        )
        recommendation["remaining_short_range_concern"] = (
            "TP1.6 does not prove global short-range correctness; it only shows that this fixture no longer signals a deeper defect once unsafe reuse is removed."
        )
        recommendation["next_step_recommendation"] = (
            "Use auto-buffered nstlist=10 as the preferred short-range validation baseline for later toy and pre-authoritative charged-system checks, "
            "but keep larger-system conclusions provisional until a denser validation tier is rerun under the same safe regime."
        )
    elif safe_weakened:
        recommendation["interpretation"] = (
            "Safe settings weaken the reproduced worsening but do not eliminate it, so a deeper short-range audit remains live."
        )
        recommendation["remaining_short_range_concern"] = (
            "At least one residual signal survives under safer settings on the same fixture."
        )
        recommendation["next_step_recommendation"] = (
            "Do not authorize broader charged-system reruns yet; isolate the surviving residual under the safest candidate regime before changing scope."
        )
    else:
        recommendation["interpretation"] = (
            "Safe settings do not materially improve the reproduced worsening on this fixture."
        )
        recommendation["remaining_short_range_concern"] = (
            "A deeper implementation issue would still be live under safe settings."
        )
        recommendation["next_step_recommendation"] = (
            "Return to short-range mechanism auditing on the same dense fixture before any later validation stage."
        )

    return comparison_rows, summary, recommendation


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
    commands.append(command_to_string(["git", "status", "--short"], ROOT))
    commands.append(command_to_string(["git", "rev-parse", "HEAD"], ROOT))
    commands.append(command_to_string([str(GMX), "--version"], ROOT))

    git_status_summary = git_output(["status", "--short"]).splitlines()
    git_commit_hash = git_output(["rev-parse", "HEAD"]).strip()
    build_version = gmx_version_text().splitlines()
    tp15b_rows = read_csv_by_run(TP15B_SWEEP)

    safe_candidates = build_safe_regime_candidates()
    write_text(RESULTS_DIR / "safe_regime_candidates.json", json.dumps(safe_candidates, indent=2) + "\n")

    results = [run_dynamic_case(run, commands) for run in RUNS]
    comparison_rows, summary, recommendation = build_summary(results, tp15b_rows)

    write_csv(
        RESULTS_DIR / "unsafe_vs_safe_comparison.csv",
        [
            "run_id",
            "role",
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
            "range_ratio_vs_unsafe_reference",
            "final_drift_delta_vs_tight_ref_kj",
            "final_drift_delta_vs_unsafe_reference_kj",
            "rebaseline_status",
        ],
        comparison_rows,
    )

    write_text(RESULTS_DIR / "rebaseline_summary.json", json.dumps(summary, indent=2) + "\n")
    write_text(RESULTS_DIR / "tp1_6_recommendation.json", json.dumps(recommendation, indent=2) + "\n")
    write_text(RESULTS_DIR / "raw_commands.txt", "\n".join(commands) + "\n")

    provenance_manifest = {
        "milestone": "TP1.6",
        "execution_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_commit_hash,
        "git_status_summary": git_status_summary,
        "build_version": build_version,
        "fixture_reference": {
            "tp1_5b_fixture_definition": str(TP15B_FIXTURE.relative_to(ROOT)),
            "tp1_5c_membership_summary": str(TP15C_SUMMARY.relative_to(ROOT)),
            "tp1_5d_decision_summary": str(TP15D_SUMMARY.relative_to(ROOT)),
            "tp1_5e_contract_verdict": str(TP15E_VERDICT.relative_to(ROOT)),
            "primary_fixture_id": "dense_nonlisted",
        },
        "commands_run": commands,
        "output_artifacts": [
            "tests/reference_results/tp1_6_rebaseline_audit/safe_regime_candidates.json",
            "tests/reference_results/tp1_6_rebaseline_audit/unsafe_vs_safe_comparison.csv",
            "tests/reference_results/tp1_6_rebaseline_audit/rebaseline_summary.json",
            "tests/reference_results/tp1_6_rebaseline_audit/tp1_6_recommendation.json",
            "tests/reference_results/tp1_6_rebaseline_audit/raw_commands.txt",
            "tests/reference_results/tp1_6_rebaseline_audit/provenance_manifest.json",
        ],
    }
    write_text(RESULTS_DIR / "provenance_manifest.json", json.dumps(provenance_manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
