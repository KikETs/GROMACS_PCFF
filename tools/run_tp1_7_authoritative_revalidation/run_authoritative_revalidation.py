#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import pathlib
import shutil
import subprocess
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[2]
GMX = ROOT / "build/bin/gmx"

TOOL_DIR = pathlib.Path(__file__).resolve().parent
WORK_DIR = TOOL_DIR / "work"
RESULTS_DIR = ROOT / "tests/reference_results/tp1_7_authoritative_revalidation"

TP13_DIR = ROOT / "tests/reference_results/tp1_3_stabilization/TRL-0"
TP14_REPORT = ROOT / "docs/validation_report_tp1_4.md"
TP15E_REPORT = ROOT / "docs/validation_report_tp1_5e.md"
TP16_REPORT = ROOT / "docs/validation_report_tp1_6.md"

ENERGY_STDIN = "Potential\nKinetic-En.\nTotal-Energy\nTemperature\nPressure\n0\n"
RUNAWAY_THRESHOLD_K = 400.0

SAFE_RUNS = [
    {
        "run_id": "safe_auto_n10_vbt0005",
        "role": "preferred_safe_baseline",
        "nstlist": 10,
        "verlet_buffer_tolerance": 0.005,
        "why_safer": "Matches the TP1.6 preferred auto-buffer baseline candidate and avoids manual-rlist reuse semantics.",
    },
    {
        "run_id": "manual_safe_n10_r0911",
        "role": "secondary_manual_safe_candidate",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "why_safer": "Uses the TP1.6 manual-safe margin that exceeded the TP1.5e critical-pair distance on the dense toy fixture.",
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


def command_to_string(cmd: list[str], cwd: pathlib.Path, stdin: str | None = None) -> str:
    rendered = f"(cd {cwd} && {' '.join(cmd)})"
    if stdin is not None:
        rendered += f"  # stdin={stdin!r}"
    return rendered


def run_command(
    cmd: list[str],
    cwd: pathlib.Path,
    log_path: pathlib.Path,
    title: str,
    stdin: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, text=True, input=stdin, capture_output=True, check=False)
    append_section(log_path, f"{title} stdout", result.stdout)
    append_section(log_path, f"{title} stderr", result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"{title} failed with code {result.returncode}")
    return result


def git_output(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout


def gmx_version_text() -> str:
    result = subprocess.run([str(GMX), "--version"], cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout


def parse_xvg(path: pathlib.Path) -> dict[str, list[float]]:
    legends: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("@ s") and " legend " in line:
            legends.append(line.split("legend", 1)[1].strip().strip('"'))
        elif line and not line.startswith(("#", "@")):
            rows.append([float(token) for token in line.split()])

    if not rows:
        raise RuntimeError(f"No data rows found in {path}")

    columns = list(zip(*rows))
    series: dict[str, list[float]] = {"time_ps": list(columns[0])}
    for index, legend in enumerate(legends, start=1):
        series[legend] = list(columns[index])
    return series


def get_series(series: dict[str, list[float]], *names: str) -> list[float]:
    for name in names:
        if name in series:
            return series[name]
    raise KeyError(f"Missing series. Tried {names!r}, found {sorted(series.keys())!r}")


def summarize_series(series: dict[str, list[float]]) -> dict[str, object]:
    time_ps = series["time_ps"]
    potential = get_series(series, "Potential")
    kinetic = get_series(series, "Kinetic-En.", "Kinetic En.")
    total = get_series(series, "Total-Energy", "Total Energy")
    temperature = get_series(series, "Temperature")
    pressure = get_series(series, "Pressure")

    runaway_onset_ps = None
    for time_value, temp_value in zip(time_ps, temperature):
        if temp_value > RUNAWAY_THRESHOLD_K:
            runaway_onset_ps = time_value
            break

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    return {
        "duration_ps": time_ps[-1],
        "num_points": len(time_ps),
        "mean_temperature_k": mean(temperature),
        "max_temperature_k": max(temperature),
        "final_temperature_k": temperature[-1],
        "runaway_onset_ps": runaway_onset_ps,
        "status": "RUNAWAY" if max(temperature) > RUNAWAY_THRESHOLD_K else "NOT_RUNAWAY",
        "mean_potential_kj": mean(potential),
        "final_potential_kj": potential[-1],
        "potential_range_kj": max(potential) - min(potential),
        "mean_kinetic_kj": mean(kinetic),
        "final_kinetic_kj": kinetic[-1],
        "mean_total_energy_kj": mean(total),
        "final_total_energy_kj": total[-1],
        "total_energy_range_kj": max(total) - min(total),
        "mean_pressure_bar": mean(pressure),
        "max_pressure_bar": max(pressure),
        "min_pressure_bar": min(pressure),
        "max_abs_pressure_bar": max(abs(value) for value in pressure),
    }


def extract_line(path: pathlib.Path, pattern: str) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if pattern in line:
            return line.strip()
    return None


def extract_reference_metrics(commands: list[str]) -> dict[str, object]:
    results_subdir = RESULTS_DIR / "historical_reference"
    results_subdir.mkdir(parents=True, exist_ok=True)

    reference_log = results_subdir / "raw_historical_reference_energy_output.txt"
    energy_cmd = [str(GMX), "energy", "-f", str(TP13_DIR / "trial.edr"), "-o", str(results_subdir / "historical_reference_energy.xvg")]
    commands.append(command_to_string(energy_cmd, ROOT, ENERGY_STDIN))
    run_command(energy_cmd, ROOT, reference_log, "historical reference energy extraction", ENERGY_STDIN)

    shutil.copy2(TP13_DIR / "trial.log", results_subdir / "raw_historical_reference_md.log")
    shutil.copy2(TP13_DIR / "trial.mdp", results_subdir / "historical_reference_trial.mdp")
    shutil.copy2(TP13_DIR / "mdout.mdp", results_subdir / "historical_reference_mdout.mdp")
    shutil.copy2(TP13_DIR / "summary.json", results_subdir / "historical_reference_tp1_3_summary.json")
    shutil.copy2(TP13_DIR / "min.gro", results_subdir / "historical_reference_min.gro")
    shutil.copy2(TP13_DIR / "system.top", results_subdir / "historical_reference_system.top")

    series = parse_xvg(results_subdir / "historical_reference_energy.xvg")
    metrics = summarize_series(series)
    metrics.update(
        {
            "run_id": "historical_tp1_3_reference",
            "role": "trusted_prior_reference",
            "executed_now": False,
            "source_artifact": str(TP13_DIR.relative_to(ROOT)),
            "nominal_tp1_3_label": "TRL-0 baseline (summary labels this NPT)",
            "actual_trial_mdp_keywords": {
                "tcouple": "v-rescale",
                "pcouple": "berendsen",
            },
            "actual_executed_keywords": {
                "tcoupl": "No",
                "pcoupl": "No",
                "nstlist": 10,
                "verlet-buffer-tolerance": 0.005,
            },
            "runtime_repulsion_line": extract_line(TP13_DIR / "trial.log", "Detected LJ repulsion power 9."),
            "runtime_kernel_line": extract_line(TP13_DIR / "trial.log", "Using plain-C-4x4 4x4 nonbonded short-range kernels"),
            "runtime_pairlist_line": extract_line(TP13_DIR / "trial.log", "updated every"),
        }
    )
    return metrics


def rerun_mdp_text(run: dict[str, object]) -> str:
    lines = [
        "integrator = md",
        "dt = 0.001",
        "nsteps = 500000",
        "cutoff-scheme = Verlet",
        f"nstlist = {run['nstlist']}",
        "pbc = xyz",
        "nstlog = 1000",
        "nstcalcenergy = 100",
        "nstenergy = 1000",
        "nstxout = 0",
        "nstvout = 0",
        "nstfout = 0",
        "nstxout-compressed = 0",
        "coulombtype = PME",
        "coulomb-modifier = Potential-shift-Verlet",
        "rcoulomb = 0.9",
        "vdw-type = Cut-off",
        "vdw-modifier = Potential-shift-Verlet",
        "rvdw = 0.9",
        "DispCorr = no",
        "pme-order = 4",
        "fourierspacing = 0.12",
        "ewald-rtol = 1e-5",
        "tcoupl = no",
        "pcoupl = no",
        "constraints = none",
        "gen_vel = yes",
        "gen_temp = 300",
        "gen_seed = -1989880213",
    ]
    if "rlist" in run:
        lines.append(f"rlist = {run['rlist']}")
    lines.append(f"verlet-buffer-tolerance = {run['verlet_buffer_tolerance']}")
    return "\n".join(lines) + "\n"


def run_safe_case(run: dict[str, object], commands: list[str]) -> dict[str, object]:
    run_id = str(run["run_id"])
    work_dir = WORK_DIR / run_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(TP13_DIR / "min.gro", work_dir / "start.gro")
    shutil.copy2(TP13_DIR / "system.top", work_dir / "system.top")
    write_text(work_dir / "run.mdp", rerun_mdp_text(run))

    grompp_cmd = [
        str(GMX),
        "grompp",
        "-f",
        "run.mdp",
        "-c",
        "start.gro",
        "-p",
        "system.top",
        "-o",
        "run.tpr",
        "-maxwarn",
        "10",
    ]
    commands.append(command_to_string(grompp_cmd, work_dir))
    run_command(grompp_cmd, work_dir, RESULTS_DIR / f"raw_{run_id}_grompp.log", f"{run_id} grompp")
    shutil.copy2(work_dir / "mdout.mdp", RESULTS_DIR / f"raw_{run_id}_mdout.mdp")

    mdrun_cmd = [str(GMX), "mdrun", "-s", "run.tpr", "-deffnm", "run", "-nt", "1"]
    commands.append(command_to_string(mdrun_cmd, work_dir))
    mdrun = run_command(mdrun_cmd, work_dir, RESULTS_DIR / f"raw_{run_id}_mdrun.log", f"{run_id} mdrun", check=False)
    if not (work_dir / "run.edr").exists():
        raise RuntimeError(f"{run_id} did not produce run.edr (returncode={mdrun.returncode})")
    shutil.copy2(work_dir / "run.log", RESULTS_DIR / f"raw_{run_id}_md.log")

    energy_cmd = [str(GMX), "energy", "-f", "run.edr", "-o", "energy.xvg"]
    commands.append(command_to_string(energy_cmd, work_dir, ENERGY_STDIN))
    run_command(energy_cmd, work_dir, RESULTS_DIR / f"raw_{run_id}_energy_output.txt", f"{run_id} energy", ENERGY_STDIN)
    shutil.copy2(work_dir / "energy.xvg", RESULTS_DIR / f"raw_{run_id}_energy.xvg")

    series = parse_xvg(work_dir / "energy.xvg")
    metrics = summarize_series(series)
    metrics.update(
        {
            "run_id": run_id,
            "role": run["role"],
            "executed_now": True,
            "nstlist": int(run["nstlist"]),
            "verlet_buffer_tolerance": float(run["verlet_buffer_tolerance"]),
            "rlist": run.get("rlist"),
            "why_safer": run["why_safer"],
            "mdrun_returncode": mdrun.returncode,
            "runtime_repulsion_line": extract_line(work_dir / "run.log", "Detected LJ repulsion power 9."),
            "runtime_kernel_line": extract_line(work_dir / "run.log", "Using plain-C-4x4 4x4 nonbonded short-range kernels"),
            "runtime_pairlist_line": extract_line(work_dir / "run.log", "updated every"),
        }
    )
    return metrics


def classify_effect(reference: dict[str, object], run: dict[str, object]) -> str:
    if run["status"] != "RUNAWAY":
        return "disappears"

    ref_onset = reference["runaway_onset_ps"]
    run_onset = run["runaway_onset_ps"]
    onset_ratio = None
    if ref_onset not in (None, 0) and run_onset is not None:
        onset_ratio = run_onset / ref_onset

    max_ratio = run["max_temperature_k"] / reference["max_temperature_k"]
    if onset_ratio is not None and onset_ratio >= 1.25 and max_ratio <= 0.90:
        return "weakens_materially"
    return "persists"


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    if not GMX.exists():
        raise SystemExit(f"Missing GROMACS binary: {GMX}")

    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []

    reference = extract_reference_metrics(commands)
    safe_runs = [run_safe_case(run, commands) for run in SAFE_RUNS]

    for run in safe_runs:
        run["effect_vs_reference"] = classify_effect(reference, run)
        run["max_temp_delta_k"] = run["max_temperature_k"] - reference["max_temperature_k"]
        run["runaway_onset_delta_ps"] = (
            None
            if reference["runaway_onset_ps"] is None or run["runaway_onset_ps"] is None
            else run["runaway_onset_ps"] - reference["runaway_onset_ps"]
        )
        run["total_energy_range_delta_kj"] = run["total_energy_range_kj"] - reference["total_energy_range_kj"]
        run["mean_pressure_delta_bar"] = run["mean_pressure_bar"] - reference["mean_pressure_bar"]

    run_matrix = {
        "milestone": "TP1.7",
        "authoritative_system_id": "dense_salt_polymer",
        "authoritative_system_source": str((ROOT / "tests/reference_results/m10_4_charged_ensemble_gate/dense_salt_polymer").relative_to(ROOT)),
        "historical_reference": {
            "artifact_dir": str(TP13_DIR.relative_to(ROOT)),
            "note": "This authoritative TP1.3 reference already used nstlist=10 with positive verlet-buffer-tolerance in metadata; no prior authoritative manual-unsafe reference artifact was found.",
        },
        "safe_runs": SAFE_RUNS,
    }
    write_text(RESULTS_DIR / "run_matrix.json", json.dumps(run_matrix, indent=2))

    comparison_rows: list[dict[str, object]] = []
    reference_row = {
        "run_id": reference["run_id"],
        "role": reference["role"],
        "executed_now": reference["executed_now"],
        "comparison_type": "historical_reference",
        "runtime_pairlist_line": reference["runtime_pairlist_line"],
        "status": reference["status"],
        "runaway_onset_ps": reference["runaway_onset_ps"],
        "max_temperature_k": reference["max_temperature_k"],
        "final_temperature_k": reference["final_temperature_k"],
        "total_energy_range_kj": reference["total_energy_range_kj"],
        "mean_pressure_bar": reference["mean_pressure_bar"],
        "effect_vs_reference": "reference",
    }
    comparison_rows.append(reference_row)
    for run in safe_runs:
        comparison_rows.append(
            {
                "run_id": run["run_id"],
                "role": run["role"],
                "executed_now": run["executed_now"],
                "comparison_type": "safe_rerun",
                "runtime_pairlist_line": run["runtime_pairlist_line"],
                "status": run["status"],
                "runaway_onset_ps": run["runaway_onset_ps"],
                "max_temperature_k": run["max_temperature_k"],
                "final_temperature_k": run["final_temperature_k"],
                "total_energy_range_kj": run["total_energy_range_kj"],
                "mean_pressure_bar": run["mean_pressure_bar"],
                "effect_vs_reference": run["effect_vs_reference"],
            }
        )
    write_csv(
        RESULTS_DIR / "unsafe_vs_safe_authoritative_comparison.csv",
        [
            "run_id",
            "role",
            "executed_now",
            "comparison_type",
            "runtime_pairlist_line",
            "status",
            "runaway_onset_ps",
            "max_temperature_k",
            "final_temperature_k",
            "total_energy_range_kj",
            "mean_pressure_bar",
            "effect_vs_reference",
        ],
        comparison_rows,
    )

    preferred_run = next(run for run in safe_runs if run["run_id"] == "safe_auto_n10_vbt0005")
    manual_safe_run = next(run for run in safe_runs if run["run_id"] == "manual_safe_n10_r0911")
    preferred_runtime_changed = preferred_run["runtime_pairlist_line"] != reference["runtime_pairlist_line"]

    if preferred_run["effect_vs_reference"] == "disappears":
        remaining_blocker = "short_range_removed_on_authoritative_tier"
        recommendation_status = "YES"
    elif preferred_run["effect_vs_reference"] == "weakens_materially":
        remaining_blocker = "mixed_unresolved"
        recommendation_status = "PARTIAL"
    else:
        if not preferred_runtime_changed:
            remaining_blocker = "more_strongly_long_range_or_mixed"
            recommendation_status = "PARTIAL"
        elif manual_safe_run["effect_vs_reference"] == "persists":
            remaining_blocker = "more_strongly_long_range_or_mixed"
            recommendation_status = "YES"
        else:
            remaining_blocker = "mixed_unresolved"
            recommendation_status = "PARTIAL"

    summary = {
        "milestone": "TP1.7",
        "reference": reference,
        "safe_runs": safe_runs,
        "authoritative_reference_is_manual_unsafe": False,
        "key_interpretation": {
            "tp1_3_authoritative_reference_pairlist_line": reference["runtime_pairlist_line"],
            "preferred_safe_run_pairlist_line": preferred_run["runtime_pairlist_line"],
            "preferred_safe_runtime_changed_vs_reference": preferred_runtime_changed,
            "preferred_safe_effect_vs_reference": preferred_run["effect_vs_reference"],
            "manual_safe_effect_vs_reference": manual_safe_run["effect_vs_reference"],
            "remaining_blocker_classification": remaining_blocker,
        },
    }
    write_text(RESULTS_DIR / "stability_summary.json", json.dumps(summary, indent=2))

    recommendation = {
        "milestone": "TP1.7",
        "source_patching_now_justified": False,
        "plain_safe_baseline_acceptable_for_future_validation": recommendation_status,
        "preferred_safe_baseline_run_id": preferred_run["run_id"],
        "historical_reference_was_manual_unsafe": False,
        "preferred_safe_runtime_changed_vs_reference": preferred_runtime_changed,
        "remaining_blocker_classification": remaining_blocker,
        "next_step_recommendation": (
            "Do not treat the plain TP1.6 auto-buffer label as sufficient on larger charged systems; "
            "use a runtime-verified safe short-range setting such as the widened manual margin for the next authoritative comparison, "
            "and focus blocker isolation on PME/long-range or mixed causes rather than pairlist code patching."
        ),
    }
    write_text(RESULTS_DIR / "tp1_7_recommendation.json", json.dumps(recommendation, indent=2))

    provenance = {
        "milestone": "TP1.7",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["rev-parse", "HEAD"]).strip(),
        "git_status_short": git_output(["status", "--short"]).splitlines(),
        "gmx_version": gmx_version_text(),
        "input_constraints": {
            "tp1_4_report": str(TP14_REPORT.relative_to(ROOT)),
            "tp1_5e_report": str(TP15E_REPORT.relative_to(ROOT)),
            "tp1_6_report": str(TP16_REPORT.relative_to(ROOT)),
        },
        "artifacts": [
            str((RESULTS_DIR / "run_matrix.json").relative_to(ROOT)),
            str((RESULTS_DIR / "unsafe_vs_safe_authoritative_comparison.csv").relative_to(ROOT)),
            str((RESULTS_DIR / "stability_summary.json").relative_to(ROOT)),
            str((RESULTS_DIR / "tp1_7_recommendation.json").relative_to(ROOT)),
        ],
    }
    write_text(RESULTS_DIR / "provenance_manifest.json", json.dumps(provenance, indent=2))
    write_text(RESULTS_DIR / "raw_commands.txt", "\n".join(commands) + "\n")


if __name__ == "__main__":
    main()
