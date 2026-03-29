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
RESULTS_DIR = ROOT / "tests/reference_results/tp1_7b_authoritative_ab"

TP13_DIR = ROOT / "tests/reference_results/tp1_3_stabilization/TRL-0"
TP16_REPORT = ROOT / "docs/validation_report_tp1_6.md"
TP17_REPORT = ROOT / "docs/validation_report_tp1_7.md"

ENERGY_STDIN = "Potential\nKinetic-En.\nTotal-Energy\nTemperature\nPressure\n0\n"
RUNAWAY_THRESHOLD_K = 400.0

RUNS = [
    {
        "run_id": "unsafe_n10_r0909",
        "role": "same_build_unsafe_reference",
        "nstlist": 10,
        "rlist": 0.909,
        "verlet_buffer_tolerance": -1,
        "why": "Transplants the TP1.5e demonstrated allowed-unsafe manual pairlist regime onto the authoritative system while keeping the TP1.3 executed baseline otherwise fixed.",
        "expected_runtime_pairlist_line": "updated every 10 steps, buffer 0.009 nm, rlist 0.909 nm",
    },
    {
        "run_id": "safe_n10_r0911",
        "role": "same_build_safe_reference",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "why": "Uses the widened manual-safe margin that was runtime-distinct and stable on the TP1.6 toy fixture.",
        "expected_runtime_pairlist_line": "updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm",
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
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout


def gmx_version_text() -> str:
    return subprocess.run([str(GMX), "--version"], cwd=ROOT, text=True, capture_output=True, check=True).stdout


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


def summarize_series(series: dict[str, list[float]]) -> dict[str, float | int | str | None]:
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

    initial_total = total[0]
    drift_values = [value - initial_total for value in total]

    return {
        "duration_ps": time_ps[-1],
        "num_points": len(time_ps),
        "status": "RUNAWAY" if max(temperature) > RUNAWAY_THRESHOLD_K else "NOT_RUNAWAY",
        "runaway_onset_ps": runaway_onset_ps,
        "mean_temperature_k": mean(temperature),
        "max_temperature_k": max(temperature),
        "final_temperature_k": temperature[-1],
        "mean_potential_kj": mean(potential),
        "final_potential_kj": potential[-1],
        "potential_range_kj": max(potential) - min(potential),
        "mean_kinetic_kj": mean(kinetic),
        "final_kinetic_kj": kinetic[-1],
        "mean_total_energy_kj": mean(total),
        "final_total_energy_kj": total[-1],
        "initial_total_energy_kj": initial_total,
        "total_energy_range_kj": max(total) - min(total),
        "max_abs_total_energy_drift_kj": max(abs(value) for value in drift_values),
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


def mdp_text(run: dict[str, object]) -> str:
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
        f"rlist = {run['rlist']}",
        f"verlet-buffer-tolerance = {run['verlet_buffer_tolerance']}",
    ]
    return "\n".join(lines) + "\n"


def run_case(run: dict[str, object], commands: list[str]) -> dict[str, object]:
    run_id = str(run["run_id"])
    work_dir = WORK_DIR / run_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(TP13_DIR / "min.gro", work_dir / "start.gro")
    shutil.copy2(TP13_DIR / "system.top", work_dir / "system.top")
    write_text(work_dir / "run.mdp", mdp_text(run))

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
    summary = summarize_series(series)
    summary.update(
        {
            "run_id": run_id,
            "role": run["role"],
            "executed_now": True,
            "nstlist": int(run["nstlist"]),
            "rlist": float(run["rlist"]),
            "verlet_buffer_tolerance": float(run["verlet_buffer_tolerance"]),
            "why": run["why"],
            "expected_runtime_pairlist_line": run["expected_runtime_pairlist_line"],
            "mdrun_returncode": mdrun.returncode,
            "runtime_repulsion_line": extract_line(work_dir / "run.log", "Detected LJ repulsion power 9."),
            "runtime_kernel_line": extract_line(work_dir / "run.log", "Using plain-C-4x4 4x4 nonbonded short-range kernels"),
            "runtime_pairlist_line": extract_line(work_dir / "run.log", "updated every"),
        }
    )
    return summary


def classify_effect(unsafe: dict[str, object], safe: dict[str, object]) -> str:
    if safe["status"] != "RUNAWAY":
        return "disappears"

    unsafe_onset = unsafe["runaway_onset_ps"]
    safe_onset = safe["runaway_onset_ps"]
    onset_delay = None
    if unsafe_onset is not None and safe_onset is not None:
        onset_delay = safe_onset - unsafe_onset

    max_temp_ratio = safe["max_temperature_k"] / unsafe["max_temperature_k"]
    energy_range_ratio = safe["total_energy_range_kj"] / unsafe["total_energy_range_kj"]
    pressure_ratio = safe["max_abs_pressure_bar"] / unsafe["max_abs_pressure_bar"]

    if onset_delay is not None and onset_delay >= 25.0 and max_temp_ratio <= 0.90:
        return "weakens_materially"
    if max_temp_ratio <= 0.90 and energy_range_ratio <= 0.50 and pressure_ratio <= 0.90:
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
    run_results = [run_case(run, commands) for run in RUNS]
    unsafe = next(run for run in run_results if run["role"] == "same_build_unsafe_reference")
    safe = next(run for run in run_results if run["role"] == "same_build_safe_reference")

    effect = classify_effect(unsafe, safe)
    safe_runtime_distinct = safe["runtime_pairlist_line"] != unsafe["runtime_pairlist_line"]
    unsafe_expected_match = unsafe["runtime_pairlist_line"] == unsafe["expected_runtime_pairlist_line"]
    safe_expected_match = safe["runtime_pairlist_line"] == safe["expected_runtime_pairlist_line"]

    run_matrix = {
        "milestone": "TP1.7b",
        "authoritative_system_id": "dense_salt_polymer",
        "authoritative_system_source": str((ROOT / "tests/reference_results/m10_4_charged_ensemble_gate/dense_salt_polymer").relative_to(ROOT)),
        "tp1_3_executed_baseline_source": str(TP13_DIR.relative_to(ROOT)),
        "comparison_runs": RUNS,
        "same_build_commit": git_output(["rev-parse", "HEAD"]).strip(),
    }
    write_text(RESULTS_DIR / "run_matrix.json", json.dumps(run_matrix, indent=2))

    runtime_distinct = {
        "milestone": "TP1.7b",
        "unsafe_run_id": unsafe["run_id"],
        "safe_run_id": safe["run_id"],
        "unsafe_runtime_pairlist_line": unsafe["runtime_pairlist_line"],
        "safe_runtime_pairlist_line": safe["runtime_pairlist_line"],
        "unsafe_expected_runtime_pairlist_line": unsafe["expected_runtime_pairlist_line"],
        "safe_expected_runtime_pairlist_line": safe["expected_runtime_pairlist_line"],
        "unsafe_matches_expected": unsafe_expected_match,
        "safe_matches_expected": safe_expected_match,
        "safe_runtime_distinct_vs_unsafe": safe_runtime_distinct,
        "shared_kernel_family": (
            unsafe["runtime_kernel_line"] == safe["runtime_kernel_line"]
            and unsafe["runtime_repulsion_line"] == safe["runtime_repulsion_line"]
        ),
    }
    write_text(RESULTS_DIR / "runtime_distinct_check.json", json.dumps(runtime_distinct, indent=2))

    comparison_rows = []
    for run in run_results:
        row = {
            "run_id": run["run_id"],
            "role": run["role"],
            "runtime_pairlist_line": run["runtime_pairlist_line"],
            "status": run["status"],
            "runaway_onset_ps": run["runaway_onset_ps"],
            "max_temperature_k": run["max_temperature_k"],
            "final_temperature_k": run["final_temperature_k"],
            "total_energy_range_kj": run["total_energy_range_kj"],
            "max_abs_total_energy_drift_kj": run["max_abs_total_energy_drift_kj"],
            "mean_pressure_bar": run["mean_pressure_bar"],
            "max_abs_pressure_bar": run["max_abs_pressure_bar"],
            "effect_vs_unsafe": "unsafe_reference" if run["role"] == "same_build_unsafe_reference" else effect,
        }
        comparison_rows.append(row)
    write_csv(
        RESULTS_DIR / "unsafe_vs_safe_authoritative_comparison.csv",
        [
            "run_id",
            "role",
            "runtime_pairlist_line",
            "status",
            "runaway_onset_ps",
            "max_temperature_k",
            "final_temperature_k",
            "total_energy_range_kj",
            "max_abs_total_energy_drift_kj",
            "mean_pressure_bar",
            "max_abs_pressure_bar",
            "effect_vs_unsafe",
        ],
        comparison_rows,
    )

    unsafe_onset = unsafe["runaway_onset_ps"]
    safe_onset = safe["runaway_onset_ps"]
    onset_delay = None if unsafe_onset is None or safe_onset is None else safe_onset - unsafe_onset
    max_temp_delta = safe["max_temperature_k"] - unsafe["max_temperature_k"]
    energy_range_delta = safe["total_energy_range_kj"] - unsafe["total_energy_range_kj"]
    pressure_delta = safe["max_abs_pressure_bar"] - unsafe["max_abs_pressure_bar"]

    if not safe_runtime_distinct:
        remaining_blocker = "unresolved_runtime_not_distinct"
        plain_safe_baseline = "NO"
    elif effect == "disappears":
        remaining_blocker = "short_range_removed_on_authoritative_tier"
        plain_safe_baseline = "YES"
    elif effect == "weakens_materially":
        remaining_blocker = "more_strongly_long_range_or_mixed"
        plain_safe_baseline = "PARTIAL"
    else:
        remaining_blocker = "more_strongly_long_range_or_mixed"
        plain_safe_baseline = "PARTIAL"

    stability_summary = {
        "milestone": "TP1.7b",
        "unsafe_run": unsafe,
        "safe_run": safe,
        "runtime_distinct": runtime_distinct,
        "comparison_metrics": {
            "runaway_effect_classification": effect,
            "runaway_onset_delay_ps": onset_delay,
            "max_temperature_delta_k": max_temp_delta,
            "total_energy_range_delta_kj": energy_range_delta,
            "max_abs_pressure_delta_bar": pressure_delta,
        },
        "remaining_blocker_classification": remaining_blocker,
    }
    write_text(RESULTS_DIR / "stability_summary.json", json.dumps(stability_summary, indent=2))

    recommendation = {
        "milestone": "TP1.7b",
        "source_patching_now_justified": False,
        "plain_safe_baseline_acceptable_for_future_validation": plain_safe_baseline,
        "safe_run_runtime_distinct": safe_runtime_distinct,
        "runaway_effect_classification": effect,
        "remaining_blocker_classification": remaining_blocker,
        "next_step_recommendation": (
            "If the same-build safe run still leaves early runaway, keep short-range code unchanged and isolate the surviving PME/long-range or mixed blocker on this same-build authoritative tier."
        ),
    }
    write_text(RESULTS_DIR / "tp1_7b_recommendation.json", json.dumps(recommendation, indent=2))

    provenance = {
        "milestone": "TP1.7b",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["rev-parse", "HEAD"]).strip(),
        "git_status_short": git_output(["status", "--short"]).splitlines(),
        "gmx_version": gmx_version_text(),
        "constraining_reports": {
            "tp1_6_report": str(TP16_REPORT.relative_to(ROOT)),
            "tp1_7_report": str(TP17_REPORT.relative_to(ROOT)),
        },
        "artifacts": [
            str((RESULTS_DIR / "run_matrix.json").relative_to(ROOT)),
            str((RESULTS_DIR / "runtime_distinct_check.json").relative_to(ROOT)),
            str((RESULTS_DIR / "unsafe_vs_safe_authoritative_comparison.csv").relative_to(ROOT)),
            str((RESULTS_DIR / "stability_summary.json").relative_to(ROOT)),
            str((RESULTS_DIR / "tp1_7b_recommendation.json").relative_to(ROOT)),
        ],
    }
    write_text(RESULTS_DIR / "provenance_manifest.json", json.dumps(provenance, indent=2))
    write_text(RESULTS_DIR / "raw_commands.txt", "\n".join(commands) + "\n")


if __name__ == "__main__":
    main()
