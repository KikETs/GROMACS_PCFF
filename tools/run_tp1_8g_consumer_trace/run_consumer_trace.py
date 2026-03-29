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
RESULTS_DIR = ROOT / "tests/reference_results/tp1_8g_consumer_trace"

TP13_DIR = ROOT / "tests/reference_results/tp1_3_stabilization/TRL-0"
TP18D_RESULTS = ROOT / "tests/reference_results/tp1_8d_coulomb_consumer_trace"
TP18E_RESULTS = ROOT / "tests/reference_results/tp1_8e_handoff_trace"
TP18F_RESULTS = ROOT / "tests/reference_results/tp1_8f_compute_globals_trace"

ENERGY_STDIN = "Potential\nKinetic-En.\nTotal-Energy\nTemperature\nPressure\n0\n"
RUNAWAY_THRESHOLD_K = 400.0
EARLY_STEP_MAX = 200

RUNS = [
    {
        "run_id": "safe_pme_shift_ref",
        "role": "baseline_reference",
        "trace_filename": "consumer_trace_baseline.csv",
        "coulombtype": "PME",
        "coulomb_modifier": "Potential-shift-Verlet",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1e-5,
        "why": "Reuses the TP1.8f authoritative safe baseline settings for immediate post-compute_globals consumer tracing.",
        "intended_path_change": "reference",
    },
    {
        "run_id": "safe_ewald_shift",
        "role": "ewald_variant",
        "trace_filename": "consumer_trace_variant.csv",
        "coulombtype": "Ewald",
        "coulomb_modifier": "Potential-shift-Verlet",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1e-5,
        "why": "Switches the Coulomb solver from PME to Ewald while keeping the same authoritative short-range baseline.",
        "intended_path_change": "pme_vs_ewald_post_compute_globals_consumers",
    },
]

STAGES = [
    "after_update_pcouple",
    "after_energy_add",
    "after_energy_print",
    "after_pressure_prev_handoff",
]

TRACE_INT_FIELDS = [
    "b_calc_ener",
    "b_calc_ener_step",
    "pressure_coupling_is_no",
    "pressure_coupling_consumer_active",
    "has_pressure_previous",
    "pressure_previous_copy_executed",
    "energy_add_called",
    "record_nonenergy_called",
    "energy_print_called",
]

TRACE_FLOAT_FIELDS = [
    "time_ps",
    "total_vir_trace",
    "total_vir_l2",
    "pres_trace",
    "pres_l2",
    "pressure_scalar",
    "potential_energy_kj",
    "kinetic_energy_kj",
    "temperature_k",
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


def command_to_string(cmd: list[str], cwd: pathlib.Path, stdin: str | None = None, env: dict[str, str] | None = None) -> str:
    rendered = f"(cd {cwd} && "
    if env:
        rendered += " ".join(f"{key}={value!r}" for key, value in env.items()) + " "
    rendered += " ".join(cmd) + ")"
    if stdin is not None:
        rendered += f"  # stdin={stdin!r}"
    return rendered


def run_command(
    cmd: list[str],
    cwd: pathlib.Path,
    log_path: pathlib.Path,
    title: str,
    stdin: str | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    process_env = None
    if env:
        process_env = dict(os.environ)
        process_env.update(env)
    result = subprocess.run(cmd, cwd=cwd, text=True, input=stdin, capture_output=True, env=process_env, check=False)
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


def summarize_energy(series: dict[str, list[float]]) -> dict[str, float | int | str | None]:
    time_ps = series["time_ps"]
    total = get_series(series, "Total-Energy", "Total Energy")
    temperature = get_series(series, "Temperature")
    pressure = get_series(series, "Pressure")
    runaway_onset_ps = None
    for time_value, temp_value in zip(time_ps, temperature):
        if temp_value > RUNAWAY_THRESHOLD_K:
            runaway_onset_ps = time_value
            break
    initial_total = total[0]
    return {
        "status": "RUNAWAY" if max(temperature) > RUNAWAY_THRESHOLD_K else "NOT_RUNAWAY",
        "runaway_onset_ps": runaway_onset_ps,
        "max_temperature_k": max(temperature),
        "final_temperature_k": temperature[-1],
        "total_energy_range_kj": max(total) - min(total),
        "max_abs_total_energy_drift_kj": max(abs(value - initial_total) for value in total),
        "max_abs_pressure_bar": max(abs(value) for value in pressure),
        "duration_ps": time_ps[-1],
        "num_points": len(time_ps),
    }


def mdp_text(run: dict[str, object]) -> str:
    lines = [
        "integrator = md",
        "dt = 0.001",
        "nsteps = 20000",
        "cutoff-scheme = Verlet",
        f"nstlist = {run['nstlist']}",
        "pbc = xyz",
        "nstlog = 1000",
        "nstcalcenergy = 100",
        "nstenergy = 100",
        "nstxout = 0",
        "nstvout = 0",
        "nstfout = 0",
        "nstxout-compressed = 0",
        f"coulombtype = {run['coulombtype']}",
        f"coulomb-modifier = {run['coulomb_modifier']}",
        "rcoulomb = 0.9",
        "vdw-type = Cut-off",
        "vdw-modifier = Potential-shift-Verlet",
        "rvdw = 0.9",
        "DispCorr = no",
        f"ewald-rtol = {run['ewald_rtol']}",
        "tcoupl = no",
        "pcoupl = no",
        "constraints = none",
        "gen_vel = yes",
        "gen_temp = 300",
        "gen_seed = -1989880213",
        f"rlist = {run['rlist']}",
        f"verlet-buffer-tolerance = {run['verlet_buffer_tolerance']}",
        f"pme-order = {run['pme_order']}",
        f"fourierspacing = {run['fourierspacing']}",
    ]
    return "\n".join(lines) + "\n"


def read_mdout_value(mdout: pathlib.Path, key: str) -> str | None:
    prefix = f"{key:<25}"
    lines = mdout.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    for line in lines:
        if line.startswith(key):
            return line.split("=", 1)[1].strip()
    return None


def extract_line(path: pathlib.Path, pattern: str) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if pattern in line:
            return line.strip()
    return None


def parse_trace_csv(path: pathlib.Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, object] = {
                "step": int(row["step"]),
                "time_ps": float(row["time_ps"]),
                "stage": row["stage"],
            }
            for field in TRACE_INT_FIELDS:
                parsed[field] = int(row[field])
            for field in TRACE_FLOAT_FIELDS[1:]:
                parsed[field] = float(row[field])
            rows.append(parsed)
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def max_abs(values: list[float]) -> float:
    return max((abs(value) for value in values), default=0.0)


def summarize_stage(rows: list[dict[str, object]]) -> dict[str, object]:
    early_rows = [row for row in rows if int(row["step"]) <= EARLY_STEP_MAX]
    summary: dict[str, object] = {
        "row_count": len(rows),
        "early_row_count": len(early_rows),
        "step_min": min((int(row["step"]) for row in rows), default=None),
        "step_max": max((int(row["step"]) for row in rows), default=None),
    }
    for field in TRACE_INT_FIELDS:
        summary[f"{field}_any"] = any(int(row[field]) for row in rows)
        summary[f"{field}_all"] = all(int(row[field]) for row in rows)
    for field in TRACE_FLOAT_FIELDS:
        full_values = [float(row[field]) for row in rows]
        early_values = [float(row[field]) for row in early_rows]
        summary[f"mean_{field}_full"] = mean(full_values)
        summary[f"mean_{field}_early"] = mean(early_values)
        summary[f"max_abs_{field}_full"] = max_abs(full_values)
        summary[f"max_abs_{field}_early"] = max_abs(early_values)
    return summary


def summarize_trace(rows: list[dict[str, object]]) -> dict[str, object]:
    by_stage = {stage: [row for row in rows if row["stage"] == stage] for stage in STAGES}
    return {
        "row_count": len(rows),
        "stages": {stage: summarize_stage(stage_rows) for stage, stage_rows in by_stage.items()},
    }


def compare_stage(reference_rows: list[dict[str, object]], variant_rows: list[dict[str, object]]) -> dict[str, object]:
    ref = {(int(row["step"]), str(row["stage"])): row for row in reference_rows}
    var = {(int(row["step"]), str(row["stage"])): row for row in variant_rows}
    shared_keys = sorted(set(ref) & set(var))
    early_keys = [key for key in shared_keys if key[0] <= EARLY_STEP_MAX]

    comparison: dict[str, object] = {
        "shared_row_count": len(shared_keys),
        "early_shared_row_count": len(early_keys),
        "stage_mismatch_count": sum(1 for key in shared_keys if str(ref[key]["stage"]) != str(var[key]["stage"])),
    }
    for field in TRACE_INT_FIELDS:
        comparison[f"{field}_mismatch_count"] = sum(
            1 for key in shared_keys if int(ref[key][field]) != int(var[key][field])
        )
    for field in TRACE_FLOAT_FIELDS:
        comparison[f"mean_abs_delta_{field}_full"] = mean(
            [abs(float(ref[key][field]) - float(var[key][field])) for key in shared_keys]
        )
        comparison[f"mean_abs_delta_{field}_early"] = mean(
            [abs(float(ref[key][field]) - float(var[key][field])) for key in early_keys]
        )
    return comparison


def compare_traces(reference_rows: list[dict[str, object]], variant_rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "all_stages": compare_stage(reference_rows, variant_rows),
        "by_stage": {
            stage: compare_stage(
                [row for row in reference_rows if row["stage"] == stage],
                [row for row in variant_rows if row["stage"] == stage],
            )
            for stage in STAGES
        },
    }


def run_case(run: dict[str, object], commands: list[str]) -> dict[str, object]:
    run_id = str(run["run_id"])
    work_dir = WORK_DIR / run_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(TP13_DIR / "min.gro", work_dir / "start.gro")
    shutil.copy2(TP13_DIR / "system.top", work_dir / "system.top")
    write_text(work_dir / "run.mdp", mdp_text(run))

    grompp_cmd = [str(GMX), "grompp", "-f", "run.mdp", "-c", "start.gro", "-p", "system.top", "-o", "run.tpr", "-maxwarn", "10"]
    commands.append(command_to_string(grompp_cmd, work_dir))
    run_command(grompp_cmd, work_dir, RESULTS_DIR / f"raw_{run_id}_grompp.log", f"{run_id} grompp")
    shutil.copy2(work_dir / "mdout.mdp", RESULTS_DIR / f"raw_{run_id}_mdout.mdp")

    trace_output = RESULTS_DIR / str(run["trace_filename"])
    env = {"GMX_TP18G_TRACE_FILE": str(trace_output)}
    mdrun_cmd = [str(GMX), "mdrun", "-s", "run.tpr", "-deffnm", "run", "-nt", "1"]
    commands.append(command_to_string(mdrun_cmd, work_dir, env=env))
    mdrun = run_command(mdrun_cmd, work_dir, RESULTS_DIR / f"raw_{run_id}_mdrun.log", f"{run_id} mdrun", env=env, check=False)
    if not (work_dir / "run.edr").exists():
        raise RuntimeError(f"{run_id} did not produce run.edr (returncode={mdrun.returncode})")
    if not trace_output.exists():
        raise RuntimeError(f"{run_id} did not produce TP1.8g trace output")
    shutil.copy2(work_dir / "run.log", RESULTS_DIR / f"raw_{run_id}_md.log")

    energy_cmd = [str(GMX), "energy", "-f", "run.edr", "-o", "energy.xvg"]
    commands.append(command_to_string(energy_cmd, work_dir, ENERGY_STDIN))
    run_command(energy_cmd, work_dir, RESULTS_DIR / f"raw_{run_id}_energy_output.txt", f"{run_id} energy", ENERGY_STDIN)
    shutil.copy2(work_dir / "energy.xvg", RESULTS_DIR / f"raw_{run_id}_energy.xvg")

    energy_summary = summarize_energy(parse_xvg(work_dir / "energy.xvg"))
    trace_rows = parse_trace_csv(trace_output)
    trace_summary = summarize_trace(trace_rows)
    mdout_path = work_dir / "mdout.mdp"
    log_path = work_dir / "run.log"
    summary = {
        "run_id": run_id,
        "role": run["role"],
        "executed_now": True,
        "why": run["why"],
        "intended_path_change": run["intended_path_change"],
        "mdrun_returncode": mdrun.returncode,
        "mdout_coulombtype": read_mdout_value(mdout_path, "coulombtype"),
        "mdout_coulomb_modifier": read_mdout_value(mdout_path, "coulomb-modifier"),
        "mdout_nstlist": read_mdout_value(mdout_path, "nstlist"),
        "mdout_rlist": read_mdout_value(mdout_path, "rlist"),
        "mdout_vbt": read_mdout_value(mdout_path, "verlet-buffer-tolerance"),
        "mdout_vdw_type": read_mdout_value(mdout_path, "vdw-type"),
        "mdout_pme_order": read_mdout_value(mdout_path, "pme-order"),
        "mdout_fourierspacing": read_mdout_value(mdout_path, "fourierspacing"),
        "mdout_pcoupl": read_mdout_value(mdout_path, "pcoupl"),
        "runtime_ordinary_ewald_line": extract_line(log_path, "Will do ordinary reciprocal space Ewald sum."),
        "runtime_solve_pme_line": extract_line(log_path, "Solve PME"),
        "runtime_pairlist_line": extract_line(log_path, "updated every"),
        "runtime_kernel_line": extract_line(log_path, "Using plain-C-4x4 4x4 nonbonded short-range kernels"),
        "trace_output_path": str(trace_output.relative_to(ROOT)),
        "energy_summary": energy_summary,
        "trace_summary": trace_summary,
    }
    return {"summary": summary, "trace_rows": trace_rows}


def build_path_map() -> dict[str, object]:
    return {
        "milestone": "TP1.8g",
        "authoritative_system_id": "dense_salt_polymer",
        "reference_run_id": "safe_pme_shift_ref",
        "paths": [
            {
                "file": "src/gromacs/mdlib/md_support.cpp",
                "function": "compute_globals",
                "role": "Upstream producer of total_vir, pres, and scalar pressure already localized in TP1.8f.",
                "why_traced": "TP1.8g starts immediately after compute_globals and does not reopen its internal pressure algebra.",
                "status": "active_upstream_reference",
            },
            {
                "file": "src/gromacs/mdrun/md.cpp",
                "function": "LegacySimulator::do_md -> update_pcouple_after_coordinates",
                "role": "Immediate post-compute_globals pressure-coupling/control-state consumer call site.",
                "why_traced": "Needed to verify whether the authoritative setup turns the pres/virial split into an operational pressure-control action.",
                "status": "active_callsite_but_expected_inert_for_pcoupl_no",
            },
            {
                "file": "src/gromacs/mdlib/coupling.cpp",
                "function": "update_pcouple_after_coordinates",
                "role": "Pressure-coupling state consumer for Berendsen/C-rescale/PR/MTTK modes.",
                "why_traced": "The authoritative TP1.8g setup keeps pcoupl = no, so this path should reduce to the No case rather than consume the split operationally.",
                "status": "inactive_for_authoritative_setup",
            },
            {
                "file": "src/gromacs/mdrun/md.cpp",
                "function": "LegacySimulator::do_md -> EnergyOutput::addDataAtEnergyStep",
                "role": "Immediate reporting/global-accumulation consumer of total_vir and pres.",
                "why_traced": "Needed to verify whether the surviving split is forwarded only into reporting/output state at this boundary.",
                "status": "active_on_energy_steps",
            },
            {
                "file": "src/gromacs/mdlib/energyoutput.cpp",
                "function": "EnergyOutput::addDataAtEnergyStep",
                "role": "Stores virial and pressure tensors into the energy-bin accumulation.",
                "why_traced": "This is the first active post-compute_globals consumer that definitely records the split under pcoupl = no.",
                "status": "active_on_energy_steps",
            },
            {
                "file": "src/gromacs/mdrun/md.cpp",
                "function": "LegacySimulator::do_md -> EnergyOutput::printStepToEnergyFile",
                "role": "Immediate reporting/output writer after accumulation.",
                "why_traced": "Needed to determine whether the split becomes operational or stays at reporting-level output in the authoritative setup.",
                "status": "active_on_log_or_energy_steps",
            },
            {
                "file": "src/gromacs/mdrun/md.cpp",
                "function": "LegacySimulator::do_md -> copy_mat(pres, state_->pres_prev)",
                "role": "Pressure-coupling state handoff for the next step.",
                "why_traced": "Needed to verify whether the surviving split is copied into later control-state under this setup.",
                "status": "inactive_for_authoritative_setup",
            },
            {
                "file": "src/gromacs/mdlib/md_support.cpp",
                "function": "set_state_entries",
                "role": "Allocates StateEntry::PressurePrevious only for pressure-coupled modes.",
                "why_traced": "Explains why the post-compute_globals pressure state handoff is absent when pcoupl = no.",
                "status": "inactive_pressure_state_consumer_for_authoritative_setup",
            },
        ],
        "tp1_4_lj_pme_path_status": "inactive",
    }


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

    baseline = next(result for result in run_results if result["summary"]["role"] == "baseline_reference")
    variant = next(result for result in run_results if result["summary"]["role"] == "ewald_variant")
    comparison = compare_traces(baseline["trace_rows"], variant["trace_rows"])

    baseline_stages = baseline["summary"]["trace_summary"]["stages"]
    variant_stages = variant["summary"]["trace_summary"]["stages"]

    pressure_control_inactive = (
        baseline_stages["after_update_pcouple"]["pressure_coupling_consumer_active_any"] is False
        and variant_stages["after_update_pcouple"]["pressure_coupling_consumer_active_any"] is False
        and baseline_stages["after_pressure_prev_handoff"]["has_pressure_previous_any"] is False
        and variant_stages["after_pressure_prev_handoff"]["has_pressure_previous_any"] is False
    )
    reporting_consumer_active = (
        baseline_stages["after_energy_add"]["energy_add_called_any"]
        and variant_stages["after_energy_add"]["energy_add_called_any"]
        and baseline_stages["after_energy_print"]["energy_print_called_any"]
        and variant_stages["after_energy_print"]["energy_print_called_any"]
    )
    consumer_boundary_classification = (
        "aggregate_or_reporting_level_only"
        if pressure_control_inactive and reporting_consumer_active
        else "mixed_or_still_unresolved"
    )

    summary = {
        "milestone": "TP1.8g",
        "baseline": baseline["summary"],
        "variant": variant["summary"],
        "comparison": comparison,
        "interpretation": {
            "tp1_4_lj_pme_path_status": "inactive",
            "pressure_control_consumer_active": not pressure_control_inactive,
            "reporting_consumer_active": reporting_consumer_active,
            "consumer_boundary_classification": consumer_boundary_classification,
            "overall_classification": "still_unresolved",
        },
    }

    recommendation = {
        "milestone": "TP1.8g",
        "source_patching_now_justified": False,
        "plain_safe_baseline_acceptable_for_later_non_rrespa_validation": "PARTIAL",
        "consumer_boundary_classification": consumer_boundary_classification,
        "overall_classification": "still_unresolved",
        "tp1_4_lj_pme_path_status": "inactive",
        "exact_next_step_recommendation": "Stop at unresolved unless there is a concrete reason to trace beyond the immediate post-compute_globals output consumers; under pcoupl = no the pressure-control consumers are inactive here, so production patching is still not justified.",
    }

    run_matrix = {
        "milestone": "TP1.8g",
        "authoritative_system_id": "dense_salt_polymer",
        "tp13_baseline_source": str(TP13_DIR.relative_to(ROOT)),
        "tp18d_source": str(TP18D_RESULTS.relative_to(ROOT)),
        "tp18e_source": str(TP18E_RESULTS.relative_to(ROOT)),
        "tp18f_source": str(TP18F_RESULTS.relative_to(ROOT)),
        "reference_run_id": "safe_pme_shift_ref",
        "comparison_runs": RUNS,
        "rationale_for_20ps_window": "TP1.8e and TP1.8f used the same 20 ps authoritative trace window with runaway onset at 0.2 ps, so TP1.8g keeps the same window while tracing only the immediate consumers after compute_globals.",
    }

    write_text(RESULTS_DIR / "run_matrix.json", json.dumps(run_matrix, indent=2) + "\n")
    write_text(RESULTS_DIR / "consumer_path_map.json", json.dumps(build_path_map(), indent=2) + "\n")
    write_text(RESULTS_DIR / "consumer_trace_summary.json", json.dumps(summary, indent=2) + "\n")
    write_text(RESULTS_DIR / "tp1_8g_recommendation.json", json.dumps(recommendation, indent=2) + "\n")
    write_text(RESULTS_DIR / "raw_commands.txt", "\n".join(commands) + "\n")

    provenance = {
        "milestone": "TP1.8g",
        "execution_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["rev-parse", "HEAD"]).strip(),
        "git_status_short": git_output(["status", "--short"]).splitlines(),
        "gmx_version": gmx_version_text().splitlines()[0],
        "inputs": {
            "tp13_baseline": str(TP13_DIR.relative_to(ROOT)),
            "tp18d_results": str(TP18D_RESULTS.relative_to(ROOT)),
            "tp18e_results": str(TP18E_RESULTS.relative_to(ROOT)),
            "tp18f_results": str(TP18F_RESULTS.relative_to(ROOT)),
        },
    }
    write_text(RESULTS_DIR / "provenance_manifest.json", json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
