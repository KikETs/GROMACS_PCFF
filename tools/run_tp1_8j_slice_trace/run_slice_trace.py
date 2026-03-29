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
RESULTS_DIR = ROOT / "tests/reference_results/tp1_8j_slice_trace"

TP13_DIR = ROOT / "tests/reference_results/tp1_3_stabilization/TRL-0"
TP18I_RESULTS = ROOT / "tests/reference_results/tp1_8i_consumer_trace"

ENERGY_STDIN = "Potential\nKinetic-En.\nTotal-Energy\nTemperature\nPressure\n0\n"
RUNAWAY_THRESHOLD_K = 400.0
EARLY_STEP_MAX = 200
CALLSITE_NAME = "post_update_compute_globals"

RUNS = [
    {
        "run_id": "safe_pme_shift_ref",
        "role": "baseline_reference",
        "trace_filename": "slice_trace_baseline.csv",
        "coulombtype": "PME",
        "coulomb_modifier": "Potential-shift-Verlet",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1e-5,
        "why": "Reuses the TP1.8i authoritative safe baseline settings while isolating only the post-update compute_globals callsite.",
        "intended_path_change": "reference",
    },
    {
        "run_id": "safe_ewald_shift",
        "role": "ewald_variant",
        "trace_filename": "slice_trace_variant.csv",
        "coulombtype": "Ewald",
        "coulomb_modifier": "Potential-shift-Verlet",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1e-5,
        "why": "Switches the Coulomb solver from PME to Ewald while keeping the same authoritative short-range baseline and isolating only the post-update compute_globals slice.",
        "intended_path_change": "pme_vs_ewald_isolated_post_update_compute_globals_slice",
    },
]

TRACE_STRING_FIELDS = ["callsite", "stage"]
TRACE_INT_FIELDS = [
    "b_gstat",
    "b_energy",
    "b_temperature",
    "compute_ekin",
    "b_read_ekin",
    "b_ekin_ave_vel",
    "b_scale_ekin",
    "have_leapfrog",
    "have_ekinh_old",
    "mpi_parallel",
    "gstat_reduction_executed",
    "temperature_group_count",
]
TRACE_FLOAT_FIELDS = [
    "time_ps",
    "v_l2_in",
    "v_max_abs_in",
    "tcstat_ekinh_trace_sum",
    "tcstat_ekinh_old_trace_sum",
    "tcstat_ekinf_trace_sum",
    "ekind_ekin_trace",
    "ekind_ekin_l2",
    "dekindl",
    "dekindl_old",
    "dvdl_ekin",
    "kinetic_energy_kj",
    "temperature_k",
    "total_energy_term_kj",
    "conserved_energy_term_kj",
]
STAGES = [
    "before_calc_ke_part",
    "after_calc_ke_part",
    "after_gstat_block",
    "after_sum_ekin",
]
TP18I_BASELINE_MDOUT = TP18I_RESULTS / "raw_safe_pme_shift_ref_mdout.mdp"
REUSE_FIELDS = [
    "integrator",
    "dt",
    "nsteps",
    "nstlist",
    "rlist",
    "verlet-buffer-tolerance",
    "coulombtype",
    "coulomb-modifier",
    "vdw-type",
    "pcoupl",
    "tcoupl",
    "pme-order",
    "fourierspacing",
]
SHORT_RANGE_FIXED_FIELDS = [
    "integrator",
    "dt",
    "nsteps",
    "nstlist",
    "rlist",
    "verlet-buffer-tolerance",
    "coulomb-modifier",
    "vdw-type",
    "rvdw",
    "rcoulomb",
    "pcoupl",
    "tcoupl",
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


def command_to_string(
    cmd: list[str],
    cwd: pathlib.Path,
    stdin: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
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
                "call_index": int(row["call_index"]),
                "step": int(row["step"]),
            }
            for field in TRACE_STRING_FIELDS:
                parsed[field] = row[field]
            for field in TRACE_INT_FIELDS:
                parsed[field] = int(row[field])
            for field in TRACE_FLOAT_FIELDS:
                parsed[field] = float(row[field])
            rows.append(parsed)
    if not rows:
        raise RuntimeError(f"No trace rows found in {path}")
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def max_abs(values: list[float]) -> float:
    return max((abs(value) for value in values), default=0.0)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def compare_mdout_fields(a: pathlib.Path, b: pathlib.Path, fields: list[str]) -> dict[str, str | bool | None]:
    result: dict[str, str | bool | None] = {}
    for field in fields:
        result[field] = (read_mdout_value(a, field) == read_mdout_value(b, field))
    return result


def rows_by_stage(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped = {stage: [] for stage in STAGES}
    extra_stage_rows: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        stage = str(row["stage"])
        if stage in grouped:
            grouped[stage].append(row)
        else:
            extra_stage_rows.setdefault(stage, []).append(row)
    grouped.update(extra_stage_rows)
    return grouped


def summarize_stage_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    early_rows = [row for row in rows if 0 <= int(row["step"]) <= EARLY_STEP_MAX]
    summary: dict[str, object] = {
        "row_count": len(rows),
        "early_row_count": len(early_rows),
        "step_min": min((int(row["step"]) for row in rows), default=None),
        "step_max": max((int(row["step"]) for row in rows), default=None),
    }
    for field in TRACE_INT_FIELDS:
        values = [int(row[field]) for row in rows]
        summary[f"{field}_any"] = any(values)
        summary[f"{field}_all"] = all(values)
    for field in TRACE_FLOAT_FIELDS:
        full_values = [float(row[field]) for row in rows]
        early_values = [float(row[field]) for row in early_rows]
        summary[f"mean_{field}_full"] = mean(full_values)
        summary[f"mean_{field}_early"] = mean(early_values)
        summary[f"max_abs_{field}_full"] = max_abs(full_values)
        summary[f"max_abs_{field}_early"] = max_abs(early_values)
    return summary


def summarize_trace(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped = rows_by_stage(rows)
    callsite_names = sorted({str(row["callsite"]) for row in rows})
    prestep_row_count = sum(1 for row in rows if int(row["step"]) < 0)
    return {
        "row_count": len(rows),
        "callsite_names": callsite_names,
        "prestep_row_count": prestep_row_count,
        "isolated_post_update_only": (callsite_names == [CALLSITE_NAME] and prestep_row_count == 0),
        "stage_names": [stage for stage in STAGES if grouped.get(stage)],
        "stages": {stage: summarize_stage_rows(stage_rows) for stage, stage_rows in grouped.items() if stage_rows},
    }


def ensure_isolated_post_update_trace(summary: dict[str, object], run_id: str) -> None:
    if not summary["isolated_post_update_only"]:
        raise RuntimeError(
            f"{run_id} did not isolate only the post-update compute_globals callsite: "
            f"callsites={summary['callsite_names']} prestep_row_count={summary['prestep_row_count']}"
        )


def compare_stage_rows(reference_rows: list[dict[str, object]], variant_rows: list[dict[str, object]]) -> dict[str, object]:
    shared_count = min(len(reference_rows), len(variant_rows))
    shared_reference = reference_rows[:shared_count]
    shared_variant = variant_rows[:shared_count]
    early_pairs = [
        (ref_row, var_row)
        for ref_row, var_row in zip(shared_reference, shared_variant)
        if 0 <= int(ref_row["step"]) <= EARLY_STEP_MAX
    ]
    summary: dict[str, object] = {
        "reference_row_count": len(reference_rows),
        "variant_row_count": len(variant_rows),
        "shared_row_count": shared_count,
        "early_shared_row_count": len(early_pairs),
        "step_mismatch_count": sum(
            1
            for ref_row, var_row in zip(shared_reference, shared_variant)
            if int(ref_row["step"]) != int(var_row["step"])
        ),
        "callsite_mismatch_count": sum(
            1
            for ref_row, var_row in zip(shared_reference, shared_variant)
            if str(ref_row["callsite"]) != str(var_row["callsite"])
        ),
    }
    for field in TRACE_INT_FIELDS:
        summary[f"{field}_mismatch_count"] = sum(
            1
            for ref_row, var_row in zip(shared_reference, shared_variant)
            if int(ref_row[field]) != int(var_row[field])
        )
    for field in TRACE_FLOAT_FIELDS:
        summary[f"mean_abs_delta_{field}_full"] = mean(
            [
                abs(float(ref_row[field]) - float(var_row[field]))
                for ref_row, var_row in zip(shared_reference, shared_variant)
            ]
        )
        summary[f"mean_abs_delta_{field}_early"] = mean(
            [abs(float(ref_row[field]) - float(var_row[field])) for ref_row, var_row in early_pairs]
        )
    return summary


def compare_traces(reference_rows: list[dict[str, object]], variant_rows: list[dict[str, object]]) -> dict[str, object]:
    reference_grouped = rows_by_stage(reference_rows)
    variant_grouped = rows_by_stage(variant_rows)
    shared_stages = [stage for stage in STAGES if reference_grouped.get(stage) and variant_grouped.get(stage)]
    return {
        "shared_stage_names": shared_stages,
        "stages": {
            stage: compare_stage_rows(reference_grouped[stage], variant_grouped[stage]) for stage in shared_stages
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
    env = {"GMX_TP18J_TRACE_FILE": str(trace_output)}
    mdrun_cmd = [str(GMX), "mdrun", "-s", "run.tpr", "-deffnm", "run", "-nt", "1"]
    commands.append(command_to_string(mdrun_cmd, work_dir, env=env))
    mdrun = run_command(mdrun_cmd, work_dir, RESULTS_DIR / f"raw_{run_id}_mdrun.log", f"{run_id} mdrun", env=env, check=False)
    if not (work_dir / "run.edr").exists():
        raise RuntimeError(f"{run_id} did not produce run.edr (returncode={mdrun.returncode})")
    if not trace_output.exists():
        raise RuntimeError(f"{run_id} did not produce TP1.8j trace output")
    shutil.copy2(work_dir / "run.log", RESULTS_DIR / f"raw_{run_id}_md.log")

    energy_cmd = [str(GMX), "energy", "-f", "run.edr", "-o", "energy.xvg"]
    commands.append(command_to_string(energy_cmd, work_dir, ENERGY_STDIN))
    run_command(energy_cmd, work_dir, RESULTS_DIR / f"raw_{run_id}_energy_output.txt", f"{run_id} energy", ENERGY_STDIN)
    shutil.copy2(work_dir / "energy.xvg", RESULTS_DIR / f"raw_{run_id}_energy.xvg")

    trace_rows = parse_trace_csv(trace_output)
    trace_summary = summarize_trace(trace_rows)
    ensure_isolated_post_update_trace(trace_summary, run_id)
    energy_summary = summarize_energy(parse_xvg(work_dir / "energy.xvg"))
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
        "mdout_tcoupl": read_mdout_value(mdout_path, "tcoupl"),
        "runtime_ordinary_ewald_line": extract_line(log_path, "Will do ordinary reciprocal space Ewald sum."),
        "runtime_solve_pme_line": extract_line(log_path, "Solve PME"),
        "runtime_pairlist_line": extract_line(log_path, "updated every"),
        "runtime_kernel_line": extract_line(log_path, "Using plain-C-4x4 4x4 nonbonded short-range kernels"),
        "trace_output_path": str(trace_output.relative_to(ROOT)),
        "energy_summary": energy_summary,
        "trace_summary": trace_summary,
    }
    return {"summary": summary, "trace_rows": trace_rows}


def build_path_map(summary: dict[str, object]) -> dict[str, object]:
    baseline_trace = summary["baseline"]["trace_summary"]
    return {
        "milestone": "TP1.8j",
        "authoritative_system_id": "dense_salt_polymer",
        "reference_run_id": "safe_pme_shift_ref",
        "expected_isolated_callsite": CALLSITE_NAME,
        "paths": [
            {
                "file": "src/gromacs/mdrun/md.cpp",
                "function": "LegacySimulator::do_md",
                "callsite": "main post-update compute_globals invocation guarded by bGStat || needHalfStepKineticEnergy || doInterSimSignal",
                "role": "The only compute_globals callsite TP1.8j keeps active for tracing.",
                "why_traced": "TP1.8i mixed this call with pre-step compute_globals invocations, so TP1.8j adds a scope marker here.",
                "status": "active_and_isolated",
            },
            {
                "file": "src/gromacs/mdrun/md.cpp",
                "function": "LegacySimulator::do_md",
                "callsite": "pre-step compute_globals initialization calls",
                "role": "Earlier compute_globals invocations that contaminated TP1.8i with step=-1 rows.",
                "why_traced": "TP1.8j excludes them explicitly to isolate only the post-update slice.",
                "status": "excluded_by_scope_marker",
            },
            {
                "file": "src/gromacs/mdlib/md_support.cpp",
                "function": "compute_globals",
                "callsite": "ScopedTp18jPostUpdateComputeGlobalsTrace gate",
                "role": "Trace-only filter that writes TP1.8j rows only for the post-update callsite.",
                "why_traced": "This is the narrowest change that isolates a single compute_globals invocation without changing production logic.",
                "status": "active_and_traced",
            },
            {
                "file": "src/gromacs/mdlib/md_support.cpp",
                "function": "calc_ke_part / calc_ke_part_normal",
                "callsite": "within isolated post-update compute_globals slice",
                "role": "Computes simulator-owned kinetic tensor state from post-update velocities.",
                "why_traced": "Needed to distinguish bounded carry-through from slice-local amplification.",
                "status": "active_and_traced",
            },
            {
                "file": "src/gromacs/mdlib/tgroup.cpp",
                "function": "sum_ekin",
                "callsite": "within isolated post-update compute_globals slice",
                "role": "Produces simulator-owned kinetic energy and temperature outputs at the isolated slice.",
                "why_traced": "This is the actual consumer output TP1.8j compares between PME and Ewald.",
                "status": "active_and_traced"
                if baseline_trace["stages"].get("after_sum_ekin")
                else "uncertain",
            },
        ],
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

    after_calc = comparison["stages"]["after_calc_ke_part"]
    after_sum = comparison["stages"]["after_sum_ekin"]
    baseline_after_sum = baseline["summary"]["trace_summary"]["stages"]["after_sum_ekin"]
    kinetic_ratio = safe_ratio(
        float(after_sum["mean_abs_delta_kinetic_energy_kj_early"]),
        float(baseline_after_sum["mean_kinetic_energy_kj_early"]),
    )
    temperature_ratio = safe_ratio(
        float(after_sum["mean_abs_delta_temperature_k_early"]),
        float(baseline_after_sum["mean_temperature_k_early"]),
    )
    kinetic_delta_amplification = safe_ratio(
        float(after_sum["mean_abs_delta_kinetic_energy_kj_early"]),
        float(after_calc["mean_abs_delta_kinetic_energy_kj_early"]),
    )
    slice_interpretation = (
        "bounded_carry_through_only"
        if (kinetic_ratio or 0.0) < 0.01 and (temperature_ratio or 0.0) < 0.01
        else "mixed_or_potential_amplification"
    )
    continue_tracing = "NO" if slice_interpretation == "bounded_carry_through_only" else "PARTIAL"

    summary = {
        "milestone": "TP1.8j",
        "baseline": baseline["summary"],
        "variant": variant["summary"],
        "baseline_reuses_tp18i_safe_settings": all(
            compare_mdout_fields(RESULTS_DIR / "raw_safe_pme_shift_ref_mdout.mdp", TP18I_BASELINE_MDOUT, REUSE_FIELDS).values()
        ),
        "baseline_vs_tp18i_field_match": compare_mdout_fields(
            RESULTS_DIR / "raw_safe_pme_shift_ref_mdout.mdp", TP18I_BASELINE_MDOUT, REUSE_FIELDS
        ),
        "short_range_fields_fixed_across_runs": all(
            compare_mdout_fields(
                RESULTS_DIR / "raw_safe_pme_shift_ref_mdout.mdp",
                RESULTS_DIR / "raw_safe_ewald_shift_mdout.mdp",
                SHORT_RANGE_FIXED_FIELDS,
            ).values()
        ),
        "baseline_vs_variant_short_range_match": compare_mdout_fields(
            RESULTS_DIR / "raw_safe_pme_shift_ref_mdout.mdp",
            RESULTS_DIR / "raw_safe_ewald_shift_mdout.mdp",
            SHORT_RANGE_FIXED_FIELDS,
        ),
        "comparison": comparison,
        "isolated_slice_checks": {
            "expected_callsite": CALLSITE_NAME,
            "baseline_isolated_post_update_only": baseline["summary"]["trace_summary"]["isolated_post_update_only"],
            "variant_isolated_post_update_only": variant["summary"]["trace_summary"]["isolated_post_update_only"],
            "baseline_prestep_row_count": baseline["summary"]["trace_summary"]["prestep_row_count"],
            "variant_prestep_row_count": variant["summary"]["trace_summary"]["prestep_row_count"],
            "baseline_callsite_names": baseline["summary"]["trace_summary"]["callsite_names"],
            "variant_callsite_names": variant["summary"]["trace_summary"]["callsite_names"],
        },
        "isolated_slice_metrics": {
            "after_sum_ekin_kinetic_energy_delta_ratio_early": kinetic_ratio,
            "after_sum_ekin_temperature_delta_ratio_early": temperature_ratio,
            "after_sum_vs_after_calc_kinetic_delta_ratio_early": kinetic_delta_amplification,
            "slice_interpretation": slice_interpretation,
        },
        "final_classification": "still_unresolved",
    }

    recommendation = {
        "milestone": "TP1.8j",
        "source_patching_now_justified": False,
        "whether_tracing_should_continue_after_tp1_8j": continue_tracing,
        "final_classification": "still_unresolved",
        "exact_next_step_recommendation": (
            "Stop at unresolved unless there is a concrete, active consumer after the isolated post-update "
            "compute_globals slice that can be shown to transform the bounded PME-vs-Ewald difference."
            if continue_tracing == "NO"
            else "If tracing continues, target only a later active consumer with clear evidence that it transforms the isolated slice output."
        ),
    }

    run_matrix = {
        "milestone": "TP1.8j",
        "authoritative_system_id": "dense_salt_polymer",
        "tp13_baseline_source": str(TP13_DIR.relative_to(ROOT)),
        "tp18i_baseline_source": str(TP18I_RESULTS.relative_to(ROOT)),
        "reference_run_id": "safe_pme_shift_ref",
        "comparison_runs": RUNS,
        "rationale_for_20ps_window": "TP1.8i used the same 20 ps authoritative window with runaway onset at 0.2 ps, so TP1.8j preserves that window while isolating only the post-update compute_globals slice.",
    }

    write_text(RESULTS_DIR / "run_matrix.json", json.dumps(run_matrix, indent=2) + "\n")
    write_text(RESULTS_DIR / "slice_trace_summary.json", json.dumps(summary, indent=2) + "\n")
    write_text(RESULTS_DIR / "callsite_path_map.json", json.dumps(build_path_map(summary), indent=2) + "\n")
    write_text(RESULTS_DIR / "tp1_8j_recommendation.json", json.dumps(recommendation, indent=2) + "\n")
    write_text(RESULTS_DIR / "raw_commands.txt", "\n".join(commands) + "\n")

    provenance = {
        "milestone": "TP1.8j",
        "execution_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["rev-parse", "HEAD"]).strip(),
        "git_status_short": git_output(["status", "--short"]).splitlines(),
        "gmx_version": gmx_version_text().splitlines()[0],
        "inputs": {
            "tp13_baseline": str(TP13_DIR.relative_to(ROOT)),
            "tp18i_results": str(TP18I_RESULTS.relative_to(ROOT)),
        },
    }
    write_text(RESULTS_DIR / "provenance_manifest.json", json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
