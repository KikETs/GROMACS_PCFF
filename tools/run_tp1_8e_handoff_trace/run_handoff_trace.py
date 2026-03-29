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
RESULTS_DIR = ROOT / "tests/reference_results/tp1_8e_handoff_trace"

TP13_DIR = ROOT / "tests/reference_results/tp1_3_stabilization/TRL-0"
TP18C_RESULTS = ROOT / "tests/reference_results/tp1_8c_coulomb_trace"
TP18D_RESULTS = ROOT / "tests/reference_results/tp1_8d_coulomb_consumer_trace"

ENERGY_STDIN = "Potential\nKinetic-En.\nTotal-Energy\nTemperature\nPressure\n0\n"
RUNAWAY_THRESHOLD_K = 400.0
EARLY_STEP_MAX = 200

RUNS = [
    {
        "run_id": "safe_pme_shift_ref",
        "role": "baseline_reference",
        "trace_filename": "handoff_trace_baseline.csv",
        "coulombtype": "PME",
        "coulomb_modifier": "Potential-shift-Verlet",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1e-5,
        "why": "Reuses the TP1.8d authoritative safe baseline settings for post-postprocess handoff tracing.",
        "intended_path_change": "reference",
    },
    {
        "run_id": "safe_ewald_shift",
        "role": "ewald_variant",
        "trace_filename": "handoff_trace_variant.csv",
        "coulombtype": "Ewald",
        "coulomb_modifier": "Potential-shift-Verlet",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1e-5,
        "why": "Switches the Coulomb solver from PME to Ewald while keeping the same authoritative short-range baseline.",
        "intended_path_change": "pme_vs_ewald_later_handoff_trace",
    },
]

TRACE_FIELDS = [
    "force_l2",
    "force_max_abs",
    "state_x_l2",
    "state_x_max_abs",
    "state_v_l2",
    "state_v_max_abs",
    "xprime_l2",
    "xprime_max_abs",
    "force_vir_trace",
    "shake_vir_trace",
    "total_vir_trace",
    "pres_trace",
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
    ]
    if run["coulombtype"] == "PME":
        lines.extend([f"pme-order = {run['pme_order']}", f"fourierspacing = {run['fourierspacing']}"])
    else:
        # Keep the same emitted mdout fields where possible for fairness documentation.
        lines.extend([f"pme-order = {run['pme_order']}", f"fourierspacing = {run['fourierspacing']}"])
    return "\n".join(lines) + "\n"


def read_mdout_value(mdout: pathlib.Path, key: str) -> str | None:
    prefix = f"{key:<25}"
    for line in mdout.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    for line in mdout.read_text(encoding="utf-8", errors="replace").splitlines():
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
                "stage": row["stage"],
                "using_mts_combined_force": int(row["using_mts_combined_force"]),
            }
            for field in TRACE_FIELDS:
                parsed[field] = float(row[field])
            rows.append(parsed)
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_trace(rows: list[dict[str, object]]) -> dict[str, object]:
    stages = sorted({str(row["stage"]) for row in rows})
    summary: dict[str, object] = {"row_count": len(rows), "stages": {}}
    for stage in stages:
        stage_rows = [row for row in rows if row["stage"] == stage]
        early_rows = [row for row in stage_rows if int(row["step"]) <= EARLY_STEP_MAX]
        summary["stages"][stage] = {
            "row_count": len(stage_rows),
            "early_row_count": len(early_rows),
            "using_mts_combined_force_any": any(int(row["using_mts_combined_force"]) for row in stage_rows),
        }
        for field in TRACE_FIELDS:
            summary["stages"][stage][f"mean_{field}_early"] = mean([float(row[field]) for row in early_rows])
            summary["stages"][stage][f"mean_{field}_full"] = mean([float(row[field]) for row in stage_rows])
    return summary


def compare_traces(reference_rows: list[dict[str, object]], variant_rows: list[dict[str, object]]) -> dict[str, object]:
    ref = {(int(row["step"]), str(row["stage"])): row for row in reference_rows}
    var = {(int(row["step"]), str(row["stage"])): row for row in variant_rows}
    shared_keys = sorted(set(ref) & set(var))
    stages = sorted({stage for _, stage in shared_keys})
    comparison: dict[str, object] = {"shared_row_count": len(shared_keys), "stages": {}}

    for stage in stages:
        stage_keys = [key for key in shared_keys if key[1] == stage]
        early_keys = [key for key in stage_keys if key[0] <= EARLY_STEP_MAX]
        stage_summary: dict[str, object] = {
            "shared_row_count": len(stage_keys),
            "early_shared_row_count": len(early_keys),
        }
        for field in TRACE_FIELDS:
            stage_summary[f"mean_abs_delta_{field}_early"] = mean(
                [abs(float(ref[key][field]) - float(var[key][field])) for key in early_keys]
            )
            stage_summary[f"mean_abs_delta_{field}_full"] = mean(
                [abs(float(ref[key][field]) - float(var[key][field])) for key in stage_keys]
            )
        comparison["stages"][stage] = stage_summary
    return comparison


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
    env = {"GMX_TP18E_TRACE_FILE": str(trace_output)}
    mdrun_cmd = [str(GMX), "mdrun", "-s", "run.tpr", "-deffnm", "run", "-nt", "1"]
    commands.append(command_to_string(mdrun_cmd, work_dir, env=env))
    mdrun = run_command(mdrun_cmd, work_dir, RESULTS_DIR / f"raw_{run_id}_mdrun.log", f"{run_id} mdrun", env=env, check=False)
    if not (work_dir / "run.edr").exists():
        raise RuntimeError(f"{run_id} did not produce run.edr (returncode={mdrun.returncode})")
    if not trace_output.exists():
        raise RuntimeError(f"{run_id} did not produce TP1.8e trace output")
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
        "runtime_ordinary_ewald_line": extract_line(log_path, "Will do ordinary reciprocal space Ewald sum."),
        "runtime_solve_pme_line": extract_line(log_path, "Solve PME"),
        "runtime_pairlist_line": extract_line(log_path, "updated every"),
        "runtime_kernel_line": extract_line(log_path, "Using plain-C-4x4 4x4 nonbonded short-range kernels"),
        "trace_output_path": str(trace_output.relative_to(ROOT)),
        "energy_summary": energy_summary,
        "trace_summary": trace_summary,
    }
    return {"summary": summary, "trace_rows": trace_rows}


def build_handoff_path_map() -> dict[str, object]:
    return {
        "milestone": "TP1.8e",
        "authoritative_system_id": "dense_salt_polymer",
        "reference_run_id": "safe_pme_shift_ref",
        "paths": [
            {
                "file": "src/gromacs/mdlib/force.cpp",
                "function": "CpuPpLongRangeNonbondeds::calculate",
                "role": "Upstream Coulomb producer boundary already separated in TP1.8c.",
                "status": "active_upstream_reference",
                "why": "TP1.8e does not re-open this boundary; it traces later handoffs.",
            },
            {
                "file": "src/gromacs/mdlib/sim_util.cpp",
                "function": "postProcessForces",
                "role": "Immediate downstream consumer already traced in TP1.8d.",
                "status": "active_upstream_reference",
                "why": "TP1.8e traces the next handoff after this consumer.",
            },
            {
                "file": "src/gromacs/mdrun/md.cpp",
                "function": "LegacySimulator::do_md",
                "role": "Caller-side handoff after do_force and before/after update and compute_globals.",
                "status": "active_and_traced",
                "why": "TP1.8e records after_do_force_return, before_update_coords, after_update_coords, and after_compute_globals here.",
            },
            {
                "file": "src/gromacs/mdlib/update.cpp",
                "function": "Update::update_coords",
                "role": "Consumes the final force buffer for integration-relevant position/velocity propagation.",
                "status": "active_and_traced_at_caller_boundary",
                "why": "TP1.8e traces the force buffer passed into update_coords and the immediate post-update state.",
            },
            {
                "file": "src/gromacs/mdlib/md_support.cpp",
                "function": "compute_globals",
                "role": "Consumes force_vir and shake_vir to form total_vir and pressure observables.",
                "status": "active_and_traced_at_caller_boundary",
                "why": "TP1.8e records the post-compute_globals handoff state to judge virial/pressure-side survival.",
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

    summary = {
        "milestone": "TP1.8e",
        "baseline": baseline["summary"],
        "variant": variant["summary"],
        "comparison": comparison,
        "interpretation": {
            "tp1_4_lj_pme_path_status": "inactive",
            "per_pair_force_or_component_decomposition_available": False,
        },
        "final_classification": "still_unresolved",
    }

    recommendation = {
        "milestone": "TP1.8e",
        "source_patching_now_justified": False,
        "plain_safe_baseline_acceptable_for_later_non_rrespa_validation": "PARTIAL",
        "final_classification": "still_unresolved",
        "tp1_4_lj_pme_path_status": "inactive",
        "exact_next_step_recommendation": "Keep the authoritative safe baseline fixed and only trace one level deeper into later force/virial state transfer if the TP1.8e handoff still leaves ambiguity; do not patch production logic yet.",
    }

    run_matrix = {
        "milestone": "TP1.8e",
        "authoritative_system_id": "dense_salt_polymer",
        "tp13_baseline_source": str(TP13_DIR.relative_to(ROOT)),
        "tp18c_source": str(TP18C_RESULTS.relative_to(ROOT)),
        "tp18d_source": str(TP18D_RESULTS.relative_to(ROOT)),
        "reference_run_id": "safe_pme_shift_ref",
        "comparison_runs": RUNS,
        "rationale_for_20ps_window": "TP1.8c and TP1.8d both used a 20 ps authoritative trace window with runaway onset at 0.2 ps, so TP1.8e keeps the same window while tracing one level deeper.",
    }

    write_text(RESULTS_DIR / "run_matrix.json", json.dumps(run_matrix, indent=2) + "\n")
    write_text(RESULTS_DIR / "handoff_path_map.json", json.dumps(build_handoff_path_map(), indent=2) + "\n")
    write_text(RESULTS_DIR / "handoff_trace_summary.json", json.dumps(summary, indent=2) + "\n")
    write_text(RESULTS_DIR / "tp1_8e_recommendation.json", json.dumps(recommendation, indent=2) + "\n")
    write_text(RESULTS_DIR / "raw_commands.txt", "\n".join(commands) + "\n")

    provenance = {
        "milestone": "TP1.8e",
        "execution_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["rev-parse", "HEAD"]).strip(),
        "git_status_short": git_output(["status", "--short"]).splitlines(),
        "gmx_version": gmx_version_text().splitlines()[0],
        "inputs": {
            "tp13_baseline": str(TP13_DIR.relative_to(ROOT)),
            "tp18c_results": str(TP18C_RESULTS.relative_to(ROOT)),
            "tp18d_results": str(TP18D_RESULTS.relative_to(ROOT)),
        },
    }
    write_text(RESULTS_DIR / "provenance_manifest.json", json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
