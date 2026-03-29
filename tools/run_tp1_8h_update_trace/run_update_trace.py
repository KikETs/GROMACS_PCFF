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
RESULTS_DIR = ROOT / "tests/reference_results/tp1_8h_update_trace"

TP13_DIR = ROOT / "tests/reference_results/tp1_3_stabilization/TRL-0"
TP18G_RESULTS = ROOT / "tests/reference_results/tp1_8g_consumer_trace"

ENERGY_STDIN = "Potential\nKinetic-En.\nTotal-Energy\nTemperature\nPressure\n0\n"
RUNAWAY_THRESHOLD_K = 400.0
EARLY_STEP_MAX = 200

RUNS = [
    {
        "run_id": "safe_pme_shift_ref",
        "role": "baseline_reference",
        "trace_filename": "update_trace_baseline.csv",
        "coulombtype": "PME",
        "coulomb_modifier": "Potential-shift-Verlet",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1e-5,
        "why": "Reuses the TP1.8g authoritative safe baseline settings for integration-state tracing.",
        "intended_path_change": "reference",
    },
    {
        "run_id": "safe_ewald_shift",
        "role": "ewald_variant",
        "trace_filename": "update_trace_variant.csv",
        "coulombtype": "Ewald",
        "coulomb_modifier": "Potential-shift-Verlet",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1e-5,
        "why": "Switches the Coulomb solver from PME to Ewald while keeping the same authoritative short-range baseline and active update path.",
        "intended_path_change": "pme_vs_ewald_integration_update_trace",
    },
]

TRACE_STR_FIELDS = [
    "integrator",
    "update_part",
    "helper_path",
    "parrinello_rahman_velocity_scaling",
    "temperature_group_mode",
]

TRACE_INT_FIELDS = [
    "pcoupl_is_no",
    "tcoupl_is_no",
    "have_partially_frozen_atoms",
    "have_constraints",
    "do_temp_couple",
    "do_nose_hoover",
    "using_simd_path",
]

TRACE_FLOAT_FIELDS = [
    "time_ps",
    "force_l2_in",
    "force_max_abs_in",
    "v_l2_before",
    "v_max_abs_before",
    "v_l2_after",
    "v_max_abs_after",
    "delta_v_l2",
    "delta_v_max_abs",
    "xprime_l2_after",
    "xprime_max_abs_after",
    "delta_xprime_from_x_l2",
    "delta_xprime_from_x_max_abs",
    "kinetic_proxy_before",
    "kinetic_proxy_after",
    "delta_kinetic_proxy",
]

TRACE_COMPARE_FIELDS = [
    "force_l2_in",
    "force_max_abs_in",
    "delta_v_l2",
    "delta_v_max_abs",
    "delta_xprime_from_x_l2",
    "delta_xprime_from_x_max_abs",
    "kinetic_proxy_after",
    "delta_kinetic_proxy",
]

TP18G_BASELINE_MDOUT = TP18G_RESULTS / "raw_safe_pme_shift_ref_mdout.mdp"
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
                "call_index": int(row["call_index"]),
                "step": int(row["step"]),
            }
            for field in TRACE_STR_FIELDS:
                parsed[field] = row[field]
            for field in TRACE_INT_FIELDS:
                parsed[field] = int(row[field])
            for field in TRACE_FLOAT_FIELDS:
                parsed[field] = float(row[field])
            rows.append(parsed)
    if not rows:
        raise RuntimeError(f"No trace rows found in {path}")
    return rows


def mean_abs_delta(
    baseline_rows: list[dict[str, object]],
    variant_rows: list[dict[str, object]],
    field: str,
    *,
    early_only: bool,
) -> float:
    values: list[float] = []
    for baseline_row, variant_row in zip(baseline_rows, variant_rows):
        if early_only and int(baseline_row["step"]) > EARLY_STEP_MAX:
            continue
        values.append(abs(float(variant_row[field]) - float(baseline_row[field])))
    if not values:
        return 0.0
    return sum(values) / len(values)


def count_mismatches(
    baseline_rows: list[dict[str, object]],
    variant_rows: list[dict[str, object]],
    field: str,
) -> int:
    mismatches = 0
    for baseline_row, variant_row in zip(baseline_rows, variant_rows):
        if baseline_row[field] != variant_row[field]:
            mismatches += 1
    return mismatches


def unique_values(rows: list[dict[str, object]], field: str) -> list[str]:
    return sorted({str(row[field]) for row in rows})


def compare_mdout_fields(a: pathlib.Path, b: pathlib.Path, fields: list[str]) -> dict[str, str | bool | None]:
    result: dict[str, str | bool | None] = {}
    for field in fields:
        a_value = read_mdout_value(a, field)
        b_value = read_mdout_value(b, field)
        result[field] = (a_value == b_value)
    return result


def build_trace_summary(
    baseline_rows: list[dict[str, object]],
    variant_rows: list[dict[str, object]],
    baseline_mdout: pathlib.Path,
    variant_mdout: pathlib.Path,
    baseline_energy_summary: dict[str, object],
    variant_energy_summary: dict[str, object],
) -> dict[str, object]:
    if len(baseline_rows) != len(variant_rows):
        raise RuntimeError("Trace row count mismatch between baseline and variant")

    step_mismatch_count = count_mismatches(baseline_rows, variant_rows, "step")
    update_part_mismatch_count = count_mismatches(baseline_rows, variant_rows, "update_part")
    helper_path_mismatch_count = count_mismatches(baseline_rows, variant_rows, "helper_path")

    summary: dict[str, object] = {
        "baseline_row_count": len(baseline_rows),
        "variant_row_count": len(variant_rows),
        "step_mismatch_count": step_mismatch_count,
        "update_part_mismatch_count": update_part_mismatch_count,
        "helper_path_mismatch_count": helper_path_mismatch_count,
        "helper_path_baseline_unique": unique_values(baseline_rows, "helper_path"),
        "helper_path_variant_unique": unique_values(variant_rows, "helper_path"),
        "integrator_baseline_unique": unique_values(baseline_rows, "integrator"),
        "integrator_variant_unique": unique_values(variant_rows, "integrator"),
        "using_simd_path_baseline_unique": sorted({int(row["using_simd_path"]) for row in baseline_rows}),
        "using_simd_path_variant_unique": sorted({int(row["using_simd_path"]) for row in variant_rows}),
        "baseline_reuses_tp18g_safe_settings": all(compare_mdout_fields(baseline_mdout, TP18G_BASELINE_MDOUT, REUSE_FIELDS).values()),
        "baseline_vs_tp18g_field_match": compare_mdout_fields(baseline_mdout, TP18G_BASELINE_MDOUT, REUSE_FIELDS),
        "short_range_fields_fixed_across_runs": all(compare_mdout_fields(baseline_mdout, variant_mdout, SHORT_RANGE_FIXED_FIELDS).values()),
        "baseline_vs_variant_short_range_match": compare_mdout_fields(baseline_mdout, variant_mdout, SHORT_RANGE_FIXED_FIELDS),
        "baseline_energy_summary": baseline_energy_summary,
        "variant_energy_summary": variant_energy_summary,
        "overall_classification": "still_unresolved",
    }

    for field in TRACE_COMPARE_FIELDS:
        summary[f"early_mean_abs_delta_{field}"] = mean_abs_delta(baseline_rows, variant_rows, field, early_only=True)
        summary[f"full_mean_abs_delta_{field}"] = mean_abs_delta(baseline_rows, variant_rows, field, early_only=False)

    return summary


def main() -> None:
    if not GMX.exists():
        raise SystemExit(f"Missing GROMACS binary: {GMX}")

    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    RESULTS_DIR.mkdir(parents=True)
    WORK_DIR.mkdir(parents=True)

    raw_commands_path = RESULTS_DIR / "raw_commands.txt"
    raw_commands: list[str] = []

    run_matrix = {
        "milestone": "TP1.8h",
        "title": "Integration-State Update Trace Under Fixed Safe Baseline",
        "system": "dense_salt_polymer",
        "reference_source": str(TP13_DIR),
        "baseline_reused_from": "TP1.8g safe_pme_shift_ref settings",
        "runs": RUNS,
    }
    write_text(RESULTS_DIR / "run_matrix.json", json.dumps(run_matrix, indent=2) + "\n")

    run_summaries: dict[str, dict[str, object]] = {}

    for run in RUNS:
        run_id = str(run["run_id"])
        run_dir = WORK_DIR / run_id
        run_dir.mkdir(parents=True)

        shutil.copy2(TP13_DIR / "min.gro", run_dir / "conf.gro")
        shutil.copy2(TP13_DIR / "system.top", run_dir / "topol.top")
        write_text(run_dir / "md.mdp", mdp_text(run))

        trace_path = RESULTS_DIR / str(run["trace_filename"])
        grompp_log = RESULTS_DIR / f"raw_{run_id}_grompp.log"
        mdrun_log = RESULTS_DIR / f"raw_{run_id}_mdrun.log"
        energy_log = RESULTS_DIR / f"raw_{run_id}_energy_output.txt"

        grompp_cmd = [
            str(GMX),
            "grompp",
            "-f",
            "md.mdp",
            "-c",
            "conf.gro",
            "-p",
            "topol.top",
            "-o",
            "md.tpr",
            "-maxwarn",
            "1",
        ]
        raw_commands.append(command_to_string(grompp_cmd, run_dir))
        run_command(grompp_cmd, run_dir, grompp_log, f"{run_id} grompp")

        mdrun_cmd = [
            str(GMX),
            "mdrun",
            "-deffnm",
            "md",
            "-ntmpi",
            "1",
            "-ntomp",
            "1",
        ]
        mdrun_env = {"GMX_TP18H_TRACE_FILE": str(trace_path)}
        raw_commands.append(command_to_string(mdrun_cmd, run_dir, env=mdrun_env))
        run_command(mdrun_cmd, run_dir, mdrun_log, f"{run_id} mdrun", env=mdrun_env)

        energy_cmd = [str(GMX), "energy", "-f", "md.edr", "-o", f"{run_id}_energy.xvg"]
        raw_commands.append(command_to_string(energy_cmd, run_dir, stdin=ENERGY_STDIN))
        run_command(energy_cmd, run_dir, energy_log, f"{run_id} energy", stdin=ENERGY_STDIN)

        shutil.copy2(run_dir / "mdout.mdp", RESULTS_DIR / f"raw_{run_id}_mdout.mdp")
        shutil.copy2(run_dir / "md.log", RESULTS_DIR / f"raw_{run_id}_md.log")
        shutil.copy2(run_dir / f"{run_id}_energy.xvg", RESULTS_DIR / f"raw_{run_id}_energy.xvg")

        series = parse_xvg(run_dir / f"{run_id}_energy.xvg")
        energy_summary = summarize_energy(series)

        run_summaries[run_id] = {
            "run_id": run_id,
            "role": run["role"],
            "trace_path": str(trace_path),
            "mdout_path": str(RESULTS_DIR / f"raw_{run_id}_mdout.mdp"),
            "pairlist_runtime_line": extract_line(RESULTS_DIR / f"raw_{run_id}_md.log", "updated every"),
            "pmse_runtime_line": extract_line(RESULTS_DIR / f"raw_{run_id}_md.log", "Solve PME"),
            "ewald_runtime_line": extract_line(RESULTS_DIR / f"raw_{run_id}_md.log", "ordinary reciprocal space Ewald"),
            "energy_summary": energy_summary,
        }

    write_text(raw_commands_path, "\n".join(raw_commands) + "\n")

    baseline_rows = parse_trace_csv(RESULTS_DIR / "update_trace_baseline.csv")
    variant_rows = parse_trace_csv(RESULTS_DIR / "update_trace_variant.csv")

    baseline_mdout = RESULTS_DIR / "raw_safe_pme_shift_ref_mdout.mdp"
    variant_mdout = RESULTS_DIR / "raw_safe_ewald_shift_mdout.mdp"

    trace_summary = build_trace_summary(
        baseline_rows,
        variant_rows,
        baseline_mdout,
        variant_mdout,
        run_summaries["safe_pme_shift_ref"]["energy_summary"],
        run_summaries["safe_ewald_shift"]["energy_summary"],
    )
    write_text(RESULTS_DIR / "update_trace_summary.json", json.dumps(trace_summary, indent=2) + "\n")

    path_map = {
        "milestone": "TP1.8h",
        "scope": "active update/integration path after force computation under pcoupl = no",
        "paths": [
            {
                "file": "src/gromacs/mdrun/md.cpp",
                "function": "LegacySimulator::do_md",
                "role": "Calls upd.update_coords after do_force and before compute_globals.",
                "why_traced": "Caller boundary that forwards the surviving PME-vs-Ewald force difference into update.",
                "status": "active",
            },
            {
                "file": "src/gromacs/mdlib/update.cpp",
                "function": "Update::Impl::update_coords",
                "role": "Entry point for integration-state updates and TP1.8h trace boundary.",
                "why_traced": "Captures pre/post aggregate state for v and xprime under the active authoritative path.",
                "status": "active",
            },
            {
                "file": "src/gromacs/mdlib/update.cpp",
                "function": "do_update_md",
                "role": "Leap-frog MD dispatcher for the active integrator.",
                "why_traced": "Determines whether the surviving difference takes the simple or general MD update path.",
                "status": "active",
            },
            {
                "file": "src/gromacs/mdlib/update.cpp",
                "function": "updateMDLeapfrogSimple / updateMDLeapfrogSimpleSimd",
                "role": "Applies v <- v + f*dt/m and xprime <- x + v*dt in the simple leap-frog path.",
                "why_traced": "Most direct place where force-side differences can carry into velocity and coordinate updates.",
                "status": "active" if "md_leapfrog_simple" in ",".join(unique_values(baseline_rows, "helper_path")) else "uncertain",
            },
            {
                "file": "src/gromacs/mdlib/update.cpp",
                "function": "updateMDLeapfrogGeneral",
                "role": "General leap-frog path for NH, anisotropic PR, or acceleration.",
                "why_traced": "Needed only to prove that the authoritative TP1.8h runs do not take this broader path.",
                "status": "inactive" if unique_values(baseline_rows, "helper_path") == unique_values(variant_rows, "helper_path") and "md_leapfrog_general" not in unique_values(baseline_rows, "helper_path") else "uncertain",
            },
            {
                "file": "src/gromacs/mdlib/update.cpp",
                "function": "doUpdateMDDoNotUpdateVelocities",
                "role": "Constraint-virial-only xprime update without storing velocities.",
                "why_traced": "Would be a confounder if constraints were active.",
                "status": "inactive" if max(int(row["have_constraints"]) for row in baseline_rows + variant_rows) == 0 else "uncertain",
            },
        ],
    }
    write_text(RESULTS_DIR / "update_path_map.json", json.dumps(path_map, indent=2) + "\n")

    recommendation = {
        "milestone": "TP1.8h",
        "source_patching_now_justified": False,
        "plain_safe_baseline_acceptable_for_later_non_rrespa_validation": "PARTIAL",
        "overall_classification": "still_unresolved",
        "next_step_recommendation": "If one final trace step is required, target the next active kinetic/temperature consumer after update rather than returning to inactive pressure-control branches.",
    }
    write_text(RESULTS_DIR / "tp1_8h_recommendation.json", json.dumps(recommendation, indent=2) + "\n")

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "gmx_version": gmx_version_text(),
        "git_head": git_output(["rev-parse", "HEAD"]).strip(),
        "git_status_short": git_output(["status", "--short"]).splitlines(),
    }
    write_text(RESULTS_DIR / "provenance_manifest.json", json.dumps(provenance, indent=2) + "\n")

    summary_table_rows = [
        {
            "run_id": run_id,
            "role": data["role"],
            "pairlist_runtime_line": data["pairlist_runtime_line"],
            "pmse_runtime_line": data["pmse_runtime_line"],
            "ewald_runtime_line": data["ewald_runtime_line"],
            "energy_summary": data["energy_summary"],
        }
        for run_id, data in run_summaries.items()
    ]
    write_text(RESULTS_DIR / "run_summaries.json", json.dumps(summary_table_rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
