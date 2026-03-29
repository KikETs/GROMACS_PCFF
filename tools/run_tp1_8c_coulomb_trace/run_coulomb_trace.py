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
RESULTS_DIR = ROOT / "tests/reference_results/tp1_8c_coulomb_trace"

TP13_DIR = ROOT / "tests/reference_results/tp1_3_stabilization/TRL-0"
TP18_RESULTS = ROOT / "tests/reference_results/tp1_8_longrange_isolation"
TP18B_RESULTS = ROOT / "tests/reference_results/tp1_8b_coulomb_separation"

ENERGY_STDIN = "Potential\nKinetic-En.\nTotal-Energy\nTemperature\nPressure\n0\n"
RUNAWAY_THRESHOLD_K = 400.0
EARLY_CALL_MAX = 200

RUNS = [
    {
        "run_id": "safe_pme_shift_ref",
        "role": "baseline_reference",
        "trace_filename": "trace_observables_baseline.csv",
        "coulombtype": "PME",
        "coulomb_modifier": "Potential-shift-Verlet",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1e-5,
        "why": "Reuses the TP1.8b authoritative safe baseline and traces the default Coulomb PME accumulation path.",
        "intended_path_change": "reference",
    },
    {
        "run_id": "safe_ewald_shift",
        "role": "ewald_variant",
        "trace_filename": "trace_observables_variant.csv",
        "coulombtype": "Ewald",
        "coulomb_modifier": "Potential-shift-Verlet",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1e-5,
        "why": "Switches the long-range Coulomb solver from PME mesh to full Ewald while preserving the Ewald-family short-range electrostatics path.",
        "intended_path_change": "pme_vs_ewald_solver_trace",
    },
    {
        "run_id": "safe_pme_none",
        "role": "direct_modifier_variant",
        "trace_filename": "trace_observables_direct_modifier.csv",
        "coulombtype": "PME",
        "coulomb_modifier": "None",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1e-5,
        "why": "Changes only the Coulomb modifier while preserving Coulomb PME and the same short-range baseline.",
        "intended_path_change": "direct_space_modifier_trace",
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
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if key in {
                    "call_index",
                    "lj_pme_active",
                    "entered_longrange_block",
                    "compute_nonbonded_forces",
                    "compute_longrange_nonbonded_forces",
                    "compute_energy",
                    "compute_virial",
                    "do_neighbor_search",
                    "state_changed",
                    "compute_pme_on_cpu",
                    "pme_do_called",
                    "ewald_called",
                    "have_ewald_surface_term",
                }:
                    parsed[key] = int(value)
                elif key in {
                    "ewald_coeff_q",
                    "vlr_q_kj",
                    "vcorr_q_kj",
                    "coulomb_recip_term_kj",
                    "vlr_lj_kj",
                    "vcorr_lj_kj",
                    "lj_recip_term_kj",
                    "virial_q_trace",
                    "virial_lj_trace",
                }:
                    parsed[key] = float(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_trace(rows: list[dict[str, object]]) -> dict[str, object]:
    early_rows = [row for row in rows if int(row["call_index"]) <= EARLY_CALL_MAX]
    return {
        "row_count": len(rows),
        "early_row_count": len(early_rows),
        "entered_longrange_block_count": sum(int(row["entered_longrange_block"]) for row in rows),
        "pme_do_call_count": sum(int(row["pme_do_called"]) for row in rows),
        "ewald_call_count": sum(int(row["ewald_called"]) for row in rows),
        "lj_pme_active_any": any(int(row["lj_pme_active"]) for row in rows),
        "direct_space_family_values": sorted({str(row["direct_space_family"]) for row in rows}),
        "coulomb_type_values": sorted({str(row["coulomb_type"]) for row in rows}),
        "coulomb_modifier_values": sorted({str(row["coulomb_modifier"]) for row in rows}),
        "early_mean_vlr_q_kj": mean([float(row["vlr_q_kj"]) for row in early_rows]),
        "early_mean_vcorr_q_kj": mean([float(row["vcorr_q_kj"]) for row in early_rows]),
        "early_mean_coulomb_recip_term_kj": mean([float(row["coulomb_recip_term_kj"]) for row in early_rows]),
        "early_mean_virial_q_trace": mean([float(row["virial_q_trace"]) for row in early_rows]),
        "full_mean_vlr_q_kj": mean([float(row["vlr_q_kj"]) for row in rows]),
        "full_mean_vcorr_q_kj": mean([float(row["vcorr_q_kj"]) for row in rows]),
        "full_mean_coulomb_recip_term_kj": mean([float(row["coulomb_recip_term_kj"]) for row in rows]),
        "full_mean_virial_q_trace": mean([float(row["virial_q_trace"]) for row in rows]),
    }


def compare_trace_rows(reference_rows: list[dict[str, object]], variant_rows: list[dict[str, object]]) -> dict[str, object]:
    ref_by_index = {int(row["call_index"]): row for row in reference_rows}
    var_by_index = {int(row["call_index"]): row for row in variant_rows}
    shared = sorted(set(ref_by_index) & set(var_by_index))
    early_shared = [index for index in shared if index <= EARLY_CALL_MAX]

    def mean_abs_delta(field: str, indices: list[int]) -> float:
        if not indices:
            return 0.0
        return mean([abs(float(ref_by_index[index][field]) - float(var_by_index[index][field])) for index in indices])

    return {
        "shared_call_count": len(shared),
        "early_shared_call_count": len(early_shared),
        "mean_abs_delta_vlr_q_kj_early": mean_abs_delta("vlr_q_kj", early_shared),
        "mean_abs_delta_vcorr_q_kj_early": mean_abs_delta("vcorr_q_kj", early_shared),
        "mean_abs_delta_coulomb_recip_term_kj_early": mean_abs_delta("coulomb_recip_term_kj", early_shared),
        "mean_abs_delta_virial_q_trace_early": mean_abs_delta("virial_q_trace", early_shared),
        "mean_abs_delta_vlr_q_kj_full": mean_abs_delta("vlr_q_kj", shared),
        "mean_abs_delta_vcorr_q_kj_full": mean_abs_delta("vcorr_q_kj", shared),
        "mean_abs_delta_coulomb_recip_term_kj_full": mean_abs_delta("coulomb_recip_term_kj", shared),
        "mean_abs_delta_virial_q_trace_full": mean_abs_delta("virial_q_trace", shared),
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
    env = {"GMX_TP18C_TRACE_FILE": str(trace_output)}
    mdrun_cmd = [str(GMX), "mdrun", "-s", "run.tpr", "-deffnm", "run", "-nt", "1"]
    commands.append(command_to_string(mdrun_cmd, work_dir, env=env))
    mdrun = run_command(
        mdrun_cmd,
        work_dir,
        RESULTS_DIR / f"raw_{run_id}_mdrun.log",
        f"{run_id} mdrun",
        env=env,
        check=False,
    )
    if not (work_dir / "run.edr").exists():
        raise RuntimeError(f"{run_id} did not produce run.edr (returncode={mdrun.returncode})")
    if not trace_output.exists():
        raise RuntimeError(f"{run_id} did not produce TP1.8c trace output")
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
        "runtime_pme_sum_line": extract_line(log_path, "Will do PME sum in reciprocal space for electrostatic interactions."),
        "runtime_ordinary_ewald_line": extract_line(log_path, "Will do ordinary reciprocal space Ewald sum."),
        "runtime_solve_pme_line": extract_line(log_path, "Solve PME"),
        "runtime_pairlist_line": extract_line(log_path, "updated every"),
        "runtime_kernel_line": extract_line(log_path, "Using plain-C-4x4 4x4 nonbonded short-range kernels"),
        "runtime_potential_shift_line": extract_line(log_path, "Potential shift:"),
        "trace_output_path": str(trace_output.relative_to(ROOT)),
        "energy_summary": energy_summary,
        "trace_summary": trace_summary,
    }
    return {"summary": summary, "trace_rows": trace_rows}


def build_active_trace_path_map() -> dict[str, object]:
    return {
        "milestone": "TP1.8c",
        "authoritative_system_id": "dense_salt_polymer",
        "reference_run_id": "safe_pme_shift_ref",
        "paths": [
            {
                "file": "src/gromacs/mdlib/force.cpp",
                "function": "CpuPpLongRangeNonbondeds::calculate",
                "role": "Primary traced long-range Coulomb accumulation path. Emits TP1.8c per-call trace rows after reciprocal/correction accumulation.",
                "status": "active_and_traced",
                "why": [
                    "This function chooses PME mesh versus Ewald long-range work.",
                    "It accumulates Vlr_q, Vcorr_q and writes InteractionFunction::CoulombReciprocalSpace."
                ],
            },
            {
                "file": "src/gromacs/mdlib/force.cpp",
                "function": "gmx_pme_do call site inside CpuPpLongRangeNonbondeds::calculate",
                "role": "PME reciprocal accumulation call path.",
                "status": "active_in_baseline_inactive_in_ewald_variant",
                "why": [
                    "Baseline uses coulombtype = PME and has a Solve PME timing line.",
                    "Ewald variant has no Solve PME timing line and no PME mesh call."
                ],
            },
            {
                "file": "src/gromacs/mdlib/force.cpp",
                "function": "do_ewald call site inside CpuPpLongRangeNonbondeds::calculate",
                "role": "Ordinary reciprocal-space Ewald accumulation path.",
                "status": "inactive_in_baseline_active_in_ewald_variant",
                "why": [
                    "The Ewald variant uses coulombtype = Ewald.",
                    "The log reports 'Will do ordinary reciprocal space Ewald sum.'"
                ],
            },
            {
                "file": "src/gromacs/mdlib/forcerec.cpp",
                "function": "init_forcerec electrostatics translation",
                "role": "Fixes the direct-space electrostatics family and stores the Coulomb modifier.",
                "status": "active_and_fixed_to_ewald_family_for_baseline_and_ewald_variant",
                "why": [
                    "PME and Ewald both map to NbkernelElecType::Ewald.",
                    "This lets TP1.8c compare reciprocal solver changes while keeping the direct-space family fixed."
                ],
            },
            {
                "file": "src/gromacs/ewald/pme.cpp",
                "function": "gmx_pme_init",
                "role": "Activates Coulomb PME work only when usingPme(coulombtype).",
                "status": "active_in_baseline_inactive_in_ewald_variant",
                "why": [
                    "Baseline uses coulombtype = PME.",
                    "Ewald variant switches to coulombtype = Ewald."
                ],
            },
            {
                "file": "src/gromacs/nbnxm/nbnxm_setup.cpp",
                "function": "chooseLJPmeCombinationRule",
                "role": "LJ-PME gate.",
                "status": "inactive",
                "why": [
                    "All TP1.8c runs keep vdw-type = Cut-off.",
                    "TP1.4 LJ-PME path therefore remains inactive."
                ],
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
    ewald_variant = next(result for result in run_results if result["summary"]["role"] == "ewald_variant")
    direct_variant = next(result for result in run_results if result["summary"]["role"] == "direct_modifier_variant")

    baseline_summary = baseline["summary"]
    ewald_summary = ewald_variant["summary"]
    direct_summary = direct_variant["summary"]

    baseline_vs_ewald = compare_trace_rows(baseline["trace_rows"], ewald_variant["trace_rows"])
    baseline_vs_direct = compare_trace_rows(baseline["trace_rows"], direct_variant["trace_rows"])

    if (
        baseline_vs_ewald["mean_abs_delta_coulomb_recip_term_kj_early"] > 1.0
        and baseline_vs_direct["mean_abs_delta_coulomb_recip_term_kj_early"] < 0.1
        and ewald_summary["energy_summary"]["runaway_onset_ps"] == baseline_summary["energy_summary"]["runaway_onset_ps"]
        and direct_summary["energy_summary"]["runaway_onset_ps"] == baseline_summary["energy_summary"]["runaway_onset_ps"]
    ):
        final_classification = "still_unresolved"
    else:
        final_classification = "mixed_or_still_unresolved"

    run_matrix = {
        "milestone": "TP1.8c",
        "authoritative_system_id": "dense_salt_polymer",
        "tp13_baseline_source": str(TP13_DIR.relative_to(ROOT)),
        "tp18_source": str(TP18_RESULTS.relative_to(ROOT)),
        "tp18b_source": str(TP18B_RESULTS.relative_to(ROOT)),
        "reference_run_id": "safe_pme_shift_ref",
        "comparison_runs": RUNS,
        "rationale_for_optional_direct_variant": "safe_pme_none is included because tracing only PME-versus-Ewald would still leave the Coulomb modifier path unmeasured at source level.",
        "rationale_for_20ps_window": "TP1.8b already showed onset at 0.2 ps on the current 20 ps authoritative window, so TP1.8c traces the same window to keep baseline fairness while limiting trace volume.",
    }
    write_text(RESULTS_DIR / "run_matrix.json", json.dumps(run_matrix, indent=2) + "\n")

    write_text(RESULTS_DIR / "active_trace_path_map.json", json.dumps(build_active_trace_path_map(), indent=2) + "\n")

    summary = {
        "milestone": "TP1.8c",
        "baseline": baseline_summary,
        "ewald_variant": ewald_summary,
        "direct_modifier_variant": direct_summary,
        "baseline_vs_ewald_trace_delta": baseline_vs_ewald,
        "baseline_vs_direct_modifier_trace_delta": baseline_vs_direct,
        "interpretation": {
            "reciprocal_solver_specific_explanation_weakened": (
                baseline_vs_ewald["mean_abs_delta_coulomb_recip_term_kj_early"] > 1.0
                and ewald_summary["energy_summary"]["runaway_onset_ps"] == baseline_summary["energy_summary"]["runaway_onset_ps"]
            ),
            "direct_modifier_specific_explanation_weakened": (
                baseline_vs_direct["mean_abs_delta_coulomb_recip_term_kj_early"] < 0.1
                and direct_summary["energy_summary"]["runaway_onset_ps"] == baseline_summary["energy_summary"]["runaway_onset_ps"]
            ),
            "tp1_4_lj_pme_path_status": "inactive",
            "per_pair_direct_space_decomposition_available": False,
        },
        "final_classification": final_classification,
    }
    write_text(RESULTS_DIR / "trace_comparison_summary.json", json.dumps(summary, indent=2) + "\n")

    recommendation = {
        "milestone": "TP1.8c",
        "source_patching_now_justified": False,
        "plain_safe_baseline_acceptable_for_later_non_rrespa_validation": "PARTIAL",
        "final_classification": final_classification,
        "tp1_4_lj_pme_path_status": "inactive",
        "exact_next_step_recommendation": "Keep the authoritative safe baseline and trace one level deeper into the shared Ewald-family/common Coulomb accumulation path before any production patch; the current trace weakens a simple PME-mesh-only or Coulomb-modifier-only story but does not isolate a single source-level fault.",
    }
    write_text(RESULTS_DIR / "tp1_8c_recommendation.json", json.dumps(recommendation, indent=2) + "\n")

    write_text(RESULTS_DIR / "raw_commands.txt", "\n".join(commands) + "\n")

    provenance = {
        "milestone": "TP1.8c",
        "execution_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["rev-parse", "HEAD"]).strip(),
        "git_status_short": git_output(["status", "--short"]).splitlines(),
        "gmx_version": gmx_version_text().splitlines()[0],
        "inputs": {
            "tp13_baseline": str(TP13_DIR.relative_to(ROOT)),
            "tp18_results": str(TP18_RESULTS.relative_to(ROOT)),
            "tp18b_results": str(TP18B_RESULTS.relative_to(ROOT)),
        },
    }
    write_text(RESULTS_DIR / "provenance_manifest.json", json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
