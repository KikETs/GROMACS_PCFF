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
RESULTS_DIR = ROOT / "tests/reference_results/tp1_8_longrange_isolation"

TP14_REPORT = ROOT / "docs/validation_report_tp1_4.md"
TP17B_RESULTS = ROOT / "tests/reference_results/tp1_7b_authoritative_ab"
TP17B_REPORT = ROOT / "docs/validation_report_tp1_7b.md"
TP13_DIR = ROOT / "tests/reference_results/tp1_3_stabilization/TRL-0"

ENERGY_STDIN = "Potential\nKinetic-En.\nTotal-Energy\nTemperature\nPressure\n0\n"
RUNAWAY_THRESHOLD_K = 400.0

RUNS = [
    {
        "run_id": "safe_pme_n10_r0911",
        "role": "safe_baseline_reference",
        "coulombtype": "PME",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1e-5,
        "why": "Reuses the TP1.7b runtime-distinct safe short-range authoritative baseline without changing the short-range regime.",
        "expected_runtime_pairlist_line": "updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm",
        "expected_longrange_runtime": "pme_coulomb",
    },
    {
        "run_id": "safe_pme_tight_fs006_po6",
        "role": "pme_accuracy_variant",
        "coulombtype": "PME",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 6,
        "fourierspacing": 0.06,
        "ewald_rtol": 1e-6,
        "why": "Tightens Coulomb PME reciprocal accuracy while holding the short-range safe baseline fixed.",
        "expected_runtime_pairlist_line": "updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm",
        "expected_longrange_runtime": "pme_coulomb",
    },
    {
        "run_id": "safe_cutoff_n10_r0911",
        "role": "no_reciprocal_cutoff_variant",
        "coulombtype": "Cut-off",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "ewald_rtol": 1e-5,
        "why": "Removes reciprocal-space Coulomb treatment while keeping the short-range safe baseline and cut-off radii fixed, to test whether the surviving runaway is strongly tied to the active PME path.",
        "expected_runtime_pairlist_line": "updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm",
        "expected_longrange_runtime": "cutoff_no_reciprocal",
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


def extract_line(path: pathlib.Path, pattern: str) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if pattern in line:
            return line.strip()
    return None


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


def mdp_text(run: dict[str, object]) -> str:
    lines = [
        "integrator = md",
        "dt = 0.001",
        "nsteps = 100000",
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
        f"coulombtype = {run['coulombtype']}",
        "coulomb-modifier = Potential-shift-Verlet",
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
        lines.extend(
            [
                f"pme-order = {run['pme_order']}",
                f"fourierspacing = {run['fourierspacing']}",
            ]
        )
    return "\n".join(lines) + "\n"


def classify_effect(reference: dict[str, object], variant: dict[str, object]) -> str:
    if variant["status"] != "RUNAWAY":
        return "disappears"

    reference_onset = reference["runaway_onset_ps"]
    variant_onset = variant["runaway_onset_ps"]
    onset_delay = None
    if reference_onset is not None and variant_onset is not None:
        onset_delay = variant_onset - reference_onset

    max_temp_ratio = float(variant["max_temperature_k"]) / float(reference["max_temperature_k"])
    energy_range_ratio = float(variant["total_energy_range_kj"]) / float(reference["total_energy_range_kj"])
    pressure_ratio = float(variant["max_abs_pressure_bar"]) / float(reference["max_abs_pressure_bar"])

    if onset_delay is not None and onset_delay >= 25.0 and max_temp_ratio <= 0.90:
        return "weakens_materially"
    if max_temp_ratio <= 0.90 and energy_range_ratio <= 0.70 and pressure_ratio <= 0.90:
        return "weakens_materially"
    return "persists"


def longrange_runtime_class(log_path: pathlib.Path) -> str:
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Will do PME sum in reciprocal space for electrostatic interactions." in log_text:
        return "pme_coulomb"
    if "Solve PME" in log_text:
        return "pme_coulomb"
    return "non_pme_coulomb"


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    log_path = work_dir / "run.log"
    summary.update(
        {
            "run_id": run_id,
            "role": run["role"],
            "executed_now": True,
            "coulombtype": run["coulombtype"],
            "nstlist": int(run["nstlist"]),
            "rlist": float(run["rlist"]),
            "verlet_buffer_tolerance": float(run["verlet_buffer_tolerance"]),
            "why": run["why"],
            "expected_runtime_pairlist_line": run["expected_runtime_pairlist_line"],
            "expected_longrange_runtime": run["expected_longrange_runtime"],
            "mdrun_returncode": mdrun.returncode,
            "runtime_repulsion_line": extract_line(log_path, "Detected LJ repulsion power 9."),
            "runtime_kernel_line": extract_line(log_path, "Using plain-C-4x4 4x4 nonbonded short-range kernels"),
            "runtime_pairlist_line": extract_line(log_path, "updated every"),
            "runtime_longrange_line": extract_line(log_path, "Will do PME sum in reciprocal space for electrostatic interactions."),
            "runtime_longrange_class": longrange_runtime_class(log_path),
            "runtime_solve_pme_line": extract_line(log_path, "Solve PME"),
        }
    )
    if run["coulombtype"] == "PME":
        summary["pme_order"] = int(run["pme_order"])
        summary["fourierspacing"] = float(run["fourierspacing"])
    else:
        summary["pme_order"] = None
        summary["fourierspacing"] = None
    return summary


def build_active_path_map(reference: dict[str, object]) -> dict[str, object]:
    return {
        "milestone": "TP1.8",
        "authoritative_system_id": "dense_salt_polymer",
        "safe_reference_run_id": reference["run_id"],
        "runtime_reference_facts": {
            "mdout_source": str((RESULTS_DIR / f"raw_{reference['run_id']}_mdout.mdp").relative_to(ROOT)),
            "log_source": str((RESULTS_DIR / f"raw_{reference['run_id']}_md.log").relative_to(ROOT)),
            "coulombtype": reference["coulombtype"],
            "runtime_longrange_class": reference["runtime_longrange_class"],
            "runtime_pairlist_line": reference["runtime_pairlist_line"],
            "runtime_kernel_line": reference["runtime_kernel_line"],
            "runtime_repulsion_line": reference["runtime_repulsion_line"],
        },
        "paths": [
            {
                "file": "src/gromacs/ewald/pme.cpp",
                "function": "gmx_pme_init",
                "physical_role": "Initializes Coulomb PME reciprocal-space work and independently toggles LJ-PME.",
                "status": "active_for_coulomb_in_safe_reference",
                "why": [
                    "TP1.7b-safe mdout uses coulombtype = PME.",
                    "TP1.7b-safe mdout uses vdw-type = Cut-off.",
                    "TP1.7b-safe log says 'Will do PME sum in reciprocal space for electrostatic interactions.'",
                    "Source sets pme->doCoulomb = usingPme(ir->coulombtype) and pme->doLJ = usingLJPme(ir->vdwtype).",
                ],
                "evidence": [
                    "tests/reference_results/tp1_7b_authoritative_ab/raw_safe_n10_r0911_mdout.mdp",
                    "tests/reference_results/tp1_7b_authoritative_ab/raw_safe_n10_r0911_md.log",
                    "src/gromacs/ewald/pme.cpp:782",
                    "src/gromacs/ewald/pme.cpp:783",
                ],
            },
            {
                "file": "src/gromacs/nbnxm/kerneldispatch.cpp",
                "function": "getCoulombKernelType",
                "physical_role": "Chooses the short-range Coulomb direct-space kernel family.",
                "status": "active_for_real_space_split_logic",
                "why": [
                    "For PME or Ewald electrostatics, the short-range direct-space kernel stays in the Ewald/EwaldTwin family rather than ReactionField.",
                    "This means TP1.8 can change the reciprocal solver while keeping the short-range Coulomb family fixed.",
                ],
                "evidence": [
                    "src/gromacs/nbnxm/kerneldispatch.cpp:113",
                    "src/gromacs/nbnxm/kerneldispatch.cpp:154",
                    "tests/reference_results/tp1_7b_authoritative_ab/raw_safe_n10_r0911_mdout.mdp",
                ],
            },
            {
                "file": "src/gromacs/nbnxm/kerneldispatch.cpp",
                "function": "getVdwKernelType",
                "physical_role": "Chooses the short-range VdW kernel family.",
                "status": "active_cutoff_vdw_lj_pme_inactive",
                "why": [
                    "Safe authoritative mdout uses vdw-type = Cut-off, so the Cut-off VdW kernel is selected.",
                    "This excludes LJ-PME reciprocal correction from the active authoritative path.",
                ],
                "evidence": [
                    "src/gromacs/nbnxm/kerneldispatch.cpp:164",
                    "src/gromacs/nbnxm/kerneldispatch.cpp:170",
                    "tests/reference_results/tp1_7b_authoritative_ab/raw_safe_n10_r0911_mdout.mdp",
                ],
            },
            {
                "file": "src/gromacs/nbnxm/nbnxm_setup.cpp",
                "function": "chooseLJPmeCombinationRule",
                "physical_role": "Prepares the LJ-PME grid combination rule only when vdw-type = Pme.",
                "status": "inactive_for_safe_reference",
                "why": [
                    "Source only enters LJ-PME combination-rule logic when forcerec.ic->vdw.type == VanDerWaalsType::Pme.",
                    "Safe authoritative mdout uses vdw-type = Cut-off.",
                ],
                "evidence": [
                    "src/gromacs/nbnxm/nbnxm_setup.cpp:462",
                    "src/gromacs/nbnxm/nbnxm_setup.cpp:464",
                    "tests/reference_results/tp1_7b_authoritative_ab/raw_safe_n10_r0911_mdout.mdp",
                ],
            },
            {
                "file": "src/gromacs/nbnxm/atomdata.cpp",
                "function": "setParamCombinationRule",
                "physical_role": "Builds LJ-PME dispersion-grid parameters for the PCFF 9-6 path only when LJ-PME is active.",
                "status": "inactive_for_safe_reference",
                "why": [
                    "The PCFF 9-6 LJ-PME grid branch is guarded by usingLJPme.",
                    "Safe authoritative mdout uses vdw-type = Cut-off, so this TP1.4-style reciprocal LJ path is inactive here.",
                ],
                "evidence": [
                    "src/gromacs/nbnxm/atomdata.cpp:400",
                    "src/gromacs/nbnxm/atomdata.cpp:401",
                    "tests/reference_results/tp1_7b_authoritative_ab/raw_safe_n10_r0911_mdout.mdp",
                    "docs/validation_report_tp1_4.md",
                ],
            },
            {
                "file": "src/gromacs/mdlib/forcerec.cpp",
                "function": "makeNonBondedParameterLists / init_forcerec path",
                "physical_role": "Disables SIMD kernels for repulsion power 9 and keeps the plain-C reference short-range kernels active.",
                "status": "active_short_range_family_but_not_lj_pme_proof_path",
                "why": [
                    "Safe authoritative log reports repulsion power 9 and plain-C kernels.",
                    "This is a short-range family fact, not evidence that LJ-PME reciprocal correction is active.",
                ],
                "evidence": [
                    "src/gromacs/mdlib/forcerec.cpp:892",
                    "src/gromacs/mdlib/forcerec.cpp:908",
                    "tests/reference_results/tp1_7b_authoritative_ab/raw_safe_n10_r0911_md.log",
                ],
            },
        ],
        "tp1_4_path_status_for_authoritative_setup": "inactive",
    }


def overall_classification(reference: dict[str, object], tight: dict[str, object], cutoff: dict[str, object]) -> str:
    if cutoff["effect_vs_reference"] in {"disappears", "weakens_materially"} and tight["effect_vs_reference"] == "persists":
        return "more_strongly_coulomb_long_range_or_reciprocal_related"
    if tight["effect_vs_reference"] in {"disappears", "weakens_materially"} or cutoff["effect_vs_reference"] in {"disappears", "weakens_materially"}:
        return "more_strongly_long_range_or_mixed"
    return "mixed_or_still_unresolved"


def recommendation_text(classification: str) -> str:
    if classification == "more_strongly_coulomb_long_range_or_reciprocal_related":
        return (
            "Keep the same-build authoritative safe short-range baseline and isolate the Coulomb reciprocal/PME path next, "
            "not the LJ-PME path that is inactive here."
        )
    if classification == "more_strongly_long_range_or_mixed":
        return (
            "Keep the same-build authoritative safe short-range baseline and next isolate the surviving Coulomb long-range path "
            "with a narrower reciprocal-space vs direct-Ewald comparison before any code patch."
        )
    return (
        "Keep the safe short-range baseline, but do not claim a purely PME blocker yet; the next step should be a narrower "
        "Coulomb long-range vs mixed-path audit on this same authoritative tier."
    )


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
    reference = next(run for run in run_results if run["role"] == "safe_baseline_reference")

    for run in run_results:
        if run is reference:
            run["effect_vs_reference"] = "reference"
        else:
            run["effect_vs_reference"] = classify_effect(reference, run)

    tight = next(run for run in run_results if run["role"] == "pme_accuracy_variant")
    cutoff = next(run for run in run_results if run["role"] == "no_reciprocal_cutoff_variant")
    blocker_classification = overall_classification(reference, tight, cutoff)

    active_path_map = build_active_path_map(reference)
    write_text(RESULTS_DIR / "active_path_map.json", json.dumps(active_path_map, indent=2) + "\n")

    run_matrix = {
        "milestone": "TP1.8",
        "authoritative_system_id": "dense_salt_polymer",
        "authoritative_system_source": "tests/reference_results/m10_4_charged_ensemble_gate/dense_salt_polymer",
        "tp1_3_executed_baseline_source": "tests/reference_results/tp1_3_stabilization/TRL-0",
        "tp1_7b_same_build_source": "tests/reference_results/tp1_7b_authoritative_ab",
        "reference_run_id": reference["run_id"],
        "comparison_runs": RUNS,
    }
    write_text(RESULTS_DIR / "run_matrix.json", json.dumps(run_matrix, indent=2) + "\n")

    comparison_rows: list[dict[str, object]] = []
    for run in run_results:
        comparison_rows.append(
            {
                "run_id": run["run_id"],
                "role": run["role"],
                "coulombtype": run["coulombtype"],
                "nstlist": run["nstlist"],
                "rlist": run["rlist"],
                "verlet_buffer_tolerance": run["verlet_buffer_tolerance"],
                "pme_order": run["pme_order"],
                "fourierspacing": run["fourierspacing"],
                "runtime_longrange_class": run["runtime_longrange_class"],
                "runtime_pairlist_line": run["runtime_pairlist_line"],
                "status": run["status"],
                "runaway_onset_ps": run["runaway_onset_ps"],
                "max_temperature_k": run["max_temperature_k"],
                "final_temperature_k": run["final_temperature_k"],
                "total_energy_range_kj": run["total_energy_range_kj"],
                "max_abs_total_energy_drift_kj": run["max_abs_total_energy_drift_kj"],
                "max_abs_pressure_bar": run["max_abs_pressure_bar"],
                "effect_vs_reference": run["effect_vs_reference"],
            }
        )
    write_csv(
        RESULTS_DIR / "longrange_variant_comparison.csv",
        list(comparison_rows[0].keys()),
        comparison_rows,
    )

    summary = {
        "milestone": "TP1.8",
        "reference_run": reference,
        "variants": [tight, cutoff],
        "comparison_metrics": {
            "pme_accuracy_variant": {
                "runaway_effect_classification": tight["effect_vs_reference"],
                "runaway_onset_delay_ps": None
                if tight["runaway_onset_ps"] is None or reference["runaway_onset_ps"] is None
                else float(tight["runaway_onset_ps"]) - float(reference["runaway_onset_ps"]),
                "max_temperature_delta_k": float(tight["max_temperature_k"]) - float(reference["max_temperature_k"]),
                "total_energy_range_delta_kj": float(tight["total_energy_range_kj"])
                - float(reference["total_energy_range_kj"]),
                "max_abs_pressure_delta_bar": float(tight["max_abs_pressure_bar"])
                - float(reference["max_abs_pressure_bar"]),
            },
            "no_reciprocal_cutoff_variant": {
                "runaway_effect_classification": cutoff["effect_vs_reference"],
                "runaway_onset_delay_ps": None
                if cutoff["runaway_onset_ps"] is None or reference["runaway_onset_ps"] is None
                else float(cutoff["runaway_onset_ps"]) - float(reference["runaway_onset_ps"]),
                "max_temperature_delta_k": float(cutoff["max_temperature_k"]) - float(reference["max_temperature_k"]),
                "total_energy_range_delta_kj": float(cutoff["total_energy_range_kj"])
                - float(reference["total_energy_range_kj"]),
                "max_abs_pressure_delta_bar": float(cutoff["max_abs_pressure_bar"])
                - float(reference["max_abs_pressure_bar"]),
            },
        },
        "remaining_blocker_classification": blocker_classification,
        "tp1_4_path_status_for_authoritative_setup": "inactive",
    }
    write_text(RESULTS_DIR / "stability_summary.json", json.dumps(summary, indent=2) + "\n")

    recommendation = {
        "milestone": "TP1.8",
        "source_patching_now_justified": False,
        "plain_safe_baseline_acceptable_for_later_non_rrespa_validation": "PARTIAL",
        "tp1_4_path_status_for_authoritative_setup": "inactive",
        "remaining_blocker_classification": blocker_classification,
        "exact_next_step_recommendation": recommendation_text(blocker_classification),
    }
    write_text(RESULTS_DIR / "tp1_8_recommendation.json", json.dumps(recommendation, indent=2) + "\n")

    write_text(RESULTS_DIR / "raw_commands.txt", "\n".join(commands) + "\n")

    provenance = {
        "milestone": "TP1.8",
        "execution_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["rev-parse", "HEAD"]).strip(),
        "git_status_short": git_output(["status", "--short"]).splitlines(),
        "gmx_version": gmx_version_text().splitlines()[0],
        "inputs": {
            "tp14_report": str(TP14_REPORT.relative_to(ROOT)),
            "tp17b_report": str(TP17B_REPORT.relative_to(ROOT)),
            "tp17b_results": str(TP17B_RESULTS.relative_to(ROOT)),
            "tp13_baseline": str(TP13_DIR.relative_to(ROOT)),
        },
        "artifacts": {
            "active_path_map": "tests/reference_results/tp1_8_longrange_isolation/active_path_map.json",
            "run_matrix": "tests/reference_results/tp1_8_longrange_isolation/run_matrix.json",
            "comparison": "tests/reference_results/tp1_8_longrange_isolation/longrange_variant_comparison.csv",
            "summary": "tests/reference_results/tp1_8_longrange_isolation/stability_summary.json",
            "recommendation": "tests/reference_results/tp1_8_longrange_isolation/tp1_8_recommendation.json",
        },
    }
    write_text(RESULTS_DIR / "provenance_manifest.json", json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
