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
RESULTS_DIR = ROOT / "tests/reference_results/tp1_8b_coulomb_separation"

TP17B_RESULTS = ROOT / "tests/reference_results/tp1_7b_authoritative_ab"
TP18_RESULTS = ROOT / "tests/reference_results/tp1_8_longrange_isolation"
TP13_DIR = ROOT / "tests/reference_results/tp1_3_stabilization/TRL-0"

ENERGY_STDIN = "Potential\nKinetic-En.\nTotal-Energy\nTemperature\nPressure\n0\n"
RUNAWAY_THRESHOLD_K = 400.0

RUNS = [
    {
        "run_id": "safe_pme_shift_ref",
        "role": "safe_baseline_reference",
        "coulombtype": "PME",
        "coulomb_modifier": "Potential-shift-Verlet",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1e-5,
        "why": "Reuses the TP1.7b/TP1.8 authoritative safe short-range baseline and the same Coulomb PME baseline.",
        "intended_path_change": "reference",
    },
    {
        "run_id": "safe_pme_tight_mesh",
        "role": "reciprocal_accuracy_variant",
        "coulombtype": "PME",
        "coulomb_modifier": "Potential-shift-Verlet",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 6,
        "fourierspacing": 0.06,
        "ewald_rtol": 1e-6,
        "why": "Perturbs reciprocal-space Coulomb accuracy while keeping the PME/direct-space family and short-range baseline fixed.",
        "intended_path_change": "reciprocal_accuracy_only",
    },
    {
        "run_id": "safe_ewald_shift",
        "role": "reciprocal_solver_variant",
        "coulombtype": "Ewald",
        "coulomb_modifier": "Potential-shift-Verlet",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "ewald_rtol": 1e-5,
        "why": "Replaces Coulomb PME mesh handling with full Ewald while preserving the Ewald-family direct-space kernel and the short-range baseline.",
        "intended_path_change": "reciprocal_solver_only_narrower_than_cutoff",
    },
    {
        "run_id": "safe_pme_none",
        "role": "direct_space_modifier_variant",
        "coulombtype": "PME",
        "coulomb_modifier": "None",
        "nstlist": 10,
        "rlist": 0.911,
        "verlet_buffer_tolerance": -1,
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1e-5,
        "why": "Perturbs direct-space Coulomb shifting while keeping Coulomb PME reciprocal treatment and the short-range baseline fixed.",
        "intended_path_change": "direct_space_modifier_only",
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
        lines.extend(
            [
                f"pme-order = {run['pme_order']}",
                f"fourierspacing = {run['fourierspacing']}",
            ]
        )
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


def runtime_longrange_mode(run: dict[str, object], log_path: pathlib.Path) -> str:
    if str(run["coulombtype"]) == "Ewald":
        return "ewald_no_pme_mesh"
    if str(run["coulombtype"]) == "PME":
        return "pme_coulomb"
    if str(run["coulombtype"]) == "Cut-off":
        return "cutoff_no_reciprocal"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Will do PME sum in reciprocal space for electrostatic interactions." in log_text:
        return "pme_or_ewald_full_electrostatics"
    return "unknown"


def pme_banner_is_generic(mdout_coulombtype: str | None, log_path: pathlib.Path) -> bool:
    return (
        mdout_coulombtype in {"PME", "Ewald"}
        and extract_line(log_path, "Will do PME sum in reciprocal space for electrostatic interactions.") is not None
    )


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

    if onset_delay is not None and onset_delay >= 5.0 and max_temp_ratio <= 0.90:
        return "weakens_materially"
    if max_temp_ratio <= 0.90 and energy_range_ratio <= 0.70 and pressure_ratio <= 0.90:
        return "weakens_materially"
    return "persists"


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

    mdout_path = work_dir / "mdout.mdp"
    log_path = work_dir / "run.log"
    series = parse_xvg(work_dir / "energy.xvg")
    summary = summarize_series(series)
    summary.update(
        {
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
            "runtime_solve_pme_line": extract_line(log_path, "Solve PME"),
            "runtime_pairlist_line": extract_line(log_path, "updated every"),
            "runtime_repulsion_line": extract_line(log_path, "Detected LJ repulsion power 9."),
            "runtime_kernel_line": extract_line(log_path, "Using plain-C-4x4 4x4 nonbonded short-range kernels"),
            "runtime_potential_shift_line": extract_line(log_path, "Potential shift:"),
        }
    )
    summary["runtime_coulomb_mode"] = runtime_longrange_mode(run, log_path)
    summary["runtime_pme_banner_is_generic_for_usingPmeOrEwald"] = pme_banner_is_generic(
        summary["mdout_coulombtype"], log_path
    )
    return summary


def build_active_path_map() -> dict[str, object]:
    return {
        "milestone": "TP1.8b",
        "authoritative_system_id": "dense_salt_polymer",
        "reference_run_id": "safe_pme_shift_ref",
        "paths": [
            {
                "file": "src/gromacs/ewald/pme.cpp",
                "function": "gmx_pme_init",
                "physical_role": "Enables Coulomb PME reciprocal-space work when coulombtype = PME.",
                "control_that_may_alter_it": "coulombtype = PME versus Ewald",
                "status": "active_in_reference",
                "why": [
                    "Reference mdout uses coulombtype = PME.",
                    "Source sets pme->doCoulomb = usingPme(ir->coulombtype).",
                ],
            },
            {
                "file": "src/gromacs/mdlib/force.cpp",
                "function": "CpuPpLongRangeNonbondeds::calculate",
                "physical_role": "Chooses CPU long-range Coulomb calculation path: PME solve versus do_ewald.",
                "control_that_may_alter_it": "coulombtype = PME versus Ewald",
                "status": "active_and_tunable",
                "why": [
                    "PME path uses computePmeOnCpu.",
                    "Ewald path calls do_ewald when coulombInteractionType_ == Ewald.",
                ],
            },
            {
                "file": "src/gromacs/mdlib/forcerec.cpp",
                "function": "init_forcerec electrostatics translation",
                "physical_role": "Sets short-range electrostatics interaction family and modifier.",
                "control_that_may_alter_it": "coulombtype and coulomb-modifier",
                "status": "active_and_tunable",
                "why": [
                    "PME and Ewald both map to NbkernelElecType::Ewald.",
                    "The Coulomb modifier is stored separately in nbkernel_elec_modifier.",
                ],
            },
            {
                "file": "src/gromacs/nbnxm/kerneldispatch.cpp",
                "function": "getCoulombKernelType",
                "physical_role": "Selects the direct-space Coulomb kernel family.",
                "control_that_may_alter_it": "coulombtype",
                "status": "active_and_tunable",
                "why": [
                    "PME/Ewald select the Ewald/EwaldTwin family.",
                    "Cut-off would switch to ReactionField, which is why TP1.8's cut-off variant was too mixed.",
                ],
            },
            {
                "file": "src/gromacs/mdtypes/interaction_const.cpp",
                "function": "initCoulombEwaldParameters",
                "physical_role": "Sets Ewald coefficient and direct-space Coulomb shift term, and prints a generic reciprocal-space banner for PME or Ewald.",
                "control_that_may_alter_it": "ewald-rtol and coulomb-modifier",
                "status": "active_and_tunable",
                "why": [
                    "ewald-rtol changes beta/splitting.",
                    "coulomb-modifier = PotShift versus None changes ewaldShift without changing coulombtype.",
                    "The log string 'Will do PME sum in reciprocal space for electrostatic interactions.' is emitted for usingPmeOrEwald and is therefore not sufficient to distinguish PME mesh from full Ewald.",
                ],
            },
            {
                "file": "src/gromacs/nbnxm/nbnxm_setup.cpp",
                "function": "chooseLJPmeCombinationRule",
                "physical_role": "Activates LJ-PME-specific reciprocal path only for vdwtype = Pme.",
                "control_that_may_alter_it": "vdw-type",
                "status": "inactive_in_reference",
                "why": [
                    "Reference mdout uses vdw-type = Cut-off.",
                    "TP1.4-style LJ-PME path therefore remains inactive here.",
                ],
            },
        ],
        "tp1_4_lj_pme_path_status": "inactive",
    }


def make_runtime_distinct_check(reference: dict[str, object], variants: list[dict[str, object]]) -> dict[str, object]:
    variant_checks: list[dict[str, object]] = []
    for variant in variants:
        changed_coulombtype = variant["mdout_coulombtype"] != reference["mdout_coulombtype"]
        changed_modifier = variant["mdout_coulomb_modifier"] != reference["mdout_coulomb_modifier"]
        changed_mesh = (
            variant["mdout_pme_order"] != reference["mdout_pme_order"]
            or variant["mdout_fourierspacing"] != reference["mdout_fourierspacing"]
        )
        same_short_range_baseline = (
            variant["mdout_nstlist"] == reference["mdout_nstlist"]
            and variant["mdout_rlist"] == reference["mdout_rlist"]
            and variant["mdout_vbt"] == reference["mdout_vbt"]
            and variant["mdout_vdw_type"] == reference["mdout_vdw_type"]
            and variant["runtime_pairlist_line"] == reference["runtime_pairlist_line"]
        )

        if variant["role"] == "reciprocal_accuracy_variant":
            intended_verified = (
                not changed_coulombtype
                and not changed_modifier
                and changed_mesh
                and same_short_range_baseline
                and variant["runtime_coulomb_mode"] == "pme_coulomb"
            )
            path_statement = "PME reciprocal accuracy changed while Coulomb PME mode and direct-space family stayed in the PME/Ewald family."
        elif variant["role"] == "reciprocal_solver_variant":
            intended_verified = (
                changed_coulombtype
                and not changed_modifier
                and same_short_range_baseline
                and reference["runtime_coulomb_mode"] == "pme_coulomb"
                and variant["runtime_coulomb_mode"] == "ewald_no_pme_mesh"
            )
            path_statement = "Reciprocal solver changed from PME mesh to full Ewald while the short-range baseline stayed fixed and the source path indicates the direct-space family remains Ewald-family."
        else:
            intended_verified = (
                not changed_coulombtype
                and changed_modifier
                and not changed_mesh
                and same_short_range_baseline
                and variant["runtime_coulomb_mode"] == "pme_coulomb"
            )
            path_statement = "Direct-space Coulomb modifier changed while Coulomb PME remained active and the short-range baseline stayed fixed."

        variant_checks.append(
            {
                "run_id": variant["run_id"],
                "role": variant["role"],
                "changed_coulombtype_vs_reference": changed_coulombtype,
                "changed_coulomb_modifier_vs_reference": changed_modifier,
                "changed_pme_mesh_controls_vs_reference": changed_mesh,
                "same_short_range_baseline_vs_reference": same_short_range_baseline,
                "reference_runtime_coulomb_mode": reference["runtime_coulomb_mode"],
                "variant_runtime_coulomb_mode": variant["runtime_coulomb_mode"],
                "runtime_pairlist_line": variant["runtime_pairlist_line"],
                "runtime_potential_shift_line": variant["runtime_potential_shift_line"],
                "runtime_pme_banner_is_generic_for_usingPmeOrEwald": variant[
                    "runtime_pme_banner_is_generic_for_usingPmeOrEwald"
                ],
                "intended_path_change_verified": intended_verified,
                "path_change_statement": path_statement,
            }
        )
    return {
        "milestone": "TP1.8b",
        "reference_run_id": reference["run_id"],
        "reference_coulombtype": reference["mdout_coulombtype"],
        "reference_coulomb_modifier": reference["mdout_coulomb_modifier"],
        "reference_runtime_coulomb_mode": reference["runtime_coulomb_mode"],
        "reference_runtime_pme_banner_is_generic_for_usingPmeOrEwald": reference[
            "runtime_pme_banner_is_generic_for_usingPmeOrEwald"
        ],
        "variant_checks": variant_checks,
    }


def overall_classification(reciprocal_accuracy: dict[str, object], reciprocal_solver: dict[str, object], direct_modifier: dict[str, object]) -> str:
    recip_help = reciprocal_accuracy["effect_vs_reference"] == "weakens_materially" or reciprocal_solver["effect_vs_reference"] == "weakens_materially"
    direct_help = direct_modifier["effect_vs_reference"] == "weakens_materially"

    if recip_help and not direct_help:
        return "stronger_reciprocal_space_or_pme_related_suspicion"
    if direct_help and not recip_help:
        return "stronger_direct_space_coulomb_suspicion"
    return "mixed_or_still_unresolved"


def recommendation_text(classification: str) -> str:
    if classification == "stronger_reciprocal_space_or_pme_related_suspicion":
        return "Keep the safe authoritative baseline and trace the Coulomb PME reciprocal/decomposition path next; LJ-PME remains inactive here."
    if classification == "stronger_direct_space_coulomb_suspicion":
        return "Keep the safe authoritative baseline and trace the direct-space Coulomb modifier/split path next before any production patch."
    return "Keep the safe authoritative baseline and next trace a narrower Coulomb real/reciprocal split path on this same authoritative system before any production patch."


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
    reference = next(run for run in run_results if run["role"] == "safe_baseline_reference")
    reciprocal_accuracy = next(run for run in run_results if run["role"] == "reciprocal_accuracy_variant")
    reciprocal_solver = next(run for run in run_results if run["role"] == "reciprocal_solver_variant")
    direct_modifier = next(run for run in run_results if run["role"] == "direct_space_modifier_variant")

    for run in run_results:
        if run is reference:
            run["effect_vs_reference"] = "reference"
        else:
            run["effect_vs_reference"] = classify_effect(reference, run)

    active_path_map = build_active_path_map()
    write_text(RESULTS_DIR / "active_coulomb_path_map.json", json.dumps(active_path_map, indent=2) + "\n")

    run_matrix = {
        "milestone": "TP1.8b",
        "authoritative_system_id": "dense_salt_polymer",
        "authoritative_system_source": "tests/reference_results/m10_4_charged_ensemble_gate/dense_salt_polymer",
        "tp1_3_executed_baseline_source": "tests/reference_results/tp1_3_stabilization/TRL-0",
        "tp1_7b_same_build_source": "tests/reference_results/tp1_7b_authoritative_ab",
        "tp1_8_source": "tests/reference_results/tp1_8_longrange_isolation",
        "reference_run_id": reference["run_id"],
        "comparison_runs": RUNS,
        "rationale_for_20ps_window": "TP1.7b and TP1.8 already showed runaway onset at 1.0 ps, so TP1.8b uses a 20 ps authoritative window to isolate early Coulomb-path sensitivity while keeping the narrower Ewald variant feasible.",
    }
    write_text(RESULTS_DIR / "run_matrix.json", json.dumps(run_matrix, indent=2) + "\n")

    runtime_check = make_runtime_distinct_check(reference, [reciprocal_accuracy, reciprocal_solver, direct_modifier])
    write_text(RESULTS_DIR / "runtime_distinct_check.json", json.dumps(runtime_check, indent=2) + "\n")

    comparison_rows: list[dict[str, object]] = []
    for run in run_results:
        comparison_rows.append(
            {
                "run_id": run["run_id"],
                "role": run["role"],
                "mdout_coulombtype": run["mdout_coulombtype"],
                "mdout_coulomb_modifier": run["mdout_coulomb_modifier"],
                "mdout_nstlist": run["mdout_nstlist"],
                "mdout_rlist": run["mdout_rlist"],
                "mdout_vbt": run["mdout_vbt"],
                "mdout_pme_order": run["mdout_pme_order"],
                "mdout_fourierspacing": run["mdout_fourierspacing"],
                "runtime_coulomb_mode": run["runtime_coulomb_mode"],
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
    write_csv(RESULTS_DIR / "coulomb_variant_comparison.csv", list(comparison_rows[0].keys()), comparison_rows)

    classification = overall_classification(reciprocal_accuracy, reciprocal_solver, direct_modifier)
    summary = {
        "milestone": "TP1.8b",
        "reference_run": reference,
        "variants": [reciprocal_accuracy, reciprocal_solver, direct_modifier],
        "comparison_metrics": {
            "reciprocal_accuracy_variant": {
                "effect": reciprocal_accuracy["effect_vs_reference"],
                "runaway_onset_delay_ps": None
                if reciprocal_accuracy["runaway_onset_ps"] is None or reference["runaway_onset_ps"] is None
                else float(reciprocal_accuracy["runaway_onset_ps"]) - float(reference["runaway_onset_ps"]),
                "max_temperature_delta_k": float(reciprocal_accuracy["max_temperature_k"]) - float(reference["max_temperature_k"]),
                "total_energy_range_delta_kj": float(reciprocal_accuracy["total_energy_range_kj"]) - float(reference["total_energy_range_kj"]),
                "max_abs_pressure_delta_bar": float(reciprocal_accuracy["max_abs_pressure_bar"]) - float(reference["max_abs_pressure_bar"]),
            },
            "reciprocal_solver_variant": {
                "effect": reciprocal_solver["effect_vs_reference"],
                "runaway_onset_delay_ps": None
                if reciprocal_solver["runaway_onset_ps"] is None or reference["runaway_onset_ps"] is None
                else float(reciprocal_solver["runaway_onset_ps"]) - float(reference["runaway_onset_ps"]),
                "max_temperature_delta_k": float(reciprocal_solver["max_temperature_k"]) - float(reference["max_temperature_k"]),
                "total_energy_range_delta_kj": float(reciprocal_solver["total_energy_range_kj"]) - float(reference["total_energy_range_kj"]),
                "max_abs_pressure_delta_bar": float(reciprocal_solver["max_abs_pressure_bar"]) - float(reference["max_abs_pressure_bar"]),
            },
            "direct_space_modifier_variant": {
                "effect": direct_modifier["effect_vs_reference"],
                "runaway_onset_delay_ps": None
                if direct_modifier["runaway_onset_ps"] is None or reference["runaway_onset_ps"] is None
                else float(direct_modifier["runaway_onset_ps"]) - float(reference["runaway_onset_ps"]),
                "max_temperature_delta_k": float(direct_modifier["max_temperature_k"]) - float(reference["max_temperature_k"]),
                "total_energy_range_delta_kj": float(direct_modifier["total_energy_range_kj"]) - float(reference["total_energy_range_kj"]),
                "max_abs_pressure_delta_bar": float(direct_modifier["max_abs_pressure_bar"]) - float(reference["max_abs_pressure_bar"]),
            },
        },
        "final_classification": classification,
        "tp1_4_lj_pme_path_status": "inactive",
    }
    write_text(RESULTS_DIR / "stability_summary.json", json.dumps(summary, indent=2) + "\n")

    recommendation = {
        "milestone": "TP1.8b",
        "source_patching_now_justified": False,
        "plain_safe_baseline_acceptable_for_later_non_rrespa_validation": "PARTIAL",
        "final_classification": classification,
        "tp1_4_lj_pme_path_status": "inactive",
        "exact_next_step_recommendation": recommendation_text(classification),
    }
    write_text(RESULTS_DIR / "tp1_8b_recommendation.json", json.dumps(recommendation, indent=2) + "\n")

    write_text(RESULTS_DIR / "raw_commands.txt", "\n".join(commands) + "\n")

    provenance = {
        "milestone": "TP1.8b",
        "execution_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["rev-parse", "HEAD"]).strip(),
        "git_status_short": git_output(["status", "--short"]).splitlines(),
        "gmx_version": gmx_version_text().splitlines()[0],
        "inputs": {
            "tp17b_results": str(TP17B_RESULTS.relative_to(ROOT)),
            "tp18_results": str(TP18_RESULTS.relative_to(ROOT)),
            "tp13_baseline": str(TP13_DIR.relative_to(ROOT)),
        },
    }
    write_text(RESULTS_DIR / "provenance_manifest.json", json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
