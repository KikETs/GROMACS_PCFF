from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from freeze_gate_a_oracle import (
    DEFAULT_GMX,
    FIXTURE_ROOT,
    LEVEL_FACTORS,
    RESPA_ENERGY_INTERVAL,
    base_env,
    capture_output,
    command_record,
    energy_summary,
    env_delta,
    make_exact_respa_mdp,
    parse_energy_dump,
    parse_event_trace,
    parse_merge_trace_dir,
    parse_total_force_dump,
    parse_trr_dump,
    restart_summary,
    run_command,
    write_commands_script,
    write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_A_MANIFEST = REPO_ROOT / "tests" / "reference_results" / "gate_a_cpu_oracle" / "oracle_manifest.json"
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_b_nb_gpu_validation"
GPU_REPEAT_COUNT = 3
OFFLOADED_ENERGY_TERMS = ("LJ (SR)", "Coulomb (SR)")
FLOAT_EPSILON = 1.1920928955078125e-07
GMX_DUMP_DECIMAL_PLACES = 5
GPU_REPRODUCIBILITY_NOTE = (
    "Binary reproducibility (-reprod) is not enabled because GROMACS rejects -nb gpu together with -reprod."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Gate B for standalone exact r-RESPA with nonbonded GPU offload only."
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument(
        "--gate-a-manifest",
        default=str(DEFAULT_GATE_A_MANIFEST),
        help="Path to the frozen Gate A CPU oracle manifest.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--build-target", default="gmx", help="CMake target to build.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step.")
    parser.add_argument("--build-dir", default=None, help="Optional explicit build directory.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks for mdrun.")
    parser.add_argument("--ntomp", type=int, default=2, help="OpenMP threads for mdrun.")
    parser.add_argument("--outer-steps", type=int, default=5, help="Number of exact r-RESPA outer steps.")
    parser.add_argument("--pair14-level", type=int, default=1, help="Exact r-RESPA pair14 level.")
    parser.add_argument(
        "--gpu-repeats",
        type=int,
        default=GPU_REPEAT_COUNT,
        help="Number of repeated GPU runs for noise-floor estimation when the path executes.",
    )
    return parser.parse_args()


def run_command_allow_failure(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("w", encoding="utf-8") if stdout_path is not None else None
    stderr_handle = stderr_path.open("w", encoding="utf-8") if stderr_path is not None else None
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    finally:
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_gpu_support(gmx_version: str) -> str:
    match = re.search(r"GPU support:\s*(.+)", gmx_version)
    return match.group(1).strip() if match is not None else "unknown"


def parse_precision_mode(gmx_version: str) -> str:
    match = re.search(r"Precision:\s*(.+)", gmx_version)
    return match.group(1).strip() if match is not None else "unknown"


def capture_optional_output(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return {
        "argv": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def fixture_dir(system_id: str) -> Path:
    return FIXTURE_ROOT / system_id


def read_gro_atom_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Unexpected GRO file, missing atom-count line: {path}")
    return int(lines[1].strip())


def build_trace_atom_indices(atom_count: int) -> str:
    if atom_count <= 0:
        raise ValueError(f"Atom count must be positive, got {atom_count}")
    return ",".join(str(atom_index) for atom_index in range(atom_count))


def mdrun_args_gate_b(args: argparse.Namespace, deffnm: Path) -> list[str]:
    return [
        "-s",
        str(deffnm.with_suffix(".tpr")),
        "-deffnm",
        str(deffnm),
        "-ntmpi",
        str(args.ntmpi),
        "-ntomp",
        str(args.ntomp),
        "-dlb",
        "no",
        "-nb",
        "gpu",
        "-pme",
        "cpu",
        "-bonded",
        "cpu",
        "-update",
        "cpu",
        "-pin",
        "off",
    ]


def trace_env_for_run(args: argparse.Namespace, run_root: Path, atom_count: int | None = None) -> dict[str, str]:
    env = base_env(args)
    all_steps = ",".join(str(step) for step in range(args.outer_steps * LEVEL_FACTORS[-1] + 1))
    env["GMX_EXACT_RESPA_RUNTIME_EVENT_TRACE_FILE"] = str(run_root / "event_trace.tsv")
    env["GMX_PCFF_RESPA_MERGE_TRACE_DIR"] = str(run_root / "merge_trace")
    env["GMX_EXACT_RESPA_TOTAL_FORCE_DUMP_FILE"] = str(run_root / "total_force.tsv")
    if atom_count is not None:
        env["GMX_PCFF_RESPA_TRACE_ATOMS"] = build_trace_atom_indices(atom_count)
    env["GMX_PCFF_RESPA_TRACE_FORCE_COMPONENTS"] = "1"
    env["GMX_PCFF_RESPA_TRACE_FORCE_COMPONENTS_STEPS"] = all_steps
    env["GMX_PCFF_RESPA_TRACE_REALSPACE_FORCE_SUBCOMPONENTS"] = "1"
    env["GMX_PCFF_RESPA_TRACE_REALSPACE_FORCE_SUBCOMPONENTS_STEPS"] = "0,2"
    env["GMX_PCFF_RESPA_TRACE_STEP1_SUBSET01_FORCEGROUP_AUDIT"] = "1"
    env["GMX_PCFF_RESPA_TRACE_STEP1_SUBSET01_FORCEGROUP_AUDIT_STEPS"] = "2"
    env["GMX_PCFF_RESPA_TRACE_CLASS2_SUBTERM_ENERGIES"] = "1"
    env["GMX_PCFF_RESPA_TRACE_CLASS2_SUBTERM_ENERGIES_STEPS"] = all_steps
    env["GMX_PCFF_RESPA_TRACE_CPU_CORRECTION_ENERGIES"] = "1"
    env["GMX_PCFF_RESPA_TRACE_CPU_CORRECTION_ENERGIES_STEPS"] = all_steps
    env["GMX_PCFF_RESPA_TRACE_MULTI_STEP_COULOMB_STEPS"] = all_steps
    env["GMX_PCFF_RESPA_M2P_TRACE_DIR"] = str(run_root / "m2p_trace")
    env["GMX_PCFF_RESPA_M2P_CASE_LABEL"] = run_root.name
    return env


def comparison_keys_by_relative_path(entries: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(entry["relative_path"]): entry for entry in entries}


def comparison_keys_by_step_level(entries: list[dict[str, object]]) -> dict[tuple[int, int], dict[str, object]]:
    return {(int(entry["step"]), int(entry["highest_active_level"])): entry for entry in entries}


def compare_event_trace(actual: list[dict[str, object]], expected: list[dict[str, object]]) -> dict[str, object]:
    mismatch = None
    for index, (actual_entry, expected_entry) in enumerate(zip(actual, expected)):
        if actual_entry != expected_entry:
            mismatch = {
                "index": index,
                "expected": expected_entry,
                "actual": actual_entry,
            }
            break
    if mismatch is None and len(actual) != len(expected):
        mismatch = {
            "index": min(len(actual), len(expected)),
            "expected": expected[len(actual)] if len(expected) > len(actual) else None,
            "actual": actual[len(expected)] if len(actual) > len(expected) else None,
        }
    return {
        "matches": mismatch is None,
        "actual_count": len(actual),
        "expected_count": len(expected),
        "first_mismatch": mismatch,
    }


def compare_total_force_entries(
    actual_entries: list[dict[str, object]], expected_entries: list[dict[str, object]]
) -> dict[str, object]:
    actual_map = comparison_keys_by_step_level(actual_entries)
    expected_map = comparison_keys_by_step_level(expected_entries)
    missing_in_actual = sorted([f"{step}:{level}" for step, level in expected_map.keys() - actual_map.keys()])
    extra_in_actual = sorted([f"{step}:{level}" for step, level in actual_map.keys() - expected_map.keys()])
    max_abs_component_delta = 0.0
    first_mismatch = None
    for key in sorted(actual_map.keys() & expected_map.keys()):
        actual_vector = list(actual_map[key]["vector_sum"])
        expected_vector = list(expected_map[key]["vector_sum"])
        component_deltas = [abs(float(a) - float(b)) for a, b in zip(actual_vector, expected_vector)]
        if component_deltas:
            local_max = max(component_deltas)
            max_abs_component_delta = max(max_abs_component_delta, local_max)
        if first_mismatch is None and any(delta != 0.0 for delta in component_deltas):
            first_mismatch = {
                "step": key[0],
                "highest_active_level": key[1],
                "expected_vector_sum": expected_vector,
                "actual_vector_sum": actual_vector,
                "component_abs_deltas": component_deltas,
            }
    return {
        "matches": not missing_in_actual and not extra_in_actual and first_mismatch is None,
        "missing_in_actual": missing_in_actual,
        "extra_in_actual": extra_in_actual,
        "max_abs_component_delta": max_abs_component_delta,
        "first_mismatch": first_mismatch,
    }


def compare_per_level_force_entries(
    actual_entries: list[dict[str, object]], expected_entries: list[dict[str, object]]
) -> dict[str, object]:
    actual_map = comparison_keys_by_relative_path(actual_entries)
    expected_map = comparison_keys_by_relative_path(expected_entries)
    missing_in_actual = sorted(expected_map.keys() - actual_map.keys())
    extra_in_actual = sorted(actual_map.keys() - expected_map.keys())
    max_abs_component_delta = 0.0
    first_mismatch = None
    for key in sorted(actual_map.keys() & expected_map.keys()):
        actual_vector = list(actual_map[key]["vector_sum"])
        expected_vector = list(expected_map[key]["vector_sum"])
        component_deltas = [abs(float(a) - float(b)) for a, b in zip(actual_vector, expected_vector)]
        if component_deltas:
            local_max = max(component_deltas)
            max_abs_component_delta = max(max_abs_component_delta, local_max)
        if first_mismatch is None and any(delta != 0.0 for delta in component_deltas):
            first_mismatch = {
                "relative_path": key,
                "expected_vector_sum": expected_vector,
                "actual_vector_sum": actual_vector,
                "component_abs_deltas": component_deltas,
            }
    return {
        "matches": not missing_in_actual and not extra_in_actual and first_mismatch is None,
        "missing_in_actual": missing_in_actual,
        "extra_in_actual": extra_in_actual,
        "max_abs_component_delta": max_abs_component_delta,
        "first_mismatch": first_mismatch,
    }


def compare_energy_frames(
    actual_frames: list[dict[str, object]], expected_frames: list[dict[str, object]]
) -> dict[str, object]:
    first_mismatch = None
    max_abs_delta = 0.0
    term_deltas: list[dict[str, object]] = []
    frame_count = min(len(actual_frames), len(expected_frames))
    for index in range(frame_count):
        actual_frame = actual_frames[index]
        expected_frame = expected_frames[index]
        actual_terms = dict(actual_frame["terms"])
        expected_terms = dict(expected_frame["terms"])
        common_terms = sorted(actual_terms.keys() & expected_terms.keys())
        deltas_for_frame = {}
        for term in common_terms:
            delta = float(actual_terms[term]) - float(expected_terms[term])
            deltas_for_frame[term] = delta
            max_abs_delta = max(max_abs_delta, abs(delta))
            if first_mismatch is None and delta != 0.0:
                first_mismatch = {
                    "frame_index": index,
                    "step": int(actual_frame["step"]),
                    "term": term,
                    "expected": float(expected_terms[term]),
                    "actual": float(actual_terms[term]),
                    "delta": delta,
                }
        term_deltas.append(
            {
                "frame_index": index,
                "step": int(actual_frame["step"]),
                "time_ps": float(actual_frame["time_ps"]),
                "term_deltas_kj_mol": deltas_for_frame,
            }
        )
    return {
        "matches": len(actual_frames) == len(expected_frames) and first_mismatch is None,
        "actual_frame_count": len(actual_frames),
        "expected_frame_count": len(expected_frames),
        "max_abs_delta_kj_mol": max_abs_delta,
        "first_mismatch": first_mismatch,
        "frames": term_deltas,
    }


def extract_virial_deltas(energy_comparison: dict[str, object]) -> dict[str, object]:
    virial_terms = {"Pressure", "Vir-XX", "Vir-YY", "Vir-ZZ", "Pres-XX", "Pres-YY", "Pres-ZZ"}
    frames = []
    max_abs_delta = 0.0
    first_nonzero = None
    for frame in energy_comparison["frames"]:
        deltas = {
            term: delta
            for term, delta in dict(frame["term_deltas_kj_mol"]).items()
            if term in virial_terms
        }
        if deltas:
            local_max = max(abs(float(value)) for value in deltas.values())
            max_abs_delta = max(max_abs_delta, local_max)
            if first_nonzero is None:
                for term, delta in deltas.items():
                    if float(delta) != 0.0:
                        first_nonzero = {
                            "frame_index": frame["frame_index"],
                            "step": frame["step"],
                            "term": term,
                            "delta": float(delta),
                        }
                        break
        frames.append(
            {
                "frame_index": frame["frame_index"],
                "step": frame["step"],
                "time_ps": frame["time_ps"],
                "term_deltas": deltas,
            }
        )
    return {
        "available": bool(frames),
        "max_abs_delta": max_abs_delta,
        "first_nonzero": first_nonzero,
        "frames": frames,
    }


def estimate_noise_floor(
    successful_runs: list[dict[str, object]],
) -> dict[str, object]:
    if len(successful_runs) < 2:
        return {
            "available": False,
            "reason": "Fewer than two successful GPU runs are available.",
            "successful_run_count": len(successful_runs),
        }

    reference = successful_runs[0]
    max_energy_delta = 0.0
    max_virial_delta = 0.0
    max_total_force_delta = 0.0
    max_per_level_force_delta = 0.0
    event_trace_identical = True
    compared_runs = []
    for run in successful_runs[1:]:
        event_comparison = compare_event_trace(run["actual_events"], reference["actual_events"])
        energy_comparison = compare_energy_frames(run["energy_frames"], reference["energy_frames"])
        total_force_comparison = compare_total_force_entries(
            run["total_force_summary"]["per_step_totals"], reference["total_force_summary"]["per_step_totals"]
        )
        per_level_force_comparison = compare_per_level_force_entries(
            run["per_level_force_totals"]["entries"], reference["per_level_force_totals"]["entries"]
        )
        virial_comparison = extract_virial_deltas(energy_comparison)
        event_trace_identical = event_trace_identical and bool(event_comparison["matches"])
        max_energy_delta = max(max_energy_delta, float(energy_comparison["max_abs_delta_kj_mol"]))
        max_virial_delta = max(max_virial_delta, float(virial_comparison["max_abs_delta"]))
        max_total_force_delta = max(max_total_force_delta, float(total_force_comparison["max_abs_component_delta"]))
        max_per_level_force_delta = max(
            max_per_level_force_delta, float(per_level_force_comparison["max_abs_component_delta"])
        )
        compared_runs.append(
            {
                "run_id": run["run_id"],
                "event_trace_matches_reference_gpu_run": event_comparison["matches"],
                "energy_max_abs_delta_kj_mol": energy_comparison["max_abs_delta_kj_mol"],
                "virial_max_abs_delta": virial_comparison["max_abs_delta"],
                "total_force_max_abs_component_delta": total_force_comparison["max_abs_component_delta"],
                "per_level_force_max_abs_component_delta": per_level_force_comparison["max_abs_component_delta"],
            }
        )

    return {
        "available": True,
        "successful_run_count": len(successful_runs),
        "reference_gpu_run_id": reference["run_id"],
        "event_trace_identical_across_successful_gpu_runs": event_trace_identical,
        "energy_max_abs_delta_kj_mol": max_energy_delta,
        "virial_max_abs_delta": max_virial_delta,
        "total_force_max_abs_component_delta": max_total_force_delta,
        "per_level_force_max_abs_component_delta": max_per_level_force_delta,
        "compared_runs": compared_runs,
    }


def max_abs_delta_for_terms(energy_comparison: dict[str, object], term_names: tuple[str, ...]) -> dict[str, object]:
    max_abs_delta = 0.0
    first_nonzero = None
    for frame in energy_comparison["frames"]:
        deltas = dict(frame["term_deltas_kj_mol"])
        for term in term_names:
            if term not in deltas:
                continue
            delta = float(deltas[term])
            max_abs_delta = max(max_abs_delta, abs(delta))
            if first_nonzero is None and delta != 0.0:
                first_nonzero = {
                    "frame_index": frame["frame_index"],
                    "step": frame["step"],
                    "term": term,
                    "delta": delta,
                }
    return {
        "terms": list(term_names),
        "max_abs_delta_kj_mol": max_abs_delta,
        "first_nonzero": first_nonzero,
    }


def force_sum_roundoff_bound_from_total_force_dump(path: Path) -> dict[str, object]:
    grouped: dict[tuple[int, int], dict[str, float | int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 7:
            step_str, _time_str, highest_level_str, _atom_str, fx_str, fy_str, fz_str = parts
        elif len(parts) == 8:
            step_str, _time_str, highest_level_str, _local_atom_str, _atom_str, fx_str, fy_str, fz_str = parts
        else:
            raise ValueError(f"Unexpected force dump line in {path}: {line}")
        key = (int(step_str), int(highest_level_str))
        bucket = grouped.setdefault(key, {"atom_count": 0, "max_abs_component": 0.0})
        bucket["atom_count"] = int(bucket["atom_count"]) + 1
        bucket["max_abs_component"] = max(
            float(bucket["max_abs_component"]),
            abs(float(fx_str)),
            abs(float(fy_str)),
            abs(float(fz_str)),
        )

    max_bound = 0.0
    worst_case = None
    for (step, highest_level), bucket in grouped.items():
        atom_count = int(bucket["atom_count"])
        max_abs_component = float(bucket["max_abs_component"])
        bound = 2.0 * atom_count * max_abs_component * FLOAT_EPSILON
        if bound > max_bound:
            max_bound = bound
            worst_case = {
                "step": step,
                "highest_active_level": highest_level,
                "atom_count": atom_count,
                "max_abs_component": max_abs_component,
                "bound": bound,
            }
    return {
        "bound": max_bound,
        "worst_case": worst_case,
    }


def force_sum_roundoff_bound_from_merge_trace_dir(path: Path) -> dict[str, object]:
    max_bound = 0.0
    worst_case = None
    for trace_path in sorted(path.glob("*.tsv")):
        atom_count = 0
        max_abs_component = 0.0
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            _atom, fx_str, fy_str, fz_str = stripped.split("\t")
            atom_count += 1
            max_abs_component = max(
                max_abs_component,
                abs(float(fx_str)),
                abs(float(fy_str)),
                abs(float(fz_str)),
            )
        if atom_count == 0:
            continue
        bound = 2.0 * atom_count * max_abs_component * FLOAT_EPSILON
        if bound > max_bound:
            max_bound = bound
            worst_case = {
                "relative_path": str(trace_path),
                "atom_count": atom_count,
                "max_abs_component": max_abs_component,
                "bound": bound,
            }
    return {
        "bound": max_bound,
        "worst_case": worst_case,
    }


def scientific_display_resolution(value: float) -> float:
    if value == 0.0:
        return 10 ** (-GMX_DUMP_DECIMAL_PLACES)
    exponent = int(math.floor(math.log10(abs(value))))
    return 10 ** (exponent - GMX_DUMP_DECIMAL_PLACES)


def assess_energy_display_resolution(
    actual_frames: list[dict[str, object]],
    expected_frames: list[dict[str, object]],
    *,
    selected_terms: tuple[str, ...] | None = None,
) -> dict[str, object]:
    first_excess = None
    max_abs_excess = 0.0
    max_allowed_abs_delta = 0.0
    frame_count = min(len(actual_frames), len(expected_frames))
    for index in range(frame_count):
        actual_terms = dict(actual_frames[index]["terms"])
        expected_terms = dict(expected_frames[index]["terms"])
        common_terms = sorted(actual_terms.keys() & expected_terms.keys())
        for term in common_terms:
            if selected_terms is not None and term not in selected_terms:
                continue
            actual_value = float(actual_terms[term])
            expected_value = float(expected_terms[term])
            delta = abs(actual_value - expected_value)
            allowed_delta = 2.0 * max(
                scientific_display_resolution(actual_value),
                scientific_display_resolution(expected_value),
            )
            max_allowed_abs_delta = max(max_allowed_abs_delta, allowed_delta)
            numeric_slack = max(1e-12, allowed_delta * 1e-9)
            excess = max(0.0, delta - allowed_delta - numeric_slack)
            max_abs_excess = max(max_abs_excess, excess)
            if first_excess is None and excess > 0.0:
                first_excess = {
                    "frame_index": index,
                    "step": int(actual_frames[index]["step"]),
                    "term": term,
                    "actual": actual_value,
                    "expected": expected_value,
                    "abs_delta": delta,
                    "allowed_abs_delta": allowed_delta,
                    "abs_excess": excess,
                }
    return {
        "within_bounds": first_excess is None,
        "max_allowed_abs_delta": max_allowed_abs_delta,
        "max_abs_excess": max_abs_excess,
        "first_excess": first_excess,
    }


def assess_gate_b_system(system_result: dict[str, object]) -> dict[str, object]:
    if system_result["main_run"]["returncode"] != 0:
        return {
            "status": "BLOCKER",
            "reasons": ["The primary Gate B GPU run did not execute successfully."],
        }

    event_order_comparison = system_result["event_order_comparison"]
    total_force_comparison = system_result["total_force_comparison"]
    per_level_force_comparison = system_result["per_level_force_comparison"]
    energy_comparison = system_result["energy_comparison"]
    virial_comparison = system_result["virial_comparison"]
    gpu_noise_floor = system_result["gpu_noise_floor"]
    restart_validation = system_result["restart_validation"]
    offloaded_energy_delta = max_abs_delta_for_terms(energy_comparison, OFFLOADED_ENERGY_TERMS)
    total_force_roundoff = force_sum_roundoff_bound_from_total_force_dump(
        Path(system_result["main_run"]["full_outputs"]["total_force_tsv"])
    )
    per_level_roundoff = force_sum_roundoff_bound_from_merge_trace_dir(
        Path(system_result["main_run"]["full_outputs"]["merge_trace_dir"])
    )
    expected_energy_frames = load_json(
        Path(system_result["gate_a_artifact_root"]) / "summaries" / "energy_terms.json"
    )["frames"]
    full_energy_display = assess_energy_display_resolution(
        system_result["main_run"]["energy_frames"],
        expected_energy_frames,
    )
    offloaded_energy_display = assess_energy_display_resolution(
        system_result["main_run"]["energy_frames"],
        expected_energy_frames,
        selected_terms=OFFLOADED_ENERGY_TERMS,
    )
    virial_display = assess_energy_display_resolution(
        system_result["main_run"]["energy_frames"],
        expected_energy_frames,
        selected_terms=("Pressure", "Vir-XX", "Vir-YY", "Vir-ZZ", "Pres-XX", "Pres-YY", "Pres-ZZ"),
    )

    total_force_within_float_roundoff = (
        float(total_force_comparison["max_abs_component_delta"]) <= float(total_force_roundoff["bound"])
    )
    per_level_within_float_roundoff = (
        float(per_level_force_comparison["max_abs_component_delta"]) <= float(per_level_roundoff["bound"])
    )

    reasons = []
    if not event_order_comparison["matches"]:
        reasons.append("Runtime event ordering diverged from the frozen Gate A oracle.")
    if total_force_comparison["missing_in_actual"] or total_force_comparison["extra_in_actual"]:
        reasons.append("Step-level total-force coverage diverged from the Gate A oracle.")
    if per_level_force_comparison["missing_in_actual"] or per_level_force_comparison["extra_in_actual"]:
        reasons.append("Per-level force-trace coverage diverged from the Gate A oracle.")
    if restart_validation["status"] != "PASS":
        reasons.append("Restart continuity did not match the standalone exact Gate A semantics.")
    if not gpu_noise_floor["available"]:
        reasons.append("GPU run-to-run noise floor was not measured.")
    elif not gpu_noise_floor["event_trace_identical_across_successful_gpu_runs"]:
        reasons.append("Repeated GPU runs changed event ordering, so the noise floor is not trustworthy.")
    if not total_force_within_float_roundoff:
        reasons.append("CPU-vs-GPU total-force deviation exceeds a conservative float-accumulation roundoff bound.")
    if not per_level_within_float_roundoff:
        reasons.append("CPU-vs-GPU per-level force deviation exceeds a conservative float-accumulation roundoff bound.")
    if not offloaded_energy_display["within_bounds"]:
        reasons.append("CPU-vs-GPU offloaded-energy delta exceeds the gmx dump display-resolution bound.")

    return {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "event_order_identical": bool(event_order_comparison["matches"]),
        "total_force_trace_coverage_preserved": not total_force_comparison["missing_in_actual"]
        and not total_force_comparison["extra_in_actual"],
        "per_level_force_trace_coverage_preserved": not per_level_force_comparison["missing_in_actual"]
        and not per_level_force_comparison["extra_in_actual"],
        "restart_semantics_unchanged": restart_validation["status"] == "PASS",
        "gpu_noise_floor_measured": bool(gpu_noise_floor["available"]),
        "gpu_repeat_event_order_identical": bool(
            gpu_noise_floor["available"] and gpu_noise_floor["event_trace_identical_across_successful_gpu_runs"]
        ),
        "total_force_max_abs_component_delta": total_force_comparison["max_abs_component_delta"],
        "per_level_force_max_abs_component_delta": per_level_force_comparison["max_abs_component_delta"],
        "total_force_float_roundoff_bound": total_force_roundoff,
        "per_level_force_float_roundoff_bound": per_level_roundoff,
        "offloaded_energy_terms": offloaded_energy_delta,
        "offloaded_energy_display_resolution_assessment": offloaded_energy_display,
        "full_energy_terms_max_abs_delta_kj_mol": energy_comparison["max_abs_delta_kj_mol"],
        "full_energy_display_resolution_assessment": full_energy_display,
        "virial_terms_max_abs_delta": virial_comparison["max_abs_delta"],
        "virial_display_resolution_assessment": virial_display,
        "total_force_within_float_roundoff_bound": total_force_within_float_roundoff,
        "per_level_force_within_float_roundoff_bound": per_level_within_float_roundoff,
    }


def find_first_failure(system_result: dict[str, object]) -> dict[str, object] | None:
    if system_result["main_run"]["returncode"] != 0:
        return {
            "field": "main_run.returncode",
            "reason": "The Gate B mdrun command did not execute successfully.",
            "returncode": system_result["main_run"]["returncode"],
        }

    for field_name in (
        "event_order_comparison",
        "total_force_comparison",
        "per_level_force_comparison",
    ):
        comparison = system_result.get(field_name)
        if comparison is not None and not comparison["matches"]:
            return {
                "field": field_name,
                "details": comparison["first_mismatch"],
            }

    energy_comparison = system_result.get("energy_comparison")
    if energy_comparison is not None and not energy_comparison["matches"]:
        return {
            "field": "energy_comparison",
            "details": energy_comparison["first_mismatch"],
        }

    restart_validation = system_result.get("restart_validation")
    if restart_validation is not None and restart_validation["status"] != "PASS":
        return {
            "field": "restart_validation",
            "details": restart_validation,
        }

    return None


def extract_failure_markers(stderr_path: Path, stdout_path: Path) -> list[str]:
    markers = []
    combined_text = ""
    if stdout_path.exists():
        combined_text += stdout_path.read_text(encoding="utf-8", errors="replace")
    if stderr_path.exists():
        combined_text += "\n" + stderr_path.read_text(encoding="utf-8", errors="replace")
    for marker in (
        "Standalone exact r-RESPA is not supported on GPUs.",
        "Standalone exact r-RESPA is not supported.",
        "Exact LAMMPS-style r-RESPA is CPU-only",
    ):
        if marker in combined_text:
            markers.append(marker)
    return markers


def collect_restart_validation(
    args: argparse.Namespace,
    gmx: Path,
    system_id: str,
    system_root: Path,
    mdp_path: Path,
    commands: list[dict[str, object]],
    base_environment: dict[str, str],
    base_env_delta: dict[str, str],
) -> dict[str, object]:
    fixture = fixture_dir(system_id)
    logs_dir = system_root / "logs"
    inputs_dir = system_root / "inputs"
    restart_full_dir = system_root / "restart_full"
    restart_split_dir = system_root / "restart_split"
    for directory in (restart_full_dir, restart_split_dir):
        directory.mkdir(parents=True, exist_ok=True)

    split_outer_steps = max(1, args.outer_steps // 2)
    split_steps = split_outer_steps * RESPA_ENERGY_INTERVAL

    restart_full_deffnm = restart_full_dir / "exact_full"
    restart_full_grompp = [
        str(gmx),
        "grompp",
        "-f",
        str(mdp_path),
        "-c",
        str(fixture / "initial_nve.gro"),
        "-p",
        str(fixture / "topol.top"),
        "-o",
        str(restart_full_deffnm.with_suffix(".tpr")),
        "-po",
        str(inputs_dir / "exact_respa_restart_full_mdout.mdp"),
        "-maxwarn",
        "1",
    ]
    restart_full_grompp_stdout = logs_dir / "grompp_restart_full.stdout"
    restart_full_grompp_stderr = logs_dir / "grompp_restart_full.stderr"
    run_command(
        restart_full_grompp,
        cwd=REPO_ROOT,
        env=base_environment,
        stdout_path=restart_full_grompp_stdout,
        stderr_path=restart_full_grompp_stderr,
    )
    commands.append(
        command_record(
            "grompp_restart_full",
            restart_full_grompp,
            cwd=REPO_ROOT,
            env_overrides=base_env_delta,
            stdout_path=restart_full_grompp_stdout,
            stderr_path=restart_full_grompp_stderr,
        )
    )

    restart_full_mdrun = [str(gmx), "mdrun", *mdrun_args_gate_b(args, restart_full_deffnm)]
    restart_full_mdrun_stdout = logs_dir / "mdrun_restart_full.stdout"
    restart_full_mdrun_stderr = logs_dir / "mdrun_restart_full.stderr"
    restart_full_result = run_command_allow_failure(
        restart_full_mdrun,
        cwd=REPO_ROOT,
        env=base_environment,
        stdout_path=restart_full_mdrun_stdout,
        stderr_path=restart_full_mdrun_stderr,
    )
    commands.append(
        command_record(
            "mdrun_restart_full",
            restart_full_mdrun,
            cwd=REPO_ROOT,
            env_overrides=base_env_delta,
            stdout_path=restart_full_mdrun_stdout,
            stderr_path=restart_full_mdrun_stderr,
        )
    )
    if restart_full_result.returncode != 0:
        return {
            "status": "BLOCKER",
            "stage": "restart_full_mdrun",
            "returncode": restart_full_result.returncode,
            "failure_markers": extract_failure_markers(restart_full_mdrun_stderr, restart_full_mdrun_stdout),
            "stdout": str(restart_full_mdrun_stdout),
            "stderr": str(restart_full_mdrun_stderr),
        }

    restart_split_deffnm = restart_split_dir / "exact_split"
    restart_split_grompp = [
        str(gmx),
        "grompp",
        "-f",
        str(mdp_path),
        "-c",
        str(fixture / "initial_nve.gro"),
        "-p",
        str(fixture / "topol.top"),
        "-o",
        str(restart_split_deffnm.with_suffix(".tpr")),
        "-po",
        str(inputs_dir / "exact_respa_restart_split_mdout.mdp"),
        "-maxwarn",
        "1",
    ]
    restart_split_grompp_stdout = logs_dir / "grompp_restart_split.stdout"
    restart_split_grompp_stderr = logs_dir / "grompp_restart_split.stderr"
    run_command(
        restart_split_grompp,
        cwd=REPO_ROOT,
        env=base_environment,
        stdout_path=restart_split_grompp_stdout,
        stderr_path=restart_split_grompp_stderr,
    )
    commands.append(
        command_record(
            "grompp_restart_split",
            restart_split_grompp,
            cwd=REPO_ROOT,
            env_overrides=base_env_delta,
            stdout_path=restart_split_grompp_stdout,
            stderr_path=restart_split_grompp_stderr,
        )
    )

    restart_split_first_mdrun = [
        str(gmx),
        "mdrun",
        *mdrun_args_gate_b(args, restart_split_deffnm),
        "-nsteps",
        str(split_steps),
    ]
    restart_split_first_stdout = logs_dir / "mdrun_restart_split_first.stdout"
    restart_split_first_stderr = logs_dir / "mdrun_restart_split_first.stderr"
    restart_split_first_result = run_command_allow_failure(
        restart_split_first_mdrun,
        cwd=REPO_ROOT,
        env=base_environment,
        stdout_path=restart_split_first_stdout,
        stderr_path=restart_split_first_stderr,
    )
    commands.append(
        command_record(
            "mdrun_restart_split_first",
            restart_split_first_mdrun,
            cwd=REPO_ROOT,
            env_overrides=base_env_delta,
            stdout_path=restart_split_first_stdout,
            stderr_path=restart_split_first_stderr,
        )
    )
    if restart_split_first_result.returncode != 0:
        return {
            "status": "BLOCKER",
            "stage": "restart_split_first_mdrun",
            "returncode": restart_split_first_result.returncode,
            "failure_markers": extract_failure_markers(restart_split_first_stderr, restart_split_first_stdout),
            "stdout": str(restart_split_first_stdout),
            "stderr": str(restart_split_first_stderr),
        }

    restart_checkpoint = restart_split_deffnm.with_suffix(".cpt")
    restart_split_second_mdrun = [
        str(gmx),
        "mdrun",
        *mdrun_args_gate_b(args, restart_split_deffnm),
        "-cpi",
        str(restart_checkpoint),
    ]
    restart_split_second_stdout = logs_dir / "mdrun_restart_split_second.stdout"
    restart_split_second_stderr = logs_dir / "mdrun_restart_split_second.stderr"
    restart_split_second_result = run_command_allow_failure(
        restart_split_second_mdrun,
        cwd=REPO_ROOT,
        env=base_environment,
        stdout_path=restart_split_second_stdout,
        stderr_path=restart_split_second_stderr,
    )
    commands.append(
        command_record(
            "mdrun_restart_split_second",
            restart_split_second_mdrun,
            cwd=REPO_ROOT,
            env_overrides=base_env_delta,
            stdout_path=restart_split_second_stdout,
            stderr_path=restart_split_second_stderr,
        )
    )
    if restart_split_second_result.returncode != 0:
        return {
            "status": "BLOCKER",
            "stage": "restart_split_second_mdrun",
            "returncode": restart_split_second_result.returncode,
            "failure_markers": extract_failure_markers(restart_split_second_stderr, restart_split_second_stdout),
            "stdout": str(restart_split_second_stdout),
            "stderr": str(restart_split_second_stderr),
        }

    restart_full_energy_dump = capture_output(
        [str(gmx), "dump", "-e", str(restart_full_deffnm.with_suffix(".edr"))], cwd=REPO_ROOT
    )
    restart_full_trr_dump = capture_output(
        [str(gmx), "dump", "-f", str(restart_full_deffnm.with_suffix(".trr"))], cwd=REPO_ROOT
    )
    restart_split_energy_dump = capture_output(
        [str(gmx), "dump", "-e", str(restart_split_deffnm.with_suffix(".edr"))], cwd=REPO_ROOT
    )
    restart_split_trr_dump = capture_output(
        [str(gmx), "dump", "-f", str(restart_split_deffnm.with_suffix(".trr"))], cwd=REPO_ROOT
    )

    write_text(system_root / "summaries" / "restart_full_energy_dump.txt", restart_full_energy_dump)
    write_text(system_root / "summaries" / "restart_full_trr_dump.txt", restart_full_trr_dump)
    write_text(system_root / "summaries" / "restart_split_energy_dump.txt", restart_split_energy_dump)
    write_text(system_root / "summaries" / "restart_split_trr_dump.txt", restart_split_trr_dump)

    restart_full_energy_frames = parse_energy_dump(restart_full_energy_dump)
    restart_split_energy_frames = parse_energy_dump(restart_split_energy_dump)
    restart_full_trr_frames = parse_trr_dump(restart_full_trr_dump)
    restart_split_trr_frames = parse_trr_dump(restart_split_trr_dump)
    summary = restart_summary(
        system_id,
        split_outer_steps,
        restart_full_energy_frames,
        restart_split_energy_frames,
        restart_full_trr_frames,
        restart_split_trr_frames,
    )
    summary["status"] = "PASS"
    return summary


def run_gpu_attempt(
    *,
    args: argparse.Namespace,
    gmx: Path,
    system_id: str,
    system_root: Path,
    attempt_name: str,
    mdp_path: Path,
    collect_trace_env: bool,
    commands: list[dict[str, object]],
) -> dict[str, object]:
    fixture = fixture_dir(system_id)
    run_root = system_root / attempt_name
    logs_dir = system_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    base_environment = base_env(args)
    trace_atom_count = read_gro_atom_count(fixture / "initial_nve.gro")
    run_environment = (
        trace_env_for_run(args, run_root, atom_count=trace_atom_count) if collect_trace_env else base_environment.copy()
    )
    base_env_delta = env_delta(base_environment, os.environ)
    run_env_delta = env_delta(run_environment, os.environ)

    deffnm = run_root / "exact_full"
    grompp = [
        str(gmx),
        "grompp",
        "-f",
        str(mdp_path),
        "-c",
        str(fixture / "initial_nve.gro"),
        "-p",
        str(fixture / "topol.top"),
        "-o",
        str(deffnm.with_suffix(".tpr")),
        "-po",
        str(system_root / "inputs" / f"{attempt_name}_mdout.mdp"),
        "-maxwarn",
        "1",
    ]
    grompp_stdout = logs_dir / f"grompp_{attempt_name}.stdout"
    grompp_stderr = logs_dir / f"grompp_{attempt_name}.stderr"
    run_command(grompp, cwd=REPO_ROOT, env=base_environment, stdout_path=grompp_stdout, stderr_path=grompp_stderr)
    commands.append(
        command_record(
            f"grompp_{attempt_name}",
            grompp,
            cwd=REPO_ROOT,
            env_overrides=base_env_delta,
            stdout_path=grompp_stdout,
            stderr_path=grompp_stderr,
        )
    )

    mdrun = [str(gmx), "mdrun", *mdrun_args_gate_b(args, deffnm)]
    mdrun_stdout = logs_dir / f"mdrun_{attempt_name}.stdout"
    mdrun_stderr = logs_dir / f"mdrun_{attempt_name}.stderr"
    result = run_command_allow_failure(
        mdrun,
        cwd=REPO_ROOT,
        env=run_environment,
        stdout_path=mdrun_stdout,
        stderr_path=mdrun_stderr,
    )
    commands.append(
        command_record(
            f"mdrun_{attempt_name}",
            mdrun,
            cwd=REPO_ROOT,
            env_overrides=run_env_delta,
            stdout_path=mdrun_stdout,
            stderr_path=mdrun_stderr,
        )
    )

    attempt_result: dict[str, object] = {
        "run_id": attempt_name,
        "artifact_root": str(run_root),
        "returncode": result.returncode,
        "stdout": str(mdrun_stdout),
        "stderr": str(mdrun_stderr),
        "failure_markers": extract_failure_markers(mdrun_stderr, mdrun_stdout),
        "grompp_stdout": str(grompp_stdout),
        "grompp_stderr": str(grompp_stderr),
        "env_overrides": run_env_delta,
        "argv": mdrun,
    }
    if result.returncode != 0:
        return attempt_result

    energy_dump = capture_output([str(gmx), "dump", "-e", str(deffnm.with_suffix(".edr"))], cwd=REPO_ROOT)
    trr_dump = capture_output([str(gmx), "dump", "-f", str(deffnm.with_suffix(".trr"))], cwd=REPO_ROOT)
    write_text(system_root / "summaries" / f"{attempt_name}_energy_dump.txt", energy_dump)
    write_text(system_root / "summaries" / f"{attempt_name}_trr_dump.txt", trr_dump)

    actual_events = parse_event_trace(run_root / "event_trace.tsv") if collect_trace_env else []
    energy_frames = parse_energy_dump(energy_dump)
    total_force_summary = parse_total_force_dump(run_root / "total_force.tsv") if collect_trace_env else {"entries": [], "per_step_totals": []}
    per_level_force_totals = parse_merge_trace_dir(run_root / "merge_trace") if collect_trace_env else {"entries": []}

    attempt_result.update(
        {
            "actual_events": actual_events,
            "energy_frames": energy_frames,
            "energy_summary": energy_summary(system_id, args.outer_steps, args.pair14_level, energy_frames),
            "total_force_summary": total_force_summary,
            "per_level_force_totals": per_level_force_totals,
            "full_outputs": {
                "tpr": str(deffnm.with_suffix(".tpr")),
                "edr": str(deffnm.with_suffix(".edr")),
                "trr": str(deffnm.with_suffix(".trr")),
                "cpt": str(deffnm.with_suffix(".cpt")),
                "gro": str(deffnm.with_suffix(".gro")),
                "log": str(deffnm.with_suffix(".log")),
                "event_trace_tsv": str(run_root / "event_trace.tsv") if collect_trace_env else None,
                "total_force_tsv": str(run_root / "total_force.tsv") if collect_trace_env else None,
                "merge_trace_dir": str(run_root / "merge_trace") if collect_trace_env else None,
                "m2p_trace_dir": str(run_root / "m2p_trace") if collect_trace_env else None,
            },
        }
    )
    return attempt_result


def collect_system_result(
    args: argparse.Namespace,
    gmx: Path,
    out_root: Path,
    system_id: str,
    gate_a_system: dict[str, object],
) -> dict[str, object]:
    system_root = out_root / system_id
    if system_root.exists():
        shutil.rmtree(system_root)
    inputs_dir = system_root / "inputs"
    logs_dir = system_root / "logs"
    summaries_dir = system_root / "summaries"
    for directory in (inputs_dir, logs_dir, summaries_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mdp_path = inputs_dir / "exact_respa_gate_b.mdp"
    write_text(mdp_path, make_exact_respa_mdp(args.outer_steps, args.pair14_level))

    commands: list[dict[str, object]] = []
    main_run = run_gpu_attempt(
        args=args,
        gmx=gmx,
        system_id=system_id,
        system_root=system_root,
        attempt_name="full",
        mdp_path=mdp_path,
        collect_trace_env=True,
        commands=commands,
    )

    system_result: dict[str, object] = {
        "system_id": system_id,
        "artifact_root": str(system_root),
        "gate_a_artifact_root": gate_a_system["artifact_root"],
        "mdp": str(mdp_path),
        "commands_json": str(summaries_dir / "commands.json"),
        "commands_sh": str(system_root / "run_commands.sh"),
        "main_run": main_run,
    }

    successful_repeats = []
    if main_run["returncode"] == 0:
        expected_event_trace = load_json(Path(gate_a_system["event_trace"]))["actual_event_trace"]
        expected_energy_frames = load_json(Path(gate_a_system["energy_terms"]))["frames"]
        expected_total_force_entries = load_json(Path(gate_a_system["total_force_summary"]))["per_step_totals"]
        expected_per_level_entries = load_json(Path(gate_a_system["per_level_force_totals"]))["entries"]

        event_order_comparison = compare_event_trace(main_run["actual_events"], expected_event_trace)
        total_force_comparison = compare_total_force_entries(
            main_run["total_force_summary"]["per_step_totals"], expected_total_force_entries
        )
        per_level_force_comparison = compare_per_level_force_entries(
            main_run["per_level_force_totals"]["entries"], expected_per_level_entries
        )
        energy_comparison = compare_energy_frames(main_run["energy_frames"], expected_energy_frames)
        virial_comparison = extract_virial_deltas(energy_comparison)

        system_result.update(
            {
                "event_order_comparison": event_order_comparison,
                "total_force_comparison": total_force_comparison,
                "per_level_force_comparison": per_level_force_comparison,
                "energy_comparison": energy_comparison,
                "virial_comparison": virial_comparison,
            }
        )

        for repeat_index in range(1, max(args.gpu_repeats, 1)):
            repeat_run = run_gpu_attempt(
                args=args,
                gmx=gmx,
                system_id=system_id,
                system_root=system_root,
                attempt_name=f"repeat_{repeat_index:02d}",
                mdp_path=mdp_path,
                collect_trace_env=True,
                commands=commands,
            )
            if repeat_run["returncode"] == 0:
                successful_repeats.append(repeat_run)
        successful_runs = [main_run, *successful_repeats]
        system_result["gpu_noise_floor"] = estimate_noise_floor(successful_runs)
        system_result["repeat_runs"] = successful_repeats

        restart_validation = collect_restart_validation(
            args,
            gmx,
            system_id,
            system_root,
            mdp_path,
            commands,
            base_env(args),
            env_delta(base_env(args), os.environ),
        )
        system_result["restart_validation"] = restart_validation
    else:
        system_result["gpu_noise_floor"] = {
            "available": False,
            "reason": "The primary Gate B GPU run did not complete successfully.",
            "successful_run_count": 0,
        }
        system_result["repeat_runs"] = []
        system_result["restart_validation"] = {
            "status": "BLOCKER",
            "reason": "Restart continuity cannot be tested because the primary Gate B GPU path did not execute.",
        }

    system_result["gate_b_assessment"] = assess_gate_b_system(system_result)
    system_result["first_failing_field"] = find_first_failure(system_result)
    write_text(summaries_dir / "system_result.json", json.dumps(system_result, indent=2, sort_keys=True) + "\n")
    write_text(summaries_dir / "commands.json", json.dumps(commands, indent=2, sort_keys=True) + "\n")
    write_commands_script(system_root / "run_commands.sh", commands)
    return system_result


def build_manifest(
    *,
    args: argparse.Namespace,
    out_root: Path,
    gate_a_manifest: dict[str, object],
    gmx: Path,
    gmx_version: str,
    gpu_inventory: dict[str, object],
    systems: list[dict[str, object]],
) -> dict[str, object]:
    gpu_support = parse_gpu_support(gmx_version)
    any_system_failed = any(system["main_run"]["returncode"] != 0 for system in systems)
    any_system_blocker = any(system["gate_b_assessment"]["status"] == "BLOCKER" for system in systems)
    all_systems_pass = all(system["gate_b_assessment"]["status"] == "PASS" for system in systems)

    status = "PASS" if all_systems_pass else "BLOCKER" if any_system_failed or any_system_blocker or gpu_support == "disabled" else "FAIL"
    gate_c_allowed = status == "PASS"

    blocking_reasons = []
    if gpu_support == "disabled":
        blocking_reasons.append("The selected gmx binary does not have GPU support enabled.")
    for system in systems:
        if system["main_run"]["returncode"] != 0:
            markers = system["main_run"]["failure_markers"]
            if markers:
                blocking_reasons.append(f"{system['system_id']}: {' | '.join(markers)}")
            else:
                blocking_reasons.append(f"{system['system_id']}: Gate B mdrun failed before comparison artifacts were produced.")
        elif system["gate_b_assessment"]["status"] != "PASS":
            for reason in system["gate_b_assessment"]["reasons"]:
                blocking_reasons.append(f"{system['system_id']}: {reason}")

    return {
        "schema_version": 1,
        "gate": "Gate B",
        "status": status,
        "objective": "Validate standalone exact r-RESPA with nonbonded GPU offload only and compare against Gate A.",
        "artifact_root": str(out_root),
        "gate_a_manifest": str(Path(args.gate_a_manifest).resolve()),
        "gate_a_status": gate_a_manifest.get("status"),
        "gmx": str(gmx),
        "gmx_version": gmx_version,
        "precision_mode": parse_precision_mode(gmx_version),
        "gpu_support": gpu_support,
        "hardware_configuration": gpu_inventory,
        "ntmpi": args.ntmpi,
        "ntomp": args.ntomp,
        "dlb": "no",
        "pme_rank_count": 0,
        "reproducibility_flags": [
            "-dlb no",
            "-pin off",
            "-nb gpu",
            "-pme cpu",
            "-bonded cpu",
            "-update cpu",
            "GMX_DISABLE_MODULAR_SIMULATOR=1",
        ],
        "binary_reproducibility_supported": False,
        "reproducibility_notes": [
            GPU_REPRODUCIBILITY_NOTE,
            "Determinism is constrained with single-rank execution, the recorded OpenMP thread count, DLB disabled, and measured repeated-run GPU noise floors.",
        ],
        "rerun_used": False,
        "normal_md_used": True,
        "comparison_basis": "Frozen Gate A CPU oracle manifest and per-system summaries.",
        "blocking_reasons": blocking_reasons,
        "systems": systems,
        "recommendation": {
            "gate_c_allowed": gate_c_allowed,
            "reason": "Gate C is allowed only if Gate B preserves event order, per-level ownership trace coverage, restart semantics, and a measured GPU noise floor while keeping offloaded-energy and force deviations within the measured GPU-noise order of magnitude."
            if gate_c_allowed
            else "Gate C must remain blocked until Gate B executes successfully and all comparisons are evidenced against Gate A.",
        },
    }


def write_manifest_markdown(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Gate B Oracle Comparison",
        "",
        f"- Status: {manifest['status']}",
        f"- Gate C allowed: {manifest['recommendation']['gate_c_allowed']}",
        f"- gmx: `{manifest['gmx']}`",
        f"- precision: `{manifest['precision_mode']}`",
        f"- GPU support: `{manifest['gpu_support']}`",
        f"- ntmpi / ntomp: `{manifest['ntmpi']}` / `{manifest['ntomp']}`",
        f"- DLB: `{manifest['dlb']}`",
        f"- PME ranks: `{manifest['pme_rank_count']}`",
        f"- Binary reproducibility supported: `{manifest['binary_reproducibility_supported']}`",
        "",
        "## Reproducibility Notes",
        "",
    ]
    for note in manifest["reproducibility_notes"]:
        lines.append(f"- Repro note: {note}")
    lines.append("")
    lines.append("## Blocking Reasons")
    lines.append("")
    if manifest["blocking_reasons"]:
        for reason in manifest["blocking_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- None")
    lines.extend(["", "## Systems", ""])
    for system in manifest["systems"]:
        lines.append(f"### {system['system_id']}")
        lines.append("")
        lines.append(f"- Gate B assessment: `{system['gate_b_assessment']['status']}`")
        for reason in system["gate_b_assessment"]["reasons"]:
            lines.append(f"- Assessment note: `{reason}`")
        lines.append(f"- Main run return code: `{system['main_run']['returncode']}`")
        lines.append(f"- First nonzero comparison field: `{system['first_failing_field']}`")
        lines.append(f"- Artifact root: `{system['artifact_root']}`")
        lines.append(f"- Command script: `{system['commands_sh']}`")
        lines.append("")
    write_text(path, "\n".join(lines) + "\n")


def maybe_build(args: argparse.Namespace, build_dir: Path | None) -> None:
    if args.skip_build:
        return
    command = [
        "cmake",
        "--build",
        str(build_dir if build_dir is not None else (Path(args.gmx).resolve().parents[1])),
        "--target",
        args.build_target,
        "-j4",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True, text=True)


def validate_inputs(gate_a_manifest: dict[str, object]) -> None:
    if gate_a_manifest.get("status") != "PASS":
        raise ValueError("Gate A manifest is not PASS; Gate B cannot use it as a frozen oracle.")


def main() -> None:
    args = parse_args()
    gmx = Path(args.gmx).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    gate_a_manifest = load_json(Path(args.gate_a_manifest).resolve())
    validate_inputs(gate_a_manifest)
    maybe_build(args, Path(args.build_dir).resolve() if args.build_dir is not None else None)

    gmx_version = capture_output([str(gmx), "--version"], cwd=REPO_ROOT)
    gpu_inventory = {
        "nvidia_smi_list": capture_optional_output(["nvidia-smi", "-L"]),
        "nvidia_smi_query": capture_optional_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "nvcc_version": capture_optional_output(["nvcc", "--version"]),
    }

    systems = []
    gate_a_systems_by_id = {system["system_id"]: system for system in gate_a_manifest["systems"]}
    for system_id in ("small_oligomer", "small_salt_polymer_box"):
        systems.append(
            collect_system_result(
                args,
                gmx,
                out_root,
                system_id,
                gate_a_systems_by_id[system_id],
            )
        )

    manifest = build_manifest(
        args=args,
        out_root=out_root,
        gate_a_manifest=gate_a_manifest,
        gmx=gmx,
        gmx_version=gmx_version,
        gpu_inventory=gpu_inventory,
        systems=systems,
    )
    write_text(out_root / "gate_b_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_manifest_markdown(out_root / "gate_b_manifest.md", manifest)


if __name__ == "__main__":
    main()
