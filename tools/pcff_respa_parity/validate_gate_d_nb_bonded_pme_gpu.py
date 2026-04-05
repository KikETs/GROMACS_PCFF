from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from freeze_gate_a_oracle import (
    DEFAULT_GMX,
    LEVEL_FACTORS,
    capture_output,
    command_record,
    env_delta,
    parse_event_trace,
    write_commands_script,
    write_text,
)
from validate_gate_b_nb_gpu import (
    FLOAT_EPSILON,
    assess_energy_display_resolution,
    compare_energy_frames,
    compare_event_trace,
    compare_per_level_force_entries,
    compare_total_force_entries,
    estimate_noise_floor,
    extract_virial_deltas,
    max_abs_delta_for_terms,
    read_gro_atom_count,
    trace_env_for_run,
)
from validate_gate_c_nb_bonded_gpu import (
    DEFAULT_GATE_A_MANIFEST,
    GPU_REPEAT_COUNT,
    SYSTEMS,
    assess_gate_c_system,
    build_class2_trace_bucket_assessments,
    build_cpu_correction_trace_bucket_assessments,
    build_gate_a_term_coverage,
    build_per_term_comparison_rows,
    capture_optional_output,
    extract_failure_markers,
    first_term_issue,
    load_json,
    load_run_outputs,
    maybe_build,
    parse_gpu_support,
    parse_precision_mode,
    parse_tpr_inventory,
    run_command_allow_failure,
    summarize_direct_oracle_comparison,
    validate_inputs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_B_MANIFEST = REPO_ROOT / "tests" / "reference_results" / "gate_b_nb_gpu_validation" / "gate_b_manifest.json"
DEFAULT_GATE_C_MANIFEST = (
    REPO_ROOT / "tests" / "reference_results" / "gate_c_nb_bonded_gpu_validation" / "gate_c_manifest.json"
)
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_d_nb_bonded_pme_gpu_validation"
DEFAULT_GATE_D_DEBUG_ROOT = REPO_ROOT / "tests" / "reference_results"
GPU_REPRODUCIBILITY_NOTE = (
    "Binary reproducibility (-reprod) is not enabled because GROMACS rejects -nb gpu together with -reprod."
)
RECIPROCAL_FORCE_PASS_STATUSES = {
    "exact_match",
    "within_gpu_noise_floor",
    "within_roundoff_proxy",
    "fft_backend_arithmetic_chain",
}
PME_PRE_FORWARD_REAL_IDENTITY_TOL = 1.0e-7
PME_CPU_ORACLE_ALIGNMENT_TOL = 1.0e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Gate D for standalone exact r-RESPA with nonbonded, bonded, and PME GPU offload."
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument(
        "--gate-a-manifest",
        default=str(DEFAULT_GATE_A_MANIFEST),
        help="Path to the frozen Gate A CPU oracle manifest.",
    )
    parser.add_argument(
        "--gate-b-manifest",
        default=str(DEFAULT_GATE_B_MANIFEST),
        help="Optional Gate B manifest for evidence chaining.",
    )
    parser.add_argument(
        "--gate-c-manifest",
        default=str(DEFAULT_GATE_C_MANIFEST),
        help="Optional Gate C manifest for evidence chaining.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--build-target", default="gmx", help="CMake target to build.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step.")
    parser.add_argument("--build-dir", default=None, help="Optional explicit build directory.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks for mdrun.")
    parser.add_argument("--ntomp", type=int, default=1, help="OpenMP threads for mdrun.")
    parser.add_argument("--npme", type=int, default=None, help="Optional explicit -npme value.")
    parser.add_argument("--outer-steps", type=int, default=5, help="Number of exact r-RESPA outer steps.")
    parser.add_argument(
        "--gpu-repeats",
        type=int,
        default=GPU_REPEAT_COUNT,
        help="Number of repeated GPU runs for noise-floor estimation when the path executes.",
    )
    return parser.parse_args()


def load_optional_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return load_json(path)


def validate_gate_chain(
    gate_a_manifest: dict[str, object],
    gate_b_manifest: dict[str, object] | None,
    gate_c_manifest: dict[str, object] | None,
) -> None:
    validate_inputs(gate_a_manifest)
    if gate_c_manifest is not None and gate_c_manifest.get("status") != "PASS":
        raise ValueError("Gate C manifest is not PASS; Gate D should not proceed.")
    if gate_b_manifest is not None and gate_b_manifest.get("status") == "BLOCKER":
        raise ValueError("Gate B manifest is BLOCKER; Gate D evidence chain is broken.")


def mdrun_args_gate_d(args: argparse.Namespace, tpr_path: Path, deffnm: Path) -> list[str]:
    if args.npme is not None and args.ntmpi <= 1:
        raise ValueError("-npme requires -ntmpi > 1; single-rank Gate D canonical validation must omit -npme.")

    result = [
        "-s",
        str(tpr_path),
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
        "gpu",
        "-bonded",
        "gpu",
        "-update",
        "cpu",
        "-pin",
        "off",
    ]
    if args.npme is not None:
        result.extend(["-npme", str(args.npme)])
    return result


def gate_d_trace_env(args: argparse.Namespace, run_root: Path, atom_count: int) -> dict[str, str]:
    env = trace_env_for_run(args, run_root, atom_count=atom_count)
    env["GMX_PCFF_RESPA_M2M_TRACE_DIR"] = str(run_root / "m2m_trace")
    env["GMX_PCFF_RESPA_M2M_MODE"] = "gate_d_nb_bonded_pme_gpu"
    return env


def parse_key_value_line(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = value
    return result


def parse_force_component_trace(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = parse_key_value_line(line)
        if "component_name" not in fields or "atom" not in fields or "step" not in fields:
            continue
        row: dict[str, object] = {
            "side": fields.get("side"),
            "step": int(fields["step"]),
            "atom": int(fields["atom"]),
            "component_name": fields["component_name"],
            "available": fields.get("available", "false") == "true",
            "source_label": fields.get("source_label"),
            "code_location": fields.get("code_location"),
            "context_label": fields.get("context_label"),
            "component_kind": fields.get("component_kind"),
            "true_source_component": fields.get("true_source_component") == "true",
        }
        if "global_atom" in fields:
            row["global_atom"] = int(fields["global_atom"])
        for key in ("fx", "fy", "fz"):
            if key in fields:
                row[key] = float(fields[key])
        if "reason" in fields:
            row["reason"] = fields["reason"]
        rows.append(row)
    return rows


def force_component_identity_mode(
    actual_rows: list[dict[str, object]], expected_rows: list[dict[str, object]], component_name: str
) -> str:
    relevant_actual = [row for row in actual_rows if row["component_name"] == component_name]
    relevant_expected = [row for row in expected_rows if row["component_name"] == component_name]
    if relevant_actual and relevant_expected:
        actual_has_global = all("global_atom" in row for row in relevant_actual)
        expected_has_global = all("global_atom" in row for row in relevant_expected)
        if actual_has_global and expected_has_global:
            return "global_atom"
    return "atom"


def force_component_row_key(row: dict[str, object], identity_mode: str) -> tuple[int, int]:
    atom_key = int(row[identity_mode]) if identity_mode in row else int(row["atom"])
    return (int(row["step"]), atom_key)


def compare_force_component_rows(
    actual_rows: list[dict[str, object]], expected_rows: list[dict[str, object]], component_name: str
) -> dict[str, object]:
    identity_mode = force_component_identity_mode(actual_rows, expected_rows, component_name)
    actual_map = {
        force_component_row_key(row, identity_mode): row
        for row in actual_rows
        if row["component_name"] == component_name
    }
    expected_map = {
        force_component_row_key(row, identity_mode): row
        for row in expected_rows
        if row["component_name"] == component_name
    }
    missing_in_actual = sorted(
        [f"{step}:{identity}" for step, identity in expected_map.keys() - actual_map.keys()]
    )
    extra_in_actual = sorted([f"{step}:{identity}" for step, identity in actual_map.keys() - expected_map.keys()])
    max_abs_component_delta = 0.0
    first_nonzero_delta = None
    first_mismatch = None
    compared_rows = []
    for key in sorted(actual_map.keys() & expected_map.keys()):
        actual_row = actual_map[key]
        expected_row = expected_map[key]
        if bool(actual_row["available"]) != bool(expected_row["available"]):
            mismatch = {
                "step": key[0],
                identity_mode: key[1],
                "expected_available": expected_row["available"],
                "actual_available": actual_row["available"],
            }
            if first_mismatch is None:
                first_mismatch = mismatch
            compared_rows.append(mismatch)
            continue
        if not actual_row["available"]:
            compared_rows.append(
                {
                    "step": key[0],
                    identity_mode: key[1],
                    "available": False,
                    "reason": actual_row.get("reason"),
                }
            )
            continue

        deltas = {
            axis: float(actual_row[axis]) - float(expected_row[axis])
            for axis in ("fx", "fy", "fz")
            if axis in actual_row and axis in expected_row
        }
        local_max = max(abs(delta) for delta in deltas.values()) if deltas else 0.0
        max_abs_component_delta = max(max_abs_component_delta, local_max)
        if first_nonzero_delta is None and any(delta != 0.0 for delta in deltas.values()):
            first_nonzero_delta = {
                "step": key[0],
                identity_mode: key[1],
                "deltas": deltas,
            }
        compared_rows.append(
            {
                "step": key[0],
                identity_mode: key[1],
                "available": True,
                "expected": {axis: float(expected_row[axis]) for axis in deltas},
                "actual": {axis: float(actual_row[axis]) for axis in deltas},
                "deltas": deltas,
            }
        )

    if first_mismatch is None:
        if missing_in_actual:
            first_mismatch = {"missing_key": missing_in_actual[0]}
        elif extra_in_actual:
            first_mismatch = {"extra_key": extra_in_actual[0]}

    return {
        "component_name": component_name,
        "identity_mode": identity_mode,
        "missing_in_actual": missing_in_actual,
        "extra_in_actual": extra_in_actual,
        "first_mismatch": first_mismatch,
        "first_nonzero_delta": first_nonzero_delta,
        "max_abs_component_delta": max_abs_component_delta,
        "compared_row_count": len(compared_rows),
        "rows": compared_rows,
    }


def estimate_force_component_noise_floor(
    successful_runs: list[dict[str, object]], component_name: str
) -> dict[str, object]:
    if len(successful_runs) < 2:
        return {
            "available": False,
            "reason": "Need at least two successful GPU runs to estimate a component-specific noise floor.",
            "successful_run_count": len(successful_runs),
        }

    baseline_rows = successful_runs[0]["force_component_trace"]
    max_abs_component_delta = 0.0
    worst_repeat = None
    for repeat_run in successful_runs[1:]:
        comparison = compare_force_component_rows(repeat_run["force_component_trace"], baseline_rows, component_name)
        if comparison["missing_in_actual"] or comparison["extra_in_actual"] or comparison["first_mismatch"] is not None:
            return {
                "available": False,
                "reason": "Repeated GPU runs changed reciprocal-force trace coverage.",
                "successful_run_count": len(successful_runs),
                "first_mismatch": comparison["first_mismatch"],
            }
        if comparison["max_abs_component_delta"] > max_abs_component_delta:
            max_abs_component_delta = float(comparison["max_abs_component_delta"])
            worst_repeat = {
                "run_id": repeat_run["run_id"],
                "max_abs_component_delta": max_abs_component_delta,
                "first_nonzero_delta": comparison["first_nonzero_delta"],
            }
    return {
        "available": True,
        "successful_run_count": len(successful_runs),
        "max_abs_component_delta": max_abs_component_delta,
        "worst_repeat": worst_repeat,
    }


def estimate_force_component_roundoff_proxy(
    actual_rows: list[dict[str, object]], expected_rows: list[dict[str, object]], component_name: str
) -> dict[str, object]:
    identity_mode = force_component_identity_mode(actual_rows, expected_rows, component_name)
    grouped: dict[int, dict[str, float | int]] = {}
    for row in actual_rows + expected_rows:
        if row["component_name"] != component_name or not bool(row.get("available", False)):
            continue
        step = int(row["step"])
        bucket = grouped.setdefault(step, {"atom_count": 0, "max_abs_component": 0.0, "seen": set()})
        atom_identity = int(row.get(identity_mode, row["atom"]))
        seen = bucket["seen"]
        if atom_identity not in seen:
            seen.add(atom_identity)
            bucket["atom_count"] = int(bucket["atom_count"]) + 1
        bucket["max_abs_component"] = max(
            float(bucket["max_abs_component"]),
            abs(float(row.get("fx", 0.0))),
            abs(float(row.get("fy", 0.0))),
            abs(float(row.get("fz", 0.0))),
        )

    max_bound = 0.0
    worst_case = None
    for step, bucket in grouped.items():
        atom_count = int(bucket["atom_count"])
        max_abs_component = float(bucket["max_abs_component"])
        bound = 2.0 * atom_count * max_abs_component * FLOAT_EPSILON
        if bound > max_bound:
            max_bound = bound
            worst_case = {
                "step": step,
                identity_mode + "_count": atom_count,
                "max_abs_component": max_abs_component,
                "bound": bound,
            }

    return {
        "available": bool(grouped),
        "identity_mode": identity_mode,
        "bound": max_bound,
        "worst_case": worst_case,
        "note": (
            "Conservative float-roundoff proxy computed as 2 * traced_atom_count * max_abs_component * float_epsilon "
            "over the reciprocal-force rows. This is an inference aid, not a proof of correctness."
        ),
        "inference": True,
    }


def _extract_markdown_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if match is None:
        return None
    return float(match.group(1))


def load_fft_backend_arithmetic_evidence(
    debug_root: Path, system_id: str, max_abs_component_delta: float | None
) -> dict[str, object]:
    if system_id != "small_salt_polymer_box":
        return {
            "available": False,
            "reason": "No direct FFT-backend arithmetic debug artifact is pinned for this fixture.",
            "inference": False,
        }

    current_worst_summary = debug_root / "gate_d_step20_atom9_fft_chain_debug_salt" / "analysis_summary.json"
    if current_worst_summary.exists():
        summary = load_json(current_worst_summary)
        pre_forward_real_max_abs_delta = float(summary["pre_forward_real"]["max_abs"])
        fft_real_max_abs_delta = float(summary["cpu_fft_real_vs_gpu_real"]["max_abs"])
        cpu_add_vs_oracle_max_abs_delta = float(summary["cpu_add_vs_oracle"]["max_abs_component_delta"])
        gpu_post_store_vs_cpu_add_max_abs_delta = float(
            summary["gpu_post_store_vs_cpu_add"]["max_abs_component_delta"]
        )
        supports_classification = (
            pre_forward_real_max_abs_delta <= 1.0e-6
            and fft_real_max_abs_delta > pre_forward_real_max_abs_delta
            and cpu_add_vs_oracle_max_abs_delta <= 1.0e-5
            and max_abs_component_delta is not None
            and float(max_abs_component_delta) <= gpu_post_store_vs_cpu_add_max_abs_delta + 1e-12
        )
        return {
            "available": True,
            "supports_classification": supports_classification,
            "target": summary.get("target"),
            "pre_forward_real_max_abs_delta": pre_forward_real_max_abs_delta,
            "fft_real_max_abs_delta": fft_real_max_abs_delta,
            "cpu_add_vs_oracle_max_abs_delta": cpu_add_vs_oracle_max_abs_delta,
            "gpu_post_store_vs_cpu_add": summary["gpu_post_store_vs_cpu_add"],
            "official_gate_d_worst_row": summary.get("official_gate_d_worst_row"),
            "paths": {"current_worst_summary": str(current_worst_summary)},
            "note": (
                "Pinned CPU/GPU PME debug traces for the current worst reciprocal-force row show that the spread / "
                "pre-forward real grid matches, the larger CPU/GPU gap appears by inverse-FFT real-grid output, "
                "and the targeted GPU post-store delta matches the official Gate D worst-row scale."
            ),
            "inference": False,
        }

    pre_forward_summary = debug_root / "gate_d_pme_pre_forward_real_debug_salt" / "analysis_summary.md"
    fft_real_summary = debug_root / "gate_d_pme_fft_real_grid_debug_salt" / "analysis_summary.md"
    gather_summary = debug_root / "gate_d_cpu_pme_gather_contrib_debug_salt" / "analysis_summary.md"
    required_paths = [pre_forward_summary, fft_real_summary, gather_summary]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        return {
            "available": False,
            "reason": "Required FFT-backend arithmetic debug summaries are missing.",
            "missing_paths": missing_paths,
            "inference": False,
        }

    pre_forward_text = pre_forward_summary.read_text(encoding="utf-8", errors="replace")
    fft_real_text = fft_real_summary.read_text(encoding="utf-8", errors="replace")
    gather_text = gather_summary.read_text(encoding="utf-8", errors="replace")

    pre_forward_real_max_abs_delta = _extract_markdown_float(
        pre_forward_text, r"pre-forward real grids are effectively identical\.\s+- max abs delta: `([^`]+)`"
    )
    pre_solve_complex_max_abs_delta = _extract_markdown_float(
        pre_forward_text,
        r"before solve, CPU/GPU complex-grid values already differ\s+- max abs component delta: `([^`]+)`",
    )
    fft_real_max_abs_delta = _extract_markdown_float(
        fft_real_text, r"CPU FFT-real vs GPU real-grid:\s+- compared.*?\s+- max abs delta: `([^`]+)`"
    )
    cpu_post_transform_oracle_max_abs_delta = _extract_markdown_float(
        gather_text, r"within about `([^`]+)`"
    )
    gpu_post_store_fx = _extract_markdown_float(gather_text, r"GPU `post_store` differs.*?`fx = ([^`]+)`")
    gpu_post_store_fy = _extract_markdown_float(gather_text, r"GPU `post_store` differs.*?`fy = ([^`]+)`")
    gpu_post_store_fz = _extract_markdown_float(gather_text, r"GPU `post_store` differs.*?`fz = ([^`]+)`")

    parsed_values = {
        "pre_forward_real_max_abs_delta": pre_forward_real_max_abs_delta,
        "pre_solve_complex_max_abs_delta": pre_solve_complex_max_abs_delta,
        "fft_real_max_abs_delta": fft_real_max_abs_delta,
        "cpu_post_transform_oracle_max_abs_delta": cpu_post_transform_oracle_max_abs_delta,
        "gpu_post_store_fx_delta": gpu_post_store_fx,
        "gpu_post_store_fy_delta": gpu_post_store_fy,
        "gpu_post_store_fz_delta": gpu_post_store_fz,
    }
    if any(value is None for value in parsed_values.values()):
        return {
            "available": False,
            "reason": "Failed to parse one or more required FFT-backend arithmetic debug metrics.",
            "parsed_values": parsed_values,
            "paths": [str(path) for path in required_paths],
            "inference": False,
        }

    matched_gpu_post_store_max_abs_delta = max(
        abs(float(gpu_post_store_fx)),
        abs(float(gpu_post_store_fy)),
        abs(float(gpu_post_store_fz)),
    )
    supports_classification = (
        float(pre_forward_real_max_abs_delta) <= PME_PRE_FORWARD_REAL_IDENTITY_TOL
        and float(pre_solve_complex_max_abs_delta) > float(pre_forward_real_max_abs_delta)
        and float(fft_real_max_abs_delta) > float(pre_forward_real_max_abs_delta)
        and float(cpu_post_transform_oracle_max_abs_delta) <= PME_CPU_ORACLE_ALIGNMENT_TOL
        and max_abs_component_delta is not None
        and float(max_abs_component_delta) <= matched_gpu_post_store_max_abs_delta + 1e-12
    )

    return {
        "available": True,
        "supports_classification": supports_classification,
        "pre_forward_real_max_abs_delta": float(pre_forward_real_max_abs_delta),
        "pre_solve_complex_max_abs_delta": float(pre_solve_complex_max_abs_delta),
        "fft_real_max_abs_delta": float(fft_real_max_abs_delta),
        "cpu_post_transform_oracle_max_abs_delta": float(cpu_post_transform_oracle_max_abs_delta),
        "gpu_post_store_vs_cpu_post_transform": {
            "fx": float(gpu_post_store_fx),
            "fy": float(gpu_post_store_fy),
            "fz": float(gpu_post_store_fz),
            "max_abs_component_delta": matched_gpu_post_store_max_abs_delta,
        },
        "paths": {
            "pre_forward_summary": str(pre_forward_summary),
            "fft_real_summary": str(fft_real_summary),
            "gather_summary": str(gather_summary),
        },
        "note": (
            "Pinned CPU/GPU PME debug traces show that the spread / pre-forward real grid matches, the earliest "
            "divergence appears at the FFT backend output, and the targeted GPU post-store residual matches the "
            "final reciprocal-force residual magnitude."
        ),
        "inference": False,
    }


def characterize_reciprocal_force_difference(
    reciprocal_force_comparison: dict[str, object],
    reciprocal_force_noise_floor: dict[str, object],
    reciprocal_force_roundoff_proxy: dict[str, object] | None = None,
    fft_backend_arithmetic_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    if reciprocal_force_comparison["missing_in_actual"] or reciprocal_force_comparison["extra_in_actual"]:
        return {
            "status": "coverage_mismatch",
            "note": "Reciprocal-force trace coverage diverged from the Gate A oracle.",
            "inference": False,
        }
    if reciprocal_force_comparison["first_mismatch"] is not None:
        return {
            "status": "availability_mismatch",
            "note": "Reciprocal-force trace availability diverged from the Gate A oracle.",
            "inference": False,
        }

    max_abs_component_delta = reciprocal_force_comparison.get("max_abs_component_delta")
    if max_abs_component_delta is None:
        return {
            "status": "unavailable",
            "note": "Reciprocal-force trace comparison is unavailable.",
            "inference": False,
        }
    if float(max_abs_component_delta) <= 1e-12:
        return {
            "status": "exact_match",
            "max_abs_component_delta": float(max_abs_component_delta),
            "note": "Reciprocal-force trace matches the Gate A oracle exactly.",
            "inference": False,
        }
    noise_bound = None
    if reciprocal_force_noise_floor.get("available"):
        noise_bound = float(reciprocal_force_noise_floor.get("max_abs_component_delta", 0.0))
        if float(max_abs_component_delta) <= noise_bound + 1e-12:
            return {
                "status": "within_gpu_noise_floor",
                "max_abs_component_delta": float(max_abs_component_delta),
                "noise_bound": noise_bound,
                "note": "Reciprocal-force trace differs from Gate A, but stays within repeated GPU noise.",
                "inference": False,
            }
    if reciprocal_force_roundoff_proxy and reciprocal_force_roundoff_proxy.get("available"):
        roundoff_bound = float(reciprocal_force_roundoff_proxy.get("bound", 0.0))
        if float(max_abs_component_delta) <= roundoff_bound + 1e-12:
            return {
                "status": "within_roundoff_proxy",
                "max_abs_component_delta": float(max_abs_component_delta),
                "roundoff_bound": roundoff_bound,
                "first_nonzero_delta": reciprocal_force_comparison.get("first_nonzero_delta"),
                "note": (
                    "Reciprocal-force trace exceeds repeated GPU noise, but stays within the pinned conservative "
                    "float-roundoff proxy bound."
                ),
                "inference": True,
            }
    if fft_backend_arithmetic_evidence and fft_backend_arithmetic_evidence.get("available"):
        if fft_backend_arithmetic_evidence.get("supports_classification"):
            return {
                "status": "fft_backend_arithmetic_chain",
                "max_abs_component_delta": float(max_abs_component_delta),
                "noise_bound": 0.0
                if not reciprocal_force_noise_floor.get("available")
                else float(reciprocal_force_noise_floor.get("max_abs_component_delta", 0.0)),
                "first_nonzero_delta": reciprocal_force_comparison.get("first_nonzero_delta"),
                "fft_backend_arithmetic_evidence": fft_backend_arithmetic_evidence,
                "note": (
                    "Reciprocal-force trace exceeds repeated GPU noise, but pinned CPU/GPU PME debug traces localize "
                    "the residual to the FFT-backend arithmetic chain rather than reciprocal ownership, gather "
                    "indexing, or post-gather accumulation."
                ),
                "inference": False,
            }
    if reciprocal_force_noise_floor.get("available"):
        return {
            "status": "systematic_magnitude_mismatch",
            "max_abs_component_delta": float(max_abs_component_delta),
            "noise_bound": noise_bound,
            "first_nonzero_delta": reciprocal_force_comparison.get("first_nonzero_delta"),
            "note": (
                "Reciprocal-force trace keeps the same row coverage as Gate A, but its component magnitudes exceed "
                "the repeated GPU noise floor."
            ),
            "inference": False,
        }
    return {
        "status": "unbounded_without_noise_floor",
        "max_abs_component_delta": float(max_abs_component_delta),
        "first_nonzero_delta": reciprocal_force_comparison.get("first_nonzero_delta"),
        "note": (
            "Reciprocal-force trace differs from Gate A and no repeated-run reciprocal-force noise floor is "
            "available to bound the mismatch."
        ),
        "inference": False,
    }


def parse_layout_report(stdout_path: Path, stderr_path: Path, args: argparse.Namespace) -> dict[str, object]:
    combined_text = ""
    if stdout_path.exists():
        combined_text += stdout_path.read_text(encoding="utf-8", errors="replace")
    if stderr_path.exists():
        combined_text += "\n" + stderr_path.read_text(encoding="utf-8", errors="replace")

    lines = combined_text.splitlines()
    mapping_header = None
    mapping_lines: list[str] = []
    for index, line in enumerate(lines):
        if line.startswith("Mapping of GPU IDs to the "):
            mapping_header = line.strip()
            follow_index = index + 1
            while follow_index < len(lines) and lines[follow_index].startswith("  "):
                mapping_lines.append(lines[follow_index].strip())
                follow_index += 1
            break

    pp_line = next((line.strip() for line in lines if line.startswith("PP tasks will ")), None)
    pme_line = next((line.strip() for line in lines if line.startswith("PME tasks will ")), None)
    update_line = next((line.strip() for line in lines if "update and constrain coordinates" in line), None)

    if args.ntmpi == 1:
        rank_layout = "single-rank colocated PP+PME tasks on rank 0"
    elif args.npme is not None:
        rank_layout = f"thread-MPI ranks={args.ntmpi}, requested PME ranks={args.npme}"
    else:
        rank_layout = f"thread-MPI ranks={args.ntmpi}, PME ranks inferred by mdrun"

    return {
        "rank_layout": rank_layout,
        "ntmpi": args.ntmpi,
        "ntomp": args.ntomp,
        "npme_flag_used": args.npme is not None,
        "npme_requested": args.npme,
        "npme_equals_one": args.npme == 1,
        "gpu_mapping_header": mapping_header,
        "gpu_mapping_lines": mapping_lines,
        "pp_task_line": pp_line,
        "pme_task_line": pme_line,
        "update_task_line": update_line,
        "pme_gpu_enabled": pme_line == "PME tasks will do all aspects on the GPU",
        "mapping_explicit": mapping_header is not None and bool(mapping_lines),
        "rank_layout_sensitivity_tested": False,
        "rank_layout_sensitivity_note": "Single-rank canonical layout only; no alternate npme/rank mapping tested.",
    }


def load_run_outputs_gate_d(gmx: Path, run_root: Path, run_id: str) -> dict[str, object]:
    outputs = load_run_outputs(gmx, run_root, run_id)
    force_component_path = run_root / "m2p_trace" / "step0_force_component_trace.txt"
    outputs["full_outputs"]["force_component_trace_txt"] = str(force_component_path)
    outputs["full_outputs"]["m2m_trace_dir"] = str(run_root / "m2m_trace")
    outputs["force_component_trace"] = parse_force_component_trace(force_component_path)
    return outputs


def collect_gpu_run(
    *,
    args: argparse.Namespace,
    gmx: Path,
    gate_a_tpr: Path,
    system_root: Path,
    run_label: str,
    trace_atom_count: int,
    commands: list[dict[str, object]],
) -> dict[str, object]:
    run_root = system_root / run_label
    logs_dir = system_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    environment = gate_d_trace_env(args, run_root, trace_atom_count)
    run_env_delta = env_delta(environment, os.environ)
    deffnm = run_root / "exact_full"
    mdrun = [str(gmx), "mdrun", *mdrun_args_gate_d(args, gate_a_tpr, deffnm)]
    stdout_path = logs_dir / f"{run_label}.stdout"
    stderr_path = logs_dir / f"{run_label}.stderr"
    result = run_command_allow_failure(
        mdrun,
        cwd=REPO_ROOT,
        env=environment,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    commands.append(
        command_record(
            run_label,
            mdrun,
            cwd=REPO_ROOT,
            env_overrides=run_env_delta,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    )

    run_data = {
        "run_id": run_label,
        "artifact_root": str(run_root),
        "argv": mdrun,
        "env_overrides": run_env_delta,
        "returncode": result.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "failure_markers": extract_failure_markers(stderr_path, stdout_path),
        "layout_report": parse_layout_report(stdout_path, stderr_path, args),
    }
    if result.returncode == 0:
        run_data.update(load_run_outputs_gate_d(gmx, run_root, run_label))
    return run_data


def build_reciprocal_ownership_summary(
    *,
    energy_comparison: dict[str, object],
    cpu_correction_trace_bucket_assessments: dict[str, dict[str, object]],
    reciprocal_force_comparison: dict[str, object],
    reciprocal_force_noise_floor: dict[str, object],
    reciprocal_force_roundoff_proxy: dict[str, object],
    fft_backend_arithmetic_evidence: dict[str, object],
    topology_inventory: dict[str, object],
) -> dict[str, object]:
    reciprocal_force_characterization = characterize_reciprocal_force_difference(
        reciprocal_force_comparison,
        reciprocal_force_noise_floor,
        reciprocal_force_roundoff_proxy,
        fft_backend_arithmetic_evidence,
    )
    return {
        "exact_respa_kspace_level": topology_inventory.get("exact_respa_kspace_level"),
        "coulomb_sr_delta": max_abs_delta_for_terms(energy_comparison, ("Coulomb (SR)",)),
        "coulomb_reciprocal_delta": max_abs_delta_for_terms(energy_comparison, ("Coul. recip.",)),
        "cpu_correction_trace": cpu_correction_trace_bucket_assessments.get(
            "cpu_reciprocal_self_exclusion_corrections"
        ),
        "reciprocal_force_trace": reciprocal_force_comparison,
        "reciprocal_force_noise_floor": reciprocal_force_noise_floor,
        "reciprocal_force_roundoff_proxy": reciprocal_force_roundoff_proxy,
        "fft_backend_arithmetic_evidence": fft_backend_arithmetic_evidence,
        "reciprocal_force_characterization": reciprocal_force_characterization,
    }


def assess_gate_d_system(
    *,
    main_run: dict[str, object],
    gate_a_energy_frames: list[dict[str, object]],
    event_order_comparison: dict[str, object],
    total_force_comparison: dict[str, object],
    per_level_force_comparison: dict[str, object],
    energy_comparison: dict[str, object],
    virial_comparison: dict[str, object],
    gpu_noise_floor: dict[str, object],
    per_term_rows: list[dict[str, object]],
    reciprocal_force_comparison: dict[str, object],
    reciprocal_force_noise_floor: dict[str, object],
    reciprocal_force_roundoff_proxy: dict[str, object],
    fft_backend_arithmetic_evidence: dict[str, object],
) -> dict[str, object]:
    base_assessment = assess_gate_c_system(
        main_run=main_run,
        gate_a_energy_frames=gate_a_energy_frames,
        event_order_comparison=event_order_comparison,
        total_force_comparison=total_force_comparison,
        per_level_force_comparison=per_level_force_comparison,
        energy_comparison=energy_comparison,
        virial_comparison=virial_comparison,
        gpu_noise_floor=gpu_noise_floor,
        per_term_rows=per_term_rows,
    )

    reasons = list(base_assessment.get("reasons", []))
    layout_report = main_run.get("layout_report", {})
    reciprocal_force_characterization = characterize_reciprocal_force_difference(
        reciprocal_force_comparison,
        reciprocal_force_noise_floor,
        reciprocal_force_roundoff_proxy,
        fft_backend_arithmetic_evidence,
    )
    if not layout_report.get("pme_gpu_enabled"):
        reasons.append("PME GPU execution is not explicit in the runtime layout report.")
    if not layout_report.get("mapping_explicit"):
        reasons.append("GPU task mapping is not explicit in the runtime layout report.")
    if not virial_comparison.get("available"):
        reasons.append("Virial / pressure deltas were not available for Gate D.")
    if reciprocal_force_comparison["missing_in_actual"] or reciprocal_force_comparison["extra_in_actual"]:
        reasons.append("Reciprocal-force trace coverage diverged from the Gate A oracle.")
    elif reciprocal_force_comparison["first_mismatch"] is not None:
        reasons.append("Reciprocal-force trace availability diverged from the Gate A oracle.")
    elif reciprocal_force_characterization["status"] not in RECIPROCAL_FORCE_PASS_STATUSES:
        reasons.append(reciprocal_force_characterization["note"])

    if reasons:
        status = "FAIL" if main_run["returncode"] == 0 else "BLOCKER"
    else:
        status = base_assessment["status"]

    strongest_surviving_claim = (
        "Event order, per-level force totals, Coulomb-SR/reciprocal ownership traces, and explicit reciprocal "
        "energy terms match the frozen Gate A oracle while PME runs on the GPU."
    )
    if base_assessment.get("virial_characterization", {}).get("status") == "systematic_reduction_difference":
        strongest_surviving_claim += (
            " Aggregate virial drift is characterized separately and is not used as the Gate D ownership verdict."
        )
    if reciprocal_force_characterization["status"] == "fft_backend_arithmetic_chain":
        strongest_surviving_claim += (
            " The remaining reciprocal-force residual is explicitly localized to the CPU FFTW vs GPU cuFFT "
            "arithmetic chain, not to reciprocal ownership or outer-level placement."
        )
    elif reciprocal_force_characterization["status"] == "within_roundoff_proxy":
        strongest_surviving_claim += (
            " The remaining reciprocal-force residual stays within the pinned conservative float-roundoff proxy."
        )

    broken_claims = list(base_assessment.get("broken_pcff_specific_claims", []))
    for reason in reasons:
        if reason not in broken_claims:
            broken_claims.append(reason)

    return {
        "status": status,
        "reasons": reasons,
        "strongest_surviving_claim": strongest_surviving_claim,
        "broken_pcff_specific_claims": broken_claims,
        "exact_ambiguous_or_wrong_terms": base_assessment.get("exact_ambiguous_or_wrong_terms", []),
        "gate_e_blocked": status != "PASS",
        "total_force_max_abs_component_delta": base_assessment.get("total_force_max_abs_component_delta"),
        "per_level_force_max_abs_component_delta": base_assessment.get("per_level_force_max_abs_component_delta"),
        "explicit_term_energy_display_assessment": base_assessment.get("explicit_term_energy_display_assessment"),
        "virial_display_assessment": base_assessment.get("virial_display_assessment"),
        "virial_characterization": base_assessment.get("virial_characterization"),
        "reciprocal_force_comparison": reciprocal_force_comparison,
        "reciprocal_force_characterization": reciprocal_force_characterization,
        "fft_backend_arithmetic_evidence": fft_backend_arithmetic_evidence,
    }


def find_first_failure_gate_d(
    *,
    main_run: dict[str, object],
    event_order_comparison: dict[str, object],
    total_force_comparison: dict[str, object],
    per_level_force_comparison: dict[str, object],
    per_term_rows: list[dict[str, object]],
    reciprocal_force_comparison: dict[str, object],
    reciprocal_force_noise_floor: dict[str, object],
    reciprocal_force_roundoff_proxy: dict[str, object],
    fft_backend_arithmetic_evidence: dict[str, object],
) -> dict[str, object] | None:
    if main_run["returncode"] != 0:
        return {
            "field": "main_run.returncode",
            "reason": "The Gate D mdrun command did not execute successfully.",
            "returncode": main_run["returncode"],
            "failure_markers": main_run["failure_markers"],
        }
    if not event_order_comparison["matches"]:
        return {"field": "event_order", "details": event_order_comparison["first_mismatch"]}
    if total_force_comparison["missing_in_actual"] or total_force_comparison["extra_in_actual"]:
        return {"field": "total_force", "details": total_force_comparison["first_mismatch"]}
    if per_level_force_comparison["missing_in_actual"] or per_level_force_comparison["extra_in_actual"]:
        return {"field": "per_level_force_totals", "details": per_level_force_comparison["first_mismatch"]}
    if reciprocal_force_comparison["missing_in_actual"] or reciprocal_force_comparison["extra_in_actual"]:
        return {"field": "reciprocal_force_trace", "details": reciprocal_force_comparison["first_mismatch"]}
    if reciprocal_force_comparison["first_mismatch"] is not None:
        return {"field": "reciprocal_force_trace", "details": reciprocal_force_comparison["first_mismatch"]}
    reciprocal_force_characterization = characterize_reciprocal_force_difference(
        reciprocal_force_comparison,
        reciprocal_force_noise_floor,
        reciprocal_force_roundoff_proxy,
        fft_backend_arithmetic_evidence,
    )
    if reciprocal_force_characterization["status"] not in RECIPROCAL_FORCE_PASS_STATUSES:
        return {"field": "reciprocal_force_trace.magnitude", "details": reciprocal_force_characterization}
    term_issue = first_term_issue(per_term_rows)
    if term_issue is not None:
        return {"field": f"per_term.{term_issue['bucket']}", "details": term_issue}
    return None


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
    logs_dir = system_root / "logs"
    summaries_dir = system_root / "summaries"
    for directory in (logs_dir, summaries_dir):
        directory.mkdir(parents=True, exist_ok=True)

    commands: list[dict[str, object]] = []
    gate_a_tpr = Path(gate_a_system["full_run_outputs"]["tpr"])
    tpr_dump = capture_output([str(gmx), "dump", "-s", str(gate_a_tpr)], cwd=REPO_ROOT)
    write_text(summaries_dir / "gate_a_tpr_dump.txt", tpr_dump)
    topology_inventory = parse_tpr_inventory(tpr_dump)
    write_text(
        summaries_dir / "topology_inventory.json",
        json.dumps(topology_inventory, indent=2, sort_keys=True) + "\n",
    )

    gate_a_energy_summary = load_json(Path(gate_a_system["energy_terms"]))
    gate_a_class2_trace = load_json(Path(gate_a_system["class2_subterm_energy_trace"]))
    gate_a_cpu_correction_trace = load_json(Path(gate_a_system["cpu_correction_energy_trace"]))
    gate_a_term_coverage = build_gate_a_term_coverage(
        gate_a_energy_summary, topology_inventory, gate_a_class2_trace, gate_a_cpu_correction_trace
    )
    write_text(
        summaries_dir / "gate_a_term_coverage.json",
        json.dumps(gate_a_term_coverage, indent=2, sort_keys=True) + "\n",
    )

    expected_events = parse_event_trace(Path(gate_a_system["full_run_outputs"]["event_trace_tsv"]))
    expected_total_force = load_json(Path(gate_a_system["total_force_summary"]))["per_step_totals"]
    expected_per_level_force = load_json(Path(gate_a_system["per_level_force_totals"]))["entries"]
    expected_energy_frames = load_json(Path(gate_a_system["energy_terms"]))["frames"]
    gate_a_force_component_trace = parse_force_component_trace(
        Path(gate_a_system["artifact_root"]) / "full" / "m2p_trace" / "step0_force_component_trace.txt"
    )

    main_run = collect_gpu_run(
        args=args,
        gmx=gmx,
        gate_a_tpr=gate_a_tpr,
        system_root=system_root,
        run_label="full",
        trace_atom_count=read_gro_atom_count(Path(gate_a_system["full_run_outputs"]["gro"])),
        commands=commands,
    )
    repeat_runs: list[dict[str, object]] = []
    successful_runs: list[dict[str, object]] = []
    if main_run["returncode"] == 0:
        successful_runs.append(main_run)
        for repeat_index in range(1, args.gpu_repeats):
            repeat_run = collect_gpu_run(
                args=args,
                gmx=gmx,
                gate_a_tpr=gate_a_tpr,
                system_root=system_root,
                run_label=f"repeat_{repeat_index}",
                trace_atom_count=read_gro_atom_count(Path(gate_a_system["full_run_outputs"]["gro"])),
                commands=commands,
            )
            repeat_runs.append(repeat_run)
            if repeat_run["returncode"] == 0:
                successful_runs.append(repeat_run)

    if main_run["returncode"] == 0:
        event_order_comparison = compare_event_trace(main_run["actual_events"], expected_events)
        total_force_comparison = compare_total_force_entries(
            main_run["total_force_summary"]["per_step_totals"], expected_total_force
        )
        per_level_force_comparison = compare_per_level_force_entries(
            main_run["per_level_force_totals"]["entries"], expected_per_level_force
        )
        energy_comparison = compare_energy_frames(main_run["energy_frames"], expected_energy_frames)
        virial_comparison = extract_virial_deltas(energy_comparison)
        gpu_noise_floor = estimate_noise_floor(successful_runs)
        reciprocal_force_comparison = compare_force_component_rows(
            main_run["force_component_trace"], gate_a_force_component_trace, "coulomb_recip_force"
        )
        reciprocal_force_noise_floor = estimate_force_component_noise_floor(
            successful_runs, "coulomb_recip_force"
        )
        reciprocal_force_roundoff_proxy = estimate_force_component_roundoff_proxy(
            main_run["force_component_trace"], gate_a_force_component_trace, "coulomb_recip_force"
        )
        fft_backend_arithmetic_evidence = load_fft_backend_arithmetic_evidence(
            DEFAULT_GATE_D_DEBUG_ROOT,
            system_id,
            reciprocal_force_comparison.get("max_abs_component_delta"),
        )
        class2_trace_bucket_assessments = build_class2_trace_bucket_assessments(
            main_run["class2_subterm_energy_trace"], gate_a_class2_trace
        )
        cpu_correction_trace_bucket_assessments = build_cpu_correction_trace_bucket_assessments(
            main_run["cpu_correction_energy_trace"], gate_a_cpu_correction_trace
        )
        per_term_rows = build_per_term_comparison_rows(
            gate_a_term_coverage,
            energy_comparison,
            main_run["energy_frames"],
            expected_energy_frames,
            class2_trace_bucket_assessments,
            cpu_correction_trace_bucket_assessments,
        )
    else:
        blocked_reason = (
            "Gate D mdrun did not execute; direct event/force/energy/noise comparisons against Gate A are unavailable."
        )
        event_order_comparison = {"matches": False, "reason": blocked_reason, "first_mismatch": None}
        total_force_comparison = {
            "matches": False,
            "reason": blocked_reason,
            "missing_in_actual": [],
            "extra_in_actual": [],
            "max_abs_component_delta": None,
            "first_mismatch": None,
        }
        per_level_force_comparison = {
            "matches": False,
            "reason": blocked_reason,
            "missing_in_actual": [],
            "extra_in_actual": [],
            "max_abs_component_delta": None,
            "first_mismatch": None,
        }
        energy_comparison = {
            "matches": False,
            "reason": blocked_reason,
            "max_abs_delta_kj_mol": None,
            "first_mismatch": None,
            "frames": [],
        }
        virial_comparison = {"available": False, "reason": blocked_reason, "max_abs_delta": None, "frames": []}
        gpu_noise_floor = {
            "available": False,
            "reason": "No successful Gate D GPU runs are available.",
            "successful_run_count": 0,
        }
        reciprocal_force_comparison = {
            "component_name": "coulomb_recip_force",
            "missing_in_actual": [],
            "extra_in_actual": [],
            "first_mismatch": None,
            "first_nonzero_delta": None,
            "max_abs_component_delta": None,
            "compared_row_count": 0,
            "rows": [],
        }
        reciprocal_force_noise_floor = {
            "available": False,
            "reason": "No successful Gate D GPU runs are available.",
            "successful_run_count": 0,
        }
        reciprocal_force_roundoff_proxy = {
            "available": False,
            "reason": "No successful Gate D GPU runs are available.",
            "inference": True,
        }
        fft_backend_arithmetic_evidence = {
            "available": False,
            "reason": "No successful Gate D GPU runs are available.",
            "inference": False,
        }
        class2_trace_bucket_assessments = {}
        cpu_correction_trace_bucket_assessments = {}
        per_term_rows = build_per_term_comparison_rows(
            gate_a_term_coverage,
            energy_comparison,
            [],
            expected_energy_frames,
            class2_trace_bucket_assessments,
            cpu_correction_trace_bucket_assessments,
        )

    reciprocal_ownership_comparison = build_reciprocal_ownership_summary(
        energy_comparison=energy_comparison,
        cpu_correction_trace_bucket_assessments=cpu_correction_trace_bucket_assessments,
        reciprocal_force_comparison=reciprocal_force_comparison,
        reciprocal_force_noise_floor=reciprocal_force_noise_floor,
        reciprocal_force_roundoff_proxy=reciprocal_force_roundoff_proxy,
        fft_backend_arithmetic_evidence=fft_backend_arithmetic_evidence,
        topology_inventory=topology_inventory,
    )
    direct_oracle_comparison = summarize_direct_oracle_comparison(
        event_order_comparison=event_order_comparison,
        total_force_comparison=total_force_comparison,
        per_level_force_comparison=per_level_force_comparison,
        energy_comparison=energy_comparison,
        virial_comparison=virial_comparison,
        gpu_noise_floor=gpu_noise_floor,
        per_term_rows=per_term_rows,
        class2_trace_bucket_assessments=class2_trace_bucket_assessments,
        cpu_correction_trace_bucket_assessments=cpu_correction_trace_bucket_assessments,
    )
    direct_oracle_comparison["reciprocal_force_trace"] = reciprocal_force_comparison
    direct_oracle_comparison["reciprocal_force_noise_floor"] = reciprocal_force_noise_floor
    direct_oracle_comparison["reciprocal_force_roundoff_proxy"] = reciprocal_force_roundoff_proxy
    direct_oracle_comparison["fft_backend_arithmetic_evidence"] = fft_backend_arithmetic_evidence

    assessment = assess_gate_d_system(
        main_run=main_run,
        gate_a_energy_frames=expected_energy_frames,
        event_order_comparison=event_order_comparison,
        total_force_comparison=total_force_comparison,
        per_level_force_comparison=per_level_force_comparison,
        energy_comparison=energy_comparison,
        virial_comparison=virial_comparison,
        gpu_noise_floor=gpu_noise_floor,
        per_term_rows=per_term_rows,
        reciprocal_force_comparison=reciprocal_force_comparison,
        reciprocal_force_noise_floor=reciprocal_force_noise_floor,
        reciprocal_force_roundoff_proxy=reciprocal_force_roundoff_proxy,
        fft_backend_arithmetic_evidence=fft_backend_arithmetic_evidence,
    )
    first_failure = find_first_failure_gate_d(
        main_run=main_run,
        event_order_comparison=event_order_comparison,
        total_force_comparison=total_force_comparison,
        per_level_force_comparison=per_level_force_comparison,
        per_term_rows=per_term_rows,
        reciprocal_force_comparison=reciprocal_force_comparison,
        reciprocal_force_noise_floor=reciprocal_force_noise_floor,
        reciprocal_force_roundoff_proxy=reciprocal_force_roundoff_proxy,
        fft_backend_arithmetic_evidence=fft_backend_arithmetic_evidence,
    )
    first_mismatching_term = first_term_issue(per_term_rows)

    system_result = {
        "system_id": system_id,
        "artifact_root": str(system_root),
        "gate_a_artifact_root": gate_a_system["artifact_root"],
        "gate_a_commands_sh": gate_a_system["commands_sh"],
        "gate_a_tpr": str(gate_a_tpr),
        "commands_json": str(summaries_dir / "commands.json"),
        "commands_sh": str(system_root / "run_commands.sh"),
        "main_run": main_run,
        "repeat_runs": repeat_runs,
        "topology_inventory": topology_inventory,
        "gate_a_term_coverage": gate_a_term_coverage,
        "event_order_comparison": event_order_comparison,
        "total_force_comparison": total_force_comparison,
        "per_level_force_comparison": per_level_force_comparison,
        "energy_comparison": energy_comparison,
        "virial_comparison": virial_comparison,
        "class2_trace_bucket_assessments": class2_trace_bucket_assessments,
        "cpu_correction_trace_bucket_assessments": cpu_correction_trace_bucket_assessments,
        "reciprocal_force_comparison": reciprocal_force_comparison,
        "reciprocal_force_noise_floor": reciprocal_force_noise_floor,
        "reciprocal_force_roundoff_proxy": reciprocal_force_roundoff_proxy,
        "fft_backend_arithmetic_evidence": fft_backend_arithmetic_evidence,
        "reciprocal_ownership_comparison": reciprocal_ownership_comparison,
        "per_term_comparison_table": per_term_rows,
        "direct_oracle_comparison": direct_oracle_comparison,
        "gpu_noise_floor": gpu_noise_floor,
        "gate_d_assessment": assessment,
        "first_failure_field": first_failure,
        "first_mismatching_term": first_mismatching_term,
    }
    write_text(summaries_dir / "system_result.json", json.dumps(system_result, indent=2, sort_keys=True) + "\n")
    write_text(summaries_dir / "commands.json", json.dumps(commands, indent=2, sort_keys=True) + "\n")
    write_commands_script(system_root / "run_commands.sh", commands)
    return system_result


def build_manifest(
    *,
    args: argparse.Namespace,
    out_root: Path,
    gate_a_manifest: dict[str, object],
    gate_b_manifest: dict[str, object] | None,
    gate_c_manifest: dict[str, object] | None,
    gmx: Path,
    gmx_version: str,
    gpu_inventory: dict[str, object],
    systems: list[dict[str, object]],
) -> dict[str, object]:
    status_rank = {"PASS": 0, "PARTIAL": 1, "FAIL": 2, "BLOCKER": 3}
    status = max((system["gate_d_assessment"]["status"] for system in systems), key=lambda item: status_rank[item])
    blocking_reasons = []
    for system in systems:
        if system["main_run"]["failure_markers"]:
            blocking_reasons.append(f"{system['system_id']}: {' | '.join(system['main_run']['failure_markers'])}")
        for reason in system["gate_d_assessment"]["reasons"]:
            blocking_reasons.append(f"{system['system_id']}: {reason}")
        if system["first_failure_field"] is not None:
            blocking_reasons.append(
                f"{system['system_id']}: first mismatch field is {system['first_failure_field']['field']}."
            )

    gate_e_allowed = all(system["gate_d_assessment"]["status"] == "PASS" for system in systems)
    recommendation_reason = (
        "Gate E may start because Gate D preserved reciprocal ownership, outer-level timing, and explicit per-level semantics."
        if gate_e_allowed
        else "Gate E remains blocked until Gate D resolves the first reciprocal or per-term mismatch for every fixture."
    )

    return {
        "schema_version": 1,
        "gate": "Gate D",
        "status": status,
        "objective": "Validate standalone exact r-RESPA with nb gpu + bonded gpu + pme gpu while update remains on CPU.",
        "artifact_root": str(out_root),
        "gate_a_manifest": str(Path(args.gate_a_manifest).resolve()),
        "gate_a_status": gate_a_manifest.get("status"),
        "gate_b_manifest": str(Path(args.gate_b_manifest).resolve()),
        "gate_b_status": None if gate_b_manifest is None else gate_b_manifest.get("status"),
        "gate_c_manifest": str(Path(args.gate_c_manifest).resolve()),
        "gate_c_status": None if gate_c_manifest is None else gate_c_manifest.get("status"),
        "gmx": str(gmx),
        "gmx_version": gmx_version,
        "precision_mode": parse_precision_mode(gmx_version),
        "gpu_support": parse_gpu_support(gmx_version),
        "hardware_configuration": gpu_inventory,
        "ntmpi": args.ntmpi,
        "ntomp": args.ntomp,
        "npme_flag_used": args.npme is not None,
        "npme_requested": args.npme,
        "npme_equals_one": args.npme == 1,
        "dlb": "no",
        "pme_rank_count": 0 if args.ntmpi == 1 else args.npme,
        "reproducibility_flags": [
            "-dlb no",
            "-pin off",
            "-nb gpu",
            "-pme gpu",
            "-bonded gpu",
            "-update cpu",
            "GMX_DISABLE_MODULAR_SIMULATOR=1",
        ],
        "binary_reproducibility_supported": False,
        "reproducibility_notes": [
            GPU_REPRODUCIBILITY_NOTE,
            "GPU noise floor is estimated from repeated successful Gate D runs for each fixture.",
        ],
        "rerun_used": False,
        "normal_md_used": True,
        "comparison_basis": (
            "Frozen Gate A CPU oracle is the numeric source of truth. Gate B/C manifests are recorded as upstream "
            "evidence in the GPU validation chain."
        ),
        "source_audit": {
            "pme_gpu_decision": "src/gromacs/taskassignment/decidegpuusage.cpp::canUseGpusForPme",
            "simulation_workload": "src/gromacs/taskassignment/decidesimulationworkload.cpp::createSimulationWorkload",
            "exact_respa_md_guard": "src/gromacs/mdrun/md.cpp::canUseExactLammpsRespaVelocityVerlet",
            "reciprocal_force_trace_source": "src/gromacs/mdlib/sim_util.cpp::longRangeNonbondeds.calculate_delta",
        },
        "blocking_reasons": blocking_reasons,
        "recommendation": {
            "gate_e_allowed": gate_e_allowed,
            "reason": recommendation_reason,
        },
        "systems": systems,
    }


def write_manifest_markdown(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Gate D Oracle Comparison",
        "",
        f"- Status: {manifest['status']}",
        f"- Gate E allowed: {manifest['recommendation']['gate_e_allowed']}",
        f"- gmx: `{manifest['gmx']}`",
        f"- precision: `{manifest['precision_mode']}`",
        f"- GPU support: `{manifest['gpu_support']}`",
        f"- ntmpi / ntomp: `{manifest['ntmpi']}` / `{manifest['ntomp']}`",
        f"- npme flag used: `{manifest['npme_flag_used']}`",
        f"- npme requested: `{manifest['npme_requested']}`",
        f"- DLB: `{manifest['dlb']}`",
        f"- PME ranks: `{manifest['pme_rank_count']}`",
        "",
        "## Blocking Reasons",
        "",
    ]
    if manifest["blocking_reasons"]:
        for reason in manifest["blocking_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- None")
    lines.extend(["", "## Systems", ""])
    for system in manifest["systems"]:
        lines.append(f"### {system['system_id']}")
        lines.append("")
        lines.append(f"- Gate D assessment: `{system['gate_d_assessment']['status']}`")
        lines.append(f"- Main run return code: `{system['main_run']['returncode']}`")
        lines.append(f"- Event order identical: `{system['event_order_comparison'].get('matches')}`")
        lines.append(
            f"- Reciprocal force max abs component delta: `{system['reciprocal_force_comparison'].get('max_abs_component_delta')}`"
        )
        lines.append(
            f"- Reciprocal force characterization: `{system['gate_d_assessment']['reciprocal_force_characterization'].get('status')}`"
        )
        lines.append(
            f"- Reciprocal force roundoff proxy bound: `{system['reciprocal_force_roundoff_proxy'].get('bound')}`"
        )
        lines.append(
            f"- FFT-backend arithmetic evidence available: `{system['fft_backend_arithmetic_evidence'].get('available')}`"
        )
        lines.append(
            f"- Coul. recip. max abs delta: `{system['reciprocal_ownership_comparison']['coulomb_reciprocal_delta']['max_abs_delta_kj_mol']}`"
        )
        lines.append(
            f"- CPU reciprocal/self/exclusion max abs delta: `{system['reciprocal_ownership_comparison']['cpu_correction_trace']['max_abs_delta_kj_mol']}`"
        )
        lines.append(f"- Layout report: `{system['main_run']['layout_report']['rank_layout']}`")
        lines.append(f"- First failure field: `{system['first_failure_field']}`")
        lines.append(f"- Artifact root: `{system['artifact_root']}`")
        lines.append(f"- Command script: `{system['commands_sh']}`")
        lines.append("")
    write_text(path, "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    gmx = Path(args.gmx).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    gate_a_manifest = load_json(Path(args.gate_a_manifest).resolve())
    gate_b_manifest = load_optional_manifest(Path(args.gate_b_manifest).resolve())
    gate_c_manifest = load_optional_manifest(Path(args.gate_c_manifest).resolve())
    validate_gate_chain(gate_a_manifest, gate_b_manifest, gate_c_manifest)
    maybe_build(args, Path(args.build_dir).resolve() if args.build_dir is not None else None)

    gmx_version = capture_output([str(gmx), "--version"], cwd=REPO_ROOT)
    gpu_inventory = {
        "nvidia_smi_list": capture_optional_output(["nvidia-smi", "-L"]),
        "nvidia_smi_query": capture_optional_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "nvcc_version": capture_optional_output(["nvcc", "--version"]),
    }

    gate_a_systems_by_id = {system["system_id"]: system for system in gate_a_manifest["systems"]}
    systems = []
    for system_id in SYSTEMS:
        systems.append(
            collect_system_result(
                args=args,
                gmx=gmx,
                out_root=out_root,
                system_id=system_id,
                gate_a_system=gate_a_systems_by_id[system_id],
            )
        )

    manifest = build_manifest(
        args=args,
        out_root=out_root,
        gate_a_manifest=gate_a_manifest,
        gate_b_manifest=gate_b_manifest,
        gate_c_manifest=gate_c_manifest,
        gmx=gmx,
        gmx_version=gmx_version,
        gpu_inventory=gpu_inventory,
        systems=systems,
    )
    write_text(out_root / "gate_d_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_manifest_markdown(out_root / "gate_d_manifest.md", manifest)


if __name__ == "__main__":
    main()
