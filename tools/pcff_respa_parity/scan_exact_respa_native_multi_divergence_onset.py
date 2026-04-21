#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_exact_respa_native_multi_runtime as runtime_probe


DEFAULT_GMX = REPO_ROOT / "build" / "bin" / "gmx"
DEFAULT_STEPS = (1, 4, 16, 64, 256, 1024, 2000)
FORCE_THRESHOLDS = (1.0e-3, 1.0e-2, 1.0e-1, 1.0)
ENERGY_THRESHOLDS = (1.0e-3, 1.0e-2, 1.0e-1, 1.0)
GRO_THRESHOLDS = (1.0e-4, 5.0e-4, 1.0e-3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan exact-r-RESPA native-multi trajectory divergence onset by rerunning the same "
            "TPR from step 0 at increasing lengths and comparing per-launch vs native-multi."
        )
    )
    parser.add_argument("--gmx", type=Path, default=DEFAULT_GMX)
    parser.add_argument("--tpr", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step-list", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--ntomp", type=int, default=6)
    parser.add_argument("--pin", choices=("off", "on", "auto"), default="off")
    parser.add_argument("--fixture-id", default="unspecified")
    return parser.parse_args()


def first_step_above(records: list[dict[str, object]], field: str, threshold: float) -> int | None:
    for record in records:
        if float(record[field]) > threshold:
            return int(record["steps"])
    return None


def run_scan(args: argparse.Namespace) -> dict[str, object]:
    records: list[dict[str, object]] = []
    outer_step_factor = runtime_probe.slowest_step_factor_from_tpr(args.gmx, args.tpr)
    for requested_steps in args.step_list:
        steps = max(int(requested_steps), outer_step_factor)
        while steps % outer_step_factor != 0:
            steps += 1
        run_args = argparse.Namespace(
            gmx=args.gmx,
            tpr=args.tpr,
            output_dir=args.output_dir / f"steps_{steps:05d}",
            steps=steps,
            ntomp=args.ntomp,
            pin=args.pin,
            fixture_id=f"{args.fixture_id}_steps_{steps}",
            mdp=None,
            topol=None,
            probe_steps=1,
        )
        baseline = runtime_probe.run_mode(run_args, "per_launch", "0")
        candidate = runtime_probe.run_mode(run_args, "native_multi", "1")
        total_force = runtime_probe.compare_vector_frames(
            candidate["total_force_frames"], baseline["total_force_frames"]
        )
        per_level_force = runtime_probe.compare_vector_frames(
            candidate["per_level_force_frames"], baseline["per_level_force_frames"]
        )
        energy = runtime_probe.compare_energy_frames(
            candidate["energy_frames"], baseline["energy_frames"], runtime_probe.ENERGY_TERMS
        )
        gro = runtime_probe.compare_gro(Path(candidate["gro"]), Path(baseline["gro"]))

        records.append(
            {
                "requested_steps": int(requested_steps),
                "steps": steps,
                "disable_simd_kernels_env": candidate["disable_simd_kernels_env"],
                "disable_simd_kernels_marker_seen": bool(candidate["disable_simd_kernels_marker_seen"])
                and bool(baseline["disable_simd_kernels_marker_seen"]),
                "baseline_ns_per_day": baseline["metrics"]["ns_per_day"],
                "candidate_ns_per_day": candidate["metrics"]["ns_per_day"],
                "speedup": (
                    float(candidate["metrics"]["ns_per_day"]) / float(baseline["metrics"]["ns_per_day"])
                    if baseline["metrics"]["ns_per_day"] and candidate["metrics"]["ns_per_day"]
                    else None
                ),
                "total_force_max_abs_component_delta": total_force["max_abs_component_delta"],
                "per_level_force_max_abs_component_delta": per_level_force["max_abs_component_delta"],
                "energy_max_abs_delta": energy["max_abs_delta"],
                "gro_max_abs_coord_delta_nm": gro["max_abs_coord_delta_nm"],
                "gro_sha256_equal": baseline["gro_sha256"] == candidate["gro_sha256"],
                "edr_sha256_equal": baseline["edr_sha256"] == candidate["edr_sha256"],
                "cpt_sha256_equal": baseline["cpt_sha256"] == candidate["cpt_sha256"],
                "first_total_force_mismatch": total_force["first_mismatch"],
                "first_per_level_force_mismatch": per_level_force["first_mismatch"],
                "first_energy_mismatch": energy["first_mismatch"],
                "first_gro_mismatch": gro["first_mismatch"],
            }
        )

    return {
        "schema_name": "exact_respa_native_multi_divergence_onset",
        "schema_version": 1,
        "fixture_id": args.fixture_id,
        "tpr": str(args.tpr),
        "ntmpi": 1,
        "ntomp": args.ntomp,
        "pin": args.pin,
        "outer_step_factor": outer_step_factor,
        "disable_simd_kernels_env": os.environ.get("GMX_DISABLE_SIMD_KERNELS", "0"),
        "exact_respa_nbnxm_serial_reduction_env": os.environ.get(
            "GMX_PCFF_EXACT_RESPA_NBNXM_SERIAL_REDUCTION", "0"
        ),
        "step_list": [int(step) for step in args.step_list],
        "records": records,
        "threshold_summary": {
            "force_max_abs_component_delta": {
                str(threshold): first_step_above(records, "total_force_max_abs_component_delta", threshold)
                for threshold in FORCE_THRESHOLDS
            },
            "energy_max_abs_delta": {
                str(threshold): first_step_above(records, "energy_max_abs_delta", threshold)
                for threshold in ENERGY_THRESHOLDS
            },
            "gro_max_abs_coord_delta_nm": {
                str(threshold): first_step_above(records, "gro_max_abs_coord_delta_nm", threshold)
                for threshold in GRO_THRESHOLDS
            },
            "first_gro_hash_mismatch_step": first_step_above(
                [{"steps": record["steps"], "mismatch": 0.0 if record["gro_sha256_equal"] else 1.0} for record in records],
                "mismatch",
                0.5,
            ),
            "first_edr_hash_mismatch_step": first_step_above(
                [{"steps": record["steps"], "mismatch": 0.0 if record["edr_sha256_equal"] else 1.0} for record in records],
                "mismatch",
                0.5,
            ),
        },
        "notes": [
            "This scan restarts both modes from the same input TPR at step 0 for each listed step count.",
            "Threshold summaries are heuristics for divergence onset and should not be read as exactness pass/fail gates.",
            "Raw per-step records remain the primary evidence.",
        ],
    }


def main() -> None:
    args = parse_args()
    args.gmx = args.gmx.resolve()
    args.tpr = args.tpr.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = run_scan(args)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary_path = args.output_dir / "report.tsv"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "\t".join(
                (
                    "fixture_id",
                    "steps",
                    "baseline_ns_per_day",
                    "candidate_ns_per_day",
                    "speedup",
                    "total_force_max_abs_component_delta",
                    "per_level_force_max_abs_component_delta",
                    "energy_max_abs_delta",
                    "gro_max_abs_coord_delta_nm",
                    "gro_sha256_equal",
                    "edr_sha256_equal",
                    "cpt_sha256_equal",
                )
            )
            + "\n"
        )
        for record in report["records"]:
            handle.write(
                "\t".join(
                    (
                        args.fixture_id,
                        str(record["steps"]),
                        str(record["baseline_ns_per_day"]),
                        str(record["candidate_ns_per_day"]),
                        str(record["speedup"]),
                        str(record["total_force_max_abs_component_delta"]),
                        str(record["per_level_force_max_abs_component_delta"]),
                        str(record["energy_max_abs_delta"]),
                        str(record["gro_max_abs_coord_delta_nm"]),
                        str(record["gro_sha256_equal"]).lower(),
                        str(record["edr_sha256_equal"]).lower(),
                        str(record["cpt_sha256_equal"]).lower(),
                    )
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
