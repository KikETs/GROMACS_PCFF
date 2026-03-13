from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
DEFAULT_BINARY = BUILD_DIR / "bin" / "mdrun-non-integrator-test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep outer-step counts and GROMACS pair14 ownership for PCFF r-RESPA drift diagnostics."
    )
    parser.add_argument(
        "--out",
        default="/tmp/pcff_respa_drift_scan",
        help="Output directory for generated references, actual summaries, and aggregate scan JSON.",
    )
    parser.add_argument(
        "--outer-steps",
        type=int,
        action="append",
        dest="outer_steps",
        help="Outer-step count to scan. Repeat the option. Default: 1, 2, 5.",
    )
    parser.add_argument(
        "--pair14-level",
        type=int,
        action="append",
        dest="pair14_levels",
        help="GROMACS mts-respa-pair14-level to scan. Repeat the option. Default: 1, 2, 3.",
    )
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="System id to scan. Repeat to select multiple systems.",
    )
    parser.add_argument(
        "--build-target",
        default="mdrun-non-integrator-test",
        help="CMake target to build once before the scan.",
    )
    parser.add_argument(
        "--binary",
        default=str(DEFAULT_BINARY),
        help="Test binary used by the GROMACS side of the harness.",
    )
    parser.add_argument(
        "--lammps-cmd",
        default="/home/user/.local/bin/lmp",
        help="LAMMPS executable used for reference regeneration.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the initial CMake build step.",
    )
    parser.add_argument(
        "--nested-prototype",
        action="store_true",
        help="Enable the CPU nested exact-r-RESPA prototype during the GROMACS runs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def comparison_metric(system_result: dict, metric_name: str) -> dict:
    for item in system_result["comparisons"]:
        if item["name"] == metric_name:
            return item
    raise KeyError(f"Missing comparison metric {metric_name}")


def main() -> None:
    args = parse_args()
    out_root = Path(args.out).resolve()
    outer_steps_values = args.outer_steps or [1, 2, 5]
    pair14_levels = args.pair14_levels or [1, 2, 3]

    if not args.skip_build:
        subprocess.run(
            ["cmake", "--build", str(BUILD_DIR), "--target", args.build_target],
            cwd=REPO_ROOT,
            check=True,
        )

    aggregate_runs = []

    for outer_steps in outer_steps_values:
        reference_root = out_root / f"outer_steps_{outer_steps}" / "reference"
        prepare_cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "pcff_respa_parity" / "prepare_reference.py"),
            "--out",
            str(reference_root),
            "--lammps-cmd",
            args.lammps_cmd,
            "--outer-steps",
            str(outer_steps),
        ]
        for system_id in args.systems or []:
            prepare_cmd.extend(["--system", system_id])
        subprocess.run(prepare_cmd, cwd=REPO_ROOT, check=True)

        for pair14_level in pair14_levels:
            combo_root = out_root / f"outer_steps_{outer_steps}" / f"pair14_level_{pair14_level}"
            actual_root = combo_root / "actual"
            summary_root = combo_root / "compare"

            run_cmd = [
                sys.executable,
                str(REPO_ROOT / "tools" / "pcff_respa_parity" / "run.py"),
                "--skip-build",
                "--binary",
                args.binary,
                "--reference-root",
                str(reference_root),
                "--actual-root",
                str(actual_root),
                "--summary-dir",
                str(summary_root),
                "--outer-steps",
                str(outer_steps),
                "--pair14-level",
                str(pair14_level),
            ]
            if args.nested_prototype:
                run_cmd.append("--nested-prototype")
            for system_id in args.systems or []:
                run_cmd.extend(["--system", system_id])
            subprocess.run(run_cmd, cwd=REPO_ROOT, check=True)

            comparison_summary = load_json(summary_root / "comparison_summary.json")
            run_summary = {
                "outer_steps": outer_steps,
                "pair14_level": pair14_level,
                "systems": [],
            }
            step0_total = 0.0
            drift_total = 0.0
            for system_result in comparison_summary["systems"]:
                step0_metric = comparison_metric(system_result, "nve:step0_potential_kcal_mol")
                drift_metric = comparison_metric(system_result, "nve:total_energy_drift_abs_kcal_mol")
                span_metric = comparison_metric(system_result, "nve:total_energy_span_kcal_mol")
                entry = {
                    "system_id": system_result["system_id"],
                    "step0_abs_delta_kcal_mol": step0_metric["abs_delta"],
                    "drift_abs_delta_kcal_mol": drift_metric["abs_delta"],
                    "span_abs_delta_kcal_mol": span_metric["abs_delta"],
                    "dominant_gap": "step0"
                    if step0_metric["abs_delta"] >= drift_metric["abs_delta"]
                    else "drift",
                }
                run_summary["systems"].append(entry)
                step0_total += step0_metric["abs_delta"]
                drift_total += drift_metric["abs_delta"]

            run_summary["aggregate"] = {
                "step0_abs_delta_sum_kcal_mol": step0_total,
                "drift_abs_delta_sum_kcal_mol": drift_total,
            }
            dump_json(combo_root / "scan_summary.json", run_summary)
            aggregate_runs.append(run_summary)

    best_by_outer = []
    for outer_steps in outer_steps_values:
        matches = [run for run in aggregate_runs if run["outer_steps"] == outer_steps]
        best = min(matches, key=lambda run: (run["aggregate"]["drift_abs_delta_sum_kcal_mol"],
                                             run["aggregate"]["step0_abs_delta_sum_kcal_mol"]))
        best_by_outer.append(
            {
                "outer_steps": outer_steps,
                "best_pair14_level": best["pair14_level"],
                **best["aggregate"],
            }
        )

    aggregate = {
        "schema_version": 1,
        "outer_steps": outer_steps_values,
        "pair14_levels": pair14_levels,
        "runs": aggregate_runs,
        "best_by_outer_steps": best_by_outer,
    }
    dump_json(out_root / "aggregate_scan.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
