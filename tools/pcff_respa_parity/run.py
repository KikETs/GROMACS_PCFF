from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
DEFAULT_BINARY = BUILD_DIR / "bin" / "mdrun-non-integrator-test"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "tests" / "reference_results" / "m6_respa"
DEFAULT_ACTUAL_ROOT = DEFAULT_REFERENCE_ROOT / "last_run_actual"
DEFAULT_SUMMARY_DIR = DEFAULT_REFERENCE_ROOT / "last_run_compare"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LAMMPS run_style respa parity harness.")
    parser.add_argument(
        "--build-target",
        default="mdrun-non-integrator-test",
        help="CMake target to build before running the GROMACS side of the harness.",
    )
    parser.add_argument(
        "--binary",
        default=str(DEFAULT_BINARY),
        help="Test binary that dumps exact r-RESPA observables.",
    )
    parser.add_argument(
        "--gtest-filter",
        default="PcffRespaObservableDump/*",
        help="GTest filter for the exact r-RESPA observable dump tests.",
    )
    parser.add_argument(
        "--reference-root",
        default=str(DEFAULT_REFERENCE_ROOT),
        help="Directory containing or receiving frozen LAMMPS reference artifacts.",
    )
    parser.add_argument(
        "--actual-root",
        default=str(DEFAULT_ACTUAL_ROOT),
        help="Directory for actual GROMACS exact r-RESPA JSON summaries.",
    )
    parser.add_argument(
        "--summary-dir",
        default=str(DEFAULT_SUMMARY_DIR),
        help="Directory for comparison JSON outputs.",
    )
    parser.add_argument(
        "--prepare-reference",
        action="store_true",
        help="Regenerate the frozen LAMMPS run_style respa reference before comparison.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the build step and run the binary directly.",
    )
    parser.add_argument(
        "--lammps-cmd",
        default="/home/user/.local/bin/lmp",
        help="LAMMPS executable forwarded to prepare_reference.py when requested.",
    )
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="System id to process. Repeat to select multiple systems.",
    )
    parser.add_argument(
        "--outer-steps",
        type=int,
        default=5,
        help="Number of outer r-RESPA steps to run in both LAMMPS and GROMACS diagnostics.",
    )
    parser.add_argument(
        "--pair14-level",
        type=int,
        default=1,
        help="GROMACS mts-respa-pair14-level used for the actual run.",
    )
    parser.add_argument(
        "--nested-prototype",
        action="store_true",
        help="Enable the CPU nested exact-r-RESPA prototype via GMX_EXACT_RESPA_NESTED_PROTOTYPE=1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_root = Path(args.reference_root).resolve()
    actual_root = Path(args.actual_root).resolve()
    summary_dir = Path(args.summary_dir).resolve()

    if args.prepare_reference:
        command = [
            sys.executable,
            str(REPO_ROOT / "tools" / "pcff_respa_parity" / "prepare_reference.py"),
            "--out",
            str(reference_root),
            "--lammps-cmd",
            args.lammps_cmd,
            "--outer-steps",
            str(args.outer_steps),
        ]
        for system_id in args.systems or []:
            command.extend(["--system", system_id])
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    if not args.skip_build:
        subprocess.run(
            ["cmake", "--build", str(BUILD_DIR), "--target", args.build_target],
            cwd=REPO_ROOT,
            check=True,
        )

    if actual_root.exists():
        shutil.rmtree(actual_root)
    actual_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["GMX_PCFF_RESPA_ACTUAL_DIR"] = str(actual_root)
    env["GMX_PCFF_RESPA_OUTER_STEPS"] = str(args.outer_steps)
    env["GMX_PCFF_RESPA_PAIR14_LEVEL"] = str(args.pair14_level)
    if args.nested_prototype:
        env["GMX_EXACT_RESPA_NESTED_PROTOTYPE"] = "1"
    subprocess.run(
        [args.binary, f"--gtest_filter={args.gtest_filter}"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    compare_command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "pcff_respa_parity" / "compare.py"),
        "--reference-root",
        str(reference_root),
        "--actual-root",
        str(actual_root),
        "--out",
        str(summary_dir),
    ]
    for system_id in args.systems or []:
        compare_command.extend(["--system", system_id])
    subprocess.run(compare_command, cwd=REPO_ROOT, check=True)

    summary = json.loads((summary_dir / "comparison_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
