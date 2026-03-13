from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
DEFAULT_BINARY = BUILD_DIR / "bin" / "mdrun-non-integrator-test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M5 PCFF short-MD parity workflow.")
    parser.add_argument(
        "--build-target",
        default="mdrun-non-integrator-test",
        help="CMake target to build before executing the parity tests.",
    )
    parser.add_argument(
        "--binary",
        default=str(DEFAULT_BINARY),
        help="Test binary to execute.",
    )
    parser.add_argument(
        "--gtest-filter",
        default="PcffShortMdParity*",
        help="GTest filter for the short-MD parity tests.",
    )
    parser.add_argument(
        "--summary-dir",
        default=str(REPO_ROOT / "tests" / "reference_results" / "m5" / "last_run"),
        help="Directory where per-case JSON summaries and the aggregate report are written.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the build step and run the binary directly.",
    )
    return parser.parse_args()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    for stale_json in summary_dir.glob("*.json"):
        stale_json.unlink()

    if not args.skip_build:
        subprocess.run(
            ["cmake", "--build", str(BUILD_DIR), "--target", args.build_target],
            cwd=REPO_ROOT,
            check=True,
        )

    env = os.environ.copy()
    env["GMX_PCFF_M5_SUMMARY_DIR"] = str(summary_dir)
    subprocess.run(
        [args.binary, f"--gtest_filter={args.gtest_filter}"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    cases = []
    for path in sorted(summary_dir.glob("*.json")):
        if path.name == "comparison_summary.json":
            continue
        with path.open("r", encoding="utf-8") as handle:
            cases.append(json.load(handle))
    if not cases:
        raise SystemExit(
            "No per-case M5 summaries were produced. Check the gtest filter and the "
            "GMX_PCFF_M5_SUMMARY_DIR workflow."
        )

    aggregate = {
        "schema_version": 1,
        "cases": cases,
        "totals": {
            "num_cases": len(cases),
            "num_pass": sum(1 for case in cases if case.get("status") == "pass"),
            "num_fail": sum(1 for case in cases if case.get("status") == "fail"),
            "num_harness_sensitive": sum(1 for case in cases if case.get("harness_notes")),
        },
        "failure_categories_present": sorted(
            {
                category
                for case in cases
                for category in case.get("observed_failure_categories", [])
            }
        ),
    }
    dump_json(summary_dir / "comparison_summary.json", aggregate)


if __name__ == "__main__":
    main()
