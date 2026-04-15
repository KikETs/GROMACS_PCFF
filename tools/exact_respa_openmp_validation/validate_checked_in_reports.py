from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from aggregate_reports import summarize_reports_from_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_REPORT_DIR = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "exact_respa_openmp_validation"
    / "host_reports"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the checked-in exact OpenMP host-report inventory. Use --strict to "
            "require the bounded desktop/workstation CPU OpenMP claim gate."
        )
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail unless the strict bounded desktop/workstation CPU OpenMP claim gate passes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_paths = sorted(HOST_REPORT_DIR.glob("*.json"))
    if not report_paths:
        raise SystemExit("No checked-in host reports were found.")

    summary = summarize_reports_from_paths(report_paths, allow_missing_tsan=not args.strict)
    with tempfile.NamedTemporaryFile(
        prefix="exact-openmp-report-inventory-",
        suffix=".json",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    tmp_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nInventory summary written to {tmp_path}")

    if args.strict:
        if not summary["pass"]:
            raise SystemExit(1)
        return

    fatal_blockers = [
        blocker
        for blocker in summary["mechanics_blockers"]
        if not blocker.startswith("Missing required topology classes:")
    ]
    if fatal_blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
