from __future__ import annotations

import argparse
from pathlib import Path

from .engine import dumps_report, run_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PT7 polymer electrolyte workflow and emit a mixed-system GROMACS topology.")
    parser.add_argument("spec", help="Polymer workflow spec JSON path")
    parser.add_argument("--out", default=None, help="Output directory for rendered GROMACS files")
    parser.add_argument("--dry-run", action="store_true", help="Render and validate without writing files")
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Compare the rendered files against an existing output directory",
    )
    args = parser.parse_args()

    report = run_file(
        args.spec,
        out_dir=None if args.out is None else Path(args.out),
        dry_run=args.dry_run,
        validate_existing=args.validate_existing,
    )
    print(dumps_report(report), end="")


if __name__ == "__main__":
    main()
