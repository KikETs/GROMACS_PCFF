from __future__ import annotations

import argparse
from pathlib import Path

from .engine import assign_file, dumps_assignment_report, write_assignment_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign deterministic repository-local PCFF nonbonded metadata.")
    parser.add_argument("path", help="Input structure path")
    parser.add_argument("--input-format", default=None, help="Override input format")
    parser.add_argument("--source-id", default=None, help="Override source id")
    parser.add_argument("--rules", default=None, help="Nonbonded rules path")
    parser.add_argument("--out", default=None, help="Write JSON report to this path")
    args = parser.parse_args()

    report = assign_file(
        args.path,
        input_format=args.input_format,
        source_id=args.source_id,
        rules_path=args.rules,
    )
    if args.out is None:
        print(dumps_assignment_report(report), end="")
    else:
        write_assignment_report(Path(args.out), report)


if __name__ == "__main__":
    main()
