from __future__ import annotations

import argparse
from pathlib import Path

from .core import dumps_report, perceive_file, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute deterministic chemistry perception from PT1 inputs.")
    parser.add_argument("input_path")
    parser.add_argument("--format", dest="input_format", default=None)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = perceive_file(
        args.input_path,
        input_format=args.input_format,
        source_id=args.source_id,
    )
    if args.out is None:
        print(dumps_report(report), end="")
    else:
        write_report(Path(args.out), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
