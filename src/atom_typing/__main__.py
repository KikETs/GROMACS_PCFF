from __future__ import annotations

import argparse
from pathlib import Path

from .engine import dumps_typing_report, type_file, write_typing_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign deterministic PT3 PCFF atom type families.")
    parser.add_argument("input_path")
    parser.add_argument("--format", dest="input_format", default=None)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--rules", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = type_file(
        args.input_path,
        input_format=args.input_format,
        source_id=args.source_id,
        rules_path=args.rules,
    )
    if args.out is None:
        print(dumps_typing_report(report), end="")
    else:
        write_typing_report(Path(args.out), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
