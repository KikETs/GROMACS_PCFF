#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import DEFAULT_REFERENCE_ROOT, build_outputs, compare_outputs_to_reference, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or validate PT8 PCFF typing validation summaries.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Write machine-readable PT8 validation summaries")
    generate_parser.add_argument("--out", type=Path, default=DEFAULT_REFERENCE_ROOT)

    validate_parser = subparsers.add_parser("validate", help="Compare regenerated summaries against a reference directory")
    validate_parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_outputs()

    if args.command == "generate":
        write_outputs(args.out, outputs)
        result = {
            "status": "generated",
            "out_dir": str(args.out),
            "files": sorted(outputs),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    mismatches = compare_outputs_to_reference(args.reference, outputs)
    result = {
        "status": "ok" if not mismatches else "mismatch",
        "reference_dir": str(args.reference),
        "files": sorted(outputs),
        "mismatches": mismatches,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
