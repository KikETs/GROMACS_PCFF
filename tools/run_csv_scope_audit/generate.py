#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import (
    DEFAULT_REFERENCE_ROOT,
    MANIFEST_ROOT,
    build_audit_outputs,
    build_manifests,
    compare_outputs_to_reference,
    load_manifests,
    validate_manifests,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or validate CSV scope manifests and coverage audit results.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser("snapshot", help="Write snapshot manifests from simulation-trajectory-aggregate.csv")
    snapshot_parser.add_argument("--csv", type=Path, required=True)
    snapshot_parser.add_argument("--out-manifests", type=Path, default=MANIFEST_ROOT)

    audit_parser = subparsers.add_parser("audit", help="Write coverage audit outputs from checked-in manifests")
    audit_parser.add_argument("--manifest-root", type=Path, default=MANIFEST_ROOT)
    audit_parser.add_argument("--out", type=Path, default=DEFAULT_REFERENCE_ROOT)

    validate_parser = subparsers.add_parser("validate", help="Compare regenerated audit outputs against a reference directory")
    validate_parser.add_argument("--manifest-root", type=Path, default=MANIFEST_ROOT)
    validate_parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_ROOT)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "snapshot":
        manifests = build_manifests(args.csv)
        write_outputs(args.out_manifests, manifests)
        result = {
            "status": "generated",
            "manifest_root": str(args.out_manifests),
            "files": sorted(manifests),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    manifests = load_manifests(args.manifest_root)
    validate_manifests(
        manifests["simulation_trajectory_aggregate_snapshot.json"],
        manifests["simulation_trajectory_aggregate_unique_smiles.json"],
        manifests["simulation_trajectory_aggregate_row_map.json"],
    )
    outputs = build_audit_outputs(args.manifest_root)

    if args.command == "audit":
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
