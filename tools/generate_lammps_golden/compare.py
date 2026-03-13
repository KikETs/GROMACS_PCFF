from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import OBSERVABLE_ORDER, enabled_observables, iter_system_records, system_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare normalized candidate outputs against a golden corpus.")
    parser.add_argument("--golden", required=True, help="Directory containing golden normalized outputs.")
    parser.add_argument("--candidate", required=True, help="Directory containing candidate normalized outputs.")
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="System id to compare. Repeat to select multiple systems. Default: all systems.",
    )
    parser.add_argument("--energy-abs-tol", type=float, default=0.0)
    parser.add_argument("--force-abs-tol", type=float, default=0.0)
    parser.add_argument("--trace-abs-tol", type=float, default=0.0)
    parser.add_argument("--report", help="Optional JSON report path.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def nearly_equal(lhs: float, rhs: float, abs_tol: float) -> bool:
    return math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=abs_tol)


def compare_mapping(golden: dict, candidate: dict, abs_tol: float, prefix: str) -> list[str]:
    failures = []
    golden_keys = sorted(golden.keys())
    candidate_keys = sorted(candidate.keys())
    if golden_keys != candidate_keys:
        failures.append(f"{prefix}: key mismatch {golden_keys} != {candidate_keys}")
        return failures
    for key in golden_keys:
        golden_value = golden[key]
        candidate_value = candidate[key]
        if isinstance(golden_value, (int, float)) and isinstance(candidate_value, (int, float)):
            if not nearly_equal(float(golden_value), float(candidate_value), abs_tol):
                failures.append(f"{prefix}.{key}: {candidate_value} != {golden_value} within {abs_tol}")
        elif golden_value != candidate_value:
            failures.append(f"{prefix}.{key}: {candidate_value!r} != {golden_value!r}")
    return failures


def compare_frames(golden: dict, candidate: dict, force_abs_tol: float) -> list[str]:
    failures = []
    if golden["fields"] != candidate["fields"]:
        failures.append(f"frame fields mismatch: {candidate['fields']} != {golden['fields']}")
        return failures
    if golden["timestep"] != candidate["timestep"]:
        failures.append(f"frame timestep mismatch: {candidate['timestep']} != {golden['timestep']}")
    if len(golden["atoms"]) != len(candidate["atoms"]):
        failures.append(f"atom count mismatch: {len(candidate['atoms'])} != {len(golden['atoms'])}")
        return failures
    for golden_atom, candidate_atom in zip(golden["atoms"], candidate["atoms"]):
        if golden_atom["id"] != candidate_atom["id"]:
            failures.append(f"atom id mismatch: {candidate_atom['id']} != {golden_atom['id']}")
            continue
        for field in golden["fields"]:
            if field == "id":
                continue
            tol = 0.0 if field in {"type"} else force_abs_tol
            if not nearly_equal(float(golden_atom[field]), float(candidate_atom[field]), tol):
                failures.append(
                    f"atom {golden_atom['id']} field {field}: {candidate_atom[field]} != {golden_atom[field]} within {tol}"
                )
    return failures


def compare_trace(golden_rows: list[dict], candidate_rows: list[dict], abs_tol: float, prefix: str) -> list[str]:
    failures = []
    if len(golden_rows) != len(candidate_rows):
        failures.append(f"{prefix}: row count mismatch {len(candidate_rows)} != {len(golden_rows)}")
        return failures
    for index, (golden_row, candidate_row) in enumerate(zip(golden_rows, candidate_rows)):
        failures.extend(compare_mapping(golden_row, candidate_row, abs_tol, f"{prefix}[{index}]"))
    return failures


def compare_system(golden_root: Path, candidate_root: Path, record: dict, tolerances: dict) -> list[str]:
    system_failures = []
    system_meta = system_metadata(record)
    observables = enabled_observables(system_meta)

    for observable in observables:
        normalized_name = system_meta["expected_observables"][observable]["normalized_output"]
        golden_path = golden_root / record["id"] / "normalized" / normalized_name
        candidate_path = candidate_root / record["id"] / "normalized" / normalized_name
        if not golden_path.exists():
            system_failures.append(f"{record['id']}:{observable}: missing golden file {golden_path}")
            continue
        if not candidate_path.exists():
            system_failures.append(f"{record['id']}:{observable}: missing candidate file {candidate_path}")
            continue

        golden_payload = load_json(golden_path)
        candidate_payload = load_json(candidate_path)

        if observable == "single_point":
            system_failures.extend(
                compare_mapping(
                    golden_payload["fields"],
                    candidate_payload["fields"],
                    tolerances["energy_abs_tol"],
                    f"{record['id']}:{observable}",
                )
            )
        elif observable == "forces":
            system_failures.extend(compare_frames(golden_payload["frame"], candidate_payload["frame"], tolerances["force_abs_tol"]))
        elif observable == "finite_difference":
            golden_checks = golden_payload["checks"]
            candidate_checks = candidate_payload["checks"]
            if len(golden_checks) != len(candidate_checks):
                system_failures.append(
                    f"{record['id']}:{observable}: check count mismatch {len(candidate_checks)} != {len(golden_checks)}"
                )
                continue
            for index, (golden_check, candidate_check) in enumerate(zip(golden_checks, candidate_checks)):
                system_failures.extend(
                    compare_mapping(
                        golden_check,
                        candidate_check,
                        tolerances["force_abs_tol"],
                        f"{record['id']}:{observable}[{index}]",
                    )
                )
        else:
            system_failures.extend(
                compare_trace(
                    golden_payload["trace"],
                    candidate_payload["trace"],
                    tolerances["trace_abs_tol"],
                    f"{record['id']}:{observable}:trace",
                )
            )
            system_failures.extend(
                compare_frames(
                    golden_payload["final_frame"],
                    candidate_payload["final_frame"],
                    tolerances["force_abs_tol"],
                )
            )

    return system_failures


def main() -> int:
    args = parse_args()
    golden_root = Path(args.golden).resolve()
    candidate_root = Path(args.candidate).resolve()
    tolerances = {
        "energy_abs_tol": args.energy_abs_tol,
        "force_abs_tol": args.force_abs_tol,
        "trace_abs_tol": args.trace_abs_tol,
    }

    failures = []
    records = iter_system_records(args.systems)
    for record in records:
        failures.extend(compare_system(golden_root, candidate_root, record, tolerances))

    report = {"passed": not failures, "failures": failures, "tolerances": tolerances}
    if args.report:
        with Path(args.report).open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if failures:
        for failure in failures:
            print(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
