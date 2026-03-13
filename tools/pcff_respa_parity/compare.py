from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "tests" / "reference_results" / "m6_respa"
DEFAULT_ACTUAL_ROOT = DEFAULT_REFERENCE_ROOT / "last_run_actual"
DEFAULT_OUT = DEFAULT_REFERENCE_ROOT / "last_run_compare"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GROMACS exact r-RESPA outputs against frozen LAMMPS golden data.")
    parser.add_argument(
        "--reference-root",
        default=str(DEFAULT_REFERENCE_ROOT),
        help="Directory containing frozen reference_summary.json files.",
    )
    parser.add_argument(
        "--actual-root",
        default=str(DEFAULT_ACTUAL_ROOT),
        help="Directory containing actual GROMACS summary JSON files.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output directory for machine-readable comparison summaries.",
    )
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="System id to compare. Default: all systems in the reference root.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_tolerances(reference_root: Path, system_id: str) -> dict[str, float]:
    tolerance_path = reference_root / system_id / "reference_summary.tsv"
    tolerances: dict[str, float] = {}
    if not tolerance_path.exists():
        return tolerances

    with tolerance_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) == 4 and fields[0] == "tolerance":
                ensemble, metric_name, value = fields[1], fields[2], fields[3]
                tolerances[f"{ensemble}:{metric_name}"] = float(value)
    return tolerances


def discover_systems(reference_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(path.name for path in reference_root.iterdir() if path.is_dir() and (path / "reference_summary.json").exists())


def compare_system(system_id: str, reference_root: Path, actual_root: Path) -> dict:
    reference = load_json(reference_root / system_id / "reference_summary.json")
    actual = load_json(actual_root / f"{system_id}_nve.json")
    tolerances = load_tolerances(reference_root, system_id)

    reference_values = reference["reference"]["nve"]
    actual_values = actual["observables"]["nve"]

    comparisons = []
    for key in sorted(reference_values):
        comparison_name = f"nve:{key}"
        actual_value = actual_values.get(key)
        present_in_actual = key in actual_values
        abs_delta = None if not present_in_actual else abs(actual_value - reference_values[key])
        tolerance = tolerances.get(comparison_name)
        comparisons.append(
            {
                "name": comparison_name,
                "reference": reference_values[key],
                "actual": actual_value,
                "delta": None if not present_in_actual else actual_value - reference_values[key],
                "abs_delta": abs_delta,
                "present_in_actual": present_in_actual,
                "tolerance": tolerance,
                "within_tolerance": None if tolerance is None or abs_delta is None else abs_delta <= tolerance,
            }
        )

    missing = [item["name"] for item in comparisons if not item["present_in_actual"]]
    failed = [
        item["name"]
        for item in comparisons
        if item["present_in_actual"] and item["tolerance"] is not None and item["within_tolerance"] is False
    ]
    has_tolerances = any(item["tolerance"] is not None for item in comparisons)
    status = "incomplete" if missing else "pass" if has_tolerances and not failed else "measured"
    return {
        "schema_version": 1,
        "system_id": system_id,
        "status": status,
        "schedule": reference["schedule"],
        "comparisons": comparisons,
        "notes": [
            "Frozen M6 3-level NVE tolerances are loaded from reference_summary.tsv and are intended to gate the exact CPU path before GPU work.",
            *reference.get("unresolved_items", []),
            *actual.get("notes", []),
        ],
        "missing_metrics": missing,
        "failed_metrics": failed,
    }


def main() -> None:
    args = parse_args()
    reference_root = Path(args.reference_root).resolve()
    actual_root = Path(args.actual_root).resolve()
    out_root = Path(args.out).resolve()

    results = []
    for system_id in discover_systems(reference_root, args.systems):
        result = compare_system(system_id, reference_root, actual_root)
        dump_json(out_root / f"{system_id}.json", result)
        results.append(result)

    aggregate = {
        "schema_version": 1,
        "systems": results,
        "totals": {
            "num_systems": len(results),
            "num_measured": sum(1 for result in results if result["status"] == "measured"),
            "num_pass": sum(1 for result in results if result["status"] == "pass"),
            "num_incomplete": sum(1 for result in results if result["status"] == "incomplete"),
        },
    }
    dump_json(out_root / "comparison_summary.json", aggregate)


if __name__ == "__main__":
    main()
