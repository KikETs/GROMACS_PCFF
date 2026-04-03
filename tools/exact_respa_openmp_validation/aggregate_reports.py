from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from collect_host_report import derive_host_local_rule


REQUIRED_CLASSES = {
    "low-core-workstation",
    "mid-core-server",
    "numa-or-chiplet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multiple host-local exact r-RESPA CPU OpenMP validation reports "
            "and decide whether a broader CPU support claim is defensible."
        )
    )
    parser.add_argument(
        "reports",
        nargs="+",
        help="Host report JSON paths emitted by collect_host_report.py.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--allow-missing-tsan",
        action="store_true",
        help="Do not fail aggregation when a host report lacks TSAN evidence.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_reports(reports: list[dict[str, Any]], allow_missing_tsan: bool) -> dict[str, Any]:
    blockers: list[str] = []
    classes_present = {report["host"]["topology_class"] for report in reports}
    missing_classes = sorted(REQUIRED_CLASSES - classes_present)
    if missing_classes:
        blockers.append(
            "Missing required topology classes: " + ", ".join(missing_classes)
        )

    for report in reports:
        host_label = report["host"]["label"]
        release_ok = bool(report["mechanics"]["release_suite"] and report["mechanics"]["release_suite"]["ok"])
        if not release_ok:
            blockers.append(f"{host_label}: release exact suite did not pass")
        tsan_suite = report["mechanics"]["tsan_suite"]
        tsan_ok = bool(tsan_suite and tsan_suite["ok"])
        if not tsan_ok and not allow_missing_tsan:
            blockers.append(f"{host_label}: TSAN exact suite did not pass")
        benchmark = report["benchmark"]
        if not benchmark or not benchmark.get("ok"):
            blockers.append(f"{host_label}: locality benchmark evidence is missing")

    derived_candidates = []
    for report in reports:
        derived = report.get("derived_host_local_rule", {})
        if not derived.get("rule_ready_for_cross_host_aggregation"):
            derived = derive_host_local_rule(
                report["host"]["topology"],
                report.get("benchmark"),
                report["mechanics"].get("release_suite"),
                report["mechanics"].get("tsan_suite"),
            )
        if not derived.get("rule_ready_for_cross_host_aggregation"):
            blockers.append(
                f"{report['host']['label']}: no cross-host-ready locality rule candidate "
                f"({derived.get('reason', 'missing reason')})"
            )
            continue
        derived_candidates.append(
            (
                report["host"]["label"],
                report["host"]["topology_class"],
                derived["production_candidate"]["basis"],
                derived["production_candidate"]["rule_text"],
            )
        )

    common_basis = None
    common_rule_text = None
    if derived_candidates:
        unique_bases = {candidate[2] for candidate in derived_candidates}
        if len(unique_bases) != 1:
            blockers.append(
                "No common production locality basis across reports: "
                + ", ".join(sorted(unique_bases))
            )
        else:
            common_basis = next(iter(unique_bases))
            common_rule_text = derived_candidates[0][3]

    pass_allowed = not blockers and common_basis is not None and common_rule_text is not None
    summary = {
        "schema_version": 1,
        "pass": pass_allowed,
        "reports": [
            {
                "host_label": report["host"]["label"],
                "topology_class": report["host"]["topology_class"],
                "model_name": report["host"]["topology"]["model_name"],
                "logical_cpus": report["host"]["topology"]["logical_cpus"],
                "numa_nodes": report["host"]["topology"]["numa_nodes"],
                "l3_group_count": len(report["host"]["topology"]["l3_groups"]),
                "release_suite_ok": bool(
                    report["mechanics"]["release_suite"]
                    and report["mechanics"]["release_suite"]["ok"]
                ),
                "tsan_suite_ok": bool(
                    report["mechanics"]["tsan_suite"]
                    and report["mechanics"]["tsan_suite"]["ok"]
                ),
                "rule_candidate_ready": bool(
                    report.get("derived_host_local_rule", {}).get("rule_ready_for_cross_host_aggregation")
                ),
            }
            for report in reports
        ],
        "blockers": blockers,
        "final_allowed_claim": (
            "Broader CPU OpenMP claim allowed"
            if pass_allowed
            else "Keep the claim host-local until the blockers are closed"
        ),
        "aggregated_rule": (
            {
                "production_rule": common_rule_text,
                "production_basis": common_basis,
                "correctness_only_rule": (
                    "Above the production locality ceiling but within mechanically validated "
                    "thread counts on each tested host"
                ),
                "unsupported_or_unproven_rule": (
                    "Untested topology classes, unvalidated affinity/runtime shapes, or "
                    "thread counts beyond measured host limits"
                ),
            }
            if pass_allowed
            else None
        ),
    }
    return summary


def main() -> None:
    args = parse_args()
    report_paths = [Path(path).resolve() for path in args.reports]
    reports = [load_report(path) for path in report_paths]
    summary = summarize_reports(reports, args.allow_missing_tsan)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not summary["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
