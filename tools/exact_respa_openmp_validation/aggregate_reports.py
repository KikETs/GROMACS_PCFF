from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from collect_host_report import (
    REPORT_SCHEMA_VERSION,
    canonical_report_filename,
    derive_host_local_rule,
)


REQUIRED_CLASSES = {
    "low-core-workstation",
    "mid-core-hybrid-desktop",
    "numa-or-chiplet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multiple host-local exact r-RESPA CPU OpenMP validation reports "
            "and decide whether a broader desktop-class CPU support claim is defensible."
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


def validate_report_identity(path: Path, report: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    schema_version = int(report.get("schema_version", 0))
    if schema_version < REPORT_SCHEMA_VERSION:
        blockers.append(
            f"{path.name}: legacy host report schema v{schema_version}; regenerate under schema v{REPORT_SCHEMA_VERSION}"
        )
        return blockers

    infra = report.get("infra")
    if not isinstance(infra, dict):
        blockers.append(f"{path.name}: missing infra metadata block")
        return blockers

    topology = report.get("host", {}).get("topology", {})
    topology_class = report.get("host", {}).get("topology_class")
    expected_filename = canonical_report_filename(
        topology,
        topology_class,
        infra.get("filename_host_suffix", ""),
    )
    if path.name != expected_filename:
        blockers.append(
            f"{path.name}: non-canonical host report filename; expected {expected_filename}"
        )

    if infra.get("report_filename") != path.name:
        blockers.append(
            f"{path.name}: stored report filename metadata does not match the tracked filename"
        )

    if not infra.get("report_filename_matches_canonical", False):
        blockers.append(f"{path.name}: report declares a non-canonical filename")

    collection_mode = infra.get("collection_mode")
    recurring_attestation = infra.get("recurring_backend_attestation", {})
    if collection_mode in {"ci", "scheduled"} and not recurring_attestation.get("attested", False):
        blockers.append(
            f"{path.name}: {collection_mode} report lacks recurring-backend attestation under schema v{REPORT_SCHEMA_VERSION}"
        )

    return blockers


def summarize_reports(reports: list[dict[str, Any]], allow_missing_tsan: bool) -> dict[str, Any]:
    mechanics_blockers: list[str] = []
    production_rule_blockers: list[str] = []
    infra_blockers: list[str] = []
    recomputed_rules: list[dict[str, Any]] = []
    classes_present = {report["host"]["topology_class"] for report in reports}
    missing_classes = sorted(REQUIRED_CLASSES - classes_present)
    if missing_classes:
        mechanics_blockers.append(
            "Missing required topology classes: " + ", ".join(missing_classes)
        )

    for report in reports:
        host_label = report["host"]["label"]
        release_ok = bool(report["mechanics"]["release_suite"] and report["mechanics"]["release_suite"]["ok"])
        if not release_ok:
            mechanics_blockers.append(f"{host_label}: release exact suite did not pass")
        tsan_suite = report["mechanics"]["tsan_suite"]
        tsan_ok = bool(tsan_suite and tsan_suite["ok"])
        if not tsan_ok and not allow_missing_tsan:
            mechanics_blockers.append(f"{host_label}: TSAN exact suite did not pass")
        benchmark = report["benchmark"]
        if not benchmark or not benchmark.get("ok"):
            mechanics_blockers.append(f"{host_label}: locality benchmark evidence is missing")
        infra = report.get("infra", {})
        tsan_status = infra.get("tsan", {}).get("status")
        if tsan_status != "backed":
            infra_blockers.append(
                f"{host_label}: multi-host TSAN-backed race evidence is not complete ({tsan_status or 'missing-status'})"
            )
        collection_mode = infra.get("collection_mode")
        if collection_mode not in {"ci", "scheduled"}:
            infra_blockers.append(
                f"{host_label}: report was collected in {collection_mode or 'unknown'} mode, not recurring CI/scheduled infra"
            )

    derived_candidates = []
    for report in reports:
        derived = derive_host_local_rule(
            report["host"]["topology"],
            report.get("benchmark"),
            report["mechanics"].get("release_suite"),
            report["mechanics"].get("tsan_suite"),
        )
        recomputed_rules.append(derived)
        if not derived.get("rule_ready_for_cross_host_aggregation"):
            production_rule_blockers.append(
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
            production_rule_blockers.append(
                "No common production locality basis across reports: "
                + ", ".join(sorted(unique_bases))
            )
        else:
            common_basis = next(iter(unique_bases))
            common_rule_text = derived_candidates[0][3]

    mechanics_claim_allowed = not mechanics_blockers
    production_rule_allowed = (
        mechanics_claim_allowed
        and not production_rule_blockers
        and common_basis is not None
        and common_rule_text is not None
    )
    g1_infra_allowed = not any(
        "TSAN-backed race evidence" in blocker for blocker in infra_blockers
    )
    g4_infra_allowed = not any(
        "recurring CI/scheduled infra" in blocker for blocker in infra_blockers
    )
    summary = {
        "schema_version": 2,
        "pass": production_rule_allowed and g1_infra_allowed and g4_infra_allowed,
        "mechanics_claim_allowed": mechanics_claim_allowed,
        "production_rule_allowed": production_rule_allowed,
        "thread_scaling_rule_allowed": production_rule_allowed,
        "scientific_md_production_handoff_implied": False,
        "g1_infra_allowed": g1_infra_allowed,
        "g4_infra_allowed": g4_infra_allowed,
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
                "tsan_status": report.get("infra", {}).get("tsan", {}).get("status"),
                "collection_mode": report.get("infra", {}).get("collection_mode"),
                "recurring_backend_attested": bool(
                    report.get("infra", {}).get("recurring_backend_attestation", {}).get("attested")
                ),
                "report_filename": report.get("infra", {}).get("report_filename"),
                "canonical_filename_ok": bool(
                    report.get("infra", {}).get("report_filename_matches_canonical")
                ),
                "rule_candidate_ready": bool(
                    derived["rule_ready_for_cross_host_aggregation"]
                ),
            }
            for report, derived in zip(reports, recomputed_rules, strict=True)
        ],
        "blockers": mechanics_blockers + production_rule_blockers + infra_blockers,
        "mechanics_blockers": mechanics_blockers,
        "production_rule_blockers": production_rule_blockers,
        "infra_blockers": infra_blockers,
        "final_allowed_claim": (
            "For single-rank, CPU-only, standalone exact r-RESPA on tested desktop/workstation CPUs, an affinity-enabled desktop-class exact CPU OpenMP mechanics claim is allowed and a shared one-L3 plateau-knee OpenMP thread-scaling envelope is allowed across the tested hosts. This is a host-local throughput statement only; it does not imply MD production handoff, ensemble readiness, transport readiness, does not cover server CPUs, and does not imply MPI or GPU coexistence support."
            if production_rule_allowed and g1_infra_allowed and g4_infra_allowed
            else (
                "For single-rank, CPU-only, standalone exact r-RESPA on tested desktop/workstation CPUs, an affinity-enabled desktop-class exact CPU OpenMP mechanics claim is allowed and a shared one-L3 plateau-knee OpenMP thread-scaling envelope is allowed across the tested hosts, but multi-host TSAN-backed race evidence or recurring automation infrastructure is still incomplete. This is a host-local throughput statement only; it does not imply MD production handoff, ensemble readiness, transport readiness, does not cover server CPUs, and does not imply MPI or GPU coexistence support."
                if production_rule_allowed
                else (
                    "For single-rank, CPU-only, standalone exact r-RESPA on tested desktop/workstation CPUs, an affinity-enabled desktop-class exact CPU OpenMP mechanics claim is allowed, but the OpenMP thread-scaling envelope remains host-local. This is not an MD production handoff or ensemble claim. It does not cover server CPUs and does not imply MPI or GPU coexistence support, and multi-host TSAN-backed race evidence is still incomplete."
                    if mechanics_claim_allowed
                    else "Keep the claim host-local until the blockers are closed"
                )
            )
        ),
        "scope_note": (
            "In this summary, the legacy `production_rule` name refers only to an OpenMP thread-scaling or throughput envelope derived from host-local benchmarks. "
            "It does not mean MD production handoff, ensemble convergence, or transport readiness."
        ),
        "desktop_mechanics_claim": (
            {
                "rule_text": (
                    "Across the tested desktop/workstation topology classes, single-rank CPU-only "
                    "standalone exact r-RESPA preserved exact affinity-enabled mechanical checks on every host. "
                    "This mechanics claim is separate from the shared OpenMP thread-scaling decision, "
                    "does not cover server CPUs, and does not imply MD production handoff, ensemble readiness, "
                    "transport readiness, MPI, or GPU coexistence support."
                ),
                "server_cpu_status": "unvalidated",
                "tsan_requirement_relaxed": bool(allow_missing_tsan),
            }
            if mechanics_claim_allowed
            else None
        ),
        "infrastructure_status": {
            "g1_infra_allowed": g1_infra_allowed,
            "g4_infra_allowed": g4_infra_allowed,
            "tsan_requirement_relaxed": bool(allow_missing_tsan),
            "recurring_automation_required": True,
        },
        "aggregated_rule": (
            {
                "production_rule": common_rule_text,
                "thread_scaling_rule": common_rule_text,
                "production_basis": common_basis,
                "scope_note": (
                    "This rule governs host-local OpenMP thread scaling only. "
                    "It is not a scientific MD production-readiness or transport-readiness statement."
                ),
                "correctness_only_rule": (
                    "Above the production locality ceiling but within mechanically validated "
                    "thread counts on each tested host"
                ),
                "unsupported_or_unproven_rule": (
                    "Server CPUs, untested topology classes, unvalidated affinity/runtime "
                    "shapes, or thread counts beyond measured host limits"
                ),
            }
            if production_rule_allowed
            else None
        ),
    }
    return summary


def summarize_reports_from_paths(report_paths: list[Path], allow_missing_tsan: bool) -> dict[str, Any]:
    reports = []
    identity_blockers: list[str] = []
    for path in report_paths:
        report = load_report(path)
        identity_blockers.extend(validate_report_identity(path, report))
        reports.append(report)

    summary = summarize_reports(reports, allow_missing_tsan)
    if identity_blockers:
        summary["pass"] = False
        summary["mechanics_claim_allowed"] = False
        summary["production_rule_allowed"] = False
        summary["g1_infra_allowed"] = False
        summary["g4_infra_allowed"] = False
        summary["blockers"] = identity_blockers + summary["blockers"]
        summary["mechanics_blockers"] = identity_blockers + summary["mechanics_blockers"]
        summary["desktop_mechanics_claim"] = None
        summary["aggregated_rule"] = None
        summary["final_allowed_claim"] = "Keep the claim host-local until stale or non-canonical reports are regenerated"
    return summary


def main() -> None:
    args = parse_args()
    report_paths = [Path(path).resolve() for path in args.reports]
    summary = summarize_reports_from_paths(report_paths, args.allow_missing_tsan)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not summary["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
