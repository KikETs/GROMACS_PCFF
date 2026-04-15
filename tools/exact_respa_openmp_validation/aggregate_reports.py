from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from collect_host_report import (
    REPORT_SCHEMA_VERSION,
    SUPPORTED_OPENMP_SYSTEM_IDS,
    SUPPORTED_PIN_MODES,
    derive_exact_openmp_support_scope,
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
            "and decide whether a bounded cross-host CPU OpenMP mechanics claim is defensible."
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
    host_local_observation_blockers: list[str] = []
    infra_blockers: list[str] = []
    recomputed_rules: list[dict[str, Any]] = []
    per_report_support_scopes: list[dict[str, Any]] = []
    host_local_observations: list[dict[str, Any]] = []
    classes_present = {report["host"]["topology_class"] for report in reports}
    missing_classes = sorted(REQUIRED_CLASSES - classes_present)
    if missing_classes:
        mechanics_blockers.append(
            "Missing required topology classes: " + ", ".join(missing_classes)
        )

    for report in reports:
        host_label = report["host"]["label"]
        support_scope = report.get("mechanics", {}).get("openmp_support_scope")
        if not isinstance(support_scope, dict):
            support_scope = derive_exact_openmp_support_scope(
                report["host"]["topology"],
                report["mechanics"].get("release_suite"),
                report["mechanics"].get("tsan_suite"),
            )
        per_report_support_scopes.append(support_scope)
        release_suite = support_scope.get("release_suite")
        release_ok = bool(release_suite and release_suite.get("ok"))
        if not release_ok:
            mechanics_blockers.append(f"{host_label}: release exact suite did not pass")
        tsan_suite = support_scope.get("tsan_suite")
        tsan_ok = bool(tsan_suite and tsan_suite.get("ok"))
        if not tsan_ok and not allow_missing_tsan:
            mechanics_blockers.append(f"{host_label}: TSAN exact suite did not pass")
        for blocker in support_scope.get("blockers", []):
            if allow_missing_tsan and blocker.startswith("tsan:"):
                continue
            mechanics_blockers.append(f"{host_label}: {blocker}")
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

    for report in reports:
        derived = report.get("derived_host_local_rule") or derive_host_local_rule(
            report["host"]["topology"],
            report.get("benchmark"),
            per_report_support_scopes[len(recomputed_rules)].get("release_suite"),
            per_report_support_scopes[len(recomputed_rules)].get("tsan_suite"),
        )
        recomputed_rules.append(derived)
        if derived.get("rule_ready_for_cross_host_aggregation"):
            candidate = derived["production_candidate"]
            host_local_observations.append(
                {
                    "host_label": report["host"]["label"],
                    "topology_class": report["host"]["topology_class"],
                    "basis": candidate["basis"],
                    "ceiling_threads_on_this_host": candidate["ceiling_threads_on_this_host"],
                    "best_shape": candidate["best_shape"],
                    "best_ntomp": candidate["best_ntomp"],
                    "best_ns_per_day": candidate["best_ns_per_day"],
                    "plateau_threshold_ns_per_day": candidate["plateau_threshold_ns_per_day"],
                }
            )
        else:
            host_local_observation_blockers.append(
                f"{report['host']['label']}: no host-local throughput observation is ready "
                f"({derived.get('reason', 'missing reason')})"
            )

    common_basis = None
    common_rule_text = None
    if host_local_observations:
        unique_bases = {observation["basis"] for observation in host_local_observations}
        if len(unique_bases) == 1:
            common_basis = next(iter(unique_bases))
            common_rule_text = (
                "Within one L3 or CCD-equivalent locality group, host-local throughput often "
                "plateaus before larger thread counts, but those benchmark observations do not "
                "expand the mechanically supported CPU OpenMP envelope."
            )
        else:
            host_local_observation_blockers.append(
                "Host-local throughput observations do not share one common locality basis: "
                + ", ".join(sorted(unique_bases))
            )

    mechanics_claim_allowed = not mechanics_blockers
    support_claim_allowed = mechanics_claim_allowed
    g1_infra_allowed = not any(
        "TSAN-backed race evidence" in blocker for blocker in infra_blockers
    )
    g4_infra_allowed = not any(
        "recurring CI/scheduled infra" in blocker for blocker in infra_blockers
    )
    supported_thread_sets = {
        tuple(scope.get("supported_thread_counts", []))
        for scope in per_report_support_scopes
        if scope.get("supported_thread_counts")
    }
    shared_supported_threads = list(next(iter(supported_thread_sets))) if len(supported_thread_sets) == 1 else None
    resolved_threads_note = (
        " In the current checked-in host inventory, those audited buckets resolve to `ntomp=2` and `ntomp=6` on every tested host."
        if shared_supported_threads == [2, 6]
        else ""
    )
    summary = {
        "schema_version": 3,
        "pass": support_claim_allowed and g1_infra_allowed and g4_infra_allowed,
        "support_claim_allowed": support_claim_allowed,
        "mechanics_claim_allowed": mechanics_claim_allowed,
        "production_rule_allowed": False,
        "thread_scaling_rule_allowed": False,
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
                    support_scope["release_suite"]
                    and support_scope["release_suite"]["ok"]
                ),
                "tsan_suite_ok": bool(
                    support_scope["tsan_suite"]
                    and support_scope["tsan_suite"]["ok"]
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
                "validated_pin_modes": [
                    pin_mode["cli_value"] for pin_mode in support_scope["supported_pin_modes"]
                ],
                "supported_thread_counts": support_scope["supported_thread_counts"],
                "correctness_only_scope_statement": support_scope["correctness_only_scope_statement"],
                "support_scope_ready": not any(
                    not (allow_missing_tsan and blocker.startswith("tsan:"))
                    for blocker in support_scope["blockers"]
                ),
                "host_local_benchmark_ready": bool(
                    derived["rule_ready_for_cross_host_aggregation"]
                ),
                "host_local_benchmark_ceiling_threads": (
                    derived["production_candidate"]["ceiling_threads_on_this_host"]
                    if derived["rule_ready_for_cross_host_aggregation"]
                    else None
                ),
                "host_local_benchmark_basis": (
                    derived["production_candidate"]["basis"]
                    if derived["rule_ready_for_cross_host_aggregation"]
                    else None
                ),
            }
            for report, derived, support_scope in zip(
                reports, recomputed_rules, per_report_support_scopes, strict=True
            )
        ],
        "blockers": mechanics_blockers + host_local_observation_blockers + infra_blockers,
        "mechanics_blockers": mechanics_blockers,
        "host_local_observation_blockers": host_local_observation_blockers,
        "infra_blockers": infra_blockers,
        "final_allowed_claim": (
            "For single-rank, CPU-only, standalone exact r-RESPA on the tested low-core-workstation, mid-core-hybrid-desktop, and numa-or-chiplet desktop/workstation hosts, exact CPU OpenMP support is limited to the audited ntomp>1 buckets `ntompSmall` and `ntompCeiling` under `-pin auto`, `-pin on`, and `-pin inherit`, backed by oracle parity and restart parity on `small_oligomer` and `small_salt_polymer_box` in both release and TSAN suites."
            + resolved_threads_note
            + " This is a discrete mechanics claim only: `ntomp=1` remains the oracle baseline, host-local throughput benchmarks do not broaden the envelope, intermediate or larger ntomp counts remain unsupported, and the claim does not imply MD production handoff, ensemble readiness, transport readiness, does not cover server CPUs, and does not imply MPI or GPU coexistence support."
            if support_claim_allowed and g1_infra_allowed and g4_infra_allowed
            else (
                "For single-rank, CPU-only, standalone exact r-RESPA on the tested desktop/workstation hosts, the audited discrete CPU OpenMP mechanics envelope is supported by checked-in parity/restart evidence, but multi-host TSAN-backed evidence or recurring automation metadata is incomplete."
                if support_claim_allowed
                else (
                    "Keep the CPU OpenMP claim host-local until the discrete affinity/restart parity blockers are closed."
                )
            )
        ),
        "scope_note": (
            "This summary freezes a bounded CPU OpenMP mechanics claim only. "
            "Host-local benchmark throughput observations are reported separately and do not create supported or correctness-only ntomp counts."
        ),
        "supported_envelope": (
            {
                "statement": (
                    "Supported CPU OpenMP mechanics are limited to the audited discrete ntomp>1 buckets "
                    "`ntompSmall` and `ntompCeiling` under `-pin auto`, `-pin on`, and `-pin inherit` on "
                    "the tested low-core-workstation, mid-core-hybrid-desktop, and numa-or-chiplet desktop/workstation hosts."
                    + resolved_threads_note
                ),
                "single_rank_only": True,
                "cpu_only": True,
                "tested_topology_classes": sorted(REQUIRED_CLASSES),
                "validated_systems": list(SUPPORTED_OPENMP_SYSTEM_IDS),
                "validated_pin_modes": [pin_mode["cli_value"] for pin_mode in SUPPORTED_PIN_MODES],
                "validated_probe_labels": ["ntompSmall", "ntompCeiling"],
                "shared_resolved_thread_counts": shared_supported_threads,
                "resolved_thread_counts_by_host": {
                    report["host"]["label"]: scope["supported_thread_counts"]
                    for report, scope in zip(reports, per_report_support_scopes, strict=True)
                },
            }
            if support_claim_allowed
            else None
        ),
        "correctness_only_envelope": {
            "status": "none",
            "statement": (
                "None. No checked-in parity/restart artifact extends CPU OpenMP support from the "
                "audited discrete ntomp buckets to intermediate counts, larger counts, or benchmark-only runs."
            ),
        },
        "unsupported_or_weak_shapes": [
            "ntomp=1 is the oracle baseline and is not counted as CPU OpenMP support.",
            "Intermediate ntomp>1 counts between the audited buckets are mechanically unvalidated.",
            "Counts above the audited ntomp ceiling are mechanically unvalidated.",
            "Host-local `-pin inherit` throughput scans are benchmark observations only and do not create supported or correctness-only ntomp counts.",
            "Server CPUs, untested topology classes, MPI, GPU coexistence, and non-audited affinity/runtime shapes remain unsupported or unproven.",
        ],
        "host_local_throughput_observations": {
            "scope_note": (
                "These host-local throughput observations come from benchmark-only `-pin inherit` scans. "
                "They are throughput notes only and do not broaden the supported CPU OpenMP mechanics envelope."
            ),
            "shared_basis": common_basis,
            "shared_rule_text": common_rule_text,
            "per_host": host_local_observations,
        },
        "desktop_mechanics_claim": (
            {
                "rule_text": (
                    "Across the tested desktop/workstation topology classes, single-rank CPU-only "
                    "standalone exact r-RESPA preserved exact ntomp>1 oracle parity and restart parity on "
                    "the audited `ntompSmall` and `ntompCeiling` buckets under `-pin auto`, `-pin on`, and "
                    "`-pin inherit`. This is a discrete mechanics claim only; it does not interpolate to "
                    "intermediate or larger ntomp counts, does not cover server CPUs, and does not imply "
                    "MD production handoff, ensemble readiness, transport readiness, MPI, or GPU coexistence support."
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
        "aggregated_rule": None,
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
        summary["support_claim_allowed"] = False
        summary["mechanics_claim_allowed"] = False
        summary["production_rule_allowed"] = False
        summary["g1_infra_allowed"] = False
        summary["g4_infra_allowed"] = False
        summary["blockers"] = identity_blockers + summary["blockers"]
        summary["mechanics_blockers"] = identity_blockers + summary["mechanics_blockers"]
        summary["desktop_mechanics_claim"] = None
        summary["supported_envelope"] = None
        summary["aggregated_rule"] = None
        summary["final_allowed_claim"] = "Keep the CPU OpenMP claim host-local until stale or non-canonical reports are regenerated"
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
