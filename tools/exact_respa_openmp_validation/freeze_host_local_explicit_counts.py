from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from collect_host_report import REPO_ROOT, cpu_model_slug


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "exact_respa_openmp_validation"
    / "report_set_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "exact_respa_openmp_validation"
    / "host_local_explicit_counts"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a compact checked-in host-local exact OpenMP explicit-count artifact from a "
            "raw run_explicit_ntomp_validation.py summary."
        )
    )
    parser.add_argument("--in", dest="input_path", required=True, help="Raw summary.json path.")
    parser.add_argument(
        "--out",
        default="",
        help="Compact checked-in JSON path. Defaults under tests/reference_results/exact_respa_openmp_validation/host_local_explicit_counts.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to report_set_manifest.json used to resolve host identity.",
    )
    parser.add_argument(
        "--profile-id",
        default="",
        help="Manifest profile id to use when cpu_model_slug alone is ambiguous.",
    )
    parser.add_argument(
        "--counts",
        default="",
        help=(
            "Optional comma-separated ntomp counts to freeze from the raw summary, "
            "for example '2,4,6,8,10,12'. Defaults to all counts present in the summary."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_profile(summary: dict[str, Any], manifest: dict[str, Any], requested_profile_id: str) -> tuple[str, dict[str, Any]]:
    profiles = manifest["profiles"]
    if requested_profile_id:
        if requested_profile_id not in profiles:
            raise KeyError(f"Unknown manifest profile id: {requested_profile_id}")
        return requested_profile_id, profiles[requested_profile_id]

    model_slug = cpu_model_slug(summary["topology"])
    matches = [
        (profile_id, profile)
        for profile_id, profile in profiles.items()
        if profile.get("cpu_model_slug") == model_slug
    ]
    if not matches:
        raise RuntimeError(f"No manifest profile matched cpu_model_slug={model_slug!r}")
    if len(matches) > 1:
        raise RuntimeError(
            f"cpu_model_slug={model_slug!r} matched multiple profiles; pass --profile-id explicitly."
        )
    return matches[0]


def parse_selected_counts(value: str) -> list[int]:
    if not value.strip():
        return []

    counts = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        parsed = int(stripped)
        if parsed <= 0:
            raise ValueError(f"ntomp counts must be positive integers, got {parsed}")
        counts.append(parsed)
    deduped = sorted(set(counts))
    if not deduped:
        raise ValueError("--counts did not contain any usable ntomp values")
    return deduped


def default_output_path(profile_id: str, summary: dict[str, Any]) -> Path:
    counts = sorted(int(value) for value in summary["requested_ntomp_counts"])
    contiguous = counts == list(range(counts[0], counts[-1] + 1))
    suffix = f"{counts[0]}_{counts[-1]}" if contiguous else "_".join(str(value) for value in counts)
    return DEFAULT_OUTPUT_DIR / f"{profile_id}_ntomp_{suffix}.json"


def normalize_pin_mode_labels(labels: list[str]) -> list[str]:
    normalized = []
    mapping = {
        "pinAuto": "auto",
        "pinOn": "on",
        "pinInherit": "inherit",
    }
    for label in labels:
        normalized.append(mapping.get(label, label))
    return normalized


def compact_count_result(result: dict[str, Any]) -> dict[str, Any]:
    if result["category"] == "oracle_baseline":
        baseline_summary = result["baseline_restart"]["summary"]
        return {
            "ntomp": 1,
            "category": "oracle_baseline",
            "ok": result["ok"],
            "support_claimable": False,
            "baseline_restart_ok": baseline_summary["ok"],
            "baseline_restart_cases": baseline_summary["required_cases"],
            "note": (
                "ntomp=1 is the oracle baseline only. It is checked in here for host-local completeness, "
                "not as OpenMP support evidence."
            ),
        }

    affinity_summary = result["affinity_bundle"]["summary"]
    restart_affinity = affinity_summary["suite_summaries"]["restart_affinity"]
    openmp_affinity = affinity_summary["suite_summaries"]["openmp_affinity"]
    openmp_parity_summary = result["openmp_parity"]["summary"]

    return {
        "ntomp": int(result["ntomp"]),
        "category": "host_local_explicit_count",
        "ok": result["ok"],
        "support_claimable": False,
        "host_local_mechanics_ok": result["ok"],
        "restart_affinity_ok": not restart_affinity["failed_required_cases"]
        and not restart_affinity["skipped_required_cases"]
        and not restart_affinity["missing_required_cases"],
        "openmp_affinity_ok": not openmp_affinity["failed_required_cases"]
        and not openmp_affinity["skipped_required_cases"]
        and not openmp_affinity["missing_required_cases"],
        "openmp_oracle_parity_ok": openmp_parity_summary["ok"],
        "validated_pin_modes": normalize_pin_mode_labels(affinity_summary["validated_pin_modes"]),
        "restart_affinity_required_cases": restart_affinity["required_cases"],
        "openmp_affinity_required_cases": openmp_affinity["required_cases"],
        "openmp_oracle_required_cases": openmp_parity_summary["required_cases"],
        "blockers": result["blockers"],
        "note": (
            "This count passed on one checked-in host only. It does not widen the cross-host bounded "
            "CPU OpenMP claim by itself."
        ),
    }


def filter_summary(summary: dict[str, Any], selected_counts: list[int]) -> dict[str, Any]:
    if not selected_counts:
        return summary

    available_counts = {int(result["ntomp"]) for result in summary["results"]}
    missing_counts = [count for count in selected_counts if count not in available_counts]
    if missing_counts:
        raise KeyError(f"Selected ntomp counts are missing from the raw summary: {missing_counts}")

    filtered_results = [
        result for result in summary["results"] if int(result["ntomp"]) in set(selected_counts)
    ]
    filtered_blockers = [
        blocker
        for blocker in summary["overall_blockers"]
        if not any(f"ntomp={count}" in blocker for count in available_counts - set(selected_counts))
    ]
    return {
        **summary,
        "requested_ntomp_counts": selected_counts,
        "results": filtered_results,
        "overall_blockers": filtered_blockers,
        "overall_ok": not filtered_blockers,
    }


def build_compact_report(summary: dict[str, Any], profile_id: str, profile: dict[str, Any]) -> dict[str, Any]:
    compact_counts = [compact_count_result(result) for result in summary["results"]]
    explicit_counts = [item for item in compact_counts if item["category"] == "host_local_explicit_count"]
    passing_explicit_counts = [
        item["ntomp"] for item in explicit_counts if item["host_local_mechanics_ok"]
    ]

    return {
        "schema_name": "exact_respa_openmp_host_local_explicit_counts",
        "schema_version": 1,
        "host_identity": {
            "profile_id": profile_id,
            "host_label": profile["host_label"],
            "topology_class": profile["topology_class"],
            "cpu_model_slug": profile["cpu_model_slug"],
            "model_name": summary["topology"]["model_name"],
            "logical_cpus": summary["topology"]["logical_cpus"],
            "affinity_visible_cpus": summary["results"][0]["affinity_visible_cpus"],
        },
        "collection": {
            "started_at": summary["started_at"],
            "finished_at": summary["finished_at"],
            "binary": summary["binary"],
            "requested_ntomp_counts": summary["requested_ntomp_counts"],
            "overall_ok": summary["overall_ok"],
            "overall_blockers": summary["overall_blockers"],
        },
        "claim_scope_statement": (
            "This checked-in artifact freezes host-local explicit ntomp evidence on one audited host. "
            "It is not a cross-host CPU OpenMP support claim and it does not broaden the bounded "
            "desktop/workstation claim frozen in cpu_exact_respa_claim."
        ),
        "non_claimable_statement": (
            "Do not use this file to claim a continuous ntomp envelope, cross-host generality, server-CPU support, "
            "MPI support, GPU coexistence support, or any broader CPU OpenMP wording."
        ),
        "passing_explicit_ntomp_counts_on_this_host": passing_explicit_counts,
        "counts": compact_counts,
    }


def main() -> None:
    args = parse_args()
    summary = load_json(Path(args.input_path).resolve())
    selected_counts = parse_selected_counts(args.counts)
    summary = filter_summary(summary, selected_counts)
    manifest = load_json(Path(args.manifest).resolve())
    profile_id, profile = resolve_profile(summary, manifest, args.profile_id)
    output_path = Path(args.out).resolve() if args.out else default_output_path(profile_id, summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compact_report = build_compact_report(summary, profile_id, profile)
    output_path.write_text(json.dumps(compact_report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(compact_report, indent=2, sort_keys=True))
    print(f"\nWrote compact checked-in report to {output_path}")


if __name__ == "__main__":
    main()
