from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from aggregate_reports import summarize_reports_from_paths
from collect_host_report import REPO_ROOT


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "exact_respa_openmp_validation"
    / "report_set_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the active/pending exact OpenMP host-report backend state without "
            "pretending that pending external hosts have already been recollected."
        )
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to report_set_manifest.json.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write validation summaries into.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Require the bounded desktop/workstation CPU OpenMP claim gate to pass. "
            "This fails on inventory drift, missing TSAN-backed evidence, missing recurring automation, "
            "or missing audited affinity/restart parity coverage for the supported ntomp buckets."
        ),
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_path(manifest: dict[str, Any], profile: dict[str, Any]) -> Path:
    return REPO_ROOT / manifest["host_report_dir"] / profile["report_filename"]


def known_report_filenames(manifest: dict[str, Any]) -> set[str]:
    return {
        profile["report_filename"]
        for profile in manifest["profiles"].values()
    }


def current_repo_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def ensure_commit_available(commit: str) -> None:
    if not commit:
        return
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if present.returncode == 0:
        return
    subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=64", "origin", commit],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def changed_paths_between(base_commit: str, head_commit: str) -> list[str]:
    if not base_commit or base_commit == head_commit:
        return []
    ensure_commit_available(base_commit)
    ensure_commit_available(head_commit)
    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{base_commit}..{head_commit}"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_commit = current_repo_commit()

    profiles = manifest["profiles"]
    active_ids = manifest.get("active_host_profiles", [])
    pending_ids = manifest.get("pending_host_profiles", [])
    host_report_dir = REPO_ROOT / manifest["host_report_dir"]
    stale_host_report_dir = REPO_ROOT / manifest["stale_host_report_dir"]
    allowed_report_refresh_paths = {
        str(Path(manifest["host_report_dir"]) / profiles[profile_id]["report_filename"])
        for profile_id in active_ids
    }

    active_paths: list[Path] = []
    active_inventory_errors: list[str] = []
    active_profiles: list[dict[str, Any]] = []
    for profile_id in active_ids:
        profile = profiles[profile_id]
        path = report_path(manifest, profile)
        active_profiles.append(
            {
                "profile_id": profile_id,
                "path": str(path),
                "exists": path.exists(),
                "tsan_required": bool(profile.get("tsan_required")),
                "collection_mode": profile.get("collection_mode"),
            }
        )
        if not path.exists():
            active_inventory_errors.append(
                f"{profile_id}: active host profile is missing its report file {path.name}"
            )
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        actual_commit = report.get("infra", {}).get("git_revision", {}).get("commit")
        changed_paths = changed_paths_between(actual_commit or "", expected_commit)
        is_report_refresh_only = (
            actual_commit != expected_commit
            and changed_paths
            and all(path in allowed_report_refresh_paths for path in changed_paths)
        )
        if actual_commit != expected_commit and not is_report_refresh_only:
            active_inventory_errors.append(
                f"{profile_id}: report was collected from commit {actual_commit or 'unknown'}, "
                f"expected current repo HEAD {expected_commit}"
            )
        active_paths.append(path)

    pending_profiles: list[dict[str, Any]] = []
    for profile_id in pending_ids:
        profile = profiles[profile_id]
        path = report_path(manifest, profile)
        pending_profiles.append(
            {
                "profile_id": profile_id,
                "path": str(path),
                "exists": path.exists(),
                "tsan_required": bool(profile.get("tsan_required")),
                "collection_mode": profile.get("collection_mode"),
            }
        )

    unknown_active_files = sorted(
        path.name
        for path in host_report_dir.glob("*.json")
        if path.name not in known_report_filenames(manifest)
    )
    stale_files = sorted(path.name for path in stale_host_report_dir.glob("*.json"))
    relaxed_aggregate_summary = (
        summarize_reports_from_paths(active_paths, allow_missing_tsan=True)
        if active_paths
        else {
            "pass": False,
            "mechanics_claim_allowed": False,
            "production_rule_allowed": False,
            "g1_infra_allowed": False,
            "g4_infra_allowed": False,
            "blockers": ["No active host reports are currently available."],
        }
    )
    strict_aggregate_summary = (
        summarize_reports_from_paths(active_paths, allow_missing_tsan=False)
        if active_paths
        else {
            "pass": False,
            "mechanics_claim_allowed": False,
            "production_rule_allowed": False,
            "g1_infra_allowed": False,
            "g4_infra_allowed": False,
            "blockers": ["No active host reports are currently available."],
        }
    )
    aggregate_summary = strict_aggregate_summary if args.strict else relaxed_aggregate_summary

    validation_summary = {
        "schema_version": 1,
        "report_semantics_version": manifest["report_semantics_version"],
        "expected_git_commit": expected_commit,
        "validation_mode": "strict" if args.strict else "relaxed",
        "active_host_profiles": active_profiles,
        "pending_host_profiles": pending_profiles,
        "active_inventory_errors": active_inventory_errors,
        "unknown_active_files": unknown_active_files,
        "stale_host_reports": stale_files,
        "aggregate_summary": aggregate_summary,
        "aggregate_allow_missing_tsan": relaxed_aggregate_summary,
        "aggregate_strict": strict_aggregate_summary,
        "backend_status": {
            "active_profile_count": len(active_ids),
            "pending_profile_count": len(pending_ids),
            "active_inventory_ok": not active_inventory_errors and not unknown_active_files,
            "g1_infra_blocked": not strict_aggregate_summary.get("g1_infra_allowed", False),
            "g4_infra_blocked": not strict_aggregate_summary.get("g4_infra_allowed", False),
            "strict_claim_pass": bool(strict_aggregate_summary.get("pass")),
        },
    }

    (out_dir / "aggregate-allow-missing-tsan.json").write_text(
        json.dumps(relaxed_aggregate_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "aggregate-strict.json").write_text(
        json.dumps(strict_aggregate_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "validation-summary.json").write_text(
        json.dumps(validation_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "stale-check.json").write_text(
        json.dumps(
            {
                "unknown_active_files": unknown_active_files,
                "stale_host_reports": stale_files,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(json.dumps(validation_summary, indent=2, sort_keys=True))

    if active_inventory_errors or unknown_active_files:
        raise SystemExit(1)
    if args.strict and not strict_aggregate_summary.get("pass", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
