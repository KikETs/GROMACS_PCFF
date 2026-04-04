from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from collect_host_report import REPO_ROOT, canonical_report_filename, cpu_model_slug, inspect_topology


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
            "Run one manifest-declared exact OpenMP host profile and emit a fresh "
            "schema-v2 host report into the active host_reports inventory."
        )
    )
    parser.add_argument("--profile-id", required=True, help="Manifest profile id to collect.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to report_set_manifest.json.",
    )
    parser.add_argument(
        "--allow-host-mismatch",
        action="store_true",
        help="Do not abort if the current CPU model slug does not match the manifest profile.",
    )
    parser.add_argument(
        "--collection-mode-override",
        choices=("manual", "manual-host", "ci", "scheduled"),
        help="Override the manifest collection mode for this run.",
    )
    parser.add_argument(
        "--expected-git-commit",
        default="",
        help="Require the collecting repo to already be at this commit before running the profile.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def current_repo_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def profile_output_path(manifest: dict[str, Any], profile: dict[str, Any]) -> Path:
    host_report_dir = REPO_ROOT / manifest["host_report_dir"]
    return host_report_dir / profile["report_filename"]


def resolve_collection_mode(
    profile: dict[str, Any],
    collection_mode_override: str | None,
) -> str:
    return collection_mode_override or profile.get("collection_mode", "manual-host")


def assert_recurring_mode_attestation(collection_mode: str) -> None:
    if collection_mode not in {"ci", "scheduled"}:
        return

    backend_attestation = os.getenv("EXACT_OPENMP_RECURRING_BACKEND")
    if backend_attestation == collection_mode:
        return

    if collection_mode == "ci" and os.getenv("GITHUB_ACTIONS") == "true":
        return

    raise SystemExit(
        "Recurring collection modes must be attested by the recurring backend. "
        f"Refusing collection_mode={collection_mode!r} without EXACT_OPENMP_RECURRING_BACKEND={collection_mode!r} "
        "or an actual GitHub Actions CI context."
    )


def build_collect_command(
    manifest: dict[str, Any],
    profile: dict[str, Any],
    output_path: Path,
    collection_mode: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "exact_respa_openmp_validation" / "collect_host_report.py"),
        "--out",
        str(output_path),
        "--topology-class",
        profile["topology_class"],
        "--host-label",
        profile["host_label"],
        "--collection-mode",
        collection_mode,
        "--release-binary",
        profile["release_binary"],
        "--gmx-bin",
        profile["gmx_bin"],
        "--fixture-root",
        profile.get("fixture_root", "tests/reference_results/m6_respa"),
        "--system",
        profile.get("system", "small_salt_polymer_box"),
    ]
    host_suffix = profile.get("filename_host_suffix", "")
    if host_suffix:
        cmd.extend(["--filename-host-suffix", host_suffix])

    tsan_binary = profile.get("tsan_binary", "")
    tsan_required = bool(profile.get("tsan_required"))
    if tsan_binary:
        cmd.extend(["--tsan-binary", tsan_binary])
        for key, value in profile.get("tsan_env", {}).items():
            cmd.extend(["--tsan-env", f"{key}={value}"])
    elif not tsan_required:
        cmd.append("--skip-tsan-gtests")
    else:
        raise ValueError(
            f"Profile {profile['host_label']} requires TSAN evidence but no tsan_binary is configured."
        )

    return cmd


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    profiles = manifest["profiles"]
    if args.profile_id not in profiles:
        raise KeyError(f"Unknown profile id: {args.profile_id}")
    profile = profiles[args.profile_id]

    topology = inspect_topology()
    expected_cpu_slug = profile["cpu_model_slug"]
    current_cpu_slug = cpu_model_slug(topology)
    if current_cpu_slug != expected_cpu_slug and not args.allow_host_mismatch:
        raise SystemExit(
            f"Current host CPU slug '{current_cpu_slug}' does not match manifest profile "
            f"'{args.profile_id}' expecting '{expected_cpu_slug}'."
        )

    expected_filename = canonical_report_filename(
        topology,
        profile["topology_class"],
        profile.get("filename_host_suffix", ""),
    )
    if expected_filename != profile["report_filename"] and not args.allow_host_mismatch:
        raise SystemExit(
            f"Manifest report filename '{profile['report_filename']}' does not match the "
            f"current host canonical filename '{expected_filename}'."
        )

    if args.expected_git_commit:
        current_commit = current_repo_commit()
        if current_commit != args.expected_git_commit:
            raise SystemExit(
                "Refusing to run host profile from the wrong repo revision. "
                f"Expected {args.expected_git_commit}, got {current_commit}."
            )

    output_path = profile_output_path(manifest, profile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collection_mode = resolve_collection_mode(profile, args.collection_mode_override)
    assert_recurring_mode_attestation(collection_mode)
    cmd = build_collect_command(
        manifest,
        profile,
        output_path,
        collection_mode,
    )
    if args.expected_git_commit:
        cmd.extend(["--expected-git-commit", args.expected_git_commit])

    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
