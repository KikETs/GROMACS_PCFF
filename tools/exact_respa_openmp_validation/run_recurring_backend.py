from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from collect_host_report import REPO_ROOT
from run_host_profile import DEFAULT_MANIFEST, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the recurring exact OpenMP backend from the central repo host. "
            "Local profiles run in-place, remote profiles are recollected over SSH "
            "and rsynced back into the active host_reports inventory."
        )
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to report_set_manifest.json.",
    )
    parser.add_argument(
        "--profile-id",
        action="append",
        dest="profile_ids",
        help="Restrict recurring backend execution to selected profile ids.",
    )
    parser.add_argument(
        "--collection-mode",
        choices=("ci", "scheduled"),
        default="scheduled",
        help="Recurring collection mode recorded in the refreshed reports.",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/exact-openmp-recurring-backend",
        help="Directory used by validate_report_set.py after recollection.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run validate_report_set.py in strict mode after recollection. "
            "Disable only for diagnostics with --no-strict."
        ),
    )
    return parser.parse_args()


def selected_profile_ids(manifest: dict[str, Any], requested_ids: list[str] | None) -> list[str]:
    if requested_ids:
        return requested_ids
    return list(manifest.get("active_host_profiles", []))


def run_checked(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env_updates: dict[str, str] | None = None,
) -> None:
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def sync_backend_to_remote(ssh_target: str, remote_repo_path: str, manifest_path: Path) -> None:
    run_checked(
        [
            "rsync",
            "-az",
            "--delete",
            str(REPO_ROOT / "tools" / "exact_respa_openmp_validation") + "/",
            f"{ssh_target}:{remote_repo_path}/tools/exact_respa_openmp_validation/",
        ]
    )
    run_checked(
        [
            "rsync",
            "-az",
            str(manifest_path),
            f"{ssh_target}:{remote_repo_path}/tests/reference_results/exact_respa_openmp_validation/report_set_manifest.json",
        ]
    )


def recollect_remote_profile(
    profile_id: str,
    profile: dict[str, Any],
    manifest_path: Path,
    collection_mode: str,
) -> None:
    backend_target = profile["recurring_backend"]["backend_target"]
    ssh_target = backend_target["ssh_target"]
    remote_repo_path = backend_target["repo_path"]
    sync_backend_to_remote(ssh_target, remote_repo_path, manifest_path)
    remote_cmd = (
        f"cd {shlex.quote(remote_repo_path)} && "
        f"EXACT_OPENMP_RECURRING_BACKEND={shlex.quote(collection_mode)} "
        f"python3 tools/exact_respa_openmp_validation/run_host_profile.py "
        f"--manifest tests/reference_results/exact_respa_openmp_validation/report_set_manifest.json "
        f"--profile-id {profile_id} "
        f"--collection-mode-override {collection_mode}"
    )
    run_checked(["ssh", ssh_target, remote_cmd])
    run_checked(
        [
            "rsync",
            "-az",
            f"{ssh_target}:{remote_repo_path}/tests/reference_results/exact_respa_openmp_validation/host_reports/{profile['report_filename']}",
            str(REPO_ROOT / "tests" / "reference_results" / "exact_respa_openmp_validation" / "host_reports") + "/",
        ]
    )


def recollect_local_profile(profile_id: str, manifest_path: Path, collection_mode: str) -> None:
    run_checked(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "exact_respa_openmp_validation" / "run_host_profile.py"),
            "--manifest",
            str(manifest_path),
            "--profile-id",
            profile_id,
            "--collection-mode-override",
            collection_mode,
        ],
        env_updates={"EXACT_OPENMP_RECURRING_BACKEND": collection_mode},
    )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    profiles = manifest["profiles"]
    profile_ids = selected_profile_ids(manifest, args.profile_ids)
    if not profile_ids:
        raise SystemExit("No active host profiles selected for recurring backend.")

    for profile_id in profile_ids:
        profile = profiles[profile_id]
        backend_target = profile.get("recurring_backend", {}).get("backend_target", {"mode": "local"})
        mode = backend_target.get("mode", "local")
        if mode == "local":
            recollect_local_profile(profile_id, manifest_path, args.collection_mode)
            continue
        if mode == "ssh":
            recollect_remote_profile(profile_id, profile, manifest_path, args.collection_mode)
            continue
        raise SystemExit(f"Unknown backend_target mode for {profile_id}: {mode}")

    run_checked(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "exact_respa_openmp_validation" / "validate_report_set.py"),
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(Path(args.out_dir).resolve()),
        ]
        + (["--strict"] if args.strict else [])
    )


if __name__ == "__main__":
    main()
