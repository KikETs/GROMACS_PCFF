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


def current_repo_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


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


REMOTE_BACKEND_MANAGED_PATHS = [
    "tools/exact_respa_openmp_validation",
    "tests/reference_results/exact_respa_openmp_validation/report_set_manifest.json",
    "tests/reference_results/exact_respa_openmp_validation/host_reports",
    "tests/reference_results/exact_respa_openmp_validation/stale_host_reports",
]


def shell_join_paths(paths: list[str]) -> str:
    return " ".join(shlex.quote(path) for path in paths)


def remote_backend_cleanup_snippet() -> str:
    managed_paths = shell_join_paths(REMOTE_BACKEND_MANAGED_PATHS)
    return f"""
tracked_backend_paths=$(git ls-files -- {managed_paths})
if [ -n "$tracked_backend_paths" ]; then
  # Restore tracked backend files first; invalid/untracked pathspecs must not block cleanup.
  # shellcheck disable=SC2086
  git restore --worktree --staged --source=HEAD -- $tracked_backend_paths >/dev/null 2>&1 || true
fi
git clean -fd -- {managed_paths} >/dev/null 2>&1 || true
"""


def prepare_remote_repo(
    ssh_target: str,
    remote_repo_path: str,
    *,
    profile_id: str,
    target_commit: str,
) -> str:
    state_path = f"/tmp/exact-openmp-{profile_id}.state"
    cleanup_backend = remote_backend_cleanup_snippet()
    prepare_cmd = f"""
cd {shlex.quote(remote_repo_path)}
dirty_output=$(git status --porcelain)
if [ -n "$dirty_output" ]; then
  while IFS= read -r line; do
    path="${{line#?? }}"
    case "$path" in
      tools/exact_respa_openmp_validation/*|\
tests/reference_results/exact_respa_openmp_validation/report_set_manifest.json|\
tests/reference_results/exact_respa_openmp_validation/host_reports/*|\
tests/reference_results/exact_respa_openmp_validation/stale_host_reports|\
tests/reference_results/exact_respa_openmp_validation/stale_host_reports/*)
        ;;
      *)
        echo "Remote repo has non-backend dirty path: $path" >&2
        exit 1
        ;;
    esac
  done <<EOF_DIRTY
$dirty_output
EOF_DIRTY
  {cleanup_backend}
fi
orig_ref=$(git symbolic-ref --quiet --short HEAD || true)
orig_commit=$(git rev-parse HEAD)
printf '%s\n%s\n' "$orig_ref" "$orig_commit" > {shlex.quote(state_path)}
git fetch --no-tags origin {shlex.quote(target_commit)}
git checkout --detach {shlex.quote(target_commit)}
"""
    run_checked(["ssh", ssh_target, prepare_cmd])
    return state_path


def restore_remote_repo(
    ssh_target: str,
    remote_repo_path: str,
    *,
    state_path: str,
) -> None:
    cleanup_backend = remote_backend_cleanup_snippet()
    restore_cmd = f"""
cd {shlex.quote(remote_repo_path)}
cat {shlex.quote(state_path)}
"""
    completed = subprocess.run(
        ["ssh", ssh_target, restore_cmd],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = completed.stdout.strip().splitlines()
    orig_ref = lines[0] if lines else ""
    orig_commit = lines[1] if len(lines) > 1 else ""
    checkout_target = orig_ref or orig_commit
    if not checkout_target:
        raise SystemExit(f"{ssh_target}: missing remote checkout restore target")
    cleanup_cmd = f"""
cd {shlex.quote(remote_repo_path)}
{cleanup_backend}
git checkout -q {shlex.quote(checkout_target)}
rm -f {shlex.quote(state_path)}
"""
    run_checked(["ssh", ssh_target, cleanup_cmd])


def recollect_remote_profile(
    profile_id: str,
    profile: dict[str, Any],
    manifest_path: Path,
    collection_mode: str,
    target_commit: str,
) -> None:
    backend_target = profile["recurring_backend"]["backend_target"]
    ssh_target = backend_target["ssh_target"]
    remote_repo_path = backend_target["repo_path"]
    remote_temp_report = f"/tmp/exact-openmp-{profile_id}.json"
    local_report_path = (
        REPO_ROOT
        / "tests"
        / "reference_results"
        / "exact_respa_openmp_validation"
        / "host_reports"
        / profile["report_filename"]
    )
    state_path = prepare_remote_repo(
        ssh_target,
        remote_repo_path,
        profile_id=profile_id,
        target_commit=target_commit,
    )
    sync_backend_to_remote(ssh_target, remote_repo_path, manifest_path)
    try:
        remote_cmd = (
            f"cd {shlex.quote(remote_repo_path)} && "
            f"EXACT_OPENMP_RECURRING_BACKEND={shlex.quote(collection_mode)} "
            f"python3 tools/exact_respa_openmp_validation/run_host_profile.py "
            f"--manifest tests/reference_results/exact_respa_openmp_validation/report_set_manifest.json "
            f"--profile-id {profile_id} "
            f"--collection-mode-override {collection_mode} "
            f"--expected-git-commit {shlex.quote(target_commit)} && "
            f"cp tests/reference_results/exact_respa_openmp_validation/host_reports/{shlex.quote(profile['report_filename'])} "
            f"{shlex.quote(remote_temp_report)}"
        )
        run_checked(["ssh", ssh_target, remote_cmd])
        run_checked(
            [
                "rsync",
                "-az",
                f"{ssh_target}:{remote_temp_report}",
                str(local_report_path),
            ]
        )
    finally:
        run_checked(["ssh", ssh_target, f"rm -f {shlex.quote(remote_temp_report)}"])
        restore_remote_repo(
            ssh_target,
            remote_repo_path,
            state_path=state_path,
        )


def recollect_local_profile(
    profile_id: str,
    manifest_path: Path,
    collection_mode: str,
    target_commit: str,
) -> None:
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
            "--expected-git-commit",
            target_commit,
        ],
        env_updates={"EXACT_OPENMP_RECURRING_BACKEND": collection_mode},
    )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    profiles = manifest["profiles"]
    target_commit = current_repo_commit()
    profile_ids = selected_profile_ids(manifest, args.profile_ids)
    if not profile_ids:
        raise SystemExit("No active host profiles selected for recurring backend.")

    for profile_id in profile_ids:
        profile = profiles[profile_id]
        backend_target = profile.get("recurring_backend", {}).get("backend_target", {"mode": "local"})
        mode = backend_target.get("mode", "local")
        if mode == "local":
            recollect_local_profile(profile_id, manifest_path, args.collection_mode, target_commit)
            continue
        if mode == "ssh":
            recollect_remote_profile(
                profile_id,
                profile,
                manifest_path,
                args.collection_mode,
                target_commit,
            )
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
