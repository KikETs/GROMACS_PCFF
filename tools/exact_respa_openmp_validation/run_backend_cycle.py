from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from collect_host_report import REPO_ROOT
from run_host_profile import DEFAULT_MANIFEST, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one backend collection cycle for the manifest-declared exact OpenMP "
            "host profiles, then validate the active report set."
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
        help="Restrict the backend cycle to specific manifest profile ids. Repeat to add more.",
    )
    parser.add_argument(
        "--collection-mode-override",
        choices=("manual", "manual-host", "ci", "scheduled"),
        help="Override the collection mode used for every collected profile in this cycle.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "output" / "tmp" / "exact-openmp-backend-cycle"),
        help="Directory for validate_report_set.py outputs.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Do not run validate_report_set.py after collection.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run validate_report_set.py in strict mode after collection. "
            "Disable only for diagnostics with --no-strict."
        ),
    )
    return parser.parse_args()


def profile_ids_for_cycle(manifest: dict[str, object], requested_ids: list[str] | None) -> list[str]:
    if requested_ids:
        return requested_ids
    active_ids = manifest.get("active_host_profiles", [])
    return list(active_ids)


def run_command(cmd: list[str], *, env_updates: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    profile_ids = profile_ids_for_cycle(manifest, args.profile_ids)
    if not profile_ids:
        raise SystemExit("No host profiles selected for the backend cycle.")

    for profile_id in profile_ids:
        collection_mode = args.collection_mode_override or manifest["profiles"][profile_id].get(
            "collection_mode",
            "manual-host",
        )
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "exact_respa_openmp_validation" / "run_host_profile.py"),
            "--manifest",
            str(manifest_path),
            "--profile-id",
            profile_id,
        ]
        if args.collection_mode_override:
            cmd.extend(["--collection-mode-override", args.collection_mode_override])
        env_updates = (
            {"EXACT_OPENMP_RECURRING_BACKEND": collection_mode}
            if collection_mode in {"ci", "scheduled"}
            else None
        )
        run_command(cmd, env_updates=env_updates)

    if args.skip_validate:
        return

    validate_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "exact_respa_openmp_validation" / "validate_report_set.py"),
        "--manifest",
        str(manifest_path),
        "--out-dir",
        str(Path(args.out_dir).resolve()),
    ]
    if args.strict:
        validate_cmd.append("--strict")
    run_command(validate_cmd)


if __name__ == "__main__":
    main()
