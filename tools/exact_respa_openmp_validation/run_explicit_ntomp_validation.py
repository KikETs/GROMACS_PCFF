from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from collect_host_report import (
    AFFINITY_SUITE_DEFINITIONS,
    REPO_ROOT,
    SUPPORTED_OPENMP_SYSTEM_IDS,
    SUPPORTED_PIN_MODES,
    command_record,
    inspect_topology,
    parse_gtest_case_statuses,
)


DEFAULT_COUNTS = (12,)
DEFAULT_BASELINE_FILTER = "PcffRespaRestartParity/*"
DEFAULT_AFFINITY_FILTER = (
    "PcffRespaRestartAffinityParity/*:"
    "PcffRespaOpenMPAffinityParity/*"
)
DEFAULT_OPENMP_PARITY_FILTER = "PcffRespaOpenMPParity/*"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "exact_respa_openmp_validation_manual"
PROCESS_RE = re.compile(r"(?:^|[\s/])(gmx(?:\s+mdrun)?|mdrun(?:-[^\s/]+)?)\b")


def pick_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def default_release_binary() -> Path | None:
    return pick_existing_path(
        [
            REPO_ROOT / "build" / "bin" / "mdrun-non-integrator-test",
            REPO_ROOT / "build-worktree" / "bin" / "mdrun-non-integrator-test",
            REPO_ROOT / "build_respa_fix_check" / "bin" / "mdrun-non-integrator-test",
            REPO_ROOT / "build_gateb_cuda" / "bin" / "mdrun-non-integrator-test",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run host-local explicit ntomp exact r-RESPA validation without widening the checked-in "
            "CPU OpenMP support claim."
        )
    )
    parser.add_argument(
        "--binary",
        default=str(default_release_binary() or ""),
        help="Path to the release mdrun-non-integrator-test binary.",
    )
    parser.add_argument(
        "--ntomp",
        action="append",
        dest="counts",
        type=int,
        help="Explicit ntomp value to validate. Repeat to add more counts. Default: 12 only.",
    )
    parser.add_argument(
        "--baseline-filter",
        default=DEFAULT_BASELINE_FILTER,
        help="GTest filter for the ntomp=1 oracle baseline restart suite.",
    )
    parser.add_argument(
        "--affinity-filter",
        default=DEFAULT_AFFINITY_FILTER,
        help="GTest filter for the affinity parity/restart bundle.",
    )
    parser.add_argument(
        "--openmp-parity-filter",
        default=DEFAULT_OPENMP_PARITY_FILTER,
        help="GTest filter for the non-affinity ntomp>1 oracle parity suite.",
    )
    parser.add_argument(
        "--wait-for-idle",
        action="store_true",
        help="Wait until no other GROMACS processes are running before starting.",
    )
    parser.add_argument(
        "--idle-poll-seconds",
        type=int,
        default=30,
        help="Polling interval while waiting for the host to become GROMACS-idle.",
    )
    parser.add_argument(
        "--build-first",
        action="store_true",
        help="Build the selected test target before running validation.",
    )
    parser.add_argument(
        "--build-jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel build jobs when --build-first is used.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Directory for logs and JSON summary. Default: timestamped directory under output/exact_respa_openmp_validation_manual.",
    )
    return parser.parse_args()


def suite_output_text(record: dict[str, Any]) -> str:
    stdout = record.get("stdout", "")
    stderr = record.get("stderr", "")
    return "\n".join(part for part in (stderr, stdout) if part)


def affinity_visible_cpus() -> int:
    return len(os.sched_getaffinity(0))


def current_gromacs_processes(exclude_pids: set[int]) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=", "-o", "args="],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    matches: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid_text, _, args = line.partition(" ")
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        if pid in exclude_pids:
            continue
        if not PROCESS_RE.search(args):
            continue
        matches.append({"pid": pid, "args": args})
    return matches


def wait_for_idle(poll_seconds: int) -> dict[str, Any]:
    observed: list[dict[str, Any]] = []
    exclude_pids = {os.getpid(), os.getppid()}
    while True:
        matches = current_gromacs_processes(exclude_pids)
        if not matches:
            return {
                "waited": bool(observed),
                "poll_seconds": poll_seconds,
                "observed_process_sets": observed,
            }
        observed.append(
            {
                "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "processes": matches,
            }
        )
        time.sleep(max(1, poll_seconds))


def build_target(binary: Path, jobs: int) -> dict[str, Any]:
    build_dir = binary.parent.parent
    cmd = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        binary.name,
        "--parallel",
        str(max(1, jobs)),
    ]
    return command_record(cmd, cwd=REPO_ROOT, check=False)


def explicit_override_env(count: int) -> dict[str, str]:
    return {
        "GMX_TEST_EXACT_RESPA_NTOMP_SMALL_OVERRIDE": str(count),
        "GMX_TEST_EXACT_RESPA_NTOMP_CEILING_OVERRIDE": str(count),
    }


def explicit_affinity_case_names(label: str) -> dict[str, list[str]]:
    cases: dict[str, list[str]] = {}
    for suite_key, suite_def in AFFINITY_SUITE_DEFINITIONS.items():
        suite_cases = []
        for system_id in SUPPORTED_OPENMP_SYSTEM_IDS:
            for pin_mode in SUPPORTED_PIN_MODES:
                suite_cases.append(
                    f"{suite_def['case_prefix']}/"
                    f"{system_id}_{pin_mode['label']}_{label}"
                )
        cases[suite_key] = suite_cases
    return cases


def summarize_affinity_result(
    count: int,
    visible_cpus: int,
    suite_output: str,
    returncode: int,
) -> dict[str, Any]:
    parsed = parse_gtest_case_statuses(suite_output)
    statuses = parsed["case_statuses"]
    required_small_cases = explicit_affinity_case_names("ntompSmall")
    redundant_ceiling_cases = explicit_affinity_case_names("ntompCeiling")

    suite_summaries: dict[str, Any] = {}
    for suite_key, suite_def in AFFINITY_SUITE_DEFINITIONS.items():
        required_cases = required_small_cases[suite_key]
        redundant_cases = redundant_ceiling_cases[suite_key]
        suite_summaries[suite_key] = {
            "suite_label": suite_def["suite_label"],
            "required_cases": required_cases,
            "passed_required_cases": [case for case in required_cases if statuses.get(case) == "ok"],
            "failed_required_cases": [case for case in required_cases if statuses.get(case) == "failed"],
            "skipped_required_cases": [case for case in required_cases if statuses.get(case) == "skipped"],
            "missing_required_cases": [case for case in required_cases if case not in statuses],
            "redundant_cases": redundant_cases,
            "skipped_redundant_cases": [case for case in redundant_cases if statuses.get(case) == "skipped"],
            "unexpected_ok_redundant_cases": [case for case in redundant_cases if statuses.get(case) == "ok"],
            "failed_redundant_cases": [case for case in redundant_cases if statuses.get(case) == "failed"],
            "missing_redundant_cases": [case for case in redundant_cases if case not in statuses],
        }

    validated_pin_modes = []
    skipped_pin_modes = []
    failed_pin_modes = []
    for pin_mode in SUPPORTED_PIN_MODES:
        pin_label = pin_mode["label"]
        pin_cases = [
            case
            for suite_cases in required_small_cases.values()
            for case in suite_cases
            if f"_{pin_label}_" in case
        ]
        if pin_cases and all(statuses.get(case) == "ok" for case in pin_cases):
            validated_pin_modes.append(pin_label)
            continue
        if any(statuses.get(case) == "failed" for case in pin_cases):
            failed_pin_modes.append(pin_label)
            continue
        if any(statuses.get(case) == "skipped" for case in pin_cases):
            skipped_pin_modes.append(pin_label)

    blockers: list[str] = []
    if returncode != 0:
        blockers.append("Affinity validation bundle returned nonzero.")
    for suite_key, suite_summary in suite_summaries.items():
        if suite_summary["failed_required_cases"]:
            blockers.append(
                f"{suite_key}: required cases failed: {', '.join(suite_summary['failed_required_cases'])}"
            )
        if suite_summary["skipped_required_cases"]:
            blockers.append(
                f"{suite_key}: required cases skipped: {', '.join(suite_summary['skipped_required_cases'])}"
            )
        if suite_summary["missing_required_cases"]:
            blockers.append(
                f"{suite_key}: required cases missing: {', '.join(suite_summary['missing_required_cases'])}"
            )
        if suite_summary["failed_redundant_cases"]:
            blockers.append(
                f"{suite_key}: redundant cases failed: {', '.join(suite_summary['failed_redundant_cases'])}"
            )
        if suite_summary["missing_redundant_cases"]:
            blockers.append(
                f"{suite_key}: redundant cases missing: {', '.join(suite_summary['missing_redundant_cases'])}"
            )
        if suite_summary["unexpected_ok_redundant_cases"]:
            blockers.append(
                f"{suite_key}: redundant cases unexpectedly executed: {', '.join(suite_summary['unexpected_ok_redundant_cases'])}"
            )

    if count >= visible_cpus:
        expected_skip_reason = (
            "pinInherit needs spare CPUs to construct a non-default inherited mask; "
            f"ntomp={count} on a {visible_cpus}-CPU affinity mask cannot satisfy that precondition."
        )
    else:
        expected_skip_reason = ""

    return {
        "gtest_case_summary": parsed,
        "suite_summaries": suite_summaries,
        "validated_pin_modes": validated_pin_modes,
        "skipped_pin_modes": skipped_pin_modes,
        "failed_pin_modes": failed_pin_modes,
        "full_pin_mode_coverage": len(validated_pin_modes) == len(SUPPORTED_PIN_MODES),
        "expected_pin_inherit_skip": bool(expected_skip_reason),
        "expected_pin_inherit_skip_reason": expected_skip_reason,
        "blockers": blockers,
    }


def summarize_openmp_parity_result(suite_output: str, returncode: int) -> dict[str, Any]:
    parsed = parse_gtest_case_statuses(suite_output)
    statuses = parsed["case_statuses"]
    required_cases = sorted(
        case_name
        for case_name in statuses
        if case_name.startswith(
            "PcffRespaOpenMPParity/PcffRespaOpenMPParityTest.ExactRespaCpuMatchesNtompOneOracle/"
        )
    )

    return {
        "gtest_case_summary": parsed,
        "required_cases": required_cases,
        "passed_required_cases": [case for case in required_cases if statuses.get(case) == "ok"],
        "failed_required_cases": [case for case in required_cases if statuses.get(case) == "failed"],
        "skipped_required_cases": [case for case in required_cases if statuses.get(case) == "skipped"],
        "missing_required_cases": [case for case in required_cases if case not in statuses],
        "ok": bool(required_cases)
        and returncode == 0
        and all(statuses.get(case) == "ok" for case in required_cases if case in statuses)
        and not [case for case in required_cases if case not in statuses],
    }


def summarize_baseline_restart_result(suite_output: str, returncode: int) -> dict[str, Any]:
    parsed = parse_gtest_case_statuses(suite_output)
    statuses = parsed["case_statuses"]
    required_cases = sorted(
        case_name
        for case_name in statuses
        if case_name.startswith(
            "PcffRespaRestartParity/PcffRespaRestartParityTest.RestartFromCheckpointMatchesFullExactRun/"
        )
    )

    return {
        "gtest_case_summary": parsed,
        "required_cases": required_cases,
        "passed_required_cases": [case for case in required_cases if statuses.get(case) == "ok"],
        "failed_required_cases": [case for case in required_cases if statuses.get(case) == "failed"],
        "skipped_required_cases": [case for case in required_cases if statuses.get(case) == "skipped"],
        "missing_required_cases": [case for case in required_cases if case not in statuses],
        "ok": bool(required_cases)
        and returncode == 0
        and all(statuses.get(case) == "ok" for case in required_cases if case in statuses)
        and not [case for case in required_cases if case not in statuses],
    }


def run_baseline_count(binary: Path, topology: dict[str, Any], baseline_filter: str) -> dict[str, Any]:
    logical_cpus = int(topology["logical_cpus"])
    visible_cpus = affinity_visible_cpus()
    baseline_cmd = [str(binary), f"--gtest_filter={baseline_filter}"]
    baseline_record = command_record(baseline_cmd, cwd=REPO_ROOT, env=os.environ.copy(), check=False)
    baseline_output = suite_output_text(baseline_record)
    baseline_summary = summarize_baseline_restart_result(baseline_output, baseline_record["returncode"])

    blockers: list[str] = []
    if not baseline_summary["ok"]:
        blockers.append("Oracle baseline restart suite did not pass cleanly.")

    return {
        "ntomp": 1,
        "category": "oracle_baseline",
        "host_logical_cpus": logical_cpus,
        "affinity_visible_cpus": visible_cpus,
        "claim_scope_note": (
            "ntomp=1 is recorded as the oracle baseline only. "
            "It is not counted as host-local OpenMP support evidence."
        ),
        "baseline_restart": {
            "command_record": baseline_record,
            "summary": baseline_summary,
        },
        "ok": not blockers,
        "blockers": blockers,
    }


def run_explicit_count(
    binary: Path,
    count: int,
    topology: dict[str, Any],
    affinity_filter: str,
    openmp_parity_filter: str,
) -> dict[str, Any]:
    logical_cpus = int(topology["logical_cpus"])
    visible_cpus = affinity_visible_cpus()
    if count <= 1:
        raise SystemExit(f"Explicit ntomp count must be > 1, got {count}.")
    if count > visible_cpus:
        raise SystemExit(
            f"Explicit ntomp count {count} exceeds the current affinity-visible CPU count {visible_cpus}."
        )

    affinity_env = os.environ.copy()
    affinity_env.update(explicit_override_env(count))
    affinity_cmd = [str(binary), f"--gtest_filter={affinity_filter}"]
    affinity_record = command_record(affinity_cmd, cwd=REPO_ROOT, env=affinity_env, check=False)
    affinity_output = suite_output_text(affinity_record)
    affinity_summary = summarize_affinity_result(
        count,
        visible_cpus,
        affinity_output,
        affinity_record["returncode"],
    )

    parity_cmd = [str(binary), "-ntomp", str(count), f"--gtest_filter={openmp_parity_filter}"]
    parity_record = command_record(parity_cmd, cwd=REPO_ROOT, env=os.environ.copy(), check=False)
    parity_output = suite_output_text(parity_record)
    parity_summary = summarize_openmp_parity_result(parity_output, parity_record["returncode"])

    overall_blockers = list(affinity_summary["blockers"])
    if not parity_summary["ok"]:
        overall_blockers.append("Non-affinity ntomp>1 oracle parity suite did not pass cleanly.")

    return {
        "ntomp": count,
        "category": "host_local_explicit_count",
        "host_logical_cpus": logical_cpus,
        "affinity_visible_cpus": visible_cpus,
        "claim_scope_note": (
            "This is host-local manual evidence for explicit ntomp counts only. "
            "It does not modify the checked-in bounded CPU OpenMP claim."
        ),
        "affinity_bundle": {
            "override_env": explicit_override_env(count),
            "command_record": affinity_record,
            "summary": affinity_summary,
        },
        "openmp_parity": {
            "command_record": parity_record,
            "summary": parity_summary,
        },
        "ok": not overall_blockers,
        "blockers": overall_blockers,
    }


def main() -> None:
    args = parse_args()
    binary = Path(args.binary).resolve() if args.binary else None
    if binary is None or not binary.exists():
        raise SystemExit("Release test binary was not found. Pass --binary explicitly.")

    counts = tuple(args.counts or DEFAULT_COUNTS)
    if len(set(counts)) != len(counts):
        raise SystemExit("Duplicate --ntomp values are not allowed.")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir).resolve() if args.out_dir else DEFAULT_OUTPUT_ROOT / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    topology = inspect_topology()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "binary": str(binary),
        "requested_ntomp_counts": list(counts),
        "topology": topology,
        "wait_for_idle": None,
        "build": None,
        "results": [],
        "overall_ok": False,
        "overall_blockers": [],
    }

    if args.wait_for_idle:
        print("Waiting for other GROMACS processes to finish...", flush=True)
        summary["wait_for_idle"] = wait_for_idle(args.idle_poll_seconds)
        print("Host is GROMACS-idle. Starting explicit ntomp validation.", flush=True)

    if args.build_first:
        print(f"Building {binary.name} before validation...", flush=True)
        build_record = build_target(binary, args.build_jobs)
        summary["build"] = build_record
        if build_record["returncode"] != 0:
            summary["overall_blockers"].append("Build step failed.")
            summary["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            summary_path = output_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps(summary, indent=2, sort_keys=True))
            raise SystemExit(build_record["returncode"])
        print("Build completed. Running explicit ntomp suites.", flush=True)

    for count in counts:
        print(f"Running explicit ntomp={count} validation...", flush=True)
        if count == 1:
            result = run_baseline_count(binary, topology, args.baseline_filter)
        else:
            result = run_explicit_count(
                binary,
                count,
                topology,
                args.affinity_filter,
                args.openmp_parity_filter,
            )
        summary["results"].append(result)

    summary["overall_blockers"] = [
        blocker for result in summary["results"] for blocker in result["blockers"]
    ]
    summary["overall_ok"] = not summary["overall_blockers"]
    summary["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSummary written to {summary_path}")

    if summary["overall_blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
