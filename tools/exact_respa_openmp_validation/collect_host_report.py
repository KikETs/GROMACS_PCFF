from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_ROOT = REPO_ROOT / "tests" / "reference_results" / "m6_respa"
DEFAULT_GTEST_FILTER = (
    "PcffRespaRestartParity/*:"
    "PcffRespaOpenMPParity/*:"
    "PcffRespaRestartAffinityParity/*:"
    "PcffRespaOpenMPAffinityParity/*"
)
DEFAULT_TSAN_FILTER = (
    "PcffRespaRestartParity/*:"
    "PcffRespaOpenMPParity/*:"
    "PcffRespaRestartAffinityParity/*:"
    "PcffRespaOpenMPAffinityParity/*"
)
DEFAULT_BENCH_COUNTS = (1, 2, 4, 6, 8, 12, 16, 24)
DEFAULT_SYSTEM = "small_salt_polymer_box"
REPORT_SCHEMA_VERSION = 3
REPORT_FILENAME_POLICY_VERSION = 1
SUPPORTED_OPENMP_SYSTEM_IDS = ("small_oligomer", "small_salt_polymer_box")
SUPPORTED_PIN_MODES = (
    {"label": "pinAuto", "cli_value": "auto"},
    {"label": "pinOn", "cli_value": "on"},
    {"label": "pinInherit", "cli_value": "inherit"},
)
AFFINITY_SUITE_DEFINITIONS = {
    "restart_affinity": {
        "suite_label": "restart parity",
        "case_prefix": (
            "PcffRespaRestartAffinityParity/"
            "PcffRespaRestartAffinityParityTest."
            "RestartFromCheckpointMatchesFullExactRunWithAffinity"
        ),
    },
    "openmp_affinity": {
        "suite_label": "ntomp>1 oracle parity",
        "case_prefix": (
            "PcffRespaOpenMPAffinityParity/"
            "PcffRespaOpenMPAffinityParityTest."
            "ExactRespaCpuMatchesNtompOneOracleWithAffinity"
        ),
    },
}
GTEST_CASE_STATUS_RE = re.compile(r"^\[\s*(OK|FAILED|SKIPPED)\s*\]\s+(.+?)(?: \(\d+ ms\))?$")


def pick_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


DEFAULT_RELEASE_BINARY = pick_existing_path(
    [
        REPO_ROOT / "build-worktree" / "bin" / "mdrun-non-integrator-test",
        REPO_ROOT / "build" / "bin" / "mdrun-non-integrator-test",
    ]
)
DEFAULT_TSAN_BINARY = pick_existing_path(
    [
        REPO_ROOT / "build-clang-tsan-o2" / "bin" / "mdrun-non-integrator-test",
        REPO_ROOT / "build-tsan-o2" / "bin" / "mdrun-non-integrator-test",
    ]
)
DEFAULT_GMX_BINARY = pick_existing_path(
    [
        REPO_ROOT / "build-worktree" / "bin" / "gmx",
        REPO_ROOT / "build" / "bin" / "gmx",
    ]
)


def recurring_backend_attestation(collection_mode: str) -> dict[str, Any]:
    backend_attestation = os.getenv("EXACT_OPENMP_RECURRING_BACKEND")
    github_actions = os.getenv("GITHUB_ACTIONS") == "true"

    if collection_mode == "ci":
        if backend_attestation == "ci":
            return {"required": True, "attested": True, "source": "env"}
        if github_actions:
            return {"required": True, "attested": True, "source": "github_actions"}
        return {"required": True, "attested": False, "source": "missing"}

    if collection_mode == "scheduled":
        if backend_attestation == "scheduled":
            return {"required": True, "attested": True, "source": "env"}
        return {"required": True, "attested": False, "source": "missing"}

    return {"required": False, "attested": False, "source": "not-required"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect one host-local exact r-RESPA CPU OpenMP validation report "
            "that can later be aggregated into a bounded cross-host CPU OpenMP mechanics claim."
        )
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--topology-class",
        required=True,
        choices=("low-core-workstation", "mid-core-hybrid-desktop", "numa-or-chiplet"),
        help="Human-audited topology class label for this host report.",
    )
    parser.add_argument(
        "--host-label",
        default=platform.node() or "unknown-host",
        help="Human-readable host label stored in the report.",
    )
    parser.add_argument(
        "--collection-mode",
        choices=("manual", "manual-host", "ci", "scheduled"),
        default="ci" if os.getenv("GITHUB_ACTIONS") == "true" else "manual",
        help="How this report was collected for infrastructure-quality accounting.",
    )
    parser.add_argument(
        "--filename-host-suffix",
        default="",
        help="Optional host-distinguishing suffix appended to the canonical report filename.",
    )
    parser.add_argument(
        "--release-binary",
        default=str(DEFAULT_RELEASE_BINARY) if DEFAULT_RELEASE_BINARY else "",
        help="Release mdrun-non-integrator-test binary used for exact parity suites.",
    )
    parser.add_argument(
        "--tsan-binary",
        default=str(DEFAULT_TSAN_BINARY) if DEFAULT_TSAN_BINARY else "",
        help="TSAN mdrun-non-integrator-test binary used for exact concurrency evidence.",
    )
    parser.add_argument(
        "--gmx-bin",
        default=str(DEFAULT_GMX_BINARY) if DEFAULT_GMX_BINARY else "",
        help="gmx binary used for locality benchmark collection.",
    )
    parser.add_argument(
        "--fixture-root",
        default=str(DEFAULT_FIXTURE_ROOT),
        help="Root containing the exact r-RESPA reference fixtures.",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="Fixture id used for the locality benchmark.",
    )
    parser.add_argument(
        "--release-filter",
        default=DEFAULT_GTEST_FILTER,
        help="GTest filter for the release exact parity/restart suites.",
    )
    parser.add_argument(
        "--tsan-filter",
        default=DEFAULT_TSAN_FILTER,
        help="GTest filter for the TSAN exact concurrency subset.",
    )
    parser.add_argument(
        "--tsan-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment variables for the TSAN binary.",
    )
    parser.add_argument(
        "--skip-release-gtests",
        action="store_true",
        help="Skip release exact-suite execution.",
    )
    parser.add_argument(
        "--skip-tsan-gtests",
        action="store_true",
        help="Skip TSAN exact-suite execution.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip locality benchmark execution.",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=20000,
        help="Number of exact r-RESPA inner steps used for the locality benchmark.",
    )
    parser.add_argument(
        "--benchmark-count",
        action="append",
        dest="benchmark_counts",
        type=int,
        help="Override benchmark ntomp probes. Repeat to add more counts.",
    )
    parser.add_argument(
        "--benchmark-shape",
        action="append",
        dest="benchmark_shapes",
        help="Restrict benchmark collection to named shapes discovered on this host.",
    )
    parser.add_argument(
        "--expected-git-commit",
        default="",
        help=(
            "Require the collecting repo to be at this exact commit before emitting a report. "
            "Used by recurring backends so fresh evidence cannot be collected from the wrong SHA."
        ),
    )
    return parser.parse_args()


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=check,
        text=True,
        capture_output=True,
    )
    return completed


def command_record(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> dict[str, Any]:
    completed = run_command(cmd, cwd=cwd, env=env, check=check)
    return {
        "command": " ".join(shlex.quote(token) for token in cmd),
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def parse_tsan_env(pairs: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"Invalid --tsan-env entry: {pair!r}")
        env[key] = value
    return env


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return re.sub(r"_+", "_", slug) or "unknown"


def cpu_model_slug(topology: dict[str, Any]) -> str:
    model_name = (topology.get("model_name") or "").lower()
    vendor = slugify(topology.get("vendor") or "")

    ryzen_match = re.search(r"ryzen\s+(\d+)\s+([0-9]{4,5}[a-z]*)", model_name)
    if ryzen_match:
        return f"amd_ryzen_{ryzen_match.group(1)}_{ryzen_match.group(2)}"

    intel_match = re.search(r"\b(i[3579])[- ]?([0-9]{4,5}[a-z]*)\b", model_name)
    if intel_match:
        return f"intel_{intel_match.group(1)}_{intel_match.group(2)}"

    if vendor and model_name:
        return f"{vendor}_{slugify(model_name)}"
    if model_name:
        return slugify(model_name)
    return "unknown_cpu"


def canonical_report_filename(
    topology: dict[str, Any],
    topology_class: str,
    host_suffix: str = "",
) -> str:
    model_slug = cpu_model_slug(topology)
    class_slug = slugify(topology_class)
    base = f"{model_slug}_{class_slug}"
    if host_suffix:
        base = f"{base}_{slugify(host_suffix)}"
    return f"{base}.json"


def current_git_revision() -> dict[str, Any]:
    head = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    return {
        "commit": head.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
    }


def summarize_tsan_status(
    *,
    skip_tsan_gtests: bool,
    tsan_binary: str,
    tsan_suite: dict[str, Any] | None,
) -> dict[str, Any]:
    if tsan_suite is not None:
        return {
            "status": "backed" if tsan_suite.get("ok") else "failed",
            "reason": (
                "TSAN exact suite completed successfully."
                if tsan_suite.get("ok")
                else "TSAN exact suite executed but did not pass."
            ),
            "binary_supplied": bool(tsan_binary),
            "suite_requested": True,
        }

    if skip_tsan_gtests and not tsan_binary:
        return {
            "status": "infra-limited",
            "reason": "TSAN exact suite was skipped because no TSAN binary/build was supplied.",
            "binary_supplied": False,
            "suite_requested": False,
        }

    if skip_tsan_gtests:
        return {
            "status": "missing",
            "reason": "TSAN exact suite was skipped even though a TSAN binary path was supplied.",
            "binary_supplied": bool(tsan_binary),
            "suite_requested": False,
        }

    return {
        "status": "missing",
        "reason": "TSAN exact suite status could not be determined.",
        "binary_supplied": bool(tsan_binary),
        "suite_requested": True,
    }


def parse_lscpu_kv() -> dict[str, str]:
    completed = run_command(["lscpu"], cwd=REPO_ROOT)
    values: dict[str, str] = {}
    for raw_line in completed.stdout.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def parse_lscpu_table() -> list[dict[str, Any]]:
    completed = run_command(
        ["lscpu", "-p=CPU,CORE,SOCKET,NODE,ONLINE"],
        cwd=REPO_ROOT,
    )
    rows: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        cpu_s, core_s, socket_s, node_s, online_s = line.split(",")
        rows.append(
            {
                "cpu": int(cpu_s),
                "core": int(core_s),
                "socket": int(socket_s),
                "node": int(node_s),
                "online": online_s.lower() == "y",
            }
        )
    return rows


def read_l3_id(cpu: int) -> int | None:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cache/index3/id")
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        return None
    return int(value)


def inspect_topology() -> dict[str, Any]:
    kv = parse_lscpu_kv()
    cpu_rows = [row for row in parse_lscpu_table() if row["online"]]
    cpu_rows.sort(key=lambda row: row["cpu"])
    l3_groups: dict[str, list[int]] = {}
    core_groups: dict[tuple[int, int, int], list[int]] = {}
    node_groups: dict[str, list[int]] = {}
    for row in cpu_rows:
        l3_id = read_l3_id(row["cpu"])
        row["l3_id"] = l3_id
        core_key = (row["socket"], row["node"], row["core"])
        core_groups.setdefault(core_key, []).append(row["cpu"])
        node_groups.setdefault(str(row["node"]), []).append(row["cpu"])
        if l3_id is not None:
            l3_groups.setdefault(str(l3_id), []).append(row["cpu"])

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "architecture": kv.get("Architecture"),
        "vendor": kv.get("Vendor ID"),
        "model_name": kv.get("Model name"),
        "sockets": int(kv.get("Socket(s)", "0")),
        "cores_per_socket": int(kv.get("Core(s) per socket", "0")),
        "threads_per_core": int(kv.get("Thread(s) per core", "0")),
        "logical_cpus": int(kv.get("CPU(s)", "0")),
        "numa_nodes": int(kv.get("NUMA node(s)", "0")),
        "l3_instances": kv.get("L3 cache"),
        "cpu_rows": cpu_rows,
        "core_groups": {str(key): value for key, value in core_groups.items()},
        "node_groups": node_groups,
        "l3_groups": l3_groups,
    }


def unique_physical_cpus(cpus: list[int], cpu_rows: list[dict[str, Any]]) -> list[int]:
    row_by_cpu = {row["cpu"]: row for row in cpu_rows}
    first_cpu_by_core: dict[tuple[int, int, int], int] = {}
    for cpu in sorted(cpus):
        row = row_by_cpu[cpu]
        core_key = (row["socket"], row["node"], row["core"])
        first_cpu_by_core.setdefault(core_key, cpu)
    return sorted(first_cpu_by_core.values())


def discover_benchmark_shapes(topology: dict[str, Any]) -> list[dict[str, Any]]:
    cpu_rows = topology["cpu_rows"]
    online_cpus = [row["cpu"] for row in cpu_rows]
    all_phys = unique_physical_cpus(online_cpus, cpu_rows)
    shapes: list[dict[str, Any]] = []

    l3_groups = [
        {"id": group_id, "cpus": sorted(cpus)}
        for group_id, cpus in topology["l3_groups"].items()
        if len(cpus) >= 2
    ]
    l3_groups.sort(key=lambda group: (len(group["cpus"]), group["id"]), reverse=True)
    if l3_groups:
        first_group = l3_groups[0]
        first_phys = unique_physical_cpus(first_group["cpus"], cpu_rows)
        if len(first_phys) >= 2:
            shapes.append(
                {
                    "name": "one_l3_phys",
                    "cpus": first_phys,
                    "locality_basis": "one_l3_group_physical_threads",
                    "description": f"One L3/cache locality group ({first_group['id']}), one hardware thread per core",
                }
            )
        if len(first_group["cpus"]) > len(first_phys):
            shapes.append(
                {
                    "name": "one_l3_all_threads",
                    "cpus": first_group["cpus"],
                    "locality_basis": "one_l3_group_all_threads",
                    "description": f"One L3/cache locality group ({first_group['id']}), all SMT threads",
                }
            )

    if len(all_phys) >= 2:
        shapes.append(
            {
                "name": "all_phys_cores",
                "cpus": all_phys,
                "locality_basis": "all_host_physical_threads",
                "description": "All physical cores on the host, one hardware thread per core",
            }
        )

    if len(online_cpus) >= 2:
        shapes.append(
            {
                "name": "full_host",
                "cpus": online_cpus,
                "locality_basis": "all_host_logical_threads",
                "description": "All online logical CPUs on the host",
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for shape in shapes:
        deduped.setdefault(",".join(str(cpu) for cpu in shape["cpus"]), shape)
    return list(deduped.values())


def exact_benchmark_mdp(nsteps: int) -> str:
    return (
        "title = pcff exact respa topology benchmark\n"
        "integrator = md-vv\n"
        "dt = 0.0005\n"
        f"nsteps = {nsteps}\n"
        "constraints = none\n"
        "cutoff-scheme = Verlet\n"
        "nstlist = 4\n"
        "rlist = 0.99\n"
        "rvdw = 0.9\n"
        "rcoulomb = 0.9\n"
        "vdwtype = Cut-off\n"
        "vdw-modifier = none\n"
        "coulombtype = PME\n"
        "coulomb-modifier = none\n"
        "ewald-rtol = 1e-6\n"
        "pme-order = 4\n"
        "fourierspacing = 0.08\n"
        "epsilon-r = 1\n"
        "pbc = xyz\n"
        "tcoupl = no\n"
        "pcoupl = no\n"
        "comm-mode = none\n"
        "verlet-buffer-tolerance = -1\n"
        "gen-vel = no\n"
        "mts = yes\n"
        "mts-mode = lammps-respa\n"
        "mts-levels = 3\n"
        "mts-level2-factor = 2\n"
        "mts-level3-factor = 4\n"
        "mts-respa-bond-level = 1\n"
        "mts-respa-angle-level = 1\n"
        "mts-respa-dihedral-level = 1\n"
        "mts-respa-improper-level = 1\n"
        "mts-respa-pair14-level = 1\n"
        "mts-respa-kspace-level = 3\n"
        "mts-respa-inner-level = 1\n"
        "mts-respa-middle-level = 2\n"
        "mts-respa-outer-level = 3\n"
        "mts-respa-inner-off = 0.30\n"
        "mts-respa-inner-on = 0.45\n"
        "mts-respa-outer-on = 0.60\n"
        "mts-respa-outer-off = 0.80\n"
        "nstcalcenergy = 4\n"
        "nstenergy = 4\n"
        "nstlog = 4\n"
        "nstxout = 0\n"
        "nstvout = 0\n"
        "nstfout = 0\n"
        "nstxout-compressed = 0\n"
    )


def parse_performance_ns_per_day(log_text: str) -> float | None:
    matches = re.findall(r"Performance:\s*([0-9.+\-eE]+)", log_text)
    if not matches:
        return None
    return float(matches[-1])


def suite_output_text(suite_result: dict[str, Any] | None) -> str:
    if not suite_result:
        return ""
    command_record = suite_result.get("command_record", {})
    stdout = command_record.get("stdout", "")
    stderr = command_record.get("stderr", "")
    return "\n".join(part for part in (stderr, stdout) if part)


def parse_gtest_case_statuses(output_text: str) -> dict[str, Any]:
    case_statuses: dict[str, str] = {}
    counts = {"ok": 0, "failed": 0, "skipped": 0}
    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        match = GTEST_CASE_STATUS_RE.match(line)
        if not match:
            continue
        status_raw, case_name = match.groups()
        if "/" not in case_name:
            continue
        status = status_raw.lower()
        case_statuses[case_name] = status
        counts[status] += 1
    return {
        "total_cases_with_status_lines": sum(counts.values()),
        "counts": counts,
        "case_statuses": case_statuses,
    }


def resolved_thread_probe_definitions(topology: dict[str, Any]) -> list[dict[str, Any]]:
    logical_cpus = max(1, int(topology.get("logical_cpus", 0) or 0))
    probes = [
        {
            "label": "ntompSmall",
            "resolved_threads": min(2, logical_cpus),
            "scope_note": "Minimal audited ntomp>1 OpenMP bucket from exact affinity parity tests.",
        },
        {
            "label": "ntompCeiling",
            "resolved_threads": min(6, logical_cpus),
            "scope_note": "Audited ceiling bucket from exact affinity parity tests.",
        },
    ]
    seen_thread_counts: set[int] = set()
    resolved: list[dict[str, Any]] = []
    for probe in probes:
        threads = int(probe["resolved_threads"])
        probe = dict(probe)
        probe["redundant"] = threads in seen_thread_counts or threads <= 1
        if threads > 1:
            seen_thread_counts.add(threads)
        resolved.append(probe)
    return resolved


def required_affinity_case_names(topology: dict[str, Any]) -> dict[str, Any]:
    probe_definitions = [probe for probe in resolved_thread_probe_definitions(topology) if not probe["redundant"]]
    suite_cases: dict[str, list[str]] = {}
    for suite_key, suite_def in AFFINITY_SUITE_DEFINITIONS.items():
        cases = []
        for system_id in SUPPORTED_OPENMP_SYSTEM_IDS:
            for pin_mode in SUPPORTED_PIN_MODES:
                for probe in probe_definitions:
                    cases.append(
                        f"{suite_def['case_prefix']}/"
                        f"{system_id}_{pin_mode['label']}_{probe['label']}"
                    )
        suite_cases[suite_key] = cases
    return {
        "probe_definitions": probe_definitions,
        "suite_cases": suite_cases,
    }


def derive_affinity_case_matrix(
    topology: dict[str, Any],
    suite_result: dict[str, Any] | None,
    suite_key: str,
) -> dict[str, Any]:
    suite_def = AFFINITY_SUITE_DEFINITIONS[suite_key]
    required = required_affinity_case_names(topology)
    required_cases = required["suite_cases"][suite_key]
    probe_definitions = required["probe_definitions"]
    parsed = parse_gtest_case_statuses(suite_output_text(suite_result))
    case_statuses = parsed["case_statuses"]

    passed_cases = sorted(case for case in required_cases if case_statuses.get(case) == "ok")
    failed_cases = sorted(case for case in required_cases if case_statuses.get(case) == "failed")
    skipped_cases = sorted(case for case in required_cases if case_statuses.get(case) == "skipped")
    missing_cases = sorted(case for case in required_cases if case not in case_statuses)

    validated_pin_modes = []
    for pin_mode in SUPPORTED_PIN_MODES:
        pin_cases = [case for case in required_cases if f"_{pin_mode['label']}_" in case]
        if pin_cases and all(case_statuses.get(case) == "ok" for case in pin_cases):
            validated_pin_modes.append(pin_mode)

    validated_systems = []
    for system_id in SUPPORTED_OPENMP_SYSTEM_IDS:
        system_cases = [case for case in required_cases if f"/{system_id}_" in case]
        if system_cases and all(case_statuses.get(case) == "ok" for case in system_cases):
            validated_systems.append(system_id)

    validated_thread_probes = []
    for probe in probe_definitions:
        probe_cases = [case for case in required_cases if case.endswith(f"_{probe['label']}")]
        if probe_cases and all(case_statuses.get(case) == "ok" for case in probe_cases):
            validated_thread_probes.append(probe)

    blockers: list[str] = []
    if suite_result is None:
        blockers.append(f"{suite_def['suite_label']}: suite result is missing")
    elif not suite_result.get("ok", False):
        blockers.append(f"{suite_def['suite_label']}: suite returned nonzero")
    if failed_cases:
        blockers.append(
            f"{suite_def['suite_label']}: required cases failed: {', '.join(failed_cases)}"
        )
    if skipped_cases:
        blockers.append(
            f"{suite_def['suite_label']}: required cases were skipped: {', '.join(skipped_cases)}"
        )
    if missing_cases:
        blockers.append(
            f"{suite_def['suite_label']}: required cases missing from suite output: {', '.join(missing_cases)}"
        )

    return {
        "suite_key": suite_key,
        "suite_label": suite_def["suite_label"],
        "required_cases": required_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "skipped_cases": skipped_cases,
        "missing_cases": missing_cases,
        "validated_pin_modes": validated_pin_modes,
        "validated_systems": validated_systems,
        "validated_thread_probes": validated_thread_probes,
        "case_status_counts": parsed["counts"],
        "ok": not blockers,
        "blockers": blockers,
    }


def annotate_suite_result(
    topology: dict[str, Any],
    suite_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if suite_result is None:
        return None
    if "gtest_case_summary" in suite_result and "exact_openmp_affinity_cases" in suite_result:
        return suite_result
    annotated = dict(suite_result)
    annotated["gtest_case_summary"] = parse_gtest_case_statuses(suite_output_text(suite_result))
    annotated["exact_openmp_affinity_cases"] = {
        suite_key: derive_affinity_case_matrix(topology, suite_result, suite_key)
        for suite_key in AFFINITY_SUITE_DEFINITIONS
    }
    return annotated


def derive_exact_openmp_support_scope(
    topology: dict[str, Any],
    release_suite: dict[str, Any] | None,
    tsan_suite: dict[str, Any] | None,
) -> dict[str, Any]:
    release_suite = annotate_suite_result(topology, release_suite)
    tsan_suite = annotate_suite_result(topology, tsan_suite)
    probe_definitions = [probe for probe in resolved_thread_probe_definitions(topology) if not probe["redundant"]]

    blockers: list[str] = []
    if not probe_definitions:
        blockers.append("No audited ntomp>1 bucket exists on this host.")

    suite_groups = {
        "release": release_suite,
        "tsan": tsan_suite,
    }
    for suite_family, suite_result in suite_groups.items():
        if suite_result is None:
            blockers.append(f"{suite_family}: exact affinity suite result is missing")
            continue
        matrices = suite_result["exact_openmp_affinity_cases"]
        for suite_key in AFFINITY_SUITE_DEFINITIONS:
            blockers.extend(
                f"{suite_family}: {blocker}" for blocker in matrices[suite_key]["blockers"]
            )

    supported_pin_modes = []
    for pin_mode in SUPPORTED_PIN_MODES:
        pin_label = pin_mode["label"]
        if all(
            suite_result is not None
            and all(
                any(validated["label"] == pin_label for validated in suite_result["exact_openmp_affinity_cases"][suite_key]["validated_pin_modes"])
                for suite_key in AFFINITY_SUITE_DEFINITIONS
            )
            for suite_result in suite_groups.values()
        ):
            supported_pin_modes.append(pin_mode)

    supported_systems = []
    for system_id in SUPPORTED_OPENMP_SYSTEM_IDS:
        if all(
            suite_result is not None
            and all(
                system_id in suite_result["exact_openmp_affinity_cases"][suite_key]["validated_systems"]
                for suite_key in AFFINITY_SUITE_DEFINITIONS
            )
            for suite_result in suite_groups.values()
        ):
            supported_systems.append(system_id)

    supported_thread_probes = []
    for probe in probe_definitions:
        if all(
            suite_result is not None
            and all(
                any(validated["label"] == probe["label"] for validated in suite_result["exact_openmp_affinity_cases"][suite_key]["validated_thread_probes"])
                for suite_key in AFFINITY_SUITE_DEFINITIONS
            )
            for suite_result in suite_groups.values()
        ):
            supported_thread_probes.append(probe)

    supported_threads = sorted({probe["resolved_threads"] for probe in supported_thread_probes})
    support_ready = (
        not blockers
        and len(supported_pin_modes) == len(SUPPORTED_PIN_MODES)
        and len(supported_systems) == len(SUPPORTED_OPENMP_SYSTEM_IDS)
        and len(supported_thread_probes) == len(probe_definitions)
    )
    if not support_ready and not blockers:
        blockers.append("The audited affinity/pin-mode matrix is incomplete on this host.")

    pin_scope_text = ", ".join(f"`-pin {pin_mode['cli_value']}`" for pin_mode in SUPPORTED_PIN_MODES)
    thread_scope_text = ", ".join(
        f"`{probe['label']}` (`ntomp={probe['resolved_threads']}`)"
        for probe in supported_thread_probes
    )
    supported_scope_statement = (
        "On this host, standalone exact r-RESPA CPU OpenMP support is limited to the audited "
        f"affinity modes {pin_scope_text} and the discrete ntomp>1 buckets {thread_scope_text}. "
        "Those buckets are supported only because checked-in release and TSAN suites both pass "
        "ntomp>1 oracle parity and restart parity on the two frozen fixtures."
        if support_ready
        else "This host does not currently earn a bounded CPU OpenMP support claim."
    )

    return {
        "support_ready": support_ready,
        "supported_scope_statement": supported_scope_statement,
        "scope_note": (
            "The supported OpenMP mechanics envelope is discrete. "
            "It does not interpolate from the audited ntomp buckets to intermediate or larger "
            "thread counts, and it does not treat benchmark-only no-crash runs as proof."
        ),
        "supported_pin_modes": supported_pin_modes,
        "supported_systems": list(supported_systems),
        "supported_thread_probes": supported_thread_probes,
        "supported_thread_counts": supported_threads,
        "correctness_only_scope_statement": (
            "None. No checked-in parity/restart artifact extends support beyond the discrete "
            "audited ntomp buckets."
        ),
        "unsupported_or_weak_shapes": [
            "ntomp=1 is the oracle baseline and is not counted as OpenMP support evidence.",
            "Intermediate ntomp>1 counts between the audited buckets are mechanically unvalidated.",
            "Counts above the audited ntomp ceiling are mechanically unvalidated.",
            "Benchmark-only `-pin inherit` throughput scans are host-local observations only and do not create supported or correctness-only ntomp counts.",
            "MPI, GPU coexistence, server CPUs, and untested topology classes remain outside this scope.",
        ],
        "release_suite": release_suite,
        "tsan_suite": tsan_suite,
        "blockers": blockers,
    }


def run_release_suite(binary: Path, gtest_filter: str) -> dict[str, Any]:
    record = command_record(
        [str(binary), f"--gtest_filter={gtest_filter}"],
        cwd=REPO_ROOT,
        check=False,
    )
    return {
        "binary": str(binary),
        "gtest_filter": gtest_filter,
        "ok": record["returncode"] == 0,
        "command_record": record,
    }


def run_tsan_suite(binary: Path, gtest_filter: str, extra_env: dict[str, str]) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(extra_env)
    record = command_record(
        [str(binary), f"--gtest_filter={gtest_filter}"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )
    return {
        "binary": str(binary),
        "gtest_filter": gtest_filter,
        "extra_env": extra_env,
        "ok": record["returncode"] == 0,
        "command_record": record,
    }


def run_locality_benchmark(
    gmx_bin: Path,
    fixture_root: Path,
    system_id: str,
    shapes: list[dict[str, Any]],
    counts: list[int],
    nsteps: int,
) -> dict[str, Any]:
    system_root = fixture_root / system_id
    if not system_root.exists():
        raise FileNotFoundError(f"Missing exact r-RESPA benchmark fixture: {system_root}")

    with tempfile.TemporaryDirectory(prefix="exact-respa-openmp-") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        mdp = tmpdir / "benchmark.mdp"
        mdp.write_text(exact_benchmark_mdp(nsteps), encoding="utf-8")
        tpr = tmpdir / "benchmark.tpr"
        grompp = command_record(
            [
                str(gmx_bin),
                "grompp",
                "-f",
                str(mdp),
                "-c",
                str(system_root / "initial_nve.gro"),
                "-p",
                str(system_root / "topol.top"),
                "-o",
                str(tpr),
                "-po",
                str(tmpdir / "mdout.mdp"),
                "-maxwarn",
                "1",
            ],
            cwd=REPO_ROOT,
            check=False,
        )
        if grompp["returncode"] != 0:
            return {
                "ok": False,
                "grompp": grompp,
                "shapes": [],
            }

        benchmark_shapes: list[dict[str, Any]] = []
        for shape in shapes:
            selected_counts = [count for count in counts if 1 <= count <= len(shape["cpus"])]
            shape_runs: list[dict[str, Any]] = []
            cpuset = ",".join(str(cpu) for cpu in shape["cpus"])
            for ntomp in selected_counts:
                stem = tmpdir / f"{shape['name']}_nt{ntomp}"
                cmd = [
                    "taskset",
                    "-c",
                    cpuset,
                    str(gmx_bin),
                    "mdrun",
                    "-s",
                    str(tpr),
                    "-deffnm",
                    str(stem),
                    "-ntmpi",
                    "1",
                    "-ntomp",
                    str(ntomp),
                    "-pin",
                    "inherit",
                    "-nb",
                    "cpu",
                    "-bonded",
                    "cpu",
                    "-pme",
                    "cpu",
                    "-update",
                    "cpu",
                ]
                record = command_record(cmd, cwd=REPO_ROOT, check=False)
                log_path = stem.with_suffix(".log")
                log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
                shape_runs.append(
                    {
                        "ntomp": ntomp,
                        "ok": record["returncode"] == 0,
                        "ns_per_day": parse_performance_ns_per_day(log_text),
                        "log_mentions_exact_mode": "lammps-respa" in log_text,
                        "log_mentions_ntomp": f"Using {ntomp} OpenMP thread" in log_text,
                        "command_record": record,
                    }
                )
            benchmark_shapes.append(
                {
                    "name": shape["name"],
                    "cpus": shape["cpus"],
                    "size": len(shape["cpus"]),
                    "description": shape["description"],
                    "locality_basis": shape["locality_basis"],
                    "runs": shape_runs,
                }
            )

    return {
        "ok": True,
        "grompp": grompp,
        "shapes": benchmark_shapes,
    }


def max_run(shape: dict[str, Any]) -> dict[str, Any] | None:
    successful = [
        run
        for run in shape["runs"]
        if run["ok"] and run["ns_per_day"] is not None and run["log_mentions_exact_mode"]
    ]
    if not successful:
        return None
    return max(successful, key=lambda run: float(run["ns_per_day"]))


def larger_shape_runs_above_threshold(
    benchmark: dict[str, Any],
    reference_shape_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    larger_shapes = [
        shape
        for shape in benchmark["shapes"]
        if len(shape["cpus"]) > reference_shape_size
    ]
    above_threshold_runs = []
    for shape in larger_shapes:
        for run in shape["runs"]:
            if (
                run["ok"]
                and run["ns_per_day"] is not None
                and run["ntomp"] > reference_shape_size
                and run["log_mentions_exact_mode"]
            ):
                above_threshold_runs.append(
                    {
                        "shape_name": shape["name"],
                        "shape_size": len(shape["cpus"]),
                        "ntomp": run["ntomp"],
                        "ns_per_day": run["ns_per_day"],
                    }
                )
    return larger_shapes, above_threshold_runs


def successful_exact_runs(shape: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        run
        for run in shape["runs"]
        if run["ok"] and run["ns_per_day"] is not None and run["log_mentions_exact_mode"]
    ]


def derive_host_local_rule(
    topology: dict[str, Any],
    benchmark: dict[str, Any] | None,
    release_suite: dict[str, Any] | None,
    tsan_suite: dict[str, Any] | None,
) -> dict[str, Any]:
    if benchmark is None or not benchmark.get("ok"):
        return {
            "rule_ready_for_cross_host_aggregation": False,
            "reason": "No locality benchmark evidence was collected on this host.",
        }

    shapes = {shape["name"]: shape for shape in benchmark["shapes"]}
    global_best: dict[str, Any] | None = None
    global_best_shape: str | None = None
    for shape in benchmark["shapes"]:
        best = max_run(shape)
        if best is None:
            continue
        if global_best is None or float(best["ns_per_day"]) > float(global_best["ns_per_day"]):
            global_best = best
            global_best_shape = shape["name"]

    if global_best is None:
        return {
            "rule_ready_for_cross_host_aggregation": False,
            "reason": "The locality benchmark produced no successful exact r-RESPA runs.",
        }

    one_l3_phys = shapes.get("one_l3_phys")
    locality_shapes = [
        shape
        for shape in benchmark["shapes"]
        if shape["name"].startswith("one_l3_")
    ]
    if one_l3_phys is not None and locality_shapes:
        locality_runs: list[dict[str, Any]] = []
        for shape in locality_shapes:
            for run in successful_exact_runs(shape):
                locality_runs.append(
                    {
                        "shape_name": shape["name"],
                        "shape_size": len(shape["cpus"]),
                        "ntomp": run["ntomp"],
                        "ns_per_day": run["ns_per_day"],
                    }
                )

        if not locality_runs:
            return {
                "rule_ready_for_cross_host_aggregation": False,
                "reason": "No successful exact locality-group runs were collected on this host.",
                "host_local_observation": {
                    "global_best_shape": global_best_shape,
                    "global_best_ntomp": global_best["ntomp"],
                    "global_best_ns_per_day": global_best["ns_per_day"],
                },
            }

        locality_best = max(locality_runs, key=lambda run: float(run["ns_per_day"]))
        plateau_threshold = 0.95 * float(locality_best["ns_per_day"])
        plateau_runs = [
            run for run in locality_runs if float(run["ns_per_day"]) >= plateau_threshold
        ]
        plateau_ceiling = max(run["ntomp"] for run in plateau_runs)
        post_plateau_runs = [
            run for run in locality_runs if run["ntomp"] > plateau_ceiling
        ]
        if not post_plateau_runs:
            return {
                "rule_ready_for_cross_host_aggregation": False,
                "reason": (
                    "No successful locality-group runs were collected above the plateau "
                    "candidate, so the host-local production knee cannot be demonstrated."
                ),
                "host_local_observation": {
                    "global_best_shape": global_best_shape,
                    "global_best_ntomp": global_best["ntomp"],
                    "global_best_ns_per_day": global_best["ns_per_day"],
                    "locality_best_shape": locality_best["shape_name"],
                    "locality_best_ntomp": locality_best["ntomp"],
                    "locality_best_ns_per_day": locality_best["ns_per_day"],
                },
            }

        meaningful_drop_threshold = 0.90 * float(locality_best["ns_per_day"])
        meaningful_drop_runs = [
            run
            for run in post_plateau_runs
            if float(run["ns_per_day"]) <= meaningful_drop_threshold
        ]
        if not meaningful_drop_runs:
            return {
                "rule_ready_for_cross_host_aggregation": False,
                "reason": (
                    "Locality-group runs above the candidate ceiling were collected, but they "
                    "do not yet show a meaningful throughput drop beyond the plateau."
                ),
                "host_local_observation": {
                    "global_best_shape": global_best_shape,
                    "global_best_ntomp": global_best["ntomp"],
                    "global_best_ns_per_day": global_best["ns_per_day"],
                    "locality_best_shape": locality_best["shape_name"],
                    "locality_best_ntomp": locality_best["ntomp"],
                    "locality_best_ns_per_day": locality_best["ns_per_day"],
                },
            }

        return {
            "rule_ready_for_cross_host_aggregation": True,
            "production_candidate": {
                "rule_text": (
                    "Within one L3 or CCD-equivalent locality group, the shared OpenMP "
                    "thread-scaling ceiling extends up to the host-local throughput plateau knee, defined as "
                    "the highest tested thread count still within 95% of the best exact rate "
                    "observed in that locality group"
                ),
                "basis": "one_l3_group_plateau_95pct",
                "ceiling_threads_on_this_host": plateau_ceiling,
                "best_shape": locality_best["shape_name"],
                "best_ntomp": locality_best["ntomp"],
                "best_ns_per_day": locality_best["ns_per_day"],
                "plateau_threshold_ns_per_day": plateau_threshold,
            },
            "correctness_only_candidate": {
                "rule_text": "Above the locality plateau knee but within mechanically validated thread counts on this host",
                "max_mechanically_validated_threads_on_this_host": topology["logical_cpus"],
            },
            "unsupported_or_unproven": {
                "rule_text": "Beyond tested thread counts or on hosts whose locality groups have not been benchmarked",
            },
            "host_local_observation": {
                "global_best_shape": global_best_shape,
                "global_best_ntomp": global_best["ntomp"],
                "global_best_ns_per_day": global_best["ns_per_day"],
                "locality_best_shape": locality_best["shape_name"],
                "locality_best_ntomp": locality_best["ntomp"],
                "locality_best_ns_per_day": locality_best["ns_per_day"],
                "meaningful_drop_ntomps": [run["ntomp"] for run in meaningful_drop_runs],
            },
            "mechanics_preconditions": {
                "release_exact_suite_ok": bool(release_suite and release_suite.get("ok")),
                "tsan_exact_suite_ok": bool(tsan_suite and tsan_suite.get("ok")),
            },
        }

    return {
        "rule_ready_for_cross_host_aggregation": False,
        "reason": (
            "This host does not expose enough one-L3 locality-group evidence to derive a "
            "plateau-based OpenMP thread-scaling envelope."
        ),
        "host_local_observation": {
            "global_best_shape": global_best_shape,
            "global_best_ntomp": global_best["ntomp"],
            "global_best_ns_per_day": global_best["ns_per_day"],
        },
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    topology = inspect_topology()
    expected_filename = canonical_report_filename(
        topology,
        args.topology_class,
        args.filename_host_suffix,
    )
    if out_path.name != expected_filename:
        raise ValueError(
            "Non-canonical host report filename. "
            f"Expected '{expected_filename}', got '{out_path.name}'."
        )

    git_revision = current_git_revision()
    if args.expected_git_commit and git_revision["commit"] != args.expected_git_commit:
        raise SystemExit(
            "Refusing to emit a host report from the wrong repo revision. "
            f"Expected {args.expected_git_commit}, got {git_revision['commit']}."
        )
    release_suite = None
    tsan_suite = None
    benchmark = None

    if not args.skip_release_gtests:
        if not args.release_binary:
            raise ValueError("--release-binary is required unless --skip-release-gtests is set")
        release_suite = annotate_suite_result(
            topology,
            run_release_suite(Path(args.release_binary), args.release_filter),
        )

    if not args.skip_tsan_gtests:
        if not args.tsan_binary:
            raise ValueError("--tsan-binary is required unless --skip-tsan-gtests is set")
        tsan_suite = annotate_suite_result(
            topology,
            run_tsan_suite(
                Path(args.tsan_binary),
                args.tsan_filter,
                parse_tsan_env(args.tsan_env),
            ),
        )

    if not args.skip_benchmark:
        if not args.gmx_bin:
            raise ValueError("--gmx-bin is required unless --skip-benchmark is set")
        all_shapes = discover_benchmark_shapes(topology)
        selected_shapes = all_shapes
        if args.benchmark_shapes:
            allowed = set(args.benchmark_shapes)
            selected_shapes = [shape for shape in all_shapes if shape["name"] in allowed]
        benchmark_counts = sorted(set(args.benchmark_counts or DEFAULT_BENCH_COUNTS))
        benchmark = run_locality_benchmark(
            Path(args.gmx_bin),
            Path(args.fixture_root),
            args.system,
            selected_shapes,
            benchmark_counts,
            args.benchmark_steps,
        )

    tsan_status = summarize_tsan_status(
        skip_tsan_gtests=args.skip_tsan_gtests,
        tsan_binary=args.tsan_binary,
        tsan_suite=tsan_suite,
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "host": {
            "label": args.host_label,
            "topology_class": args.topology_class,
            "topology": topology,
        },
        "infra": {
            "collection_mode": args.collection_mode,
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": git_revision,
            "expected_git_commit": args.expected_git_commit or None,
            "report_filename_policy_version": REPORT_FILENAME_POLICY_VERSION,
            "report_filename": out_path.name,
            "canonical_report_filename": expected_filename,
            "report_filename_matches_canonical": out_path.name == expected_filename,
            "cpu_model_slug": cpu_model_slug(topology),
            "filename_host_suffix": slugify(args.filename_host_suffix) if args.filename_host_suffix else "",
            "recurring_backend_attestation": recurring_backend_attestation(args.collection_mode),
            "ci_context": {
                "github_actions": os.getenv("GITHUB_ACTIONS") == "true",
                "github_run_id": os.getenv("GITHUB_RUN_ID"),
                "github_job": os.getenv("GITHUB_JOB"),
                "github_workflow": os.getenv("GITHUB_WORKFLOW"),
            },
            "tsan": tsan_status,
        },
        "mechanics": {
            "release_suite": release_suite,
            "tsan_suite": tsan_suite,
            "openmp_support_scope": derive_exact_openmp_support_scope(
                topology,
                release_suite,
                tsan_suite,
            ),
        },
        "benchmark": benchmark,
    }
    report["derived_host_local_rule"] = derive_host_local_rule(
        topology,
        benchmark,
        release_suite,
        tsan_suite,
    )
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
