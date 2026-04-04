from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect one host-local exact r-RESPA CPU OpenMP validation report "
            "that can later be aggregated into a broader desktop-class CPU support claim."
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
    if one_l3_phys is not None:
        one_l3_best = max_run(one_l3_phys)
        l3_size = len(one_l3_phys["cpus"])
        larger_shapes, larger_shape_above = larger_shape_runs_above_threshold(benchmark, l3_size)
        if not larger_shapes:
            return {
                "rule_ready_for_cross_host_aggregation": False,
                "reason": (
                    "No larger-than-one_l3 locality shape was observed on this host, so the "
                    "one-L3 physical-core ceiling cannot be stress-tested."
                ),
                "host_local_observation": {
                    "global_best_shape": global_best_shape,
                    "global_best_ntomp": global_best["ntomp"],
                    "global_best_ns_per_day": global_best["ns_per_day"],
                },
            }
        if not larger_shape_above:
            return {
                "rule_ready_for_cross_host_aggregation": False,
                "reason": (
                    "Larger locality shapes exist on this host, but no runs above the one-L3 "
                    "physical-core size were collected."
                ),
                "host_local_observation": {
                    "global_best_shape": global_best_shape,
                    "global_best_ntomp": global_best["ntomp"],
                    "global_best_ns_per_day": global_best["ns_per_day"],
                },
            }
        if (
            one_l3_best is not None
            and float(one_l3_best["ns_per_day"]) >= 0.95 * float(global_best["ns_per_day"])
            and all(
                float(run["ns_per_day"]) <= 0.8 * float(global_best["ns_per_day"])
                for run in larger_shape_above
            )
        ):
            return {
                "rule_ready_for_cross_host_aggregation": True,
                "production_candidate": {
                    "rule_text": "Up to one L3 or CCD-equivalent locality group using one hardware thread per physical core",
                    "basis": "one_l3_group_physical_threads",
                    "ceiling_threads_on_this_host": len(one_l3_phys["cpus"]),
                    "best_shape": "one_l3_phys",
                    "best_ntomp": one_l3_best["ntomp"],
                    "best_ns_per_day": one_l3_best["ns_per_day"],
                },
                "correctness_only_candidate": {
                    "rule_text": "Above the production locality ceiling but within mechanically validated thread counts on this host",
                    "max_mechanically_validated_threads_on_this_host": topology["logical_cpus"],
                },
                "unsupported_or_unproven": {
                    "rule_text": "Beyond tested thread counts or on hosts whose locality groups have not been benchmarked",
                },
                "host_local_observation": {
                    "global_best_shape": global_best_shape,
                    "global_best_ntomp": global_best["ntomp"],
                    "global_best_ns_per_day": global_best["ns_per_day"],
                },
                "mechanics_preconditions": {
                    "release_exact_suite_ok": bool(release_suite and release_suite.get("ok")),
                    "tsan_exact_suite_ok": bool(tsan_suite and tsan_suite.get("ok")),
                },
            }

    return {
        "rule_ready_for_cross_host_aggregation": False,
        "reason": (
            "This host does not yet show a stable performance drop on larger locality shapes "
            "beyond the one-L3 physical-core size."
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
    release_suite = None
    tsan_suite = None
    benchmark = None

    if not args.skip_release_gtests:
        if not args.release_binary:
            raise ValueError("--release-binary is required unless --skip-release-gtests is set")
        release_suite = run_release_suite(Path(args.release_binary), args.release_filter)

    if not args.skip_tsan_gtests:
        if not args.tsan_binary:
            raise ValueError("--tsan-binary is required unless --skip-tsan-gtests is set")
        tsan_suite = run_tsan_suite(
            Path(args.tsan_binary),
            args.tsan_filter,
            parse_tsan_env(args.tsan_env),
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

    report = {
        "schema_version": 1,
        "host": {
            "label": args.host_label,
            "topology_class": args.topology_class,
            "topology": topology,
        },
        "mechanics": {
            "release_suite": release_suite,
            "tsan_suite": tsan_suite,
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
