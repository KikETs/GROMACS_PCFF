#!/usr/bin/env python3
"""Host-local LAMMPS MPI x OpenMP throughput sweep for PolyGen PCFF inputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


PERF_RE = re.compile(r"Performance:\s+([0-9.]+)\s+ns/day")


def parse_counts(text: str) -> list[int]:
    out: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"core counts must be positive, got {value}")
        out.append(value)
    return out


def factor_cases(total: int) -> list[tuple[int, int]]:
    return [(mpi, total // mpi) for mpi in range(1, total + 1) if total % mpi == 0]


def cpu_ids(count: int) -> str:
    visible = os.sched_getaffinity(0)
    ordered = sorted(visible)
    if len(ordered) < count:
        raise RuntimeError(f"requested {count} CPUs but affinity exposes only {len(ordered)}: {ordered}")
    return ",".join(str(v) for v in ordered[:count])


def copy_inputs(md_dir: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ["equil_stage01_em.restart"]:
        src = md_dir / name
        if not src.exists():
            raise FileNotFoundError(f"required benchmark restart is missing: {src}")
        shutil.copy2(src, run_dir / name)
    for name in ["ion_parameters", "molecular_templates"]:
        src = md_dir / name
        dst = run_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        if src.exists():
            shutil.copytree(src, dst)


def write_input(path: Path, *, steps: int, thermo: int) -> None:
    path.write_text(
        f"""echo both
units           real
boundary        p p p
atom_style      full

pair_style      lj/class2/coul/long 9.5
kspace_style    pppm 0.0001
pair_modify     mix sixthpower
pair_modify     tail yes
bond_style      class2
angle_style     class2
dihedral_style  class2
improper_style  class2
read_restart    equil_stage01_em.restart

neighbor        3.0 bin
neigh_modify    delay 0 every 1 check yes

variable        time equal step*dt+0.000001

include         ./ion_parameters/Li.params
include         ./ion_parameters/TFSI.params
special_bonds   lj/coul 0.0 0.0 1.0

timestep        2.0
run_style       respa 3 2 2 bond 1 angle 1 dihedral 1 improper 1 pair 2 kspace 3
thermo_style    custom step v_time press temp pe ke etotal
thermo_modify   flush yes
thermo          {thermo}

velocity        all create 353 1540264 dist gaussian mom yes rot yes
fix             1 all nvt temp 353 353 1000
run             {steps}
unfix           1
""",
        encoding="utf-8",
    )


def run_case(
    *,
    run_dir: Path,
    lmp_binary: Path,
    mpirun_binary: Path | None,
    total_cores: int,
    mpi_ranks: int,
    omp_threads: int,
    steps: int,
    timeout: int,
) -> dict[str, object]:
    copy_inputs(args.md_dir, run_dir)
    thermo = max(100, steps // 4)
    write_input(run_dir / "bench.in", steps=steps, thermo=thermo)

    cpuset = cpu_ids(total_cores)
    lmp_cmd = [
        str(lmp_binary),
        "-nonbuf",
        "-sf",
        "omp",
        "-pk",
        "omp",
        str(omp_threads),
        "-in",
        "bench.in",
    ]
    if mpi_ranks == 1:
        cmd = ["taskset", "-c", cpuset, *lmp_cmd]
    else:
        if mpirun_binary is None:
            raise FileNotFoundError("mpirun is required for mpi_ranks > 1")
        cmd = [
            "taskset",
            "-c",
            cpuset,
            str(mpirun_binary),
            "--oversubscribe",
            "--map-by",
            "slot",
            "--bind-to",
            "none",
            "-np",
            str(mpi_ranks),
            *lmp_cmd,
        ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    if mpi_ranks > 1:
        env["OMP_PROC_BIND"] = "false"
        env.pop("OMP_PLACES", None)
    else:
        env.setdefault("OMP_PROC_BIND", "close")
        env.setdefault("OMP_PLACES", "cores")
    env.setdefault("OMPI_MCA_btl_vader_single_copy_mechanism", "cma")

    stdout_path = run_dir / "stdout.log"
    with stdout_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            cwd=str(run_dir),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=timeout,
        )

    text_parts = []
    for candidate in [stdout_path, run_dir / "log.lammps"]:
        if candidate.exists():
            text_parts.append(candidate.read_text(encoding="utf-8", errors="ignore"))
    text = "\n".join(text_parts)
    matches = PERF_RE.findall(text)
    ns_per_day = float(matches[-1]) if matches else None
    return {
        "total_cores": total_cores,
        "mpi_ranks": mpi_ranks,
        "omp_threads": omp_threads,
        "ns_per_day": ns_per_day,
        "returncode": proc.returncode,
        "cpuset": cpuset,
        "run_dir": str(run_dir),
        "cmd": cmd,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--lmp-binary", required=True, type=Path)
    parser.add_argument("--mpirun-binary", type=Path, default=None)
    parser.add_argument("--core-counts", default="6,8,10,12")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=600)
    ns = parser.parse_args()

    global args
    args = ns
    ns.md_dir = ns.md_dir.expanduser().resolve()
    ns.outdir = ns.outdir.expanduser().resolve()
    ns.lmp_binary = ns.lmp_binary.expanduser().resolve()
    ns.mpirun_binary = ns.mpirun_binary.expanduser().resolve() if ns.mpirun_binary else None

    if not ns.lmp_binary.exists():
        raise FileNotFoundError(f"LAMMPS binary not found: {ns.lmp_binary}")
    if ns.mpirun_binary is not None and not ns.mpirun_binary.exists():
        raise FileNotFoundError(f"mpirun binary not found: {ns.mpirun_binary}")

    ns.outdir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for total in parse_counts(ns.core_counts):
        for mpi_ranks, omp_threads in factor_cases(total):
            label = f"c{total}_mpi{mpi_ranks}_omp{omp_threads}"
            run_dir = ns.outdir / label
            try:
                result = run_case(
                    run_dir=run_dir,
                    lmp_binary=ns.lmp_binary,
                    mpirun_binary=ns.mpirun_binary,
                    total_cores=total,
                    mpi_ranks=mpi_ranks,
                    omp_threads=omp_threads,
                    steps=ns.steps,
                    timeout=ns.timeout,
                )
            except Exception as exc:
                result = {
                    "total_cores": total,
                    "mpi_ranks": mpi_ranks,
                    "omp_threads": omp_threads,
                    "ns_per_day": None,
                    "returncode": -1,
                    "error": repr(exc),
                    "run_dir": str(run_dir),
                }
            results.append(result)
            speed = result.get("ns_per_day")
            print(
                f"{label}: "
                f"{speed if speed is not None else 'failed'} ns/day "
                f"rc={result.get('returncode')}",
                flush=True,
            )

    result_json = ns.outdir / "results.json"
    result_csv = ns.outdir / "results.csv"
    result_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with result_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["total_cores", "mpi_ranks", "omp_threads", "ns_per_day", "returncode", "cpuset", "run_dir", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    ok = [r for r in results if isinstance(r.get("ns_per_day"), (float, int)) and int(r.get("returncode", 1)) == 0]
    if ok:
        best = max(ok, key=lambda row: float(row["ns_per_day"]))
        (ns.outdir / "best.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
        print("BEST", json.dumps(best, sort_keys=True), flush=True)
        return 0
    print("No successful sweep cases.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
