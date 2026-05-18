#!/usr/bin/env python3
"""Sweep LAMMPS MPI/OpenMP layouts per staged PolyGen input.

The output JSON is intentionally shaped for polygen_multisystem_worker.py
`--lammps-stage-layout-file`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


STAGE_INPUTS = {
    "equil_stage00_pre_em": "equil_stage00_pre_em.in",
    "equil_stage01_em": "equil_stage01_em.in",
    "equil_stage02_dynamics": "equil_stage02_dynamics.in",
    "prod_stage00_nvt": "production_stage00_nvt.in",
    "prod_stage01_clusters": "production_stage01_clusters.in",
}


def discover_stage_inputs(md_dir: Path) -> dict[str, Path]:
    stages: dict[str, Path] = {}
    for stage in ("equil_stage00_pre_em", "equil_stage01_em"):
        path = md_dir / STAGE_INPUTS[stage]
        if path.exists():
            stages[stage] = path
    resume_inputs = md_dir / "resume_inputs"
    legacy_inputs = []
    if resume_inputs.is_dir():
        legacy_inputs = sorted(resume_inputs.glob("lammps_equil_*.in"))
        for path in legacy_inputs:
            stages[path.stem] = path
    if not legacy_inputs:
        path = md_dir / STAGE_INPUTS["equil_stage02_dynamics"]
        if path.exists():
            stages["equil_stage02_dynamics"] = path
    for stage in ("prod_stage00_nvt", "prod_stage01_clusters"):
        path = md_dir / STAGE_INPUTS[stage]
        if path.exists():
            stages[stage] = path
    return stages


def detect_physical_cpu_list() -> list[int]:
    affinity = set(os.sched_getaffinity(0))
    try:
        raw = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,ONLINE"], text=True)
    except Exception:
        return sorted(affinity)
    core_to_cpu: dict[tuple[int, int], int] = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu_s, core_s, socket_s, online_s = [x.strip() for x in line.split(",")[:4]]
        if online_s.upper() != "Y":
            continue
        cpu = int(cpu_s)
        if cpu not in affinity:
            continue
        key = (int(socket_s), int(core_s))
        if key not in core_to_cpu or cpu < core_to_cpu[key]:
            core_to_cpu[key] = cpu
    return sorted(core_to_cpu.values()) or sorted(affinity)


def cpu_list(n: int) -> str:
    cpus = detect_physical_cpu_list()
    if len(cpus) < n:
        affinity = sorted(os.sched_getaffinity(0))
        for cpu in affinity:
            if cpu not in cpus:
                cpus.append(cpu)
            if len(cpus) >= n:
                break
    return ",".join(str(x) for x in cpus[:n])


def divisors(total: int) -> list[tuple[int, int, bool]]:
    out: list[tuple[int, int, bool]] = []
    for mpi in range(1, total + 1):
        if total % mpi != 0:
            continue
        omp = total // mpi
        if omp == 1:
            out.append((mpi, omp, False))
        else:
            out.append((mpi, omp, True))
    return out


def build_cmd(
    *,
    lmp: Path,
    mpirun: Path | None,
    stage_input: str,
    mpi_ranks: int,
    omp_threads: int,
    openmp: bool,
) -> list[str]:
    lmp_cmd = [str(lmp), "-nonbuf"]
    if openmp:
        lmp_cmd += ["-sf", "omp", "-pk", "omp", str(omp_threads)]
    lmp_cmd += ["-in", stage_input]
    if mpi_ranks <= 1:
        return lmp_cmd
    if mpirun is None:
        raise RuntimeError("mpi_ranks > 1 requires --mpirun")
    base = [str(mpirun)]
    try:
        version = subprocess.check_output([str(mpirun), "--version"], text=True, stderr=subprocess.STDOUT, timeout=5)
    except Exception:
        version = ""
    if "Open MPI" in version or Path(mpirun).name in {"orterun", "mpirun.openmpi", "mpiexec.openmpi"}:
        # taskset constrains the sweep to the requested core/thread budget.
        # Letting OpenMPI also bind ranks by physical core breaks 8/10/12-way
        # tests on 6-core/12-thread hosts, so MPI binding is disabled here.
        base += ["--use-hwthread-cpus", "--map-by", "slot", "--bind-to", "none"]
    return [*base, "-np", str(mpi_ranks), *lmp_cmd]


def copy_stage_workspace(md_dir: Path, run_dir: Path, stage_input: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.restart", "*.data", "*.lmp"):
        for src in md_dir.glob(pattern):
            dst = run_dir / src.name
            if not dst.exists():
                dst.symlink_to(src.resolve())
    for name in ("ion_parameters", "molecular_templates"):
        src = md_dir / name
        dst = run_dir / name
        if src.is_dir() and not dst.exists():
            shutil.copytree(src, dst)
    state_src = md_dir / ".resume_state"
    state_dst = run_dir / ".resume_state"
    if state_src.is_dir() and not state_dst.exists():
        shutil.copytree(state_src, state_dst, symlinks=True)
    traj = md_dir / "traj.lammpstrj"
    if traj.exists():
        dst = run_dir / traj.name
        if not dst.exists():
            dst.symlink_to(traj.resolve())
    dst_input = run_dir / stage_input.name
    text = stage_input.read_text(encoding="utf-8", errors="ignore")
    dst_input.write_text(text, encoding="utf-8")
    return dst_input


def shorten_input(path: Path, steps: int, thermo: int) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"(?m)^(\s*thermo\s+).*$", rf"\g<1>{max(1, thermo)}", text)
    text = re.sub(r"(?m)^(\s*variable\s+nave\s+equal\s+).*$", r"\g<1>2", text)
    text = re.sub(r"(?m)^(\s*variable\s+dnave\s+equal\s+).*$", r"\g<1>1", text)
    text = re.sub(r"(?m)^(\s*variable\s+nskip\s+equal\s+).*$", r"\g<1>0", text)
    text = re.sub(r"(?m)^(\s*run\s+)(?:\$\{[^}]+\}|[0-9]+).*$", rf"\g<1>{max(1, steps)}", text)
    text = re.sub(r"(?m)^(\s*minimize\s+\S+\s+\S+\s+)\d+(\s+)\d+.*$", rf"\g<1>50\g<2>200", text)
    if "thermo_modify" not in text:
        text = re.sub(r"(?m)^(\s*thermo_style\b.*)$", r"\1\nthermo_modify   flush yes", text)
    path.write_text(text, encoding="utf-8")


def seed_stage_states(
    *,
    md_dir: Path,
    outdir: Path,
    lmp: Path,
    mpirun: Path | None,
    stages: list[str],
    discovered: dict[str, Path],
    mpi_ranks: int,
    omp_threads: int,
    openmp: bool,
    steps: int,
    thermo: int,
    timeout: int,
) -> Path:
    state_dir = outdir / "state_source"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.restart", "*.data", "*.lmp"):
        for src in md_dir.glob(pattern):
            dst = state_dir / src.name
            if not dst.exists():
                dst.symlink_to(src.resolve())
    for name in ("ion_parameters", "molecular_templates"):
        src = md_dir / name
        dst = state_dir / name
        if src.is_dir() and not dst.exists():
            shutil.copytree(src, dst)
    resume_src = md_dir / "resume_inputs"
    if resume_src.is_dir():
        shutil.copytree(resume_src, state_dir / "resume_inputs")
    (state_dir / ".resume_state").mkdir(exist_ok=True)
    for stage in stages:
        stage_input = discovered.get(stage)
        if stage_input is None or not stage_input.exists():
            continue
        rel = stage_input.relative_to(md_dir)
        dst = state_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage_input, dst)
        shorten_input(dst, steps=steps, thermo=thermo)
        cmd = build_cmd(
            lmp=lmp,
            mpirun=mpirun,
            stage_input=str(rel),
            mpi_ranks=mpi_ranks,
            omp_threads=omp_threads,
            openmp=openmp,
        )
        total_threads = mpi_ranks * (omp_threads if openmp else 1)
        cmd = ["taskset", "-c", cpu_list(total_threads), *cmd]
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(omp_threads if openmp else 1)
        env["OMPI_MCA_btl_vader_single_copy_mechanism"] = "emulated"
        log_path = state_dir / f"seed_{stage}.stdout.log"
        try:
            proc = subprocess.run(
                cmd,
                cwd=state_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
            log_text = proc.stdout
            rc = proc.returncode
        except subprocess.TimeoutExpired as exc:
            log_text = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            rc = -1
        log_path.write_text(log_text, encoding="utf-8")
        if rc != 0:
            raise RuntimeError(f"seed stage failed: {stage}, rc={rc}, log={log_path}")
    return state_dir


def parse_perf(log_text: str) -> dict[str, float | None]:
    ns_day = None
    loop_s = None
    steps = None
    atoms = None
    m = re.search(r"Performance:\s+([0-9.]+)\s+ns/day", log_text)
    if m:
        ns_day = float(m.group(1))
    m = re.search(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+\S+\s+procs\s+for\s+(\d+)\s+steps\s+with\s+(\d+)\s+atoms", log_text)
    if m:
        loop_s = float(m.group(1))
        steps = float(m.group(2))
        atoms = float(m.group(3))
    return {"ns_per_day": ns_day, "loop_s": loop_s, "steps": steps, "atoms": atoms}


def run_one(
    *,
    md_dir: Path,
    outdir: Path,
    lmp: Path,
    mpirun: Path | None,
    stage: str,
    stage_input: Path,
    total_cores: int,
    mpi_ranks: int,
    omp_threads: int,
    openmp: bool,
    steps: int,
    thermo: int,
    timeout: int,
) -> dict[str, Any]:
    label = f"{stage}_c{total_cores}_mpi{mpi_ranks}_omp{omp_threads}_{'omp' if openmp else 'mpi'}"
    run_dir = outdir / "runs" / label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_path = copy_stage_workspace(md_dir, run_dir, stage_input)
    shorten_input(input_path, steps=steps, thermo=thermo)
    cmd = build_cmd(
        lmp=lmp,
        mpirun=mpirun,
        stage_input=input_path.name,
        mpi_ranks=mpi_ranks,
        omp_threads=omp_threads,
        openmp=openmp,
    )
    total_threads = mpi_ranks * (omp_threads if openmp else 1)
    cmd = ["taskset", "-c", cpu_list(total_threads), *cmd]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(omp_threads if openmp else 1)
    env["OMPI_MCA_btl_vader_single_copy_mechanism"] = "emulated"
    log_path = run_dir / "stdout.log"
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=run_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        rc = proc.returncode
        log_text = proc.stdout
        error = ""
    except subprocess.TimeoutExpired as exc:
        rc = -1
        log_text = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        error = f"timeout:{timeout}s"
    wall_s = time.time() - t0
    log_path.write_text(log_text, encoding="utf-8")
    perf = parse_perf(log_text)
    return {
        "stage": stage,
        "total_cores": total_cores,
        "mpi_ranks": mpi_ranks,
        "omp_threads": omp_threads,
        "openmp": openmp,
        "returncode": rc,
        "wall_s": wall_s,
        "error": error,
        "log": str(log_path),
        **perf,
    }


def choose_best(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for stage in sorted({r["stage"] for r in rows}):
        ok = [r for r in rows if r["stage"] == stage and int(r["returncode"]) == 0]
        if not ok:
            continue
        with_perf = [r for r in ok if r.get("ns_per_day") is not None]
        if with_perf:
            chosen = max(with_perf, key=lambda r: float(r["ns_per_day"]))
        else:
            with_loop = [r for r in ok if r.get("loop_s") is not None]
            chosen = min(with_loop or ok, key=lambda r: float(r.get("loop_s") or r["wall_s"]))
        best[stage] = {
            "mpi_ranks": int(chosen["mpi_ranks"]),
            "omp_threads": int(chosen["omp_threads"]),
            "openmp": bool(chosen["openmp"]),
            "source": str(chosen["log"]),
        }
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--md-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--lmp", type=Path, required=True)
    p.add_argument("--mpirun", type=Path, default=None)
    p.add_argument("--totals", default="8,10,12")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--thermo", type=int, default=100)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--stages", default="auto")
    p.add_argument("--seed-states", action="store_true")
    p.add_argument("--seed-steps", type=int, default=100)
    p.add_argument("--seed-thermo", type=int, default=100)
    p.add_argument("--seed-timeout", type=int, default=180)
    args = p.parse_args()

    md_dir = args.md_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    totals = [int(x) for x in str(args.totals).split(",") if x.strip()]
    discovered = discover_stage_inputs(md_dir)
    if str(args.stages).strip().lower() == "auto":
        stages = list(discovered)
    else:
        stages = [x.strip() for x in str(args.stages).split(",") if x.strip()]

    if args.seed_states:
        seed_mpi = max(totals)
        md_dir = seed_stage_states(
            md_dir=md_dir,
            outdir=outdir,
            lmp=args.lmp.resolve(),
            mpirun=args.mpirun.resolve() if args.mpirun else None,
            stages=stages,
            discovered=discovered,
            mpi_ranks=seed_mpi,
            omp_threads=1,
            openmp=False,
            steps=int(args.seed_steps),
            thermo=int(args.seed_thermo),
            timeout=int(args.seed_timeout),
        )
        discovered = discover_stage_inputs(md_dir)

    rows: list[dict[str, Any]] = []
    for stage in stages:
        stage_input = discovered.get(stage)
        if stage_input is None:
            input_name = STAGE_INPUTS.get(stage, stage)
            stage_input = md_dir / input_name
        if not stage_input.exists():
            rows.append({"stage": stage, "returncode": -2, "error": f"missing_input:{stage_input}"})
            continue
        for total in totals:
            for mpi_ranks, omp_threads, openmp in divisors(total):
                row = run_one(
                    md_dir=md_dir,
                    outdir=outdir,
                    lmp=args.lmp.resolve(),
                    mpirun=args.mpirun.resolve() if args.mpirun else None,
                    stage=stage,
                    stage_input=stage_input,
                    total_cores=total,
                    mpi_ranks=mpi_ranks,
                    omp_threads=omp_threads,
                    openmp=openmp,
                    steps=args.steps,
                    thermo=args.thermo,
                    timeout=args.timeout,
                )
                rows.append(row)
                print(row, flush=True)

    result_csv = outdir / "results.csv"
    fieldnames = sorted({k for row in rows for k in row})
    with result_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = choose_best(rows)
    (outdir / "best_stage_layouts.json").write_text(
        json.dumps({"stage_layouts": best, "results_csv": str(result_csv)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("results:", result_csv, flush=True)
    print("best:", outdir / "best_stage_layouts.json", flush=True)


if __name__ == "__main__":
    main()
