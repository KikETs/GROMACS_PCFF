#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from polygen_stage_metric_audit import DEFAULT_NOTEBOOK, load_notebook_config
from sweep_polygen_gpu_offload_ntomp import (
    force_mdp_key,
    mode_args_env,
    parse_mdrun_log,
    stage_runtime_env,
)

OUT_ROOT = REPO / "output" / "polygen_pcff_gromacs_initial_em_notebook"
STRICT_WORK = OUT_ROOT / "gromacs_gpu_hybrid_strict_pme5"
LAMMPS_WORK = OUT_ROOT / "lammps_openmp"
DEFAULT_LMP = Path("/home/kiket/anaconda3/envs/MD/bin/lmp")
DEFAULT_GMX_CPU = REPO / "build-znver4" / "bin" / "gmx"
DEFAULT_GMX_GPU = REPO / "build_gateb_cuda" / "bin" / "gmx"

LAMMPS_LOOP_RE = re.compile(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs for\s+(\d+)\s+steps")


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Short production-style benchmark matching PolyGen BASE/production.in: "
            "LAMMPS run_style respa 3 2 2 ... pair 2 kspace 3, tdamp=floor(1000*dt), "
            "and trajectory output every 2000 outer LAMMPS steps / 4 ps."
        )
    )
    parser.add_argument("--out", type=Path, default=REPO / "output" / "speed_probes" / f"polygen_production_in_style_{stamp}")
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--duration-ps", type=float, default=20.0)
    parser.add_argument("--prod-dt-fs", type=float, default=2.0)
    parser.add_argument("--gmx-base-dt-fs", type=float, default=0.5)
    parser.add_argument("--temperature-k", type=float, default=353.0)
    parser.add_argument("--gmx-start-work", type=Path, default=STRICT_WORK)
    parser.add_argument("--gmx-start-stem", default="13_eq13_nvt_fixed_volume_1000ps_chunk0005")
    parser.add_argument("--gmx-template-mdp-stem", default="14_prod01_nvt_10000ps_chunk0001")
    parser.add_argument("--gmx-cpu", type=Path, default=DEFAULT_GMX_CPU)
    parser.add_argument("--gmx-gpu", type=Path, default=DEFAULT_GMX_GPU)
    parser.add_argument("--lmp", type=Path, default=DEFAULT_LMP)
    parser.add_argument("--cpuset", default="0-23")
    parser.add_argument("--gmx-cpu-ntomp-list", nargs="+", type=int, default=[12, 16, 20, 24])
    parser.add_argument("--gmx-gpu-ntomp-list", nargs="+", type=int, default=[4, 6, 8, 12, 16, 20])
    parser.add_argument("--lammps-ntomp-list", nargs="+", type=int, default=[12, 16, 20, 24])
    parser.add_argument(
        "--gmx-gpu-modes",
        nargs="+",
        default=["nb_gpu_pme_cpu_bonded_all", "nb_gpu_pme_gpu_bonded_all_order4"],
    )
    parser.add_argument("--run-lammps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-gmx-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-gmx-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    return parser.parse_args()


def run_command(cmd: list[str | Path], *, cwd: Path, env: dict[str, str], log: Path, timeout_sec: float) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [str(x) for x in cmd],
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        captured = exc.stdout or ""
        if isinstance(captured, bytes):
            captured = captured.decode("utf-8", errors="replace")
        log.write_text("$ " + " ".join(str(x) for x in cmd) + f"\nTIMEOUT after {elapsed:.3f} s\n" + captured)
        raise
    elapsed = time.monotonic() - started
    log.write_text("$ " + " ".join(str(x) for x in cmd) + "\n" + completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with code {completed.returncode}: {' '.join(str(x) for x in cmd)}")
    return elapsed


def lammps_input(*, restart: Path, nsteps: int, dump_every: int, temperature_k: float, prod_dt_fs: float) -> str:
    return f"""\
echo both
variable        tlo         equal {temperature_k:.9g}
variable        prod_dt     index {prod_dt_fs:.9g}
variable        dumpfile    index traj.lammpstrj

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

read_restart    {restart}

neighbor        3.0 bin
neigh_modify    delay 0 every 1 check yes
special_bonds   lj/coul 0.0 0.0 1.0

variable        time        equal step*dt+0.000001
variable        etotal      equal etotal
variable        pe          equal pe
variable        ke          equal ke
variable        tdamp       equal floor(1000*dt)
variable        pdamp       equal floor(10000*dt)
variable        nave        equal floor(10*500*1e3/dt)
variable        dnave       equal v_nave-1
variable        nskip       equal floor(10*200*1e3/dt)

timestep        ${{prod_dt}}
run_style       respa 3 2 2 bond 1 angle 1 dihedral 1 improper 1 pair 2 kspace 3
reset_timestep  0

thermo_style    custom step v_time press temp pe ke etotal
thermo          20000
fix             1 all nvt temp ${{tlo}} ${{tlo}} ${{tdamp}}
fix             2 all ave/time 500 1 500 v_time c_thermo_temp c_thermo_press v_pe v_ke v_etotal file nVT_instantaneous.txt off 1
fix             3 all ave/time 1 49999 50000 v_time c_thermo_temp c_thermo_press v_pe v_ke v_etotal file nVT_averages.txt off 1

dump            1 all custom {dump_every} ${{dumpfile}} id mol type mass q x y z ix iy iz
dump_modify     1 sort id
restart         100000 nvt.restart

run             {nsteps}
unfix           1
unfix           2
unfix           3
undump          1
write_restart   final.restart
"""


def parse_lammps_log(path: Path, prod_dt_fs: float) -> dict[str, Any]:
    text = path.read_text(errors="ignore")
    loops = []
    for match in LAMMPS_LOOP_RE.finditer(text):
        seconds = float(match.group(1))
        procs = int(match.group(2))
        steps = int(match.group(3))
        ns_per_day = steps * prod_dt_fs / 1_000_000.0 / seconds * 86400.0 if seconds > 0 else None
        loops.append({"seconds": seconds, "procs": procs, "steps": steps, "ns_per_day": ns_per_day})
    loop = loops[-1] if loops else {}
    return {
        "ns_per_day": loop.get("ns_per_day"),
        "seconds": loop.get("seconds"),
        "steps": loop.get("steps"),
        "procs": loop.get("procs"),
    }


def prepare_polygen_gmx_mdp(template: Path, out: Path, *, duration_ps: float, base_dt_fs: float, pme_order: int) -> None:
    shutil.copy2(template, out)
    nsteps = int(round(duration_ps * 1000.0 / base_dt_fs))
    xtc_stride = int(round(4.0 * 1000.0 / base_dt_fs))
    energy_stride = int(round(40.0 * 1000.0 / base_dt_fs))
    for key, value in {
        "dt": f"{base_dt_fs / 1000.0:.9f}",
        "nsteps": str(nsteps),
        "exact-respa-levels": "3",
        "exact-respa-level2-factor": "2",
        "exact-respa-level3-factor": "4",
        "exact-respa-bond-level": "1",
        "exact-respa-angle-level": "1",
        "exact-respa-dihedral-level": "1",
        "exact-respa-improper-level": "1",
        "exact-respa-pair14-level": "2",
        "exact-respa-pair-level": "2",
        "exact-respa-kspace-level": "3",
        "pme-order": str(pme_order),
        "tau-t": "2.000000",
        "nsttcouple": "4",
        "nstcalcenergy": str(energy_stride),
        "nstenergy": str(energy_stride),
        "nstlog": str(energy_stride),
        "nstxout": "0",
        "nstvout": "0",
        "nstfout": "0",
        "nstxout-compressed": str(xtc_stride),
    }.items():
        force_mdp_key(out, key, value)


def run_lammps_cases(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    restart = (LAMMPS_WORK / "relaxed.restart").resolve()
    if not restart.exists():
        raise FileNotFoundError(restart)
    nsteps = int(round(args.duration_ps * 1000.0 / args.prod_dt_fs))
    dump_every = int(round(4.0 * 1000.0 / args.prod_dt_fs))
    base_env = os.environ.copy()
    base_env.update({"OMP_PROC_BIND": "close", "OMP_PLACES": "threads"})
    for ntomp in args.lammps_ntomp_list:
        case = args.out / "lammps_cpu" / f"ntomp{ntomp:02d}"
        if case.exists():
            shutil.rmtree(case)
        case.mkdir(parents=True)
        script = case / "production_polygen_style.in"
        script.write_text(
            lammps_input(
                restart=restart,
                nsteps=nsteps,
                dump_every=dump_every,
                temperature_k=args.temperature_k,
                prod_dt_fs=args.prod_dt_fs,
            )
        )
        env = dict(base_env)
        env["OMP_NUM_THREADS"] = str(ntomp)
        cmd: list[str | Path] = [args.lmp, "-nonbuf", "-sf", "omp", "-pk", "omp", str(ntomp), "-in", script.name]
        if args.cpuset:
            cmd = ["taskset", "-c", args.cpuset, *cmd]
        row = {"engine": "lammps", "mode": "cpu_omp", "ntomp": ntomp, "case_dir": str(case)}
        try:
            elapsed = run_command(cmd, cwd=case, env=env, log=case / "stdout.log", timeout_sec=args.timeout_sec)
            metrics = parse_lammps_log(case / "log.lammps", args.prod_dt_fs)
            row.update(metrics)
            row["elapsed_s"] = elapsed
            row["error"] = None
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
        write_results(args.out, rows)
        print(f"LAMMPS ntomp={ntomp}: ns/day={row.get('ns_per_day')} error={row.get('error')}", flush=True)


def run_gmx_case(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    engine: str,
    mode: str,
    ntomp: int,
    gmx: Path,
    mdrun_extra: list[str],
    mode_env: dict[str, str],
    pme_order: int,
    config: dict[str, Any],
) -> None:
    work = args.gmx_start_work.resolve()
    top = work / "topol.top"
    template = work / f"{args.gmx_template_mdp_stem}.mdp"
    start_gro = work / f"{args.gmx_start_stem}.hi.gro"
    if not start_gro.exists():
        start_gro = work / f"{args.gmx_start_stem}.gro"
    for path in (gmx, top, template, start_gro):
        if not path.exists():
            raise FileNotFoundError(path)
    case = args.out / engine / mode / f"ntomp{ntomp:02d}"
    if case.exists():
        shutil.rmtree(case)
    case.mkdir(parents=True)
    shutil.copy2(top, case / "topol.top")
    shutil.copy2(start_gro, case / start_gro.name)
    mdp = case / "prod_polygen_style.mdp"
    prepare_polygen_gmx_mdp(template, mdp, duration_ps=args.duration_ps, base_dt_fs=args.gmx_base_dt_fs, pme_order=pme_order)

    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    env["GMX_BINARY_NAME"] = gmx.name
    env["OMP_NUM_THREADS"] = str(ntomp)
    env.update({"OMP_PROC_BIND": "close", "OMP_PLACES": "threads"})
    env.update(mode_env)
    env = stage_runtime_env(config, env)

    tpr = case / "prod_polygen_style.tpr"
    deffnm = case / "prod_polygen_style"
    row = {"engine": engine, "mode": mode, "ntomp": ntomp, "case_dir": str(case), "pme_order": pme_order}
    try:
        run_command(
            [gmx, "grompp", "-f", mdp.name, "-c", start_gro.name, "-p", "topol.top", "-o", tpr.name, "-maxwarn", "5"],
            cwd=case,
            env=env,
            log=case / "grompp.stdout.log",
            timeout_sec=args.timeout_sec,
        )
        cmd: list[str | Path] = [
            gmx,
            "mdrun",
            "-s",
            tpr.name,
            "-deffnm",
            deffnm.name,
            "-ntmpi",
            "1",
            "-ntomp",
            str(ntomp),
            "-pin",
            "off",
            "-dlb",
            "no",
            "-notunepme",
            *mdrun_extra,
        ]
        if args.cpuset:
            cmd = ["taskset", "-c", args.cpuset, *cmd]
        elapsed = run_command(cmd, cwd=case, env=env, log=case / "mdrun.stdout.log", timeout_sec=args.timeout_sec)
        metrics = parse_mdrun_log(deffnm.with_suffix(".log"))
        row.update(metrics)
        row["elapsed_s"] = elapsed
        row["error"] = None
    except Exception as exc:
        row["error"] = str(exc)
    rows.append(row)
    write_results(args.out, rows)
    print(f"{engine} {mode} ntomp={ntomp}: ns/day={row.get('ns_per_day')} error={row.get('error')}", flush=True)


def write_results(out: Path, rows: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: (r.get("ns_per_day") is not None, float(r.get("ns_per_day") or -1)), reverse=True)
    (out / "summary.json").write_text(json.dumps(sorted_rows, indent=2, sort_keys=True) + "\n")
    fields = [
        "engine",
        "mode",
        "ntomp",
        "pme_order",
        "ns_per_day",
        "ms_per_step",
        "elapsed_s",
        "seconds",
        "steps",
        "procs",
        "error",
        "case_dir",
    ]
    with (out / "summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    args = parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    config = load_notebook_config(args.notebook.resolve())
    manifest = {
        "claim_boundary": "Short production-style speed probe only; not a full equilibration or transport parity audit.",
        "polygen_production_in": "/home/kiket/Desktop/test/MY_PAPER_RELATED/LAMMPS_BATCH/BASE/production.in",
        "matched_settings": {
            "lammps_run_style": "respa 3 2 2 bond 1 angle 1 dihedral 1 improper 1 pair 2 kspace 3",
            "prod_dt_fs": args.prod_dt_fs,
            "gmx_exact_respa_levels": 3,
            "gmx_level2_factor": 2,
            "gmx_level3_factor": 4,
            "pair_level": 2,
            "kspace_level": 3,
            "tdamp_ps": 2.0,
            "trajectory_stride_ps": 4.0,
        },
        "duration_ps": args.duration_ps,
        "cpuset": args.cpuset,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    rows: list[dict[str, Any]] = []
    if args.run_lammps:
        run_lammps_cases(args, rows)
    if args.run_gmx_cpu:
        for ntomp in args.gmx_cpu_ntomp_list:
            run_gmx_case(
                args=args,
                rows=rows,
                engine="gromacs_cpu",
                mode="cpu_pme5",
                ntomp=ntomp,
                gmx=args.gmx_cpu.resolve(),
                mdrun_extra=["-nb", "cpu", "-pme", "cpu", "-bonded", "cpu", "-update", "cpu"],
                mode_env={},
                pme_order=5,
                config=config,
            )
    if args.run_gmx_gpu:
        for mode in args.gmx_gpu_modes:
            mdrun_extra, mode_env = mode_args_env(mode)
            pme_order = 4 if "order4" in mode else 5
            for ntomp in args.gmx_gpu_ntomp_list:
                run_gmx_case(
                    args=args,
                    rows=rows,
                    engine="gromacs_gpu",
                    mode=mode,
                    ntomp=ntomp,
                    gmx=args.gmx_gpu.resolve(),
                    mdrun_extra=mdrun_extra,
                    mode_env=mode_env,
                    pme_order=pme_order,
                    config=config,
                )
    write_results(args.out, rows)
    print(f"wrote {args.out / 'summary.tsv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
