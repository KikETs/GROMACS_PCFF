#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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
LAMMPS_WORK = OUT_ROOT / "lammps_openmp"
GMX_STRICT_WORK = OUT_ROOT / "gromacs_gpu_hybrid_strict_pme5"

DEFAULT_LMP = Path(os.environ.get("LMP_BIN", "lmp"))
DEFAULT_GMX_CPU = REPO / "build-znver4" / "bin" / "gmx"
DEFAULT_GMX_GPU = REPO / "build_gateb_cuda" / "bin" / "gmx"

ATM_TO_BAR = 1.01325
KCAL_TO_KJ = 4.184
ANG3_TO_NM3 = 1.0e-3

LAMMPS_LOOP_RE = re.compile(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs for\s+(\d+)\s+steps")


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Short strict production physical-audit probe for PolyGen Example production.in style "
            "r-RESPA. This is intentionally a short audit, not a transport/statistical production gate."
        )
    )
    parser.add_argument("--out", type=Path, default=REPO / "output" / "strict_prod_parity_audits" / f"polygen_strict_prod_{stamp}")
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--duration-ps", type=float, default=20.0)
    parser.add_argument("--prod-dt-fs", type=float, default=2.0)
    parser.add_argument("--gmx-base-dt-fs", type=float, default=0.5)
    parser.add_argument("--energy-interval-ps", type=float, default=1.0)
    parser.add_argument("--trajectory-interval-ps", type=float, default=2.0)
    parser.add_argument("--temperature-k", type=float, default=353.0)
    parser.add_argument("--lmp", type=Path, default=DEFAULT_LMP)
    parser.add_argument("--gmx-cpu", type=Path, default=DEFAULT_GMX_CPU)
    parser.add_argument("--gmx-gpu", type=Path, default=DEFAULT_GMX_GPU)
    parser.add_argument("--energy-reader", type=Path, default=DEFAULT_GMX_CPU)
    parser.add_argument("--gmx-work", type=Path, default=GMX_STRICT_WORK)
    parser.add_argument("--gmx-start", default="14_prod01_nvt_10000ps_chunk0001.lammps_relaxed_box.g96")
    parser.add_argument("--gmx-template-mdp", default="14_prod01_nvt_10000ps_chunk0001.mdp")
    parser.add_argument("--lammps-ntomp", type=int, default=16)
    parser.add_argument("--gmx-cpu-ntomp", type=int, default=20)
    parser.add_argument("--gmx-gpu-ntomp", type=int, default=12)
    parser.add_argument("--lammps-omp-proc-bind", default="close")
    parser.add_argument("--lammps-omp-places", default="threads")
    parser.add_argument("--gmx-cpu-omp-proc-bind", default="close")
    parser.add_argument("--gmx-cpu-omp-places", default="threads")
    parser.add_argument("--gmx-gpu-omp-proc-bind", default="close")
    parser.add_argument("--gmx-gpu-omp-places", default="threads")
    parser.add_argument(
        "--gmx-gpu-mode",
        default="nb_gpu_pme_cpu_bonded_all",
        choices=[
            "nb_gpu_pme_cpu_bonded_cpu",
            "nb_gpu_pme_cpu_bonded_pair14",
            "nb_gpu_pme_cpu_bonded_class2_pair14",
            "nb_gpu_pme_cpu_bonded_all",
        ],
        help=(
            "Strict PME5 GPU mode. bonded_all is the current fastest direct physical-audit mode."
        ),
    )
    parser.add_argument(
        "--gmx-gpu-direct-energy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use direct GPU strict energy/pressure output instead of force-only GPU plus CPU rerun.",
    )
    parser.add_argument("--cpuset", default="0-23")
    parser.add_argument("--run-lammps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-gmx-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-gmx-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-sec", type=float, default=900.0)
    return parser.parse_args()


def omp_bind_env(*, num_threads: int, proc_bind: str, places: str) -> dict[str, str]:
    return {
        "OMP_NUM_THREADS": str(num_threads),
        "OMP_PROC_BIND": str(proc_bind),
        "OMP_PLACES": str(places),
    }


def run_command(
    cmd: list[str | Path],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Path,
    timeout_sec: float,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [str(x) for x in cmd],
            cwd=cwd,
            env=env,
            input=input_text,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        captured = exc.stdout or ""
        if isinstance(captured, bytes):
            captured = captured.decode("utf-8", errors="replace")
        log.write_text("$ " + " ".join(str(x) for x in cmd) + f"\nTIMEOUT after {elapsed:.3f} s\n" + captured)
        raise
    elapsed = time.monotonic() - started
    log.write_text(
        "$ "
        + " ".join(str(x) for x in cmd)
        + f"\n# elapsed_s={elapsed:.6f}\n"
        + completed.stdout,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with code {completed.returncode}: {' '.join(str(x) for x in cmd)}")
    return completed


def remove_mdp_key(path: Path, key: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    filtered = [
        raw
        for raw in lines
        if not (raw.strip().startswith(f"{key} ") or raw.strip().startswith(f"{key}="))
    ]
    path.write_text("\n".join(filtered) + "\n", encoding="utf-8")


def lammps_input(*, restart: Path, nsteps: int, dump_every: int, thermo_every: int, temperature_k: float, prod_dt_fs: float) -> str:
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
variable        sysvol      equal vol
variable        sysmass     equal mass(all)/6.0221367e+23
variable        sysdensity  equal v_sysmass/v_sysvol/1.0e-24
variable        tdamp       equal floor(1000*dt)
variable        pdamp       equal floor(10000*dt)

timestep        ${{prod_dt}}
run_style       respa 2 4
reset_timestep  0

thermo_style    custom step v_time press temp pe ke etotal vol v_sysdensity
thermo          {thermo_every}
thermo_modify   flush yes
fix             1 all nvt temp ${{tlo}} ${{tlo}} ${{tdamp}}
dump            1 all custom {dump_every} ${{dumpfile}} id mol type mass q x y z ix iy iz
dump_modify     1 sort id

run             {nsteps}
undump          1
unfix           1
write_restart   final.restart
"""


def parse_lammps_thermo(log_path: Path, prod_dt_fs: float) -> tuple[dict[str, Any], dict[str, list[float]]]:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    columns: list[str] | None = None
    samples: dict[str, list[float]] = {}
    for raw in lines:
        parts = raw.split()
        if not parts:
            continue
        if parts[:3] == ["Step", "v_time", "Press"]:
            columns = parts
            samples = {name: [] for name in columns}
            continue
        if columns is None or len(parts) != len(columns):
            continue
        try:
            values = [float(x) for x in parts]
        except ValueError:
            continue
        for name, value in zip(columns, values, strict=True):
            samples[name].append(value)
    if not samples or not samples.get("Step"):
        raise RuntimeError(f"No LAMMPS thermo table parsed from {log_path}")

    loops = []
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for match in LAMMPS_LOOP_RE.finditer(text):
        seconds = float(match.group(1))
        procs = int(match.group(2))
        steps = int(match.group(3))
        ns_per_day = steps * prod_dt_fs / 1_000_000.0 / seconds * 86400.0 if seconds > 0 else None
        loops.append({"seconds": seconds, "procs": procs, "steps": steps, "ns_per_day": ns_per_day})

    metrics = {
        "potential_kj_mol": [v * KCAL_TO_KJ for v in samples["PotEng"]],
        "kinetic_kj_mol": [v * KCAL_TO_KJ for v in samples["KinEng"]],
        "total_energy_kj_mol": [v * KCAL_TO_KJ for v in samples["TotEng"]],
        "temperature_k": samples["Temp"],
        "pressure_bar": [v * ATM_TO_BAR for v in samples["Press"]],
        "volume_nm3": [v * ANG3_TO_NM3 for v in samples["Volume"]],
        "density_g_cm3": samples["v_sysdensity"],
    }
    out: dict[str, Any] = {"sample_count": len(samples["Step"]), "speed": loops[-1] if loops else {}}
    for key, values in metrics.items():
        out[f"{key}_initial"] = values[0]
        out[f"{key}_final"] = values[-1]
        out[f"{key}_mean"] = sum(values) / len(values)
    return out, metrics


def parse_xvg(path: Path) -> tuple[list[str], list[list[float]]]:
    labels: list[str] = []
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("@") and "legend" in stripped:
            match = re.search(r'legend\s+"([^"]+)"', stripped)
            if match:
                labels.append(match.group(1))
            continue
        if stripped.startswith(("#", "@")):
            continue
        parts = stripped.split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    return labels, rows


def read_box_volume_nm3(path: Path) -> float:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if path.suffix.lower() == ".g96":
        in_box = False
        for raw in lines:
            if raw.strip() == "BOX":
                in_box = True
                continue
            if in_box and raw.strip() == "END":
                break
            if in_box:
                vals = [float(x) for x in raw.split()]
                if len(vals) >= 3:
                    return vals[0] * vals[1] * vals[2]
    vals = [float(x) for x in lines[-1].split()]
    if len(vals) >= 3:
        return vals[0] * vals[1] * vals[2]
    raise RuntimeError(f"Could not parse box from {path}")


def summarize_gromacs_energy(
    case: Path,
    *,
    deffnm: str = "prod_polygen_style",
    energy_reader: Path,
    env: dict[str, str],
    timeout_sec: float,
    mass_g_from_lammps: float | None,
) -> dict[str, Any]:
    edr = case / f"{deffnm}.edr"
    xvg = case / "energy_terms.xvg"
    selection = "Potential\nKinetic-En.\nTotal-Energy\nTemperature\nPressure\nPres-XX\nPres-YY\nPres-ZZ\n0\n"
    run_command(
        [energy_reader, "energy", "-f", edr.name, "-o", xvg.name],
        cwd=case,
        env=env,
        log=case / "energy.stdout.log",
        timeout_sec=timeout_sec,
        input_text=selection,
    )
    labels, rows = parse_xvg(xvg)
    if not rows:
        raise RuntimeError(f"No GROMACS energy rows parsed from {xvg}")
    label_to_index = {label: i + 1 for i, label in enumerate(labels)}
    mapping = {
        "potential_kj_mol": "Potential",
        "kinetic_kj_mol": "Kinetic En.",
        "total_energy_kj_mol": "Total Energy",
        "temperature_k": "Temperature",
        "pressure_bar": "Pressure",
    }
    out: dict[str, Any] = {"sample_count": len(rows), "energy_labels": labels}
    for metric, label in mapping.items():
        idx = label_to_index.get(label)
        if idx is None:
            out[f"{metric}_initial"] = None
            out[f"{metric}_final"] = None
            out[f"{metric}_mean"] = None
            continue
        values = [row[idx] for row in rows]
        out[f"{metric}_initial"] = values[0]
        out[f"{metric}_final"] = values[-1]
        out[f"{metric}_mean"] = sum(values) / len(values)
    gro = case / "prod_polygen_style.gro"
    start = case / "start.g96"
    volume_nm3 = read_box_volume_nm3(gro if gro.exists() else start)
    out["volume_nm3_initial"] = volume_nm3
    out["volume_nm3_final"] = volume_nm3
    out["volume_nm3_mean"] = volume_nm3
    if mass_g_from_lammps is not None:
        density = mass_g_from_lammps / (volume_nm3 * 1.0e-21)
        out["density_g_cm3_initial"] = density
        out["density_g_cm3_final"] = density
        out["density_g_cm3_mean"] = density
    return out


def prepare_gmx_mdp(
    template: Path,
    out: Path,
    *,
    duration_ps: float,
    base_dt_fs: float,
    energy_interval_ps: float,
    trajectory_interval_ps: float,
    suppress_energy: bool = False,
    write_trr: bool = False,
) -> None:
    shutil.copy2(template, out)
    remove_mdp_key(out, "exact-respa-level3-factor")
    nsteps = int(round(duration_ps * 1000.0 / base_dt_fs))
    if suppress_energy:
        # GPU exact-rRESPA narrow mode currently asserts when pressure/virial
        # output is requested. Keep the GPU trajectory run force-only and
        # compute physical metrics with a CPU rerun from the produced TRR.
        energy_stride = max(4, nsteps * 2)
    else:
        energy_stride = max(4, int(round(energy_interval_ps * 1000.0 / base_dt_fs)))
        energy_stride += (-energy_stride) % 4
    traj_stride = max(4, int(round(trajectory_interval_ps * 1000.0 / base_dt_fs)))
    traj_stride += (-traj_stride) % 4
    for key, value in {
        "dt": f"{base_dt_fs / 1000.0:.9f}",
        "nsteps": str(nsteps),
        "exact-respa": "yes",
        "exact-respa-levels": "2",
        "exact-respa-level2-factor": "4",
        "exact-respa-bond-level": "1",
        "exact-respa-angle-level": "1",
        "exact-respa-dihedral-level": "1",
        "exact-respa-improper-level": "1",
        "exact-respa-pair14-level": "2",
        "exact-respa-pair-level": "2",
        "exact-respa-kspace-level": "2",
        "pme-order": "5",
        "tau-t": "2.000000",
        "nsttcouple": "4",
        "nstcalcenergy": str(energy_stride),
        "nstenergy": str(energy_stride),
        "nstlog": str(energy_stride),
        "nstxout": str(traj_stride if write_trr else 0),
        "nstvout": str(traj_stride if write_trr else 0),
        "nstfout": "0",
        "nstxout-compressed": str(traj_stride),
    }.items():
        force_mdp_key(out, key, value)


def prepare_gmx_rerun_mdp(source_mdp: Path, out: Path, *, base_dt_fs: float) -> None:
    shutil.copy2(source_mdp, out)
    for key, value in {
        "exact-respa": "no",
        "nsteps": "0",
        "nstcalcenergy": "4",
        "nstenergy": "4",
        "nstlog": "4",
        "nstxout": "0",
        "nstvout": "0",
        "nstfout": "0",
        "nstxout-compressed": "0",
        "dt": f"{base_dt_fs / 1000.0:.9f}",
    }.items():
        force_mdp_key(out, key, value)


def write_comparison(out: Path, summaries: dict[str, dict[str, Any]]) -> None:
    reference = summaries.get("lammps_cpu")
    rows: list[dict[str, Any]] = []
    metrics = [
        "volume_nm3",
        "density_g_cm3",
        "temperature_k",
        "pressure_bar",
        "potential_kj_mol",
        "kinetic_kj_mol",
        "total_energy_kj_mol",
    ]
    stats = ["initial", "final", "mean"]
    for lane, summary in summaries.items():
        if lane == "lammps_cpu":
            continue
        for metric in metrics:
            for stat in stats:
                key = f"{metric}_{stat}"
                gmx_value = summary.get(key)
                lmp_value = reference.get(key) if reference else None
                delta = None if gmx_value is None or lmp_value is None else gmx_value - lmp_value
                pct = None
                if delta is not None and lmp_value not in (None, 0):
                    pct = 100.0 * delta / abs(lmp_value)
                rows.append(
                    {
                        "lane": lane,
                        "metric": metric,
                        "stat": stat,
                        "lammps_value": lmp_value,
                        "gromacs_value": gmx_value,
                        "delta_gromacs_minus_lammps": delta,
                        "pct_delta_vs_abs_lammps": pct,
                    }
                )
    with (out / "metric_comparison.csv").open("w", newline="") as handle:
        fields = ["lane", "metric", "stat", "lammps_value", "gromacs_value", "delta_gromacs_minus_lammps", "pct_delta_vs_abs_lammps"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_lammps(args: argparse.Namespace) -> dict[str, Any]:
    restart = (LAMMPS_WORK / "relaxed.restart").resolve()
    if not restart.exists():
        raise FileNotFoundError(restart)
    case = args.out / "lammps_cpu"
    if case.exists():
        shutil.rmtree(case)
    case.mkdir(parents=True)
    nsteps = int(round(args.duration_ps * 1000.0 / args.prod_dt_fs))
    dump_every = max(1, int(round(args.trajectory_interval_ps * 1000.0 / args.prod_dt_fs)))
    thermo_every = max(1, int(round(args.energy_interval_ps * 1000.0 / args.prod_dt_fs)))
    script = case / "production_strict_audit.in"
    script.write_text(
        lammps_input(
            restart=restart,
            nsteps=nsteps,
            dump_every=dump_every,
            thermo_every=thermo_every,
            temperature_k=args.temperature_k,
            prod_dt_fs=args.prod_dt_fs,
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        omp_bind_env(
            num_threads=args.lammps_ntomp,
            proc_bind=args.lammps_omp_proc_bind,
            places=args.lammps_omp_places,
        )
    )
    cmd: list[str | Path] = [args.lmp, "-nonbuf", "-sf", "omp", "-pk", "omp", str(args.lammps_ntomp), "-in", script.name]
    if args.cpuset:
        cmd = ["taskset", "-c", args.cpuset, *cmd]
    run_command(cmd, cwd=case, env=env, log=case / "stdout.log", timeout_sec=args.timeout_sec)
    summary, _ = parse_lammps_thermo(case / "log.lammps", args.prod_dt_fs)
    summary.update(
        {
            "case_dir": str(case),
            "ntomp": args.lammps_ntomp,
            "cpuset": args.cpuset,
            "omp_proc_bind": args.lammps_omp_proc_bind,
            "omp_places": args.lammps_omp_places,
        }
    )
    return summary


def run_gromacs(
    args: argparse.Namespace,
    *,
    lane: str,
    gmx: Path,
    ntomp: int,
    omp_proc_bind: str,
    omp_places: str,
    mdrun_extra: list[str],
    mode_env: dict[str, str],
    config: dict[str, Any],
    mass_g_from_lammps: float | None,
) -> dict[str, Any]:
    source_work = args.gmx_work.resolve()
    start = source_work / args.gmx_start
    top = source_work / "topol.top"
    template = source_work / args.gmx_template_mdp
    for path in (gmx, start, top, template):
        if not path.exists():
            raise FileNotFoundError(path)
    case = args.out / lane
    if case.exists():
        shutil.rmtree(case)
    case.mkdir(parents=True)
    shutil.copy2(start, case / "start.g96")
    shutil.copy2(top, case / "topol.top")
    mdp = case / "prod_polygen_style.mdp"
    use_cpu_rerun_energy = lane == "gromacs_gpu_strict" and not args.gmx_gpu_direct_energy
    prepare_gmx_mdp(
        template,
        mdp,
        duration_ps=args.duration_ps,
        base_dt_fs=args.gmx_base_dt_fs,
        energy_interval_ps=args.energy_interval_ps,
        trajectory_interval_ps=args.trajectory_interval_ps,
        suppress_energy=use_cpu_rerun_energy,
        write_trr=use_cpu_rerun_energy,
    )

    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    env["GMX_BINARY_NAME"] = gmx.name
    env.update(omp_bind_env(num_threads=ntomp, proc_bind=omp_proc_bind, places=omp_places))
    env.update(mode_env)
    env = stage_runtime_env(config, env)

    run_command(
        [gmx, "grompp", "-f", mdp.name, "-c", "start.g96", "-p", "topol.top", "-o", "prod_polygen_style.tpr", "-maxwarn", "5"],
        cwd=case,
        env=env,
        log=case / "grompp.stdout.log",
        timeout_sec=args.timeout_sec,
    )
    cmd: list[str | Path] = [
        gmx,
        "mdrun",
        "-s",
        "prod_polygen_style.tpr",
        "-deffnm",
        "prod_polygen_style",
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
    run_command(cmd, cwd=case, env=env, log=case / "mdrun.stdout.log", timeout_sec=args.timeout_sec)
    deffnm_for_energy = "prod_polygen_style"
    if use_cpu_rerun_energy:
        trr = case / "prod_polygen_style.trr"
        if not trr.exists() or trr.stat().st_size == 0:
            raise RuntimeError(f"GPU strict trajectory missing for CPU rerun: {trr}")
        rerun_mdp = case / "rerun_energy.mdp"
        prepare_gmx_rerun_mdp(mdp, rerun_mdp, base_dt_fs=args.gmx_base_dt_fs)
        rerun_env = os.environ.copy()
        rerun_env["GMX_MAXBACKUP"] = "-1"
        rerun_env["GMX_BINARY_NAME"] = args.energy_reader.resolve().name
        rerun_env["OMP_NUM_THREADS"] = str(args.gmx_cpu_ntomp)
        rerun_env.update({"OMP_PROC_BIND": "close", "OMP_PLACES": "threads"})
        rerun_env = stage_runtime_env(config, rerun_env)
        run_command(
            [
                args.energy_reader.resolve(),
                "grompp",
                "-f",
                rerun_mdp.name,
                "-c",
                "start.g96",
                "-p",
                "topol.top",
                "-o",
                "rerun_energy.tpr",
                "-maxwarn",
                "20",
            ],
            cwd=case,
            env=rerun_env,
            log=case / "rerun_grompp.stdout.log",
            timeout_sec=args.timeout_sec,
        )
        rerun_cmd: list[str | Path] = [
            args.energy_reader.resolve(),
            "mdrun",
            "-s",
            "rerun_energy.tpr",
            "-rerun",
            trr.name,
            "-deffnm",
            "rerun_energy",
            "-ntmpi",
            "1",
            "-ntomp",
            str(args.gmx_cpu_ntomp),
            "-pin",
            "off",
            "-nb",
            "cpu",
            "-pme",
            "cpu",
            "-bonded",
            "cpu",
            "-update",
            "cpu",
        ]
        if args.cpuset:
            rerun_cmd = ["taskset", "-c", args.cpuset, *rerun_cmd]
        run_command(
            rerun_cmd,
            cwd=case,
            env=rerun_env,
            log=case / "rerun_mdrun.stdout.log",
            timeout_sec=args.timeout_sec,
        )
        deffnm_for_energy = "rerun_energy"
    summary = summarize_gromacs_energy(
        case,
        deffnm=deffnm_for_energy,
        energy_reader=args.energy_reader.resolve(),
        env=env,
        timeout_sec=args.timeout_sec,
        mass_g_from_lammps=mass_g_from_lammps,
    )
    speed = parse_mdrun_log(case / "prod_polygen_style.log")
    summary.update(
        {
            "case_dir": str(case),
            "ntomp": ntomp,
            "cpuset": args.cpuset,
            "omp_proc_bind": omp_proc_bind,
            "omp_places": omp_places,
            "speed": speed,
            "mdrun_extra": [str(item) for item in mdrun_extra],
            "gpu_direct_energy": lane != "gromacs_gpu_strict" or args.gmx_gpu_direct_energy,
            "gmx_gpu_mode": args.gmx_gpu_mode if lane == "gromacs_gpu_strict" else None,
        }
    )
    return summary


def main() -> int:
    args = parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    config = load_notebook_config(args.notebook.resolve())
    manifest = {
        "claim_boundary": "Short strict production physical audit. It checks 20 ps-scale thermodynamic consistency only; it is not a full equilibration, production, or transport-ready gate.",
        "polygen_reference": str(REPO.parent / "PolyGen" / "Example-simulation-files" / "production" / "production.in"),
        "matched_settings": {
            "run_style": "LAMMPS respa 2 4",
            "gmx_exact_respa_levels": 2,
            "gmx_level2_factor": 4,
            "pme_order": 5,
            "tdamp_ps": 2.0,
            "trajectory_interval_ps": args.trajectory_interval_ps,
            "energy_interval_ps": args.energy_interval_ps,
            "gmx_start": str((args.gmx_work / args.gmx_start).resolve()),
            "gmx_start_box": "LAMMPS relaxed.lmp box remapped g96 when the default start file is used",
        },
        "runtime_affinity_defaults": {
            "lammps": {
                "ntomp": args.lammps_ntomp,
                "cpuset": args.cpuset,
                "omp_proc_bind": args.lammps_omp_proc_bind,
                "omp_places": args.lammps_omp_places,
            },
            "gromacs_cpu": {
                "ntomp": args.gmx_cpu_ntomp,
                "cpuset": args.cpuset,
                "omp_proc_bind": args.gmx_cpu_omp_proc_bind,
                "omp_places": args.gmx_cpu_omp_places,
            },
            "gromacs_gpu": {
                "ntomp": args.gmx_gpu_ntomp,
                "cpuset": args.cpuset,
                "omp_proc_bind": args.gmx_gpu_omp_proc_bind,
                "omp_places": args.gmx_gpu_omp_places,
            },
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summaries: dict[str, dict[str, Any]] = {}
    mass_g_from_lammps: float | None = None
    if args.run_lammps:
        summaries["lammps_cpu"] = run_lammps(args)
        density = summaries["lammps_cpu"].get("density_g_cm3_initial")
        volume = summaries["lammps_cpu"].get("volume_nm3_initial")
        if density is not None and volume is not None:
            mass_g_from_lammps = density * volume * 1.0e-21
        print(f"LAMMPS strict prod audit ns/day={summaries['lammps_cpu'].get('speed', {}).get('ns_per_day')}", flush=True)

    if args.run_gmx_cpu:
        summaries["gromacs_cpu_strict"] = run_gromacs(
            args,
            lane="gromacs_cpu_strict",
            gmx=args.gmx_cpu.resolve(),
            ntomp=args.gmx_cpu_ntomp,
            omp_proc_bind=args.gmx_cpu_omp_proc_bind,
            omp_places=args.gmx_cpu_omp_places,
            mdrun_extra=["-nb", "cpu", "-pme", "cpu", "-bonded", "cpu", "-update", "cpu"],
            mode_env={},
            config=config,
            mass_g_from_lammps=mass_g_from_lammps,
        )
        print(f"GROMACS CPU strict prod audit ns/day={summaries['gromacs_cpu_strict'].get('speed', {}).get('ns_per_day')}", flush=True)

    if args.run_gmx_gpu:
        mdrun_extra, mode_env = mode_args_env(args.gmx_gpu_mode)
        summaries["gromacs_gpu_strict"] = run_gromacs(
            args,
            lane="gromacs_gpu_strict",
            gmx=args.gmx_gpu.resolve(),
            ntomp=args.gmx_gpu_ntomp,
            omp_proc_bind=args.gmx_gpu_omp_proc_bind,
            omp_places=args.gmx_gpu_omp_places,
            mdrun_extra=mdrun_extra,
            mode_env=mode_env,
            config=config,
            mass_g_from_lammps=mass_g_from_lammps,
        )
        print(f"GROMACS GPU strict prod audit ns/day={summaries['gromacs_gpu_strict'].get('speed', {}).get('ns_per_day')}", flush=True)

    (args.out / "metric_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if "lammps_cpu" in summaries:
        write_comparison(args.out, summaries)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
