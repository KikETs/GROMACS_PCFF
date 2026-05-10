#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from run_polygen_same_state_probe import (
    ATM_TO_BAR,
    GMX,
    KCAL_TO_KJ,
    OUT_ROOT,
    gro_volume_nm3,
    parse_energy_xvg,
    replace_mdp_value,
)


REPO = Path(__file__).resolve().parents[2]
GMX_CPU_WORK = OUT_ROOT / "gromacs_cpu_openmp"
LAMMPS_LOG = OUT_ROOT / "lammps_openmp/equil_from_em.stdout.log"
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def run(
    cmd: list[str | Path],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        env=env,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def run_live_to_file(
    cmd: list[str | Path],
    cwd: Path,
    stdout_path: Path,
    *,
    env: dict[str, str] | None = None,
    label: str = "run",
) -> int:
    heartbeat_s = max(0.0, float(os.environ.get("PCFF_EQ01_PROBE_HEARTBEAT_S", "10")))
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while proc.poll() is None:
            if heartbeat_s > 0:
                print(f"{label}: running pid={proc.pid} elapsed_s={time.monotonic() - started:.1f}", flush=True)
                time.sleep(heartbeat_s)
            else:
                time.sleep(0.25)
        return proc.returncode


def parse_lammps_eq01_rows(log_path: Path) -> list[dict[str, float]]:
    cmd_re = re.compile(r"\$ .* -in .*lammps_equil_01_eq01_nvt_0p5fs_50ps_chunk0001\.in")
    in_eq01 = False
    header: list[str] | None = None
    rows: list[dict[str, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if cmd_re.search(line):
            in_eq01 = True
            header = None
            rows = []
            continue
        if not in_eq01:
            continue
        if "Loop time of" in line:
            break
        fields = line.split()
        if fields and fields[0] == "Step" and "v_time" in fields and "PotEng" in fields:
            header = fields
            continue
        if header is None:
            continue
        vals = FLOAT_RE.findall(line)
        if len(vals) == len(header):
            rows.append({key: float(value) for key, value in zip(header, vals)})
    if not rows:
        raise RuntimeError(f"No eq01 LAMMPS thermo rows found in {log_path}")
    return rows


def summarize_lammps_eq01(rows: list[dict[str, float]], target_ps: float) -> dict[str, float]:
    start_fs = rows[0]["v_time"]
    target_fs = start_fs + target_ps * 1000.0
    row = min(rows, key=lambda r: abs(r["v_time"] - target_fs))

    def avg(key: str) -> float:
        selected = [r[key] for r in rows if r["v_time"] <= row["v_time"]]
        return sum(selected) / len(selected)

    return {
        "lammps_step": row["Step"],
        "lammps_elapsed_ps": (row["v_time"] - start_fs) / 1000.0,
        "lammps_time_fs": row["v_time"],
        "lammps_pressure_bar": row["Press"] * ATM_TO_BAR,
        "lammps_temperature_k": row["Temp"],
        "lammps_potential_kj_mol": row["PotEng"] * KCAL_TO_KJ,
        "lammps_kinetic_kj_mol": row["KinEng"] * KCAL_TO_KJ,
        "lammps_total_kj_mol": row["TotEng"] * KCAL_TO_KJ,
        "lammps_pressure_mean_bar": avg("Press") * ATM_TO_BAR,
        "lammps_temperature_mean_k": avg("Temp"),
        "lammps_potential_mean_kj_mol": avg("PotEng") * KCAL_TO_KJ,
        "lammps_kinetic_mean_kj_mol": avg("KinEng") * KCAL_TO_KJ,
        "lammps_total_mean_kj_mol": avg("TotEng") * KCAL_TO_KJ,
        "lammps_sample_count_used": sum(1 for r in rows if r["v_time"] <= row["v_time"]),
    }


def make_eq01_mdp(
    base_mdp: Path,
    out_mdp: Path,
    target_ps: float,
    nstlist: int,
    sample_ps: float,
    continuation: str,
) -> None:
    text = base_mdp.read_text(encoding="utf-8", errors="ignore")
    base_steps = int(round(target_ps / 0.000125))
    sample_steps = max(4, int(round(sample_ps / 0.000125)))
    for key, value in {
        "nsteps": str(base_steps),
        "nstlist": str(nstlist),
        "nstcalcenergy": str(sample_steps),
        "nstenergy": str(sample_steps),
        "nstlog": str(sample_steps),
        "nstxout": "0",
        "nstvout": "0",
        "nstfout": "0",
        "nstxout-compressed": "0",
        "continuation": continuation,
        "gen-vel": "no",
        "pcoupl": "no",
    }.items():
        text = replace_mdp_value(text, key, value)
    out_mdp.write_text(text, encoding="utf-8")


def run_gmx_eq01(
    root: Path,
    target_ps: float,
    sample_ps: float,
    ntomp: int,
    nstlist: int,
    mass_mode: str,
    nhc_integrator: str,
    pre_trotter: str,
    post_trotter: str,
    continuation: str,
    reprod: bool,
) -> dict[str, float | int | str]:
    nhc_label = nhc_integrator or "default"
    trotter_label = f"pre{pre_trotter}_post{post_trotter}"
    reprod_label = "reprod" if reprod else "fast"
    work = root / f"gmx_eq01_{target_ps:g}ps_nstlist{nstlist}_{mass_mode}_nhc{nhc_label}_{trotter_label}_{reprod_label}"
    work.mkdir(parents=True, exist_ok=True)
    mdp = work / "eq01_probe.mdp"
    tpr = work / "eq01_probe.tpr"
    deffnm = work / "eq01_probe"
    make_eq01_mdp(GMX_CPU_WORK / "01_eq01_nvt_50ps.mdp", mdp, target_ps, nstlist, sample_ps, continuation)
    shutil.copy2(GMX_CPU_WORK / "01_eq01_nvt_50ps.lammps_velocity.gro", work / "system.gro")
    shutil.copy2(GMX_CPU_WORK / "topol.top", work / "topol.top")
    grompp = run([GMX, "grompp", "-f", mdp, "-c", "system.gro", "-p", "topol.top", "-o", tpr, "-maxwarn", "5"], cwd=work)
    (work / "grompp.stdout.log").write_text(grompp.stdout, encoding="utf-8")
    if grompp.returncode != 0:
        return {"status": "grompp_failed", "returncode": grompp.returncode}

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(ntomp),
            "OMP_PROC_BIND": "true",
            "OMP_PLACES": "cores",
            "GMX_PCFF_MTTK_MASS_MODE": mass_mode,
            "GMX_PCFF_MTTK_LAMMPS_NATOMS": "7075",
            "GMX_PCFF_EWALD_BETA_INV_A": "0.21160096",
            "GMX_PCFF_EWALD_REAL_ONLY": "1",
            "GMX_PCFF_EXACT_RESPA_NBNXM_OWNER_STEP_SCALAR_FALLBACK": "0",
            "GMX_PCFF_EXACT_RESPA_FUSED_INITIAL_DRIFT": "1",
            "GMX_PCFF_EXACT_RESPA_PRE_TROTTER": pre_trotter,
            "GMX_PCFF_EXACT_RESPA_POST_TROTTER": post_trotter,
        }
    )
    if nhc_integrator:
        env["GMX_PCFF_NHC_INTEGRATOR"] = nhc_integrator
    mdrun_cmd: list[str | Path] = [
        "taskset",
        "-c",
        "0-11",
        GMX,
        "mdrun",
        "-s",
        tpr,
        "-deffnm",
        deffnm,
        "-ntmpi",
        "1",
        "-ntomp",
        str(ntomp),
        "-pin",
        "off",
        "-dlb",
        "no",
        "-notunepme",
        "-update",
        "cpu",
        "-nb",
        "cpu",
        "-bonded",
        "cpu",
    ]
    if reprod:
        mdrun_cmd.append("-reprod")
    returncode = run_live_to_file(
        mdrun_cmd,
        cwd=work,
        env=env,
        stdout_path=work / "mdrun.stdout.log",
        label=work.name,
    )
    out: dict[str, float | int | str] = {
        "status": "ok" if returncode == 0 else "mdrun_failed",
        "returncode": returncode,
        "workdir": str(work),
    }
    if returncode != 0:
        return out
    terms = "\n".join(["Potential", "Kinetic En.", "Total Energy", "Temperature", "Pressure", "Volume", "Density", "0", ""])
    energy = run([GMX, "energy", "-f", deffnm.with_suffix(".edr"), "-o", "selected.xvg"], cwd=work, stdin=terms)
    (work / "energy.stdout.log").write_text(energy.stdout, encoding="utf-8")
    out.update(parse_energy_xvg(work / "selected.xvg"))
    if deffnm.with_suffix(".gro").exists():
        out["gro_volume_nm3"] = gro_volume_nm3(deffnm.with_suffix(".gro"))
    return out


def add_deltas(row: dict[str, float | int | str]) -> None:
    pairs = {
        "Potential": "lammps_potential_kj_mol",
        "Kinetic En.": "lammps_kinetic_kj_mol",
        "Total Energy": "lammps_total_kj_mol",
        "Temperature": "lammps_temperature_k",
        "Pressure": "lammps_pressure_bar",
        "Potential_mean": "lammps_potential_mean_kj_mol",
        "Kinetic En._mean": "lammps_kinetic_mean_kj_mol",
        "Total Energy_mean": "lammps_total_mean_kj_mol",
        "Temperature_mean": "lammps_temperature_mean_k",
        "Pressure_mean": "lammps_pressure_mean_bar",
    }
    for gmx_key, lmp_key in pairs.items():
        if gmx_key in row and lmp_key in row:
            safe = gmx_key.lower().replace(" ", "_").replace(".", "")
            row[f"{safe}_delta"] = float(row[gmx_key]) - float(row[lmp_key])


def write_summary(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO / "output/probes/eq01_nvt_probe_20260503")
    parser.add_argument("--target-ps", type=float, default=5.0)
    parser.add_argument("--sample-ps", type=float, default=5.0)
    parser.add_argument("--ntomp", type=int, default=12)
    parser.add_argument("--nstlist", type=int, nargs="+", default=[80, 1])
    parser.add_argument("--mass-mode", nargs="+", default=["lammps_tchain"])
    parser.add_argument("--nhc-integrator", choices=["", "lammps"], nargs="+", default=["lammps"])
    parser.add_argument("--pre-trotter", choices=["none", "two", "three", "two-three", "three-two"], nargs="+", default=["two"])
    parser.add_argument("--post-trotter", choices=["none", "two", "three", "two-three", "three-two"], nargs="+", default=["three"])
    parser.add_argument("--continuation", choices=["yes", "no"], default="no")
    parser.add_argument("--reprod", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lammps_rows = parse_lammps_eq01_rows(LAMMPS_LOG)
    lammps_ref = summarize_lammps_eq01(lammps_rows, args.target_ps)
    rows: list[dict[str, float | int | str]] = []
    summary = root / "eq01_nvt_probe_summary.csv"
    for nstlist in args.nstlist:
        for mass_mode in args.mass_mode:
            for nhc_integrator in args.nhc_integrator:
                for pre_trotter in args.pre_trotter:
                    for post_trotter in args.post_trotter:
                        row: dict[str, float | int | str] = {
                            "target_ps": args.target_ps,
                            "sample_ps": args.sample_ps,
                            "nstlist": nstlist,
                            "mass_mode": mass_mode,
                            "nhc_integrator": nhc_integrator or "default",
                            "pre_trotter": pre_trotter,
                            "post_trotter": post_trotter,
                            "continuation": args.continuation,
                            "reprod": int(args.reprod),
                        }
                        row.update(lammps_ref)
                        row.update(
                            run_gmx_eq01(
                                root,
                                args.target_ps,
                                args.sample_ps,
                                args.ntomp,
                                nstlist,
                                mass_mode,
                                nhc_integrator,
                                pre_trotter,
                                post_trotter,
                                args.continuation,
                                args.reprod,
                            )
                        )
                        add_deltas(row)
                        rows.append(row)
                        write_summary(summary, rows)
                        print(row, flush=True)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
