#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from run_polygen_same_state_probe import (
    ATM_TO_BAR,
    BRIDGE_SCRIPT,
    GMX,
    KCAL_TO_KJ,
    LMP,
    OUT_ROOT,
    bridge_lammps_data,
    gro_volume_nm3,
    parse_energy_xvg,
    replace_mdp_value,
    write_gro_with_velocities,
    write_lammps_run0_input,
)


REPO = Path(__file__).resolve().parents[2]
LAMMPS_WORK = OUT_ROOT / "lammps_openmp"
GMX_CPU_WORK = OUT_ROOT / "gromacs_cpu_openmp"
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
    heartbeat_s = max(0.0, float(os.environ.get("PCFF_SHORT_PROP_HEARTBEAT_S", "5")))
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


def parse_lammps_rows(log_path: Path) -> list[dict[str, float]]:
    header: list[str] | None = None
    rows: list[dict[str, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Step" in line and "v_time" in line and "PotEng" in line:
            header = line.split()
            continue
        if header is None:
            continue
        vals = FLOAT_RE.findall(line)
        if len(vals) == len(header):
            rows.append({key: float(value) for key, value in zip(header, vals)})
    if not rows:
        raise RuntimeError(f"No LAMMPS thermo rows found in {log_path}")
    return rows


def parse_last_lammps_row(log_path: Path) -> dict[str, float]:
    rows = parse_lammps_rows(log_path)
    return rows[-1]


def summarize_lammps_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    row = rows[-1]

    def avg(key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)

    return {
        "lammps_step": row["Step"],
        "lammps_time_ps": row["v_time"] / 1000.0,
        "lammps_pressure_bar": row["Press"] * ATM_TO_BAR,
        "lammps_volume_nm3": row["Volume"] / 1000.0,
        "lammps_density_g_cm3": row["v_sysdensity"],
        "lammps_temperature_k": row["Temp"],
        "lammps_potential_kj_mol": row["PotEng"] * KCAL_TO_KJ,
        "lammps_kinetic_kj_mol": row["KinEng"] * KCAL_TO_KJ,
        "lammps_total_kj_mol": row["TotEng"] * KCAL_TO_KJ,
        "lammps_pressure_mean_bar": avg("Press") * ATM_TO_BAR,
        "lammps_volume_mean_nm3": avg("Volume") / 1000.0,
        "lammps_density_mean_g_cm3": avg("v_sysdensity"),
        "lammps_temperature_mean_k": avg("Temp"),
        "lammps_potential_mean_kj_mol": avg("PotEng") * KCAL_TO_KJ,
        "lammps_kinetic_mean_kj_mol": avg("KinEng") * KCAL_TO_KJ,
        "lammps_total_mean_kj_mol": avg("TotEng") * KCAL_TO_KJ,
        "lammps_sample_count": len(rows),
    }


def make_gmx_short_mdp(
    base_mdp: Path,
    out_mdp: Path,
    base_steps: int,
    nstlist: int,
    ensemble: str,
    sample_base_steps: int,
) -> None:
    text = base_mdp.read_text(encoding="utf-8", errors="ignore")
    stride = max(4, sample_base_steps)
    for key, value in {
        "nsteps": str(base_steps),
        "nstlist": str(nstlist),
        "nstcalcenergy": str(stride),
        "nstenergy": str(stride),
        "nstlog": str(stride),
        "nstxout": "0",
        "nstvout": "0",
        "nstfout": "0",
        "nstxout-compressed": "0",
        "continuation": "yes",
        "gen-vel": "no",
        "exact-respa-inner-level": "0",
        "exact-respa-middle-level": "0",
        "exact-respa-outer-level": "0",
        "exact-respa-inner-off": "0",
        "exact-respa-inner-on": "0",
        "exact-respa-outer-on": "0",
        "exact-respa-outer-off": "0",
    }.items():
        text = replace_mdp_value(text, key, value)
    if ensemble == "nve":
        for key, value in {
            "tcoupl": "no",
            "pcoupl": "no",
        }.items():
            text = replace_mdp_value(text, key, value)
    elif ensemble == "nvt":
        text = replace_mdp_value(text, "pcoupl", "no")
    out_mdp.write_text(text, encoding="utf-8")


def prepare_start(root: Path, lammps_ntomp: int) -> tuple[Path, Path]:
    restart = LAMMPS_WORK / ".resume_state/equil_01_eq01_nvt_0p5fs_50ps_chunk0001.restart"
    if not restart.exists():
        raise FileNotFoundError(restart)
    data_out = root / "eq01_endpoint_for_eq02.lmp"
    lmp_in = root / "lammps_eq01_write_data.in"
    write_lammps_run0_input(lmp_in, restart.resolve(), data_out, True)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": str(lammps_ntomp), "OMP_PROC_BIND": "true", "OMP_PLACES": "cores"})
    proc = run(
        [LMP, "-nonbuf", "-sf", "omp", "-pk", "omp", str(lammps_ntomp), "-in", lmp_in],
        cwd=root,
        env=env,
    )
    (root / "lammps_eq01_write_data.stdout.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"LAMMPS write_data failed with code {proc.returncode}")
    bridge_out = root / "bridge_eq01"
    bridge_lammps_data(data_out, bridge_out)
    gro = bridge_out / "system_with_velocities.gro"
    write_gro_with_velocities(bridge_out / "system.gro", data_out, gro)
    return gro, bridge_out / "topol.top"


def write_lammps_eq02_short_input(
    path: Path,
    restart: Path,
    outer_steps: int,
    full_outer_steps: int,
    ensemble: str,
    sample_outer_steps: int,
) -> None:
    start_step = 120380
    stop_step = start_step + full_outer_steps
    if ensemble == "nve":
        fix_line = "fix             1 all nve"
        fix_state_terms = ""
    elif ensemble == "nvt":
        fix_line = "fix             1 all nvt temp ${tlo} ${tlo} ${tdamp}"
        fix_state_terms = ""
    else:
        fix_line = "fix             1 all npt temp ${tlo} ${tlo} ${tdamp} iso ${plo} ${pmi} ${pdamp} drag 0 mtk yes nreset 20000"
        fix_state_terms = " " + " ".join(f"f_1[{i}]" for i in range(1, 15))
    path.write_text(
        f"""echo both
variable        tlo         equal 353
variable        plo         equal 1
variable        pmi         equal 500.5

units           real
boundary        p p p
atom_style      full

pair_style      lj/class2/coul/long 9.500000
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
kspace_modify   compute no

variable        sysmass     equal mass(all)/6.0221367e+23
variable        sysdensity  equal v_sysmass/vol/1.0e-24
variable        time        equal step*dt+0.000001
variable        tdamp       equal floor(100*dt)
variable        pdamp       equal floor(1000*dt)

compute         p_full all pressure thermo_temp
compute         p_vir all pressure NULL virial

timestep        0.5
run_style       respa 2 4
{fix_line}
thermo_style    custom step v_time press c_p_full[1] c_p_full[2] c_p_full[3] c_p_full[4] c_p_full[5] c_p_full[6] c_p_vir[1] c_p_vir[2] c_p_vir[3] c_p_vir[4] c_p_vir[5] c_p_vir[6] vol v_sysdensity temp pe ke etotal{fix_state_terms}
thermo          {max(1, sample_outer_steps)}
thermo_modify   flush yes
run             {outer_steps} start {start_step} stop {stop_step}
unfix           1
""",
        encoding="utf-8",
    )


def lammps_short(
    root: Path,
    outer_steps: int,
    full_outer_steps: int,
    lammps_ntomp: int,
    ensemble: str,
    sample_outer_steps: int,
) -> dict[str, float]:
    work = root / f"lammps_{ensemble}_outer{outer_steps}"
    work.mkdir(parents=True, exist_ok=True)
    restart = LAMMPS_WORK / ".resume_state/equil_01_eq01_nvt_0p5fs_50ps_chunk0001.restart"
    inp = work / "eq02_short.in"
    write_lammps_eq02_short_input(
        inp, restart.resolve(), outer_steps, full_outer_steps, ensemble, sample_outer_steps
    )
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": str(lammps_ntomp), "OMP_PROC_BIND": "true", "OMP_PLACES": "cores"})
    log = work / "eq02_short.stdout.log"
    returncode = run_live_to_file(
        [LMP, "-nonbuf", "-sf", "omp", "-pk", "omp", str(lammps_ntomp), "-in", inp],
        cwd=work,
        env=env,
        stdout_path=log,
        label=f"lammps_{ensemble}_outer{outer_steps}",
    )
    if returncode != 0:
        raise RuntimeError(f"LAMMPS short run failed for outer_steps={outer_steps}; see {log}")
    return summarize_lammps_rows(parse_lammps_rows(log))


def lammps_short_from_log(log: Path) -> dict[str, float]:
    if not log.exists():
        raise FileNotFoundError(log)
    return summarize_lammps_rows(parse_lammps_rows(log))


def gmx_short(
    root: Path,
    start_gro: Path,
    topol: Path,
    outer_steps: int,
    ntomp: int,
    nstlist: int,
    mass_mode: str,
    veta_scale: float,
    ensemble: str,
    sample_outer_steps: int,
    nhc_integrator: str,
    owner_scalar_fallback: int,
    pre_trotter: str,
    post_trotter: str,
    extended_update: str,
    inline_box_remap: int,
    pdamp_ps: float,
    pressure_mass_scale: float,
    fused_initial_drift: int,
    reprod: bool,
) -> dict[str, float | str | int]:
    nhc_label = nhc_integrator or "default"
    work = root / (
        f"gmx_{ensemble}_outer{outer_steps}_{mass_mode}_nstlist{nstlist}"
        f"_pre{pre_trotter}_post{post_trotter}_nhc{nhc_label}_fallback{owner_scalar_fallback}"
        f"_ext{extended_update}_inline{inline_box_remap}"
    )
    work.mkdir(parents=True, exist_ok=True)
    base_steps = outer_steps * 4
    mdp = work / "eq02_short.mdp"
    tpr = work / "eq02_short.tpr"
    deffnm = work / "eq02_short"
    make_gmx_short_mdp(
        GMX_CPU_WORK / "02_eq02_npt_compress_100ps_chunk0001.mdp",
        mdp,
        base_steps,
        nstlist,
        ensemble,
        sample_outer_steps * 4,
    )
    shutil.copy2(start_gro, work / "system.gro")
    shutil.copy2(topol, work / "topol.top")
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
            "GMX_PCFF_EWALD_BETA_INV_A": "0.21160096",
            "GMX_PCFF_EWALD_REAL_ONLY": "1",
            "GMX_PCFF_EXACT_RESPA_ALLOW_LINEAR_COM_REMOVAL": "1",
            "GMX_PCFF_EXACT_RESPA_NBNXM_OWNER_STEP_SCALAR_FALLBACK": str(owner_scalar_fallback),
            "GMX_PCFF_EXACT_RESPA_FUSED_INITIAL_DRIFT": str(fused_initial_drift),
            "GMX_PCFF_EXACT_RESPA_PRE_TROTTER": pre_trotter,
            "GMX_PCFF_EXACT_RESPA_POST_TROTTER": post_trotter,
        }
    )
    if nhc_integrator:
        env["GMX_PCFF_NHC_INTEGRATOR"] = nhc_integrator
    if ensemble == "npt":
        env.update(
            {
                "GMX_PCFF_MTTK_MASS_MODE": mass_mode,
                "GMX_PCFF_MTTK_LAMMPS_NATOMS": "7075",
                "GMX_PCFF_MTTK_LAMMPS_PDAMP_PS": f"{pdamp_ps:.9g}",
                "GMX_PCFF_MTTK_PRESSURE_MASS_SCALE": f"{pressure_mass_scale:.9g}",
                "GMX_PCFF_MTTK_BOXV_INTEGRATOR": "lammps",
                "GMX_PCFF_MTTK_NRESET_STEPS": "80000",
                "GMX_PCFF_EXACT_RESPA_MTTK_EXTENDED_UPDATE": extended_update,
                "GMX_PCFF_EXACT_RESPA_MTTK_INLINE_BOX_REMAP": str(inline_box_remap),
                "GMX_PCFF_EXACT_RESPA_MTTK_VETA_SCALE": f"{veta_scale:.9g}",
                "GMX_PCFF_REFP_RAMP_START_BAR": f"{1.0 * ATM_TO_BAR:.9g}",
                "GMX_PCFF_REFP_RAMP_END_BAR": f"{500.5 * ATM_TO_BAR:.9g}",
                "GMX_PCFF_REFP_RAMP_DURATION_PS": "50",
            }
        )
        if os.environ.get("PCFF_SHORT_PROP_TRACE_MTTK"):
            env["GMX_PCFF_MTTK_STATE_TRACE_FILE"] = str(work / "mttk_state_trace.csv")
            env["GMX_PCFF_MTTK_BOXV_TRACE_FILE"] = str(work / "mttk_boxv_trace.csv")
            env["GMX_PCFF_MTTK_STATE_TRACE_STRIDE"] = "1"
    mdrun_cmd = [
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
        "-nb",
        "cpu",
        "-pme",
        "cpu",
        "-bonded",
        "cpu",
        "-update",
        "cpu",
    ]
    if reprod:
        mdrun_cmd.append("-reprod")
    returncode = run_live_to_file(
        mdrun_cmd,
        cwd=work,
        env=env,
        stdout_path=work / "mdrun.stdout.log",
        label=f"gmx_{ensemble}_outer{outer_steps}_{mass_mode}_scale{pressure_mass_scale:.3g}",
    )
    out: dict[str, float | str | int] = {"status": "ok" if returncode == 0 else "mdrun_failed", "returncode": returncode}
    if returncode != 0:
        return out
    terms = "\n".join(["Potential", "Kinetic En.", "Total Energy", "Temperature", "Pressure", "Volume", "Density", "0", ""])
    energy = run([GMX, "energy", "-f", deffnm.with_suffix(".edr"), "-o", "selected.xvg"], cwd=work, stdin=terms)
    (work / "energy.stdout.log").write_text(energy.stdout, encoding="utf-8")
    out.update(parse_energy_xvg(work / "selected.xvg"))
    if deffnm.with_suffix(".gro").exists():
        out["gro_volume_nm3"] = gro_volume_nm3(deffnm.with_suffix(".gro"))
    return out


def add_deltas(row: dict[str, float | str | int]) -> None:
    lammps_samples = int(row.get("lammps_sample_count", 0))
    gromacs_samples = int(row.get("gmx_sample_count", 0))
    row["mean_comparison_aligned_samples"] = int(
        lammps_samples > 0 and lammps_samples == gromacs_samples
    )
    if not row["mean_comparison_aligned_samples"]:
        row["mean_comparison_note"] = (
            f"mean deltas use unaligned sample counts "
            f"(lammps={lammps_samples}, gromacs={gromacs_samples}); "
            "prefer endpoint deltas for this probe"
        )
    pairs = {
        "Temperature": "lammps_temperature_k",
        "Pressure": "lammps_pressure_bar",
        "Volume": "lammps_volume_nm3",
        "Potential": "lammps_potential_kj_mol",
        "Kinetic En.": "lammps_kinetic_kj_mol",
        "Total Energy": "lammps_total_kj_mol",
        "Temperature_mean": "lammps_temperature_mean_k",
        "Pressure_mean": "lammps_pressure_mean_bar",
        "Volume_mean": "lammps_volume_mean_nm3",
        "Potential_mean": "lammps_potential_mean_kj_mol",
        "Kinetic En._mean": "lammps_kinetic_mean_kj_mol",
        "Total Energy_mean": "lammps_total_mean_kj_mol",
    }
    for gmx_key, lmp_key in pairs.items():
        if gmx_key in row and lmp_key in row:
            row[f"{gmx_key.lower().replace(' ', '_').replace('.', '')}_delta"] = float(row[gmx_key]) - float(row[lmp_key])
    if "Density" in row and "lammps_density_g_cm3" in row:
        row["density_g_cm3_delta"] = float(row["Density"]) * 0.001 - float(row["lammps_density_g_cm3"])


def write_summary(path: Path, rows: list[dict[str, float | str | int]]) -> None:
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
    parser.add_argument("--root", type=Path, default=REPO / "output/probes/eq02_short_propagation_20260502")
    parser.add_argument("--outer-steps", type=int, nargs="+", default=[1, 10, 1000])
    parser.add_argument("--full-outer-steps", type=int, default=100000)
    parser.add_argument("--ntomp", type=int, default=12)
    parser.add_argument(
        "--lammps-ntomp",
        type=int,
        default=None,
        help=(
            "OpenMP thread count for the LAMMPS reference. Defaults to --ntomp. "
            "Use 1 for deterministic oracle probes; LAMMPS OMP trajectories can "
            "diverge between repeated runs over multi-ps NVT/NPT windows."
        ),
    )
    parser.add_argument("--nstlist", type=int, nargs="+", default=[1])
    parser.add_argument("--mass-mode", nargs="+", default=["lammps"])
    parser.add_argument("--veta-scale", type=float, default=1.0)
    parser.add_argument("--sample-outer-steps", type=int, default=0)
    parser.add_argument("--ensemble", choices=["npt", "nvt", "nve"], default="npt")
    parser.add_argument("--nhc-integrator", choices=["", "lammps"], default="lammps")
    parser.add_argument("--owner-scalar-fallback", type=int, choices=[0, 1], default=0)
    parser.add_argument("--pre-trotter", choices=["none", "two", "three", "two-three", "three-two"], default="two")
    parser.add_argument("--post-trotter", choices=["none", "two", "three", "two-three", "three-two"], default="three")
    parser.add_argument(
        "--extended-update",
        choices=["none", "velocity-only", "velocity-lammps-remap", "lammps-remap"],
        default="velocity-lammps-remap",
    )
    parser.add_argument("--inline-box-remap", type=int, choices=[0, 1], default=1)
    parser.add_argument("--pdamp-ps", type=float, default=0.5)
    parser.add_argument("--pressure-mass-scale", type=float, default=1.0)
    parser.add_argument("--fused-initial-drift", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--reuse-lammps-log",
        type=Path,
        default=None,
        help="Parse an existing LAMMPS short-run log instead of rerunning the LAMMPS reference.",
    )
    parser.add_argument("--reprod", action="store_true")
    args = parser.parse_args()
    lammps_ntomp = args.lammps_ntomp if args.lammps_ntomp is not None else args.ntomp

    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    print(f"bridge script: {BRIDGE_SCRIPT}", flush=True)
    start_gro, topol = prepare_start(args.root, lammps_ntomp)
    rows: list[dict[str, float | str | int]] = []
    summary = args.root / "eq02_short_propagation_summary.csv"
    for outer_steps in args.outer_steps:
        sample_outer_steps = args.sample_outer_steps if args.sample_outer_steps > 0 else outer_steps
        if args.reuse_lammps_log is not None:
            lmp = lammps_short_from_log(args.reuse_lammps_log.resolve())
        else:
            lmp = lammps_short(
                args.root,
                outer_steps,
                args.full_outer_steps,
                lammps_ntomp,
                args.ensemble,
                sample_outer_steps,
            )
        for mass_mode in args.mass_mode:
            for nstlist in args.nstlist:
                row: dict[str, float | str | int] = {
                    "outer_steps": outer_steps,
                    "base_steps": outer_steps * 4,
                    "ensemble": args.ensemble,
                    "lammps_ntomp": lammps_ntomp,
                    "gromacs_ntomp": args.ntomp,
                    "mass_mode": mass_mode,
                    "nstlist": nstlist,
                    "nhc_integrator": args.nhc_integrator or "default",
                    "owner_scalar_fallback": args.owner_scalar_fallback,
                    "pre_trotter": args.pre_trotter,
                    "post_trotter": args.post_trotter,
                    "extended_update": args.extended_update,
                    "inline_box_remap": args.inline_box_remap,
                    "pdamp_ps": args.pdamp_ps,
                    "pressure_mass_scale": args.pressure_mass_scale,
                    "fused_initial_drift": args.fused_initial_drift,
                    "reprod": int(args.reprod),
                }
                row.update(lmp)
                row.update(
                    gmx_short(
                        args.root,
                        start_gro,
                        topol,
                        outer_steps,
                        args.ntomp,
                        nstlist,
                        mass_mode,
                        args.veta_scale,
                        args.ensemble,
                        sample_outer_steps,
                        args.nhc_integrator,
                        args.owner_scalar_fallback,
                        args.pre_trotter,
                        args.post_trotter,
                        args.extended_update,
                        args.inline_box_remap,
                        args.pdamp_ps,
                        args.pressure_mass_scale,
                        args.fused_initial_drift,
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
