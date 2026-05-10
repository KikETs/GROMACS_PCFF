#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
GMX = REPO / "build_gateb_cuda/bin/gmx"
OUT_ROOT = REPO / "output/polygen_pcff_gromacs_initial_em_notebook"
GMX_CPU_WORK = OUT_ROOT / "gromacs_cpu_openmp"
LAMMPS_WORK = OUT_ROOT / "lammps_openmp"
LAMMPS = Path("/home/kiket/anaconda3/envs/MD/bin/lmp")

ATM_TO_BAR = 1.01325
LAMMPS_VELOCITY_A_PER_FS_TO_NM_PER_PS = 100.0
GMX_PRESFAC_BAR_PER_KJMOL_NM3 = 16.605390671738468
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
TENSOR_COMPONENTS_6 = (("XX", 1), ("YY", 2), ("ZZ", 3), ("XY", 4), ("XZ", 5), ("YZ", 6))


def replace_mdp_value(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^({re.escape(key)}\s*=\s*).*$", re.M)
    if pattern.search(text):
        return pattern.sub(rf"\g<1>{value}", text)
    return text.rstrip() + f"\n{key:<24}= {value}\n"


def make_mdp(base_mdp: Path, out_mdp: Path, nsteps: int, nstcouple: int, grid: int | None) -> None:
    text = base_mdp.read_text()
    energy_stride = max(1, min(4000, nsteps))
    replacements = {
        "nsteps": str(nsteps),
        "nstlist": "1",
        "nstcalcenergy": str(energy_stride),
        "nstenergy": str(energy_stride),
        "nstlog": str(energy_stride),
        "nsttcouple": str(nstcouple),
        "nstpcouple": str(nstcouple),
        "DispCorr": "AllEnerPres",
        "coulombtype": "PME",
        "pme-order": "5",
    }
    for key, value in replacements.items():
        text = replace_mdp_value(text, key, value)
    if grid is not None:
        for key in ("fourier-nx", "fourier-ny", "fourier-nz"):
            text = replace_mdp_value(text, key, str(grid))
    out_mdp.write_text(text)


def run(cmd: list[str | Path], cwd: Path, env: dict[str, str] | None = None, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        env=env,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


def parse_energy_xvg(path: Path) -> dict[str, float]:
    rows: list[list[float]] = []
    legends: list[str] = []
    if not path.exists():
        return {}
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("@") and "legend" in line:
            m = re.search(r'legend\s+"([^"]+)"', line)
            if m:
                legends.append(m.group(1))
        if not line or line.startswith(("#", "@")):
            continue
        vals = [float(x) for x in line.split()]
        rows.append(vals)
    if not rows:
        return {}
    last = rows[-1]
    result = {"time_ps": last[0]}
    for i, name in enumerate(legends, start=1):
        if i < len(last):
            result[name] = last[i]
    return result


def gro_volume_nm3(path: Path) -> float:
    vals = [float(x) for x in path.read_text(errors="ignore").splitlines()[-1].split()[:3]]
    return vals[0] * vals[1] * vals[2]


def parse_gmx_perf(log: Path) -> float:
    if not log.exists():
        return math.nan
    m = re.search(r"Performance:\s+([0-9.]+)(?:\s+ns/day)?", log.read_text(errors="ignore"))
    return float(m.group(1)) if m else math.nan


def parse_gmx_perf_any(paths: list[Path]) -> float:
    for path in paths:
        value = parse_gmx_perf(path)
        if not math.isnan(value):
            return value
    return math.nan


def read_lammps_velocity_dump(path: Path) -> dict[int, tuple[float, float, float]]:
    text = path.read_text(errors="ignore").splitlines()
    velocities: dict[int, tuple[float, float, float]] = {}
    i = 0
    while i < len(text):
        line = text[i]
        if not line.startswith("ITEM: ATOMS"):
            i += 1
            continue
        fields = line.split()[2:]
        required = ["id", "vx", "vy", "vz"]
        missing = [field for field in required if field not in fields]
        if missing:
            raise ValueError(f"{path} missing velocity dump field(s): {missing}")
        idx = {field: fields.index(field) for field in required}
        i += 1
        while i < len(text) and not text[i].startswith("ITEM:"):
            parts = text[i].split()
            atom_id = int(parts[idx["id"]])
            vx = float(parts[idx["vx"]]) * LAMMPS_VELOCITY_A_PER_FS_TO_NM_PER_PS
            vy = float(parts[idx["vy"]]) * LAMMPS_VELOCITY_A_PER_FS_TO_NM_PER_PS
            vz = float(parts[idx["vz"]]) * LAMMPS_VELOCITY_A_PER_FS_TO_NM_PER_PS
            velocities[atom_id] = (vx, vy, vz)
            i += 1
        break
    if not velocities:
        raise ValueError(f"No ITEM: ATOMS id vx vy vz block found in {path}")
    return velocities


def write_gro_with_lammps_velocities(source_gro: Path, velocity_dump: Path, out_gro: Path) -> Path:
    velocities = read_lammps_velocity_dump(velocity_dump)
    lines = source_gro.read_text(errors="ignore").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid GRO file: {source_gro}")
    natoms = int(lines[1].strip())
    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"{source_gro} has fewer atom rows than declared")

    missing: list[int] = []
    out = [lines[0], lines[1]]
    for raw in atom_lines:
        atom_id = int(raw[15:20])
        coord_fields = raw[20:].split()
        if len(coord_fields) < 3:
            raise ValueError(f"Could not parse coordinates for atom {atom_id} in {source_gro}")
        x, y, z = (float(coord_fields[0]), float(coord_fields[1]), float(coord_fields[2]))
        if atom_id not in velocities:
            missing.append(atom_id)
            vx = vy = vz = 0.0
        else:
            vx, vy, vz = velocities[atom_id]
        out.append(f"{raw[:20]}{x:18.12f}{y:18.12f}{z:18.12f}{vx:18.12f}{vy:18.12f}{vz:18.12f}")
    if missing:
        raise ValueError(f"{velocity_dump} is missing {len(missing)} atom velocities; first missing IDs: {missing[:10]}")
    out.extend(lines[2 + natoms :])
    out_gro.write_text("\n".join(out) + "\n")
    return out_gro


def parse_lammps_table_rows(log_path: Path) -> list[dict[str, float]]:
    lines = log_path.read_text(errors="ignore").splitlines()
    header: list[str] | None = None
    rows: list[dict[str, float]] = []
    for line in lines:
        if "Step" in line and "v_time" in line:
            header = line.split()
            continue
        if header is None:
            continue
        vals = FLOAT_RE.findall(line)
        if len(vals) == len(header):
            rows.append({key: float(value) for key, value in zip(header, vals)})
    return rows


def parse_lammps_eq04_chunk1_reference(log_path: Path, target_outer_step: int) -> dict[str, float]:
    lines = log_path.read_text(errors="ignore").splitlines()
    in_segment = False
    header: list[str] | None = None
    rows: list[dict[str, float]] = []
    for line in lines:
        if "lammps_equil_04_eq04_npt_compress_500ps_chunk0001.in" in line:
            in_segment = True
            header = None
            rows = []
            continue
        if in_segment and line.startswith("$ ") and "lammps_equil_04_eq04_npt_compress_500ps_chunk0001.in" not in line:
            break
        if not in_segment:
            continue
        if "Step" in line and "v_time" in line:
            header = line.split()
            continue
        if header is None:
            continue
        vals = FLOAT_RE.findall(line)
        if len(vals) == len(header):
            row = {key: float(value) for key, value in zip(header, vals)}
            rows.append(row)
    for row in rows:
        if int(row.get("Step", -1)) == target_outer_step:
            return row
    raise RuntimeError(f"LAMMPS eq04 chunk1 step {target_outer_step} not found in {log_path}")


def parse_lammps_reference_row(log_path: Path, target_outer_step: int) -> dict[str, float]:
    for row in parse_lammps_table_rows(log_path):
        if int(row.get("Step", -1)) == target_outer_step:
            return row
    raise RuntimeError(f"LAMMPS reference step {target_outer_step} not found in {log_path}")


def run_lammps_eq04_reference(
    root: Path,
    target_outer_step: int,
    ntomp: int,
    ramp_outer_steps: int,
    thermo_outer_stride: int,
) -> tuple[Path, dict[str, float]]:
    if target_outer_step <= 0:
        raise ValueError("LAMMPS reference target step must be positive")
    if ramp_outer_steps < target_outer_step:
        raise ValueError(
            f"LAMMPS ramp_outer_steps ({ramp_outer_steps}) must be >= target_outer_step ({target_outer_step})"
        )
    work = root / "lammps_ref"
    work.mkdir(parents=True, exist_ok=True)
    restart = LAMMPS_WORK / ".resume_state/equil_03_minimize.restart"
    if not restart.exists():
        raise FileNotFoundError(f"LAMMPS eq03 restart missing: {restart}")
    lmp_input = work / "eq04_tensor_reference.in"
    thermo_stride = max(1, min(thermo_outer_stride, target_outer_step))
    fix_state_terms = " ".join(f"f_1[{i}]" for i in range(1, 14))
    lmp_input.write_text(
        f"""echo both
variable        tlo         equal 353

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

variable        sysmass     equal mass(all)/6.0221367e+23
variable        sysdensity  equal v_sysmass/vol/1.0e-24
variable        time        equal step*dt+0.000001
variable        tdamp       equal floor(100*dt)
variable        pdamp       equal floor(1000*dt)

timestep        2
run_style       respa 2 4

reset_timestep  0
velocity        all create ${{tlo}} 63862 dist gaussian mom yes rot yes
compute         p_full all pressure thermo_temp
compute         p_vir all pressure NULL virial
fix             1 all npt temp 353 353 ${{tdamp}} iso 1 1600.6 ${{pdamp}} drag 0 mtk yes nreset 20000
thermo_style    custom step v_time press c_p_full[1] c_p_full[2] c_p_full[3] c_p_full[4] c_p_full[5] c_p_full[6] c_p_vir[1] c_p_vir[2] c_p_vir[3] c_p_vir[4] c_p_vir[5] c_p_vir[6] {fix_state_terms} vol v_sysdensity temp pe ke etotal
thermo          {thermo_stride}
thermo_modify   flush yes
run             {target_outer_step} start 0 stop {ramp_outer_steps}
unfix           1
""",
    )
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": str(ntomp), "OMP_PROC_BIND": "true", "OMP_PLACES": "cores"})
    proc = run([LAMMPS, "-nonbuf", "-sf", "omp", "-pk", "omp", str(ntomp), "-in", lmp_input], cwd=work, env=env)
    stdout = work / "eq04_tensor_reference.stdout.log"
    stdout.write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"LAMMPS reference failed with code {proc.returncode}; see {stdout}")
    return stdout, parse_lammps_reference_row(stdout, target_outer_step)


def run_variant(
    root: Path,
    name: str,
    mode: str,
    nsteps: int,
    nstcouple: int,
    grid: int | None,
    ntomp: int,
    start_gro: Path,
    topol: Path,
    velocity_dump: Path | None,
    mass_mode: str,
    force_scalar_pairloop: bool,
    owner_scalar_fallback: bool,
) -> dict[str, float | str | int]:
    work = root / name
    work.mkdir(parents=True, exist_ok=True)
    base_mdp = GMX_CPU_WORK / "04_eq04_npt_compress_500ps_chunk0001.mdp"
    mdp = work / f"{name}.mdp"
    tpr = work / f"{name}.tpr"
    deffnm = work / name
    make_mdp(base_mdp, mdp, nsteps, nstcouple, grid)

    gro = start_gro
    if velocity_dump is not None:
        gro = write_gro_with_lammps_velocities(start_gro, velocity_dump, work / f"{name}.start_lammps_velocity.gro")
    grompp = run(
        [GMX, "grompp", "-f", mdp, "-c", gro, "-p", topol, "-o", tpr, "-maxwarn", "2"],
        cwd=work,
    )
    (work / "grompp.stdout.log").write_text(grompp.stdout)
    if grompp.returncode != 0:
        return {"variant": name, "mode": mode, "status": "grompp_failed", "returncode": grompp.returncode}

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(ntomp),
            "OMP_PROC_BIND": "true",
            "OMP_PLACES": "cores",
            "GMX_PCFF_MTTK_MASS_MODE": mass_mode,
            "GMX_PCFF_MTTK_LAMMPS_NATOMS": "7075",
            "GMX_PCFF_MTTK_LAMMPS_PDAMP_PS": "2",
            "GMX_PCFF_MTTK_NRESET_STEPS": "80000",
            "GMX_PCFF_REFP_RAMP_START_BAR": f"{1.0 * ATM_TO_BAR:.9g}",
            "GMX_PCFF_REFP_RAMP_END_BAR": f"{1600.6 * ATM_TO_BAR:.9g}",
            "GMX_PCFF_REFP_RAMP_DURATION_PS": "200",
            "GMX_PCFF_EWALD_BETA_INV_A": "0.23761431",
            "GMX_PCFF_EXACT_RESPA_FUSED_INITIAL_DRIFT": "1",
        }
    )
    if force_scalar_pairloop:
        env["GMX_PCFF_EXACT_RESPA_DISABLE_NBNXM_NARROW"] = "1"
    else:
        env.pop("GMX_PCFF_EXACT_RESPA_DISABLE_NBNXM_NARROW", None)
    if owner_scalar_fallback:
        env["GMX_PCFF_EXACT_RESPA_NBNXM_OWNER_STEP_SCALAR_FALLBACK"] = "1"
    else:
        env["GMX_PCFF_EXACT_RESPA_NBNXM_OWNER_STEP_SCALAR_FALLBACK"] = "0"
    env.pop("GMX_PCFF_EWALD_REAL_ONLY", None)
    if mode != "legacy":
        env["GMX_PCFF_EXACT_RESPA_MTTK_EXTENDED_UPDATE"] = mode
    else:
        env.pop("GMX_PCFF_EXACT_RESPA_MTTK_EXTENDED_UPDATE", None)
    if mode in {"lammps-remap", "velocity-lammps-remap", "velocity-position", "position-only"}:
        env["GMX_PCFF_EXACT_RESPA_MTTK_INLINE_BOX_REMAP"] = "1"
    else:
        env.pop("GMX_PCFF_EXACT_RESPA_MTTK_INLINE_BOX_REMAP", None)

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
        "-nb",
        "cpu",
        "-pme",
        "cpu",
        "-bonded",
        "cpu",
        "-update",
        "cpu",
        "-reprod",
    ]
    mdrun = run(mdrun_cmd, cwd=work, env=env)
    (work / f"{name}.stdout.log").write_text(mdrun.stdout)

    row: dict[str, float | str | int] = {
        "variant": name,
        "mode": mode,
        "status": "ok" if mdrun.returncode == 0 else "mdrun_failed",
        "returncode": mdrun.returncode,
        "nsteps": nsteps,
        "nstcouple": nstcouple,
        "grid": grid or 0,
        "mass_mode": mass_mode,
        "disable_nbnxm_narrow": 1 if force_scalar_pairloop else 0,
        "owner_scalar_fallback": 1 if owner_scalar_fallback else 0,
        "start_volume_nm3": gro_volume_nm3(gro),
        "perf_ns_day": parse_gmx_perf_any([Path(f"{deffnm}.log"), work / f"{name}.stdout.log"]),
    }
    if Path(f"{deffnm}.gro").exists():
        row["gro_volume_nm3"] = gro_volume_nm3(Path(f"{deffnm}.gro"))
    edr = Path(f"{deffnm}.edr")
    if edr.exists():
        terms = "\n".join(
            [
                "Potential",
                "Kinetic En.",
                "Total Energy",
                "Temperature",
                "Pres. DC",
                "Pressure",
                "Vir-XX",
                "Vir-XY",
                "Vir-XZ",
                "Vir-YX",
                "Vir-YY",
                "Vir-YZ",
                "Vir-ZX",
                "Vir-ZY",
                "Vir-ZZ",
                "Pres-XX",
                "Pres-XY",
                "Pres-XZ",
                "Pres-YX",
                "Pres-YY",
                "Pres-YZ",
                "Pres-ZX",
                "Pres-ZY",
                "Pres-ZZ",
                "Box-Vel-XX",
                "Box-Vel-YY",
                "Box-Vel-ZZ",
                "Box-X",
                "Box-Y",
                "Box-Z",
                "Volume",
                "Density",
                "Coul. recip.",
                "0",
                "",
            ]
        )
        energy = run([GMX, "energy", "-f", edr, "-o", work / f"{name}_selected.xvg"], cwd=work, stdin=terms)
        (work / f"{name}_energy.stdout.log").write_text(energy.stdout)
        row.update(parse_energy_xvg(work / f"{name}_selected.xvg"))
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsteps", type=int, default=80000)
    parser.add_argument("--nstcouple", type=int, default=4)
    parser.add_argument("--grid", type=int, default=15)
    parser.add_argument("--ntomp", type=int, default=12)
    parser.add_argument("--root", type=Path, default=REPO / "output/probes/eq04_fullpme_mttk_probe_current")
    parser.add_argument("--start-gro", type=Path, default=GMX_CPU_WORK / "04_eq04_npt_compress_500ps_chunk0001.lammps_velocity.gro")
    parser.add_argument("--topol", type=Path, default=GMX_CPU_WORK / "topol.top")
    parser.add_argument("--velocity-dump", type=Path, default=None)
    parser.add_argument("--mass-mode", default="lammps_pmass_pchain")
    parser.add_argument(
        "--allow-nbnxm-narrow",
        action="store_true",
        help="Deprecated compatibility flag; NBNXM narrow is now enabled by default with owner-step scalar fallback.",
    )
    parser.add_argument(
        "--force-scalar-pairloop",
        action="store_true",
        help="Disable the NBNXM narrow path entirely and use the full scalar exact pair-loop diagnostic oracle.",
    )
    parser.add_argument(
        "--disable-owner-scalar-fallback",
        action="store_true",
        help="Restore the old all-NBNXM owner/outer-step path. This is expected to drift on PolyGen.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["legacy", "lammps-remap", "velocity-position", "velocity-lammps-remap"],
    )
    parser.add_argument(
        "--run-lammps-ref",
        action="store_true",
        help="Run a fresh short LAMMPS eq04 reference with pressure tensor and virial tensor thermo columns.",
    )
    parser.add_argument(
        "--lammps-ref-log",
        type=Path,
        default=None,
        help="Reuse an existing LAMMPS tensor reference log instead of the notebook log.",
    )
    parser.add_argument(
        "--lammps-ramp-outer-steps",
        type=int,
        default=100000,
        help="LAMMPS run stop value used for the eq04 pressure ramp. Must match the original stage run length.",
    )
    parser.add_argument(
        "--lammps-thermo-outer-stride",
        type=int,
        default=10000,
        help="LAMMPS thermo stride for the eq04 tensor reference. Defaults to the original notebook schedule.",
    )
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    args.start_gro = args.start_gro.resolve()
    args.topol = args.topol.resolve()
    if args.velocity_dump is not None:
        args.velocity_dump = args.velocity_dump.resolve()

    target_outer_step = args.nsteps // 4
    lammps_source = args.lammps_ref_log.resolve() if args.lammps_ref_log is not None else LAMMPS_WORK / "equil_from_em.stdout.log"
    if args.run_lammps_ref:
        lammps_source, lmp = run_lammps_eq04_reference(
            args.root,
            target_outer_step,
            args.ntomp,
            args.lammps_ramp_outer_steps,
            args.lammps_thermo_outer_stride,
        )
    elif args.lammps_ref_log is not None:
        lmp = parse_lammps_reference_row(lammps_source, target_outer_step)
    else:
        lmp = parse_lammps_eq04_chunk1_reference(lammps_source, target_outer_step)
    lmp_ref = {
        "lammps_source": str(lammps_source),
        "lammps_step": target_outer_step,
        "lammps_time_ps": lmp["v_time"] / 1000.0,
        "lammps_pressure_bar": lmp["Press"] * ATM_TO_BAR,
        "lammps_volume_nm3": lmp["Volume"] / 1000.0,
        "lammps_density_g_cm3": lmp["v_sysdensity"],
        "lammps_temperature_k": lmp["Temp"],
        "lammps_potential_kj_mol": lmp["PotEng"] * 4.184,
        "lammps_kinetic_kj_mol": lmp["KinEng"] * 4.184,
        "lammps_total_kj_mol": lmp["TotEng"] * 4.184,
    }
    for comp, idx in TENSOR_COMPONENTS_6:
        full_key = f"c_p_full[{idx}]"
        vir_key = f"c_p_vir[{idx}]"
        if full_key in lmp:
            lmp_ref[f"lammps_pressure_{comp.lower()}_bar"] = lmp[full_key] * ATM_TO_BAR
        if vir_key in lmp:
            lmp_ref[f"lammps_virial_pressure_{comp.lower()}_bar"] = lmp[vir_key] * ATM_TO_BAR
    for idx in range(1, 14):
        key = f"f_1[{idx}]"
        if key in lmp:
            lmp_ref[f"lammps_fix1_{idx}"] = lmp[key]
    if "f_1[7]" in lmp:
        lmp_ref["lammps_omega"] = lmp["f_1[7]"]
    if "f_1[8]" in lmp:
        lmp_ref["lammps_omega_dot_per_fs"] = lmp["f_1[8]"]
        lmp_ref["lammps_omega_dot_ps_inv"] = lmp["f_1[8]"] * 1000.0
    rows: list[dict[str, float | str | int]] = []
    for mode in args.variants:
        name = mode.replace("-", "_") + f"_{args.nsteps}"
        row = run_variant(
            args.root,
            name,
            mode,
            args.nsteps,
            args.nstcouple,
            args.grid,
            args.ntomp,
            args.start_gro,
            args.topol,
            args.velocity_dump,
            args.mass_mode,
            args.force_scalar_pairloop,
            not args.disable_owner_scalar_fallback,
        )
        row.update(lmp_ref)
        if "Volume" in row:
            row["volume_delta_nm3"] = float(row["Volume"]) - lmp_ref["lammps_volume_nm3"]
            row["volume_pct_vs_lammps"] = 100.0 * float(row["volume_delta_nm3"]) / lmp_ref["lammps_volume_nm3"]
        if "Temperature" in row:
            row["temperature_delta_k"] = float(row["Temperature"]) - lmp_ref["lammps_temperature_k"]
        if "Pressure" in row:
            row["pressure_delta_bar"] = float(row["Pressure"]) - lmp_ref["lammps_pressure_bar"]
        if "Potential" in row:
            row["potential_delta_kj_mol"] = float(row["Potential"]) - lmp_ref["lammps_potential_kj_mol"]
        if "Box-Vel-XX" in row and "Box-X" in row and float(row["Box-X"]) != 0:
            row["gmx_veta_ps_inv"] = float(row["Box-Vel-XX"]) / float(row["Box-X"])
            if "lammps_omega_dot_ps_inv" in lmp_ref:
                row["omega_dot_delta_ps_inv"] = row["gmx_veta_ps_inv"] - float(lmp_ref["lammps_omega_dot_ps_inv"])
        volume_nm3 = float(row["Volume"]) if "Volume" in row else math.nan
        for comp, _idx in TENSOR_COMPONENTS_6:
            lower = comp.lower()
            gmx_pressure_key = f"Pres-{comp}"
            lmp_pressure_key = f"lammps_pressure_{lower}_bar"
            if gmx_pressure_key in row and lmp_pressure_key in lmp_ref:
                row[f"pressure_{lower}_delta_bar"] = float(row[gmx_pressure_key]) - float(lmp_ref[lmp_pressure_key])
            gmx_virial_key = f"Vir-{comp}"
            lmp_virial_key = f"lammps_virial_pressure_{lower}_bar"
            if gmx_virial_key in row and not math.isnan(volume_nm3):
                gmx_virial_pressure = -2.0 * float(row[gmx_virial_key]) * GMX_PRESFAC_BAR_PER_KJMOL_NM3 / volume_nm3
                row[f"gmx_virial_pressure_{lower}_bar_est"] = gmx_virial_pressure
                if lmp_virial_key in lmp_ref:
                    row[f"virial_pressure_{lower}_delta_bar"] = gmx_virial_pressure - float(lmp_ref[lmp_virial_key])
        rows.append(row)
        print(row, flush=True)

    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (args.root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(args.root / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
