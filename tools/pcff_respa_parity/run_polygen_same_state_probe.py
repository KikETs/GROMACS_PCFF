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
GMX = Path(os.environ.get("GMX_BIN", REPO / "build_gateb_cuda/bin/gmx"))
LMP = Path("/home/kiket/anaconda3/envs/MD/bin/lmp")
BRIDGE_REPO = Path("/home/kiket/Desktop/test/GROMACS_PCFF-lunar-data-bridge")
BRIDGE_SCRIPT = BRIDGE_REPO / "tools/pcff_fixture_bridge/lammps_data_bridge.py"
OUT_ROOT = REPO / "output/polygen_pcff_gromacs_initial_em_notebook"
GMX_CPU_WORK = OUT_ROOT / "gromacs_cpu_openmp"
LAMMPS_WORK = OUT_ROOT / "lammps_openmp"

ATM_TO_BAR = 1.01325
KCAL_TO_KJ = 4.184
GMX_PRESFAC_BAR_PER_KJMOL_NM3 = 16.605390671738468
VEL_A_PER_FS_TO_NM_PER_PS = 100.0
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
TENSOR_COMPONENTS_6 = (("xx", 1), ("yy", 2), ("zz", 3), ("xy", 4), ("xz", 5), ("yz", 6))


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


def replace_mdp_value(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^({re.escape(key)}\s*=\s*).*$", re.M)
    if pattern.search(text):
        return pattern.sub(rf"\g<1>{value}", text)
    return text.rstrip() + f"\n{key:<24}= {value}\n"


def read_mdp_value(path: Path, key: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^;#\n]+)", re.M)
    match = pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
    if not match:
        return None
    return match.group(1).strip()


def make_eval_mdp(base_mdp: Path, out_mdp: Path, grid: int | None) -> None:
    text = base_mdp.read_text(encoding="utf-8")
    for key, value in {
        "integrator": "md-vv",
        "dt": "0.0005",
        "nsteps": "0",
        "nstcalcenergy": "4",
        "nstenergy": "4",
        "nstlog": "4",
        "nstlist": "1",
        "gen-vel": "no",
        "continuation": "yes",
        "DispCorr": "AllEnerPres",
        "coulombtype": "PME",
        "pme-order": "5",
    }.items():
        text = replace_mdp_value(text, key, value)
    if grid is not None:
        for key in ("fourier-nx", "fourier-ny", "fourier-nz"):
            text = replace_mdp_value(text, key, str(grid))
    out_mdp.write_text(text, encoding="utf-8")


def parse_last_lammps_row(log_path: Path) -> dict[str, float]:
    header: list[str] | None = None
    rows: list[dict[str, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Step" in line and "c_p_full" in line and "c_p_vir" in line:
            header = line.split()
            continue
        if header is None:
            continue
        vals = FLOAT_RE.findall(line)
        if len(vals) == len(header):
            rows.append({key: float(value) for key, value in zip(header, vals)})
    if not rows:
        raise RuntimeError(f"No LAMMPS thermo row found in {log_path}")
    return rows[-1]


def parse_energy_xvg(path: Path) -> dict[str, float]:
    legends: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("@") and "legend" in line:
            m = re.search(r'legend\s+"([^"]+)"', line)
            if m:
                legends.append(m.group(1))
        if not line or line.startswith(("#", "@")):
            continue
        rows.append([float(x) for x in line.split()])
    if not rows:
        return {}
    last = rows[-1]
    out = {"time_ps": last[0]}
    for i, name in enumerate(legends, start=1):
        if i < len(last):
            out[name] = last[i]
            out[f"{name}_mean"] = sum(row[i] for row in rows if i < len(row)) / len(rows)
    out["gmx_sample_count"] = len(rows)
    return out


def parse_lammps_data_velocities(path: Path) -> dict[int, tuple[float, float, float]]:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    velocities: dict[int, tuple[float, float, float]] = {}
    in_velocities = False
    for raw in text:
        stripped = raw.strip()
        if stripped == "Velocities":
            in_velocities = True
            continue
        if not in_velocities:
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^[A-Za-z]", stripped):
            break
        parts = stripped.split()
        if len(parts) >= 4 and parts[0].lstrip("-").isdigit():
            velocities[int(parts[0])] = (
                float(parts[1]) * VEL_A_PER_FS_TO_NM_PER_PS,
                float(parts[2]) * VEL_A_PER_FS_TO_NM_PER_PS,
                float(parts[3]) * VEL_A_PER_FS_TO_NM_PER_PS,
            )
    if not velocities:
        raise RuntimeError(f"No Velocities section found in {path}")
    return velocities


def write_gro_with_velocities(source_gro: Path, lammps_data: Path, out_gro: Path) -> None:
    velocities = parse_lammps_data_velocities(lammps_data)
    lines = source_gro.read_text(encoding="utf-8", errors="ignore").splitlines()
    natoms = int(lines[1].strip())
    out = [lines[0], lines[1]]
    missing: list[int] = []
    for raw in lines[2 : 2 + natoms]:
        atom_id = int(raw[15:20])
        fields = raw[20:].split()
        x, y, z = (float(fields[0]), float(fields[1]), float(fields[2]))
        if atom_id not in velocities:
            missing.append(atom_id)
            vx = vy = vz = 0.0
        else:
            vx, vy, vz = velocities[atom_id]
        out.append(f"{raw[:20]}{x:18.12f}{y:18.12f}{z:18.12f}{vx:18.12f}{vy:18.12f}{vz:18.12f}")
    if missing:
        raise RuntimeError(f"Missing {len(missing)} velocities in {lammps_data}; first IDs {missing[:10]}")
    out.extend(lines[2 + natoms :])
    out_gro.write_text("\n".join(out) + "\n", encoding="utf-8")


def gro_volume_nm3(path: Path) -> float:
    x, y, z = (float(v) for v in path.read_text(encoding="utf-8", errors="ignore").splitlines()[-1].split()[:3])
    return x * y * z


def write_lammps_run0_input(path: Path, restart: Path, data_out: Path, kspace_compute_no: bool) -> None:
    kspace_line = "kspace_modify   compute no\n" if kspace_compute_no else ""
    path.write_text(
        f"""echo both
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
{kspace_line}variable        sysvol      equal vol
variable        sysmass     equal mass(all)/6.0221367e+23
variable        sysdensity  equal v_sysmass/v_sysvol/1.0e-24
compute         t_thermo all temp
compute         p_full all pressure t_thermo
compute         p_vir all pressure NULL virial
thermo_style    custom step temp press pxx pyy pzz pxy pxz pyz vol v_sysdensity pe ke etotal ebond eangle edihed eimp evdwl ecoul elong etail c_p_full c_p_full[1] c_p_full[2] c_p_full[3] c_p_full[4] c_p_full[5] c_p_full[6] c_p_vir c_p_vir[1] c_p_vir[2] c_p_vir[3] c_p_vir[4] c_p_vir[5] c_p_vir[6]
thermo          1
thermo_modify   flush yes
write_data      {data_out}
run             0 post no
""",
        encoding="utf-8",
    )


def bridge_lammps_data(data_path: Path, bridge_out: Path) -> None:
    bridge_out.mkdir(parents=True, exist_ok=True)
    proc = run(
        [
            sys.executable,
            BRIDGE_SCRIPT,
            "--data",
            data_path,
            "--out",
            bridge_out,
            "--system-id",
            "PolyGen_same_state_probe",
            "--display-name",
            "PolyGen same-state probe",
            "--category",
            "polymer_box",
            "--pair-style",
            "lj/class2/coul/long",
            "--pair-style-arg",
            "9.5",
            "--pair-modify",
            "mix sixthpower",
            "--special-bonds",
            "lj/coul 0.0 0.0 1.0 angle no dihedral no",
            "--kspace-style",
            "pppm 0.0001",
        ],
        cwd=BRIDGE_REPO,
    )
    (bridge_out.parent / "bridge.stdout.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Bridge failed with code {proc.returncode}; see {bridge_out.parent / 'bridge.stdout.log'}")


def run_gmx_eval(
    root: Path,
    name: str,
    base_mdp: Path,
    gro: Path,
    topol: Path,
    grid: int | None,
    ntomp: int,
    ewald_real_only: bool,
    ewald_beta_inv_a: float,
    mttk_mass_mode: str,
    mttk_pdamp_ps: float | None,
) -> dict[str, float | str | int]:
    work = root / f"gmx_eval_{name}"
    work.mkdir(parents=True, exist_ok=True)
    mdp = work / "eval.mdp"
    tpr = work / "eval.tpr"
    make_eval_mdp(base_mdp, mdp, grid)
    shutil.copy2(gro, work / "system.gro")
    shutil.copy2(topol, work / "topol.top")
    grompp = run([GMX, "grompp", "-f", mdp, "-c", "system.gro", "-p", "topol.top", "-o", tpr, "-maxwarn", "5"], cwd=work)
    (work / "grompp.stdout.log").write_text(grompp.stdout, encoding="utf-8")
    row: dict[str, float | str | int] = {"variant": name, "grid": grid or 0, "status": "ok", "returncode": 0}
    if grompp.returncode != 0:
        row.update({"status": "grompp_failed", "returncode": grompp.returncode})
        return row
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(ntomp),
            "OMP_PROC_BIND": "true",
            "OMP_PLACES": "cores",
            "GMX_PCFF_EXACT_RESPA_NBNXM_OWNER_STEP_SCALAR_FALLBACK": "1",
            "GMX_PCFF_EXACT_RESPA_FUSED_INITIAL_DRIFT": "1",
            "GMX_PCFF_EXACT_RESPA_ALLOW_LINEAR_COM_REMOVAL": "1",
        }
    )
    base_text = base_mdp.read_text(encoding="utf-8", errors="ignore")
    if re.search(r"^exact-respa\s*=\s*yes\s*$", base_text, re.M | re.I):
        env["GMX_PCFF_EWALD_BETA_INV_A"] = f"{ewald_beta_inv_a:.8g}"
    if re.search(r"^pcoupl\s*=\s*MTTK\s*$", base_text, re.M | re.I):
        tau_p = read_mdp_value(base_mdp, "tau-p")
        pdamp_ps = mttk_pdamp_ps if mttk_pdamp_ps is not None else (float(tau_p) if tau_p else 2.0)
        env.update(
            {
                "GMX_PCFF_MTTK_MASS_MODE": mttk_mass_mode,
                "GMX_PCFF_MTTK_LAMMPS_NATOMS": "7075",
                "GMX_PCFF_MTTK_LAMMPS_PDAMP_PS": f"{pdamp_ps:.9g}",
                "GMX_PCFF_MTTK_NRESET_STEPS": "80000",
                "GMX_PCFF_EXACT_RESPA_MTTK_EXTENDED_UPDATE": "velocity-lammps-remap",
                "GMX_PCFF_EXACT_RESPA_MTTK_INLINE_BOX_REMAP": "1",
            }
        )
    if ewald_real_only:
        env["GMX_PCFF_EWALD_REAL_ONLY"] = "1"
    else:
        env.pop("GMX_PCFF_EWALD_REAL_ONLY", None)
    mdrun = run(
        [
            "taskset",
            "-c",
            "0-11",
            GMX,
            "mdrun",
            "-s",
            tpr,
            "-deffnm",
            "eval",
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
        ],
        cwd=work,
        env=env,
    )
    (work / "mdrun.stdout.log").write_text(mdrun.stdout, encoding="utf-8")
    row["returncode"] = mdrun.returncode
    if mdrun.returncode != 0:
        row["status"] = "mdrun_failed"
        return row
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
    energy = run([GMX, "energy", "-f", "eval.edr", "-o", "eval_components.xvg"], cwd=work, stdin=terms)
    (work / "energy.stdout.log").write_text(energy.stdout, encoding="utf-8")
    row.update(parse_energy_xvg(work / "eval_components.xvg"))
    if (work / "eval.gro").exists():
        row["gro_volume_nm3"] = gro_volume_nm3(work / "eval.gro")
    return row


def lammps_reference_values(row: dict[str, float]) -> dict[str, float]:
    out = {
        "lammps_step": row["Step"],
        "lammps_temperature_k": row["Temp"],
        "lammps_pressure_bar": row["Press"] * ATM_TO_BAR,
        "lammps_volume_nm3": row["Volume"] / 1000.0,
        "lammps_density_g_cm3": row["v_sysdensity"],
        "lammps_potential_kj_mol": row["PotEng"] * KCAL_TO_KJ,
        "lammps_kinetic_kj_mol": row["KinEng"] * KCAL_TO_KJ,
        "lammps_total_kj_mol": row["TotEng"] * KCAL_TO_KJ,
    }
    for comp, idx in TENSOR_COMPONENTS_6:
        out[f"lammps_pressure_{comp}_bar"] = row[f"c_p_full[{idx}]"] * ATM_TO_BAR
        out[f"lammps_virial_pressure_{comp}_bar"] = row[f"c_p_vir[{idx}]"] * ATM_TO_BAR
    return out


def add_deltas(row: dict[str, float | str | int], ref: dict[str, float]) -> None:
    pairs = {
        "Temperature": "lammps_temperature_k",
        "Pressure": "lammps_pressure_bar",
        "Volume": "lammps_volume_nm3",
        "Potential": "lammps_potential_kj_mol",
        "Kinetic En.": "lammps_kinetic_kj_mol",
        "Total Energy": "lammps_total_kj_mol",
    }
    for gmx_key, lmp_key in pairs.items():
        if gmx_key in row:
            safe = re.sub(r"[^A-Za-z0-9]+", "_", gmx_key).strip("_").lower()
            row[f"{safe}_delta"] = float(row[gmx_key]) - ref[lmp_key]
    if "Density" in row:
        row["density_g_cm3_delta"] = 0.001 * float(row["Density"]) - ref["lammps_density_g_cm3"]
    volume_nm3 = float(row["Volume"]) if "Volume" in row else math.nan
    for comp, _idx in TENSOR_COMPONENTS_6:
        up = comp.upper()
        pressure_key = f"Pres-{up}"
        if pressure_key in row:
            row[f"pressure_{comp}_delta_bar"] = float(row[pressure_key]) - ref[f"lammps_pressure_{comp}_bar"]
        virial_key = f"Vir-{up}"
        if virial_key in row and not math.isnan(volume_nm3):
            gmx_virial_pressure = -2.0 * float(row[virial_key]) * GMX_PRESFAC_BAR_PER_KJMOL_NM3 / volume_nm3
            row[f"gmx_virial_pressure_{comp}_bar"] = gmx_virial_pressure
            row[f"virial_pressure_{comp}_delta_bar"] = gmx_virial_pressure - ref[f"lammps_virial_pressure_{comp}_bar"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="eq04_npt_compress_500ps_chunk0001")
    parser.add_argument(
        "--restart",
        type=Path,
        default=LAMMPS_WORK / ".resume_state/equil_04_eq04_npt_compress_500ps_chunk0001.restart",
    )
    parser.add_argument(
        "--base-mdp",
        type=Path,
        default=GMX_CPU_WORK / "04_eq04_npt_compress_500ps_chunk0001.mdp",
    )
    parser.add_argument("--root", type=Path, default=REPO / "output/probes/eq04_same_state_chunk0001_20260502")
    parser.add_argument("--ntomp", type=int, default=12)
    parser.add_argument(
        "--lammps-ntomp",
        type=int,
        default=None,
        help="OpenMP thread count for the LAMMPS run-0 reference. Defaults to --ntomp.",
    )
    parser.add_argument("--kspace-compute-no", action="store_true")
    parser.add_argument("--grids", nargs="*", default=["auto", "15"])
    parser.add_argument("--ewald-beta-inv-a", type=float, default=0.23761431)
    parser.add_argument("--mttk-mass-mode", default="lammps_pmass")
    parser.add_argument("--mttk-pdamp-ps", type=float, default=None)
    args = parser.parse_args()
    lammps_ntomp = args.lammps_ntomp if args.lammps_ntomp is not None else args.ntomp

    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    data_out = args.root / f"{args.stage}_endpoint.lmp"
    lmp_in = args.root / "lammps_write_data_and_run0.in"
    write_lammps_run0_input(lmp_in, args.restart.resolve(), data_out, args.kspace_compute_no)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": str(lammps_ntomp), "OMP_PROC_BIND": "true", "OMP_PLACES": "cores"})
    lmp_proc = run([LMP, "-nonbuf", "-sf", "omp", "-pk", "omp", str(lammps_ntomp), "-in", lmp_in], cwd=args.root, env=env)
    lmp_log = args.root / "lammps_write_data_and_run0.stdout.log"
    lmp_log.write_text(lmp_proc.stdout, encoding="utf-8")
    if lmp_proc.returncode != 0:
        raise RuntimeError(f"LAMMPS run0 failed with code {lmp_proc.returncode}; see {lmp_log}")

    bridge_out = args.root / "bridge"
    bridge_lammps_data(data_out, bridge_out)
    gro_with_vel = args.root / "bridge/system_with_velocities.gro"
    write_gro_with_velocities(bridge_out / "system.gro", data_out, gro_with_vel)

    ref = lammps_reference_values(parse_last_lammps_row(lmp_log))
    rows: list[dict[str, float | str | int]] = []
    for grid_arg in args.grids:
        if grid_arg == "auto":
            grid = None
            name = "auto_grid"
        else:
            grid = int(grid_arg)
            name = f"grid{grid}"
        row = run_gmx_eval(
            args.root,
            name,
            args.base_mdp.resolve(),
            gro_with_vel,
            bridge_out / "topol.top",
            grid,
            args.ntomp,
            args.kspace_compute_no,
            args.ewald_beta_inv_a,
            args.mttk_mass_mode,
            args.mttk_pdamp_ps,
        )
        row.update(ref)
        add_deltas(row, ref)
        rows.append(row)
        print(row, flush=True)

    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    summary = args.root / "same_state_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
