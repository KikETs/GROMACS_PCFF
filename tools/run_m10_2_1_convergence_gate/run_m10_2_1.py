#!/home/kiket/anaconda3/envs/MD/bin/python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys
import os
import re
import math
import shutil
import numpy as np

# Add repo root to sys.path to import from tools
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.pcff_fixture_bridge.common import (
    build_typed_ir,
    render_gromacs_topology,
    parse_lammps_data,
    ANGSTROM_TO_NM,
    KCAL_TO_KJ,
)

KCAL_PER_A_TO_KJ_PER_NM = KCAL_TO_KJ / ANGSTROM_TO_NM

def create_gro_from_lammps(lammps_data_path: Path, gro_path: Path):
    data = parse_lammps_data(lammps_data_path)
    box_x = (data["box"]["x"]["hi"] - data["box"]["x"]["lo"]) * ANGSTROM_TO_NM
    box_y = (data["box"]["y"]["hi"] - data["box"]["y"]["lo"]) * ANGSTROM_TO_NM
    box_z = (data["box"]["z"]["hi"] - data["box"]["z"]["lo"]) * ANGSTROM_TO_NM
    
    with gro_path.open("w") as f:
        f.write("Generated from LAMMPS data\n")
        f.write(f"{len(data['atoms']):>5d}\n")
        for i, atom in enumerate(data["atoms"], start=1):
            x = atom["x_angstrom"] * ANGSTROM_TO_NM
            y = atom["y_angstrom"] * ANGSTROM_TO_NM
            z = atom["z_angstrom"] * ANGSTROM_TO_NM
            f.write(f"{1:>5d}{'MOL':<5s}{f'A{i}':>5s}{atom['id']:>5d}{x:15.7f}{y:15.7f}{z:15.7f}\n")
        f.write(f"{box_x:15.7f}{box_y:15.7f}{box_z:15.7f}\n")

def get_mdp_npt(dt: float, nsteps: int):
    return f"""
integrator  = md
dt          = {dt}
nsteps      = {nsteps}
cutoff-scheme = Verlet
vdw-type     = Cut-off
rvdw         = 1.2
coulombtype  = PME
rcoulomb     = 1.2
pbc          = xyz
nstxout      = 0
nstvout      = 0
nstfout      = 0
nstenergy    = 100
tcoupl      = v-rescale
tc-grps     = System
tau-t       = 0.1
ref-t       = 300
gen-vel     = yes
gen-temp    = 300
pcoupl      = Berendsen
pcoupltype  = isotropic
tau-p       = 1.0
compressibility = 4.5e-5
ref-p       = 1.0
"""

def run_gmx(cmd: list[str], work_dir: Path, log_name: str, nthreads: int = 12):
    gmx_bin = str(REPO_ROOT / "build" / "bin" / "gmx")
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    
    full_cmd = [gmx_bin] + cmd
    if "mdrun" in cmd:
        full_cmd += ["-nt", str(nthreads)]
        
    result = subprocess.run(full_cmd, cwd=work_dir, capture_output=True, text=True, env=env, errors="replace")
    (work_dir / f"{log_name}.stdout").write_text(result.stdout)
    (work_dir / f"{log_name}.stderr").write_text(result.stderr)
    return result.returncode == 0

def run_lammps_npt(work_dir: Path, dt: float, nsteps: int, nprocs: int = 12):
    # Using mpirun if available, else just lmp
    with (work_dir / "system.in").open("w") as f:
        f.write("""
units real
atom_style full
boundary p p p
pair_style lj/class2/coul/long 9.0 9.0
pair_modify mix sixthpower
bond_style class2
angle_style class2
dihedral_style class2
improper_style none
kspace_style pppm 1.0e-4
special_bonds lj/coul 0.0 0.0 1.0 angle no dihedral no
read_data system.data
pair_coeff 1 1 0.12 3.40
pair_coeff 2 2 0.08 3.00
bond_coeff 1 1.48 230.0 -32.0 7.0
angle_coeff 1 111.0 32.0 -3.6 1.0
angle_coeff 1 bb 5.0 1.48 1.48
angle_coeff 1 ba 1.7 1.4 1.48 1.48
dihedral_coeff 1 0.9 0.0 0.5 180.0 0.3 0.0
dihedral_coeff 1 mbt 0.14 -0.09 0.05 1.48
dihedral_coeff 1 ebt 0.12 -0.06 0.03 0.10 -0.04 0.02 1.48 1.48
dihedral_coeff 1 at 0.05 -0.03 0.02 0.04 -0.02 0.01 111.0 111.0
dihedral_coeff 1 aat 0.22 111.0 111.0
dihedral_coeff 1 bb13 0.16 1.48 1.48
""")
        f.write(f"\ntimestep {dt * 1000}\n")
        f.write("fix 1 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0\n")
        f.write("thermo 100\n")
        f.write("thermo_style custom step temp pe ke etotal press vol density\n")
        f.write(f"run {nsteps}\n")
    
    cmd = ["mpirun", "-np", str(nprocs), "lmp", "-in", "system.in"]
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    (work_dir / "lammps.stdout").write_text(result.stdout)
    (work_dir / "lammps.stderr").write_text(result.stderr)
    return result.returncode == 0

def parse_xvg(path: Path):
    data = []
    if not path.exists(): return data
    with path.open("r") as f:
        for line in f:
            if line.startswith(("#", "@")): continue
            parts = line.split()
            if parts: data.append([float(x) for x in parts])
    return np.array(data)

def analyze_ensemble_blocks(data, nblocks=5):
    # data: [steps, fields]
    nsteps = len(data)
    block_size = nsteps // nblocks
    block_means = []
    for i in range(nblocks):
        block = data[i*block_size : (i+1)*block_size]
        block_means.append(np.mean(block, axis=0))
    block_means = np.array(block_means)
    
    overall_mean = np.mean(data, axis=0)
    overall_std = np.std(data, axis=0)
    sem = np.std(block_means, axis=0) / math.sqrt(nblocks)
    
    # Check for drift: slope of block means
    drifts = []
    for i in range(data.shape[1]):
        if nblocks > 1:
            slope = np.polyfit(range(nblocks), block_means[:, i], 1)[0]
            # Normalize slope by std to see if it's significant
            drifts.append(slope / (overall_std[i] if overall_std[i] > 0 else 1.0))
        else:
            drifts.append(0.0)
            
    return {
        "mean": overall_mean.tolist(),
        "std": overall_std.tolist(),
        "sem": sem.tolist(),
        "block_means": block_means.tolist(),
        "normalized_drifts": drifts
    }

def get_convergence_status(stats, field_idx, threshold=0.2):
    drift = abs(stats["normalized_drifts"][field_idx])
    if drift < threshold:
        return "converged enough"
    elif drift < threshold * 3:
        return "trending but not converged"
    else:
        return "failed / unstable"

def run_gmx_async(cmd: list[str], work_dir: Path, log_name: str, nthreads: int = 12):
    gmx_bin = str(REPO_ROOT / "build" / "bin" / "gmx")
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    full_cmd = [gmx_bin] + cmd
    if "mdrun" in cmd:
        full_cmd += ["-nt", str(nthreads)]
    return subprocess.Popen(full_cmd, cwd=work_dir, stdout=open(work_dir/f"{log_name}.stdout", "w"), stderr=open(work_dir/f"{log_name}.stderr", "w"), env=env)

def run_lammps_async(work_dir: Path, dt: float, nsteps: int, nprocs: int = 12):
    with (work_dir / "system.in").open("w") as f:
        f.write(f"""
units real
atom_style full
boundary p p p
pair_style lj/class2/coul/long 9.0 9.0
pair_modify mix sixthpower
bond_style class2
angle_style class2
dihedral_style class2
improper_style none
kspace_style pppm 1.0e-4
special_bonds lj/coul 0.0 0.0 1.0 angle no dihedral no
read_data system.data
pair_coeff 1 1 0.12 3.40
pair_coeff 2 2 0.08 3.00
bond_coeff 1 1.48 230.0 -32.0 7.0
angle_coeff 1 111.0 32.0 -3.6 1.0
angle_coeff 1 bb 5.0 1.48 1.48
angle_coeff 1 ba 1.7 1.4 1.48 1.48
dihedral_coeff 1 0.9 0.0 0.5 180.0 0.3 0.0
dihedral_coeff 1 mbt 0.14 -0.09 0.05 1.48
dihedral_coeff 1 ebt 0.12 -0.06 0.03 0.10 -0.04 0.02 1.48 1.48
dihedral_coeff 1 at 0.05 -0.03 0.02 0.04 -0.02 0.01 111.0 111.0
dihedral_coeff 1 aat 0.22 111.0 111.0
dihedral_coeff 1 bb13 0.16 1.48 1.48
timestep {dt * 1000}
fix 1 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0
thermo 100
thermo_style custom step temp pe ke etotal press vol density
run {nsteps}
""")
    cmd = ["mpirun", "-np", str(nprocs), "lmp", "-in", "system.in"]
    return subprocess.Popen(cmd, cwd=work_dir, stdout=open(work_dir/"lammps.stdout", "w"), stderr=open(work_dir/"lammps.stderr", "w"))

def check_system(sid: str, results_root: Path):
    work_dir = results_root / sid
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- M10.2.1 Convergence Gate for {sid} ---")
    system_dir = REPO_ROOT / "testdata" / "lammps_golden" / "systems" / "small_oligomer"
    
    # Replicate
    data_file = system_dir / "lammps" / "system.data"
    shutil.copy(data_file, work_dir / "orig.data")
    with open(work_dir / "replicate.in", "w") as f:
        f.write("units real\natom_style full\nread_data orig.data\nreplicate 4 4 4\nwrite_data system.data nocoeff\n")
    subprocess.run(["lmp", "-in", "replicate.in"], cwd=work_dir, capture_output=True, text=True, check=True)

    dt = 0.002
    nsteps_equil = 5000
    nsteps_prod = 50000
    
    # GMX Setup
    ir = build_typed_ir({"id": "small_oligomer", "path": "systems/small_oligomer"}, REPO_ROOT / "testdata" / "lammps_golden")
    top_content = render_gromacs_topology(ir).replace("OLI 1", "OLI 64")
    (work_dir / "system.top").write_text(top_content)
    create_gro_from_lammps(work_dir / "system.data", work_dir / "system.gro")
    
    # Run Equil (Sequential, fast)
    work_dir.joinpath("equil.mdp").write_text(get_mdp_npt(dt, nsteps_equil))
    run_gmx(["grompp", "-f", "equil.mdp", "-c", "system.gro", "-p", "system.top", "-o", "equil.tpr", "-maxwarn", "10"], work_dir, "grompp_equil")
    run_gmx(["mdrun", "-s", "equil.tpr", "-c", "equil.gro", "-e", "equil.edr", "-g", "equil.log"], work_dir, "mdrun_equil")
    
    # Run Production in Parallel
    print(f"Starting GROMACS and LAMMPS in parallel (100ps)...")
    work_dir.joinpath("run.mdp").write_text(get_mdp_npt(dt, nsteps_prod))
    run_gmx(["grompp", "-f", "run.mdp", "-c", "equil.gro", "-p", "system.top", "-o", "run.tpr", "-maxwarn", "10"], work_dir, "grompp")
    
    p_gmx = run_gmx_async(["mdrun", "-s", "run.tpr", "-c", "out.gro", "-e", "run.edr", "-g", "run.log"], work_dir, "mdrun", nthreads=12)
    p_lmp = run_lammps_async(work_dir, dt, nsteps_equil + nsteps_prod, nprocs=12)
    
    p_gmx.wait()
    p_lmp.wait()
    
    # Extract GMX
    energy_input = "Potential\nTemperature\nPressure\nVolume\nDensity\n0\n"
    subprocess.run([str(REPO_ROOT / "build" / "bin" / "gmx"), "energy", "-f", "run.edr", "-o", "energy.xvg"], cwd=work_dir, input=energy_input, capture_output=True, text=True)
    gmx_data = parse_xvg(work_dir / "energy.xvg")
    
    # Parse LMP
    lmp_raw = []
    with (work_dir / "log.lammps").open("r") as f:
        in_thermo = False
        for line in f:
            clean_line = line.strip()
            if clean_line.startswith("Step"): in_thermo = True; continue
            if clean_line.startswith("Loop time"): in_thermo = False; continue
            if in_thermo:
                parts = clean_line.split()
                if parts and parts[0].isdigit(): lmp_raw.append([float(x) for x in parts])
    lmp_data = np.array(lmp_raw)
    
    gmx_stats = analyze_ensemble_blocks(gmx_data)
    lmp_stats = analyze_ensemble_blocks(lmp_data[nsteps_equil // 100 :])
    
    report = {
        "system_id": sid,
        "duration_ps": nsteps_prod * dt,
        "gmx": {
            "potential_energy": {"mean": gmx_stats["mean"][1], "sem": gmx_stats["sem"][1], "status": get_convergence_status(gmx_stats, 1)},
            "temperature": {"mean": gmx_stats["mean"][2], "sem": gmx_stats["sem"][2], "status": get_convergence_status(gmx_stats, 2)},
            "density": {"mean": gmx_stats["mean"][5], "sem": gmx_stats["sem"][5], "status": get_convergence_status(gmx_stats, 5)}
        },
        "lmp": {
            "potential_energy": {"mean": lmp_stats["mean"][2] * KCAL_TO_KJ, "sem": lmp_stats["sem"][2] * KCAL_TO_KJ, "status": get_convergence_status(lmp_stats, 2)},
            "temperature": {"mean": lmp_stats["mean"][1], "sem": lmp_stats["sem"][1], "status": get_convergence_status(lmp_stats, 1)},
            "density": {"mean": lmp_stats["mean"][7] * 1000, "sem": lmp_stats["sem"][7] * 1000, "status": get_convergence_status(lmp_stats, 7)}
        }
    }
    
    temp_diff = abs(report["gmx"]["temperature"]["mean"] - report["lmp"]["temperature"]["mean"])
    dens_diff_rel = abs(report["gmx"]["density"]["mean"] - report["lmp"]["density"]["mean"]) / report["gmx"]["density"]["mean"]
    report["parity_status"] = "pass" if (temp_diff < 5.0 and dens_diff_rel < 0.05) else "partial"
    
    with (work_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Convergence Report for {sid}:")
    print(f"  GMX Temp: {report['gmx']['temperature']['mean']:.2f} K ({report['gmx']['temperature']['status']})")
    print(f"  LMP Temp: {report['lmp']['temperature']['mean']:.2f} K ({report['lmp']['temperature']['status']})")
    print(f"  GMX Density: {report['gmx']['density']['mean']:.2f} kg/m3 ({report['gmx']['density']['status']})")
    print(f"  LMP Density: {report['lmp']['density']['mean']:.2f} kg/m3 ({report['lmp']['density']['status']})")
    print(f"  Parity Status: {report['parity_status']}")
    
    return report

def main():
    results_root = REPO_ROOT / "tests" / "reference_results" / "m10_2_1_convergence_gate"
    results_root.mkdir(parents=True, exist_ok=True)
    
    results = []
    results.append(check_system("small_oligomer_medium_100ps", results_root))
    
    with (results_root / "m10_2_1_summary.json").open("w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
