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
            # Use 15.7f for high precision coordinates in .gro
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

def run_gmx(cmd: list[str], work_dir: Path, log_name: str):
    gmx_bin = str(REPO_ROOT / "build" / "bin" / "gmx")
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    full_cmd = [gmx_bin] + cmd
    result = subprocess.run(full_cmd, cwd=work_dir, capture_output=True, text=True, env=env, errors="replace")
    (work_dir / f"{log_name}.stdout").write_text(result.stdout)
    (work_dir / f"{log_name}.stderr").write_text(result.stderr)
    return result.returncode == 0

def analyze_ensemble(data, start_idx):
    subset = data[start_idx:]
    if len(subset) == 0: return np.array([]), np.array([])
    means = np.mean(subset, axis=0)
    stds = np.std(subset, axis=0)
    return means, stds

def parse_xvg(path: Path):
    data = []
    if not path.exists(): return data
    with path.open("r") as f:
        for line in f:
            if line.startswith(("#", "@")): continue
            parts = line.split()
            if parts: data.append([float(x) for x in parts])
    return np.array(data)

def check_system(sid: str, corpus_root: Path, results_root: Path):
    work_dir = results_root / sid
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- M10.2 Ensemble Validation for {sid} ---")
    
    dt = 0.001 
    nsteps_equil = 2000 
    nsteps_prod = 8000 
    
    system_dir = corpus_root / "systems" / "small_oligomer"
    
    # Replicate to medium scale (4x4x4 = 64 molecules)
    def replicate_lammps(system_dir: Path, work_dir: Path, nx, ny, nz):
        data_file = system_dir / "lammps" / "system.data"
        shutil.copy(data_file, work_dir / "orig.data")
        gen_in = work_dir / "replicate.in"
        with gen_in.open("w") as f:
            f.write("units real\natom_style full\nread_data orig.data\n")
            f.write(f"replicate {nx} {ny} {nz}\n")
            f.write("write_data system.data nocoeff\n")
        subprocess.run(["lmp", "-in", "replicate.in"], cwd=work_dir, capture_output=True, text=True, check=True)

    replicate_lammps(system_dir, work_dir, 4, 4, 4)
    
    # GMX
    ir = build_typed_ir({"id": "small_oligomer", "path": "systems/small_oligomer"}, corpus_root)
    top_content = render_gromacs_topology(ir)
    top_content = top_content.replace("OLI 1", "OLI 64")
    (work_dir / "system.top").write_text(top_content)
    
    create_gro_from_lammps(work_dir / "system.data", work_dir / "system.gro")
    
    # Equil
    (work_dir / "equil.mdp").write_text(get_mdp_npt(dt, nsteps_equil))
    print(f"Running GROMACS Equil ({nsteps_equil} steps)...")
    run_gmx(["grompp", "-f", "equil.mdp", "-c", "system.gro", "-p", "system.top", "-o", "equil.tpr", "-maxwarn", "10"], work_dir, "grompp_equil")
    run_gmx(["mdrun", "-s", "equil.tpr", "-c", "equil.gro", "-e", "equil.edr", "-g", "equil.log", "-nt", "8"], work_dir, "mdrun_equil")

    # Prod
    (work_dir / "run.mdp").write_text(get_mdp_npt(dt, nsteps_prod))
    print(f"Running GROMACS Prod ({nsteps_prod} steps)...")
    run_gmx(["grompp", "-f", "run.mdp", "-c", "equil.gro", "-p", "system.top", "-o", "run.tpr", "-maxwarn", "10"], work_dir, "grompp")
    run_gmx(["mdrun", "-s", "run.tpr", "-c", "out.gro", "-e", "run.edr", "-g", "run.log", "-nt", "8"], work_dir, "mdrun")
    
    energy_input = "Potential\nTemperature\nPressure\nVolume\nDensity\n0\n"
    subprocess.run([str(REPO_ROOT / "build" / "bin" / "gmx"), "energy", "-f", "run.edr", "-o", "energy.xvg"], cwd=work_dir, input=energy_input, capture_output=True, text=True)
    gmx_data = parse_xvg(work_dir / "energy.xvg")
    
    # LAMMPS
    def run_lammps_npt(work_dir: Path, dt: float, n_equil: int, n_prod: int):
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
kspace_style pppm 1.0e-6
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
            f.write(f"run {n_equil + n_prod}\n")
        subprocess.run(["lmp", "-in", "system.in"], cwd=work_dir, capture_output=True, text=True, check=True)

    print(f"Running LAMMPS NPT ({nsteps_equil + nsteps_prod} steps)...")
    run_lammps_npt(work_dir, dt, nsteps_equil, nsteps_prod)
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
    
    # Analysis (Skip equilibration steps)
    gmx_means, gmx_stds = analyze_ensemble(gmx_data, 0)
    lmp_means, lmp_stds = analyze_ensemble(lmp_data, nsteps_equil // 100)
    
    report = {
        "system_id": sid,
        "nsteps_prod": nsteps_prod,
        "potential_energy": {"gmx": float(gmx_means[1]), "lmp": float(lmp_means[2]) * KCAL_TO_KJ, "diff_rel": float(abs(gmx_means[1] - lmp_means[2]*KCAL_TO_KJ)/abs(gmx_means[1]))},
        "temperature": {"gmx": float(gmx_means[2]), "lmp": float(lmp_means[1]), "diff": float(abs(gmx_means[2] - lmp_means[1]))},
        "density": {"gmx": float(gmx_means[5]), "lmp": float(lmp_means[7]) * 1000, "diff_rel": float(abs(gmx_means[5] - lmp_means[7]*1000)/gmx_means[5])},
        "status": "pass" if abs(gmx_means[2] - lmp_means[1]) < 10.0 else "fail"
    }
    
    with (work_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Ensemble Parity Status: {report['status']}")
    print(f"Mean Temp Diff: {report['temperature']['diff']:.2f} K")
    print(f"Mean Density Rel Diff: {report['density']['diff_rel']:.4%}")
    return report

def dump_json(path, data):
    with open(path, "w") as f: json.dump(data, f, indent=2)

def main():
    corpus_root = REPO_ROOT / "testdata" / "lammps_golden"
    results_root = REPO_ROOT / "tests" / "reference_results" / "m10_2_ensemble_gate"
    results = []
    results.append(check_system("small_oligomer_medium", corpus_root, results_root))
    dump_json(results_root / "m10_2_summary.json", results)

if __name__ == "__main__":
    main()
