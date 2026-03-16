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

def get_mdp_min():
    return """
integrator  = steep
nsteps      = 500
emtol       = 100.0
emstep      = 0.01
cutoff-scheme = Verlet
vdw-type     = Cut-off
rvdw         = 0.9
coulombtype  = PME
rcoulomb     = 0.9
pbc          = xyz
"""

def get_mdp_npt(dt: float, nsteps: int):
    return f"""
integrator  = md
dt          = {dt}
nsteps      = {nsteps}
cutoff-scheme = Verlet
vdw-type     = Cut-off
rvdw         = 0.9
coulombtype  = PME
rcoulomb     = 0.9
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

def run_lammps_async(work_dir: Path, dt: float, nsteps: int, nprocs: int = 12):
    cmd = ["mpirun", "-np", str(nprocs), "lmp", "-in", "run.in"]
    return subprocess.Popen(cmd, cwd=work_dir, stdout=open(work_dir/"lammps.stdout", "w"), stderr=open(work_dir/"lammps.stderr", "w"))

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
    if len(data) == 0: return {}
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
    
    drifts = []
    for i in range(data.shape[1]):
        if nblocks > 1:
            try:
                slope = np.polyfit(range(nblocks), block_means[:, i], 1)[0]
                drifts.append(slope / (overall_std[i] if overall_std[i] > 0 else 1.0))
            except:
                drifts.append(0.0)
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
    if not stats: return "unknown"
    drift = abs(stats["normalized_drifts"][field_idx])
    if drift < threshold:
        return "converged enough"
    elif drift < threshold * 3:
        return "trending but not converged"
    else:
        return "failed / unstable"

def check_system(sid: str, corpus_root: Path, results_root: Path):
    work_dir = results_root / sid
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- M10.4 Charged Ensemble Recovery Gate for {sid} ---")
    system_dir = corpus_root / "systems" / sid
    
    dt = 0.001 
    nsteps_equil = 20000 # 20 ps
    nsteps_prod = 100000 # 100 ps
    
    # GMX
    ir = build_typed_ir({"id": sid, "path": f"systems/{sid}"}, corpus_root)
    top_content = render_gromacs_topology(ir)
    (work_dir / "system.top").write_text(top_content)
    
    from tools.run_m10_0_short_workflow.run_m10_0 import create_gro_from_lammps
    create_gro_from_lammps(system_dir / "lammps" / "system.data", work_dir / "system.gro")
    
    # Min
    work_dir.joinpath("min.mdp").write_text(get_mdp_min())
    print("Running GROMACS Min...")
    run_gmx(["grompp", "-f", "min.mdp", "-c", "system.gro", "-p", "system.top", "-o", "min.tpr", "-maxwarn", "10"], work_dir, "grompp_min")
    run_gmx(["mdrun", "-s", "min.tpr", "-c", "min.gro", "-e", "min.edr", "-g", "min.log"], work_dir, "mdrun_min")

    # Equil
    work_dir.joinpath("equil.mdp").write_text(get_mdp_npt(dt, nsteps_equil))
    print(f"Running GROMACS Equil ({nsteps_equil} steps)...")
    run_gmx(["grompp", "-f", "equil.mdp", "-c", "min.gro", "-p", "system.top", "-o", "equil.tpr", "-maxwarn", "10"], work_dir, "grompp_equil")
    run_gmx(["mdrun", "-s", "equil.tpr", "-c", "equil.gro", "-e", "equil.edr", "-g", "equil.log"], work_dir, "mdrun_equil")
    
    # Prod
    work_dir.joinpath("run.mdp").write_text(get_mdp_npt(dt, nsteps_prod))
    run_gmx(["grompp", "-f", "run.mdp", "-c", "equil.gro", "-p", "system.top", "-o", "run.tpr", "-maxwarn", "10"], work_dir, "grompp")
    
    print(f"Starting GROMACS and LAMMPS production ({nsteps_prod} steps)...")
    shutil.copy(system_dir / "lammps" / "system.data", work_dir / "system.data")
    
    # LAMMPS setup
    orig_in = system_dir / "lammps" / "system.in"
    with orig_in.open("r") as f:
        lines = f.readlines()
    with (work_dir / "run.in").open("w") as f:
        for line in lines:
            if line.startswith("read_data"):
                f.write("read_data system.data\n")
            elif any(x in line for x in ["timestep", "run", "fix"]): continue
            else: f.write(line)
        f.write(f"\ntimestep {dt * 1000}\n")
        f.write("minimize 1.0e-4 1.0e-6 100 1000\n")
        f.write("fix 1 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0\n")
        f.write("thermo 100\n")
        f.write("thermo_style custom step temp pe ke epair ebond eangle edihed press vol density\n")
        f.write(f"run {nsteps_equil + nsteps_prod}\n")

    # Run both
    gmx_bin = str(REPO_ROOT / "build" / "bin" / "gmx")
    p_gmx = subprocess.Popen([gmx_bin, "mdrun", "-s", "run.tpr", "-c", "out.gro", "-e", "run.edr", "-g", "run.log", "-nt", "12"], cwd=work_dir, stdout=open(work_dir/"mdrun.stdout", "w"), stderr=open(work_dir/"mdrun.stderr", "w"))
    p_lmp = run_lammps_async(work_dir, dt, nsteps_equil + nsteps_prod, nprocs=12)
    
    p_gmx.wait()
    p_lmp.wait()
    
    # Analysis
    energy_input = "Potential\nTemperature\nPressure\nVolume\nDensity\n0\n"
    subprocess.run([gmx_bin, "energy", "-f", "run.edr", "-o", "energy.xvg"], cwd=work_dir, input=energy_input, capture_output=True, text=True)
    gmx_data = parse_xvg(work_dir / "energy.xvg")
    
    lmp_raw = []
    if (work_dir / "log.lammps").exists():
        with (work_dir / "log.lammps").open("r") as f:
            in_thermo = False
            for line in f:
                clean_line = line.strip()
                if clean_line.startswith("Step"): in_thermo = True; continue
                if clean_line.startswith("Loop time"): in_thermo = False; continue
                if in_thermo:
                    parts = clean_line.split()
                    if parts and parts[0].isdigit() and len(parts) == 11:
                        lmp_raw.append([float(x) for x in parts])
    lmp_data = np.array(lmp_raw)
    
    gmx_stats = analyze_ensemble_blocks(gmx_data)
    lmp_stats = analyze_ensemble_blocks(lmp_data[nsteps_equil // 100 :])
    
    # GMX: 1:Pot, 2:Temp, 4:Volume, 5:Density
    # LMP: 2:PE, 1:Temp, 9:Volume, 10:Density
    
    dens_gmx = gmx_stats["mean"][5]
    dens_lmp = lmp_stats["mean"][10] * 1000
    dens_diff_rel = abs(dens_gmx - dens_lmp) / dens_gmx

    vol_gmx = gmx_stats["mean"][4]
    vol_lmp = lmp_stats["mean"][9] / 1000
    vol_diff_rel = abs(vol_gmx - vol_lmp) / vol_gmx
    
    report = {
        "system_id": sid,
        "duration_ps": nsteps_prod * dt,
        "gmx": {
            "potential_energy": {"mean": gmx_stats["mean"][1], "sem": gmx_stats["sem"][1], "status": get_convergence_status(gmx_stats, 1)},
            "temperature": {"mean": gmx_stats["mean"][2], "sem": gmx_stats["sem"][2], "status": get_convergence_status(gmx_stats, 2)},
            "volume": {"mean": vol_gmx, "sem": gmx_stats["sem"][4], "status": get_convergence_status(gmx_stats, 4)},
            "density": {"mean": dens_gmx, "sem": gmx_stats["sem"][5], "status": get_convergence_status(gmx_stats, 5)}
        },
        "lmp": {
            "potential_energy": {"mean": lmp_stats["mean"][2] * KCAL_TO_KJ, "sem": lmp_stats["sem"][2] * KCAL_TO_KJ, "status": get_convergence_status(lmp_stats, 2)},
            "temperature": {"mean": lmp_stats["mean"][1], "sem": lmp_stats["sem"][1], "status": get_convergence_status(lmp_stats, 1)},
            "volume": {"mean": vol_lmp, "sem": lmp_stats["sem"][9] / 1000, "status": get_convergence_status(lmp_stats, 9)},
            "density": {"mean": dens_lmp, "sem": lmp_stats["sem"][10] * 1000, "status": get_convergence_status(lmp_stats, 10)}
        },
        "density_parity_rel_diff": dens_diff_rel,
        "volume_parity_rel_diff": vol_diff_rel,
        "parity_status": "pass" if dens_diff_rel < 0.10 else "partial" # Looser for charged
    }
    
    with (work_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Results for {sid}:")
    print(f"  GMX Density: {dens_gmx:.2f} kg/m3")
    print(f"  LMP Density: {dens_lmp:.2f} kg/m3")
    print(f"  Rel Diff: {dens_diff_rel:.2%}")
    print(f"  Status: {report['parity_status']}")
    
    return report

def main():
    corpus_root = REPO_ROOT / "testdata" / "lammps_golden"
    results_root = REPO_ROOT / "tests" / "reference_results" / "m10_4_charged_ensemble_gate"
    results_root.mkdir(parents=True, exist_ok=True)
    
    results = []
    results.append(check_system("dense_salt_polymer", corpus_root, results_root))
    
    with (results_root / "m10_4_summary.json").open("w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
