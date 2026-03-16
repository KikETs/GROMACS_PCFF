#!/usr/bin/env python3
"""
TP1.2 - Charged Long-Equilibration Rerun Script
Using custom GROMACS 2027.0-dev with PCFF/Class2 support.
Refined to avoid numpy dependency.
"""

import os
import subprocess
import json
import shutil
import argparse
import math

def run_command(cmd, cwd=None):
    print(f"Executing: {' '.join(cmd)}")
    if cmd[0] == "gmx":
        cmd[0] = "/home/kiket/바탕화면/test/GROMACS_PCFF/build/bin/gmx"
    
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise Exception(f"Command failed with return code {result.returncode}")
    return result.stdout

def create_mdp(filename, duration_ps=5000):
    mdp_content = f"""
; TP1.2 MDP - 5ns NPT Equilibration
integrator              = md
dt                      = 0.001     ; 1.0 fs
nsteps                  = {int(duration_ps / 0.001)}

; Neighbor searching
cutoff-scheme           = Verlet
ns_type                 = grid
nstlist                 = 10
rcoulomb                = 0.9
rvdw                    = 0.9

; Electrostatics
coulombtype             = PME
pme_order               = 4
fourierspacing          = 0.12
ewald_rtol              = 1e-5

; VdW
vdw-type                = Cut-off
vdw-modifier            = None
DispCorr                = no

; Temperature coupling
tcouple                 = v-rescale
tc-grps                 = System
tau_t                   = 0.5
ref_t                   = 300

; Pressure coupling
pcouple                 = berendsen
pcoupletype             = isotropic
tau_p                   = 5.0
compressibility         = 4.5e-5
ref_p                   = 1.0

; Velocity generation
gen_vel                 = yes
gen_temp                = 300
gen_seed                = -1
"""
    with open(filename, 'w') as f:
        f.write(mdp_content)

def linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi*yi for xi, yi in zip(x, y))
    sum_xx = sum(xi*xi for xi in x)
    
    denominator = (n * sum_xx - sum_x * sum_x)
    if denominator == 0:
        return 0, 0
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept

def analyze_recovery(xvg_file, output_path, duration_ps):
    data = []
    with open(xvg_file, 'r') as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            cols = line.split()
            if len(cols) >= 5:
                data.append([float(x) for x in cols])
    
    # Analysis in 1ns blocks (1000ps)
    num_blocks = int(duration_ps / 1000)
    blocks = []
    drift_csv = os.path.join(output_path, "drift_analysis.csv")
    
    with open(drift_csv, 'w') as f:
        f.write("block_idx,start_ps,end_ps,dens_mean,dens_drift_per_100ps,pot_eng_mean,temp_mean\n")
        
        for i in range(num_blocks):
            start_t = i * 1000
            end_t = (i + 1) * 1000
            
            b_data = [row for row in data if start_t <= row[0] < end_t]
            if len(b_data) < 2:
                continue
            
            b_time = [row[0] for row in b_data]
            b_dens = [row[1] for row in b_data]
            b_pot = [row[2] for row in b_data]
            b_temp = [row[4] for row in b_data]
            
            slope, _ = linear_regression(b_time, b_dens)
            drift_100ps = slope * 100
            
            mean_dens = sum(b_dens) / len(b_dens)
            mean_pot = sum(b_pot) / len(b_pot)
            mean_temp = sum(b_temp) / len(b_temp)
            
            blocks.append({
                "index": i,
                "window_ns": [start_t/1000, end_t/1000],
                "density_mean": mean_dens,
                "density_drift_per_100ps": drift_100ps,
                "potential_energy_mean": mean_pot,
                "temperature_mean": mean_temp
            })
            
            f.write(f"{i},{start_t},{end_t},{mean_dens},{drift_100ps},{mean_pot},{mean_temp}\n")

    last_block = blocks[-1]
    rel_drift = abs(last_block["density_drift_per_100ps"]) / last_block["density_mean"] * 100

    if rel_drift < 0.05:
        overall_status = "production-entry ready"
    elif rel_drift < 0.2:
        overall_status = "partial / extend equilibration"
    else:
        overall_status = "unresolved / unstable"

    summary = {
        "milestone": "TP1.2",
        "system_id": "dense_salt_polymer",
        "equilibration_duration_ns": duration_ps / 1000.0,
        "recovery_status": {
            "overall": overall_status,
            "final_relative_drift_pct_per_100ps": rel_drift
        },
        "block_analysis": {
            "units": {"density": "kg/m^3", "potential_energy": "kJ/mol", "temperature": "K"},
            "blocks": blocks
        }
    }
    
    with open(os.path.join(output_path, "recovery_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return overall_status

def main():
    parser = argparse.ArgumentParser(description="Run TP1.2 recovery.")
    parser.add_argument("--system", default="dense_salt_polymer", help="Target system ID")
    parser.add_argument("--duration_ps", type=float, default=5000.0, help="Equilibration duration in ps")
    parser.add_argument("--output_dir", default="tests/reference_results/tp1_charged_recovery/dense_salt_polymer", help="Output directory")
    args = parser.parse_args()

    project_root = "/home/kiket/바탕화면/test/GROMACS_PCFF"
    output_path = os.path.join(project_root, args.output_dir)
    os.makedirs(output_path, exist_ok=True)

    src_dir = os.path.join(project_root, "tests/reference_results/m10_4_charged_ensemble_gate/dense_salt_polymer")
    top_file = os.path.join(src_dir, "system.top")
    gro_file = os.path.join(src_dir, "system.gro")

    shutil.copy(top_file, os.path.join(output_path, "system.top"))
    shutil.copy(gro_file, os.path.join(output_path, "system.gro"))

    # 1. Minimization
    min_mdp = os.path.join(output_path, "min.mdp")
    with open(min_mdp, 'w') as f:
        f.write("integrator = steep\nemtol = 100.0\nnsteps = 500\ncutoff-scheme = Verlet\nrcoulomb = 0.9\nrvdw = 0.9\n")
    
    run_command(["gmx", "grompp", "-f", "min.mdp", "-c", "system.gro", "-p", "system.top", "-o", "min.tpr", "-maxwarn", "5"], cwd=output_path)
    run_command(["gmx", "mdrun", "-s", "min.tpr", "-deffnm", "min", "-ntmpi", "1"], cwd=output_path)

    # 2. Long Equilibration
    equil_mdp = os.path.join(output_path, "tp1_equil.mdp")
    create_mdp(equil_mdp, duration_ps=args.duration_ps)

    run_command(["gmx", "grompp", "-f", "tp1_equil.mdp", "-c", "min.gro", "-p", "system.top", "-o", "tp1_equil.tpr", "-maxwarn", "5"], cwd=output_path)
    run_command(["gmx", "mdrun", "-s", "tp1_equil.tpr", "-deffnm", "tp1_equil", "-ntmpi", "1"], cwd=output_path)

    # 3. Post-processing
    gmx_bin = "/home/kiket/바탕화면/test/GROMACS_PCFF/build/bin/gmx"
    energy_input = "Density\nPotential\nVolume\nTemperature\n0\n"
    process = subprocess.Popen([gmx_bin, "energy", "-f", "tp1_equil.edr", "-o", "energy.xvg"], 
                               cwd=output_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    process.communicate(input=energy_input)

    # 4. Analyze
    status = analyze_recovery(os.path.join(output_path, "energy.xvg"), output_path, args.duration_ps)
    print(f"Recovery status: {status}")

    # 5. Manifest
    manifest = {
        "system_id": args.system,
        "atom_count": 270,
        "composition": "Na/Cl salt in polymer matrix",
        "source": "M10.4 provenance",
        "verification": "TP1.2 authoritative rerun"
    }
    with open(os.path.join(output_path, "system_manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)

    print("TP1.2 Run Completed.")

if __name__ == "__main__":
    main()
