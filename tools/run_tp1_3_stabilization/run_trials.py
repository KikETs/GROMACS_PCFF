#!/usr/bin/env python3
"""
TP1.3 - Stabilization Diagnostic Runner
"""

import os
import subprocess
import json
import shutil
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GMX_BIN = os.environ.get("GMX_BIN", str(REPO_ROOT / "build" / "bin" / "gmx"))

def run_command(cmd, cwd=None):
    # print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr
    return True, result.stdout

def create_mdp(filename, dt_fs=1.0, ensemble='NPT', duration_ps=500.0, gen_vel='yes', coulomb='PME', tau_t=0.5):
    dt = dt_fs / 1000.0
    nsteps = int(duration_ps / dt)
    
    pcouple = "berendsen" if ensemble == 'NPT' else "no"
    
    mdp_content = f"""
integrator              = md
dt                      = {dt}
nsteps                  = {nsteps}
cutoff-scheme           = Verlet
rcoulomb                = 0.9
rvdw                    = 0.9
coulombtype             = {coulomb}
pme_order               = 4
fourierspacing          = 0.12
ewald_rtol              = 1e-5
vdw-type                = Cut-off
DispCorr                = no
tcouple                 = v-rescale
tc-grps                 = System
tau_t                   = {tau_t}
ref_t                   = 300
pcouple                 = {pcouple}
pcoupletype             = isotropic
tau_p                   = 5.0
compressibility         = 4.5e-5
ref_p                   = 1.0
gen_vel                 = {gen_vel}
gen_temp                = 300
gen_seed                = -1
"""
    with open(filename, 'w') as f:
        f.write(mdp_content)

def analyze_xvg(xvg_file):
    data = []
    if not os.path.exists(xvg_file): return None
    with open(xvg_file, 'r') as f:
        for line in f:
            if line.startswith(('#', '@')): continue
            cols = line.split()
            if len(cols) >= 3: data.append([float(x) for x in cols])
    if not data: return None
    
    times = [r[0] for r in data]
    pots = [r[1] for r in data]
    temps = [r[2] for r in data]
    
    mean_temp = sum(temps) / len(temps)
    max_temp = max(temps)
    
    status = "STABLE"
    if max_temp > 400: status = "RUNAWAY"
    elif mean_temp > 350: status = "UNSTABLE"
    
    return {
        "mean_temp": mean_temp,
        "max_temp": max_temp,
        "final_temp": temps[-1],
        "status": status,
        "duration": times[-1]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", help="Run specific trial (TRL-0 to TRL-4)")
    args = parser.parse_args()

    project_root = str(REPO_ROOT)
    output_base = os.path.join(project_root, "tests/reference_results/tp1_3_stabilization")
    os.makedirs(output_base, exist_ok=True)

    # Auth system source
    src_dir = os.path.join(project_root, "tests/reference_results/m10_4_charged_ensemble_gate/dense_salt_polymer")
    top_file = os.path.join(src_dir, "system.top")
    gro_file = os.path.join(src_dir, "system.gro")

    trials = {
        "TRL-0": {"dt": 1.0, "ens": "NPT", "dur": 500.0, "note": "Baseline"},
        "TRL-1": {"dt": 0.5, "ens": "NPT", "dur": 500.0, "note": "dt=0.5fs"},
        "TRL-2": {"dt": 1.0, "ens": "NVT", "dur": 500.0, "note": "NVT only"},
        "TRL-3": {"dt": 0.5, "ens": "NVT", "dur": 500.0, "note": "NVT + dt=0.5"},
        "TRL-4": {"dt": 1.0, "ens": "NPT", "dur": 500.0, "note": "Relaxed start"},
        "TRL-5": {"dt": 1.0, "ens": "NPT", "dur": 500.0, "note": "Cut-off (No PME)"},
        "TRL-6": {"dt": 1.0, "ens": "NPT", "dur": 500.0, "note": "Strong Thermostat (tau=0.01)"}
    }

    if args.trial:
        run_list = [args.trial]
    else:
        run_list = list(trials.keys())

    matrix_results = []

    for tid in run_list:
        print(f"--- Running {tid} ({trials[tid]['note']}) ---")
        trial_dir = os.path.join(output_base, tid)
        os.makedirs(trial_dir, exist_ok=True)
        
        shutil.copy(top_file, os.path.join(trial_dir, "system.top"))
        shutil.copy(gro_file, os.path.join(trial_dir, "system.gro"))

        # Step 0: Minimization
        min_mdp = os.path.join(trial_dir, "min.mdp")
        with open(min_mdp, 'w') as f:
            f.write("integrator=steep\nemtol=500.0\nnsteps=200\ncutoff-scheme=Verlet\nrcoulomb=0.9\nrvdw=0.9\n")
        
        success, err = run_command([GMX_BIN, "grompp", "-f", "min.mdp", "-c", "system.gro", "-p", "system.top", "-o", "min.tpr", "-maxwarn", "5"], cwd=trial_dir)
        if not success:
            print(f"{tid} grompp-min failed: {err}")
            continue
        
        run_command([GMX_BIN, "mdrun", "-s", "min.tpr", "-deffnm", "min", "-ntmpi", "1"], cwd=trial_dir)

        # Step 1: Trial run
        mdp_file = os.path.join(trial_dir, "trial.mdp")
        t_conf = trials[tid]
        coulomb = "Cut-off" if tid == "TRL-5" else "PME"
        tau_t = 0.01 if tid == "TRL-6" else 0.5
        
        if tid == "TRL-4":
            # Extra relaxation: 100ps NVT at 0.5fs
            relax_mdp = os.path.join(trial_dir, "relax.mdp")
            create_mdp(relax_mdp, dt_fs=0.5, ensemble='NVT', duration_ps=100.0, coulomb=coulomb, tau_t=tau_t)
            run_command([GMX_BIN, "grompp", "-f", "relax.mdp", "-c", "min.gro", "-p", "system.top", "-o", "relax.tpr", "-maxwarn", "5"], cwd=trial_dir)
            run_command([GMX_BIN, "mdrun", "-s", "relax.tpr", "-deffnm", "relax", "-ntmpi", "1"], cwd=trial_dir)
            
            create_mdp(mdp_file, dt_fs=t_conf["dt"], ensemble=t_conf["ens"], duration_ps=t_conf["dur"], gen_vel='no', coulomb=coulomb, tau_t=tau_t)
            start_conf = "relax.gro"
        else:
            create_mdp(mdp_file, dt_fs=t_conf["dt"], ensemble=t_conf["ens"], duration_ps=t_conf["dur"], coulomb=coulomb, tau_t=tau_t)
            start_conf = "min.gro"

        success, err = run_command([GMX_BIN, "grompp", "-f", "trial.mdp", "-c", start_conf, "-p", "system.top", "-o", "trial.tpr", "-maxwarn", "5"], cwd=trial_dir)
        if not success:
            print(f"{tid} grompp failed: {err}")
            continue

        run_command([GMX_BIN, "mdrun", "-s", "trial.tpr", "-deffnm", "trial", "-ntmpi", "1"], cwd=trial_dir)

        # Extract stats
        energy_input = "Potential\nTemperature\n0\n"
        process = subprocess.Popen([GMX_BIN, "energy", "-f", "trial.edr", "-o", "energy.xvg"], 
                                   cwd=trial_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process.communicate(input=energy_input)

        analysis = analyze_xvg(os.path.join(trial_dir, "energy.xvg"))
        if analysis:
            analysis["trial_id"] = tid
            analysis["config"] = t_conf
            print(f"Result: {analysis['status']} (T_avg={analysis['mean_temp']:.1f}K, T_max={analysis['max_temp']:.1f}K)")
            
            with open(os.path.join(trial_dir, "summary.json"), 'w') as f:
                json.dump(analysis, f, indent=2)
            matrix_results.append(analysis)

    with open(os.path.join(output_base, "trial_matrix_results.json"), 'w') as f:
        json.dump(matrix_results, f, indent=2)

if __name__ == "__main__":
    main()
