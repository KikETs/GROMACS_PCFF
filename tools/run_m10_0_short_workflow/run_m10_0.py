from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys
import os

# Add repo root to sys.path to import from tools
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.pcff_fixture_bridge.common import (
    build_typed_ir,
    render_gromacs_topology,
    parse_lammps_data,
    ANGSTROM_TO_NM,
)

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
            f.write(f"{1:>5d}{'MOL':<5s}{f'A{i}':>5s}{atom['id']:>5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write(f"{box_x:10.5f}{box_y:10.5f}{box_z:10.5f}\n")

def write_mdp(path: Path, content: str):
    path.write_text(content)

MDP_MIN = """
integrator  = steep
nsteps      = 100
emtol       = 10.0
emstep      = 0.01
cutoff-scheme = Verlet
vdw-type     = Cut-off
rvdw         = 0.9
coulombtype  = PME
rcoulomb     = 0.9
pbc          = xyz
"""

MDP_EQUIL = """
integrator  = md
dt          = 0.001
nsteps      = 100
cutoff-scheme = Verlet
vdw-type     = Cut-off
rvdw         = 0.9
coulombtype  = PME
rcoulomb     = 0.9
pbc          = xyz
tcoupl      = v-rescale
tc-grps     = System
tau-t       = 0.1
ref-t       = 300
gen-vel     = yes
gen-temp    = 300
"""

MDP_PROD = """
integrator  = md
dt          = 0.001
nsteps      = 100
cutoff-scheme = Verlet
vdw-type     = Cut-off
rvdw         = 0.9
coulombtype  = PME
rcoulomb     = 0.9
pbc          = xyz
tcoupl      = v-rescale
tc-grps     = System
tau-t       = 0.1
ref-t       = 300
"""

def run_gmx(cmd: list[str], work_dir: Path, log_name: str):
    gmx_bin = str(REPO_ROOT / "build" / "bin" / "gmx")
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    
    full_cmd = [gmx_bin] + cmd
    print(f"Executing: {' '.join(full_cmd)}")
    
    result = subprocess.run(
        full_cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
        env=env,
        errors="replace"
    )
    
    (work_dir / f"{log_name}.stdout").write_text(result.stdout)
    (work_dir / f"{log_name}.stderr").write_text(result.stderr)
    
    if result.returncode != 0:
        print(f"Error in {log_name}: {result.stderr}")
        return False, result.stdout + result.stderr
    return True, result.stdout + result.stderr

def check_stability(output: str):
    if "NaN" in output or "Inf" in output:
        return False, "Found NaN or Inf in output"
    return True, "Stable"

def check_system(system_id: str, corpus_root: Path, results_root: Path):
    system_dir = corpus_root / "systems" / system_id
    work_dir = results_root / system_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- M10.0 Workflow Gate for {system_id} ---")
    
    # 1. Topology Generation
    typed_ir = build_typed_ir({"id": system_id, "path": f"systems/{system_id}"}, corpus_root)
    top_content = render_gromacs_topology(typed_ir)
    (work_dir / "system.top").write_text(top_content)
    
    # 2. GRO Generation
    create_gro_from_lammps(system_dir / "lammps" / "system.data", work_dir / "system.gro")
    
    stages = [
        ("min", MDP_MIN, "system.gro", "system.top"),
        ("equil", MDP_EQUIL, "min.gro", "system.top"),
        ("prod", MDP_PROD, "equil.gro", "system.top")
    ]
    
    results = {
        "system_id": system_id,
        "stages": {}
    }
    
    last_gro = "system.gro"
    for stage_name, mdp_content, input_gro, top_file in stages:
        print(f"Running stage: {stage_name}")
        mdp_file = work_dir / f"{stage_name}.mdp"
        mdp_file.write_text(mdp_content)
        
        # grompp
        success, output = run_gmx([
            "grompp", "-f", f"{stage_name}.mdp", "-c", input_gro, "-p", top_file, "-o", f"{stage_name}.tpr", "-maxwarn", "10"
        ], work_dir, f"grompp_{stage_name}")
        
        if not success:
            results["stages"][stage_name] = {"status": "fail", "reason": "grompp failed"}
            break
            
        # mdrun
        success, output = run_gmx([
            "mdrun", "-s", f"{stage_name}.tpr", "-c", f"{stage_name}.gro", "-e", f"{stage_name}.edr", "-g", f"{stage_name}.log", "-o", f"{stage_name}.trr"
        ], work_dir, f"mdrun_{stage_name}")
        
        if not success:
            results["stages"][stage_name] = {"status": "fail", "reason": "mdrun failed"}
            break
            
        stable, message = check_stability(output)
        results["stages"][stage_name] = {"status": "pass", "stability": message}
        if not stable:
            print(f"Unstable: {message}")
            break
            
        last_gro = f"{stage_name}.gro"

    summary_file = work_dir / "workflow_summary.json"
    with summary_file.open("w") as f:
        json.dump(results, f, indent=2)
    
    final_status = "pass" if len(results["stages"]) == 3 and all(s["status"] == "pass" for s in results["stages"].values()) else "fail"
    print(f"Final Status: {final_status}\n")
    return results

def main():
    corpus_root = REPO_ROOT / "testdata" / "lammps_golden"
    results_root = REPO_ROOT / "tests" / "reference_results" / "m10_0_short_workflow"
    
    systems = ["small_oligomer", "small_salt_polymer_box"]
    
    aggregate = []
    for sid in systems:
        res = check_system(sid, corpus_root, results_root)
        aggregate.append(res)
        
    with (results_root / "m10_0_summary.json").open("w") as f:
        json.dump(aggregate, f, indent=2)

if __name__ == "__main__":
    main()
