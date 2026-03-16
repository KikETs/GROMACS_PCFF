from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys
import os
import re
import math
import shutil

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
            f.write(f"{1:>5d}{'MOL':<5s}{f'A{i}':>5s}{atom['id']:>5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write(f"{box_x:10.5f}{box_y:10.5f}{box_z:10.5f}\n")

def get_mdp(protocol: str, dt: float, nsteps: int, has_kspace: bool):
    coulomb_type = "PME" if has_kspace else "Cut-off"
    
    mdp = f"""
integrator  = md
dt          = {dt}
nsteps      = {nsteps}
cutoff-scheme = Verlet
vdw-type     = Cut-off
rvdw         = 0.9
coulombtype  = {coulomb_type}
rcoulomb     = 0.9
pbc          = xyz
nstxout      = 10
nstvout      = 0
nstfout      = 0
nstenergy    = 10
"""
    if protocol == "nve":
        mdp += "tcoupl = no\npcoupl = no\n"
    elif protocol == "nvt":
        mdp += """
tcoupl      = v-rescale
tc-grps     = System
tau-t       = 0.1
ref-t       = 300
gen-vel     = yes
gen-temp    = 300
"""
    elif protocol == "npt":
        mdp += """
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
    return mdp

def run_gmx(cmd: list[str], work_dir: Path, log_name: str):
    gmx_bin = str(REPO_ROOT / "build" / "bin" / "gmx")
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    
    full_cmd = [gmx_bin] + cmd
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
    return result.returncode == 0

def run_lammps(system_dir: Path, work_dir: Path, protocol: str, dt: float, nsteps: int):
    orig_in = system_dir / "lammps" / "system.in"
    data_file = system_dir / "lammps" / "system.data"
    
    shutil_copy = True # Just use shutil
    import shutil
    shutil.copy(data_file, work_dir / "system.data")
    
    with orig_in.open("r") as f:
        lines = f.readlines()
    
    new_in = work_dir / "system.in"
    with new_in.open("w") as f:
        for line in lines:
            if line.startswith("read_data"):
                f.write("read_data system.data\n")
            elif line.startswith("timestep"):
                continue
            elif "run" in line and not line.strip().startswith("#"):
                continue
            elif "fix" in line and ("nvt" in line or "nve" in line or "npt" in line):
                continue
            else:
                f.write(line)
        
        f.write(f"\ntimestep {dt * 1000}\n") # GMX dt is ps, LAMMPS real units dt is fs
        
        if protocol == "nve":
            f.write("fix 1 all nve\n")
        elif protocol == "nvt":
            f.write("fix 1 all nvt temp 300.0 300.0 100.0\n")
        elif protocol == "npt":
            f.write("fix 1 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0\n")
            
        f.write("thermo 10\n")
        f.write("thermo_style custom step temp pe ke etotal press vol\n")
        f.write(f"run {nsteps}\n")
        
    subprocess.run(["lmp", "-in", "system.in"], cwd=work_dir, capture_output=True, text=True, check=True)

def parse_xvg(path: Path):
    data = []
    if not path.exists(): return data
    with path.open("r") as f:
        for line in f:
            if line.startswith(("#", "@")): continue
            parts = line.split()
            if parts:
                data.append([float(x) for x in parts])
    return data

def check_system(system_id: str, protocol: str, dt: float, nsteps: int, corpus_root: Path, results_root: Path):
    test_id = f"{system_id}_{protocol}_dt{dt}"
    work_dir = results_root / test_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- M10.1: {test_id} ---")
    
    system_dir = corpus_root / "systems" / system_id
    typed_ir = build_typed_ir({"id": system_id, "path": f"systems/{system_id}"}, corpus_root)
    has_kspace = typed_ir["styles"]["kspace_style"] is not None
    
    # GMX
    (work_dir / "system.top").write_text(render_gromacs_topology(typed_ir))
    create_gro_from_lammps(system_dir / "lammps" / "system.data", work_dir / "system.gro")
    (work_dir / "run.mdp").write_text(get_mdp(protocol, dt, nsteps, has_kspace))
    
    success = run_gmx(["grompp", "-f", "run.mdp", "-c", "system.gro", "-p", "system.top", "-o", "run.tpr", "-maxwarn", "10"], work_dir, "grompp")
    if not success: return {"test_id": test_id, "status": "fail", "reason": "grompp"}
    
    success = run_gmx(["mdrun", "-s", "run.tpr", "-c", "out.gro", "-e", "run.edr", "-g", "run.log", "-o", "run.trr"], work_dir, "mdrun")
    if not success: return {"test_id": test_id, "status": "fail", "reason": "mdrun"}
    
    # Extract GMX energies
    energy_input = "Potential\nKinetic-En.\nTemperature\nPressure\nVolume\n0\n"
    subprocess.run([str(REPO_ROOT / "build" / "bin" / "gmx"), "energy", "-f", "run.edr", "-o", "energy.xvg"], cwd=work_dir, input=energy_input, capture_output=True, text=True)
    gmx_data = parse_xvg(work_dir / "energy.xvg")
    
    # LAMMPS
    run_lammps(system_dir, work_dir, protocol, dt, nsteps)
    # Parse log.lammps
    lmp_data = []
    log_lmp = work_dir / "log.lammps"
    if log_lmp.exists():
        with log_lmp.open("r") as f:
            in_thermo = False
            for line in f:
                clean_line = line.strip()
                if clean_line.startswith("Step"): 
                    in_thermo = True
                    continue
                if clean_line.startswith("Loop time"): 
                    in_thermo = False
                    continue
                if in_thermo:
                    parts = clean_line.split()
                    if parts and parts[0].isdigit():
                        try:
                            lmp_data.append([float(x) for x in parts])
                        except ValueError:
                            continue
                
    # Summary
    # Step Temp PotEng KinEng TotEng Press Volume (LAMMPS)
    # 0    1    2      3      4      5     6
    # Time Pot  Kin    Temp   Press  Vol (GMX)
    # 0    1    2      3      4      5
    
    # Compare start/end PE
    lmp_pe_start = lmp_data[0][2] * KCAL_TO_KJ
    lmp_pe_end = lmp_data[-1][2] * KCAL_TO_KJ
    gmx_pe_start = gmx_data[0][1]
    gmx_pe_end = gmx_data[-1][1]
    
    report = {
        "test_id": test_id,
        "status": "pass",
        "dt": dt,
        "pe_start": {"lmp": lmp_pe_start, "gmx": gmx_pe_start, "diff": abs(lmp_pe_start - gmx_pe_start)},
        "pe_end": {"lmp": lmp_pe_end, "gmx": gmx_pe_end, "diff": abs(lmp_pe_end - gmx_pe_end)},
    }
    dump_json(work_dir / "report.json", report)
    return report

def dump_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

def main():
    corpus_root = REPO_ROOT / "testdata" / "lammps_golden"
    results_root = REPO_ROOT / "tests" / "reference_results" / "m10_1_trajectory_gate"
    
    # Protocols for neutral system (small_oligomer)
    results = []
    # NVE for timestep convergence
    results.append(check_system("small_oligomer", "nve", 0.0001, 100, corpus_root, results_root)) # 0.1 fs
    results.append(check_system("small_oligomer", "nve", 0.0005, 100, corpus_root, results_root)) # 0.5 fs
    # NVT for trajectory trend
    results.append(check_system("small_oligomer", "nvt", 0.001, 100, corpus_root, results_root)) # 1.0 fs
    # NPT for barostat sanity
    results.append(check_system("small_oligomer", "npt", 0.001, 100, corpus_root, results_root))
    
    # Optional charged case (small_salt_polymer_box)
    results.append(check_system("small_salt_polymer_box", "nvt", 0.001, 100, corpus_root, results_root))
    
    dump_json(results_root / "m10_1_summary.json", results)

if __name__ == "__main__":
    main()
