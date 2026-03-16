from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys
import re

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

# 1 kcal/mol = 4.184 kJ/mol

def create_gro_from_lammps(lammps_data_path: Path, gro_path: Path):
    data = parse_lammps_data(lammps_data_path)
    box_x = 10.0
    box_y = 10.0
    box_z = 10.0
    
    with gro_path.open("w") as f:
        f.write("Generated from LAMMPS data with high precision\n")
        f.write(f"{len(data['atoms']):>5d}\n")
        for i, atom in enumerate(data["atoms"], start=1):
            x = atom["x_angstrom"] * ANGSTROM_TO_NM
            y = atom["y_angstrom"] * ANGSTROM_TO_NM
            z = atom["z_angstrom"] * ANGSTROM_TO_NM
            # Use 15.7f for high precision coordinates in .gro
            f.write(f"{1:>5d}{'MOL':<5s}{f'A{i}':>5s}{atom['id']:>5d}{x:15.7f}{y:15.7f}{z:15.7f}\n")
        f.write(f"{box_x:15.7f}{box_y:15.7f}{box_z:15.7f}\n")

def create_mdp(mdp_path: Path):
    content = """
integrator  = md
nsteps      = 0
cutoff-scheme = Verlet
vdw-type     = Cut-off
rvdw         = 2.0
coulombtype  = Cut-off
rcoulomb     = 2.0
pbc          = xyz
"""
    mdp_path.write_text(content)

def run_lammps(system_dir: Path, work_dir: Path) -> float:
    orig_in = system_dir / "lammps" / "system.in"
    new_in = work_dir / "system.in"
    data_file = system_dir / "lammps" / "system.data"
    
    local_data_file = work_dir / "system.data"
    local_data_file.write_bytes(data_file.read_bytes())
    
    with orig_in.open("r") as f:
        lines = f.readlines()
    
    with new_in.open("w") as f:
        for line in lines:
            if line.startswith("read_data"):
                f.write("read_data system.data\n")
            else:
                f.write(line)
        f.write("\nthermo_style custom step pe\n")
        f.write("run 0\n")
    
    result = subprocess.run(
        ["lmp", "-in", "system.in"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
        errors="replace"
    )
    
    pe = None
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        if "Step" in line and "PotEng" in line:
            for j in range(i + 1, len(lines)):
                values = lines[j].split()
                if len(values) >= 2 and values[0].isdigit():
                    pe = float(values[1])
                    break
            if pe is not None:
                break
    if pe is None:
        print(result.stdout)
        raise ValueError("Could not find potential energy in LAMMPS output")
    return pe

def run_gromacs(top_path: Path, gro_path: Path, mdp_path: Path, work_dir: Path) -> float:
    gmx_bin = str(REPO_ROOT / "build" / "bin" / "gmx")
    subprocess.run(
        [gmx_bin, "grompp", "-f", str(mdp_path), "-c", str(gro_path), "-p", str(top_path), "-o", "topol.tpr", "-maxwarn", "10"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
        errors="replace"
    )
    
    subprocess.run(
        [gmx_bin, "mdrun", "-s", "topol.tpr", "-rerun", str(gro_path), "-e", "ener.edr"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
        errors="replace"
    )
    
    energy_input = "Potential\n0\n"
    result = subprocess.run(
        [gmx_bin, "energy", "-f", "ener.edr", "-o", "energy.xvg"],
        input=energy_input,
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
        errors="replace"
    )
    
    pe = None
    lines = result.stdout.splitlines()
    for line in lines:
        if line.startswith("Potential"):
            tokens = line.split()
            if len(tokens) >= 2:
                pe = float(tokens[1])
                break
    
    if pe is None:
        print(result.stdout)
        raise ValueError("Could not find potential energy in GROMACS output")
    return pe

def check_system(system_id: str, corpus_root: Path, results_root: Path):
    system_dir = corpus_root / "systems" / system_id
    work_dir = results_root / system_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Running mixing parity check for {system_id} ---")
    
    typed_ir = build_typed_ir({"id": system_id, "path": f"systems/{system_id}"}, corpus_root)
    top_content = render_gromacs_topology(typed_ir)
    top_path = work_dir / "system.top"
    top_path.write_text(top_content)
    
    gro_path = work_dir / "system.gro"
    create_gro_from_lammps(system_dir / "lammps" / "system.data", gro_path)
    
    mdp_path = work_dir / "min.mdp"
    create_mdp(mdp_path)
    
    lammps_pe = run_lammps(system_dir, work_dir)
    lammps_pe_kj = lammps_pe * KCAL_TO_KJ
    
    gromacs_pe = run_gromacs(top_path, gro_path, mdp_path, work_dir)
    
    print(f"LAMMPS Potential Energy: {lammps_pe_kj:.6f} kJ/mol")
    print(f"GROMACS Potential Energy: {gromacs_pe:.6f} kJ/mol")
    diff = abs(lammps_pe_kj - gromacs_pe)
    print(f"Difference: {diff:.6e} kJ/mol")
    
    # 0.01 tolerance for energy
    status = "pass" if diff < 0.01 else "fail"
    print(f"Status: {status}\n")
    
    report = {
        "system_id": system_id,
        "lammps_pe_kcal": lammps_pe,
        "lammps_pe_kj": lammps_pe_kj,
        "gromacs_pe_kj": gromacs_pe,
        "difference_kj": diff,
        "status": status
    }
    
    with (work_dir / "mixing_parity_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    return report

def main():
    systems = ["mixing_toy"]
    corpus_root = REPO_ROOT / "testdata" / "lammps_golden"
    results_root = REPO_ROOT / "tests" / "reference_results" / "pt8_4_1_mixing_parity"
    
    summary = []
    for sid in systems:
        report = check_system(sid, corpus_root, results_root)
        summary.append(report)
        
    with (results_root / "mixing_parity_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
