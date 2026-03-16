from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys

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
    # Increase box size to avoid cut-off errors in GROMACS
    box_x = 2.0
    box_y = 2.0
    box_z = 2.0
    
    with gro_path.open("w") as f:
        f.write("Generated from LAMMPS data\n")
        f.write(f"{len(data['atoms']):>5d}\n")
        for i, atom in enumerate(data["atoms"], start=1):
            # Shift coordinates to be positive if needed, but here just convert A to nm
            # and use residue name 'MOL' and atom name 'A'
            x = atom["x_angstrom"] * ANGSTROM_TO_NM
            y = atom["y_angstrom"] * ANGSTROM_TO_NM
            z = atom["z_angstrom"] * ANGSTROM_TO_NM
            # GROMACS .gro format:
            # 5 positions for residue number, 5 for residue name, 5 for atom name, 5 for atom number, 
            # 8.3f for x, y, z
            f.write(f"{1:>5d}{'MOL':<5s}{f'A{i}':>5s}{atom['id']:>5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write(f"{box_x:10.5f}{box_y:10.5f}{box_z:10.5f}\n")

def create_mdp(mdp_path: Path):
    content = """
integrator  = md
nsteps      = 0
cutoff-scheme = Verlet
vdw-type     = Cut-off
rvdw         = 0.9
coulombtype  = Cut-off
rcoulomb     = 0.9
pbc          = xyz
"""
    mdp_path.write_text(content)

def run_lammps(system_dir: Path, work_dir: Path) -> float:
    # We need to run LAMMPS and get potential energy.
    # system.in already exists. We'll modify it slightly to output energy.
    orig_in = system_dir / "lammps" / "system.in"
    new_in = work_dir / "system.in"
    data_file = system_dir / "lammps" / "system.data"
    
    # Copy data file to work_dir
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
    
    # Parse potential energy from output
    # Typical line: Step PotEng
    #                0   -12.345
    pe = None
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        if "Step" in line and "PotEng" in line:
            # Look at subsequent lines until we find the data
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
    # 1. grompp
    subprocess.run(
        [gmx_bin, "grompp", "-f", str(mdp_path), "-c", str(gro_path), "-p", str(top_path), "-o", "topol.tpr"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
        errors="replace"
    )
    
    # 2. mdrun -rerun
    subprocess.run(
        [gmx_bin, "mdrun", "-s", "topol.tpr", "-rerun", str(gro_path), "-e", "ener.edr"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
        errors="replace"
    )
    
    # 3. energy to get Potential
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
    
    # Parse potential energy from output
    # Typical line: Potential                    5.0761         --          0          0  (kJ/mol)
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

def main():
    system_id = "bond_toy"
    corpus_root = REPO_ROOT / "testdata" / "lammps_golden"
    system_dir = corpus_root / "systems" / system_id
    
    work_dir = REPO_ROOT / "tests" / "reference_results" / "pt8_1_sanity" / system_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running sanity check for {system_id}...")
    
    # GROMACS topology
    typed_ir = build_typed_ir({"id": system_id, "path": f"systems/{system_id}"}, corpus_root)
    top_content = render_gromacs_topology(typed_ir)
    top_path = work_dir / "system.top"
    top_path.write_text(top_content)
    
    # GROMACS gro
    gro_path = work_dir / "system.gro"
    create_gro_from_lammps(system_dir / "lammps" / "system.data", gro_path)
    
    # GROMACS mdp
    mdp_path = work_dir / "min.mdp"
    create_mdp(mdp_path)
    
    # Run LAMMPS
    lammps_pe = run_lammps(system_dir, work_dir)
    # LAMMPS energy is in kcal/mol. Convert to kJ/mol for comparison.
    lammps_pe_kj = lammps_pe * 4.184
    
    # Run GROMACS
    gromacs_pe = run_gromacs(top_path, gro_path, mdp_path, work_dir)
    
    print(f"LAMMPS Potential Energy: {lammps_pe_kj:.6f} kJ/mol")
    print(f"GROMACS Potential Energy: {gromacs_pe:.6f} kJ/mol")
    diff = abs(lammps_pe_kj - gromacs_pe)
    print(f"Difference: {diff:.6e} kJ/mol")
    
    status = "pass" if diff < 1e-3 else "fail"
    print(f"Status: {status}")
    
    report = {
        "system_id": system_id,
        "lammps_pe_kcal": lammps_pe,
        "lammps_pe_kj": lammps_pe_kj,
        "gromacs_pe_kj": gromacs_pe,
        "difference_kj": diff,
        "status": status
    }
    
    with (work_dir / "sanity_report.json").open("w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
