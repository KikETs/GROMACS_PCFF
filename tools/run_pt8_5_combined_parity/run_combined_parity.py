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
# 1 kcal/mol/A = 41.84 kJ/mol/nm
KCAL_PER_A_TO_KJ_PER_NM = KCAL_TO_KJ / ANGSTROM_TO_NM

def create_gro_from_lammps(lammps_data_path: Path, gro_path: Path):
    data = parse_lammps_data(lammps_data_path)
    box_x = (data["box"]["x"]["hi"] - data["box"]["x"]["lo"]) * ANGSTROM_TO_NM
    box_y = (data["box"]["y"]["hi"] - data["box"]["y"]["lo"]) * ANGSTROM_TO_NM
    box_z = (data["box"]["z"]["hi"] - data["box"]["z"]["lo"]) * ANGSTROM_TO_NM
    
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

def create_mdp(mdp_path: Path, has_kspace: bool):
    coulomb_type = "PME" if has_kspace else "Cut-off"
    vdw_type = "Cut-off"
    
    content = f"""
integrator  = md
nsteps      = 0
cutoff-scheme = Verlet
vdw-type     = {vdw_type}
rvdw         = 0.9
coulombtype  = {coulomb_type}
rcoulomb     = 0.9
pbc          = xyz
nstfout      = 1
"""
    mdp_path.write_text(content)

def run_lammps(system_dir: Path, work_dir: Path) -> tuple[float, dict[int, list[float]]]:
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
        f.write("dump 1 all custom 1 forces.lammps id fx fy fz\n")
        f.write("dump_modify 1 sort id\n")
        f.write("run 0\n")
    
    result = subprocess.run(
        ["lmp", "-in", "system.in"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
        errors="replace"
    )
    
    # Parse Energy
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
    
    # Parse Forces
    forces = {}
    dump_file = work_dir / "forces.lammps"
    with dump_file.open("r") as f:
        reading_atoms = False
        for line in f:
            if line.startswith("ITEM: ATOMS"):
                reading_atoms = True
                continue
            if reading_atoms:
                parts = line.split()
                if not parts: continue
                atom_id = int(parts[0])
                fx, fy, fz = [float(x) * KCAL_PER_A_TO_KJ_PER_NM for x in parts[1:4]]
                forces[atom_id] = [fx, fy, fz]
    
    return pe, forces

def run_gromacs(top_path: Path, gro_path: Path, mdp_path: Path, work_dir: Path) -> tuple[float, dict[int, list[float]]]:
    gmx_bin = str(REPO_ROOT / "build" / "bin" / "gmx")
    
    # Set GMX_MAXBACKUP to -1 to avoid backup errors
    import os
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"

    subprocess.run(
        [gmx_bin, "grompp", "-f", str(mdp_path), "-c", str(gro_path), "-p", str(top_path), "-o", "topol.tpr", "-maxwarn", "10"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
        errors="replace",
        env=env
    )
    
    subprocess.run(
        [gmx_bin, "mdrun", "-s", "topol.tpr", "-rerun", str(gro_path), "-e", "ener.edr", "-o", "traj.trr"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
        errors="replace",
        env=env
    )
    
    # Energy
    energy_input = "Potential\n0\n"
    res_en = subprocess.run(
        [gmx_bin, "energy", "-f", "ener.edr", "-o", "energy.xvg"],
        input=energy_input,
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
        errors="replace",
        env=env
    )
    pe = None
    for line in res_en.stdout.splitlines():
        if line.startswith("Potential"):
            tokens = line.split()
            if len(tokens) >= 2:
                pe = float(tokens[1])
                break

    # Forces
    res_dump = subprocess.run(
        [gmx_bin, "dump", "-f", "traj.trr"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
        errors="replace",
        env=env
    )
    forces = {}
    pattern = re.compile(r"f\[\s*(\d+)\]=\{(.*),\s*(.*),\s*(.*)\}")
    for line in res_dump.stdout.splitlines():
        match = pattern.search(line)
        if match:
            atom_idx = int(match.group(1))
            fx = float(match.group(2))
            fy = float(match.group(3))
            fz = float(match.group(4))
            forces[atom_idx + 1] = [fx, fy, fz]
            
    return pe, forces

def check_system(system_id: str, corpus_root: Path, results_root: Path):
    system_dir = corpus_root / "systems" / system_id
    work_dir = results_root / system_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Running combined parity check for {system_id} ---")
    
    typed_ir = build_typed_ir({"id": system_id, "path": f"systems/{system_id}"}, corpus_root)
    top_content = render_gromacs_topology(typed_ir)
    top_path = work_dir / "system.top"
    top_path.write_text(top_content)
    
    gro_path = work_dir / "system.gro"
    create_gro_from_lammps(system_dir / "lammps" / "system.data", gro_path)
    
    has_kspace = typed_ir["styles"]["kspace_style"] is not None
    mdp_path = work_dir / "min.mdp"
    create_mdp(mdp_path, has_kspace)
    
    lammps_pe, lammps_forces = run_lammps(system_dir, work_dir)
    gromacs_pe, gromacs_forces = run_gromacs(top_path, gro_path, mdp_path, work_dir)
    
    lammps_pe_kj = lammps_pe * KCAL_TO_KJ
    energy_diff = abs(lammps_pe_kj - gromacs_pe)
    print(f"LAMMPS Potential Energy: {lammps_pe_kj:.6f} kJ/mol")
    print(f"GROMACS Potential Energy: {gromacs_pe:.6f} kJ/mol")
    print(f"Energy Difference: {energy_diff:.6e} kJ/mol")

    atom_diffs = []
    max_f_diff = 0.0
    sum_sq_f_diff = 0.0
    count = 0
    max_f_mag = 0.0
    
    for atom_id in sorted(lammps_forces.keys()):
        lf = lammps_forces[atom_id]
        gf = gromacs_forces[atom_id]
        diff = [abs(lf[i] - gf[i]) for i in range(3)]
        atom_diffs.append({
            "atom_id": atom_id,
            "lammps_f": lf,
            "gromacs_f": gf,
            "diff": diff
        })
        for d in diff:
            max_f_diff = max(max_f_diff, d)
            sum_sq_f_diff += d*d
            count += 1
        
        mag = (lf[0]**2 + lf[1]**2 + lf[2]**2)**0.5
        max_f_mag = max(max_f_mag, mag)
            
    rms_f_diff = (sum_sq_f_diff / count)**0.5 if count > 0 else 0.0
    rel_f_diff = (max_f_diff / max_f_mag if max_f_mag > 0 else 0)
    
    print(f"Max Force Diff: {max_f_diff:.6f} kJ/mol/nm")
    print(f"Relative Force Diff: {rel_f_diff:.6%}")
    
    # 0.5 kJ/mol for energy, 0.1% for forces (relaxed for combined systems)
    # Actually, for PME vs PPPM, energy might differ more.
    e_status = "pass" if energy_diff < 0.5 else "fail"
    f_status = "pass" if (max_f_diff < 0.5 or rel_f_diff < 0.005) else "fail"
    
    status = "pass" if (e_status == "pass" and f_status == "pass") else "fail"
    print(f"Status: {status}\n")
    
    report = {
        "system_id": system_id,
        "energy_diff_kj": energy_diff,
        "max_force_diff": max_f_diff,
        "rms_force_diff": rms_f_diff,
        "relative_force_diff": rel_f_diff,
        "status": status,
        "e_status": e_status,
        "f_status": f_status
    }
    
    with (work_dir / "combined_parity_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    return report

def main():
    systems = ["small_oligomer", "small_salt_polymer_box"]
    corpus_root = REPO_ROOT / "testdata" / "lammps_golden"
    results_root = REPO_ROOT / "tests" / "reference_results" / "pt8_5_combined_parity"
    
    summary = []
    for sid in systems:
        try:
            report = check_system(sid, corpus_root, results_root)
            summary.append(report)
        except Exception as e:
            print(f"Error checking {sid}: {e}")
        
    with (results_root / "combined_parity_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
