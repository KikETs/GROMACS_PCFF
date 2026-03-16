from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys
import re
import itertools
import os
import shutil
import math

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
DEG_TO_RAD = math.pi / 180.0

def kcal_to_kj(val):
    return val * 4.184

def dump_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def create_gro_from_lammps(lammps_data_path: Path, gro_path: Path):
    data = parse_lammps_data(lammps_data_path)
    box_x = 4.0
    box_y = 4.0
    box_z = 4.0
    
    with gro_path.open("w") as f:
        f.write("High precision search gro\n")
        f.write(f"{len(data['atoms']):>5d}\n")
        for i, atom in enumerate(data["atoms"], start=1):
            x = atom["x_angstrom"] * ANGSTROM_TO_NM
            y = atom["y_angstrom"] * ANGSTROM_TO_NM
            z = atom["z_angstrom"] * ANGSTROM_TO_NM
            f.write(f"{1:>5d}{'MOL':<5s}{f'A{i}':>5s}{atom['id']:>5d}{x:15.7f}{y:15.7f}{z:15.7f}\n")
        f.write(f"{box_x:15.7f}{box_y:15.7f}{box_z:15.7f}\n")

def run_lammps(system_dir: Path, work_dir: Path) -> dict[int, list[float]]:
    orig_in = system_dir / "lammps" / "system.in"
    data_file = system_dir / "lammps" / "system.data"
    
    local_in = work_dir / "system.in"
    local_data = work_dir / "system.data"
    local_data.write_bytes(data_file.read_bytes())
    
    with orig_in.open("r") as f:
        lines = f.readlines()
    
    with local_in.open("w") as f:
        for line in lines:
            if line.startswith("read_data"):
                f.write("read_data system.data\n")
            else:
                f.write(line)
        f.write("\ndump 1 all custom 1 forces.lammps id fx fy fz\n")
        f.write("dump_modify 1 sort id\n")
        f.write("run 0\n")
    
    subprocess.run(["lmp", "-in", "system.in"], cwd=work_dir, capture_output=True, text=True, check=True)
    
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
    return forces

def run_variant(vid: str, top_content: str, gro_path: Path, results_root: Path) -> dict[int, list[float]]:
    gmx_bin = str(REPO_ROOT / "build" / "bin" / "gmx")
    variant_dir = results_root / vid
    variant_dir.mkdir(parents=True, exist_ok=True)
    
    local_gro = variant_dir / "search.gro"
    shutil.copy(gro_path, local_gro)
    
    top_path = variant_dir / "system.top"
    top_path.write_text(top_content)
    
    mdp_path = variant_dir / "min.mdp"
    mdp_path.write_text("integrator=md\nnsteps=0\ncutoff-scheme=Verlet\nnstfout=1\n")
    
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    
    subprocess.run([gmx_bin, "grompp", "-f", "min.mdp", "-c", "search.gro", "-p", "system.top", "-o", "topol.tpr", "-maxwarn", "10"], cwd=variant_dir, capture_output=True, text=True, check=True, env=env)
    subprocess.run([gmx_bin, "mdrun", "-s", "topol.tpr", "-rerun", "search.gro", "-o", "traj.trr"], cwd=variant_dir, capture_output=True, text=True, check=True, env=env)
    res = subprocess.run([gmx_bin, "dump", "-f", "traj.trr"], cwd=variant_dir, capture_output=True, text=True, check=True, env=env)
    
    forces = {}
    pattern = re.compile(r"f\[\s*(\d+)\]=\{(.*),\s*(.*),\s*(.*)\}")
    for line in res.stdout.splitlines():
        match = pattern.search(line)
        if match:
            forces[int(match.group(1)) + 1] = [float(match.group(2)), float(match.group(3)), float(match.group(4))]
    return forces

def evaluate_variant(lammps_f, gmx_f):
    max_diff = 0.0
    max_mag = 0.0
    for aid in lammps_f:
        lf, gf = lammps_f[aid], gmx_f[aid]
        diffs = [abs(lf[i] - gf[i]) for i in range(3)]
        max_diff = max(max_diff, *diffs)
        max_mag = max(max_mag, (lf[0]**2 + lf[1]**2 + lf[2]**2)**0.5)
    return max_diff, (max_diff / max_mag if max_mag > 0 else 0)

def main():
    sid = "improper_toy"
    corpus_root = REPO_ROOT / "testdata" / "lammps_golden"
    results_root = REPO_ROOT / "tests" / "reference_results" / "pt8_3_2_improper_search"
    if results_root.exists():
        shutil.rmtree(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    
    base_work = results_root / "baseline"
    base_work.mkdir()
    lammps_forces = run_lammps(corpus_root / "systems" / sid, base_work)
    gro_path = results_root / "search.gro"
    create_gro_from_lammps(corpus_root / "systems" / sid / "lammps" / "system.data", gro_path)
    
    typed_ir = build_typed_ir({"id": sid, "path": f"systems/{sid}"}, corpus_root)
    coeffs = typed_ir["improper_types"][0]
    
    k_vals = [kcal_to_kj(coeffs["aa"][k]) for k in ["k1_kcal_mol", "k2_kcal_mol", "k3_kcal_mol"]]
    t_vals = [coeffs["aa"][t] for t in ["theta0_1_deg", "theta0_2_deg", "theta0_3_deg"]]
    
    k_perms = list(itertools.permutations(k_vals))
    t_perms = list(itertools.permutations(t_vals))
    # Representative permutations
    atom_perms = [[1, 2, 3, 4], [2, 1, 3, 4]]
    
    # Baseline variant
    from tools.pcff_fixture_bridge.common import render_gromacs_topology
    typed_ir = build_typed_ir({"id": sid, "path": f"systems/{sid}"}, corpus_root)
    base_top = render_gromacs_topology(typed_ir)
    base_gmx_f = run_variant("baseline_check", base_top, gro_path, results_root)
    base_max_d, base_rel_d = evaluate_variant(lammps_forces, base_gmx_f)
    print(f"Verified Baseline Check RelErr: {base_rel_d:.6%}")

    # Kernal swap check (Swapping k1/k2 in baseline top)
    # Baseline was: 1 2 3 4 12 104.6 0.0 5.0208 4.184 3.7656 110.0 109.0 108.0
    # Swapped: 1 2 3 4 12 104.6 0.0 4.184 5.0208 3.7656 110.0 109.0 108.0
    swapped_top = base_top.replace("1   2   3   4 12 104.60000000 0.00000000 5.02080000 4.18400000 3.76560000",
                                   "1   2   3   4 12 104.60000000 0.00000000 4.18400000 5.02080000 3.76560000")
    swap_gmx_f = run_variant("kernel_swap_check", swapped_top, gro_path, results_root)
    swap_max_d, swap_rel_d = evaluate_variant(lammps_forces, swap_gmx_f)
    print(f"Verified Swapped Mapping Check RelErr: {swap_rel_d:.6%}")

    # Extract [ bonds ] and [ angles ] from base_top to inject into variants
    bond_angle_part = []
    lines = base_top.splitlines()
    in_ba = False
    for line in lines:
        if line.startswith("[ bonds ]") or line.startswith("[ angles ]"):
            in_ba = True
        elif line.startswith("[ dihedrals ]"):
            in_ba = False
        if in_ba:
            bond_angle_part.append(line)
    
    ba_str = "\n".join(bond_angle_part)

    variants = []
    
    for a_perm in atom_perms:
        for kp in k_perms:
            for tp in t_perms:
                for k_unit in ["raw", "rad"]:
                    for t_unit in ["raw", "rad"]:
                        vid = f"a{''.join(map(str, a_perm))}_k{k_unit}_t{t_unit}_kp{k_perms.index(kp)}_tp{t_perms.index(tp)}"
                        
                        kp_adj = [k * (1.0 if k_unit == "raw" else (180.0/math.pi)**2) for k in kp]
                        tp_adj = [t * (1.0 if t_unit == "raw" else DEG_TO_RAD) for t in tp]
                        
                        top_lines = [
                            "[ defaults ]", "1 4 yes 1.0 1.0 9.0", "",
                            "[ atomtypes ]", "T1 12.011 0.0 A 0.35 0.0", "",
                            "[ moleculetype ]", "IMPR 3", "",
                            "[ atoms ]", "1 T1 1 IMPR A1 1 0.0 12.011", "2 T1 1 IMPR A2 2 0.0 12.011", "3 T1 1 IMPR A3 3 0.0 12.011", "4 T1 1 IMPR A4 4 0.0 12.011", "",
                            ba_str,
                            "[ dihedrals ]",
                            f"{a_perm[0]} {a_perm[1]} {a_perm[2]} {a_perm[3]} 12 {kcal_to_kj(coeffs['main']['k0_kcal_mol'])} {coeffs['main']['chi0_deg']} " + " ".join(map(str, kp_adj)) + " " + " ".join(map(str, tp_adj)),
                            "",
                            "[ system ]", "Search", "",
                            "[ molecules ]", "IMPR 1"
                        ]
                        
                        try:
                            gmx_forces = run_variant(vid, "\n".join(top_lines), gro_path, results_root)
                            max_d, rel_d = evaluate_variant(lammps_forces, gmx_forces)
                            variants.append({
                                "vid": vid, "atoms": a_perm, "rel_diff": rel_d
                            })
                            if rel_d < 0.001:
                                print(f"Found match! {vid} {rel_d:.6%}")
                        except Exception as e:
                            continue

    variants.sort(key=lambda x: x["rel_diff"])
    dump_json(results_root / "search_results.json", variants)
    
    print(f"Tested {len(variants)} variants.")
    if variants:
        best = variants[0]
        print(f"Best: {best['vid']} RelErr: {best['rel_diff']:.6%}")

if __name__ == "__main__":
    main()
