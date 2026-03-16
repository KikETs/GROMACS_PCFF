import os
import subprocess
import json
import math

GMX_BIN = "/home/kiket/바탕화면/test/GROMACS_PCFF/build/bin/gmx"

def run_command(cmd, cwd=None, input_str=None):
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    result = subprocess.run(cmd, cwd=cwd, input=input_str, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        return False, result.stdout, result.stderr
    return True, result.stdout, result.stderr

def create_top(filename, ftype, params):
    reppow = 9.0 if ftype in ["nb", "ljc14", "nb_grid"] else 12.0
    sigma = 0.4
    eps = 0.1
    if ftype == "nb_grid":
        sigma = params[0]
        eps = params[1]
        
    content = f"""
[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ rep-pow
1 2 yes 1.0 1.0 {reppow}

[ atomtypes ]
; name mass charge ptype sigma epsilon
C {12.011} {0.0} A {sigma} {eps}

[ moleculetype ]
; name nrexcl
MOL 1

[ atoms ]
; nr type resnr residue atom cgnr charge mass
1 C 1 MOL C1 1 0.0 12.011
2 C 1 MOL C2 1 0.0 12.011
3 C 1 MOL C3 1 0.0 12.011
4 C 1 MOL C4 1 0.0 12.011

"""
    if ftype == "bond":
        content += "[ bonds ]\n; ai aj funct params\n1 2 11 " + " ".join(map(str, params)) + "\n"
    elif ftype == "angle":
        content += "[ angles ]\n; ai aj ak funct params\n1 2 3 11 " + " ".join(map(str, params)) + "\n"
    elif ftype == "dihedral":
        content += "[ dihedrals ]\n; ai aj ak al funct params\n1 2 3 4 13 " + " ".join(map(str, params)) + "\n"
    elif ftype == "improper":
        content += "[ dihedrals ]\n; ai aj ak al funct params\n1 2 3 4 12 " + " ".join(map(str, params)) + "\n"
    elif ftype == "nb":
        content += "[ pairs ]\n; ai aj funct params\n1 4 1 " + " ".join(map(str, params)) + "\n"
    elif ftype == "ljc14":
        content += "[ bonds ]\n1 2 1 0.15 1000\n2 3 1 0.15 1000\n3 4 1 0.15 1000\n"
        content += "[ pairs ]\n; ai aj funct params\n1 4 2 1.0 1.0 -1.0 " + " ".join(map(str, params)) + "\n"
    elif ftype == "nb_grid":
        pass

    content += """
[ system ]
Audit System

[ molecules ]
MOL 1
"""
    with open(filename, 'w') as f:
        f.write(content)

def create_gro(filename, coords):
    content = "Audit System\n4\n"
    for i, c in enumerate(coords):
        name = f"C{i+1}"
        content += f"{1:5d}MOL  {name:>5s}{i+1:5d}{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}\n"
    content += "   10.000   10.000   10.000\n"
    with open(filename, 'w') as f:
        f.write(content)

def create_mdp(filename):
    content = """
integrator = md
nsteps = 1
nstxout = 1
nstfout = 1
nstenergy = 1
cutoff-scheme = Verlet
rcoulomb = 1.0
rvdw = 1.0
vdw-type = Cut-off
"""
    with open(filename, 'w') as f:
        f.write(content)

def get_energy_forces_virial(work_dir):
    success, out, err = run_command([GMX_BIN, "grompp", "-f", "test.mdp", "-c", "test.gro", "-p", "test.top", "-o", "test.tpr", "-maxwarn", "5"], cwd=work_dir)
    if not success:
        print(f"grompp failed: {err}")
        return None, None, None
    
    success, out, err = run_command([GMX_BIN, "mdrun", "-s", "test.tpr", "-deffnm", "test", "-ntmpi", "1"], cwd=work_dir)
    if not success:
        print(f"mdrun failed: {err}")
        return None, None, None
    
    success, out, err = run_command([GMX_BIN, "energy", "-f", "test.edr", "-o", "energy.xvg"], cwd=work_dir, input_str="Potential\nVir-XX\nVir-YY\nVir-ZZ\n0\n")
    if not success:
        print(f"gmx energy failed: {err}")
        return None, None, None
    
    energy = None
    virial = [0.0, 0.0, 0.0]
    try:
        with open(os.path.join(work_dir, "energy.xvg"), 'r') as f:
            for line in f:
                if line.startswith(('#', '@')): continue
                cols = line.split()
                energy = float(cols[1])
                if len(cols) >= 5:
                    virial = [float(cols[2]), float(cols[3]), float(cols[4])]
                break
    except Exception as e:
        print(f"Error reading energy.xvg: {e}")
            
    success, out, err = run_command([GMX_BIN, "dump", "-f", "test.trr"], cwd=work_dir)
    forces = []
    lines = out.split('\n')
    for i in range(len(lines)):
        if "f[" in lines[i]:
            parts = lines[i].split('=')[1].strip('{} \n').split(',')
            forces.append([float(p) for p in parts])
            if len(forces) == 4: break
            
    return energy, forces, virial

def audit_kernel(name, ftype, params, coords, delta=1e-3):
    print(f"Auditing kernel: {name} ...")
    work_dir = f"tests/reference_results/k1_kernel_audit/{name}"
    os.makedirs(work_dir, exist_ok=True)
    
    create_mdp(os.path.join(work_dir, "test.mdp"))
    create_top(os.path.join(work_dir, "test.top"), ftype, params)
    create_gro(os.path.join(work_dir, "test.gro"), coords)
    
    e0, f0, v0 = get_energy_forces_virial(work_dir)
    if e0 is None:
        print(f"Failed to get baseline for {name}")
        return None
    
    # Calculate expected virial from forces and coordinates
    # GROMACS Virial: Xi_ab = -0.5 * sum r_ij,a * F_ij,b
    # Here we use -0.5 * sum x_i,a * F_i,b which is equivalent for non-PBC
    exp_virial = [0.0, 0.0, 0.0]
    for a in range(4):
        exp_virial[0] += -0.5 * coords[a][0] * f0[a][0]
        exp_virial[1] += -0.5 * coords[a][1] * f0[a][1]
        exp_virial[2] += -0.5 * coords[a][2] * f0[a][2]
            
    vir_diff = [v0[i] - exp_virial[i] for i in range(3)]
    max_vir_diff = max(abs(d) for d in vir_diff)
    
    num_forces = [[0.0, 0.0, 0.0] for _ in range(4)]
    for atom in range(4):
        for dim in range(3):
            c_plus = [c[:] for c in coords]
            c_plus[atom][dim] += delta
            create_gro(os.path.join(work_dir, "test.gro"), c_plus)
            e_plus, _, _ = get_energy_forces_virial(work_dir)
            
            c_minus = [c[:] for c in coords]
            c_minus[atom][dim] -= delta
            create_gro(os.path.join(work_dir, "test.gro"), c_minus)
            e_minus, _, _ = get_energy_forces_virial(work_dir)
            
            num_forces[atom][dim] = -(e_plus - e_minus) / (2 * delta)
            
    max_diff = 0.0
    max_f = 0.0
    for a in range(4):
        for d in range(3):
            diff = abs(f0[a][d] - num_forces[a][d])
            if diff > max_diff: max_diff = diff
            if abs(f0[a][d]) > max_f: max_f = abs(f0[a][d])
            
    rel_diff = max_diff / (max_f + 1e-9)
    
    result = {
        "name": name,
        "energy": e0,
        "analytic_forces": f0,
        "numerical_forces": num_forces,
        "virial": v0,
        "expected_virial": exp_virial,
        "vir_diff": vir_diff,
        "max_diff": max_diff,
        "rel_diff": rel_diff,
        "status": "PASS" if rel_diff < 1e-2 and max_vir_diff < 1.0 else "FAIL"
    }
    return result

def main():
    coords = [
        [0.0, 0.0, 0.0],
        [0.15, 0.0, 0.0],
        [0.285, 0.1, 0.0],
        [0.41, 0.12, 0.11]
    ]
    
    results = []
    
    # Nonbonded 9-6 Grid (should use kernel_ref_inner.h)
    results.append(audit_kernel("nb_grid_9_6", "nb_grid", [0.4, 0.1], coords))
    
    # Nonbonded 9-6 Pairs (should use pairs.cpp)
    results.append(audit_kernel("nb_pairs_9_6", "nb", [0.4, 0.1], coords))

    print(json.dumps(results, indent=2))
    os.makedirs("tests/reference_results/k1_kernel_audit", exist_ok=True)
    with open("tests/reference_results/k1_kernel_audit/consistency_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
