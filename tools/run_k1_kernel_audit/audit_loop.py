import os
import subprocess
import json
import math

GMX_BIN = "/home/kiket/바탕화면/test/GROMACS_PCFF/build/bin/gmx"

def run_command(cmd, cwd=None, input_str=None):
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    result = subprocess.run(cmd, cwd=cwd, input=input_str, capture_output=True, text=True, env=env)
    return result.returncode == 0, result.stdout, result.stderr

def create_top(filename, ftype, params):
    reppow = 9.0
    content = f"""
[ defaults ]
1 2 yes 1.0 1.0 {reppow}
[ atomtypes ]
C 12.011 0.0 A 0.0 0.0
[ moleculetype ]
MOL 3
[ atoms ]
1 C 1 MOL C1 1 1.0 12.011
2 C 1 MOL C2 1 -1.0 12.011
3 C 1 MOL C3 1 1.0 12.011
4 C 1 MOL C4 1 -1.0 12.011
"""
    if ftype == "dihedral":
        content += "[ dihedrals ]\n1 2 3 4 13 " + " ".join(map(str, params)) + "\n"
    elif ftype == "improper":
        content += "[ dihedrals ]\n1 2 3 4 12 " + " ".join(map(str, params)) + "\n"
    content += "[ system ]\nAudit\n[ molecules ]\nMOL 1\n"
    with open(filename, 'w') as f:
        f.write(content)

def create_gro(filename, coords):
    content = "Audit\n4\n"
    for i, c in enumerate(coords):
        name = f"C{i+1}"
        content += f"{1:5d}MOL  {name:>5s}{i+1:5d}{c[0]:8.5f}{c[1]:8.5f}{c[2]:8.5f}\n"
    content += "   10.000   10.000   10.000\n"
    with open(filename, 'w') as f:
        f.write(content)

def create_mdp(filename):
    content = "integrator=md\nnsteps=1\nnstxout=1\nnstfout=1\ncutoff-scheme=Verlet\nrcoulomb=1.0\nrvdw=1.0\n"
    with open(filename, 'w') as f:
        f.write(content)

def get_forces(work_dir):
    success, out, err = run_command([GMX_BIN, "grompp", "-f", "test.mdp", "-c", "test.gro", "-p", "test.top", "-o", "test.tpr", "-maxwarn", "5"], cwd=work_dir)
    if not success:
        print(f"grompp failed: {err}")
        return []
    success, out, err = run_command([GMX_BIN, "mdrun", "-s", "test.tpr", "-deffnm", "test", "-ntmpi", "1"], cwd=work_dir)
    if not success:
        print(f"mdrun failed: {err}")
        return []
    success, out, err = run_command([GMX_BIN, "dump", "-f", "test.trr"], cwd=work_dir)
    if not success:
        print(f"gmx dump failed: {err}")
        return []
    forces = []
    lines = out.split('\n')
    for i in range(len(lines)):
        if "f[" in lines[i]:
            parts = lines[i].split('=')[1].strip('{} \n').split(',')
            forces.append([float(p) for p in parts])
            if len(forces) == 4: break
    return forces

def audit_loop(name, ftype, params):
    print(f"Loop audit for {name}...")
    work_dir = f"tests/reference_results/k1_kernel_audit/loop_{name}"
    os.makedirs(work_dir, exist_ok=True)
    create_mdp(os.path.join(work_dir, "test.mdp"))
    create_top(os.path.join(work_dir, "test.top"), ftype, params)
    
    coords = [[0,0,0], [0.15,0,0], [0.28,0.1,0], [0.4,0.1,0.1]]
    
    n_steps = 100
    radius = 0.01
    work = 0.0
    
    for i in range(n_steps):
        angle = 2 * math.pi * i / n_steps
        angle_next = 2 * math.pi * (i+1) / n_steps
        
        # Move atom 3 in a circle in XY plane
        c_i = [c[:] for c in coords]
        c_i[3][0] += radius * math.cos(angle)
        c_i[3][1] += radius * math.sin(angle)
        
        create_gro(os.path.join(work_dir, "test.gro"), c_i)
        f_i = get_forces(work_dir)
        
        dx = radius * (math.cos(angle_next) - math.cos(angle))
        dy = radius * (math.sin(angle_next) - math.sin(angle))
        
        # dW = F . dr
        work += f_i[3][0] * dx + f_i[3][1] * dy
        
    print(f"  Total Work over loop: {work:.6f} kJ/mol")
    return work

def main():
    dih_params = [3.3, 0.0, 2.5, 3.14, 1.6, 0.0, 5.0, -3.3, 1.6, 0.15, 4.1, -2.0, 0.8, 4.6, -1.2, 0.4, 0.15, 0.15, 0.25, -0.12, 0.06, 0.2, -0.1, 0.04, 1.95, 1.95, 0.8, 1.95, 1.95, 0.75, 0.15, 0.15]
    w_dih = audit_loop("dihedral", "dihedral", dih_params)
    
    imp_params = [100.0, 0.0, 5.0, 4.1, 3.7, 1.92, 1.90, 1.88]
    w_imp = audit_loop("improper", "improper", imp_params)
    
    results = {"dihedral_work": w_dih, "improper_work": w_imp}
    with open("tests/reference_results/k1_kernel_audit/loop_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
