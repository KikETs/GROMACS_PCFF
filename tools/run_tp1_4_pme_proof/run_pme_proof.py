
import os
import subprocess
import json

GMX_BIN = "/home/kiket/바탕화면/test/GROMACS_PCFF/build/bin/gmx"

def write_top(filename):
    with open(filename, 'w') as f:
        f.write("[ defaults ]\n")
        f.write("1 4 yes 1.0 1.0 9.0\n\n")
        f.write("[ atomtypes ]\n")
        f.write("T1 12.011 0.0 A 0.35 0.3\n\n")
        f.write("[ moleculetype ]\n")
        f.write("MOL 1\n")
        f.write("[ atoms ]\n")
        f.write("1 T1 1 MOL A1 1 0.0 12.011\n\n")
        f.write("[ system ]\n")
        f.write("TP1.4\n")
        f.write("[ molecules ]\n")
        f.write("MOL 2\n")

def write_mdp(filename):
    with open(filename, 'w') as f:
        f.write("vdwtype = PME\n")
        f.write("ljpme-combination-rule = Geometric\n")
        f.write("rvdw = 1.0\n")
        f.write("rcoulomb = 1.0\n")
        f.write("ewald-rtol-vdw = 1e-5\n")
        f.write("ewald-rtol = 1e-5\n")
        f.write("cutoff-scheme = Verlet\n")
        f.write("pbc = xyz\n")
        f.write("nsteps = 0\n")

def write_gro(filename, d):
    with open(filename, 'w') as f:
        f.write("TP1.4\n")
        f.write("2\n")
        f.write(f"{1:5d}MOL     A1    1{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}\n")
        f.write(f"{1:5d}MOL     A1    2{d:8.3f}{0.0:8.3f}{0.0:8.3f}\n")
        f.write("   3.000   3.000   3.000\n")

def run_gmx(cmd, input_str=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, input=input_str)

def main():
    work_dir = "/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_4_pme_proof/work"
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    write_top("topol.top")
    write_mdp("test.mdp")
    
    # 0.8 to 1.2 in steps of 0.01
    distances = [round(0.8 + 0.01 * i, 4) for i in range(41)]
    results = []
    
    for d in distances:
        write_gro("test.gro", d)
        run_gmx(f"{GMX_BIN} grompp -f test.mdp -c test.gro -p topol.top -o test.tpr -maxwarn 2")
        # Run energy to get potential
        run_gmx(f"{GMX_BIN} mdrun -s test.tpr -rerun test.gro -e test.edr -o test.trr -g test.log -nt 1")
        
        # Extract energy
        res = run_gmx(f"{GMX_BIN} energy -f test.edr", input_str="Potential\n0\n")
        potential = None
        lines = res.stdout.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("Potential"):
                try:
                    # Look for the line after "Potential" headers
                    for j in range(i+1, len(lines)):
                        parts = lines[j].split()
                        if len(parts) >= 2 and parts[0] == "Potential":
                            potential = float(parts[1])
                            break
                        if len(parts) >= 2 and j > i+1: # Fallback to standard table
                            potential = float(parts[1])
                            break
                except:
                    continue
        
        # Extract forces
        run_gmx(f"{GMX_BIN} traj -f test.trr -s test.tpr -of force.xvg", input_str="System\n")
        force_x = 0
        if os.path.exists("force.xvg"):
            with open("force.xvg") as f:
                for line in f:
                    if not line.startswith(('#', '@')):
                        tokens = line.split()
                        if len(tokens) >= 5:
                            force_x = float(tokens[4]) # force on atom 2 in X
        
        results.append({"distance": d, "potential": potential, "force_x": force_x})
        print(f"d={d:.4f}, potential={potential}, force_x={force_x}")

    # Write CSV manually
    os.chdir("..")
    with open("pme_energy_force_scan.csv", 'w') as f:
        f.write("distance,potential,force_x\n")
        for res in results:
            f.write(f"{res['distance']},{res['potential']},{res['force_x']}\n")
    
    # Analysis
    # Compare potential at 0.99, 1.00, 1.01
    p099 = next(r for r in results if r['distance'] == 0.99)['potential']
    p100 = next(r for r in results if r['distance'] == 1.00)['potential']
    p101 = next(r for r in results if r['distance'] == 1.01)['potential']
    
    potential_jump = p101 - p099 # simplified
    
    summary = {
        "potential_at_0.99": p099,
        "potential_at_1.00": p100,
        "potential_at_1.01": p101,
        "potential_jump_near_cutoff": potential_jump
    }
    
    with open("pme_continuity_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
