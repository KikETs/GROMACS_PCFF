
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GMX_BIN = os.environ.get("GMX_BIN", str(REPO_ROOT / "build" / "bin" / "gmx"))

def create_top(filename):
    with open(filename, 'w') as f:
        f.write("""
[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ rep-pow
1 2 yes 1.0 1.0 12.0

[ atomtypes ]
; name mass charge ptype sigma epsilon
C 12.011 0.0 A 0.4 0.1

[ moleculetype ]
; name nrexcl
MOL 1

[ atoms ]
; nr type resnr residue atom cgnr charge mass
1 C 1 MOL C1 1 0.0 12.011
2 C 1 MOL C2 1 0.0 12.011

[ system ]
Split Test System

[ molecules ]
MOL 1
""")

def create_gro(filename, dist):
    with open(filename, 'w') as f:
        f.write("Split Test System\n")
        f.write("2\n")
        f.write(f"    1MOL     C1    1   0.000   0.000   0.000\n")
        f.write(f"    1MOL     C2    2   {dist:.3f}   0.000   0.000\n")
        f.write("   5.000   5.000   5.000\n")

def create_mdp(filename, rcut):
    with open(filename, 'w') as f:
        f.write(f"""
integrator = md
nsteps = 0
cutoff-scheme = Verlet
nstlist = 1
nstfout = 1
rlist = {rcut}
rcoulomb = {rcut}
rvdw = {rcut}
coulombtype = PME
vdwtype = PME
fourierspacing = 0.1
pme-order = 4
ewald-rtol = 1e-5
ewald-rtol-lj = 1e-5
""")

def run_gmx(rcut, dist):
    gmx_bin = GMX_BIN
    create_top('system.top')
    create_gro('system.gro', dist)
    create_mdp('test.mdp', rcut)
    
    try:
        subprocess.run([gmx_bin, "grompp", "-f", "test.mdp", "-c", "system.gro", "-p", "system.top", "-o", "test.tpr", "-maxwarn", "10"], check=True, capture_output=True)
        subprocess.run([gmx_bin, "mdrun", "-s", "test.tpr", "-rerun", "system.gro", "-e", "energy.edr", "-deffnm", "test"], check=True, capture_output=True)
        
        # Extract energy
        input_str = "LJ-(SR)\nLJ-recip.\nPotential\n0\n"
        subprocess.run([gmx_bin, "energy", "-f", "energy.edr", "-o", "energy.xvg"], input=input_str.encode(), capture_output=True)
        
        # Extract forces
        result = subprocess.run([gmx_bin, "dump", "-f", "test.trr"], check=True, capture_output=True, text=True)
        force = 0
        for line in result.stdout.split('\n'):
            if "f[    1]" in line:
                force = float(line.split('=')[1].strip('{} \n').split(',')[0])
                break

        # Parse energy.xvg
        with open('energy.xvg', 'r') as f:
            for line in f:
                if line.startswith(('#', '@')): continue
                cols = line.split()
                if len(cols) >= 4:
                    return float(cols[1]), float(cols[2]), float(cols[3]), force
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None

if __name__ == "__main__":
    dist = 0.5
    print("rcut, sr, recip, potential, force")
    for rcut in [0.7, 0.8, 0.9, 1.0, 1.1]:
        sr, recip, pot, force = run_gmx(rcut, dist)
        if sr is not None:
            print(f"{rcut:.2f}, {sr:12.6f}, {recip:12.6f}, {pot:12.6f}, {force:12.6f}")
