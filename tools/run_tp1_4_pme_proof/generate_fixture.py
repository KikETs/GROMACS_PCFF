
import os
import subprocess

def create_top(filename):
    with open(filename, 'w') as f:
        f.write("""
[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ rep-pow
1 4 yes 1.0 1.0 9.0

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
PME Proof System

[ molecules ]
MOL 1
""")

def create_gro(filename, dist):
    with open(filename, 'w') as f:
        f.write("PME Proof System\n")
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
ns-type = grid
nstlist = 1
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
    create_top('system.top')
    create_gro('system.gro', dist)
    create_mdp('test.mdp', rcut)
    
    # We need to find the gmx binary. Assuming 'gmx' is in the path.
    # The prompt says we are inside a local repository with a custom bridge.
    # I should check if there's a build directory.
    gmx_bin = "/home/kiket/바탕화면/test/GROMACS_PCFF/build/bin/gmx" 
    
    try:
        subprocess.run([gmx_bin, "grompp", "-f", "test.mdp", "-c", "system.gro", "-p", "system.top", "-o", "test.tpr", "-maxwarn", "10"], check=True, capture_output=True)
        result = subprocess.run([gmx_bin, "mdrun", "-s", "test.tpr", "-rerun", "system.gro", "-e", "energy.edr"], check=True, capture_output=True)
        
        # Extract energy using gmx energy
        # For LJ-PME, we want "Lennard-Jones (SR)" and "LJ [mesh]"
        input_str = "Lennard-Jones-(SR)\nLJ-Recip\n0\n"
        result = subprocess.run([gmx_bin, "energy", "-f", "energy.edr", "-o", "energy.xvg"], input=input_str.encode(), capture_output=True)
        
        # Parse energy.xvg
        sr = 0
        recip = 0
        with open('energy.xvg', 'r') as f:
            for line in f:
                if line.startswith('#') or line.startswith('@'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    sr = float(parts[1])
                    recip = float(parts[2])
        return sr, recip
    except Exception as e:
        print(f"Error running GROMACS: {e}")
        return None, None

if __name__ == "__main__":
    # Scan distance near 0.8 nm (rcut is usually around 1.0)
    # We vary rcut to move the split point.
    dist = 0.5
    print("rcut, sr, recip, total")
    for rcut in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        sr, recip = run_gmx(rcut, dist)
        if sr is not None:
            print(f"{rcut:.2f}, {sr:.6f}, {recip:.6f}, {sr+recip:.6f}")
