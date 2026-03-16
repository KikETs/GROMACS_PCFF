
import os
import subprocess
import json

def create_top(filename, comb_rule, reppow):
    with open(filename, 'w') as f:
        f.write(f"""
[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ rep-pow
1 {comb_rule} yes 1.0 1.0 {reppow}

[ atomtypes ]
; name mass charge ptype sigma epsilon
C1 12.011 0.0 A 0.4 0.1
C2 12.011 0.0 A 0.5 0.2

[ moleculetype ]
; name nrexcl
MOL 1

[ atoms ]
; nr type resnr residue atom cgnr charge mass
1 C1 1 MOL C1 1 0.0 12.011
2 C2 1 MOL C2 1 0.0 12.011

[ system ]
PME Proof System

[ molecules ]
MOL 1
""")

def create_gro(filename):
    with open(filename, 'w') as f:
        f.write("PME Proof System\n")
        f.write("2\n")
        f.write(f"    1MOL     C1    1   0.000   0.000   0.000\n")
        f.write(f"    1MOL     C2    2   0.500   0.000   0.000\n")
        f.write("   5.000   5.000   5.000\n")

def create_mdp(filename, vdwtype):
    with open(filename, 'w') as f:
        f.write(f"""
integrator = md
nsteps = 0
cutoff-scheme = Verlet
ns-type = grid
nstlist = 1
rlist = 1.0
rcoulomb = 1.0
rvdw = 1.0
coulombtype = PME
vdwtype = {vdwtype}
fourierspacing = 0.1
pme-order = 4
ewald-rtol = 1e-5
ewald-rtol-lj = 1e-5
""")

def inspect_tpr(comb_rule, reppow, vdwtype):
    gmx_bin = "/home/kiket/바탕화면/test/GROMACS_PCFF/build/bin/gmx"
    
    create_top('system.top', comb_rule, reppow)
    create_gro('system.gro')
    create_mdp('test.mdp', vdwtype)
    
    try:
        subprocess.run([gmx_bin, "grompp", "-f", "test.mdp", "-c", "system.gro", "-p", "system.top", "-o", "test.tpr", "-maxwarn", "10"], check=True, capture_output=True)
        result = subprocess.run([gmx_bin, "dump", "-s", "test.tpr"], check=True, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    # Case 1: SixthPower mixing, repulsion 9.0, Cut-off
    print("=== Case 1: SixthPower, 9.0, Cut-off ===")
    dump1 = inspect_tpr(4, 9.0, "Cut-off")
    # Search for nbfp matrix
    for line in dump1.split('\n'):
        if "nbfp[" in line and "0, 1" in line:
            print(line)
            
    # Case 2: SixthPower mixing, repulsion 12.0 (if possible), PME
    # Wait, SixthPower requires 9.0. Let's use comb-rule 2 (Geometric) with 9.0
    print("=== Case 2: Geometric, 9.0, PME ===")
    # This will fail in mdrun, but grompp might pass.
    dump2 = inspect_tpr(2, 9.0, "PME")
    for line in dump2.split('\n'):
        if "nbfp[" in line and "0, 1" in line:
            print(line)
        if "lj_comb" in line:
            print(line)
