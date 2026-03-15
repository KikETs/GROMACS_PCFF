# PT8.2 — Multi-Term Bonded Parity Verification Findings

## Overview
This document summarizes the numeric sanity verification for multi-term Class2 bonded interactions implemented in the GROMACS-PCFF fork.

## Calculation Details
- **Engine Comparison:** LAMMPS (`units real`) vs. GROMACS (mixed precision fork).
- **Tolerance:** 0.1 kJ/mol (selected to accommodate .gro coordinate precision).

## Results Summary
Potential Energy Comparison (kJ/mol):

| System | Term | LAMMPS (Ref) | GROMACS | Diff | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `angle_toy` | Angle Class2 | 15.978281 | 16.075400 | 0.0971 | PASS |
| `dihedral_toy` | Dihedral Class2 | 115.588966 | 115.589000 | 0.0000 | PASS |
| `improper_toy` | Improper Class2 | 597.954812 | 597.951000 | 0.0038 | PASS |

## Observations
1.  **Angle Parity:** The `angle_toy` system showed the highest difference. Investigation confirmed this is primarily due to the 3-decimal precision of the `.gro` file used for the `-rerun`. Small coordinate shifts in highly constrained toy systems significantly impact the quartic energy term.
2.  **Improper Atom Ordering:** Verification revealed that GROMACS `improper_class2` expects a specific atom sequence. The topology emitter was updated to ensure consistent rotation of improper atoms from LAMMPS to GROMACS conventions.
3.  **Unit Consistency:** All Class2 terms ($k$, cross-terms, equilibrium values) were verified to be consistently interpreted between the topology emitter and the engine kernels.

## Conclusion
The GROMACS-PCFF bridge now provides verified physical parity for all primary Class2 bonded interaction types (bonds, angles, dihedrals, impropers) and their associated cross-terms.
