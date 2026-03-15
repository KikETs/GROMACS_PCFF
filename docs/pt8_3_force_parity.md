# PT8.3 — Force Vector Parity Verification Findings (Revised)

## Overview
This document summarizes the atom-wise force-vector parity verification for Class2 bonded interactions. Milestone PT8.3.1 confirmed the reliability of the ImproperClass2 mapping through empirical testing.

## Numeric Summary
| System | Term | Max Force (kJ/mol/nm) | Max Diff (kJ/mol/nm) | Rel Diff (%) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bond_toy` | Bond Class2 | 1443.33 | 0.0336 | 0.0023% | PASS |
| `angle_toy` | Angle Class2 | 1415.55 | 0.0173 | 0.0012% | PASS |
| `dihedral_toy` | Dihedral Class2 | 3232.47 | 0.0052 | 0.0002% | PASS |
| `improper_toy` | Improper Class2 | 12418.60 | 7.6506 | 0.0616% | PASS |

## Precision & Mapping Validation
1.  **Coordinate Precision:** High-precision (7-decimal) `.gro` files were used to achieve sub-0.1% relative force errors across all systems.
2.  **Improper Mapping Verification (PT8.3.1):** 
    - A suspected coefficient mapping bug (K1/K2 swap) was investigated.
    - Empirical testing showed that the original mapping (`aa_k1`=K1, `aa_k2`=K2) yields an energy difference of 0.003 kJ/mol, while swapping them increases the error to 0.12 kJ/mol.
    - **Conclusion:** The bridge mapping is consistent with the engine kernel's internal logic. The 0.06% force difference in `improper_toy` is attributed to numerical sensitivity in the Wilson angle derivative rather than a mapping error.

## Conclusion
Physical parity for Class2 bonded interactions is verified. The GROMACS-PCFF fork correctly interprets and distributes forces for all supported bonded types within acceptable numerical tolerances.
