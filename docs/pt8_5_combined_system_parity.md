# PT8.5 — Combined Small-System Parity Verification Findings

## Overview
This document summarizes the numeric parity verification for combined Class2 interaction sets in realistic small systems.

## Results Summary
| System | Type | Energy Diff (kJ/mol) | Max Force Diff (kJ/mol/nm) | Rel Force Diff (%) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `small_oligomer` | Neutral | 0.0184 | 0.1714 | 0.0042% | PASS |
| `small_salt_polymer_box` | Charged | 0.3455 | 3.6875 | 0.0368% | PASS |

## Observations
1.  **Bonded/Nonbonded Interplay:** The successful validation of `small_oligomer` confirms that the GROMACS-PCFF fork correctly handles the simultaneous calculation of Class2 bond, angle, and dihedral terms alongside LJ 9-6 nonbonded interactions and 1-4 pair exclusions.
2.  **Electrostatic Consistency:** In `small_salt_polymer_box`, despite the use of different reciprocal-space algorithms (PME in GROMACS vs. PPPM in LAMMPS), energy parity was maintained within 0.35 kJ/mol. This confirms the correct interpretation of PCFF partial charges and Coulomb scaling.
3.  **Force Stability:** The relative force errors remain extremely low ($< 0.04\%$), proving that the combined force distribution is physically consistent across the system.

## Conclusion
The GROMACS-PCFF bridge is numerically consistent for combined interaction sets on these frozen small fixtures. This is a small-fixture mechanics result, not a claim of broad PCFF charged readiness, dense charged ensemble parity, or charged transport validity.
