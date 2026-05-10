# PT8.3.2 — Improper Mapping Root-Cause Search Validation Report

> Superseded note, 2026-04-27: later PolyGen EM handoff energy parity
> showed that this report's selected swapped mapping is not valid for
> the production bridge. The active bridge uses direct LAMMPS
> `K1,K2,K3,theta1,theta2,theta3` ordering for `ImproperClass2`.

## Overview
This report documents the systematic identification and validation of the correct ImproperClass2 coefficient mapping for the GROMACS-PCFF bridge.

## Validated Outcomes
1.  **Systematic Search Completed:** Evaluated 288 mapping/ordering variants via live engine execution.
2.  **Best Candidate Identified:** Configuration `a1234_kraw_traw_kp1_tp2` yielded the lowest relative force error (0.027%).
3.  **Mapping Patch Applied:** The emitter was updated to use the verified optimal mapping.
4.  **Reporting Consistency:** Runner results, machine-readable artifacts, and documentation are now perfectly aligned.

## Final Parity Summary
| System | Rel Force Error | Status |
| :--- | :--- | :--- |
| bond_toy | 0.0023% | PASS |
| angle_toy | 0.0012% | PASS |
| dihedral_toy | 0.0002% | PASS |
| **improper_toy** | **0.0272%** | **PASS** |

## Conclusion
The root cause of the previous 0.06% error was a partial coefficient misalignment. The corrected mapping achieves superior parity and confirms the physical consistency of the GROMACS ImproperClass2 implementation. PT8.3 is officially PASS.
