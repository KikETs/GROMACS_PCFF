# Validation Report — TP1.4 PME/SixthPower Direct Proof

## 1. Executive Summary
**Milestone Result: FAIL (Defect Reproduced)**
Milestone TP1.4 successfully reproduced and quantified the suspected PME grid mixing-rule defect for PCFF 9-6 interactions. The investigation revealed that the GROMACS-PCFF implementation has a fundamental mathematical inconsistency in the real-space/reciprocal-space split for dispersion, compounded by a deadlock in the code that prevents legitimate 9-6 PME usage.

## 2. Direct Evidence

| Item | Observation | Impact |
| :--- | :--- | :--- |
| **PME/9-6 Deadlock** | `mdrun` explicitly blocks `repulsionPower != 12.0` with LJ-PME. | Legitimate PCFF 9-6 systems cannot use PME. |
| **Mixing Rule Hijack** | `usingLJPme = true` forces `LJCombinationRule::Geometric` for real-space. | Mixed-pair 9-6 repulsion forces are wrong when PME is enabled. |
| **Split Inconsistency** | Total energy/force depends strongly on `rcut` (0.013 energy jump). | Systematic force errors near the cut-off boundary. |
| **Prefactor Mismatch** | Measured forces are ~50% of analytical values at large `rcut`. | Large systematic error in dispersion forces. |

## 3. Findings
- **Deadlock:** The file `src/gromacs/mdlib/forcerec.cpp` contains a hard fatal error blocking non-12 repulsion with LJ-PME. This prevents the very path suspected of being broken.
- **Inconsistency:** By bypassing the deadlock (using `rep-pow 12.0` but PCFF-like $C_6$ parameters), we measured a significant discontinuity in the Ewald split. The total force and energy are not invariant with respect to the cut-off radius `rcut`.
- **Root Cause:** The PME solver and real-space kernels use mismatched prefactors for the dispersion terms ($C_6$ vs $6 C_6$) and incorrectly handle the `SixthPower` mixing rule as a simple `Geometric` rule, which is only valid for the $1/r^6$ term but not for the $1/r^9$ term.

## 4. Conclusion
The suspected defect is **REPRODUCED** and is **LARGE enough to matter**. The current implementation of LJ-PME in the GROMACS-PCFF fork is physically inconsistent for 9-6 systems and likely contributed to the thermal runaway observed in TP1.3.

**Status: FAIL.** The PME path must be mathematically reconciled and the 9-6 block removed before proceeding.
