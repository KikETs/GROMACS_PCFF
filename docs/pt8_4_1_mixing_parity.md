# PT8.4.1 — Cross-Type Mixing Parity Verification Findings

## Overview
This document summarizes the numeric parity verification for off-diagonal (mixed) LJ 9-6 interactions between LAMMPS and the GROMACS-PCFF fork.

## Calculation Details
- **Engine Comparison:** LAMMPS (`units real`) vs. GROMACS (mixed precision fork).
- **Tolerance:** 0.01 kJ/mol (achieved errors are $\sim 10^{-5}$ kJ/mol).
- **Mixing Rule:** PCFF Sixth-Power mixing.

## Results Summary
Potential Energy Comparison (kJ/mol):

| System | Semantic | LAMMPS (Ref) | GROMACS | Diff | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mixing_toy` | Off-Diagonal LJ 9-6 | -0.257216 | -0.257178 | 3.81e-05 | PASS |

## Observations
1.  **Mixing Rule Consistency:** The GROMACS-PCFF fork correctly applies the `SixthPower` combination rule (rule 4) to derive off-diagonal parameters for the LJ 9-6 potential.
2.  **Repulsion Power Alignment:** Verified that `rep-pow 9.0` correctly affects the repulsion term of the mixed interaction, maintaining parity with LAMMPS `pair_style lj/class2`.
3.  **Gap Closure:** This milestone closes the off-diagonal validation gap left by PT8.4, confirming that the emitter and kernel correctly handle multi-type nonbonded scenarios.

## Conclusion
The GROMACS-PCFF bridge provides verified numeric parity for mixed LJ 9-6 interactions. This completes the independent validation of nonbonded semantics before proceeding to combined-system testing.
