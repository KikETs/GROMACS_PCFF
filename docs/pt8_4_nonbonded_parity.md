# PT8.4 — Nonbonded & 1-4 / Exclusions Parity Verification Findings

## Overview
This document summarizes the numeric parity verification for LJ 9-6 and 1-4/Exclusion semantics between LAMMPS and the GROMACS-PCFF fork.

## Calculation Details
- **Engine Comparison:** LAMMPS (`units real`) vs. GROMACS (mixed precision fork).
- **Tolerance:** 0.01 kJ/mol (achieved errors are significantly lower).

## Results Summary
Potential Energy Comparison (kJ/mol):

| System | Semantic | LAMMPS (Ref) | GROMACS | Diff | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `lj96_toy` | Pure LJ 9-6 | -0.183572 | -0.183554 | 1.84e-05 | PASS |
| `exclusion_toy` | 1-4 / Exclusions | -0.095356 | -0.095356 | 1.04e-10 | PASS |

## Observations
1.  **LJ 9-6 Consistency:** The GROMACS-PCFF fork correctly interprets the `rep-pow 9.0` directive in the topology to switch from standard 12-6 to 9-6 behavior. The agreement with LAMMPS `pair_style lj/class2` is excellent.
2.  **Combination Rule Accuracy:** PCFF-style sigma/epsilon mixing (SixthPower rule) was verified to match LAMMPS `mix sixthpower`.
3.  **Exclusion Semantics:** The bridge correctly maps LAMMPS `special_bonds` to GROMACS `nrexcl` and `[ pairs ]`. In `exclusion_toy`, setting `nrexcl 3` successfully suppressed 1-2 and 1-3 interactions, allowing only the 1-4 pair interaction to contribute to the energy.
4.  **Coulomb Path:** While not the primary focus, initial checks show that `fudgeQQ 1.0` correctly includes 1-4 Coulomb interactions at full strength, as expected for PCFF.

## Conclusion
The GROMACS-PCFF bridge provides verified numeric parity for the critical nonbonded semantics required for PCFF simulations. The implementation of LJ 9-6 and exclusion rules is physically consistent with the LAMMPS reference.
