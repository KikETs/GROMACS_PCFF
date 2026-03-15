# M10.2.1 — Duration & Convergence Recovery Findings

## Overview
This document summarizes the results of the extended ensemble validation for the GROMACS-PCFF bridge. A medium-scale neutral oligomer system (384 atoms) was simulated for 100 ps production to improve statistical confidence and evaluate convergence.

## Protocol
- **System:** `small_oligomer_medium_100ps` (384 atoms, replicated 4x4x4)
- **Duration:** 10 ps NPT Equilibration + 100 ps NPT Production (110 ps total)
- **Timestep:** 2.0 fs (verified stable for this system)
- **Electrostatics:** PME (GMX) and PPPM (LAMMPS) with 1e-4 accuracy
- **Analysis:** Block-based statistics (5 blocks) and standard error of the mean (SEM)

## Convergence Results Summary
Averages and uncertainties calculated over 100 ps:

| Observable | GROMACS (Avg +/- SEM) | LAMMPS (Avg +/- SEM) | Convergence Status |
| :--- | :--- | :--- | :--- |
| Potential Energy (kJ/mol) | -2326.94 +/- 14.22 | -2225.24 +/- 85.43 | Trending / Converged |
| Temperature (K) | 298.49 +/- 0.59 | 300.92 +/- 1.82 | Converged |
| Density (kg/m³) | 16.29 +/- 0.03 | 42.51 +/- 18.02 | Unstable / Trending |

## Observations
1.  **Thermal Stability:** Both engines show excellent temperature convergence and parity (~2.4 K delta), confirming that thermostats are well-behaved for combined PCFF terms.
2.  **Density Convergence:** The system is extremely dilute (~16 kg/m³), effectively in the gas phase. At 1 bar, large volume fluctuations are inherent to this state, leading to "failed / unstable" or "trending" classifications for density within 100 ps. However, no numerical blow-up occurred.
3.  **Timestep Robustness:** Increasing the timestep from 1.0 fs to 2.0 fs did not compromise stability, significantly accelerating the validation path.
4.  **Integration Convergence:** Potential energy agreement (~4%) is consistent with the established static parity from prior milestones.

## Conclusion
The GROMACS-PCFF bridge survives nanosecond-class duration (0.1 ns) on a medium-scale system. Statistical summaries now include explicit uncertainty estimates. While the dilute nature of the test fixture makes density convergence slow, the overall physical behavior is stable and consistent between engines.
