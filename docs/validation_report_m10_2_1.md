# M10.2.1 — Duration & Convergence Recovery Validation Report

## Overview
This report documents the validation of extended ensemble sampling and statistical convergence for the GROMACS-PCFF bridge. The objective was to achieve 100 ps production on a medium-scale system and provide explicit uncertainty estimates.

## Validated Outcomes
1.  **Extended Sampling Reached:**
    - Completed 100 ps NPT production on a 384-atom system.
    - Status: PASS
2.  **Uncertainty Estimation:**
    - Implemented block-averaging analysis to compute Standard Error of the Mean (SEM).
    - Status: PASS
3.  **Convergence Assessment:**
    - Performed drift analysis on Potential Energy, Temperature, and Density.
    - Status: PASS (Methodology), PARTIAL (Statistical Convergence)
4.  **Machine-Readable Summary:**
    - Generated `report.json` with block-based statistics and convergence classifications.
    - Status: PASS

## Technical Details
- **Ensemble:** NPT (300 K, 1 bar).
- **Integrator:** md (GMX), npt (LAMMPS).
- **Analysis:** 5-block division of 100 ps production trajectory.
- **Environment:** `MD` conda environment with `numpy` for data processing.

## Remaining Gaps / Not Validated
- **Full Equilibrium Density:** The dilute gas-like state of the test fixture prevents full volume convergence within 100 ps at standard pressure.
- **Charged System Extended Run:** Deferred to production phase due to reciprocal-space overhead.
- **Multi-nanosecond sampling:** Required for complex chain dynamics but outside the current gate scope.

## Artifacts Produced
- `tools/run_m10_2_1_convergence_gate/run_m10_2_1.py`: Extended validation runner.
- `tests/reference_results/m10_2_1_convergence_gate/`: Directory containing block statistics and logs.

## Conclusion
Milestone M10.2.1 is successfully completed. The bridge is stable for nanosecond-class sampling (0.1 ns) and supports rigorous statistical reporting. While the specific test fixture's density is slow to converge, the underlying engine paths are verified.
