# TP0 — Transport Acceptance Criteria

## 1. Overview
This document defines the success metrics for transport-property validation milestones. These criteria are frozen to prevent drift in PASS/FAIL interpretations during the validation campaign.

## 2. Observable Accuracy Targets

| Observable | System Type | Pass Criteria (Rel. Error) | Partial Pass | Fail Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **Diffusion ($D$)** | Neutral | $< 15\%$ | $15\% - 30\%$ | $> 30\%$ |
| **Diffusion ($D$)** | Charged | $< 20\%$ | $20\% - 40\%$ | $> 40\%$ |
| **Conductivity ($\sigma_{NE}$)** | Charged | $< 25\%$ | $25\% - 50\%$ | $> 50\%$ |
| **Transference Number ($t_+$)** | Charged | $\pm 0.05$ (abs) | $\pm 0.10$ (abs) | $>\pm 0.10$ (abs) |

## 3. Convergence Requirements
An observable is only valid for comparison if the following convergence criteria are met:
- **MSD Linearity:** $R^2 > 0.99$ for the primary species in the fitting window (20% - 80%).
- **Block Stability:** Relative uncertainty from 5-block averaging must be $< 10\%$.
- **Density Stability:** Drift in production density must be $< 0.1\%$ per ns.

## 4. Milestone PASS/FAIL Logic
- **PASS:** All primary observables for all in-scope systems meet "Pass Criteria".
- **PARTIAL:** At least 50% of observables meet "Pass Criteria" and none are "Fail".
- **FAIL:** Any observable in any in-scope system meets "Fail Criteria", or convergence is not reached within frozen durations.

## 5. Reporting Requirements
Each validation result must report:
1.  **Estimator Used:** (e.g., Einstein MSD).
2.  **Fitting Window:** (e.g., 2 ns to 8 ns).
3.  **Reference Value:** LAMMPS result with source provenance.
4.  **Absolute & Relative Error:** Relative to the LAMMPS reference.
5.  **Statistical Uncertainty:** Based on block averaging.
6.  **Simulation MDP/Metadata:** Checksum or filename of the settings used.
