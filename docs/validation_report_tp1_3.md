# TP1.3 Root-Cause Analysis & Stabilization Validation Report

## 1. Executive Summary
**Milestone TP1.3 Result: SUPERSEDED HISTORICAL DIAGNOSTIC**
Milestone TP1.3 performed a structured diagnostic of the `dense_salt_polymer` instability, but its stability verdict and root-cause inference are no longer authoritative because the runner did not apply the intended coupling keys.

Supersession note: this TP1.3 conclusion is historical and must not be used as the current exact-system verdict. The TP1.3 runner used the same wrong-key pattern (`tcouple` / `pcouple` / `gen_vel`) that prevented the intended thermostat/barostat contract from appearing in `mdout.mdp`. The later corrected exact TP1 rerun resolves the thermal-runaway blocker for the exact 5 ns `dense_salt_polymer` NPT protocol only; see [TP1 Charged Long-Equilibration Recovery](validation_report_tp1.md).

## 2. Evidence Table

| Item | Status | Evidence |
| :--- | :--- | :--- |
| Root-Cause Matrix | **DONE** | Defined in `docs/tp1_3_stabilization_matrix.md`. |
| Stabilization Trials | **DONE** | 7 trials executed using `tools/run_tp1_3_stabilization/run_trials.py`. |
| Machine-readable evidence | **DONE** | Results in `tests/reference_results/tp1_3_stabilization/trial_matrix_results.json`. |
| Dominant Factor ID | **SUPERSEDED** | Historical inference is not authoritative after the corrected-key rerun. |
| System Classification | **SUPERSEDED** | Corrected exact TP1 thermal-runaway recovery now passes; endpoint continuation safety remains blocked. |

## 3. Findings
- **Timestep Sensitivity:** Reducing $\Delta t$ to 0.5 fs did not improve stability.
- **Barostat Impact:** Instability persists in NVT-only ensembles.
- **Electrostatics:** Switching to Cut-off worsened the blow-up, suggesting non-bonded implementation issues.
- **Thermostat:** Even a extremely strong thermostat ($\tau_t = 0.01$ ps) failed to suppress systematic heating.

## 4. Conclusion
This TP1.3 report remains useful only as a historical diagnostic record. The current TP1 verdict is defined by [TP1 Charged Long-Equilibration Recovery](validation_report_tp1.md), not by the obsolete TP1.3 stabilization matrix.
