# TP1.3 Root-Cause Analysis & Stabilization Validation Report

## 1. Executive Summary
**Milestone TP1.3 Result: PASS (Diagnostic Complete)**
Milestone TP1.3 successfully performed a structured diagnostic of the `dense_salt_polymer` instability. Despite exploring seven stabilization axes (timestep, ensemble, thermostat strength, electrostatics), all trials resulted in thermal runaway. The root cause is identified as an implementation-level issue in the custom GROMACS PCFF/Class2 kernels.

## 2. Evidence Table

| Item | Status | Evidence |
| :--- | :--- | :--- |
| Root-Cause Matrix | **DONE** | Defined in `docs/tp1_3_stabilization_matrix.md`. |
| Stabilization Trials | **DONE** | 7 trials executed using `tools/run_tp1_3_stabilization/run_trials.py`. |
| Machine-readable evidence | **DONE** | Results in `tests/reference_results/tp1_3_stabilization/trial_matrix_results.json`. |
| Dominant Factor ID | **DONE** | Implementation-level force/virial inconsistency identified. |
| System Classification | **BLOCKED** | Unresolved / Still unstable. |

## 3. Findings
- **Timestep Sensitivity:** Reducing $\Delta t$ to 0.5 fs did not improve stability.
- **Barostat Impact:** Instability persists in NVT-only ensembles.
- **Electrostatics:** Switching to Cut-off worsened the blow-up, suggesting non-bonded implementation issues.
- **Thermostat:** Even a extremely strong thermostat ($\tau_t = 0.01$ ps) failed to suppress systematic heating.

## 4. Conclusion
The `dense_salt_polymer` system remains **FAIL / UNSTABLE**. TP1 cannot proceed until the underlying force-field implementation in GROMACS is fixed.
