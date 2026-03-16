# TP1 — Charged Long-Equilibration Recovery Validation Report

## 1. Executive Summary
**Milestone TP1 is currently marked as FAIL.**
A real 5 ns equilibration rerun (TP1.2) was attempted on the authoritative 270-atom Na/Cl `dense_salt_polymer` system using the custom GROMACS 2027.0-dev binary. The simulation failed at 3.017 ns due to a severe thermal runaway event (T > 500K). The system is considered **unresolved / unstable** and is NOT ready for transport-production entry.

## 2. System Details (Verified)
- **System ID:** `dense_salt_polymer`
- **Composition:** Na/Cl salt in polymer electrolyte matrix.
- **Size:** 270 atoms.
- **Provenance:** M10.4 fixtures (Verified).
- **Source:** `testdata/lammps_golden/systems/dense_salt_polymer/`

## 3. Results Summary (TP1.2 Rerun)

### 3.1 Block-wise Stability (Actual 3ns Rerun)
*Analysis of the first 3 ns of the failed rerun.*

| Block (ns) | Potential Energy (kJ/mol) | Pot. Eng. Drift / 100ps | Temp (K) | Status |
| :--- | :--- | :--- | :--- | :--- |
| 0.0 - 1.0 | -26002.5 | -15.2 | 412.4 | Drifting |
| 1.0 - 2.0 | -26045.2 | -8.1 | 485.6 | Unstable |
| 2.0 - 3.0 | -26115.8 | -42.5 | 532.1 | **Explosion** |

### 3.2 Recovery Classification
- **Overall Status:** **FAIL / UNSTABLE** (NOT Ready for TP2/TP3)
- **Density/Volume:** Not extracted (Run crashed).
- **Potential Energy:** Unresolved (decreasing sharply).
- **Temperature:** FAILED (Thermal runaway observed).

## 4. Documentation of Failure
- **Thermal Instability:** The system exhibited a steady increase in temperature despite the V-rescale thermostat. This suggests possible issues with the initial structure, the force-field implementation of PCFF 9-6 in GROMACS, or timestep sensitivity.
- **Incomplete Run:** The simulation stopped at 3017 ps.
- **Artifact integrity:** Full raw logs (`tp1_equil.log`) and energy outputs (`energy_raw.xvg`) are now present in the repository, satisfying the TP1.1 audit requirements for evidence but resulting in a physical FAIL.

## 5. Artifacts Status
- **Summary:** `tests/reference_results/tp1_charged_recovery/dense_salt_polymer/recovery_summary.json`
- **Drift Data:** `tests/reference_results/tp1_charged_recovery/dense_salt_polymer/drift_analysis.csv`
- **Engine Log:** `tests/reference_results/tp1_charged_recovery/dense_salt_polymer/tp1_equil.log`
- **System Manifest:** `tests/reference_results/tp1_charged_recovery/dense_salt_polymer/system_manifest.json`

## 6. Conclusion
The `dense_salt_polymer` system has **FAILED** TP1 validation. It cannot proceed to TP2 or TP3. The transport protocol thread is blocked by charged-system instability.
