# TP1 — Charged Long-Equilibration Recovery Plan & Rerun Result

## 1. System Selection (Verified)
- **Target System:** `dense_salt_polymer`
- **Identity:** 270 atoms, Na/Cl salt in generic polymer matrix.
- **Source:** `testdata/lammps_golden/systems/dense_salt_polymer/`

## 2. Rerun Summary (TP1.2)
A 5 ns rerun was attempted but failed at 3.017 ns.

### 2.1 Execution Details
- **Runner:** `tools/run_tp1_2_charged_recovery/run_tp1.py`
- **Engine:** custom GROMACS 2027.0-dev (PCFF support).
- **Execution Command:** `python3 tools/run_tp1_2_charged_recovery/run_tp1.py --duration_ps 5000.0`
- **Actual Duration:** 3.017 ns.
- **Termination Reason:** Implicit crash or shell timeout following thermal runaway.

### 2.2 Actual Simulation Settings
- **Timestep:** 1.0 fs.
- **Thermostat:** V-rescale ($T=300$ K, $\tau_t=0.5$ ps).
- **Barostat:** Berendsen ($P=1.0$ bar, $\tau_p=5.0$ ps).
- **Electrostatics:** PME, $r_{coulomb}=0.9$ nm, `ewald-rtol=1e-5`.
- **VdW:** Cut-off, $r_{vdw}=0.9$ nm, `DispCorr=no` (9-6 potential constraint).

## 3. Recovery Analysis Results (TP1.2)
| Window (ns) | Potential Energy (kJ/mol) | Temperature (K) |
| :--- | :--- | :--- |
| 0.0 - 1.0 | -26002.5 | 412.4 |
| 1.0 - 2.0 | -26045.2 | 485.6 |
| 2.0 - 3.0 | -26115.8 | 532.1 |

**Classification:** **FAIL / UNSTABLE.**

## 4. Required Artifacts (Satisfied)
The following artifacts are now committed and verifiable:
- `tools/run_tp1_2_charged_recovery/run_tp1.py`
- `tools/run_tp1_2_charged_recovery/analyze_tp1_2.py`
- `tests/reference_results/tp1_charged_recovery/dense_salt_polymer/`
    - `tp1_equil.log` (Full engine log up to crash)
    - `recovery_summary.json` (Refined analysis)
    - `drift_analysis.csv` (1 ns block statistics)
    - `energy_raw.xvg` (Energy trace)
    - `system_manifest.json` (Verification manifest)

## 5. Conclusion
TP1 validation is **FAIL**. TP2 and TP3 are currently blocked by charged-system instability in GROMACS with the current PCFF settings.
