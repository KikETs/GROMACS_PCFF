# TP1.2 Artifact Contract

This contract defines the required outputs for a future TP1.2 rerun to be considered PASS.

## 1. Input Requirements
- **System:** `dense_salt_polymer` (270 atoms).
- **Initial Configuration:** LAMMPS data/input converted to GROMACS.
- **Protocol:** `TP0_charged_equilibration_freeze_v1`.

## 2. Output Artifacts (The Mandatory Set)
All artifacts MUST be placed in `tests/reference_results/tp1_charged_recovery/dense_salt_polymer/`.

| Filename | Type | Description |
| :--- | :--- | :--- |
| `tp1_equil.log` | Text (GROMACS) | Full log of the 5 ns equilibration run. |
| `recovery_summary.json` | JSON (Metadata) | Machine-readable analysis of density and energy stability. |
| `drift_analysis.csv` | CSV (Data) | Per-window (1 ns) drift statistics. |
| `energy.xvg` | XVG (Data) | Time-series of density, volume, and potential energy. |
| `system_manifest.json` | JSON (Audit) | Verification of the 270-atom Na/Cl polymer identity. |

## 3. Validation Criteria
- **GROMACS Log Validation:** Log must show completion of the full 5 ns duration with a 1 fs timestep.
- **Density Stability:** Linear drift in the final 1 ns block must be $< 0.05\%$ per 100 ps.
- **System Parity:** The final configuration from TP1.2 must be bitwise-consistent with the expected Na/Cl structure (within GROMACS floating-point precision).

## 4. Replacement Protocol
The artifacts produced in TP1.2 will **fully replace** the current UNTRUSTED set from the failed TP1 audit.
The `status` field in `tests/reference_results/transport_protocol_metadata/tp1_status.json` will be updated to `PASS` only after a manual audit of these artifacts.
