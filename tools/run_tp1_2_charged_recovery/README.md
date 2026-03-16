# TP1.2 — Charged Long-Equilibration Rerun Tooling

This directory contains the scaffold for the future TP1.2 rerun.
The original TP1 audit failed due to missing raw logs and system identity mismatch.
TP1.2 is required to produce auditable evidence.

## Runner Location
`tools/run_tp1_2_charged_recovery/run_tp1.py`

## Expected Inputs
- **GROMACS Structure/Topology:** From `testdata/lammps_golden/systems/dense_salt_polymer/` (after conversion to GROMACS).
- **Protocol Settings:** Derived from `docs/tp1_charged_long_equilibration_recovery.md` (5 ns NPT, 1 fs timestep).

## Expected Outputs (The TP1.2 Artifact Contract)
For a TP1.2 run to be considered PASS, the following MUST be committed to `tests/reference_results/tp1_charged_recovery/dense_salt_polymer/`:

1.  **`tp1_equil.log`**: Complete GROMACS engine log.
2.  **`recovery_summary.json`**: Machine-readable summary (overwriting the current UNTRUSTED version).
3.  **`drift_analysis.csv`**: Per-window density and energy drift values.
4.  **`energy.xvg`**: Raw energy/density trace from GROMACS `gmx energy`.
5.  **`system_manifest.json`**: Verification of system identity (270-atom Na/Cl polymer).

## Usage (Future)
```bash
python run_tp1.py --system dense_salt_polymer --duration_ns 5.0
```
