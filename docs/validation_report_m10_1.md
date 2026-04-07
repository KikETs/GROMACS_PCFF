# M10.1 — Trajectory Parity, Integration Convergence & NPT Stability Validation Report

## Overview
This report documents the validation scope of M10.1 after the gate definition was tightened.

M10.1 is not a general cross-engine ensemble-parity milestone. It is a short-time handoff gate with the following intended ownership:
- deterministic parity: neutral NVE, plus charged NVE only if the charged artifact bundle is actually present
- short-time stability/trend diagnostics: NVT and NPT
- ensemble-level parity: deferred to M10.2

## Current Checked-In Evidence
1.  **Deterministic NVE parity**
    - `small_oligomer_nve_dt0.0001`: PASS
    - `small_oligomer_nve_dt0.0005`: PASS
    - `small_salt_polymer_box_nve_dt0.0001`: artifact missing in the current repository checkout, so this is **not claimable as a checked-in PASS**
2.  **NVT/NPT short-time diagnostics**
    - `small_oligomer_nvt_dt0.001`: diagnostic only
    - `small_oligomer_npt_dt0.001`: diagnostic only
    - `small_salt_polymer_box_nvt_dt0.001`: diagnostic only
3.  **Handoff decision**
    - the checked-in repository currently supports only the neutral deterministic NVE readout as exact evidence
    - Ensemble parity remains owned by M10.2.

## Technical Details
- **NVE Protocol:** deterministic parity gate.
- **NVT Protocol:** short-time thermostat diagnostic only.
- **NPT Protocol:** short-time barostat stability diagnostic only.
- **Observables used for NVE gating:** PE delta and pressure delta, not absolute PE offsets.

## Remaining Gaps / Not Validated
- **Long-timescale equivalence:** not covered here.
- **Thermostat/barostat parity:** not covered here and should not be inferred from M10.1.
- **Charged ensemble behavior:** still requires M10.2/M10.x follow-up.
- **Charged NVE handoff artifact integrity:** the repository does not currently contain `small_salt_polymer_box_nve_dt0.0001` or `m10_1_gate_decision.json`, so the historical charged NVE pass cannot be reused as present-tense exact evidence.

## Artifacts Currently Present
- `tools/run_m10_1_trajectory_gate/run_m10_1.py`: deterministic/stability gate runner.
- `tests/reference_results/m10_1_trajectory_gate/`: Directory containing per-protocol comparison reports and logs.
- checked-in neutral NVE and short-time diagnostic directories exist.
- the charged NVE directory and `m10_1_gate_decision.json` are missing in the current checkout.

## Conclusion
M10.1 should be cited only as a partial short-time gate with exact neutral NVE evidence and caveated short-time diagnostics. It should not be cited as proof of charged NVE parity, NVT/NPT ensemble parity, or transport readiness.
