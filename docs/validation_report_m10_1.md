# M10.1 — Trajectory Parity, Integration Convergence & NPT Stability Validation Report

## Overview
This report documents the validation scope of M10.1 after the gate definition was tightened.

M10.1 is not a general cross-engine ensemble-parity milestone. It is a short-time handoff gate with the following ownership:
- deterministic parity: neutral and charged NVE only
- short-time stability/trend diagnostics: NVT and NPT
- ensemble-level parity: deferred to M10.2

## Validated Outcomes
1.  **Deterministic NVE parity**
    - `small_oligomer_nve_dt0.0001`: PASS
    - `small_oligomer_nve_dt0.0005`: PASS
    - `small_salt_polymer_box_nve_dt0.0001`: PASS
2.  **NVT/NPT short-time diagnostics**
    - `small_oligomer_nvt_dt0.001`: diagnostic only
    - `small_oligomer_npt_dt0.001`: diagnostic only
    - `small_salt_polymer_box_nvt_dt0.001`: diagnostic only
3.  **Handoff decision**
    - M10.1 gate status is determined by deterministic NVE passes only.
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

## Artifacts Produced
- `tools/run_m10_1_trajectory_gate/run_m10_1.py`: deterministic/stability gate runner.
- `tests/reference_results/m10_1_trajectory_gate/`: Directory containing per-protocol comparison reports and logs.
- `tests/reference_results/m10_1_trajectory_gate/m10_1_gate_decision.json`: explicit handoff decision artifact.

## Conclusion
M10.1 is complete only as a deterministic handoff gate. It should not be cited as proof of NVT/NPT ensemble parity. Any claim about ensemble-level transport readiness must be based on M10.2 or later milestones.
