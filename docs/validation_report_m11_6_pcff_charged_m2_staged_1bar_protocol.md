# M11.6 PCFF Charged M2 Staged 1 Bar Protocol

Date: 2026-04-07

This is a staged-protocol artifact for the ambient `1 bar` M2 follow-up. The campaign summary now reports `PASS`, but the passing claim is limited to pressure-preconditioned staged `1 bar` dense parity.

## Scope

The frozen protocol is:

- precondition stage: `250 bar`, `100 ps`
- target stage: `1 bar`, `100 ps`
- final target analysis window: `50 ps`
- density relative-difference threshold: `<= 0.05`
- volume relative-difference threshold: `<= 0.05`
- required systems:
  - `gate_h_dense_salt_polymer_2x2x2`
  - `monoglyme_ethane_litfsi_1to1_dense18`

The target claim boundary, if the campaign passes, is only:

> pressure-preconditioned `1 bar` staged dense charged density/volume parity

It is not:

- ambient `1 bar` equilibrium dense parity
- generic charged dense-box readiness
- charged transport readiness
- broad PCFF chemistry readiness

## Protocol Artifacts

- Protocol: `tests/reference_results/pcff_charged_expansion/m2_broad_v4_staged_250bar_to_1bar/m2_staged_1bar_protocol.json`
- Campaign summary: `tests/reference_results/pcff_charged_expansion/m2_broad_v4_staged_250bar_to_1bar/m2_staged_1bar_campaign_summary.json`
- Runner: `tools/run_pcff_charged_m2_broad/run_m2_staged_1bar.py`

## Anti-Cherry-Pick Rule

The staged campaign can pass only if every predeclared required system passes the staged target gate. A single passing system is not enough.

## Result

Campaign verdict: `PASS`

System-level target-stage results:

| System | Target Density Rel Diff | Target Volume Rel Diff | Verdict |
|---|---:|---:|---|
| `gate_h_dense_salt_polymer_2x2x2` | `0.02789669955166171` | `0.027156417682212955` | `PASS` |
| `monoglyme_ethane_litfsi_1to1_dense18` | `0.037764712115395525` | `0.03977580260686641` | `PASS` |

The result is stronger than the prior high-pressure-only M11.4 boundary because it adds a staged `1 bar` target gate after a paired `250 bar` precondition.

The result is still weaker than an ambient `1 bar` equilibrium claim because the target stage starts from pressure-preconditioned dense coordinates.
