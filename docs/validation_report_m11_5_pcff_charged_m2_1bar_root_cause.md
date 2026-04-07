# M11.5 PCFF Charged M2 Ambient 1 Bar Root-Cause Note

Date: 2026-04-07

This is a root-cause note, not an M2 broader PASS artifact.

## Scope

System under probe: `monoglyme_ethane_litfsi_1to1_dense18`.

Question: why does the ambient `1 bar` broader M2 dense NPT parity campaign fail while the explicit `250 bar` protocol passes?

## Evidence Artifacts

- Summary: `tests/reference_results/pcff_charged_expansion/m2_1bar_root_cause/m2_1bar_root_cause_summary.json`
- Analyzer: `tools/run_pcff_charged_m2_broad/analyze_1bar_root_cause.py`
- Checksum manifest: `tests/reference_results/pcff_charged_expansion/m2_1bar_root_cause/sha256_manifest.txt`

## Observed Facts

The formal ambient `1 bar` campaign remains failed:

- GROMACS final-window density: `12.763570504990021 kg/m^3`
- LAMMPS final-window density: `22.0575029500998 kg/m^3`
- Density relative difference: `0.42135016217090554`
- Volume relative difference: `0.5708016395087039`

LAMMPS-only ambient `1 bar` with the larger `4.0 A` neighbor skin reproduces gas-like expansion:

- Initial density: `880.5959 kg/m^3`
- Final density at `100 ps`: `14.317396 kg/m^3`
- Final `50 ps` mean density: `23.378330289421157 kg/m^3`

Therefore the LAMMPS neighbor skin/check setting is not the primary cause of the ambient failure.

The `250 bar` dense endpoint behaves differently when released to `1 bar`:

- LAMMPS `250 bar` endpoint to `1 bar`, final `50 ps` mean density: `1332.7929123752494 kg/m^3`
- GROMACS `250 bar` endpoint to `1 bar`, `10-20 ps` mean density: `1255.4987043663366 kg/m^3`

Therefore the ambient failure is path-dependent. The direct `1 bar` release from the generated dense18 state expands or cavitates to gas-like volumes, while a pressure-preconditioned endpoint remains condensed over the tested diagnostic horizons.

## Current Interpretation

The most supported current failure mode is pressure-path / initial-basin dependence, not parser/emitter failure and not a LAMMPS neighbor-list artifact.

The GROMACS endpoint-to-`1 bar` probe is only `20 ps`. It is diagnostic evidence only. It cannot be used as formal M2 parity evidence.

## Non-Claims

This note does not claim:

- ambient `1 bar` broader M2 PASS
- fully symmetric ambient dense parity
- transport readiness
- production readiness
- broad PCFF charged dense-system readiness

## Next Action

Do not reinterpret the existing `250 bar` formal campaign as ambient parity. If ambient M2 is still the target, freeze a new staged ambient protocol before running it. The candidate protocol should explicitly include pressure preconditioning and a final `1 bar` target stage, then apply the same gate to both predeclared systems with no cherry-pick escape.
