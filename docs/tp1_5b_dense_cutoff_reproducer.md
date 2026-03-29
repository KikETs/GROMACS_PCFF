# TP1.5b — Dense Cut-off Reproducer Notes

## Goal

Build the smallest dense periodic cut-off-only fixture that can reproduce the direction of worsening seen in TP1.3 cut-off runs, then separate:
- pairlist / `nstlist` / `rlist` / buffer behavior
- plain-C cut-off/reference kernel behavior
- listed-vs-nonlisted routing behavior

## Constraining Evidence Carried In

From TP1.3:
- the worse run is `TRL-5`
- `TRL-5` uses cut-off electrostatics and cut-off vdW
- `TRL-0` and `TRL-5` both still report repulsion power `9` and plain-C-4x4 short-range kernels

From TP1.5:
- blanket exclusion failure was weakened
- simple shift/PBC bookkeeping failure was weakened
- remaining live families were dense multi-atom cut-off application, pairlist/buffer behavior, and listed-vs-nonlisted routing

## Fixture Strategy

### Primary fixture: `dense_nonlisted`

Why this geometry:
- 4 atoms is the smallest setup that let us combine close contacts with a distinct near-cutoff cross-pair
- the `A2-A3` distance is `0.905 nm`, so it sits just beyond the physical cut-off while still being close enough to stress list lifetime behavior
- the system is periodic and cut-off only, which keeps it inside TP1.5b scope

### Sister fixture: `dense_routed_sister`

Why it exists:
- same dense geometry
- adds minimal exclusions and one explicit pair
- lets us ask whether the reproduced worsening depends on listed/nonlisted routing, without broadening into a new topology family

## Executed Separation Logic

### Pairlist / buffer axis

Used the same primary fixture while sweeping only:
- `nstlist`
- explicit `rlist`
- auto buffer vs explicit no-buffer

Key separation result:
- `n1_r0909` == `tight_ref_n1_r1200`
- `n10_r0909` and `n20_r0909` both worsen
- `auto_buffer_n10_vbt0005` returns to tight-reference behavior

This is the strongest evidence TP1.5b produced.

### Kernel / reference-path axis

Used fixed-frame reruns on the same dense primary fixture:
- `r0900`
- `r0909`

Observed:
- potential diff `0.0`
- force-component diff `0.0`

That weakens a simple fixed-frame cut-off kernel miscompute explanation.

### Listed-vs-nonlisted axis

Used the routed sister fixture in two ways:
- fixed-frame rerun at `r0900` and `r0909`
- dynamic comparison between `tight_ref` and `n10_r0909`

Observed:
- static invariance holds
- routed dynamic pairlist ratio is `1.0`, unlike the nonlisted fixture

Bounded meaning:
- routing still changes the overall physics of the sister fixture
- but it is not the main source of the pairlist-sensitive worsening reproduced on the primary dense fixture

## Conservative Interpretation

Confirmed:
- TP1.3-style cut-off worsening direction can be reproduced on a minimal dense periodic cut-off-only fixture
- the worsening is sensitive to pairlist lifetime controls

Weakened:
- fixed-frame kernel miscompute independent of pairlist lifetime
- listed-vs-nonlisted routing as the main driver of the reproduced worsening

Still unresolved:
- the exact per-step inclusion / pruning / accumulation defect inside the plain-C cut-off runtime family
- whether the same mechanism fully explains the larger TP1.3 dense salt-polymer instability

## Artifacts

Primary machine-readable artifacts:
- `tests/reference_results/tp1_5b_dense_cutoff_audit/dense_fixture_definition.json`
- `tests/reference_results/tp1_5b_dense_cutoff_audit/dense_cutoff_baseline_results.csv`
- `tests/reference_results/tp1_5b_dense_cutoff_audit/pairlist_sweep_results.csv`
- `tests/reference_results/tp1_5b_dense_cutoff_audit/runtime_path_trace.json`
- `tests/reference_results/tp1_5b_dense_cutoff_audit/tp1_5b_suspicion_ranking.json`

Additional evidence:
- `tests/reference_results/tp1_5b_dense_cutoff_audit/listed_vs_nonlisted_checks.csv`
- `tests/reference_results/tp1_5b_dense_cutoff_audit/raw_commands.txt`
- `tests/reference_results/tp1_5b_dense_cutoff_audit/provenance_manifest.json`
- raw logs, energy traces, and force dumps for all key runs under the same directory

## Exact Next Step

TP1.6 should stay narrow:
- instrument pairlist lifetime behavior on `dense_nonlisted`
- compare `tight_ref_n1_r1200`, `n1_r0909`, and `n10_r0909`
- preserve per-step inclusion/pruning evidence
- patch only after a concrete lifetime or routing failure is directly observed
