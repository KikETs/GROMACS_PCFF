# R-RESPA M2f Validation Report

- Milestone: `R-RESPA M2f — Exact Pair-Write Ownership Proof for Excluded-Pair Coulomb Correction`
- Worktree: `..`
- Branch: `respa-m2-exact-three-level`
- Head commit at run start: `0ebd29773e7b49b85e4f53c703babe989fd6b360`

## Scope

- Fixture: `dense_oligomer` only
- Integrator path: exact 3-level `lammps-respa` only
- Timestep: coarse `dt = 0.0005 ps` only
- Step: `0` only
- Target path: first `excludedPairs` outer write into `forceWithVirial`
- Out of scope: merge-stage re-analysis, full-system TRL-5, production claims, TP1.xx work

## Starting Boundary

- M2 proved the exact 3-level path runs on the validated microfixture harness.
- M2b localized the dense step-0 energy defect to excluded-pair Coulomb correction ownership.
- M2c localized the dense step-0 force symptom to the excluded correction vector.
- M2d closed `postProcessForces` and `combineMtsForces` as the first duplication stage.
- M2e did not prove the first illegal write site; it only proved the first observed excluded outer write event.

## Files Changed

- `src/gromacs/mdlib/sim_util.cpp`
- `src/gromacs/nbnxm/pairlistset.cpp`
- `tools/run_respa_m2_microfixtures/run_respa_m2.py`
- `docs/validation_report_respa_m2f.md`

## Commands Run

- Build:
  - `cmake --build ../ab_builds/respa_m2_exact_three_level --target gmx -j4`
- M2f harness:
  - `python3 ../tools/run_respa_m2_microfixtures/run_respa_m2.py --gmx-bin ../ab_builds/respa_m2_exact_three_level/bin/gmx --fixture dense_oligomer --exact-pair-write-ownership-proof --milestone-name 'R-RESPA M2f' --out ../tests/reference_results/r_respa_m2f_exact_pair_write_ownership_proof`
- Full per-case command log:
  - `tests/reference_results/r_respa_m2f_exact_pair_write_ownership_proof/raw_commands.txt`

## Exact Pair-Write Evidence

The exact first excluded outer write boundary is captured, not just loop-level snapshots:

- before:
  - `tests/reference_results/r_respa_m2f_exact_pair_write_ownership_proof/dense_oligomer/dt_0p0005/exact_three_level/step0_outer_excluded_write_ord000_before.tsv`
- event:
  - `tests/reference_results/r_respa_m2f_exact_pair_write_ownership_proof/dense_oligomer/dt_0p0005/exact_three_level/step0_outer_excluded_write_ord000_event.tsv`
- after:
  - `tests/reference_results/r_respa_m2f_exact_pair_write_ownership_proof/dense_oligomer/dt_0p0005/exact_three_level/step0_outer_excluded_write_ord000_after.tsv`

The traced pair is:

- `ai = 0`
- `aj = 1`
- `shift_index = 22`
- `pair_list = excludedPairs`
- `contribution = outer`
- `buffer = forceWithVirial`
- `scalar = correction_scalar = -13.3029`

The before/after delta matches the per-pair event vector to numerical noise:

- `l2 = 2.2381524271391137e-06`
- `max_abs = 1.9073486328125e-06`

So M2f proves that this exact pair-write really writes the logged correction vector into the outer accumulator.

## Storage Identity Evidence

Pointer/backing-storage identity is dumped in:

- `tests/reference_results/r_respa_m2f_exact_pair_write_ownership_proof/dense_oligomer/dt_0p0005/exact_three_level/step0_force_storage_identity.txt`

Key findings:

- outer physical target:
  - `outer_accumulator_force_ptr = 0x5893814bd210`
- outer virial-backed force buffer:
  - `outer_accumulator_virial_ptr = 0x5893814bd210`
- outer shift-force buffer:
  - `outer_outputs_shift_force_ptr = 0x5893814c6e00`
- level-0 shift buffer:
  - `0.shift_force_ptr = 0x5893814c4800`
- level-1 shift buffer:
  - `1.shift_force_ptr = 0x5893814c5b00`

Within this narrow run:

- outer force buffer is disjoint from shift-force backing storage
- outer force buffer is disjoint from level-0 and level-1 shift buffers
- `outer_aliases_shift = false`

This rules out the specific aliasing hypothesis between the traced outer physical target and the shift-force storage.

## Ownership Lineage Evidence

The same pair is traced through pairlist construction artifacts:

- builder append trace:
  - `tests/reference_results/r_respa_m2f_exact_pair_write_ownership_proof/dense_oligomer/dt_0p0005/exact_three_level/step0_pairlist_builder_append_trace.txt`
- pairlist preview:
  - `tests/reference_results/r_respa_m2f_exact_pair_write_ownership_proof/dense_oligomer/dt_0p0005/exact_three_level/step0_plain_pairlist_preview.txt`
- membership scan:
  - `tests/reference_results/r_respa_m2f_exact_pair_write_ownership_proof/dense_oligomer/dt_0p0005/exact_three_level/step0_pair_key_membership_scan.txt`

These agree on the same first excluded entry:

- `kind = excludedPairs`
- `ordinal = 0`
- `ai = 0`
- `aj = 1`
- `shift_index = 22`

But topology parsing of `system.top` shows the same physical pair is already present in an earlier ownership bucket:

- `tests/reference_results/r_respa_m2f_exact_pair_write_ownership_proof/dense_oligomer/dt_0p0005/exact_three_level/system.top`
- pair `(1, 2)` appears under `[ bonds ]`

That is the key downgrade point. M2f proves the first visible consumer write, but not that this write is the first illegal owner.

## Independent Contribution Identity

The write is tied to the excluded-pair Coulomb correction by two independent checks:

1. exact boundary metadata logs `pair_list = excludedPairs` and `scalar = correction_scalar`
2. the same pair-write before/after delta matches the logged event vector directly, without relying on later total-force correlation

This is stronger than M2c/M2e, but still not enough for PASS because earlier ownership/spec error remains alive.

## Known-Good Control

The same tracing method was applied to a clean control pair path:

- control files:
  - `tests/reference_results/r_respa_m2f_exact_pair_write_ownership_proof/dense_oligomer/dt_0p0005/exact_three_level/step0_outer_pairs_write_ord0_before.tsv`
  - `tests/reference_results/r_respa_m2f_exact_pair_write_ownership_proof/dense_oligomer/dt_0p0005/exact_three_level/step0_outer_pairs_write_ord0_event.tsv`
  - `tests/reference_results/r_respa_m2f_exact_pair_write_ownership_proof/dense_oligomer/dt_0p0005/exact_three_level/step0_outer_pairs_write_ord0_after.tsv`

The chosen control pair is:

- `ai = 0`
- `aj = 4`
- `kind = pairs`
- `topology_sources = []`

For that control write:

- before/after delta matches the logged event vector exactly
  - `max_abs = 0`

This shows the tracing method can observe a clean single write on a non-problem path without inventing a false duplicate.

## Instrumentation Perturbation Control

M2f also runs a trace-off exact control:

- trace-off work dir:
  - `tests/reference_results/r_respa_m2f_exact_pair_write_ownership_proof/dense_oligomer/dt_0p0005/exact_three_level_trace_off`

Step-0 trace-on vs trace-off agreement:

- force diff `max_abs = 0`
- potential diff `= 0`

So the instrumentation did not materially perturb the narrow target quantities used for this milestone.

## Verdict

- `FIRST VISIBLE CONSUMER PROVEN; EARLIER OWNERSHIP STILL ALIVE`

## Why This Is Not PASS

PASS required proof that the traced first excluded outer write was itself the first illegal owner.
M2f does not have that proof, because the exact same pair is already present in an earlier topology ownership bucket (`bond`).
That leaves a live alternative explanation:

- the traced write may be the first visible consumer of an already-bad ownership/spec set, not the first illegal owner.

## Minimal Next Step

- Stay on `dense_oligomer`, coarse `dt = 0.0005`, step `0` only.
- Trace how pair `(0, 1)` enters `plainPairlist.excludedPairs` despite already belonging to the bonded ownership set.
