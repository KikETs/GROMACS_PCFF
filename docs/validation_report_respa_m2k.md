# R-RESPA M2k Validation Report

- Milestone: `R-RESPA M2k — Narrow Patch Proof for Excluded-Correction Outer Promotion`
- Worktree: `/home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2`
- Branch: `respa-m2-exact-three-level`
- Clean head commit at run start: `cb211f1ee161c3abc5b4b143430385049b27f982`

## Scope

- Fixture: `dense_oligomer` only
- Integrator path: exact 3-level `lammps-respa` only
- Timestep: coarse `dt = 0.0005 ps` only
- Step relevance: `0` only
- Target pair: `(0,1)` only
- Control pair: `(0,4)` only
- Out of scope: all-pair closure, all-step closure, full-system claims, production claims

## Patch Candidates

- Patch-shape A
  - semantic intent: cut excluded-pair correction at raw `outerScalar` formation in the excludedPairs outer path
  - touched region: `src/gromacs/mdlib/sim_util.cpp`
- Patch-shape B
  - semantic intent: preserve raw `outerScalar` formation but block excluded-pair correction from becoming `effectiveOuterScalar`
  - touched region: `src/gromacs/mdlib/sim_util.cpp`

## Locked-Scope Results

- Baseline exact remains broken for target `(0,1)`
  - `outer_scalar_effective = -13.3029`
  - `actual_outer_write_executed = true`
  - exact step-0 total force does not close to plain
- Patch-shape A closes the target in locked scope
  - `outer_scalar_effective = 0`
  - `actual_outer_write_executed = false`
  - exact step-0 total-force diff vs plain: `max_abs = 6.866455078125e-05`
- Patch-shape B also closes the target in locked scope
  - `outer_scalar_effective = 0`
  - `actual_outer_write_executed = false`
  - exact step-0 total-force diff vs plain: `max_abs = 6.866455078125e-05`
- Control `(0,4)` remains unchanged under both patches
  - stays on the standard `pairs` path
  - keeps `effective_outer_active = true`
  - keeps `actual_outer_write_executed = true`

## Minimality Comparison

- Patch-shape A works, but changes raw `outerScalar` formation itself.
- Patch-shape B works with the same target closure and control preservation, while preserving raw `outerScalar` formation and only cutting the promotion into physical outer contribution.
- Therefore Patch-shape B is the narrower safer candidate.

## Verdict

- `PASS`
- preferred patch: `PATCH-SHAPE-B`

## Narrow Conclusion

Within the locked scope `dense_oligomer`, exact 3-level, coarse `dt = 0.0005`, step `0`, pair `(0,1)`:

- at least one narrow patch candidate is proven to close the target bug and preserve control
- both A and B work in locked scope
- B is narrower because it preserves raw scalar bookkeeping and only removes excluded correction from effective outer physical promotion
