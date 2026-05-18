# R-RESPA M2j Validation Report

- Milestone: `R-RESPA M2j — Dispatch-Internal Minimal-Fix Isolation Inside exact_excludedPairs_dispatch_contract`
- Worktree: `..`
- Branch: `respa-m2-exact-three-level`
- Head commit at run start: `3be070f6714214d8c30addd012a3e34978f8569f`

## Scope

- Fixture: `dense_oligomer` only
- Integrator path: exact 3-level `lammps-respa` only
- Timestep: coarse `dt = 0.0005 ps` only
- Step relevance: `0` only
- Target pair: `(0,1)` only
- Narrow control: `(0,4)` only
- Out of scope: `generate_excl` origin, merge-stage, broad physics claims, production claims

## Starting Boundary

- M2i already proved that the first bad downstream site is the exact excluded-pairs dispatch contract.
- M2j therefore does not re-prove dispatch membership. It decomposes the bad dispatch site into:
  - `includePair` admission
  - active contribution selection
  - outer routing / `forceWithVirial`
  - and one narrower correction-to-outer probe

## Files Changed

- `src/gromacs/mdlib/sim_util.cpp`
- `tools/run_respa_m2_microfixtures/run_respa_m2.py`
- `docs/validation_report_respa_m2j.md`

## Commands Run

- Build:
  - `cmake --build ../ab_builds/respa_m2_exact_three_level --target gmx -j4`
- M2j harness:
  - `python3 ../tools/run_respa_m2_microfixtures/run_respa_m2.py --gmx-bin ../ab_builds/respa_m2_exact_three_level/bin/gmx --fixture dense_oligomer --dispatch-minimal-fix-isolation --milestone-name 'R-RESPA M2j' --out ../tests/reference_results/r_respa_m2j_dispatch_minimal_fix_isolation`
- Full command log:
  - `tests/reference_results/r_respa_m2j_dispatch_minimal_fix_isolation/raw_commands.txt`

## Baseline Dispatch-Internal Trace

For the target pair `(0,1)` inside the exact excluded-pairs dispatch:

1. `dispatch_internal_include_pair`
   - `include_pair_effective = true`
   - target is admitted into the excluded-pairs consumer
2. `dispatch_internal_active_contributions`
   - baseline active set is `inner,middle,outer`
   - only the outer path is live for the target because `inner_scalar = 0`, `middle_scalar = 0`, `outer_scalar = -13.3029`
3. `dispatch_internal_outer_routing`
   - `outer_force_write_eligible = true`
   - `outer_routing_target = forceWithVirial`
   - `actual_outer_write_executed = true`

This means the baseline exact path turns valid exclusion bookkeeping into a live physical outer-force consumer inside the bad dispatch site.

## Counterfactual Probes

### 1. `includePair` restricted

- Admission is blocked for `(0,1)` before consumer evaluation.
- Result: first-bad semantics disappears.
- Interpretation: changing only `includePair` is sufficient, but not minimal.

### 2. active contributions narrowed

- Admission remains.
- Effective active set becomes `inner,middle`.
- `effective_outer_active = false`
- Result: first-bad semantics disappears.
- Interpretation: narrowing active contributions is sufficient, but not minimal.

### 3. outer routing suppressed

- Admission remains.
- Outer contribution remains live with `effective_outer_scalar = -13.3029`
- Physical outer write is suppressed.
- Result: the earlier bad semantics remains, but physical realization is blocked.
- Interpretation: outer routing is not the first bad semantic sub-decision.

### 4. narrower correction-to-outer suppression

- Admission remains.
- Active set remains `inner,middle,outer`.
- `effective_outer_active = true`
- `effective_outer_scalar = 0`
- Physical outer write disappears.
- Result: first-bad semantics disappears while admission, active contribution configuration, and routing framework remain otherwise baseline.
- Interpretation: this is the narrowest proven minimal-fix candidate.

## Candidate Verdicts

- `includePair policy`: `SUFFICIENT-BUT-NOT-MINIMAL`
- `activeContributions configuration`: `SUFFICIENT-BUT-NOT-MINIMAL`
- `outer routing / forceWithVirial selection`: `NOT-CAUSAL`
- narrower proven candidate: `excluded correction -> outer contribution selection` = `MINIMAL-FIX-CANDIDATE`

## Reference Reconciliation

For pair `(0,1)`:

- reference bookkeeping contract from the same run keeps the pair excluded from the standard physical nonbonded consumer
- baseline exact dispatch re-admits the pair and promotes `correction_scalar` into a live outer contribution
- the narrower correction-to-outer suppression probe preserves admission but removes the live outer contribution

Therefore the first minimal semantic defect inside the already-bad dispatch site is not generic admission, not generic active-contribution configuration, and not routing alone. It is the promotion of excluded-pair correction into the effective outer physical contribution.

## Control Pair

Control pair `(0,4)` stays clean under all probes:

- admitted through the standard `pairs` path
- keeps `effective_outer_active = true`
- keeps `actual_outer_write_executed = true`
- does not flip semantics under the target-only probes

So the method distinguishes the bad target contract from a clean nearby runtime contract at the same trace depth.

## Verdict

- `DISPATCH-INTERNAL MINIMAL FIX CANDIDATE ISOLATED`

## Narrow Conclusion

Within the locked scope `dense_oligomer`, exact 3-level, coarse `dt = 0.0005`, step `0`, pair `(0,1)`:

- the first bad dispatch site from M2i remains correct
- inside that site, the narrowest proven minimal-fix candidate is the excluded correction to outer-contribution promotion path
- broader fixes at the `includePair` or generic active-contribution level would work, but are not minimal
