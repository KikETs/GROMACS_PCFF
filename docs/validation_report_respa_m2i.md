# R-RESPA M2i Validation Report

- Milestone: `R-RESPA M2i — First Downstream Mis-Consumption Trace for Valid Exclusion Set in Exact 3-Level Path`
- Worktree: `..`
- Branch: `respa-m2-exact-three-level`
- Head commit at run start: `7f5413f02c50930fae7e023729044c98e9bbf707`

## Scope

- Fixture: `dense_oligomer` only
- Integrator path: exact 3-level `lammps-respa` only
- Timestep: coarse `dt = 0.0005 ps` only
- Step relevance: `0` only
- Target pair: `(0,1)` only
- Narrow control: `(0,4)` only
- Out of scope: `generate_excl` origin, merge-stage, full-system TRL-5, production claims

## Starting Boundary

- M2h already proved that the bond-to-exclusion transition for `(0,1)` inside `generate_excl` is baseline-intended under `[ bonds ] 1 2` with `nrexcl = 3`.
- Therefore M2i traces only the downstream runtime contract, starting from valid runtime exclusion membership.

## Files Changed

- `src/gromacs/mdlib/sim_util.cpp`
- `tools/run_respa_m2_microfixtures/run_respa_m2.py`
- `docs/validation_report_respa_m2i.md`

## Commands Run

- Build:
  - `cmake --build ../ab_builds/respa_m2_exact_three_level --target gmx -j4`
- M2i harness:
  - `python3 ../tools/run_respa_m2_microfixtures/run_respa_m2.py --gmx-bin ../ab_builds/respa_m2_exact_three_level/bin/gmx --fixture dense_oligomer --downstream-misconsumption-trace --milestone-name 'R-RESPA M2i' --out ../tests/reference_results/r_respa_m2i_downstream_misconsumption_trace`
- Full command log:
  - `tests/reference_results/r_respa_m2i_downstream_misconsumption_trace/raw_commands.txt`

## Pair (0,1) Downstream Contract

The stage-by-stage contract stays legal until the exact excluded-pair consumer is admitted:

1. `runtime_exclusions_input`
   - target pair `(0,1)` is present in valid runtime exclusions
   - this is exclusion membership only, not physical force ownership
2. `append_plain_pairlist_branch`
   - target takes `branch = excludedPairs` because the interaction bit is cleared
   - this still means “not eligible for the standard physical nonbonded consumer”
3. `plain_pairlist_membership`
   - target is `in_plain_excluded = 1`, `in_plain_pairs = 0`
   - this remains valid bookkeeping
4. `excluded_pairs_dispatch_contract`
   - target is re-admitted with `include_rule = always_true`
   - the excluded-membership container is now treated as an exact nonbonded consumer input
5. `consumer_pair_eval`
   - target has `correction_scalar = -13.3029`
   - `outer_scalar = -13.3029`
   - `outer_force_write_eligible = true`
   - the valid exclusion membership has now become an actual physical outer-force consumer

## Earliest Misuse

The first downstream semantic misuse is:

- stage: `exact_excludedPairs_dispatch_contract`
- code site: `computeLammpsRespaNonbondedCpu()` dispatch of
  - `processPairlist(plainPairlist.excludedPairs, 0.0_real, 0.0_real, [](const int, const int) { return true; }, ...)`

Reason:

- reference semantics for pair `(0,1)` are already fixed by the cleared interaction bit and `branch = excludedPairs`
- so the pair is validly excluded from the standard physical nonbonded consumer
- the exact 3-level path first breaks that contract when it admits the entire `excludedPairs` container into a nonbonded correction consumer with `include_rule = always_true`
- the later pair-specific evaluation only confirms the misuse by showing a non-zero outer correction force written to `forceWithVirial`

## Reference Reconciliation

For pair `(0,1)`:

- reference semantics on the same run:
  - `masked_value = 0`
  - `branch = excludedPairs`
  - standard physical nonbonded consumer eligibility = false
- exact 3-level downstream semantics:
  - `excludedPairs` dispatch admits the pair with `include_rule = always_true`
  - pair-specific evaluation gives non-zero `outer_scalar`
  - physical outer-force consumer eligibility = true

That is the first contract divergence. The bug is no longer “why is the pair excluded?” The bug is “why does the exact path reinterpret excluded bookkeeping as a physical outer-force consumer?”

## Control Pair

Control pair `(0,4)` stays clean at the same trace depth:

- `branch = pairs`
- `in_plain_pairs = 1`, `in_plain_excluded = 0`
- admitted through the standard `pairs` dispatch
- pair-specific evaluation remains in the standard physical nonbonded consumer path

So the method distinguishes a clean runtime contract from the bad target contract on the same run.

## Verdict

- `FIRST DOWNSTREAM MIS-CONSUMPTION SITE IDENTIFIED`

## Narrow Conclusion

Within the locked scope `dense_oligomer`, exact 3-level, coarse `dt = 0.0005`, step `0`, pair `(0,1)`:

- `generate_excl` remains baseline-intended
- the first bad downstream site is not exclusions generation or merge/postprocess
- the first bad site is the exact excluded-pairs dispatch contract in `computeLammpsRespaNonbondedCpu()`
- the pair-specific outer correction write is the first concrete physical manifestation of that earlier contract misuse
