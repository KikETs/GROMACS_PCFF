# R-RESPA M2e Validation Report

- Milestone: `R-RESPA M2e — Dense Step-0 Early Accumulation-Site Trace for Excluded-Pair Coulomb Correction`
- Worktree: `..`
- Branch: `respa-m2-exact-three-level`
- Head commit at run start: `0d6c1d0e61e47c22e43f2bd7e1960e23ccb058cd`

## Scope

- Fixture: `dense_oligomer` only
- Timestep: coarse `dt = 0.0005 ps` only
- Step: `0` only
- Path under test: exact 3-level `lammps-respa`
- Out of scope: full-system TRL-5, production claims, TP1.xx work, performance, transport

## Starting Boundary

- M2 already proved the exact 3-level path runs on the validated microfixture harness.
- M2b localized the dense step-0 energy defect to excluded-pair Coulomb correction ownership.
- M2c localized the dense step-0 force defect to a duplicated excluded-pair Coulomb correction vector.
- M2d exonerated `postProcessForces` and `combineMtsForces` as the first duplication stage.
- M2e only traces the earlier accumulation path into the outer `forceWithVirial` buffer.

## Files Changed

- `src/gromacs/mdlib/sim_util.cpp`
- `tools/run_respa_m2_microfixtures/run_respa_m2.py`
- `docs/validation_report_respa_m2e.md`

## Commands Run

- Build:
  - `cmake --build ../ab_builds/respa_m2_exact_three_level --target gmx -j4`
- M2e harness:
  - `python3 ../tools/run_respa_m2_microfixtures/run_respa_m2.py --gmx-bin ../ab_builds/respa_m2_exact_three_level/bin/gmx --fixture dense_oligomer --dense-early-accumulation-trace --milestone-name 'R-RESPA M2e' --out ../tests/reference_results/r_respa_m2e_dense_early_accumulation_site_trace`
- Full per-case command log:
  - `tests/reference_results/r_respa_m2e_dense_early_accumulation_site_trace/raw_commands.txt`

## Comparison Basis

- Exact fixture source:
  - `testdata/lammps_golden/systems/dense_oligomer/system.json`
- Coordinate/topology generation path:
  - `tools/run_respa_m2_microfixtures/run_respa_m2.py`
- Plain reference:
  - `md-vv` plain Verlet on the same dense microfixture
- Bounded simpler split:
  - `pme_legacy_side_reference`
  - not direct archived-M1 continuity

## Early Accumulation Trace

M2e traces the outer `forceWithVirial` buffer through these step-0 checkpoints:

- initial outer buffer:
  - `tests/reference_results/r_respa_m2e_dense_early_accumulation_site_trace/dense_oligomer/dt_0p0005/exact_three_level/step0_level2_initial_outer_virial.tsv`
- after plain pairs dispatch:
  - `tests/reference_results/r_respa_m2e_dense_early_accumulation_site_trace/dense_oligomer/dt_0p0005/exact_three_level/step0_level2_after_pairs_virial.tsv`
- after excluded pairs dispatch:
  - `tests/reference_results/r_respa_m2e_dense_early_accumulation_site_trace/dense_oligomer/dt_0p0005/exact_three_level/step0_level2_after_excluded_pairs_virial.tsv`
- before long-range nonbonded:
  - `tests/reference_results/r_respa_m2e_dense_early_accumulation_site_trace/dense_oligomer/dt_0p0005/exact_three_level/step0_level2_before_longrange_virial.tsv`
- after long-range nonbonded:
  - `tests/reference_results/r_respa_m2e_dense_early_accumulation_site_trace/dense_oligomer/dt_0p0005/exact_three_level/step0_level2_after_longrange_virial.tsv`
- pre-postprocess outer reconciliation:
  - `tests/reference_results/r_respa_m2e_dense_early_accumulation_site_trace/dense_oligomer/dt_0p0005/exact_three_level/step0_level2_pre_postprocess_virial.tsv`
- first excluded-pair outer write event:
  - `tests/reference_results/r_respa_m2e_dense_early_accumulation_site_trace/dense_oligomer/dt_0p0005/exact_three_level/step0_outer_first_excluded_write.tsv`
- direct excluded correction dump:
  - `tests/reference_results/r_respa_m2e_dense_early_accumulation_site_trace/dense_oligomer/dt_0p0005/exact_three_level/exact_excluded_pairs_correction_force.tsv`

## Strongest Confirmed Finding

The first illegal physical write into the outer `forceWithVirial` buffer occurs during the excluded-pairs outer dispatch inside `computeLammpsRespaNonbondedCpu()`.

Direct code basis:

- excluded-pair correction is computed through `correctionScalar` in
  - `src/gromacs/mdlib/sim_util.cpp:881-898`
- outer contribution sets
  - `scalar = correctionScalar + bareCoulombScalar * splitWeights.outer + factorLj * rawLjScalar * splitWeights.outer`
  - `src/gromacs/mdlib/sim_util.cpp:932-934`
- for the excluded-pairs dispatch in exact mode, `factorCoulomb = 0` and `factorLj = 0`, so the outer scalar reduces to the correction term
- that force is then written directly into the active outer accumulator with
  - `rvec_inc(accumulator.force[ai], force)`
  - `rvec_dec(accumulator.force[aj], force)`
  - `src/gromacs/mdlib/sim_util.cpp:967-968`

Direct trace basis:

- initial outer buffer is exactly zero
  - `max_abs = 0`
- after-excluded delta matches the dumped excluded correction vector
  - `max_abs = 8.106231689453125e-06`
  - alignment `= 0.999999999999996`
- before-longrange buffer equals after-excluded exactly
  - `max_abs = 0`
- after-longrange buffer equals pre-postprocess exactly
  - `max_abs = 0`
- removing the excluded correction from the outer contribution after the legitimate long-range update nearly closes against plain
  - corrected `max_abs = 6.771087646484375e-05`

The first-write event metadata is also specific:

- `pair_list = excludedPairs`
- `contribution = outer`
- `buffer = forceWithVirial`
- `alias_with_shift = false`
- `scalar = correction_scalar = -13.3029`

This rules out:

- zero-init failure
- aliasing between outer virial and shift-force buffers
- a later first-write in `postProcessForces`
- a later first-write in `combineMtsForces`

## Brief Exoneration of Later Stages

- `step0_level2_after_longrange_virial.tsv` equals `step0_level2_pre_postprocess_virial.tsv` exactly.
- M2d already showed `postProcessForces` does not create the extra correction.
- M2d already showed `combineMtsForces` reconstructs the final physical total from postprocessed inputs to numerical noise.

So the duplicated correction survives later stages, but it is already present before them.

## Strongest Minimal Fix Hypothesis

Hypothesis, not yet validated:

- in the exact 3-level path, the excluded-pairs dispatch should not physically add `correctionScalar` into the outer `forceWithVirial` buffer
- the narrowest candidate fix is the outer branch of `processPairlist(plainPairlist.excludedPairs, 0, 0, ...)` in `computeLammpsRespaNonbondedCpu()`
- that candidate fix should preserve the legitimate later long-range nonbonded write in `do_force()`

This is a narrow ownership hypothesis, not a production-ready fix claim.

## Verdict

- `EARLY ACCUMULATION SITE LOCALIZED`

## Exact Next Step

- Stay on `dense_oligomer`, coarse `dt = 0.0005`, step `0` only.
- Patch only the excluded-pairs outer correction write path in `computeLammpsRespaNonbondedCpu()`.
- Re-run the same M2e harness and require the raw exact step-0 total force to match plain directly, without subtracting the dumped excluded correction vector.
