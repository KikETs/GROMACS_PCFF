# R-RESPA M2d Validation Report

- Milestone: `R-RESPA M2d — Dense Step-0 Duplicate-Correction Merge Trace`
- Worktree: `/home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2`
- Branch: `respa-m2-exact-three-level`
- Head commit at run start: `a060b7f81888b4f724de2a07bd3f29b79d706505`

## Scope

- Fixture: `dense_oligomer` only
- Timestep: coarse `dt = 0.0005 ps` only
- Step: `0` only
- Path under test: exact 3-level `lammps-respa`
- Out of scope: full-system TRL-5, production claims, TP1.xx work, performance, transport

## Starting Boundary

- M2 already proved the exact 3-level path really runs on the validated microfixture harness.
- M2b narrowed the dense step-0 energy bookkeeping defect to excluded-pair Coulomb correction ownership.
- M2c narrowed the dense force-side defect to a duplicated excluded-pair Coulomb correction contribution.
- M2d only traces where that extra correction survives into the physical total force.

## Files Changed

- `src/gromacs/mdlib/sim_util.cpp`
- `tools/run_respa_m2_microfixtures/run_respa_m2.py`
- `docs/validation_report_respa_m2d.md`

## Commands Run

- Build:
  - `cmake --build /home/kiket/바탕화면/test/ab_builds/respa_m2_exact_three_level --target gmx -j4`
- M2d harness:
  - `python3 /home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tools/run_respa_m2_microfixtures/run_respa_m2.py --gmx-bin /home/kiket/바탕화면/test/ab_builds/respa_m2_exact_three_level/bin/gmx --fixture dense_oligomer --dense-merge-trace --milestone-name 'R-RESPA M2d' --out /home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tests/reference_results/r_respa_m2d_dense_duplicate_correction_merge_trace`
- Full per-case command log:
  - `tests/reference_results/r_respa_m2d_dense_duplicate_correction_merge_trace/raw_commands.txt`

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

## Merge-Stage Trace

M2d traces the same excluded-pair Coulomb correction vector through these stage dumps:

- pre-postprocess outer buffer:
  - `tests/reference_results/r_respa_m2d_dense_duplicate_correction_merge_trace/dense_oligomer/dt_0p0005/exact_three_level/step0_level2_pre_postprocess_virial.tsv`
- post-postprocess outer buffer:
  - `tests/reference_results/r_respa_m2d_dense_duplicate_correction_merge_trace/dense_oligomer/dt_0p0005/exact_three_level/step0_level2_post_postprocess_shift.tsv`
- post-combine physical total:
  - `tests/reference_results/r_respa_m2d_dense_duplicate_correction_merge_trace/dense_oligomer/dt_0p0005/exact_three_level/step0_physical_postcombine.tsv`
- post-combine impulse total:
  - `tests/reference_results/r_respa_m2d_dense_duplicate_correction_merge_trace/dense_oligomer/dt_0p0005/exact_three_level/step0_impulse_postcombine.tsv`
- direct excluded correction dump:
  - `tests/reference_results/r_respa_m2d_dense_duplicate_correction_merge_trace/dense_oligomer/dt_0p0005/exact_three_level/exact_excluded_pairs_correction_force.tsv`

## Strongest Confirmed Finding

The duplicated excluded-pair Coulomb correction is already present before `postProcessForces`.

Direct evidence:

- outer pre-vs-post `postProcessForces` delta is exactly zero
  - `l2 = 0`
  - `max_abs = 0`
- `combineMtsForces` reconstructs the final physical total from the postprocessed inputs to numerical noise
  - reconstruction `l2 = 2.758485476721565e-05`
  - max-abs `= 3.814697265625e-06`
- subtracting the dumped excluded correction vector from that reconstructed total collapses the exact-vs-plain step-0 force mismatch
  - original `l2 = 742.3482152950181`
  - corrected `l2 = 0.0006322382118554644`

This means:

- the extra correction is not first created by `postProcessForces`
- the extra correction is not first created by `combineMtsForces`
- the extra correction is already inside the outer contribution before `postProcessForces`

## Strongest Remaining Limitation

M2d closes the merge-stage localization on the validated dense microfixture at step 0, but it does not yet identify the earlier accumulation site that caused the duplicated correction to already be present in the outer buffer.

## Verdict

- `DUPLICATION LOCALIZED BEFORE POSTPROCESS`

## Exact Next Step

- Stay on `dense_oligomer`, coarse `dt = 0.0005`, step `0` only and trace where the excluded-pair Coulomb correction first enters the outer `forceWithVirial` buffer before `postProcessForces`.
