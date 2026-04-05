# R-RESPA M2c Validation Report

- Milestone: `R-RESPA M2c — Dense Exact-Path Force-Buffer Ownership Isolation`
- Worktree: `/home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2`
- Branch: `respa-m2-exact-three-level`
- Head commit at run start: `47a2c482027d7c910f527b0d8cd619281bcf9022`

## Scope

- Fixture: `dense_oligomer` only
- Step: `0` only
- Path under test: exact 3-level `lammps-respa`
- Out of scope: full-system TRL-5, production claims, TP1.xx work, performance, transport

## Files Changed

- `src/gromacs/mdlib/sim_util.cpp`
- `tools/run_respa_m2_microfixtures/run_respa_m2.py`

## Commands Run

- Build:
  - `cmake --build /home/kiket/바탕화면/test/ab_builds/respa_m2_exact_three_level --target gmx -j4`
- M2c harness:
  - `python3 /home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tools/run_respa_m2_microfixtures/run_respa_m2.py --gmx-bin /home/kiket/바탕화면/test/ab_builds/respa_m2_exact_three_level/bin/gmx --fixture dense_oligomer --dense-force-ownership-isolation --milestone-name 'R-RESPA M2c' --out /home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tests/reference_results/r_respa_m2c_dense_force_buffer_ownership_isolation`
- Full per-case command log:
  - `tests/reference_results/r_respa_m2c_dense_force_buffer_ownership_isolation/raw_commands.txt`

## Comparison Basis

- Exact fixture source:
  - `testdata/lammps_golden/systems/dense_oligomer/system.json`
- Coordinate/topology generation path:
  - `tools/run_respa_m2_microfixtures/run_respa_m2.py`
- Plain reference:
  - `md-vv` plain Verlet on the same microfixture
- Bounded simpler split:
  - `pme_legacy_side_reference`
  - Not direct archived-M1 continuity

## Strongest Confirmed Finding

- The excluded-pair Coulomb correction force is routed to outer level `3` (internal index `2`), initially through `forceWithVirial`, then merged via `postProcessForces`, then carried into the physical total force by `combineMtsForces`.
- Direct dump path:
  - `tests/reference_results/r_respa_m2c_dense_force_buffer_ownership_isolation/dense_oligomer/dt_0p0005/exact_three_level/exact_excluded_pairs_correction_force.tsv`
- Exact step-0 total force minus that dumped correction vector collapses the dense exact-vs-plain step-0 force mismatch from:
  - `l2 = 742.3482152950181`
  - to `l2 = 0.000633334307547299`
- The dumped correction vector aligns with the exact-minus-plain step-0 force delta at:
  - `cosine = 0.9999999999996434`

## Strongest Unresolved Uncertainty

- This milestone localizes the dense step-0 force defect to duplicated excluded-pair Coulomb correction ownership on the validated microfixture harness.
- It does not yet implement or validate the fix.
- It also does not generalize beyond `dense_oligomer` step `0`.

## Artifacts

- Top-level summary:
  - `tests/reference_results/r_respa_m2c_dense_force_buffer_ownership_isolation/summary.json`
- Dense fixture summary:
  - `tests/reference_results/r_respa_m2c_dense_force_buffer_ownership_isolation/dense_oligomer/fixture_summary.json`
- Exact total-force dump:
  - `tests/reference_results/r_respa_m2c_dense_force_buffer_ownership_isolation/dense_oligomer/dt_0p0005/exact_three_level/exact_total_force.tsv`
- Plain total-force dump:
  - `tests/reference_results/r_respa_m2c_dense_force_buffer_ownership_isolation/dense_oligomer/dt_0p0005/plain_verlet/plain_total_force.tsv`

## Verdict

- `FORCE DEFECT LOCALIZED TO DUPLICATED EXCLUDED-PAIR COULOMB FORCE OWNERSHIP`

## Exact Next Step

- Keep scope on `dense_oligomer` step `0` and patch only the excluded-pair Coulomb correction force ownership path in the exact 3-level implementation.
