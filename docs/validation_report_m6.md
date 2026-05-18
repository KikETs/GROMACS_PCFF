# M6 Validation Report

## Milestone Result

M6 is closed for the intended CPU-only 3-level prototype scope.

What is now true:

- `mts-mode = lammps-respa` has a real CPU runtime path, not only parser/design scaffolding.
- the frozen 3-level `small_oligomer` and `small_salt_polymer_box` NVE fixtures now pass strict automated regression against the frozen LAMMPS reference with explicit tolerances
- exact same-coordinate total-force parity versus unsplit GROMACS remains at about `1e-3 kJ/mol/nm` or better
- legacy MTS behavior and the exact-mode parser/runtime regressions still pass

What is still not true:

- GPU support exists
- arbitrary exact-mode restart/checkpoint equivalence is covered
- exact 2-level mode is a supported production path

## Implemented Coverage

Implemented:

- `mts-mode = lammps-respa`
- explicit ownership of `bond`, `angle`, `dihedral`, `improper`, `pair14`, `kspace`
- exact inner / middle / outer split of CPU real-space nonbonded work
- explicit base-step trace for the intended LAMMPS recursive event ordering
- dedicated CPU `integrator = md-vv` exact propagation path
- machine-readable LAMMPS-vs-GROMACS parity harness under `tools/pcff_respa_parity/`
- strict in-tree regression for the frozen M6 NVE fixtures in [pcff_short_md.cpp](../src/programs/mdrun/tests/pcff_short_md.cpp)

Intentionally not implemented:

- GPU nonbonded, PME, bonded, update, or GPU-resident scheduling
- thermostats, barostats, constraints, DD, free-energy perturbation in exact mode
- LJ-PME or shifted real-space modifiers inside exact mode

## Key Fix In This Pass

The dominant remaining blocker before GPU work was not pairlist buffering anymore. It was the exact kick path in [md.cpp](../src/gromacs/mdrun/md.cpp).

Facts:

- on slow MTS steps, [force.h](../src/gromacs/mdlib/force.h) documents that `force()` holds the physical total force
- [sim_util.cpp](../src/gromacs/mdlib/sim_util.cpp) combines `F_level0 + sum(F_slow)` into that same `force()` buffer
- the exact kick helper previously used `force()` directly for level 0

That meant the level-0 kick on slow steps used the physical total force instead of the fast force. Slow-level impulses were therefore counted twice: once inside the level-0 kick and again through explicit slow-level kicks.

The fix now reconstructs the exact level-0 force for exact kicks as:

- `F_level0 = F_physical_total - sum(F_slow_levels)`

This reconstruction is applied only in the exact `lammps-respa` kick path.

## Test Matrix

Executed in the current validation state:

- `build/bin/gmxpreprocess-test --gtest_filter='GetIrTest.MtsAcceptsExactLammpsRespaDefinitions:GetIrTest.MtsAcceptsExactLammpsRespaWithVelocityVerletIntegrator:GetIrTest.MtsRejectsExactLammpsRespaWithTwoLevels:GetIrTest.MtsRejectsLegacyForceListsInExactLammpsRespaMode:GetIrTest.MtsRejectsExactLammpsRespaWhenOuterCutoffExceedsPairCutoff:GetIrTest.MtsRejectsExactLammpsRespaWithReactionFieldCoulomb:GetIrTest.MtsRejectsExactLammpsRespaWithShiftedCoulombModifier:GetIrTest.MtsRejectsExactLammpsRespaWithShiftedVdwModifier:GetIrTest.MtsRejectsExactLammpsRespaWithoutPairlistBuffer'`
- `build/bin/mdtypes-test --gtest_filter='MultipleTimeStepping.AcceptsVelocityVerletOnlyForExactLammpsRespa:MultipleTimeStepping.FlattenedBaseStepTraceMatchesRecursiveLammpsReference:MultipleTimeStepping.ReportsLammpsRespaBaseStepTraceForThreeLevelSchedule:MultipleTimeStepping.RejectsTwoLevelExactLammpsRespaSchedule'`
- `build/bin/mdrun-test --gtest_filter='*ExactLammpsRespa*'`
- `build/bin/mdrun-non-integrator-test --gtest_filter='PcffSinglePointParity*:*PcffRespaObservableDump*'`
- `build/bin/mdrun-non-integrator-test --gtest_filter='PcffRespaRestartParity*'`
- `python3 tools/pcff_respa_parity/run.py --skip-build`
- `python3 tools/pcff_respa_parity/force_compare.py --skip-build`

Results:

- `gmxpreprocess-test`: 9 passed
- `mdtypes-test`: 4 passed
- `mdrun-test --gtest_filter='*ExactLammpsRespa*'`: 2 passed
- `mdrun-non-integrator-test --gtest_filter='PcffSinglePointParity*:*PcffRespaObservableDump*'`: 6 passed
- `mdrun-non-integrator-test --gtest_filter='PcffRespaRestartParity*'`: 2 passed
- `python3 tools/pcff_respa_parity/run.py --skip-build`: completed and updated [comparison_summary.json](../tests/reference_results/m6_respa/last_run_compare/comparison_summary.json)
- `python3 tools/pcff_respa_parity/force_compare.py --skip-build`: completed and updated [aggregate_force_compare.json](../tests/reference_results/m6_respa/force_compare_last/aggregate_force_compare.json)

## Numerical Agreement

### Frozen LAMMPS NVE parity

Current deltas from [comparison_summary.json](../tests/reference_results/m6_respa/last_run_compare/comparison_summary.json):

- `small_oligomer`
  - `step0_potential_kcal_mol`: `-0.0028642265`
  - `initial_total_kcal_mol`: `-0.0028656323`
  - `final_total_kcal_mol`: `-0.0025483825`
  - `total_energy_drift_abs_kcal_mol`: `-0.0003172498`
  - `total_energy_span_kcal_mol`: `-0.0003172498`
- `small_salt_polymer_box`
  - `step0_potential_kcal_mol`: `-0.0153952263`
  - `initial_total_kcal_mol`: `-0.0153982040`
  - `final_total_kcal_mol`: `-0.0131378602`
  - `total_energy_drift_abs_kcal_mol`: `-0.0022603436`
  - `total_energy_span_kcal_mol`: `-0.0021429201`

The frozen structural observables are closer than `1e-4 nm` in both systems.

These values are now enforced in-tree through the M6 reference TSV tolerance contract:

- [small_oligomer/reference_summary.tsv](../tests/reference_results/m6_respa/small_oligomer/reference_summary.tsv)
- [small_salt_polymer_box/reference_summary.tsv](../tests/reference_results/m6_respa/small_salt_polymer_box/reference_summary.tsv)

The same frozen contract now also includes step-0 virial-pressure tensor components in `atm`:

- `small_oligomer`: all `xx/yy/zz/xy/xz/yz` components pass within `5 atm`
- `small_salt_polymer_box`: all `xx/yy/zz/xy/xz/yz` components pass within `5 atm`

### Same-coordinate force parity

Current same-coordinate exact-vs-unsplit force deltas from [aggregate_force_compare.json](../tests/reference_results/m6_respa/force_compare_last/aggregate_force_compare.json):

- overall worst component delta: `9.765625e-4 kJ/mol/nm`
- overall worst atom-norm delta: `1.0066175843793117e-3 kJ/mol/nm`
- overall worst RMS component delta: `1.922203076950665e-4 kJ/mol/nm`

This matters because it separates force correctness from propagation correctness:

- the exact force split is consistent with the unsplit Hamiltonian on identical coordinates
- the earlier large short-NVE mismatch was not a force-field bookkeeping problem anymore
- the remaining pre-fix blocker was the kick decomposition

## Scope Assessment

This is a real CPU scheduler prototype, not only design scaffolding.

It is also not silently changing unrelated integrator behavior:

- exact mode remains opt-in through `mts = yes` and `mts-mode = lammps-respa`
- the dedicated `md-vv` path is only used for the supported exact subset
- unsupported settings still fail explicitly

## Remaining Limits Before GPU Work

1. Exact 3-level CPU NVE parity is now strong enough for the frozen fixtures. Broader exact-mode breadth is still missing.
2. Exact 2-level mode is now rejected explicitly instead of being left half-supported.
3. Virial parity is frozen only for the step-0 tensor in the two M6 fixtures. Broader virial coverage is still missing.
4. Restart/checkpoint smoke is covered only for outer-boundary restarts. Arbitrary mid-period exact-mode termination is not supported.
5. GPU work must preserve the current strict M6 regression. If GPU work lands without re-running [pcff_short_md.cpp](../src/programs/mdrun/tests/pcff_short_md.cpp) and the Python harness under [tools/pcff_respa_parity](../tools/pcff_respa_parity), the validation bar drops immediately.

## Readiness Assessment

Ready now:

- start GPU nonbonded work on top of the frozen 3-level CPU exact path
- keep the current M6 NVE fixtures as the CPU reference gate for that work

Not ready to claim:

- full exact-mode breadth across all schedules and algorithms
- production restart parity away from outer-force boundaries
- GPU correctness before the current M6 parity tests are wired into the GPU development loop
