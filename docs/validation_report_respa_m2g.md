# R-RESPA M2g Validation Report

- Milestone: `R-RESPA M2g — Upstream Ownership/Spec Handoff Trace for Bonded Pair Entering excludedPairs`
- Worktree: `..`
- Branch: `respa-m2-exact-three-level`
- Head commit at run start: `a1d0876e7a35320719e660f672bdca44c4ec3ebf`

## Scope

- Fixture: `dense_oligomer` only
- Integrator path: exact 3-level `lammps-respa` only
- Timestep: coarse `dt = 0.0005 ps` only
- Step: `0` only
- Target pair: `(0,1)` only
- Out of scope: merge-stage re-analysis, force-buffer merge logic, full-system TRL-5, production claims

## Starting Boundary

- M2 proved the exact 3-level path runs on the validated microfixture harness.
- M2b localized the dense step-0 energy defect to excluded-pair Coulomb correction ownership.
- M2c localized the force symptom to the duplicated excluded correction vector.
- M2d closed `postProcessForces` and `combineMtsForces` as the first duplication stage.
- M2f proved the first visible excluded outer write, but not the earliest upstream bad owner.

## Files Changed

- `src/gromacs/gmxpreprocess/gpp_nextnb.cpp`
- `src/gromacs/nbnxm/pairlist.cpp`
- `src/gromacs/nbnxm/pairlistset.cpp`
- `tools/run_respa_m2_microfixtures/run_respa_m2.py`
- `docs/validation_report_respa_m2g.md`

## Commands Run

- Build:
  - `cmake --build ../ab_builds/respa_m2_exact_three_level --target gmx -j4`
- M2g harness:
  - `python3 ../tools/run_respa_m2_microfixtures/run_respa_m2.py --gmx-bin ../ab_builds/respa_m2_exact_three_level/bin/gmx --fixture dense_oligomer --upstream-ownership-handoff-trace --milestone-name 'R-RESPA M2g' --out ../tests/reference_results/r_respa_m2g_upstream_ownership_handoff_trace`
- Full command log:
  - `tests/reference_results/r_respa_m2g_upstream_ownership_handoff_trace/raw_commands.txt`

## Pair (0,1) Lineage

The exact narrow lineage is now explicit:

1. Topology inputs
   - `system.top` has `[ moleculetype ] ... nrexcl = 3`
   - `system.top` has `[ bonds ] 1 2`
2. `generate_excl()` output
   - atom `0` exclusions become `0,1,2,3`
   - target pair `(0,1)` is present
   - control pair `(0,4)` is absent
3. Runtime pairlist construction input
   - `top.excls` arrives at pairlist construction with the same exclusion membership
4. Runtime exclusion-bit clear
   - the `(0,1)` interaction bit is explicitly cleared by `jAtom_in_topology_exclusions`
5. `appendPlainPairlistCpu()` branch
   - target pair takes `branch = excludedPairs`
   - control pair takes `branch = pairs`
6. Plain pairlist append / membership
   - target pair is appended as `excludedPairs ordinal=0`
   - control pair remains in `pairs`

## Earliest Divergence

The earliest concrete bad handoff is:

- stage: `generate_excl_output`
- fault class: `upstream_spec_overlap_policy_defect`

Reason:

- pair `(0,1)` is already bonded in topology
- under `nrexcl = 3`, `generate_excl()` materializes that bonded pair into the atom-0 exclusion list
- later runtime stages only consume that already-bad dual-membership state

This closes the earlier live alternative from M2f. The bad state is not first introduced by `appendPlainPairlistCpu()`.

## Append-Branch Proof

The exact append branch is now proven by direct rule inputs:

- runtime bit clear:
  - `mask_before = 2246`
  - `bit_mask = 2`
  - `masked_before = 2`
  - `mask_after = 2244`
  - `masked_after = 0`
  - `rule = jAtom_in_topology_exclusions`
- append branch:
  - `masked_value = 0`
  - `predicate_mask_nonzero = false`
  - `predicate_excluded_branch = true`
  - `branch = excludedPairs`

So the append branch fires because the pair has already been marked excluded by upstream topology exclusions.

## Dual-Membership Result

Yes. Pair `(0,1)` simultaneously belongs to:

- bonded topology ownership world: `bond`
- exclusion world: `generate_excl()` output for atom `0`

The first materialized dual-membership point is `generate_excl_output`.

## Known-Good Control

Control pair `(0,4)` stays clean under the same instrumentation:

- no topology ownership source
- not present in `generate_excl()` output
- not present in runtime exclusions input
- `masked_value = 1`
- `branch = pairs`
- appended as `pairs ordinal=0`

So the tracing method distinguishes a normal lineage from the bad target lineage on the same run.

## Verdict

- `EARLIEST OWNERSHIP/SPEC HANDOFF DEFECT IDENTIFIED`

## Narrow Conclusion

Within the locked scope `dense_oligomer`, exact 3-level, coarse `dt=0.0005`, step `0`, the earliest bad handoff is upstream of runtime pair packing:

- bonded topology + `nrexcl = 3`
- `generate_excl()` materialization into exclusions
- runtime exclusion-bit clear
- faithful `excludedPairs` append

This is not a full-system claim and not a fix. It only identifies the earliest bad ownership/spec handoff for the traced pair.
