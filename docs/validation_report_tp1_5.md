# Validation Report — TP1.5 Cut-off Regression / Exclusion-Reference Path Audit

## 1. Executive Summary

**Milestone Result: PARTIAL**

TP1.5 narrowed the cut-off-only regression path beyond TP1.3/K1/TP1.4. The severe `TRL-5` worsening is confirmed to run on the plain-C 4x4 cut-off short-range path, not on PME reciprocal-space code. That means a PME-only explanation is insufficient.

The current minimal executed checks do **not** support a blanket exclusion-mask failure. An exclusion-sensitive cut-off 9-6 parity fixture still matches the LAMMPS reference, and a shift-sensitive periodic cut-off fixture is invariant within tight relative tolerance. The remaining suspicion is therefore narrower: a dense multi-atom application issue on the cut-off-only plain-C reference path, with neighbor-list/pairlist population differences still unresolved.

## 2. Prior Evidence Boundaries

Observed cut-off-only symptom from TP1.3:
- `TRL-5` (`Cut-off`) mean temperature `826.8 K`, max `1070.4 K`
- `TRL-0` (`PME`) mean temperature `738.0 K`, max `829.9 K`

Why PME alone is insufficient:
- `TRL-5` uses `coulombtype = Cut-off` and `vdw-type = Cut-off`
- it therefore bypasses PME reciprocal electrostatics entirely
- both `TRL-0` and `TRL-5` still report:
  - `Detected LJ repulsion power 9.`
  - `Using plain-C-4x4 4x4 nonbonded short-range kernels`

Bounded interpretation:
- TP1.4 remains `PARTIAL` and continues to support a real PME split inconsistency
- TP1.5 adds that cut-off-only worsening must involve a non-PME path as well

## 3. Localized Cut-off-Only Path

The current cut-off-only path map is preserved in:
- `tests/reference_results/tp1_5_cutoff_audit/cutoff_path_trace.json`

Most relevant path segments:

1. `src/gromacs/mdlib/forcerec.cpp::init_forcerec`
   - detects LJ repulsion power `9`
   - disables SIMD nonbonded kernels
   - forces the plain-C reference path
2. `src/gromacs/nbnxm/nbnxm_setup.cpp::chooseLJCombinationRule`
   - selects `LJCombinationRule::None` for non-12 repulsion
   - uses the full pair matrix for cut-off LJ 9-6
3. `src/gromacs/nbnxm/nbnxm_setup.cpp::init_nb_verlet`
   - builds the NBNxM cut-off short-range setup
   - applies pairlist tuning via `setupDynamicPairlistPruning`
4. `src/gromacs/nbnxm/kerneldispatch.cpp::getCoulombKernelType`
   - maps `CoulombInteractionType::Cut` to the `ReactionField` kernel family
5. `src/gromacs/nbnxm/kerneldispatch.cpp::getVdwKernelType`
   - selects the cut-off LJ kernel with `LJCombinationRule::None`
6. `src/gromacs/nbnxm/kerneldispatch.cpp::nbnxn_kernel_cpu`
   - dispatches the plain-C-4x4 CPU kernel
7. `src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h`
   - applies exclusions, cut-off masking, RF-style Coulomb, LJ 9-6, and shift bookkeeping inside the plain-C reference loop

## 4. Executed Minimal Checks

### A. Exclusion-sensitive cut-off parity fixture

Fixture:
- reused the existing `exclusion_toy` 4-atom 9-6 cut-off system
- path exercised:
  - exclusion handling
  - listed/nonlisted split
  - plain-C cut-off reference loop

Result:
- GROMACS potential: `-0.095356 kJ/mol`
- LAMMPS potential: `-0.095356000104 kJ/mol`
- absolute difference: `1.04e-10 kJ/mol`

Interpretation:
- a blanket exclusion-mask failure is **not** supported
- a blanket listed-vs-nonlisted split failure is also weakened

### B. Shift-sensitive periodic cut-off fixture

Fixture:
- 2 atoms, periodic `2 x 2 x 2 nm` box
- cut-off-only, repulsion power `9`
- compared two PBC-equivalent frames

Result:
- potential relative difference: `5.33e-06`
- force-magnitude relative difference: `6.93e-06`

Interpretation:
- no direct shift-bookkeeping defect was reproduced on this minimal fixture

### C. TP1.3 pairlist cadence comparison

Static but direct log evidence:
- `TRL-0`: `updated every 10 steps, buffer 0.000 nm, rlist 0.900 nm`
- `TRL-5`: `updated every 10 steps, buffer 0.009 nm, rlist 0.909 nm`

Interpretation:
- neighbor-list cadence alone is weakened as the sole explanation
- pairlist radius/buffer differences remain unresolved

## 5. Confirmed / Plausible / Unresolved

Confirmed:
- the cut-off-only worsening runs on the same plain-C-4x4 short-range kernel family as the PME trials
- the cut-off-only worsening is therefore not explainable by PME reciprocal-space defects alone

Plausible contributor:
- dense multi-atom application of the plain-C cut-off reference loop

Weakened:
- blanket exclusion-mask bug
- blanket listed-vs-nonlisted split bug
- blanket shift/PBC bookkeeping bug
- neighbor-list cadence as the sole explanation

Unresolved:
- whether dense pairlist population / buffer-radius differences on the plain-C cut-off path amplify the regression in ways not hit by the current minimal sparse fixtures

## 6. PME-Only Assessment

TP1.5 result:
- PME-only explanation: **split**

Meaning:
- TP1.4 remains relevant for PME runs
- TP1.5 shows cut-off-only worsening needs an additional non-PME path explanation

## 7. Verdict

TP1.5 improved path localization and executed two minimal cut-off-only checks, but it did not directly reproduce a dense-system cut-off defect on a new minimal fixture. That is enough for a bounded audit result, not for a full closure.

Verdict: **PARTIAL**
