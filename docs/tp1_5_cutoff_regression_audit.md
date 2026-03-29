# TP1.5 — Cut-off Regression Audit Notes

## Goal

Explain why cut-off-only runs in TP1.3 regressed more severely than PME runs without broadening into transport work, code refactoring, or TP1.4 re-interpretation.

## Evidence Summary

Cut-off-only symptom:
- `TRL-5` is worse than the PME baseline in TP1.3 by both mean and peak temperature

Why PME alone is insufficient:
- `TRL-5` removes PME electrostatics entirely
- `TRL-0` and `TRL-5` still share the same repulsion-power-9 plain-C 4x4 short-range kernel family

Candidate path families entering TP1.5:
- exclusion mask handling
- listed-vs-nonlisted split
- reference loop / fallback path
- shift handling / periodic image bookkeeping
- neighbor-list update cadence

## What TP1.5 Actually Demonstrated

1. Path localization is stronger than before.
   - The cut-off-only path is now localized through `init_forcerec`, `init_nb_verlet`, `getCoulombKernelType`, `getVdwKernelType`, `nbnxn_kernel_cpu`, and `kernel_ref_inner.h`.

2. PME-only is no longer a sufficient explanation.
   - TP1.4 remains valid for PME split inconsistency.
   - TP1.5 shows the worse cut-off regression sits on a path that does not use PME reciprocal space.

3. Blanket exclusion failure is weakened.
   - The executed exclusion-sensitive 9-6 cut-off fixture still matches LAMMPS to machine precision.

4. Blanket shift failure is weakened.
   - The executed periodic-image invariance check is consistent within small relative tolerance on the cut-off path.

5. Neighbor-list cadence is weakened, not cleared.
   - TP1.3 keeps `nstlist = 10` in both `TRL-0` and `TRL-5`.
   - The pairlist radius/buffer is not identical, so the whole pairlist family is not closed.

## What TP1.5 Did Not Prove

TP1.5 did **not** prove:
- that exclusion handling is broken
- that listed/nonlisted splitting is broken
- that shift bookkeeping is broken
- that the remaining dense cut-off regression is dominated by one single line of code

Those claims would overreach the current evidence.

## Best Current Interpretation

The strongest bounded interpretation is:

- TP1.3 cut-off-only worsening is not a PME-only story
- the cut-off-only path of interest is the plain-C 4x4 short-range reference loop family used for repulsion power 9
- sparse exclusion and sparse shift checks do not fail
- the remaining likely problem is denser multi-atom application on that cut-off path, with pairlist/buffer effects still unresolved

## Machine-readable Artifacts

- `tests/reference_results/tp1_5_cutoff_audit/cutoff_fixture_definition.json`
- `tests/reference_results/tp1_5_cutoff_audit/cutoff_path_trace.json`
- `tests/reference_results/tp1_5_cutoff_audit/cutoff_regression_summary.json`
- `tests/reference_results/tp1_5_cutoff_audit/tp1_5_suspicion_ranking.json`
- `tests/reference_results/tp1_5_cutoff_audit/exclusion_mask_checks.csv`
- `tests/reference_results/tp1_5_cutoff_audit/raw_commands.txt`
- `tests/reference_results/tp1_5_cutoff_audit/provenance_manifest.json`

## Conservative Next Step

If TP1.5 is extended later, the next minimal step should be a dense 4-atom cut-off-only periodic fixture that varies pairlist radius or contact density while keeping exclusions absent. That would test the remaining dense reference-loop / pairlist-population suspicion directly without reopening TP1.4 or starting transport work.
