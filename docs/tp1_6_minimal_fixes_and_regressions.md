# TP1.6 — Minimal Fixes and Focused Regressions

## Scope decision

TP1.6 did **not** carry forward TP1.4/TP1.5 conclusions blindly. The current repository state was re-checked first.

### Selected for fix
- **Mixed-type 9-6 LJ-PME startup assert**
  - Confirmed by code-path inspection and direct execution.
  - Smallest fix was to relax stale LJ-PME pair-rule assertions in setup/dispatch.

### Deferred
- **Existing TP1.4 split mismatch**
  - Still reproduced.
  - Root cause was not narrowed to a single non-speculative math patch in this milestone.

### Rejected from fix set
- **TP1.5 "ghost energy"**
  - Current source and docs show RF shift contributions on excluded pairs/self interactions are expected for the Verlet scheme.
  - No current-code failing fixture justified a code change.

## Evidence snapshots

### Pre-fix mixed-type 9-6 LJ-PME
- `tests/reference_results/tp1_6_regressions/pre_fix_mixed_type_9_6_mdrun_stderr.txt`
- Failure: assertion abort in `src/gromacs/nbnxm/nbnxm_setup.cpp:554`

### Post-fix mixed-type 9-6 LJ-PME
- `tests/reference_results/tp1_6_regressions/post_fix_mixed_type_9_6_mdrun_stderr.txt`
- `tests/reference_results/tp1_6_regressions/post_fix_mixed_type_9_6_energy.xvg`
- Result: `mdrun` completed, `Potential = -4.385970 kJ/mol`

### Existing TP1.4 split scan
- `tests/reference_results/tp1_6_regressions/pre_fix_tp1_4_existing_split_stdout.txt`
- `tests/reference_results/tp1_6_regressions/post_fix_tp1_4_existing_split_stdout.txt`
- Result: essentially unchanged drift across `rcut`

## Exact next-step recommendation

Use a new milestone to isolate the remaining TP1.4 split mismatch at kernel level. Start from the unchanged `test_split.py` regression, then construct a parity fixture that compares:

1. direct-space LJ-PME correction terms,
2. reciprocal-space terms,
3. total force/energy invariance as `rcut` moves,

while changing only one dimension at a time:

- 12-6 homogeneous
- 9-6 homogeneous
- 9-6 mixed-type

Do not patch LJ-PME force/energy formulas before one of those fixtures points to a single correction path.
