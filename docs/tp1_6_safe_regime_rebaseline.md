# TP1.6 Safe Regime Re-Baselining

## Goal

TP1.6 is a validation-regime milestone. It does not try to fix pairlist code. It asks whether the dense cut-off worsening from TP1.5b survives once the same fixture is moved from the TP1.5e `ALLOWED-UNSAFE` regime to safer pairlist/buffer settings.

## Constraining Prior Evidence

- TP1.5b: `n10_r0909` widened the dense fixture total-energy range from `8.657745` to `12.576325 kJ/mol`
- TP1.5c: the first cross-run membership divergence appeared under `n1_r0909` versus `n10_r0909`
- TP1.5d: the branch-level difference was rebuild cadence, not pruning
- TP1.5e: the omission was reclassified as `ALLOWED-UNSAFE`, because the critical pair was outside manual `rlist` at the last rebuild

So TP1.6 had one narrow job: rerun the same cut-off family under safer settings and see whether the worsening disappears, weakens, or persists.

## Safe-Regime Definition

The candidate set is recorded in [safe_regime_candidates.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_6_rebaseline_audit/safe_regime_candidates.json).

Cases executed:

- `tight_ref_n1_r1200`
  - role: tight diagnostic reference
  - why safer: large manual list margin
  - intended use: reference only
- `n10_r0909`
  - role: unsafe reference
  - why unsafe: manual `rlist = 0.909` leaves the critical pair outside the list at the last rebuild
- `n1_r0909`
  - role: safe control
  - why safer: rebuild every step, so no stale reuse
  - intended use: diagnostic control, not the preferred baseline
- `n10_r0911`
  - role: manual safe candidate
  - why safer: manual `rlist = 0.911` exceeds the TP1.5e critical distance at step `170`
  - intended use: secondary manual candidate
- `auto_buffer_n10_vbt0005`
  - role: preferred safe baseline candidate
  - why safer: positive `verlet-buffer-tolerance` restores automatic safe list sizing
  - intended use: preferred validation baseline candidate

## Controlled Reruns

All runs used the same TP1.5b `dense_nonlisted` fixture and kept the following fixed:

- topology
- coordinates
- box
- charges
- `rep-pow = 9`
- cut-off family (`Cut-off` / `plain-C-4x4`)
- seed and integrator family

Only pairlist/buffer settings changed.

The raw command trail is in [raw_commands.txt](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_6_rebaseline_audit/raw_commands.txt). Raw logs and energy outputs were preserved for all five runs. Final-frame force dumps were also preserved for the key comparison runs, but they were not used as a cross-run correctness metric because the dynamic trajectories diverge.

## Unsafe vs Safe Results

The main comparison table is [unsafe_vs_safe_comparison.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_6_rebaseline_audit/unsafe_vs_safe_comparison.csv).

Key rows:

- `n10_r0909`
  - runtime: `updated every 10 steps, buffer 0.009 nm, rlist 0.909 nm`
  - total-energy range: `12.576325 kJ/mol`
  - status: `worsening_persists`
- `n1_r0909`
  - runtime: `updated every 1 steps, buffer 0.009 nm, rlist 0.909 nm`
  - total-energy range: `8.657745 kJ/mol`
  - status: `worsening_removed`
- `n10_r0911`
  - runtime: `updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm`
  - total-energy range: `8.657745 kJ/mol`
  - status: `worsening_removed`
- `auto_buffer_n10_vbt0005`
  - runtime: `updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm`
  - total-energy range: `8.657745 kJ/mol`
  - status: `worsening_removed`

This is the main TP1.6 result: once the fixture leaves the unsafe reuse regime, the reproduced worsening vanishes on the same runtime family.

## Interpretation

The strongest evidence now points to unsafe-regime behavior, not to a surviving implementation issue on this fixture.

Why:

- the unsafe case still reproduces when rerun now
- safer regimes recover the tight reference exactly
- the preferred auto-buffer candidate and the safe manual-margin candidate converge to the same runtime pairlist line: `buffer 0.011 nm, rlist 0.911 nm`

What TP1.6 does **not** prove:

- that all short-range handling is globally correct
- that larger dense charged systems will automatically behave the same way
- that no other short-range issue survives outside this fixture family

So TP1.6 supports a safe re-baselining, not global closure.

## Recommendation Boundary

The narrow recommendation is:

- `auto_buffer_n10_vbt0005` is the preferred safe validation baseline for later toy and pre-authoritative charged-system checks
- `n1_r0909` remains useful as a diagnostic control
- `n10_r0911` is acceptable as a secondary manual-margin candidate, but auto-buffer is the cleaner baseline

The acceptance level is only partial for later validation beyond this toy fixture family. Larger-system reruns still need to demonstrate the same stability under the safe regime before being treated as authoritative.

## Source Patching

Source-level pairlist patching is still not justified in TP1.6.

Reason:

- TP1.5e already removed the contract-violation claim
- TP1.6 shows that safe settings remove the reproduced worsening on the same fixture
- patching code now would skip the more important regime/baseline correction
