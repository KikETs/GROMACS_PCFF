# TP1.5e Pairlist Contract Audit

## Goal

TP1.5e asks a narrower question than TP1.5d: is the reproduced omission for pair `(1,4, shift 21)` under `n10_r0909` a real implementation defect, or is it allowed by the current pairlist reuse contract when the user manually sets `rlist = 0.909` and disables automatic Verlet buffering with `verlet-buffer-tolerance = -1`?

## Constraining Prior Evidence

- TP1.5b showed that the dense cut-off worsening is pairlist-sensitive:
  - `n1_r0909` matches tight reference
  - `n10_r0909` worsens
  - `auto_buffer_n10_vbt0005` returns to tight-reference behavior
- TP1.5c showed that the first cross-run membership difference appears at step `171`
- TP1.5d showed that the relevant branch difference is rebuild cadence, not pruning

TP1.5e tightens one important interpretation point from TP1.5d: step `171` is the first step below `rlist`, not the first step below the actual cutoff. The pair first crosses below the real interaction cutoff only at step `172`.

## Contract Localization

The contract-relevant path map is stored in [contract_path_map.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5e_pairlist_contract_audit/contract_path_map.json).

Key locations:

- [sim_util.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp)
  - `doPairSearch`: rebuild scheduling
- [pairlistsets.h](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlistsets.h)
  - `numStepsWithPairlist`, `isDynamicPruningStepCpu`: list age and pruning cadence
- [pairlist_tuning.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlist_tuning.cpp)
  - `supportsDynamicPairlistGenerationInterval`: requires `verletbuf_tol > 0`
  - `setupDynamicPairlistPruning`: only enables dual-list pruning when that contract is active
- [calc_verletbuf.h](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdlib/calc_verletbuf.h)
  - positive `verlet-buffer-tolerance` defines an average energy-jump target over the list lifetime
- [nbnxm_setup.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/nbnxm_setup.cpp)
  - plain-C cut-off path still receives pairlist contract setup

External basis checked on `2026-03-19`:

- GROMACS Manual 2024.4 `mdp-options`: <https://manual.gromacs.org/documentation/2024.4/user-guide/mdp-options.html>
- GROMACS Manual 2024.4 release notes: <https://manual.gromacs.org/2024.4/release-notes/2024/2024.4.html>

## Controlled Reconstruction

TP1.5e reran the same TP1.5b `dense_nonlisted` fixture under:

- `n1_r0909`
- `n10_r0909`

Core physics stayed fixed:

- same topology
- same coordinates
- same box
- same charges
- same `rep-pow = 9`
- same `rcoulomb = 0.9`, `rvdw = 0.9`
- same `rlist = 0.909`
- same integrator family and seed

Only `nstlist` changed from `1` to `10`. The trace window was widened to steps `160-180`.

## Rebuild History

The machine-readable histories are:

- [rebuild_history_n1.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5e_pairlist_contract_audit/rebuild_history_n1.csv)
- [rebuild_history_n10.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5e_pairlist_contract_audit/rebuild_history_n10.csv)

The critical `n10_r0909` facts are:

- last rebuild before the TP1.5d divergence: step `170`
- next rebuild after that: step `180`
- dynamic pruning: disabled throughout
- pair `(1,4, shift 21)` distance at step `170`: `0.9095347524 nm`
- manual outer-list radius at step `170`: `0.9089999795 nm`
- shortfall at rebuild: `0.0005347729 nm`
- first step below `rlist`: `171`
- first step below the actual cutoff `0.9 nm`: `172`

So the pair is omitted on steps `171-179` because it was not eligible for inclusion at the last rebuild, not because a later pruning branch removed it.

## Margin Analysis

The compact comparison is in [pair_1_4_margin_analysis.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5e_pairlist_contract_audit/pair_1_4_margin_analysis.csv).

Important rows:

- `n10_r0909`, step `170`
  - distance above cutoff: `+0.0095347524 nm`
  - distance above manual `rlist`: `+0.0005347729 nm`
  - result: pair correctly absent from the rebuilt list
- `n10_r0909`, step `171`
  - distance above cutoff: `+0.0038842320 nm`
  - distance below `rlist`: `-0.0051157475 nm`
  - result: pair would qualify for a fresh outer list, but no rebuild occurs
- `n10_r0909`, step `172`
  - distance below cutoff: `-0.0015858054 nm`
  - result: actual missing interaction starts while the reused list still lacks the pair
- `auto_buffer_n10_vbt0005`, reference at step `170`
  - `rlist ≈ 0.911 nm`
  - same pair distance is now `0.0014652476 nm` inside the list
  - TP1.5b already showed this auto-buffered setup removes the worsening

## Compliance Decision

The strongest supported classification is `ALLOWED-UNSAFE`.

Why:

- the pair is outside the manually requested list radius at the last rebuild
- dynamic pruning is not active, so there is no inner/outer refresh contract to rescue it
- the automatic buffer path chooses a larger `rlist` and eliminates the issue on the same fixture family

What this does **not** prove:

- that the larger TP1.3 failure is fully explained by this manual-buffer regime
- that manual `rlist` is always a bad choice
- that a source-level bug does not exist elsewhere

It only proves that this specific omission is consistent with the current manual pairlist contract.

## Patching Decision

Minimal production patching is not justified now.

Reason:

- TP1.5e found no direct contract violation
- the omission is explained by manual `rlist` reuse semantics on this fixture
- patching pairlist code before proving a stronger intended contract would risk solving the wrong problem

## Next Step

For TP1.6, the next rational split is:

1. If the validation goal is physics safety, move dense cut-off checks to auto-buffered settings.
2. If manual `rlist` mode must remain a supported target regime, first prove that the intended contract should be stronger than the current one before changing code.
