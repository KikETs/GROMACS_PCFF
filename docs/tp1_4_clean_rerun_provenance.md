# TP1.4-RR — Clean Rerun Provenance Notes

## Scope

This document does not broaden TP1.4 scientifically. It records what was rerun, how raw evidence was preserved, and why the result is still bounded as a dirty-tree rerun.

## Cleanliness Assessment

Facts:
- `git rev-parse HEAD` recorded commit `3c0b91be8eba68d22d6404cf70e4439fbb1dd0d6`
- `build/bin/gmx --version` reports `3c0b91be8e... (dirty)`
- `git status --short` shows uncommitted changes, including:
  - `src/gromacs/nbnxm/nbnxm_setup.cpp`
  - `src/gromacs/nbnxm/kerneldispatch.cpp`

Interpretation:
- a truly pristine TP1.4 rerun is not possible in the current working tree
- TP1.4-RR therefore records a **current rerun on a dirty tree**, not a historical or pristine proof

## Reproduced Fixture

Identity:
- 2 atoms in a `5 x 5 x 5 nm` periodic box
- atom types `A` and `B`
- fixed separation `0.5 nm`
- `comb-rule = 4`
- `rep-pow = 9`
- `vdwtype = PME`
- `lj-pme-comb-rule = geometric`
- scan values `rcut = 0.7, 0.8, 0.9, 1.0, 1.1 nm`

Tracked fixture inputs:
- `tests/reference_results/tp1_4_pme_proof/rerun_inputs/system.top`
- `tests/reference_results/tp1_4_pme_proof/rerun_inputs/system.gro`
- `tests/reference_results/tp1_4_pme_proof/rerun_inputs/test_rcut_*.mdp`

## Raw Evidence Preserved

Tracked raw outputs:
- `tests/reference_results/tp1_4_pme_proof/raw_grompp.log`
- `tests/reference_results/tp1_4_pme_proof/raw_mdrun_rerun.log`
- `tests/reference_results/tp1_4_pme_proof/raw_energy_output.txt`
- `tests/reference_results/tp1_4_pme_proof/raw_force_dump.txt`
- `tests/reference_results/tp1_4_pme_proof/commands_run.txt`

Tracked derived outputs:
- `tests/reference_results/tp1_4_pme_proof/pme_energy_force_scan.csv`
- `tests/reference_results/tp1_4_pme_proof/pme_continuity_summary.json`
- `tests/reference_results/tp1_4_pme_proof/tp1_4_suspicion_update.json`
- `tests/reference_results/tp1_4_pme_proof/tp1_4_provenance_manifest.json`

Archived prior non-pristine outputs:
- `tests/reference_results/tp1_4_pme_proof/prior_nonpristine/`

## What Improved Relative To Earlier TP1.4 Evidence

Before TP1.4-RR:
- derived scan outputs existed
- raw force provenance in tracked reference-results paths did not exist
- exact commands and rerun inputs were not preserved in tracked reference-results paths
- current report did not cleanly separate rerun evidence from earlier non-pristine outputs

After TP1.4-RR:
- raw and derived evidence both exist under tracked reference-results paths
- the force extraction source is preserved explicitly via `gmx dump` output
- the rerun is tied to commit/build metadata
- prior outputs are archived separately instead of silently overwritten

## Remaining Weakness

The main bottleneck is not missing logs anymore. It is build/source cleanliness.

Because the current rerun depends on a dirty working tree and a dirty binary, TP1.4-RR cannot honestly claim PASS-level provenance. Any future PASS-level audit still requires the same fixture and artifact set to be rerun from a clean commit and clean rebuild.

## Conclusion

TP1.4-RR hardened provenance enough to close the earlier audit gap around missing raw execution evidence. It did **not** close the clean-tree provenance gap. The correct bounded conclusion is:

- TP1.4 defect reproduction: confirmed again
- TP1.4 raw provenance: materially improved
- TP1.4 PASS readiness: still `PARTIAL`
