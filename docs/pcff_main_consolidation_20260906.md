# PCFF canonical main consolidation — 2026-09-06

## Canonical source and validation provenance

Use local `main` for subsequent implementation-paper source work. Its clean working directory is `/home/kiket/Desktop/test/GROMACS_PCFF_main`.

`main` was fast-forwarded from `b525c1daf1` to the validated implementation freeze `f7c9ede84e14b4fb87932ab5c8992067128b1177`, then merged with `lunar-pcff-converter-evidence` (`1e0e40be45a80fc15ef83e9797f5a5f9e678473c`) in merge commit `57f5da4fdcefa0f608523273dbabfbb292b56e65`.

The frozen branch and checkout remain unchanged. Numerical measurements and the 95 historical core-test results retain their original frozen-commit provenance. The merged C++ engine, API, CMake files, and top-level build configuration are byte-identical to that freeze. This consolidation does not establish a new numerical validation result. The known strict large-system GPU checkpoint limitation remains open.

The original `/home/kiket/Desktop/test/GROMACS_PCFF` checkout remains on its previous branch with all dirty files intact. Use the canonical checkout above when making new source changes. Simulation outputs and the paper materials package remain at their existing locations; they were not moved or overwritten. No push was performed.

## Integrated changes

- All 14 implementation commits between old `main` and the freeze, including exact r-RESPA runtime, restart, COM, scalar, analysis, and bridge corrections.
- The committed LUNAR converter, its scripts, claim-boundary document, and historical smoke evidence.
- The pending LUNAR converter correction to use the existing shortest-bond-path 1–4 pair generator, plus a regression test for a ring where explicit dihedral endpoints are not a valid 1–4 pair. Existing frozen bridge safeguards and tests were retained.

## Historical branches retained as references

The following is a content review, not a claim that every reference was merged or deleted. Keeping historical references avoids losing diagnostic snapshots.

| Branch group | Disposition |
|---|---|
| `agent/pcff-respa-bottleneck-optimizations`, `fix/pcff-exact-respa-com-20260905` | Already ancestors of the frozen implementation; included in main. |
| `agent/pcff-respa-checkpoint-boundary-fix` | Unique commit is patch-equivalent to an included fix (`git cherry` reports `-`). |
| `phase1-typing-bundle-salvage`, `runner-doc-salvage-20260405` | Unique patches are already included (`git cherry` reports `-`). |
| `lunar-pcff-converter-evidence` | Merged with history. |
| `paper/pcff-implementation-freeze-20260905` | Included; preserved as the immutable measurement anchor. |
| `backup/pending-code-runtime-20260329`, `backup/tp1-audits-20260329` | Old stash/audit snapshots retained without importing unvalidated snapshot code into the frozen engine. |
| `tmp/merge-sim-1775363578`, `tmp/test-merge-respa-m2`, `tmp/test-merge-runner` | Temporary merge/snapshot history retained; not a production implementation source. |
| `tp1.19g-clean-replay` | Single-frame diagnostic dump instrumentation retained separately; not imported into the validated engine. |

## Uncommitted changes in the original checkout

A binary Git patch, the original refs/status, and 15 tracked/untracked source or documentation files were backed up before integration. Their SHA-256 hashes and the complete tracked diff were checked unchanged afterward.

| Pending area | Decision |
|---|---|
| LUNAR converter pair generation and its new ring test | Integrated into main. |
| Shared fixture bridge and old test-file edits | Kept the frozen versions, which already include shortest-path generation plus newer image-flag, coordinate-precision, and triclinic rejection checks. Copying the old files wholesale would remove these fixes. |
| Transport analyzer, multisystem analyzer, worker | Kept the frozen implementation. Older fixed-ion-type overrides and old runtime settings are not replacements for the validated automatic ion mapping and current GPU runtime. Original changes remain recoverable. |
| `lammps_stage_layout_sweep.py` | Pending segmented-restart copy helper remains in the original checkout; outside this source/evidence consolidation. |
| `run_polygen_same_state_probe.py` | Pending diagnostic options remain in the original checkout; not incorporated into the frozen protocol. |
| `polygen_remote_speed_sweep.py` | Did not adopt the pending change raising box/grid guard defaults to `1e9`. |
| Five untracked analysis/status/repair scripts and the untracked validation note | Backed up and retained in the original checkout; not promoted to validated release tools. |
| Untracked simulation and scratch outputs | Left in place. |

This is one canonical branch for continued work, with archival references and explicitly deferred local experiments preserved. It is not a blind union of every historical working tree.

## Verification performed for consolidation

- 73 focused Python tests passed: fixture bridge, multisystem worker, exact image handoff, and extended-state lineage. This includes the newly integrated LUNAR ring regression.
- All six LUNAR Python source files passed syntax parsing.
- `git diff --check` passed.
- Both the frozen implementation and committed LUNAR branch are ancestors of main.
- C++ engine/API/build sources unchanged relative to the frozen commit.
- Original dirty files and patch preserved; frozen branch unchanged.

No new MD or 6270-row smoke campaign was run. The LUNAR `6270 / 6270` reports are historical preprocessing evidence associated with the pre-consolidation converter, not a rerun of the corrected converter. See [the LUNAR claim boundary](lunar_gromacs_pcff_converter_claim_boundary.md) for fallback, smoke geometry, and physical-validation exclusions.

## Local artifact locations

- Consolidation audit, backups and JUnit XML: `/home/kiket/Desktop/test/GROMACS_PCFF/output/branch_consolidation_20260906/`
- Prior paper materials: `/home/kiket/Desktop/test/GROMACS_PCFF/output/paper_materials_20260906/`
- Frozen measurement validation: `/home/kiket/Desktop/test/GROMACS_PCFF/output/implementation_closeout_20260905/final_validation/`

The existing paper-materials ZIP retains its original contents and checksum. The LUNAR material merged here is an additional historical evidence source and must be cited separately with its original preprocessing-only scope; it is not silently substituted into the frozen numerical tables.
