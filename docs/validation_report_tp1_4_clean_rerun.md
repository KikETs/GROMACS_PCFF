# Validation Report — TP1.4-RR Clean Rerun And Provenance Hardening

## 1. Executive Summary

**Milestone Result: PARTIAL**

TP1.4-RR reran the existing minimal periodic 9-6 LJ-PME split fixture and preserved raw execution provenance under tracked `tests/reference_results/tp1_4_pme_proof/` paths. The rerun again shows strong `rcut` dependence for a fixed 2-atom geometry, so the original TP1.4 physical inconsistency still reproduces on the current build.

This milestone does **not** establish a pristine clean-tree proof. The repository and built `gmx` binary are both dirty, and uncommitted source changes in the 9-6 LJ-PME startup path remain present. TP1.4 is therefore better evidenced than before, but still not PASS-ready.

## 2. What Was Rerun Now

Current rerun evidence:
- same 2-atom mixed-type periodic fixture used by TP1.4
- same fixed distance: `0.5 nm`
- same topology defaults: `comb-rule = 4`, `rep-pow = 9`
- same LJ-PME settings: `vdwtype = PME`, `lj-pme-comb-rule = geometric`
- same `rcut = rlist = rcoulomb = rvdw` scan: `0.7, 0.8, 0.9, 1.0, 1.1 nm`

New provenance hardening in this rerun:
- tracked raw `grompp` output
- tracked raw `mdrun -rerun` output
- tracked raw `gmx energy` output
- tracked raw `gmx dump` output used for force extraction
- tracked exact command list
- tracked fixture inputs used for the rerun
- tracked manifest with commit, build version, and dirty-tree status

## 3. Current Rerun Evidence Vs Prior Non-Pristine Evidence

Current rerun evidence:
- `tests/reference_results/tp1_4_pme_proof/pme_fixture_definition.json`
- `tests/reference_results/tp1_4_pme_proof/pme_energy_force_scan.csv`
- `tests/reference_results/tp1_4_pme_proof/pme_continuity_summary.json`
- `tests/reference_results/tp1_4_pme_proof/tp1_4_suspicion_update.json`
- `tests/reference_results/tp1_4_pme_proof/tp1_4_provenance_manifest.json`
- `tests/reference_results/tp1_4_pme_proof/raw_grompp.log`
- `tests/reference_results/tp1_4_pme_proof/raw_mdrun_rerun.log`
- `tests/reference_results/tp1_4_pme_proof/raw_energy_output.txt`
- `tests/reference_results/tp1_4_pme_proof/raw_force_dump.txt`
- `tests/reference_results/tp1_4_pme_proof/commands_run.txt`
- `tests/reference_results/tp1_4_pme_proof/rerun_inputs/`

Prior non-pristine evidence archived, not re-labeled as clean:
- `tests/reference_results/tp1_4_pme_proof/prior_nonpristine/`

Those archived files existed before TP1.4-RR regenerated the tracked rerun artifacts. They should not be treated as historical pristine provenance.

## 4. Direct Rerun Finding

The current rerun reproduced the same large split inconsistency:

| `rcut` (nm) | LJ (SR) | LJ recip. | Potential | Force x on atom 2 |
| :--- | :--- | :--- | :--- | :--- |
| 0.7 | 6.943827 | -41.748489 | -34.804665 | -14.40740 |
| 0.8 | 3.124440 | -18.999409 | -15.874969 | -6.63457 |
| 0.9 | 1.544618 | -9.550016 | -8.005398 | -4.29121 |
| 1.0 | 0.819228 | -5.205198 | -4.385970 | -3.14929 |
| 1.1 | 0.456348 | -3.030363 | -2.574015 | -2.38829 |

Derived continuity metrics from the rerun:
- `potential_span = 32.23065 kJ/mol`
- `force_span = 12.01911`
- `relative_span_vs_rcut_1p1 = 12.52`
- `invariance_check = FAILED`

This is still direct evidence that the exercised LJ-PME split is not physically continuous for the fixed TP1.4 mixed-type fixture.

## 5. What Provenance Weaknesses Were Fixed

Fixed in TP1.4-RR:
- raw force extraction source is now preserved, not only parsed into CSV/JSON
- raw energy extraction output is now preserved
- exact commands run are now preserved
- fixture inputs used for the scan are now preserved
- the rerun is tied to explicit commit/build metadata
- prior non-pristine artifacts are separated from the current rerun outputs

Still imperfect:
- the rerun is not on a clean source tree
- the built binary identifies itself as dirty
- source changes in `src/gromacs/nbnxm/nbnxm_setup.cpp` and `src/gromacs/nbnxm/kerneldispatch.cpp` affect the 9-6 LJ-PME startup path

## 6. Verdict

TP1.4 now has stronger reproducibility and raw provenance than before, but it still does **not** support PASS-level audit.

Conservative status:
- current rerun evidence supports: defect reproduced and plausibly large enough to matter
- current rerun evidence does **not** support: pristine clean-tree proof
- TP1.4 PASS readiness: **still PARTIAL**

## 7. Reporting

- files changed
  - `docs/validation_report_tp1_4_clean_rerun.md`
  - `docs/tp1_4_clean_rerun_provenance.md`
  - `tools/run_tp1_4_pme_proof/run_pme_proof.py`
  - `tests/reference_results/tp1_4_pme_proof/*`
- commands run
  - `git status --short`
  - `git rev-parse HEAD`
  - `build/bin/gmx --version`
  - `python3 tools/run_tp1_4_pme_proof/run_pme_proof.py`
- fixtures executed
  - 2-atom mixed-type periodic 9-6 LJ-PME split scan over `rcut = 0.7, 0.8, 0.9, 1.0, 1.1 nm`
- strongest confirmed finding
  - the rerun again shows large monotonic drift in total potential and force for a fixed geometry
- strongest unresolved uncertainty
  - whether the same defect would reproduce identically on a pristine source/build state remains unverified in this repository state
- exact next step recommendation
  - rerun the same TP1.4 fixture from a clean commit and clean rebuild, preserving the same raw provenance artifact set without changing fixture design
- verdict
  - `PARTIAL`
