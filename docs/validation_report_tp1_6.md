# Validation Report — TP1.6 Safe Validation Regime Re-Baselining

## Verdict

- milestone: `TP1.6`
- source patching now justified: `NO`
- safe baseline acceptable for later validation: `PARTIAL`
- overall verdict: `PASS`

## Outcome

TP1.6 reran the TP1.5b `dense_nonlisted` fixture under one unsafe reference and three safer pairlist/buffer regimes. The unsafe reference, `n10_r0909`, reproduced the same widened total-energy range seen in TP1.5b: `12.576325 kJ/mol`. All three safer regimes removed that worsening on the same fixture:

- `n1_r0909`: `8.657745 kJ/mol`
- `n10_r0911`: `8.657745 kJ/mol`
- `auto_buffer_n10_vbt0005`: `8.657745 kJ/mol`

That means TP1.6 found no surviving short-range implementation signal on this fixture once the allowed-unsafe reuse regime was removed. The strongest supported interpretation is still local: the prior worsening on `dense_nonlisted` was unsafe-regime behavior, not a confirmed remaining short-range code defect.

## Boundaries

- confirmed:
  - the prior unsafe case was actually rerun
  - at least one safe regime was actually rerun
  - the safe candidates all return the total-energy range to the TP1.5b tight reference on the same fixture
  - the preferred safe candidate, `auto_buffer_n10_vbt0005`, uses the same `plain-C-4x4` short-range runtime family, with a larger safe buffer: `updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm`
- not confirmed:
  - global short-range correctness on larger charged systems
  - that no other short-range issue exists outside this fixture family

## Key Artifacts

- candidates: [safe_regime_candidates.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_6_rebaseline_audit/safe_regime_candidates.json)
- comparison table: [unsafe_vs_safe_comparison.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_6_rebaseline_audit/unsafe_vs_safe_comparison.csv)
- summary: [rebaseline_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_6_rebaseline_audit/rebaseline_summary.json)
- recommendation: [tp1_6_recommendation.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_6_rebaseline_audit/tp1_6_recommendation.json)
- raw commands: [raw_commands.txt](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_6_rebaseline_audit/raw_commands.txt)

## Reporting

- files changed
  - [tools/run_tp1_6_rebaseline_audit/run_rebaseline_audit.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_6_rebaseline_audit/run_rebaseline_audit.py)
  - [tools/run_tp1_6_rebaseline_audit/README.md](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_6_rebaseline_audit/README.md)
  - [docs/validation_report_tp1_6.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/validation_report_tp1_6.md)
  - [docs/tp1_6_safe_regime_rebaseline.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/tp1_6_safe_regime_rebaseline.md)
  - [tests/reference_results/tp1_6_rebaseline_audit](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_6_rebaseline_audit)
- commands run
  - `git status --short`
  - `git rev-parse HEAD`
  - `build/bin/gmx --version`
  - multiple `sed -n ...` and `rg -n ...` inspections over TP1.5b/TP1.5c/TP1.5d/TP1.5e evidence and existing dense runner code
  - `python3 -m py_compile tools/run_tp1_6_rebaseline_audit/run_rebaseline_audit.py`
  - `python3 tools/run_tp1_6_rebaseline_audit/run_rebaseline_audit.py`
  - exact `gmx grompp`, `gmx mdrun`, `gmx energy`, `gmx dump` commands: [raw_commands.txt](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_6_rebaseline_audit/raw_commands.txt)
- fixtures executed
  - TP1.5b `dense_nonlisted` under `tight_ref_n1_r1200`
  - TP1.5b `dense_nonlisted` under `n10_r0909`
  - TP1.5b `dense_nonlisted` under `n1_r0909`
  - TP1.5b `dense_nonlisted` under `n10_r0911`
  - TP1.5b `dense_nonlisted` under `auto_buffer_n10_vbt0005`
- strongest confirmed finding
  - on the same dense fixture, the unsafe `n10_r0909` widening disappears completely once the pairlist regime is made safe; `n10_r0911` and `auto_buffer_n10_vbt0005` both recover the tight-reference total-energy range exactly
- strongest unresolved uncertainty
  - TP1.6 only re-baselined this toy dense fixture family; it does not yet show that larger charged-system validation is fully safe under the same regime
- exact next step recommendation
  - use `auto_buffer_n10_vbt0005` semantics as the preferred short-range validation baseline for later toy and pre-authoritative charged-system checks, but do not treat that as global closure until a denser validation tier is rerun under the same safe regime
- verdict
  - `PASS`
