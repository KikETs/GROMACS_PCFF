# Validation Report — TP1.7 Authoritative Charged-System Revalidation Under Safe Short-Range Baseline

## Verdict

- milestone: `TP1.7`
- source patching now justified: `NO`
- plain safe baseline acceptable for future validation: `PARTIAL`
- overall verdict: `PASS`

## Outcome

TP1.7 reran the authoritative charged system `dense_salt_polymer` against the trusted TP1.3 `TRL-0` reference and two explicit safe-regime candidates.

The main result is not what the milestone premise suggested:

- the authoritative TP1.3 reference was **not** a confirmed manual-unsafe pairlist regime
- the preferred TP1.6-style auto-buffer rerun, `safe_auto_n10_vbt0005`, reproduced the same runtime short-range line as TP1.3:
  - `updated every 10 steps, buffer 0.000 nm, rlist 0.900 nm`
- a stronger manual-safe margin, `manual_safe_n10_r0911`, did change the runtime line to:
  - `updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm`
  - and reduced total-energy range from `72.15625` to `8.535156 kJ/mol`
  - but runaway still started at `1 ps`, and `max_temperature_k` remained `801.986 K`

That means short-range tightening helps one stability metric, but it does not remove the authoritative runaway. The remaining blocker is now more strongly long-range / PME-related or mixed, not a clear surviving pairlist-code defect.

## Boundary Findings

- confirmed:
  - TP1.3 `TRL-0` historical reference was re-extracted from `trial.edr`, not accepted on prose alone
  - the authoritative TP1.3 reference runtime line is `updated every 10 steps, buffer 0.000 nm, rlist 0.900 nm`
  - the preferred TP1.6 auto-buffer candidate does **not** change that runtime line on this system
  - the widened manual-safe candidate `rlist = 0.911` does change the runtime short-range regime and sharply reduces total-energy range
  - both safe reruns still remain `RUNAWAY`
- not confirmed:
  - that authoritative charged-system instability is now dominantly PME-only
  - global short-range correctness
  - that a production short-range code patch is warranted

## Key Artifacts

- run matrix: [run_matrix.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7_authoritative_revalidation/run_matrix.json)
- comparison table: [unsafe_vs_safe_authoritative_comparison.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7_authoritative_revalidation/unsafe_vs_safe_authoritative_comparison.csv)
- stability summary: [stability_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7_authoritative_revalidation/stability_summary.json)
- recommendation: [tp1_7_recommendation.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7_authoritative_revalidation/tp1_7_recommendation.json)
- raw commands: [raw_commands.txt](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7_authoritative_revalidation/raw_commands.txt)

## Reporting

- files changed
  - [tools/run_tp1_7_authoritative_revalidation/run_authoritative_revalidation.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_7_authoritative_revalidation/run_authoritative_revalidation.py)
  - [tools/run_tp1_7_authoritative_revalidation/README.md](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_7_authoritative_revalidation/README.md)
  - [docs/validation_report_tp1_7.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/validation_report_tp1_7.md)
  - [docs/tp1_7_authoritative_safe_baseline_revalidation.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/tp1_7_authoritative_safe_baseline_revalidation.md)
  - [tests/reference_results/tp1_7_authoritative_revalidation](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7_authoritative_revalidation)
- commands run
  - `git status --short`
  - `sed -n ... AGENTS.md`
  - multiple `sed -n ...` and `rg -n ...` inspections over TP1.3 / TP1.4 / TP1.5e / TP1.6 artifacts and the authoritative system source
  - `build/bin/gmx --version`
  - `python3 -m py_compile tools/run_tp1_7_authoritative_revalidation/run_authoritative_revalidation.py`
  - `python3 tools/run_tp1_7_authoritative_revalidation/run_authoritative_revalidation.py`
  - exact `gmx energy`, `gmx grompp`, and `gmx mdrun` commands: [raw_commands.txt](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7_authoritative_revalidation/raw_commands.txt)
- systems executed
  - trusted prior authoritative reference extraction from `tests/reference_results/tp1_3_stabilization/TRL-0`
  - `dense_salt_polymer` under `safe_auto_n10_vbt0005`
  - `dense_salt_polymer` under `manual_safe_n10_r0911`
- strongest confirmed finding
  - on the authoritative system, the plain TP1.6 auto-buffer candidate does not actually change the runtime pairlist regime, while the widened manual-safe margin does improve total-energy behavior but still leaves immediate runaway
- strongest unresolved uncertainty
  - whether the remaining dominant blocker is the TP1.4 PME/SixthPower issue specifically, or a broader long-range / mixed interaction defect
- exact next step recommendation
  - use a runtime-verified safe short-range setting for the next authoritative comparison, then isolate the surviving long-range / PME blocker instead of patching pairlist code
- verdict
  - `PASS`
