# Validation Report — TP1.7b Same-Build Unsafe-vs-Safe Authoritative A/B Revalidation

## Verdict

- milestone: `TP1.7b`
- source patching now justified: `NO`
- plain safe baseline acceptable for future validation: `PARTIAL`
- overall verdict: `PASS`

## Outcome

TP1.7b created the same-build authoritative A/B that TP1.7 lacked.

Two current-build runs were executed on the same `dense_salt_polymer` charged system with all non-target physics held fixed:

- unsafe reference: `nstlist = 10`, `rlist = 0.909`, `verlet-buffer-tolerance = -1`
- safe reference: `nstlist = 10`, `rlist = 0.911`, `verlet-buffer-tolerance = -1`

The safe run is genuinely runtime-distinct:

- unsafe: `updated every 10 steps, buffer 0.009 nm, rlist 0.909 nm`
- safe: `updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm`

But the same-build A/B result is still conservative:

- unsafe remains `RUNAWAY`, onset `1.0 ps`, `max_temperature_k = 814.205`
- safe remains `RUNAWAY`, onset `1.0 ps`, `max_temperature_k = 801.986`
- safe improves total-energy range from `12.607422` to `8.535156 kJ/mol`
- that is a real weakening in one short-range-sensitive metric, but not a material removal of authoritative runaway

So TP1.7b strengthens the interpretation that the remaining blocker is more strongly long-range / PME-related or mixed than short-range pairlist-specific. It does **not** prove PME dominance.

## Boundaries

- confirmed:
  - a same-build unsafe authoritative reference was actually rerun
  - a same-build safe authoritative run was actually rerun
  - the safe run is runtime-distinct from the unsafe run
  - the shared runtime family remains `repulsion power 9` plus `plain-C-4x4`
  - safer short-range settings reduce total-energy range and max temperature modestly, but do not delay runaway onset
- not confirmed:
  - PME dominance
  - global short-range correctness
  - that short-range production code patching is warranted

## Key Artifacts

- run matrix: [run_matrix.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/run_matrix.json)
- runtime-distinct check: [runtime_distinct_check.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/runtime_distinct_check.json)
- comparison table: [unsafe_vs_safe_authoritative_comparison.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/unsafe_vs_safe_authoritative_comparison.csv)
- summary: [stability_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/stability_summary.json)
- recommendation: [tp1_7b_recommendation.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/tp1_7b_recommendation.json)
- raw commands: [raw_commands.txt](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/raw_commands.txt)

## Reporting

- files changed
  - [run_authoritative_ab.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_7b_authoritative_ab/run_authoritative_ab.py)
  - [README.md](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_7b_authoritative_ab/README.md)
  - [validation_report_tp1_7b.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/validation_report_tp1_7b.md)
  - [tp1_7b_same_build_authoritative_ab.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/tp1_7b_same_build_authoritative_ab.md)
  - [tp1_7b_authoritative_ab](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab)
- commands run
  - `git status --short`
  - `build/bin/gmx --version`
  - multiple `sed -n ...` and `rg -n ...` inspections over TP1.3 / TP1.6 / TP1.7 evidence and authoritative runtime artifacts
  - `python3 -m py_compile tools/run_tp1_7b_authoritative_ab/run_authoritative_ab.py`
  - `python3 tools/run_tp1_7b_authoritative_ab/run_authoritative_ab.py`
  - exact `gmx grompp`, `gmx mdrun`, and `gmx energy` commands: [raw_commands.txt](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/raw_commands.txt)
- fixtures executed
  - `dense_salt_polymer` under `unsafe_n10_r0909`
  - `dense_salt_polymer` under `safe_n10_r0911`
- strongest confirmed finding
  - same-build short-range safety changes are real and runtime-distinct on the authoritative system, but they do not materially weaken the early runaway
- strongest unresolved uncertainty
  - whether the surviving blocker is specifically the TP1.4 LJ-PME split defect or a broader long-range / mixed nonbonded issue
- exact next step recommendation
  - keep short-range code unchanged and isolate the surviving PME / long-range or mixed blocker on this same-build authoritative tier
- verdict
  - `PASS`
