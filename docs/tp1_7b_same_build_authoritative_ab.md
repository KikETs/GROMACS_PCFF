# TP1.7b — Same-Build Authoritative Unsafe-vs-Safe A/B

## Scope

TP1.7b does one thing only: it replaces TP1.7’s cross-build / non-unsafe reference weakness with a same-build authoritative A/B comparison.

This milestone does not:

- start r-RESPA work
- start transport calculations
- patch production code
- claim TP1.4 is proven dominant

## Constraining Evidence

- TP1.6 showed on the dense toy fixture that `n10_r0909` was unsafe and `n10_r0911` was safe.
- TP1.7 showed that the authoritative `auto_buffer_n10_vbt0005` run was not runtime-distinct from the historical TP1.3 artifact.
- TP1.7 hostile audit therefore required a same-build unsafe-vs-safe authoritative A/B.

## Comparison Design

Authoritative system:

- [dense_salt_polymer](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/m10_4_charged_ensemble_gate/dense_salt_polymer)

Baseline held fixed from the actually executed TP1.3 style run:

- start structure: [min.gro](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_3_stabilization/TRL-0/min.gro)
- topology: [system.top](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_3_stabilization/TRL-0/system.top)
- `dt = 0.001`
- `nsteps = 500000`
- `coulombtype = PME`
- `rcoulomb = 0.9`
- `vdw-type = Cut-off`
- `rvdw = 0.9`
- `pme-order = 4`
- `fourierspacing = 0.12`
- `tcoupl = no`
- `pcoupl = no`
- `gen_seed = -1989880213`

Unsafe vs safe axis:

1. unsafe authoritative reference
   - `nstlist = 10`
   - `rlist = 0.909`
   - `verlet-buffer-tolerance = -1`
2. safe authoritative reference
   - `nstlist = 10`
   - `rlist = 0.911`
   - `verlet-buffer-tolerance = -1`

Important honesty note:

- this unsafe reference is **not** a historical TP1.3 artifact claim
- it is a same-build authoritative transplant of the only directly demonstrated unsafe short-range regime from TP1.5e / TP1.6

## Runtime-Distinct Verification

From [runtime_distinct_check.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/runtime_distinct_check.json):

- unsafe runtime line: `updated every 10 steps, buffer 0.009 nm, rlist 0.909 nm`
- safe runtime line: `updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm`
- both match expected
- both keep the same kernel family:
  - `Detected LJ repulsion power 9.`
  - `Using plain-C-4x4 4x4 nonbonded short-range kernels`

So the A/B comparison is genuinely same-build and runtime-distinct.

## Direct Results

From [unsafe_vs_safe_authoritative_comparison.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/unsafe_vs_safe_authoritative_comparison.csv):

| run | runtime pairlist line | onset (ps) | max T (K) | final T (K) | total-energy range (kJ/mol) | max abs total drift (kJ/mol) | effect |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| unsafe | `buffer 0.009, rlist 0.909` | `1.0` | `814.205` | `683.261` | `12.607` | `6.867` | reference |
| safe | `buffer 0.011, rlist 0.911` | `1.0` | `801.986` | `762.275` | `8.535` | `5.793` | persists |

Pressure signal:

- unsafe `max_abs_pressure_bar = 4869.487`
- safe `max_abs_pressure_bar = 4817.828`

## Interpretation

### What TP1.7b proves

- The safe authoritative run is genuinely runtime-distinct from the unsafe authoritative run.
- Moving from the unsafe to the safe short-range regime reduces:
  - total-energy range
  - max absolute total-energy drift
  - max temperature slightly
- But the safe run still enters runaway immediately at `1 ps`.

### What TP1.7b does not prove

- PME dominance
- TP1.4 dominance
- global short-range correctness
- that short-range code patching should begin

## Recommendation Boundary

The same-build A/B result is stronger than TP1.7, but still conservative:

- short-range safety helps, but not materially enough to remove authoritative runaway
- the remaining blocker is more strongly long-range / mixed than short-range pairlist-specific
- short-range production code patching remains unjustified

Next step:

- keep the same-build authoritative tier
- keep the safe manual short-range regime
- isolate the surviving PME / long-range or mixed nonbonded blocker

## Artifacts

- matrix: [run_matrix.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/run_matrix.json)
- runtime-distinct check: [runtime_distinct_check.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/runtime_distinct_check.json)
- comparison: [unsafe_vs_safe_authoritative_comparison.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/unsafe_vs_safe_authoritative_comparison.csv)
- summary: [stability_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/stability_summary.json)
- recommendation: [tp1_7b_recommendation.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/tp1_7b_recommendation.json)
- provenance: [provenance_manifest.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7b_authoritative_ab/provenance_manifest.json)
