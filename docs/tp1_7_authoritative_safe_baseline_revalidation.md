# TP1.7 — Authoritative Safe-Baseline Revalidation

## Scope

TP1.7 only revalidates the authoritative charged-system tier under safer short-range settings. It does not patch production code, does not start r-RESPA work, and does not claim TP1.4 is solved.

## Constraining Prior Evidence

- TP1.3 showed `dense_salt_polymer` runaway on the authoritative tier.
- TP1.4 reproduced a real 9-6 LJ-PME split inconsistency, but only at `PARTIAL` linkage to TP1.3.
- TP1.5e classified the dense toy pair omission as `ALLOWED-UNSAFE`, not a code bug.
- TP1.6 showed that the toy dense fixture stopped widening once pairlist/buffer settings became safe.

What TP1.7 had to decide was narrower:

1. does the authoritative charged system actually use the same unsafe short-range regime that TP1.5e found on the toy fixture?
2. if not, does enforcing a safer short-range regime still materially weaken the charged-system runaway?

## Exact Authoritative Reference

System identity:

- source system: [dense_salt_polymer](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/m10_4_charged_ensemble_gate/dense_salt_polymer)
- TP1.3 reference artifact: [TRL-0](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_3_stabilization/TRL-0)

Important correction:

- nominal TP1.3 metadata labeled `TRL-0` as `NPT`
- actual executed TP1.3 artifact did **not** run with active temperature or pressure coupling
- basis:
  - [trial.mdp](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_3_stabilization/TRL-0/trial.mdp) uses misspelled `tcouple` / `pcouple`
  - [mdout.mdp](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_3_stabilization/TRL-0/mdout.mdp) records `tcoupl = No`, `pcoupl = No`

This matters because TP1.7 had to rerun the **actual executed baseline**, not the intended label.

## Comparison Matrix

Historical reference:

- `historical_tp1_3_reference`
- source: `TRL-0/trial.edr`
- runtime line: `updated every 10 steps, buffer 0.000 nm, rlist 0.900 nm`
- classification: trusted prior reference, not a confirmed manual-unsafe artifact

Executed safe runs:

1. `safe_auto_n10_vbt0005`
   - `nstlist = 10`
   - `verlet-buffer-tolerance = 0.005`
   - intended role: TP1.6 preferred safe auto-buffer baseline
2. `manual_safe_n10_r0911`
   - `nstlist = 10`
   - `rlist = 0.911`
   - `verlet-buffer-tolerance = -1`
   - intended role: widened manual-safe margin from TP1.6

All runs held fixed:

- topology
- start coordinates
- box
- PME electrostatics
- repulsion power 9
- timestep `1 fs`
- starting velocity seed `-1989880213`

## Direct Results

From [unsafe_vs_safe_authoritative_comparison.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7_authoritative_revalidation/unsafe_vs_safe_authoritative_comparison.csv):

| run | runtime pairlist line | runaway onset (ps) | max T (K) | final T (K) | total-energy range (kJ/mol) | effect |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| historical TP1.3 | `buffer 0.000, rlist 0.900` | `1.0` | `829.866` | `737.177` | `72.156` | reference |
| safe auto n10 | `buffer 0.000, rlist 0.900` | `1.0` | `820.864` | `732.787` | `71.877` | persists |
| manual safe n10 | `buffer 0.011, rlist 0.911` | `1.0` | `801.986` | `762.275` | `8.535` | persists |

## Interpretation

### What TP1.7 confirms

- The authoritative TP1.3 tier was not simply the same manual-unsafe regime from TP1.5e.
- The preferred TP1.6 auto-buffer label did not actually change the runtime pairlist line on this larger charged system.
- A genuinely wider short-range margin can improve total-energy stability sharply.
- Even with that wider margin, runaway still starts immediately and remains severe.

### What TP1.7 weakens

- the idea that the authoritative runaway is mainly the same short-range pairlist omission seen on the toy fixture
- the idea that pairlist-code patching is the next justified move

### What remains unresolved

- whether the surviving blocker is mainly the TP1.4 PME/SixthPower inconsistency
- or another mixed long-range interaction problem

The evidence is enough to say the blocker is now more strongly long-range / mixed than short-range pairlist-specific. It is **not** enough to call TP1.4 proven dominant.

## Recommendation Boundary

- Source patching remains unjustified.
- A plain “TP1.6 preferred safe baseline” label is not sufficient by itself for larger systems, because on this authoritative tier it did not actually alter the runtime pairlist line.
- Future authoritative validation should use a runtime-verified safe short-range regime, then focus diagnosis on PME / long-range or mixed causes.

## Artifacts

- matrix: [run_matrix.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7_authoritative_revalidation/run_matrix.json)
- comparison: [unsafe_vs_safe_authoritative_comparison.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7_authoritative_revalidation/unsafe_vs_safe_authoritative_comparison.csv)
- summary: [stability_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7_authoritative_revalidation/stability_summary.json)
- recommendation: [tp1_7_recommendation.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7_authoritative_revalidation/tp1_7_recommendation.json)
- provenance: [provenance_manifest.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_7_authoritative_revalidation/provenance_manifest.json)
