# Validation Report — TP1.8 Authoritative Long-Range / PME Blocker Isolation

## Verdict

- milestone: `TP1.8`
- source patching now justified: `NO`
- plain safe baseline acceptable for later non-rRESPA validation: `PARTIAL`
- overall verdict: `PASS`

## Outcome

TP1.8 reused the TP1.7b same-build authoritative charged-system setup with the short-range safe baseline fixed at:

- `nstlist = 10`
- `rlist = 0.911`
- `verlet-buffer-tolerance = -1`

The active-path check is the first hard result:

- Coulomb PME is active in the authoritative safe reference.
- LJ-PME is **inactive** in the authoritative safe reference because `vdw-type = Cut-off`.
- Therefore the TP1.4 LJ-PME/SixthPower path is **not** the active reciprocal path in this authoritative setup.

TP1.8 then executed two long-range isolation variants while keeping the short-range safe baseline fixed:

1. tighter Coulomb PME accuracy
2. no-reciprocal Coulomb cut-off variant

The executed 100 ps authoritative-tier comparison gives a conservative result:

- safe PME baseline: `RUNAWAY`, onset `1.0 ps`, `max_temperature_k = 801.986`
- tighter PME accuracy: `RUNAWAY`, onset `1.0 ps`, `max_temperature_k = 795.445`
- no-reciprocal cut-off: `RUNAWAY`, onset `1.0 ps`, `max_temperature_k = 758.498`

But the decisive point is not the small temperature decrease. It is that:

- tightening PME accuracy does **not** materially change the runaway classification
- removing reciprocal-space Coulomb does **not** delay runaway onset either
- and the no-reciprocal cut-off variant greatly worsens total-energy conservation

That combination does **not** support a simple “PME precision bug” story. The remaining blocker stays **mixed or still unresolved**.

## Verified Active Path

From the TP1.8 safe reference raw files:

- [raw_safe_pme_n10_r0911_mdout.mdp](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation/raw_safe_pme_n10_r0911_mdout.mdp)
  - `coulombtype = PME`
  - `vdw-type = Cut-off`
- [raw_safe_pme_n10_r0911_md.log](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation/raw_safe_pme_n10_r0911_md.log)
  - `Will do PME sum in reciprocal space for electrostatic interactions.`
  - `Detected LJ repulsion power 9.`
  - `Using plain-C-4x4 4x4 nonbonded short-range kernels`
  - `updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm`

Source-path interpretation:

- [pme.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/ewald/pme.cpp#L782)
  - `pme->doCoulomb = usingPme(ir->coulombtype)`
- [pme.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/ewald/pme.cpp#L783)
  - `pme->doLJ = usingLJPme(ir->vdwtype)`
- [kerneldispatch.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/kerneldispatch.cpp#L113)
  - PME/Ewald direct-space Coulomb uses the Ewald kernel family, not ReactionField
- [kerneldispatch.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/kerneldispatch.cpp#L164)
  - `vdw-type = Cut-off` keeps the Cut-off VdW kernel family
- [nbnxm_setup.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/nbnxm_setup.cpp#L462)
  - LJ-PME combination rule handling only activates when `vdw-type = Pme`
- [atomdata.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/atomdata.cpp#L400)
  - the PCFF 9-6 LJ-PME grid path is gated by `usingLJPme`

So the active authoritative reciprocal path is Coulomb PME, not TP1.4’s LJ-PME path.

## Long-Range Isolation Variants

Executed comparison matrix:

| run | long-range change only | runtime class | onset (ps) | max T (K) | total-energy range (kJ/mol) | max abs pressure (bar) | effect |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `safe_pme_n10_r0911` | baseline | `pme_coulomb` | `1.0` | `801.986` | `5.793` | `2804.175` | reference |
| `safe_pme_tight_fs006_po6` | `fourierspacing 0.12 -> 0.06`, `pme-order 4 -> 6`, `ewald-rtol 1e-5 -> 1e-6` | `pme_coulomb` | `1.0` | `795.445` | `5.457` | `4112.617` | persists |
| `safe_cutoff_n10_r0911` | `coulombtype PME -> Cut-off` with short-range baseline fixed | `non_pme_coulomb` | `1.0` | `758.498` | `135.209` | `4059.436` | persists |

Artifacts:

- [active_path_map.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation/active_path_map.json)
- [run_matrix.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation/run_matrix.json)
- [longrange_variant_comparison.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation/longrange_variant_comparison.csv)
- [stability_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation/stability_summary.json)
- [tp1_8_recommendation.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation/tp1_8_recommendation.json)

## Interpretation Boundary

What TP1.8 confirms:

- the authoritative safe reference really uses Coulomb PME
- the authoritative safe reference does **not** use LJ-PME
- tightening Coulomb PME accuracy does not materially weaken the early runaway
- removing reciprocal-space Coulomb also does not remove or delay the early runaway

What TP1.8 does **not** confirm:

- PME dominance
- TP1.4 dominance in the authoritative setup
- global short-range correctness
- that production source patching should begin

The no-reciprocal cut-off variant is an isolation tool, not a physically equivalent replacement. Its huge total-energy range increase means it cannot be used as proof that long-range electrostatics are unimportant. It only shows that simply removing reciprocal-space Coulomb does not cleanly rescue the run.

## Reporting

- files changed
  - [run_longrange_isolation.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_8_longrange_isolation/run_longrange_isolation.py)
  - [README.md](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_8_longrange_isolation/README.md)
  - [validation_report_tp1_8.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/validation_report_tp1_8.md)
  - [tp1_8_authoritative_longrange_isolation.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/tp1_8_authoritative_longrange_isolation.md)
  - [tp1_8_longrange_isolation](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation)
- commands run
  - `git status --short`
  - `build/bin/gmx --version`
  - multiple `sed -n ...` and `rg -n ...` inspections over TP1.4 / TP1.7b evidence, authoritative mdout/log files, and active PME source paths
  - `python3 -m py_compile tools/run_tp1_8_longrange_isolation/run_longrange_isolation.py`
  - `python3 tools/run_tp1_8_longrange_isolation/run_longrange_isolation.py`
  - exact `gmx grompp`, `gmx mdrun`, and `gmx energy` commands: [raw_commands.txt](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation/raw_commands.txt)
- fixtures executed
  - authoritative `dense_salt_polymer` under `safe_pme_n10_r0911`
  - authoritative `dense_salt_polymer` under `safe_pme_tight_fs006_po6`
  - authoritative `dense_salt_polymer` under `safe_cutoff_n10_r0911`
- strongest confirmed finding
  - the authoritative safe setup uses Coulomb PME while TP1.4’s LJ-PME path is inactive, and neither tighter PME accuracy nor no-reciprocal cut-off removes the immediate runaway
- strongest unresolved uncertainty
  - whether the surviving blocker is a narrower Coulomb reciprocal/decomposition defect or a broader mixed electrostatics problem
- exact next step recommendation
  - keep the safe short-range baseline and run a narrower Coulomb long-range vs mixed-path audit on the same authoritative tier before any production patching
- verdict
  - `PASS`
