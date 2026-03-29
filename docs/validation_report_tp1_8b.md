# Validation Report — TP1.8b Narrow Coulomb Long-Range Separation

## Verdict

- milestone: `TP1.8b`
- source patching now justified: `NO`
- plain safe baseline acceptable for later non-rRESPA validation: `PARTIAL`
- overall verdict: `PASS`

## Outcome

TP1.8b reused the authoritative safe short-range baseline from TP1.7b/TP1.8 and kept these settings fixed across all comparisons:

- `nstlist = 10`
- `rlist = 0.911`
- `verlet-buffer-tolerance = -1`
- `vdw-type = Cut-off`

The active-path verification is the first hard result:

- the authoritative safe baseline uses `coulombtype = PME`
- the authoritative safe baseline keeps `vdw-type = Cut-off`
- TP1.4's LJ-PME/SixthPower path therefore remains **inactive**

TP1.8b then ran three narrower Coulomb variants than TP1.8's coarse `coulombtype = Cut-off` comparison:

1. tighter PME mesh and reciprocal accuracy
2. `coulombtype = Ewald` with the same short-range baseline
3. `coulomb-modifier = None` with `coulombtype = PME` unchanged

The conservative result is:

- all three variants remain `RUNAWAY`
- all three variants still cross the runaway threshold at `0.2 ps`
- none of the narrowed Coulomb changes materially weakens the early authoritative runaway

That does **not** support PME dominance. It also does **not** support a clean direct-space-only explanation. The remaining blocker stays `mixed_or_still_unresolved`.

## Narrowed Runtime Verification

The TP1.8b reference raw files show:

- `raw_safe_pme_shift_ref_mdout.mdp`
  - `coulombtype = PME`
  - `coulomb-modifier = Potential-shift-Verlet`
  - `vdw-type = Cut-off`
- `raw_safe_pme_shift_ref_md.log`
  - `Using plain-C-4x4 4x4 nonbonded short-range kernels`
  - `updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm`
  - `Solve PME ...`

The key runtime caveat is important:

- the log banner `Will do PME sum in reciprocal space for electrostatic interactions.` is **generic** for `usingPmeOrEwald`
- it is printed from `src/gromacs/mdtypes/interaction_const.cpp:initCoulombEwaldParameters`
- so it cannot by itself distinguish PME mesh from full Ewald

For TP1.8b, the Ewald variant is treated as runtime-distinct because:

- `raw_safe_ewald_shift_mdout.mdp` uses `coulombtype = Ewald`
- `raw_safe_ewald_shift_md.log` has no `Solve PME` timing line
- source path inspection shows the long-range CPU path switches from `computePmeOnCpu` to `do_ewald`

Relevant source basis:

- `src/gromacs/ewald/pme.cpp:gmx_pme_init`
  - Coulomb PME is gated by `usingPme(ir->coulombtype)`
- `src/gromacs/mdlib/force.cpp:CpuPpLongRangeNonbondeds::calculate`
  - PME uses `gmx_pme_do`
  - Ewald uses `do_ewald`
- `src/gromacs/mdlib/forcerec.cpp:init_forcerec`
  - PME and Ewald both stay in the Ewald-family direct-space electrostatics
- `src/gromacs/nbnxm/kerneldispatch.cpp:getCoulombKernelType`
  - `Cut-off` would switch to a different direct-space Coulomb family, which is why TP1.8's cut-off variant was too mixed

## Comparison

| run | intended Coulomb change | runtime mode | onset (ps) | max T (K) | total-energy range (kJ/mol) | max abs pressure (bar) | effect |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `safe_pme_shift_ref` | reference | `pme_coulomb` | `0.2` | `758.727` | `5.621` | `4493.854` | reference |
| `safe_pme_tight_mesh` | reciprocal accuracy only | `pme_coulomb` | `0.2` | `770.215` | `6.223` | `3954.032` | persists |
| `safe_ewald_shift` | reciprocal solver change, narrower than cutoff | `ewald_no_pme_mesh` | `0.2` | `788.438` | `5.012` | `4435.358` | persists |
| `safe_pme_none` | direct-space Coulomb modifier only | `pme_coulomb` | `0.2` | `779.377` | `5.361` | `3333.198` | persists |

Artifacts:

- `tests/reference_results/tp1_8b_coulomb_separation/active_coulomb_path_map.json`
- `tests/reference_results/tp1_8b_coulomb_separation/run_matrix.json`
- `tests/reference_results/tp1_8b_coulomb_separation/runtime_distinct_check.json`
- `tests/reference_results/tp1_8b_coulomb_separation/coulomb_variant_comparison.csv`
- `tests/reference_results/tp1_8b_coulomb_separation/stability_summary.json`
- `tests/reference_results/tp1_8b_coulomb_separation/tp1_8b_recommendation.json`

## Interpretation Boundary

What TP1.8b supports:

- TP1.4's LJ-PME path is inactive in the authoritative setup
- a narrower reciprocal-solver change from PME mesh to Ewald is feasible and was actually executed
- a narrower direct-space modifier change from `Potential-shift-Verlet` to `None` is feasible and was actually executed
- none of these narrower Coulomb changes materially weakens the authoritative runaway

What TP1.8b does **not** support:

- PME dominance
- direct-space Coulomb dominance
- global short-range correctness
- production source patching

The evidence is too mixed for a single-path blame assignment:

- tighter PME accuracy slightly changes metrics but does not rescue onset
- Ewald versus PME mesh slightly changes metrics but does not rescue onset
- direct-space modifier change also fails to rescue onset

So the surviving blocker stays `mixed_or_still_unresolved`.

## Reporting

- files changed
  - `tools/run_tp1_8b_coulomb_separation/run_coulomb_separation.py`
  - `tools/run_tp1_8b_coulomb_separation/README.md`
  - `docs/validation_report_tp1_8b.md`
  - `docs/tp1_8b_narrow_coulomb_separation.md`
  - `tests/reference_results/tp1_8b_coulomb_separation/`
- commands run
  - `git status --short`
  - `build/bin/gmx --version`
  - multiple `sed -n ...` / `rg -n ...` inspections over TP1.7b / TP1.8 evidence, authoritative safe-baseline logs, and Coulomb-path source locations
  - `python3 -m py_compile tools/run_tp1_8b_coulomb_separation/run_coulomb_separation.py`
  - `python3 tools/run_tp1_8b_coulomb_separation/run_coulomb_separation.py`
  - exact `gmx grompp`, `gmx mdrun`, and `gmx energy` commands: `tests/reference_results/tp1_8b_coulomb_separation/raw_commands.txt`
- fixtures executed
  - authoritative `dense_salt_polymer` under `safe_pme_shift_ref`
  - authoritative `dense_salt_polymer` under `safe_pme_tight_mesh`
  - authoritative `dense_salt_polymer` under `safe_ewald_shift`
  - authoritative `dense_salt_polymer` under `safe_pme_none`
- strongest confirmed finding
  - the authoritative setup keeps TP1.4's LJ-PME path inactive, and neither a narrower reciprocal-solver change nor a narrower direct-space Coulomb modifier change materially weakens the early runaway
- strongest unresolved uncertainty
  - which narrower Coulomb real/reciprocal split mechanism still drives the surviving instability signal, if any, versus a broader mixed electrostatics problem
- exact next step recommendation
  - keep the same authoritative safe baseline and add source-level Coulomb tracing around PME-versus-Ewald long-range accumulation before any production patching
- verdict
  - `PASS`
