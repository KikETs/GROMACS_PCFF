# TP1.8b Narrow Coulomb Separation Notes

## Scope

TP1.8 already showed two things:

- the authoritative safe setup uses Coulomb PME, not LJ-PME
- the coarse `coulombtype = Cut-off` comparison was too mixed because it also switched the direct-space Coulomb kernel family

TP1.8b narrows that comparison. The goal is not to prove a final root cause. The goal is to see whether the surviving authoritative runaway reacts more strongly to:

- reciprocal-space Coulomb treatment
- direct-space Coulomb treatment
- or neither in a clean enough way to break the tie

## Active Coulomb Path

Reference setup:

- `coulombtype = PME`
- `coulomb-modifier = Potential-shift-Verlet`
- `vdw-type = Cut-off`
- `nstlist = 10`
- `rlist = 0.911`
- `verlet-buffer-tolerance = -1`

Source path map:

1. `src/gromacs/ewald/pme.cpp:gmx_pme_init`
   - physical role: activates Coulomb PME only when `usingPme(ir->coulombtype)`
   - status: active in reference
2. `src/gromacs/mdlib/force.cpp:CpuPpLongRangeNonbondeds::calculate`
   - physical role: chooses `gmx_pme_do` versus `do_ewald`
   - status: active and tunable
3. `src/gromacs/mdlib/forcerec.cpp:init_forcerec`
   - physical role: translates `coulombtype` into the short-range electrostatics family and stores the modifier separately
   - status: active and tunable
4. `src/gromacs/nbnxm/kerneldispatch.cpp:getCoulombKernelType`
   - physical role: selects the direct-space Coulomb kernel family
   - status: active and tunable
   - important: `PME` and `Ewald` both stay in the Ewald-family path, while `Cut-off` changes family
5. `src/gromacs/mdtypes/interaction_const.cpp:initCoulombEwaldParameters`
   - physical role: sets Ewald parameters and prints the reciprocal-space banner
   - status: active and tunable
   - important: the banner `Will do PME sum in reciprocal space for electrostatic interactions.` is printed for `usingPmeOrEwald`, so it is not a solver discriminator
6. `src/gromacs/nbnxm/nbnxm_setup.cpp:chooseLJPmeCombinationRule`
   - physical role: LJ-PME-specific path gate
   - status: inactive in reference because `vdw-type = Cut-off`

## Variant Design

Reference:

- `safe_pme_shift_ref`
- keeps the TP1.7b/TP1.8 authoritative safe short-range baseline unchanged

Narrow reciprocal-accuracy variant:

- `safe_pme_tight_mesh`
- changes only `pme-order 4 -> 6`, `fourierspacing 0.12 -> 0.06`, `ewald-rtol 1e-5 -> 1e-6`
- why narrower than TP1.8 cutoff:
  - `coulombtype` stays `PME`
  - direct-space Coulomb family stays the same
  - short-range baseline stays the same

Narrow reciprocal-solver variant:

- `safe_ewald_shift`
- changes `coulombtype PME -> Ewald`
- keeps `coulomb-modifier = Potential-shift-Verlet`
- why narrower than TP1.8 cutoff:
  - based on `forcerec.cpp` and `kerneldispatch.cpp`, the direct-space electrostatics family remains Ewald-family instead of switching to the `Cut-off`/`ReactionField` family
  - the short-range pairlist baseline stays fixed
- residual ambiguity:
  - `coulombtype` still changes more than the PME mesh controls, so this is narrower than cutoff, not perfectly reciprocal-only

Narrow direct-space modifier variant:

- `safe_pme_none`
- changes `coulomb-modifier Potential-shift-Verlet -> None`
- keeps `coulombtype = PME`
- why narrower:
  - reciprocal PME remains active
  - pairlist baseline stays fixed
  - VdW path stays fixed

## Runtime Verification

What stayed fixed across all runs:

- same `dense_salt_polymer` start structure and topology
- same `nstlist = 10`
- same `rlist = 0.911`
- same `verlet-buffer-tolerance = -1`
- same `vdw-type = Cut-off`
- same `rvdw = 0.9`
- same timestep, seed, and ensemble family

What changed:

- `safe_pme_tight_mesh`
  - `mdout`: `coulombtype = PME`, `coulomb-modifier = Potential-shift-Verlet`, `pme-order = 6`, `fourierspacing = 0.06`
  - runtime: same pairlist line, same `Solve PME` family, tighter Ewald shift term
- `safe_ewald_shift`
  - `mdout`: `coulombtype = Ewald`, `coulomb-modifier = Potential-shift-Verlet`
  - runtime: same pairlist line, no `Solve PME` line
  - interpretation basis: distinctness comes from `mdout` plus source path plus absence of `Solve PME`, not from the generic reciprocal banner
- `safe_pme_none`
  - `mdout`: `coulombtype = PME`, `coulomb-modifier = None`
  - runtime: same pairlist line, same `Solve PME` family, direct-space `Ewald -0.000e+00` shift line

Machine-readable verification:

- `tests/reference_results/tp1_8b_coulomb_separation/runtime_distinct_check.json`

## Observed Results

Short version:

- baseline `safe_pme_shift_ref`: `RUNAWAY`, onset `0.2 ps`
- tighter PME mesh: `RUNAWAY`, onset `0.2 ps`
- Ewald solver: `RUNAWAY`, onset `0.2 ps`
- direct-space modifier None: `RUNAWAY`, onset `0.2 ps`

Detailed comparison:

| run | max T (K) | total-energy range (kJ/mol) | max abs pressure (bar) | effect |
| :--- | :--- | :--- | :--- | :--- |
| `safe_pme_shift_ref` | `758.727` | `5.621` | `4493.854` | reference |
| `safe_pme_tight_mesh` | `770.215` | `6.223` | `3954.032` | persists |
| `safe_ewald_shift` | `788.438` | `5.012` | `4435.358` | persists |
| `safe_pme_none` | `779.377` | `5.361` | `3333.198` | persists |

What this means:

- reciprocal-only tightening does not rescue the run
- PME mesh versus full Ewald also does not rescue the run
- direct-space modifier change also does not rescue the run

There is no clean monotonic pattern that would justify saying:

- "this is clearly PME precision"
- or "this is clearly direct-space Coulomb shift"

## Conclusion

Confirmed:

- the authoritative setup is not running TP1.4's LJ-PME path
- TP1.8b executed narrower Coulomb-path variants than TP1.8's coarse cutoff comparison
- all narrowed variants preserved the safe short-range baseline
- all narrowed variants still show immediate runaway

Not confirmed:

- reciprocal-space / PME dominance
- direct-space Coulomb dominance
- a source-level patch target

Best classification:

- `mixed_or_still_unresolved`

The most defensible next step is narrower source-level tracing around the authoritative Coulomb long-range accumulation path, especially the PME-versus-Ewald split that `force.cpp` chooses, while keeping the safe short-range baseline fixed.
