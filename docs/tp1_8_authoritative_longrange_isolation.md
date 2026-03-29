# TP1.8 — Authoritative Long-Range / PME Blocker Isolation

## Scope

TP1.8 does not revisit short-range pairlist code. TP1.6 and TP1.7b already established the current safe short-range baseline and showed that fixing the unsafe pairlist regime alone does not materially remove the authoritative runaway.

This milestone asks a narrower question:

- with the short-range safe baseline fixed
- what long-range path is actually active
- and does changing only that long-range path materially weaken the authoritative runaway

## Constraining Evidence

- TP1.4 reproduced a real LJ-PME split inconsistency for PCFF-style 9-6 mixed pairs, but only on `vdwtype = PME`.
- TP1.7b showed the authoritative charged system still runs away under the same-build safe short-range regime.
- TP1.7b did **not** prove whether the surviving blocker was specifically PME, broadly long-range, or mixed.

## Active-Path Verification

The authoritative TP1.8 safe baseline was verified from runtime files, not assumed.

Safe reference facts:

- [raw_safe_pme_n10_r0911_mdout.mdp](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation/raw_safe_pme_n10_r0911_mdout.mdp)
  - `coulombtype = PME`
  - `vdw-type = Cut-off`
  - `nstlist = 10`
  - `rlist = 0.911`
  - `verlet-buffer-tolerance = -1`
- [raw_safe_pme_n10_r0911_md.log](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation/raw_safe_pme_n10_r0911_md.log)
  - `Will do PME sum in reciprocal space for electrostatic interactions.`
  - `Detected LJ repulsion power 9.`
  - `Using plain-C-4x4 4x4 nonbonded short-range kernels`
  - `updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm`

Source implications:

- [pme.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/ewald/pme.cpp#L782) and [pme.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/ewald/pme.cpp#L783)
  - Coulomb PME is active when `coulombtype = PME`
  - LJ-PME is active only when `vdwtype = Pme`
- [nbnxm_setup.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/nbnxm_setup.cpp#L462) and [atomdata.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/atomdata.cpp#L400)
  - the TP1.4-style LJ-PME grid path is inactive here because `vdw-type = Cut-off`
- [kerneldispatch.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/kerneldispatch.cpp#L113)
  - PME/Ewald direct-space Coulomb shares the Ewald kernel family
- [kerneldispatch.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/kerneldispatch.cpp#L164)
  - VdW remains the plain cut-off path

That means TP1.8 is isolating the Coulomb long-range path, not the TP1.4 LJ-PME path.

## Comparison Matrix

Short-range fixed across all runs:

- same authoritative system: `dense_salt_polymer`
- same start structure and topology from TP1.3 executed baseline
- `nstlist = 10`
- `rlist = 0.911`
- `verlet-buffer-tolerance = -1`
- `vdw-type = Cut-off`
- `rvdw = 0.9`
- `repulsion power = 9`
- same `dt`, `seed`, PME/Cut-off family unless the long-range axis itself changes it

Long-range variants:

1. `safe_pme_n10_r0911`
   - reference safe authoritative baseline
2. `safe_pme_tight_fs006_po6`
   - tighter PME mesh/accuracy only
   - `fourierspacing = 0.06`
   - `pme-order = 6`
   - `ewald-rtol = 1e-6`
3. `safe_cutoff_n10_r0911`
   - removes reciprocal-space Coulomb
   - `coulombtype = Cut-off`
   - keeps the same short-range baseline and cut-off radii

The third variant is intentionally a coarse isolation tool. It is not a physically equivalent replacement for PME.

## Direct Results

### 1. Tighter PME accuracy

Runtime verification:

- [raw_safe_pme_tight_fs006_po6_mdout.mdp](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation/raw_safe_pme_tight_fs006_po6_mdout.mdp)
  - `coulombtype = PME`
  - `fourierspacing = 0.06`
  - `pme-order = 6`
  - `rlist = 0.911`
- [raw_safe_pme_tight_fs006_po6_md.log](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation/raw_safe_pme_tight_fs006_po6_md.log)
  - still reports PME reciprocal electrostatics
  - still reports `plain-C-4x4`
  - still reports `buffer 0.011 nm, rlist 0.911 nm`

Outcome:

- onset stays `1.0 ps`
- `max_temperature_k` drops only from `801.986` to `795.445`
- `total_energy_range_kj` changes only from `5.793` to `5.457`
- pressure extremes are not improved

Interpretation:

- this is not evidence for a simple Coulomb PME precision problem

### 2. No-reciprocal cut-off

Runtime verification:

- [raw_safe_cutoff_n10_r0911_mdout.mdp](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation/raw_safe_cutoff_n10_r0911_mdout.mdp)
  - `coulombtype = Cut-off`
  - `rlist = 0.911`
  - `vdw-type = Cut-off`
- [raw_safe_cutoff_n10_r0911_md.log](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8_longrange_isolation/raw_safe_cutoff_n10_r0911_md.log)
  - no PME reciprocal line
  - still reports `plain-C-4x4`
  - still reports `buffer 0.011 nm, rlist 0.911 nm`

Outcome:

- onset still `1.0 ps`
- `max_temperature_k` drops to `758.498`
- but `total_energy_range_kj` explodes to `135.209`
- pressure extremes remain very large

Interpretation:

- simply removing reciprocal-space Coulomb does not cleanly rescue the authoritative runaway
- but because this changes the electrostatics model, it does **not** prove PME is irrelevant
- the result is still mixed and not patch-grade

## Conclusion

TP1.8 materially narrows one thing:

- TP1.4’s LJ-PME path is not the active authoritative reciprocal path

TP1.8 does **not** materially narrow the remaining blocker to a single Coulomb PME defect:

- tighter PME accuracy is not a material lever
- no-reciprocal cut-off also leaves immediate runaway
- therefore the surviving blocker remains **mixed or still unresolved**

Patch boundary:

- no source-level production patch is justified from TP1.8

Next step:

- keep the safe short-range baseline
- keep the same authoritative system
- build a narrower Coulomb long-range vs mixed-path audit before any code changes
