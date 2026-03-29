# Validation Report — TP1.4 PME/SixthPower Direct Proof

## 1. Executive Summary
**Milestone Result: FAIL (Defect Reproduced and Plausibly Large Enough to Matter)**

TP1.4 directly tested the LJ-PME split used for PCFF-style 9-6 interactions with the smallest periodic mixed-type fixture we could justify: two atoms in a periodic box, fixed at 0.5 nm, with `rep-pow = 9` and `vdwtype = PME`. The expected property is simple: if the real-space/reciprocal-space split is internally consistent, total LJ energy and force for that fixed pair should stay approximately invariant when `rcut` moves.

That did not happen. Across `rcut = 0.7 -> 1.1 nm`, the total potential shifted from `-34.804665` to `-2.574015 kJ/mol` and the force on atom 2 shifted from `-14.4074` to `-2.38829`. This is direct numerical evidence of a split inconsistency. The magnitude is large enough to matter physically, but TP1.4 alone does **not** prove this defect is the dominant TP1.3 cause.

## 2. K1 Basis and Path Localization

K1 already established that isolated 9-6 pair-force mathematics is internally consistent. TP1.4 therefore targeted the PME split path on top of those pair interactions rather than the standalone 9-6 pair kernel.

The current source path localized in TP1.4 is:

1. `src/gromacs/mdlib/forcerec.cpp`
   - builds full pair prefactors as `6*C6` and `repulsionPower*C_repulsive`
   - separately builds LJ-PME grid correction parameters from C6 only
2. `src/gromacs/nbnxm/atomdata.cpp`
   - routes 9-6 LJ-PME pair parameters through `LJCombinationRule::None`
   - still prepares geometric C6 grid data for LJ-PME
3. `src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h`
   - evaluates the real-space 9-6 pair term from the full pair matrix
   - applies LJ-Ewald correction through `c6grid`
4. `src/gromacs/ewald/pme.cpp`
   - allocates a single geometric LJ grid for non-LB LJ-PME

This gives a directly testable split: full-matrix real-space pair treatment plus geometric reciprocal C6 correction.

## 3. Fixture and Method

### Minimal fixture
- 2 atoms
- mixed atom types `A` and `B`
- periodic box `5 x 5 x 5 nm`
- fixed pair distance `0.5 nm`
- topology defaults: `comb-rule = 4`, `rep-pow = 9`
- MDP: `vdwtype = PME`, `lj-pme-comb-rule = geometric`, `coulombtype = Cut-off`

### Scan
- held coordinates fixed
- varied `rcut = rvdw = rcoulomb = rlist` over `0.7, 0.8, 0.9, 1.0, 1.1 nm`
- measured:
  - `LJ (SR)`
  - `LJ recip.`
  - total potential
  - force on atom 2 in `x`

Machine-readable fixture definition:
- `tests/reference_results/tp1_4_pme_proof/pme_fixture_definition.json`

## 4. Direct Evidence

| `rcut` (nm) | LJ (SR) | LJ recip. | Total potential | Force on atom 2 (x) |
| :--- | :--- | :--- | :--- | :--- |
| 0.7 | 6.943827 | -41.748489 | -34.804665 | -14.40740 |
| 0.8 | 3.124440 | -18.999409 | -15.874969 | -6.63457 |
| 0.9 | 1.544618 | -9.550016 | -8.005398 | -4.29121 |
| 1.0 | 0.819228 | -5.205198 | -4.385970 | -3.14929 |
| 1.1 | 0.456348 | -3.030363 | -2.574015 | -2.38829 |

Derived continuity metrics:
- `potential_span = 32.23065 kJ/mol`
- `force_span = 12.01911`
- `relative_span_vs_rcut_1p1 = 12.52`

Expected continuity versus observed behavior:
- Expected: total force and potential approximately invariant versus `rcut`
- Observed: large monotonic drift in both quantities

Artifacts:
- `tests/reference_results/tp1_4_pme_proof/pme_energy_force_scan.csv`
- `tests/reference_results/tp1_4_pme_proof/pme_continuity_summary.json`
- `tests/reference_results/tp1_4_pme_proof/tp1_4_suspicion_update.json`

## 5. Interpretation

What TP1.4 confirms:
- A real numerical inconsistency exists in the LJ-PME split exercised by 9-6 PCFF-like mixed pairs.
- The defect is reproduced on a minimal periodic fixture with no transport calculations and no large charged-system reruns.
- The size of the drift is too large to dismiss as harmless cutoff noise.

What TP1.4 does **not** confirm:
- the exact mathematical subterm that is wrong
- that this is the dominant TP1.3 cause
- that the defect alone explains the full charged-system runaway

## 6. Commands Run

- `git status --short`
- `sed -n ... docs/k1_code_path_map.md`
- `sed -n ... docs/k1_kernel_consistency_audit.md`
- `sed -n ... docs/validation_report_k1.md`
- `sed -n ... src/gromacs/mdlib/forcerec.cpp`
- `sed -n ... src/gromacs/nbnxm/atomdata.cpp`
- `sed -n ... src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h`
- `sed -n ... src/gromacs/nbnxm/simd_kernel_inner.h`
- `sed -n ... src/gromacs/ewald/pme.cpp`
- `python3 tools/run_tp1_4_pme_proof/run_pme_proof.py`

## 7. Conclusion

TP1.4 reproduced the suspected PME/SixthPower inconsistency directly. Verdict: **defect reproduced and plausibly large enough to matter**. The exact subterm mismatch remains unresolved, so the next step should be a narrower kernel-level isolation milestone, not a broad production rerun and not a broad code patch.
