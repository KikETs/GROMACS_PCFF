# TP1.4 — PME/SixthPower Direct Proof

## Objective

Directly test whether the current LJ-PME path used for PCFF-style 9-6 interactions is physically continuous across the real-space / reciprocal-space split.

## Why This Milestone Exists

K1 already showed that isolated 9-6 pair mathematics is internally consistent. That shifts the burden to the PME split path:

- pair-space nonbonded evaluation
- reciprocal-space LJ grid correction
- their matching at the split boundary

TP1.4 therefore avoids large system reruns and instead uses the smallest periodic fixture that can expose a split inconsistency.

## Localized Path

The path exercised in this milestone is:

- `src/gromacs/mdlib/forcerec.cpp`
  - runtime pair prefactors are stored as `6*C6` and `repulsionPower*C_repulsive`
  - `makeLJPmeC6GridCorrectionParameters()` builds a separate C6 grid correction table
- `src/gromacs/nbnxm/atomdata.cpp`
  - 9-6 LJ-PME pair parameters fall back to `LJCombinationRule::None`
  - LJ-PME still prepares geometric C6 grid data
- `src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h`
  - real-space 9-6 pair interaction uses the full pair matrix
  - LJ-Ewald correction subtracts `c6grid`-based reciprocal contribution
- `src/gromacs/ewald/pme.cpp`
  - non-LB LJ-PME uses one geometric LJ grid

This is the precise split TP1.4 tests.

## Minimal Fixture

Fixture definition:
- 2 atoms, mixed types `A` and `B`
- cubic periodic box: `5.0 nm`
- fixed interatomic distance: `0.5 nm`
- topology defaults: `comb-rule = 4`, `rep-pow = 9`
- MDP:
  - `vdwtype = PME`
  - `lj-pme-comb-rule = geometric`
  - `coulombtype = Cut-off`
  - `ewald-rtol-lj = 1e-5`

Why mixed types:
- a same-type fixture can show split drift
- a mixed-type fixture is stricter because it directly exercises the suspected SixthPower/PME mixed-pair path

## Measurement Strategy

Hold the pair geometry fixed and move only the split boundary:

- `rcut = 0.7, 0.8, 0.9, 1.0, 1.1 nm`

For each `rcut`, record:

- `LJ (SR)`
- `LJ recip.`
- total potential
- force on atom 2 in `x`

If the split is correct, the total force and potential should stay approximately invariant.

## Results

| `rcut` (nm) | LJ (SR) | LJ recip. | Potential | Force x (atom 2) |
| :--- | :--- | :--- | :--- | :--- |
| 0.7 | 6.943827 | -41.748489 | -34.804665 | -14.40740 |
| 0.8 | 3.124440 | -18.999409 | -15.874969 | -6.63457 |
| 0.9 | 1.544618 | -9.550016 | -8.005398 | -4.29121 |
| 1.0 | 0.819228 | -5.205198 | -4.385970 | -3.14929 |
| 1.1 | 0.456348 | -3.030363 | -2.574015 | -2.38829 |

Continuity metrics:
- `potential_span = 32.23065 kJ/mol`
- `force_span = 12.01911`
- `relative_span_vs_rcut_1p1 = 12.52`

## Direct Conclusion

The total LJ energy and force are not even approximately invariant with respect to `rcut`. That is direct evidence of a physical inconsistency in the LJ-PME split exercised by this 9-6 mixed-pair fixture.

Classification:
- **defect reproduced and plausibly large enough to matter**

## Limits

This milestone does **not** establish:

- the single exact algebraic term at fault
- that this is the dominant TP1.3 cause
- that transport calculations should begin

Those would require a narrower localization step first.

## Preserved Outputs

- `tools/run_tp1_4_pme_proof/run_pme_proof.py`
- `tools/run_tp1_4_pme_proof/run_logs.json`
- `tests/reference_results/tp1_4_pme_proof/pme_fixture_definition.json`
- `tests/reference_results/tp1_4_pme_proof/pme_energy_force_scan.csv`
- `tests/reference_results/tp1_4_pme_proof/pme_continuity_summary.json`
- `tests/reference_results/tp1_4_pme_proof/tp1_4_suspicion_update.json`
