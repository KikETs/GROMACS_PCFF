# Validation Report M4

## Scope frozen in M4

M4 implements the CPU-side nonbonded semantics required for PCFF/Class2 parity in the current fork:

- Class2 `9-6` Lennard-Jones semantics
- sixth-power mixing semantics
- generated and explicit `1-4` LJ semantics
- exclusion semantics
- coupling with long-range Coulomb treatment on the CPU path

M4 still does **not** implement:

- GPU/CUDA kernels
- `r-RESPA`
- LJ-PME support for Class2 `9-6`
- free-energy support for non-`12` repulsion powers

The frozen target remains
[docs/pcff_respa_reference_spec.md](./pcff_respa_reference_spec.md).

## Architecture entry points touched

### Topology and mixing semantics

- [api/legacy/include/gromacs/mdtypes/md_enums.h](../api/legacy/include/gromacs/mdtypes/md_enums.h)
  adds `CombinationRule::SixthPower`.
- [src/gromacs/mdtypes/md_enums.cpp](../src/gromacs/mdtypes/md_enums.cpp)
  adds enum-string support.
- [src/gromacs/gmxpreprocess/toppush.cpp](../src/gromacs/gmxpreprocess/toppush.cpp)
  implements sixth-power atom-type mixing.
- [src/gromacs/gmxpreprocess/topio.cpp](../src/gromacs/gmxpreprocess/topio.cpp)
  updates generated `1-4` scaling semantics for sixth-power mixing.
- [src/gromacs/gmxpreprocess/convparm.cpp](../src/gromacs/gmxpreprocess/convparm.cpp)
  converts sigma/epsilon input to exact Class2 `9-6` coefficient form for both normal nonbonded and listed pair interactions.

### Runtime evaluation

- [src/gromacs/mdtypes/interaction_const.cpp](../src/gromacs/mdtypes/interaction_const.cpp)
  generalizes VdW shift/switch handling from fixed `12`-power to arbitrary repulsion power.
- [src/gromacs/mdlib/forcerec.cpp](../src/gromacs/mdlib/forcerec.cpp)
  allows non-`12` repulsion powers on CPU and routes them away from unsupported optimized kernels.
- [src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h](../src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h)
  evaluates exact `r^-9` repulsion in the plain-C nonbonded kernel.
- [src/gromacs/listed_forces/pairs.cpp](../src/gromacs/listed_forces/pairs.cpp)
  generalizes listed-pair evaluation and softcore-related coefficient handling to arbitrary repulsion powers.
- [src/gromacs/tables/forcetable.cpp](../src/gromacs/tables/forcetable.cpp)
  updates table scaling from hard-coded `1/12` to `1/reppow`.

## Exact semantics implemented

### Nonbonded Class2 `9-6`

For mixed or self nonbonded pairs under sixth-power mixing, preprocessing stores the exact internal coefficient form

- `c6 = 18 * epsilon * sigma^6`
- `c_rep = 18 * epsilon * sigma^9`

with `c_rep` carried in the existing `c12` slot as an internal storage detail only. Runtime interpretation is controlled by `reppow = 9`, so this remains numerically exact rather than an approximation.

### Generated and explicit `1-4`

For listed `1-4` interactions under sixth-power mixing, preprocessing stores the direct Class2 pair form

- `c6 = 3 * epsilon * sigma^6`
- `c_rep = 2 * epsilon * sigma^9`

This matches the frozen PCFF/Class2 semantics and is tested explicitly for generated `1-4` pairs.

### Mixing rule

Sixth-power mixing is implemented as:

- `sigma_ij = ((sigma_i^6 + sigma_j^6) / 2)^(1/6)`
- `epsilon_ij = 2 * sqrt(epsilon_i * epsilon_j) * sigma_i^3 * sigma_j^3 / (sigma_i^6 + sigma_j^6)`

Negative sigma handling remains consistent with existing GROMACS convention:

- attractive term disabled (`c6 = 0`)
- repulsive term evaluated with `abs(sigma)`

### Coulomb coupling and exclusions

- explicit exclusion handling is preserved in the CPU nonbonded path
- listed `LJ14` and `LJC14Q` paths remain distinct
- tabulated/PME-style Coulomb treatment remains separable from Class2 `9-6` VdW on the tested CPU path

## Validation artifacts added

- [src/gromacs/gmxpreprocess/tests/convparm.cpp](../src/gromacs/gmxpreprocess/tests/convparm.cpp)
- [src/gromacs/gmxpreprocess/tests/grompp_directives.cpp](../src/gromacs/gmxpreprocess/tests/grompp_directives.cpp)
- [src/gromacs/listed_forces/tests/pairs.cpp](../src/gromacs/listed_forces/tests/pairs.cpp)
- [src/gromacs/nbnxm/tests/pcff_class2_nonbonded.cpp](../src/gromacs/nbnxm/tests/pcff_class2_nonbonded.cpp)
- [src/programs/mdrun/tests/pcff_short_md.cpp](../src/programs/mdrun/tests/pcff_short_md.cpp)
- [tests/reference_results/m4/README.md](../tests/reference_results/m4/README.md)
- [tests/reference_results/m4/small_oligomer/single_point.json](../tests/reference_results/m4/small_oligomer/single_point.json)
- [tests/reference_results/m4/small_salt_polymer_box/single_point.json](../tests/reference_results/m4/small_salt_polymer_box/single_point.json)
- [tests/reference_schema/test_corpus_schema.py](../tests/reference_schema/test_corpus_schema.py)

## Agreement summary

### Parser and coefficient conversion

The preprocessing tests now verify:

- exact nonbonded `9-6` coefficient conversion for `LennardJonesShortRange`
- exact direct `1-4` coefficient conversion for `LennardJones14`
- sixth-power mixing with generated `1-4` parameters from `grompp`
- failure on invalid `rep-pow` for sixth-power mixing

These checks are exact coefficient-semantic checks, not loose tolerance tests.

### Pair-curve energy and force agreement

[pairs.cpp](../src/gromacs/listed_forces/tests/pairs.cpp) validates the listed `1-4` runtime against the analytic Class2 `9-6` curve.

Acceptance tolerances:

- energy: `2e-4`
- force: `2e-3`
- Coulomb separation residual: `1e-7` to `2e-3` depending on the checked quantity

All M4 listed-pair tests pass.

### Plain-C CPU nonbonded kernel agreement

[pcff_class2_nonbonded.cpp](../src/gromacs/nbnxm/tests/pcff_class2_nonbonded.cpp) validates:

- exact `9-6` energy and force curves across multiple distances
- independence of Coulomb energy from the added Class2 VdW term in the tabulated/PME-style path
- explicit exclusion suppression of both VdW and Coulomb interactions

Acceptance tolerances:

- VdW energy: `2e-4`
- force on the test axis: `2e-3`
- excluded interaction residuals: `1e-8`

All M4 plain-C CPU nonbonded tests pass.

### Frozen LAMMPS reference outputs

Whole-system LAMMPS normalized outputs are now committed for:

- [small_oligomer](../tests/reference_results/m4/small_oligomer)
- [small_salt_polymer_box](../tests/reference_results/m4/small_salt_polymer_box)

Key frozen single-point values:

| System | `pe` | `evdwl` | `ecoul` | `elong` |
| --- | ---: | ---: | ---: | ---: |
| `small_oligomer` | `25.320359` | `-0.15323047` | `9.4886598` | `-20.012769` |
| `small_salt_polymer_box` | `92.11412` | `-0.33788696` | `6.3682874` | `-64.379354` |

The committed JSON payloads are schema-tested and key observables are frozen in
[tests/reference_schema/test_corpus_schema.py](../tests/reference_schema/test_corpus_schema.py).

### Whole-system single-point regression against frozen M4 outputs

[pcff_short_md.cpp](../src/programs/mdrun/tests/pcff_short_md.cpp)
now contains a dedicated `PcffSinglePointParity` regression that runs GROMACS CPU single-point
calculations from committed GROMACS fixtures under
[tests/reference_results/m4](../tests/reference_results/m4)
and compares them against the frozen LAMMPS `single_point.json` and `forces.json` payloads for:

- `small_oligomer`
- `small_salt_polymer_box`

Acceptance tolerances:

- bonded energy breakdown: `5e-4 kcal/mol`
- total VdW energy: `2e-2 kcal/mol`
- total electrostatic energy: `7e-2 kcal/mol`
- total potential energy: `6e-2 kcal/mol`
- per-component force: `9e-2 kcal/mol/angstrom`

## Tests run

Narrow M4 subsets:

- `./build/bin/gmxpreprocess-test --gtest_filter='*PcffClass2Nonbonded*:*SixthPower*'`
- `./build/bin/listed_forces-test --gtest_filter='PcffClass2PairCurveTest.*'`
- `./build/bin/nbnxm-test --gtest_filter='PcffClass2NonbondedCurveTest.*'`
- `./build/bin/mdrun-non-integrator-test --gtest_filter='PcffSinglePointParity*'`
- `pytest -q tests/reference_schema/test_corpus_schema.py`

Result summary:

- `gmxpreprocess-test`: `4 passed`
- `listed_forces-test`: `2 passed`
- `nbnxm-test`: `9 passed`
- `mdrun-non-integrator-test`: `4 passed`
- `pytest tests/reference_schema`: `6 passed`

Broader regression reruns:

- `./build/bin/gmxpreprocess-test`
- `./build/bin/listed_forces-test`
- `./build/bin/nbnxm-test`

Result summary:

- `gmxpreprocess-test`: `259 passed`, `40 skipped`, `0 failed`
- `listed_forces-test`: `155 passed`, `0 failed`
- `nbnxm-test`: exit code `0`, with only pre-existing skip classes in generic kernel coverage

## Known caveats and unresolved issues before M5

1. M4 correctness now includes automated whole-system CPU single-point energy and force regression against the frozen LAMMPS outputs for `small_oligomer` and `small_salt_polymer_box`. What is still missing is automated short-MD trajectory parity at the M4 layer.
2. Non-`12` repulsion powers currently fall back to the plain-C CPU nonbonded path. This is correct but intentionally not fast.
3. LJ-PME for Class2 `9-6` is still unsupported.
4. Free-energy support for non-`12` repulsion powers remains explicitly disabled.
5. GPU kernels are still absent.
6. The frozen M4 corpus still uses synthetic Class2 coefficients, not a published production PCFF parameter set.

## Readiness for M5

M4 is sufficient for CPU-first short-MD parity work in this limited sense:

- the preprocessing path now preserves exact Class2 `9-6`, sixth-power mixing, and `1-4` semantics
- the CPU listed-pair and plain-C nonbonded kernels evaluate exact `9-6` forces and energies
- exclusions and Coulomb coupling are covered by automated tests
- whole-system LAMMPS normalized outputs are frozen in-repo for the two M4 fixtures

What M4 still does **not** justify is moving to GPU or `r-RESPA`. The next defensible step is M5: use the frozen `small_oligomer` and `small_salt_polymer_box` references to add short-MD CPU parity checks before any CUDA or multi-time-step work.
