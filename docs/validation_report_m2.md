# Validation Report M2

## Scope frozen in M2

M2 implements CPU-side support for the PCFF/Class2 bonded terms below and stops there:

- `BondClass2`
- `AngleClass2`
- `ImproperClass2`

M2 does **not** implement:

- full Class2 dihedral terms
- `run_style respa`
- `lj/class2`
- `lj/class2/coul/long`
- any GPU/CUDA path

The reference target remains the frozen M1 specification in
[docs/pcff_respa_reference_spec.md](/home/user/바탕화면/gromacs/docs/pcff_respa_reference_spec.md).

## Architectural entry points touched

### Topology and function typing

- [api/legacy/include/gromacs/topology/ifunc.h](/home/user/바탕화면/gromacs/api/legacy/include/gromacs/topology/ifunc.h)
  adds `BondClass2`, `AngleClass2`, and `ImproperClass2`.
- [src/gromacs/topology/ifunc.cpp](/home/user/바탕화면/gromacs/src/gromacs/topology/ifunc.cpp)
  registers the interaction definitions and parameter counts.
- [api/legacy/include/gromacs/topology/idef.h](/home/user/바탕화면/gromacs/api/legacy/include/gromacs/topology/idef.h)
  extends `t_iparams` with Class2 bond, angle, and improper parameter storage.
- [src/gromacs/topology/idef.cpp](/home/user/바탕화면/gromacs/src/gromacs/topology/idef.cpp)
  adds human-readable parameter printing.
- [src/gromacs/fileio/tpxio.cpp](/home/user/바탕화면/gromacs/src/gromacs/fileio/tpxio.cpp)
  serializes the new parameter blocks into `.tpr`.

### Preprocessing and topology conversion

- [src/gromacs/gmxpreprocess/topdirs.cpp](/home/user/바탕화면/gromacs/src/gromacs/gmxpreprocess/topdirs.cpp)
  maps GROMACS topology directive function numbers to the new Class2 function types.
- [src/gromacs/gmxpreprocess/convparm.cpp](/home/user/바탕화면/gromacs/src/gromacs/gmxpreprocess/convparm.cpp)
  converts raw topology parameters into runtime `t_iparams`, including degree-to-radian conversion for angular equilibrium values.
- [src/gromacs/gmxpreprocess/toputil.cpp](/home/user/바탕화면/gromacs/src/gromacs/gmxpreprocess/toputil.cpp)
  prints the new function numbers back out for topology round-tripping.

### CPU runtime evaluation

- [src/gromacs/listed_forces/bonded.cpp](/home/user/바탕화면/gromacs/src/gromacs/listed_forces/bonded.cpp)
  adds CPU listed-force kernels and dispatch wiring for the three new interaction types.

## Formulas implemented

### BondClass2

Implemented as the LAMMPS quartic bond:

`E = K2 (r-r0)^2 + K3 (r-r0)^3 + K4 (r-r0)^4`

with analytic forces from `dE/dr`.

### AngleClass2

Implemented as the sum of:

- angle term
- bond-bond cross term
- bond-angle cross term

using the frozen M1 coefficient ordering and internal radian representation.

### ImproperClass2

Implemented as the sum of:

- improper out-of-plane term
- angle-angle coupling term

with the LAMMPS/Class2 ordering where the second atom is the central atom.

## Validation artifacts added

- [src/gromacs/listed_forces/tests/pcff_class2.cpp](/home/user/바탕화면/gromacs/src/gromacs/listed_forces/tests/pcff_class2.cpp)
- [src/gromacs/gmxpreprocess/tests/convparm.cpp](/home/user/바탕화면/gromacs/src/gromacs/gmxpreprocess/tests/convparm.cpp)
- [src/gromacs/gmxpreprocess/tests/grompp_directives.cpp](/home/user/바탕화면/gromacs/src/gromacs/gmxpreprocess/tests/grompp_directives.cpp)
- [tests/reference_results/m2/README.md](/home/user/바탕화면/gromacs/tests/reference_results/m2/README.md)
- [tests/reference_results/m2/bond_toy.tsv](/home/user/바탕화면/gromacs/tests/reference_results/m2/bond_toy.tsv)
- [tests/reference_results/m2/angle_toy.tsv](/home/user/바탕화면/gromacs/tests/reference_results/m2/angle_toy.tsv)
- [tests/reference_results/m2/improper_toy.tsv](/home/user/바탕화면/gromacs/tests/reference_results/m2/improper_toy.tsv)

M2 also fixes a generator-side normalization bug in
[tools/generate_lammps_golden/common.py](/home/user/바탕화면/gromacs/tools/generate_lammps_golden/common.py)
so real LAMMPS thermo headers are canonicalized correctly before reference summaries are derived.

## Energy agreement summary

Golden comparisons are executed against machine-readable summaries generated from LAMMPS toy systems.

| System | Compared observables | Acceptance tolerance |
| --- | --- | --- |
| `bond_toy` | `pe`, `ebond` | `2e-5 kcal/mol` |
| `angle_toy` | `pe`, `ebond`, `eangle` | `5e-5 kcal/mol` |
| `improper_toy` | `pe`, `ebond`, `eangle`, `eimp` | `5e-4 kcal/mol` |

All M2 golden energy tests pass on CPU with those tolerances.

## Force agreement summary

Golden force comparisons use LAMMPS `real` force units, i.e. `(kcal/mol)/Angstrom`.

| System | Golden-force tolerance |
| --- | --- |
| `bond_toy` | `5e-3` |
| `angle_toy` | `2e-2` |
| `improper_toy` | `6e-2` |

All M2 golden force tests pass on CPU with those tolerances.

Finite-difference checks are also automated:

| System | Checked components | Step | Acceptance tolerance |
| --- | --- | --- | --- |
| `bond_toy` | atom 1 `x`, atom 2 `x` | `1e-4 nm` | `2e-1 kJ/mol/nm` |
| `angle_toy` | atom 2 `x,y`, atom 3 `x,y` | `1e-4 nm` | `5e-1 kJ/mol/nm` |
| `improper_toy` | atom 4 `z` | `1e-4 nm` | `5.0 kJ/mol/nm` |

All listed finite-difference checks pass.

## Tests run

The narrow M2-focused test subsets were run first:

- `./build/bin/listed_forces-test --gtest_filter='PcffClass2FormulaTest.*:PcffClass2GoldenTest.*:PcffClass2ForceValidationTest.*'`
- `./build/bin/gmxpreprocess-test --gtest_filter='ConvertInteractionsPcffClass2Test.BondClass2ParametersAreStoredVerbatim:ConvertInteractionsPcffClass2Test.AngleClass2ConvertsAnglesToRadians:ConvertInteractionsPcffClass2Test.ImproperClass2ConvertsAnglesToRadians:PcffClass2DirectiveTest.ParsesBondAngleAndImproperClass2DirectivesWithoutDihedralClass2:PcffClass2DirectiveTest.RejectsMalformedBondClass2Parameters:PcffClass2DirectiveTest.RejectsMalformedAngleClass2Parameters:PcffClass2DirectiveTest.RejectsMalformedImproperClass2Parameters'`

These explicit filters are required because later milestones add `DihedralClass2` and PCFF nonbonded tests that also match broader `PcffClass2*` globs.

Broader regression coverage was then rerun:

- `./build/bin/listed_forces-test`
- `./build/bin/gmxpreprocess-test`

Result summary:

- `listed_forces-test`: `141 passed`
- `gmxpreprocess-test`: `252 passed`, `40 skipped`, `0 failed`

The skips are pre-existing unsupported/irrelevant interaction cases in generic conversion coverage, not new M2 failures.

## Known deviations and explicit limitations

1. Only bond, angle, and improper Class2 terms are implemented. Full Class2 dihedral support is intentionally absent.
2. No PCFF nonbonded support is included yet. `lj/class2` and `lj/class2/coul/long` remain unimplemented.
3. No `r-RESPA` work is included yet.
4. No GPU/CUDA implementation is included yet.
5. The current parser exposure is numeric function-type based in topology directives. M2 does not add a higher-level user-facing PCFF syntax layer.
6. Improper finite-difference validation currently covers only a numerically stable subset of components for the toy geometry. It does not yet constitute full Jacobian-style force validation for arbitrary improper geometries.
7. The improper runtime path treats singular or near-singular geometries conservatively; M2 does not yet define a broader robustness policy for linearized improper configurations.
8. The current golden regression set for M2 is toy-system focused. It is sufficient for bonded kernel bring-up, but not for claiming whole-force-field PCFF parity.

## Unresolved parser/runtime limitations

- No restart-compatibility policy is defined yet for the new parameter types beyond `.tpr` serialization.
- No topology import/export bridge exists yet for LAMMPS-native Class2 data files; M2 only adds the GROMACS-side bonded function plumbing.
- No energy-group or decomposition policy has been frozen yet for later full PCFF + long-range validation.
- No mixed-precision or SIMD sensitivity study has been completed for the improper kernel.

## Readiness for M3

M2 is ready for the next CPU-only bonded milestone only in this narrow sense:

- the Class2 bonded data path now exists from topology parsing through `.tpr` serialization to CPU listed-force evaluation;
- the implemented formulas are locked against toy-system LAMMPS golden references;
- parser failures for malformed parameter counts are covered by tests.

M3 should **not** start with GPU work. The next defensible step is full Class2 dihedral CPU design and validation, followed by PCFF nonbonded parity, and only then any `r-RESPA` or CUDA work.
