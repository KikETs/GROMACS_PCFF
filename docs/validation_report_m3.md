# Validation Report M3

## Scope frozen in M3

M3 implements CPU-side support for full PCFF/Class2 dihedral terms and stops there.

Included in M3:

- `DihedralClass2` primary dihedral term
- middle-bond-torsion cross term
- end-bond-torsion cross term
- angle-torsion cross term
- angle-angle-torsion cross term
- bond-bond-1,3 cross term

Still excluded from M3:

- `run_style respa`
- GPU/CUDA kernels
- `lj/class2`
- `lj/class2/coul/long`
- any approximation that collapses Class2 dihedral semantics into an existing simpler torsion model

The reference target remains the frozen M1 specification in
[docs/pcff_respa_reference_spec.md](/home/user/바탕화면/gromacs/docs/pcff_respa_reference_spec.md).

## Architecture decision

M3 adds a distinct `DihedralClass2` interaction path instead of reusing existing proper-dihedral APIs.

That decision is necessary because full PCFF/Class2 dihedral semantics require one primary torsion term plus five additional cross-term families with a parameter layout that does not match `ProperDihedrals`, `RestrictedDihedrals`, `RyckaertBellemansDihedrals`, or Fourier torsions.

The CPU reference path now runs through these subsystems:

- [ifunc.h](/home/user/바탕화면/gromacs/api/legacy/include/gromacs/topology/ifunc.h)
- [idef.h](/home/user/바탕화면/gromacs/api/legacy/include/gromacs/topology/idef.h)
- [ifunc.cpp](/home/user/바탕화면/gromacs/src/gromacs/topology/ifunc.cpp)
- [idef.cpp](/home/user/바탕화면/gromacs/src/gromacs/topology/idef.cpp)
- [tpxio.cpp](/home/user/바탕화면/gromacs/src/gromacs/fileio/tpxio.cpp)
- [topdirs.cpp](/home/user/바탕화면/gromacs/src/gromacs/gmxpreprocess/topdirs.cpp)
- [convparm.cpp](/home/user/바탕화면/gromacs/src/gromacs/gmxpreprocess/convparm.cpp)
- [toppush.cpp](/home/user/바탕화면/gromacs/src/gromacs/gmxpreprocess/toppush.cpp)
- [toputil.cpp](/home/user/바탕화면/gromacs/src/gromacs/gmxpreprocess/toputil.cpp)
- [bonded.cpp](/home/user/바탕화면/gromacs/src/gromacs/listed_forces/bonded.cpp)

## Parameter traceability and topology semantics

`DihedralClass2` is stored as an explicit 32-parameter block in `t_iparams`.

The flattened GROMACS topology order is:

1. `k1 phi1 k2 phi2 k3 phi3`
2. `mbt_f1 mbt_f2 mbt_f3 mbt_r0`
3. `ebt_f1_1 ebt_f2_1 ebt_f3_1 ebt_f1_2 ebt_f2_2 ebt_f3_2 ebt_r0_1 ebt_r0_2`
4. `at_f1_1 at_f2_1 at_f3_1 at_f1_2 at_f2_2 at_f3_2 at_theta0_1 at_theta0_2`
5. `aat_k aat_theta0_1 aat_theta0_2`
6. `bb13t_k bb13t_r10 bb13t_r30`

Conversion rules implemented in preprocessing:

- phase and equilibrium angles are read in degrees and converted to radians in [convparm.cpp](/home/user/바탕화면/gromacs/src/gromacs/gmxpreprocess/convparm.cpp)
- force/length coefficients are stored verbatim in the same physical units already used by the bonded runtime
- `.tpr` serialization/deserialization includes all 32 fields through [tpxio.cpp](/home/user/바탕화면/gromacs/src/gromacs/fileio/tpxio.cpp)

M3 also extends topology parsing so Class2 dihedrals are not truncated by older fixed-width token handling. That parser change lives in [toppush.cpp](/home/user/바탕화면/gromacs/src/gromacs/gmxpreprocess/toppush.cpp).

## Term-by-term energy agreement

Term isolation is covered in [pcff_class2_dihedral.cpp](/home/user/바탕화면/gromacs/src/gromacs/listed_forces/tests/pcff_class2_dihedral.cpp) with analytic checks for:

- primary dihedral contribution
- middle-bond-torsion contribution
- end-bond-torsion contribution
- angle-torsion contribution
- angle-angle-torsion contribution
- bond-bond-1,3 contribution
- full summed contribution

The frozen LAMMPS regression summary for `dihedral_toy` is stored in
[dihedral_toy.tsv](/home/user/바탕화면/gromacs/tests/reference_results/m3/dihedral_toy.tsv).

Reference values from LAMMPS are:

- `pe = 27.626426 kcal/mol`
- `ebond = 13.697359 kcal/mol`
- `eangle = 12.503652 kcal/mol`
- `edihed = 1.4254148 kcal/mol`

Automated M3 regression compares the GROMACS CPU result against those frozen totals and passes.

## Total force agreement

The same frozen summary stores per-atom LAMMPS forces in `real` units:

- atom 1: `(1.57699, 20.4136, 1.55479)`
- atom 2: `(75.1704, 7.52742, 8.3573)`
- atom 3: `(-14.3327, -11.2292, 32.4421)`
- atom 4: `(-62.4148, -16.7118, -42.3542)`

The M3 force regression converts those values to GROMACS internal units and checks total analytic force agreement on the CPU path.

Finite-difference validation is automated for the `dihedral_toy` geometry on selected components:

- atom 2 `x`
- atom 2 `z`
- atom 4 `x`
- atom 4 `z`

The finite-difference step is `1e-5 nm`, and the acceptance tolerance is `3.0 kJ/mol/nm`.

## Convention decisions

The following decisions are now explicit:

1. Full PCFF/Class2 dihedral is a dedicated interaction type, not an alias of an existing GROMACS torsion.
2. Phases and equilibrium angles are stored internally in radians after preprocessing conversion from topology degrees.
3. Runtime geometry and derivative handling follow the LAMMPS Class2 implementation closely, including conservative clamping around numerically delicate trigonometric intermediates.
4. Mirror-related sign behavior is tested explicitly: mirrored geometry flips the signed dihedral convention while preserving total energy.
5. `+180` and `-180` phase edge cases are treated as physically equivalent and tested directly.

## Numerical stability notes

Targeted stress coverage added in [pcff_class2_dihedral.cpp](/home/user/바탕화면/gromacs/src/gromacs/listed_forces/tests/pcff_class2_dihedral.cpp):

- near-linear geometry remains finite and non-NaN
- mirrored geometry preserves energy while changing the signed torsion orientation
- phase periodicity at `+180/-180` remains consistent
- malformed parameter-count input is rejected in [grompp_directives.cpp](/home/user/바탕화면/gromacs/src/gromacs/gmxpreprocess/tests/grompp_directives.cpp)

One numerical detail matters: the `+180/-180` force-equivalence check needs a `1e-5` force tolerance in single precision. A stricter `1e-6` threshold is not stable enough to be a useful regression gate.

## Tests run

Narrow M3 subsets:

- `./build/bin/listed_forces-test --gtest_filter='*PcffClass2Dihedral*'`
- `./build/bin/gmxpreprocess-test --gtest_filter='*PcffClass2*'`

Result summary:

- `listed_forces-test`: `12 passed`
- `gmxpreprocess-test`: `9 passed`

Broader regression reruns:

- `./build/bin/listed_forces-test`
- `./build/bin/gmxpreprocess-test`

Result summary:

- `listed_forces-test`: `153 passed`
- `gmxpreprocess-test`: `255 passed`, `40 skipped`, `0 failed`

The `40 skipped` cases are generic pre-existing conversion skips, not new M3 failures.

## Explicit unresolved issues before M4

1. There is no user-facing warning path yet equivalent to LAMMPS `Dihedral::problem()` diagnostics for problematic Class2 torsion geometries. M3 matches the conservative clamp behavior in the kernel, but not the warning emission.
2. M3 validates full Class2 dihedral only on the bonded toy reference. It does not yet prove parity in oligomer or polymer-box systems where nonbonded PCFF terms also matter.
3. Restart policy is still limited to `.tpr` serialization coverage. There is no broader restart/reproducibility contract frozen yet for later M4 work.
4. The current force finite-difference checks are targeted, not exhaustive Jacobian validation over all coordinates and pathological geometries.
5. No mixed-precision, SIMD, or threaded reproducibility study has been done for the new dihedral path.

## Readiness for M4

M3 is sufficient as a CPU bonded reference for the next milestone in this narrow sense:

- full PCFF/Class2 dihedral energy and force evaluation now exists on CPU
- parameter parsing, conversion, serialization, and runtime dispatch are wired end to end
- the primary term and all five required cross terms are individually tested
- regression against frozen LAMMPS `dihedral_toy` reference data passes

What M3 does **not** justify is jumping to GPU or `r-RESPA`. The next technically defensible step is nonbonded PCFF parity on CPU, because without `lj/class2` and `lj/class2/coul/long`, larger-system validation will remain underdetermined.
