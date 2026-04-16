# Repulsion-Power-9 Specialized SIMD Microkernel Design Note

## Scope Freeze

This follow-on track is intentionally narrower than the admission milestone.

In scope:
- CPU `nbnxm` short-range exact `repulsionPower = 9` SIMD path
- specialization against the admitted generic non-`12` SIMD path already in runtime
- exact `LJ (SR)` `9-6` semantics for the admitted PCFF/Class2 route
- exact-r-RESPA CPU integration that consumes the specialized path

Explicitly out of scope:
- GPU or hybrid implications
- LJ-PME SIMD claims
- Buckingham
- free-energy perturbation with non-`12` repulsion
- dispersion correction with non-`12` repulsion
- generic non-`12` optimization claims beyond `repulsionPower = 9`
- broad chemistry claims beyond the audited fixtures
- `pair14` listed-force semantics; they remain on their existing runtime path and are not the new SIMD capability

## Old vs New Boundary

Old boundary:
- runtime admitted a real CPU SIMD exact `repulsionPower = 9` path
- the admitted path still used the generic non-`12` repulsion evaluation path inside the SIMD inner math

New boundary:
- runtime can explicitly select a specialized exact CPU SIMD `repulsionPower = 9` microkernel path
- the generic admitted SIMD path remains reachable for baseline comparison with an environment override

## Inner-Math Specialization

The generic admitted SIMD path evaluated non-`12` repulsion with a runtime `pow(rInv, repulsionPower)` path.

For the admitted `repulsionPower = 9` route, the specialized exact identity is:

- `r^-9 = r^-6 * r^-2 * r^-1`

This replaces the generic power call with an explicit multiplication chain while preserving the exact `9-6`
interaction shape. The specialization is exact algebra for the admitted power; it is not a tuned approximation
and it does not reinterpret `9-6` as `12-6`.

## Runtime Selection

Default admitted route:
- specialized exact CPU SIMD `repulsionPower = 9` path

Baseline override:
- `GMX_DISABLE_REPULSION_POWER_9_SIMD_SPECIALIZATION=1`
- keeps the admitted generic CPU SIMD `repulsionPower = 9` path reachable for parity and performance comparison

Plain-C reference override:
- `GMX_DISABLE_SIMD_KERNELS=1`

The three runtime shapes are therefore distinguishable in logs:
- plain-C reference
- generic admitted SIMD baseline
- specialized admitted SIMD path

## Affected File / Function Inventory

- `src/gromacs/mdtypes/interaction_const.h`
  adds runtime state for the repulsion-power-9 SIMD specialization choice
- `src/gromacs/mdlib/forcerec.cpp`
  selects specialized vs generic admitted SIMD route and emits runtime log markers
- `src/gromacs/nbnxm/simd_lennardjones_functions.h`
  implements the exact `r^-9` SIMD multiplication chain inside the admitted `C6/C12` path
- `src/gromacs/nbnxm/tests/pcff_class2_nonbonded.cpp`
  deterministic generic-SIMD vs specialized-SIMD parity coverage
- `src/programs/mdrun/tests/pcff_short_md.cpp`
  single-point and exact-r-RESPA runtime parity, plus runtime-selection log checks

## Validation Plan

Deterministic fixtures:
- generic SIMD vs specialized SIMD for two-atom `9-6` + Coulomb-table coupling
- generic SIMD vs specialized SIMD for the small oligomer no-pairs scaffold

Medium integration fixtures:
- single-point generic SIMD vs specialized SIMD on `small_oligomer`
- single-point generic SIMD vs specialized SIMD on `small_salt_polymer_box`
- exact-r-RESPA generic SIMD vs specialized SIMD on the same fixtures
- specialized default restart continuity on the same exact-r-RESPA fixtures

Compared outputs:
- step-0 energy breakdown
- force components
- virial / pressure tensor components where available
- exact-r-RESPA total and per-level force dumps
- exact-r-RESPA trace files, observables, and final snapshot continuity

## Performance Characterization Plan

Three-way benchmark:
1. plain-C reference
2. generic admitted SIMD baseline
3. specialized admitted SIMD path

Host-local driver:
- `tools/pcff_respa_parity/bench_repulsion_power_9_simd_specialized_cpu.py`

Required framing:
- compare specialized SIMD directly against the generic admitted SIMD baseline
- keep host and runtime envelope explicit
- do not widen the claim to GPU, hybrid, LJ-PME, or generic CPU scaling without new evidence
