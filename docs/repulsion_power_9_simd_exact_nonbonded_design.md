# Repulsion-Power-9 SIMD Exact Nonbonded Design Note

## Scope Freeze

This capability is intentionally narrow.

In scope:
- CPU `nbnxm` SIMD short-range nonbonded execution for PCFF/Class2 `9-6` (`repulsionPower = 9`) systems
- exact `LJ (SR)` force and energy semantics matching the existing plain-C reference kernels
- coupled short-range Coulomb paths already supported by the same CPU `nbnxm` kernels
- exclusion handling in the same SIMD nonbonded path
- exact r-RESPA CPU integration that consumes the admitted SIMD nonbonded path

Explicitly out of scope:
- Buckingham
- free-energy perturbation with non-`12` repulsion
- dispersion correction with non-`12` repulsion
- LJ-PME admission for this SIMD capability track
- GPU or hybrid admission claims derived from CPU SIMD success

## Admission Rule

`repulsionPower = 9` is admitted to CPU SIMD only when all of the following hold:

- `vdwtype = Cut-off`
- `vdw-modifier = none` or `Potential-shift`
- Buckingham is not active
- dispersion correction is off
- free-energy perturbation is off

All other non-`12` runtime shapes remain on the existing plain-C reference path or separate runtime guards.

## Affected Runtime / Kernel Files

- `src/gromacs/mdlib/forcerec.cpp`
  runtime admission policy for non-`12` repulsion powers
- `src/gromacs/nbnxm/nbnxm_setup.cpp`
  CPU `nbnxm` parameter-combination selection for non-`12` repulsion
- `src/gromacs/nbnxm/simd_lennardjones_functions.h`
  SIMD `r^-6` and `r^-repulsionPower` evaluation
- `src/gromacs/nbnxm/simd_kernel.h`
  CPU SIMD short-range nonbonded kernel entry point
- `src/gromacs/nbnxm/tests/pcff_class2_nonbonded.cpp`
  small deterministic fixture coverage across plain-C and SIMD CPU kernels
- `src/programs/mdrun/tests/pcff_short_md.cpp`
  medium-scale single-point and exact-r-RESPA integration validation

## Validation Plan

Small deterministic fixtures:
- two-atom `9-6` energy/force curve
- exclusion suppression
- Coulomb-table coupling with `9-6`
- CPU kernel cross-checks across `Cpu1x1_PlainC`, `Cpu4x4_PlainC`, and available SIMD kernels

Medium integration fixtures:
- single-point plain-C vs admitted CPU SIMD comparison on `small_oligomer` and `small_salt_polymer_box`
- exact r-RESPA plain-C vs admitted CPU SIMD comparison on the same audited fixtures

Compared outputs:
- step-0 energy breakdown
- force components
- virial-pressure tensor components
- exact r-RESPA per-level force dumps and final trajectory continuity

Host-local performance characterization:
- `tools/pcff_respa_parity/bench_repulsion_power_9_simd_exact_cpu.py`
- `docs/repulsion_power_9_simd_exact_nonbonded_perf.md`

## Capability Delta

Old boundary:
- repulsion-power-9 CPU runtime forcibly disabled SIMD nonbonded kernels and always used plain-C reference kernels

New boundary:
- repulsion-power-9 CPU runtime admits a real SIMD nonbonded path for the validated short-range `9-6` scope above
- unsupported chemistry and long-range modes remain outside the claim
