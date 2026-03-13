# M7 Validation Report

## Milestone Result

M7 is closed for the intended CUDA scope:

- CUDA support exists for the PCFF/Class2 nonbonded real-space path
- CPU-vs-GPU single-point parity is checked on representative frames
- the validated path now uses actual GPU PME, not CPU PME fallback

What this milestone does **not** claim:

- bonded Class2 terms run on the GPU
- GPU-resident execution is complete

## Implemented Scope

The new GPU work is intentionally narrow.

Implemented:

- CUDA nonbonded real-space support for non-`12` repulsion powers, including PCFF `9-6`
- propagation of `repulsionPower` into GPU kernel parameters
- exact `r^-9` evaluation in the CUDA real-space kernel
- explicit CPU-vs-GPU regression through the production `mdrun` path
- GPU performance smoke coverage for repeated single-point launches

Still intentionally excluded:

- GPU bonded Class2 kernels
- any change to M6 exact `lammps-respa` behavior

## Architectural Entry Points

### CUDA nonbonded kernel data

- [gpu_types_common.h](/home/user/바탕화면/gromacs/src/gromacs/nbnxm/gpu_types_common.h)
  adds `repulsionPower` and `inverseRepulsionPower` to `NBParamGpu`.
- [nbnxm_gpu_data_mgmt.cpp](/home/user/바탕화면/gromacs/src/gromacs/nbnxm/nbnxm_gpu_data_mgmt.cpp)
  populates those fields from `interaction_const` and rejects unsupported `ForceSwitch` + non-`12` combinations on the GPU.

### CUDA real-space evaluation

- [nbnxm_cuda_kernel.cuh](/home/user/바탕화면/gromacs/src/gromacs/nbnxm/cuda/nbnxm_cuda_kernel.cuh)
  evaluates the repulsive term as `r^-12` only for the legacy fast path and falls back to `powf(inv_r, repulsionPower)` otherwise.
- [kernel_gpu_ref.cpp](/home/user/바탕화면/gromacs/src/gromacs/nbnxm/kernels_reference/kernel_gpu_ref.cpp)
  mirrors the same generic repulsion-power semantics in the GPU reference kernel.

### Validation path

- [pcff_short_md.cpp](/home/user/바탕화면/gromacs/src/programs/mdrun/tests/pcff_short_md.cpp)
  adds:
  - `PcffGpuSinglePointParityTest`
  - `PcffGpuPerfSmokeTest`
  - GPU capability gating
  - unique output-prefix handling for CPU/GPU dual-run tests

The unique output-prefix helper matters. Without it, two `SimulationRunner` instances in one test can reuse the same temporary filenames and overwrite each other's `.edr` / `.trr` outputs. That would create a false-green parity result.

## Validation Strategy

The acceptance test is the production `mdrun` path, not a handcrafted low-level harness.

That choice is deliberate:

- it exercises the same preprocessing, pairlist, PME coupling, and energy/force reporting path that users will actually run
- it reduces the chance of validating a synthetic test harness instead of the real runtime path

The GPU parity tests force this execution split:

- `-nb gpu`
- `-pme gpu`
- `-pmefft gpu`
- `-bonded cpu`
- `-update cpu`
- `-notunepme`

This is the right M7 cut:

- it validates the CUDA real-space nonbonded path first
- it validates the coupled short-range + reciprocal GPU execution path on a single rank
- it still avoids over-claiming GPU-resident bonded/update execution that was not tested

## Tests Run

Executed in the CUDA build under [build-cuda](/home/user/바탕화면/gromacs/build-cuda):

- `./build-cuda/bin/mdrun-non-integrator-test --gtest_filter='PcffGpuSinglePointParity*:*PcffGpuPerfSmokeTest*'`
- `./build-cuda/bin/nbnxm-test --gtest_filter='PcffClass2NonbondedCurveTest.*'`
- `./build-cuda/bin/mdrun-non-integrator-test --gtest_filter='PcffSinglePointParity*'`

Results:

- `PcffGpuSinglePointParity*`: 2 passed
- `PcffGpuPerfSmokeTest*`: 1 passed
- `PcffClass2NonbondedCurveTest.*`: 9 passed
- `PcffSinglePointParity*`: 4 passed

## Numerical Checks

### CPU-vs-GPU single-point parity

Representative systems:

- `small_oligomer`
- `small_salt_polymer_box`

Compared quantities:

- `bond`
- `angle`
- `dihedral`
- `lj14`
- `ljsr`
- `coul14`
- `coulsr`
- `coul_recip`
- `potential_total`
- per-atom force components from `.trr`

Acceptance tolerances enforced in [pcff_short_md.cpp](/home/user/바탕화면/gromacs/src/programs/mdrun/tests/pcff_short_md.cpp):

- energy breakdown terms: `8e-3 kcal/mol`
- force component parity: `6e-2 kJ/mol/nm`

Both representative systems passed within those explicit tolerances.

### CPU regression guard in CUDA build

The existing CPU single-point golden tests also pass in the CUDA-enabled build:

- `PcffSinglePointParity*`: 4 passed

This matters because M7 must not silently degrade the already-frozen CPU PCFF path.

## Known Limits

1. The automated parity evidence covers the single-rank `-nb gpu -pme gpu -pmefft gpu` path. It does **not** validate separate PME ranks or GPU PME decomposition.
2. `ForceSwitch` with non-`12` repulsion power is rejected on the GPU path. That is an explicit unsupported mode, not a silent approximation.
3. The current tests validate representative frames, not long trajectories.
4. M7 adds no GPU support for bonded Class2 terms.

## Readiness Assessment

Ready now:

- use CUDA for the PCFF/Class2 real-space nonbonded path
- use the new CPU-vs-GPU `mdrun` parity tests as a gate for later GPU work

Not ready to claim:

- full GPU-resident PCFF execution
- bonded GPU parity
- multi-rank GPU PME decomposition parity
