# M8 Validation Report

## Milestone Result

M8 is closed for the intended scope:

- the PCFF/Class2 nonbonded GPU path is validated under a GPU-resident-compatible runtime pattern
- correctness is preserved when enabling GPU PME, GPU update, and GPU X/F buffer ops in the supported subset
- no physics semantics were changed

What M8 does **not** claim:

- full no-copy execution
- GPU bonded Class2 support
- MTS / exact `lammps-respa` compatibility with GPU-resident execution
- multi-rank GPU-resident parity

## Scope Actually Validated

The validated M8 runtime subset is:

- single rank
- `integrator = md`
- `-nb gpu`
- `-pme gpu`
- `-pmefft gpu`
- `-bonded cpu`
- `-update gpu`
- no MTS
- no constraints

The `integrator = md` restriction is not a PCFF choice. It is an existing GPU-update runtime restriction in
[decidegpuusage.cpp](../src/gromacs/taskassignment/decidegpuusage.cpp#L779).

## Why This Subset Matters

M8 is about dataflow correctness, not new force kernels.

The important requirement is that the PCFF `9-6` nonbonded path does not break when the surrounding runtime keeps more state on the GPU:

- coordinates updated on GPU
- PME executed on GPU
- nonbonded coordinate transforms executed from device buffers
- nonbonded force reduction executed on GPU on non-virial steps

That is the maximum useful subset before bonded GPU work exists.

## Implementation Changes

No production CUDA kernel changes were required beyond M7.

The M8 code changes are targeted validation and runtime-selection coverage in
[pcff_short_md.cpp](../src/programs/mdrun/tests/pcff_short_md.cpp):

- `PcffGpuResidentParityTest`
- `getGpuResidentSkipMessages()`
- `makeGpuResidentNveMdp()`
- `makeGpuResidentShortMdCaller()`
- short-MD CPU-vs-GPU observable and final-state comparison helpers

This is the correct cut. There was no evidence that the PCFF GPU kernels themselves needed more production changes to participate in the existing GPU-resident machinery.

## Dataflow Basis

The relevant runtime rules are:

- GPU X/F buffer ops are only allowed when nonbonded is offloaded and MTS is off:
  [decidesimulationworkload.cpp](../src/gromacs/taskassignment/decidesimulationworkload.cpp#L160)
- GPU update or direct GPU communication requires those buffer ops:
  [decidesimulationworkload.cpp](../src/gromacs/taskassignment/decidesimulationworkload.cpp#L166)
- per-step GPU F buffer ops are disabled on virial steps:
  [decidesimulationworkload.cpp](../src/gromacs/taskassignment/decidesimulationworkload.cpp#L301)
- GPU PME force reduction depends on GPU F buffer ops being active:
  [decidesimulationworkload.cpp](../src/gromacs/taskassignment/decidesimulationworkload.cpp#L304)
- with GPU update and CPU local force work, coordinates are still copied back to the CPU:
  [sim_util.cpp](../src/gromacs/mdlib/sim_util.cpp#L2174)
- when GPU update is active and the step is not a search step, coordinates stay on device and are consumed through device events rather than a fresh H2D copy:
  [sim_util.cpp](../src/gromacs/mdlib/sim_util.cpp#L2187)
- when GPU X buffer ops are active, nonbonded coordinate conversion uses device coordinates directly:
  [sim_util.cpp](../src/gromacs/mdlib/sim_util.cpp#L2267)

## Test Design

The resident-style M8 short-MD test intentionally uses:

- `integrator = md`
- `nsteps = 20`
- `nstcalcenergy = 20`
- `nstenergy = 20`
- `nstlog = 20`
- `nstxout = 20`
- `nstvout = 20`
- `nstfout = 0`

Reason:

- `nstcalcenergy = 1` would compute virial every step
- that would force `useGpuFBufferOps = false` every step
- then the test would no longer exercise the intended resident-style force reduction path

This choice is documented inline in
[pcff_short_md.cpp](../src/programs/mdrun/tests/pcff_short_md.cpp#L1280).

## Tests Run

Executed in the CUDA build:

- `./build-cuda/bin/mdrun-non-integrator-test --gtest_filter='PcffGpuResidentParity*'`
- `./build-cuda/bin/mdrun-non-integrator-test --gtest_filter='PcffGpuSinglePointParity*:*PcffGpuPerfSmokeTest*:*PcffGpuResidentParity*'`
- `./build-cuda/bin/mdrun-non-integrator-test --gtest_filter='PcffSinglePointParity*'`

Results:

- `PcffGpuResidentParity*`: 2 passed
- combined `PcffGpuSinglePointParity*:*PcffGpuPerfSmokeTest*:*PcffGpuResidentParity*`: 5 passed
- `PcffSinglePointParity*`: 4 passed

## Numerical Agreement

The M8 resident test compares CPU vs GPU for:

- `step0_potential_kcal_mol`
- `initial_total_kcal_mol`
- `final_total_kcal_mol`
- `total_energy_drift_abs_kcal_mol`
- `total_energy_span_kcal_mol`
- polymer end-to-end distance
- polymer radius of gyration
- ion distance when present
- final coordinate RMS / max difference
- final velocity RMS / max difference

Acceptance tolerances in
[pcff_short_md.cpp](../src/programs/mdrun/tests/pcff_short_md.cpp):

- scalar energy observables: `3e-2 kcal/mol`
- structural observables: `2e-4 nm`
- final coordinate RMS: `2e-4 nm`
- final coordinate max: `6e-4 nm`
- final velocity RMS: `1.5e-3 nm/ps`
- final velocity max: `5e-3 nm/ps`

Representative systems:

- `small_oligomer`
- `small_salt_polymer_box`

Both passed.

The GPU log is also checked for:

- `PP task will update and constrain coordinates on the GPU`
- `PME tasks will do all aspects on the GPU`

So this is not a fake resident test running on the CPU update path.

## Scope Limits

1. Bonded Class2 remains on the CPU, so coordinate staging back to the host still occurs when CPU local force work exists.
2. M8 proves compatibility with the existing GPU-resident machinery. It does not remove all host-device transfers.
3. `integrator = md-vv` is not covered in this resident path because GPU update currently rejects it.
4. MTS and exact `lammps-respa` remain outside M8.

## Readiness Assessment

Ready now:

- treat the PCFF nonbonded CUDA path as compatible with the existing single-rank GPU-resident update/PME/buffer-op model
- use the new resident-parity test as a gate before optional bonded GPU work

Not ready to claim:

- full bonded + nonbonded GPU residency
- multi-rank GPU-resident parity
- exact-MTS GPU residency
