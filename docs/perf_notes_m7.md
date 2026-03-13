# M7 Performance Notes

## What Was Measured

M7 adds a smoke test, not a benchmark.

The performance-oriented test is
[PcffGpuPerfSmokeTest](/home/user/바탕화면/gromacs/src/programs/mdrun/tests/pcff_short_md.cpp)
and it runs the `small_oligomer` single-point case on the GPU three times.

The pass condition is intentionally weak:

- elapsed wall-clock must be finite
- elapsed wall-clock must be positive

This is enough to catch obvious failures such as:

- GPU path not launching
- pathological stalls
- runtime returning invalid timing data

It is **not** enough to claim speedup.

## Why The Smoke Test Is Deliberately Weak

The measured runtime includes more than kernel execution:

- `grompp`-generated runtime setup
- `mdrun` startup
- file I/O
- energy and trajectory output
- GPU initialization overhead

The M7 parity tests also force:

- `-pme gpu`
- `-pmefft gpu`
- `-bonded cpu`
- `-update cpu`

So the observed runtime is not a pure measure of the short-range CUDA kernel alone. It includes GPU PME as well.

## Environment Used

Validation was run in the CUDA build at [build-cuda](/home/user/바탕화면/gromacs/build-cuda) with:

- `GMX_GPU=CUDA`
- release build
- single precision
- 1 MPI rank
- 1 OpenMP thread

The machine has an NVIDIA GPU available and the runtime logs showed:

- `1 GPU selected for this run`
- `PP tasks will do non-perturbed short-ranged interactions on the GPU`
- `PME tasks will do all aspects on the GPU`

## Observed Runtime Scale

During the executed single-point parity and regression runs, the `mdrun` logs reported wall times on the order of:

- about `0.011 s` for the `small_salt_polymer_box` GPU single-point run
- about `0.014 s` to `0.015 s` for the representative single-point runs in the CUDA build

Do not over-interpret those numbers.

They are startup-inclusive, machine-specific, and too small to be a stable benchmark.

## Recommended Use

Use the M7 performance smoke test only as a regression sentinel:

- it should remain finite
- it should remain obviously non-pathological
- it should continue to exercise the GPU real-space + PME path

Do **not** use it for:

- CPU-vs-GPU speedup claims
- cross-machine comparisons
- tuning decisions
- publication-quality performance numbers

## What Is Still Missing

Before making stronger GPU performance claims, the project still needs:

1. a dedicated benchmark harness with fixed warmup and repeated steady-state sampling
2. separate reporting for GPU PME vs GPU real-space time contributions
3. larger representative systems
4. profiling under the eventual GPU-resident execution path
