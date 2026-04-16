# Repulsion-Power-9 Exact-r-RESPA CPU Patch Performance Envelope

## Bottom Line

Once the wrong-benchmark issue is removed, the real exact-`r-RESPA` CPU optimization target is
[`computeExactRespaNonbondedCpu()`](/home/kiket/Desktop/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp:3498).

That path now has a repulsion-power-9 specialization:

- generic baseline:
  - `std::pow(rinv, repulsionPower)` for non-12 repulsion
- specialized path:
  - exact multiplication chain for `r^-9`

The specialization is runtime-distinguishable through:

- `GMX_DISABLE_REPULSION_POWER_9_EXACT_RESPA_CPU_SPECIALIZATION=1`
- exact-`r-RESPA` CPU patch log markers in the first step

On the larger charged audited fixture, the specialized exact-`r-RESPA` CPU patch path is modestly faster
than the generic patch baseline at `ntomp=1,2,6`.

## Driver

Benchmark script:

- [`tools/pcff_respa_parity/bench_repulsion_power_9_exact_respa_cpu_patch.py`](/home/kiket/Desktop/test/GROMACS_PCFF/tools/pcff_respa_parity/bench_repulsion_power_9_exact_respa_cpu_patch.py)

Executed output bundle:

- [`output/repulsion_power_9_exact_respa_cpu_patch_perf/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_exact_respa_cpu_patch_perf/summary.md)
- [`output/repulsion_power_9_exact_respa_cpu_patch_perf/summary.json`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_exact_respa_cpu_patch_perf/summary.json)

Audited run settings:

- host: `AMD Ryzen 9 9900X 12-Core Processor`
- binary: [`build/bin/gmx`](/home/kiket/Desktop/test/GROMACS_PCFF/build/bin/gmx)
- exact `r-RESPA` CPU pair splitting
- `pin=on`
- `steps=200`
- `repeats=1`
- audited fixtures:
  - `small_oligomer`
  - `small_salt_polymer_box`
  - `gate_h_dense_salt_polymer_2x2x2`

## Implementation Boundary

The optimized code path is the exact CPU patch path, not the admitted SIMD short-range kernel.

Relevant implementation points:

- path dispatch:
  - [`src/gromacs/mdlib/sim_util.cpp`](/home/kiket/Desktop/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp:8189)
- generic baseline:
  - [`src/gromacs/mdlib/sim_util.cpp`](/home/kiket/Desktop/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp:4325)
- specialization toggle:
  - [`src/gromacs/mdlib/sim_util.cpp`](/home/kiket/Desktop/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp:186)
- step-0 log markers:
  - [`src/gromacs/mdlib/sim_util.cpp`](/home/kiket/Desktop/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp:8203)

## Exactness Basis

Specialization parity was revalidated before benchmarking:

- single-point generic SIMD vs specialized SIMD:
  - `./build/bin/mdrun-non-integrator-test --gtest_filter='PcffSinglePointParity/PcffSinglePointParityTest.CpuSimdPower9SpecializedMatchesGenericBaseline/*'`
- exact-`r-RESPA` CPU patch generic vs specialized:
  - `./build/bin/mdrun-non-integrator-test --gtest_filter='PcffRespaObservableDump/PcffRespaObservableDumpTest.ExactRespaCpuPower9PatchSpecializedMatchesGenericBaseline/*'`
- exact-`r-RESPA` plain-C vs SIMD admission:
  - `./build/bin/mdrun-non-integrator-test --gtest_filter='PcffRespaObservableDump/PcffRespaObservableDumpTest.ExactRespaCpuSimdMatchesPlainCReference/*'`
- restart continuity:
  - `./build/bin/mdrun-non-integrator-test --gtest_filter='PcffRespaRestartParity/PcffRespaRestartParityTest.RestartFromCheckpointMatchesFullExactRun/*'`

## Observed Host-Local Envelope

### `gate_h_dense_salt_polymer_2x2x2`

| ntomp | generic ns/day | specialized ns/day | specialized/generic wall | generic Force s | specialized Force s | specialized/generic Force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.704 | 3.753 | 1.013 | 2.024000 | 1.980000 | 1.022 |
| 2 | 3.644 | 3.857 | 1.058 | 2.107000 | 1.969000 | 1.070 |
| 6 | 3.789 | 4.039 | 1.066 | 2.087000 | 1.944000 | 1.074 |

`Update` also moves in the same direction on this larger charged fixture:

| ntomp | generic Update s | specialized Update s | specialized/generic Update |
| --- | ---: | ---: | ---: |
| 1 | 1.162000 | 1.140000 | 1.019 |
| 2 | 1.173000 | 1.110000 | 1.057 |
| 6 | 1.140000 | 1.059000 | 1.076 |

### Tiny audited fixtures

The tiny `6`-atom and `10`-atom fixtures remain noisy and mixed:

- `small_oligomer`
  - `ntomp=2` regresses
  - `ntomp=6` is near parity
- `small_salt_polymer_box`
  - `ntomp=2` regresses
  - `ntomp=6` is a small win

Those shapes are still valid exactness gates, but they remain weak standalone performance evidence.

## Interpretation

What the current evidence supports:

- the exact-`r-RESPA` CPU patch specialization is real and runtime-distinguishable
- the larger charged audited fixture shows a small but consistent host-local win for the specialized
  exact-`r-RESPA` CPU patch path
- optimizing `computeExactRespaNonbondedCpu()` was the right next target after the wrong-benchmark
  diagnosis

What the current evidence does not support:

- a broad CPU scaling claim across hosts
- a claim that the tiny audited fixtures are meaningful performance probes
- any implication about the admitted specialized SIMD microkernel wall-clock envelope
- any implication about GPU, hybrid, or LJ-PME

## Claim Boundary

Allowed:

- `On the audited Ryzen 9 9900X host, the specialized exact-r-RESPA CPU repulsion-power-9 patch path modestly improves the generic exact-r-RESPA CPU patch baseline on the larger charged audited fixture.`
- `The exact-r-RESPA CPU patch path is now a separate optimized runtime surface from the admitted SIMD short-range kernel path.`

Not allowed:

- `exact-r-RESPA CPU pair splitting is broadly faster on CPU`
- `the tiny fixtures establish a stable scaling story`
- any implication about GPU, hybrid, or multi-host portability
