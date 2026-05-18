# M9 Validation Report

## Milestone Result

M9 is closed as an evidence-driven evaluation milestone, but **no new PCFF bonded GPU kernels were added**.

That is the correct result for the current code base.

The decision basis is:

- the existing GPU bonded path is not semantically safe for PCFF/class2 listed `1-4` interactions
- the measured upside from offloading class2 bonded work is modest even under an unrealistic perfect-offload assumption
- adding four new bonded GPU kernels plus parity maintenance is not justified by the current evidence

## Scope Evaluated

M9 asked whether bonded GPU offload is worth adding for:

- `BondClass2`
- `AngleClass2`
- `ImproperClass2`
- `DihedralClass2`

The scope did **not** include:

- changing nonbonded PCFF GPU physics semantics
- weakening CPU parity gates
- porting all bonded interactions regardless of payoff

## Blocking Correctness Issue Found First

Before discussing performance, the existing GPU bonded/listed path had to be checked for PCFF correctness.

That check found a real semantic blocker:

- the current GPU listed-pair kernel hardcodes `12-6` force/energy evaluation:
  [listed_forces_gpu_internal_shared.h](../src/gromacs/listed_forces/listed_forces_gpu_internal_shared.h#L751)
- the CPU listed-pair path already uses the general repulsion power from `fr->ic->vdw.repulsionPower`:
  [pairs.cpp](../src/gromacs/listed_forces/pairs.cpp#L656)

PCFF/class2 uses sixth-power mixing and `9-6` listed `1-4` interactions, so the existing `-bonded gpu` path would have been unsafe for PCFF if left unchecked.

The milestone therefore adds an explicit runtime gate:

- [listed_forces_gpu_impl.cpp](../src/gromacs/listed_forces/listed_forces_gpu_impl.cpp#L129)

That gate rejects `-bonded gpu` when `reppow != 12` and the topology contains GPU-capable listed interactions.

## What Was Added

M9 adds three things:

1. a correctness guard that rejects unsafe bonded-GPU use for PCFF/class2:
   [listed_forces_gpu_impl.cpp](../src/gromacs/listed_forces/listed_forces_gpu_impl.cpp#L145)
2. unit tests that prove the guard behaves as intended:
   [listed_forces_gpu.cpp](../src/gromacs/listed_forces/tests/listed_forces_gpu.cpp#L59)
3. a benchmark-oriented smoke test that measures whether class2 bonded work is large enough to justify a GPU port:
   [pcff_short_md.cpp](../src/programs/mdrun/tests/pcff_short_md.cpp#L1895)

No class2 bonded CUDA kernels were added in M9.

## Tests Run

Executed in the CUDA build at [build-cuda](../build-cuda):

- `./build-cuda/bin/listed_forces-test --gtest_filter='ListedForcesGpuInputSupportTest.*'`
- `./build-cuda/bin/mdrun-non-integrator-test --gtest_filter='PcffGpuSinglePointParity*:*PcffGpuResidentParity*:*PcffGpuPerfSmokeTest.ReplicatedSaltBoxBenchmarkReportsCpuBondedShare'`
- `./build-cuda/bin/mdrun-non-integrator-test --gtest_filter='PcffGpuSinglePointParity*:*PcffGpuResidentParity*:*PcffGpuPerfSmokeTest*'`

Observed results:

- `ListedForcesGpuInputSupportTest.*`: `2 passed`
- `PcffGpuSinglePointParity*:*PcffGpuResidentParity*:*PcffGpuPerfSmokeTest.ReplicatedSaltBoxBenchmarkReportsCpuBondedShare`: `5 passed`
- `PcffGpuSinglePointParity*:*PcffGpuResidentParity*:*PcffGpuPerfSmokeTest*`: `6 passed`

The guard tests are:

- `RejectsSixthPowerRepulsionForBondedGpu`
- `AcceptsTwelveSixListedPairsForBondedGpu`

The benchmark smoke test is:

- `ReplicatedSaltBoxBenchmarkReportsCpuBondedShare`

The existing nonbonded GPU parity gates also continued to pass:

- [PcffGpuSinglePointParityTest](../src/programs/mdrun/tests/pcff_short_md.cpp#L1805)
- [PcffGpuResidentParityTest](../src/programs/mdrun/tests/pcff_short_md.cpp#L1937)

That matters because M9 must not silently damage the already validated M7/M8 GPU path.

## Benchmark Method

The benchmark fixture is generated from the frozen salt/polymer system and replicated `8 x 8 x 8`:

- input builder:
  [writeReplicatedSaltBoxFixture](../src/programs/mdrun/tests/pcff_short_md.cpp#L1495)
- benchmark MDP:
  [makeGpuBondedBenchmarkMdp](../src/programs/mdrun/tests/pcff_short_md.cpp#L1608)
- benchmark driver:
  [benchmarkCpuBondedOnReplicatedSaltBox](../src/programs/mdrun/tests/pcff_short_md.cpp#L1643)

Validated runtime mode:

- `-nb gpu`
- `-pme gpu`
- `-pmefft gpu`
- `-bonded cpu`
- `-update gpu`

The test extracts:

- total `Force` wall time from the mdrun wallcycle table
- bonded and `1-4` flop shares from the `M E G A - F L O P S A C C O U N T I N G` table

Extraction helpers:

- [extractWallcycleSeconds](../src/programs/mdrun/tests/pcff_short_md.cpp#L1416)
- [extractMegaFlopsPercent](../src/programs/mdrun/tests/pcff_short_md.cpp#L1444)

## Measured Result

The executed benchmark reported:

- atoms: `5120`
- replicas: `512`
- `Force` wall time: about `0.14 s` to `0.17 s` across repeated smoke executions
- `Bonds`: `3.2 %`
- `Angles`: `7.9 %`
- `Propers`: `9.0 %`
- `Impropers`: `0.0 %`
- `1,4 nonbonded interactions`: `3.5 %`
- class2 bonded subtotal: `20.1 %`
- class2 bonded + listed `1-4` subtotal: `23.6 %`
- perfect-offload upper bound from class2 bonded only: `1.25156 x`
- perfect-offload upper bound from class2 bonded + listed `1-4`: `1.3089 x`

Those numbers are emitted directly by:

- [PcffGpuPerfSmokeTest.ReplicatedSaltBoxBenchmarkReportsCpuBondedShare](../src/programs/mdrun/tests/pcff_short_md.cpp#L1895)

## Interpretation

This does **not** justify implementing PCFF bonded GPU kernels yet.

Reasons:

1. the measured upside is limited even before accounting for kernel launch overhead, extra data staging, maintenance cost, and parity burden
2. the broader listed-GPU path is already blocked by incorrect `9-6` listed-pair semantics
3. porting `BondClass2`, `AngleClass2`, `ImproperClass2`, and full `DihedralClass2` would be a large maintenance surface for a small projected whole-run speedup

If the `1-4` GPU path were also generalized to `9-6`, the theoretical upper bound still rises only from about `1.25x` to about `1.31x` on this representative benchmark.

That is not enough to justify the implementation cost today.

## Recommended Usage Conditions

Recommended now:

- keep PCFF/class2 bonded terms on the CPU
- continue using the validated GPU path for nonbonded real-space + PME
- reject `-bonded gpu` for sixth-power / `reppow = 9` systems

Do not enable bonded GPU for PCFF/class2 unless all of the following become true:

1. the listed GPU path is generalized to exact `9-6` `1-4` semantics
2. new profiling on larger representative systems shows a materially larger bonded share than the current benchmark
3. per-term CPU-vs-GPU parity tests exist for each ported class2 bonded interaction

## What M9 Did Not Do

M9 did **not** add:

- `BondClass2` GPU kernels
- `AngleClass2` GPU kernels
- `ImproperClass2` GPU kernels
- `DihedralClass2` GPU kernels

This is intentional, not missing work.

## Readiness Assessment

Ready now:

- treat M9 as a completed evaluation milestone
- keep the new guard in place so unsafe PCFF bonded-GPU runs fail early
- keep using M7/M8 GPU parity tests as the regression gate for the supported nonbonded GPU path

Not ready now:

- optional bonded GPU offload for PCFF/class2
- any claim that bonded GPU offload is performance-critical for the current representative systems
