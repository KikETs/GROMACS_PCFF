# Repulsion-Power-9 `ntomp=6` Regression Measurement Protocol

## Question Under Audit

Why does the specialized exact CPU SIMD repulsion-power-9 path show a host-local win at some thread counts but regress against the admitted generic SIMD baseline at `ntomp=6` on the audited host?

This note freezes the measurement stack used for the host-local diagnosis. It does not reopen admission, exactness, or public-capability scope.

## Host And Runtime

- Host: `AMD Ryzen 9 9900X 12-Core Processor`
- Logical CPUs: `24`
- Physical cores: `12`
- L3 cache: `64 MiB (2 instances)`
- GROMACS binary: [`build/bin/gmx`](/home/kiket/Desktop/test/GROMACS_PCFF/build/bin/gmx)
- Subcounter diagnostic build: [`build_subcounters/bin/gmx`](/home/kiket/Desktop/test/GROMACS_PCFF/build_subcounters/bin/gmx)
  - configured with `GMX_CYCLE_SUBCOUNTERS=ON`
- Exact runtime shape: CPU-only `md-vv` exact `r-RESPA` with `-nb cpu -pme cpu -bonded cpu -update cpu -notunepme -ntmpi 1`

## Controlled Axes

- Modes:
  - `generic`: admitted exact SIMD baseline with `GMX_DISABLE_REPULSION_POWER_9_SIMD_SPECIALIZATION=1`
  - `specialized`: default specialized exact repulsion-power-9 SIMD path
- Small audited fixtures:
  - `small_oligomer` from `tests/reference_results/m6_respa/small_oligomer`
  - `small_salt_polymer_box` from `tests/reference_results/m6_respa/small_salt_polymer_box`
- Larger charged relevance fixture:
  - `gate_h_dense_salt_polymer_2x2x2` from `tests/reference_results/gate_h_fixture_scaffold/gate_h_dense_salt_polymer_2x2x2/generated`
- Thread counts:
  - small fixtures: `ntomp=1,2,4,6,8,12`
  - larger charged fixture: `ntomp=2,6`
- Pinning:
  - `pin=on`
  - `pin=off`

## Metrics Collected

- Wall-clock throughput from the `Performance:` line: `ns/day`
- Wallcycle entries from `REAL CYCLE AND TIME ACCOUNTING`:
  - `Domain decomp.`
  - `Neighbor search`
  - `Force`
  - `PME mesh`
  - `NB X/F buffer ops.`
  - `Update`
  - `Total`
- Wallcycle subcounters from `Breakdown of PP / PME activities` in the subcounter build:
  - `NB F kernel`
  - `NB F buffer ops.`
  - `NB X buffer ops.`
  - `Listed buffer ops.`
  - `Clear force buffer`
  - `Bonded F`
- Affinity report when `GMX_REPORT_CPU_AFFINITY=1` was enabled for the pinned probe

## Fixed Run Settings

- Small-fixture matrix:
  - `steps=8000`
  - `repeats=2`
- Small-fixture repeat-depth probe:
  - `steps=8000`
  - `repeats=6`
  - `ntomp=2,6`
- Larger charged relevance check:
  - `steps=1000`
  - `repeats=2`
  - `ntomp=2,6`

## Tooling

- Driver script:
  - [`tools/pcff_respa_parity/bench_repulsion_power_9_simd_specialized_cpu.py`](/home/kiket/Desktop/test/GROMACS_PCFF/tools/pcff_respa_parity/bench_repulsion_power_9_simd_specialized_cpu.py)
- Summary outputs:
  - [`output/repulsion_power_9_scaling_diagnostic/pin_on/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/pin_on/summary.md)
  - [`output/repulsion_power_9_scaling_diagnostic/pin_off/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/pin_off/summary.md)
  - [`output/repulsion_power_9_scaling_diagnostic/pin_on_repeatdepth_2_6/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/pin_on_repeatdepth_2_6/summary.md)
  - [`output/repulsion_power_9_scaling_diagnostic/gate_h_pin_on/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/gate_h_pin_on/summary.md)
- Subcounter probe logs:
  - [`output/repulsion_power_9_subcounter_probe/small_oligomer/ntomp6/generic/run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_subcounter_probe/small_oligomer/ntomp6/generic/run.log)
  - [`output/repulsion_power_9_subcounter_probe/small_salt_polymer_box/ntomp6/generic/run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_subcounter_probe/small_salt_polymer_box/ntomp6/generic/run.log)
  - [`output/repulsion_power_9_subcounter_probe/gate_h_dense_salt_polymer_2x2x2/ntomp6/generic/run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_subcounter_probe/gate_h_dense_salt_polymer_2x2x2/ntomp6/generic/run.log)

## Affinity / Topology Basis

- `pin=on`, `ntomp=6` probe used CPUs `0-5` on the audited host:
  - [`output/ntomp6_affinity_probe/generic/run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/ntomp6_affinity_probe/generic/run.log:469)
  - [`output/ntomp6_affinity_probe/specialized/run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/ntomp6_affinity_probe/specialized/run.log:469)
- Host topology shows two L3 instances, and CPUs `0-5` map to the same L3 slice.

## Measurement Limits

- `perf stat` and PMU counters were unavailable on this host because `perf_event_paranoid=4`.
- The current wallcycle report does not expose a bonded-only CPU timer for this exact runtime path.
- `Force` is a kernel-adjacent proxy, not an isolated nonbonded-only microkernel timer.
- The audited performance fixtures all use exact `r-RESPA` pair splitting: `mts = yes`, `mts-mode = lammps-respa`, `mts-respa-inner/middle/outer-*`.
- For that runtime shape, the CPU exact-nonbonded path is not guaranteed to use `nbv->dispatchNonbondedKernel()`.
- Therefore SIMD admission markers in the log are not, by themselves, proof that the specialized SIMD short-range kernel performed the timed real-space force work.
- The small audited fixtures are extremely small:
  - `small_oligomer`: `6` atoms
  - `small_salt_polymer_box`: `10` atoms
  - these shapes are suitable for correctness gates, but weak as standalone OpenMP scaling evidence
