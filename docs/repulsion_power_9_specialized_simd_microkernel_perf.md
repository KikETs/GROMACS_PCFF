# Repulsion-Power-9 Specialized SIMD Microkernel Performance Envelope

## Bottom Line

The old exact-`r-RESPA` performance interpretation is withdrawn.

Those runs used exact `r-RESPA` pair splitting and did not time the admitted specialized SIMD short-range
nonbonded kernel in the real-space force path. They are no longer valid evidence for the specialized
microkernel claim.

The valid host-local benchmark is now the non-MTS short-MD CPU path that actually executes
`dispatchNonbondedKernel()` and records non-zero `NB F kernel` time.

On that valid benchmark:

- the specialized repulsion-power-9 SIMD microkernel is measurably faster than the generic admitted SIMD
  baseline in `NB F kernel` time at every audited thread count
- the broader wall-clock result is still host-local and mixed

So the honest claim is:

`the specialized microkernel itself is faster on the valid CPU short-range benchmark, but that does not
translate into a broad audited wall-clock speedup claim on this host.`

## Withdrawn Benchmark

Withdrawn as microkernel evidence:

- historical bundle:
  - [`output/repulsion_power_9_simd_specialized_cpu_perf/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_simd_specialized_cpu_perf/summary.md)
- reason:
  - exact `r-RESPA` pair splitting routes CPU force evaluation through
    [`computeExactRespaNonbondedCpu()`](/home/kiket/Desktop/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp:3498),
    not the admitted specialized SIMD short-range kernel

See:

- [`docs/repulsion_power_9_ntomp6_regression_analysis.md`](/home/kiket/Desktop/test/GROMACS_PCFF/docs/repulsion_power_9_ntomp6_regression_analysis.md)

## Valid Benchmark Driver

Reproducible host-local benchmark script:

- [`tools/pcff_respa_parity/bench_repulsion_power_9_simd_shortmd_cpu.py`](/home/kiket/Desktop/test/GROMACS_PCFF/tools/pcff_respa_parity/bench_repulsion_power_9_simd_shortmd_cpu.py)

Executed output bundle:

- [`output/repulsion_power_9_simd_shortmd_cpu_perf/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_simd_shortmd_cpu_perf/summary.md)
- [`output/repulsion_power_9_simd_shortmd_cpu_perf/summary.json`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_simd_shortmd_cpu_perf/summary.json)

Audited run settings:

- host: `AMD Ryzen 9 9900X 12-Core Processor`
- binary: [`build_subcounters/bin/gmx`](/home/kiket/Desktop/test/GROMACS_PCFF/build_subcounters/bin/gmx)
- CPU-only non-MTS short-MD
- `pin=on`
- `steps=200`
- `repeats=1`
- audited fixture:
  - `gate_h_dense_salt_polymer_2x2x2`

## Measurement Basis

Two runtime layers were compared on the same repulsion-power-9 system:

1. generic admitted SIMD baseline
2. specialized repulsion-power-9 SIMD path

Three metrics matter:

1. `ns/day`
   whole-run wall-clock throughput from the `Performance:` line
2. `Force s`
   wallcycle `Force` time
3. `NB F kernel`
   wallcycle subcounter for the actual CPU short-range nonbonded kernel

`NB F kernel` is the decisive metric here, because this benchmark shape actually emits it as non-zero.

## Observed Host-Local Envelope

### `gate_h_dense_salt_polymer_2x2x2`

| ntomp | generic SIMD ns/day | specialized SIMD ns/day | specialized/generic wall | generic NB F kernel s | specialized NB F kernel s | specialized/generic NB F kernel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 27.283 | 31.445 | 1.153 | 0.094000 | 0.059000 | 1.593 |
| 2 | 49.061 | 42.559 | 0.867 | 0.048000 | 0.038000 | 1.263 |
| 6 | 109.711 | 103.140 | 0.940 | 0.018000 | 0.013000 | 1.385 |

Supporting `Force` proxy:

| ntomp | generic Force s | specialized Force s | specialized/generic Force |
| --- | ---: | ---: | ---: |
| 1 | 0.130000 | 0.095000 | 1.368 |
| 2 | 0.067000 | 0.061000 | 1.098 |
| 6 | 0.026000 | 0.021000 | 1.238 |

## Interpretation

What the current evidence supports:

- the specialized exact SIMD microkernel is not a fake enable
- on a valid non-MTS CPU benchmark, the specialized path reduces `NB F kernel` time at every audited
  thread count on the audited host
- the microkernel speedup is strongest at `ntomp=1` in this measured shape and remains positive at
  `ntomp=2` and `ntomp=6`

What the current evidence does not support:

- a broad wall-clock speedup claim
- a claim that faster `NB F kernel` time automatically improves total throughput at higher thread counts
- any implication about exact `r-RESPA` pair splitting
- any implication about GPU, hybrid, LJ-PME, Buckingham, or broader chemistry support

The critical host-local observation is that the faster microkernel does not guarantee a faster whole run:

- `ntomp=1`: specialized wins in both kernel time and wall time
- `ntomp=2`, `ntomp=6`: specialized still wins in `NB F kernel`, but loses in total wall time

That means the remaining wall-clock bottleneck on this audited host is outside the microkernel itself.

## Claim Boundary

Allowed:

- `On the audited Ryzen 9 9900X host, the repulsion-power-9 specialized exact CPU SIMD microkernel reduces actual NB F kernel time on a valid non-MTS short-range benchmark.`
- `The microkernel speedup does not yet justify a broad wall-clock CPU speedup claim.`

Not allowed:

- `the specialized repulsion-power-9 SIMD path is broadly faster on CPU`
- `the old exact-r-RESPA benchmark proved the microkernel speedup`
- any implication about GPU or hybrid support
- any implication about LJ-PME SIMD support
