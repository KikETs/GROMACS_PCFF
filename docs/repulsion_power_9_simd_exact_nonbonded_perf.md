# Repulsion-Power-9 CPU Exact SIMD Performance Envelope

## Bottom Line

The new capability is real, but the current audited host-local performance envelope is narrow.

On the audited Ryzen 9 9900X host, the admitted CPU SIMD exact `9-6` path improves the `ntomp=1`
exact-r-RESPA campaign against the forced plain-C reference on both audited fixtures, but it does
not support a broad scaling claim at `ntomp=2` or `ntomp=6`.

That means the honest public claim is:

- validated CPU SIMD exact repulsion-power-9 runtime capability exists
- host-local throughput improvement is present at `ntomp=1` on the audited fixtures
- no general CPU threading speedup claim is supported by the current audited envelope

## Benchmark Driver

Reproducible host-local benchmark script:

- `tools/pcff_respa_parity/bench_repulsion_power_9_simd_exact_cpu.py`

Executed output bundle:

- `output/repulsion_power_9_simd_exact_cpu_perf/summary.json`
- `output/repulsion_power_9_simd_exact_cpu_perf/summary.md`

Benchmark settings used for the current envelope:

- host: `AMD Ryzen 9 9900X 12-Core Processor`
- exact-r-RESPA CPU path
- `pin=on`
- `steps=8000`
- `repeats=2`
- audited fixtures:
  - `small_oligomer`
  - `small_salt_polymer_box`

## Measurement Basis

Two metrics are reported together:

1. `ns/day`
   campaign wall-clock throughput from the standard `mdrun` performance line
2. `Force s`
   `REAL CYCLE AND TIME ACCOUNTING` wallcycle time for `Force`

`Force s` is only a kernel-adjacent proxy. It is not isolated nonbonded-only timing, because the
same exact-r-RESPA campaign still includes PME and bonded work. That limitation is explicit and is
why the document does not overclaim a pure kernel speedup.

## Observed Host-Local Envelope

### `small_oligomer`

| ntomp | plain-C ns/day | SIMD ns/day | wall speedup | plain-C Force s | SIMD Force s | force speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 739.138 | 756.956 | 1.024 | 0.100000 | 0.095500 | 1.047 |
| 2 | 878.937 | 797.168 | 0.907 | 0.105500 | 0.111500 | 0.946 |
| 6 | 1125.205 | 971.441 | 0.863 | 0.097000 | 0.128500 | 0.755 |
| 12 | 546.417 | 574.080 | 1.051 | 0.182500 | 0.170000 | 1.074 |

### `small_salt_polymer_box`

| ntomp | plain-C ns/day | SIMD ns/day | wall speedup | plain-C Force s | SIMD Force s | force speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 586.300 | 651.197 | 1.111 | 0.129500 | 0.114000 | 1.136 |
| 2 | 709.554 | 670.056 | 0.944 | 0.115000 | 0.132000 | 0.871 |
| 6 | 906.091 | 691.504 | 0.763 | 0.127500 | 0.183500 | 0.695 |
| 12 | 473.466 | 493.214 | 1.042 | 0.209500 | 0.198000 | 1.058 |

## Interpretation

What these numbers support:

- the admitted SIMD path is not a fake enable flag; it runs and can outperform the plain-C oracle
  on this host in at least the `ntomp=1` exact runtime
- the new path is compatible with the exact-r-RESPA runtime that was validated separately

What these numbers do not support:

- a broad OpenMP scaling claim for the admitted SIMD exact `9-6` path
- a claim that SIMD wins across all tested thread counts
- a claim that the host-local `Force s` proxy is a pure nonbonded-kernel timer

## Claim Boundary

The narrow claim that fits the current evidence is:

`CPU exact short-range repulsion-power-9 admission now reaches a validated SIMD nonbonded path, and on the audited Ryzen 9 9900X host it improves the audited exact-r-RESPA runtime at ntomp=1 against the forced plain-C oracle.`

The following claims would be dishonest today:

- `repulsion-power-9 SIMD exact nonbonded scales better than plain-C across OpenMP thread counts`
- `repulsion-power-9 SIMD exact nonbonded is broadly faster on CPU`
- any implication about GPU or hybrid support
