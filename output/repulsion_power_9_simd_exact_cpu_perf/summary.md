# Repulsion-Power-9 CPU Exact SIMD Benchmark

This file is host-local. It is not a cross-machine claim.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `/home/kiket/Desktop/test/GROMACS_PCFF/build/bin/gmx`
- steps per run: `8000`
- repeats per point: `2`
- pin mode: `on`

## Measurement Notes

- `ns/day` is the wall-clock campaign metric from the mdrun performance line.
- `Force s` is the `REAL CYCLE AND TIME ACCOUNTING` wallcycle entry for `Force`.
- `Force s` is kernel-adjacent, not isolated nonbonded-only timing. PME and bonded work remain in the same exact r-RESPA campaign.

## small_oligomer

| ntomp | plain-C ns/day | SIMD ns/day | wall speedup | plain-C Force s | SIMD Force s | force speedup | plain-C scaling | SIMD scaling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 739.138 | 756.956 | 1.024 | 0.100000 | 0.095500 | 1.047 | 1.000 | 1.000 |
| 2 | 878.937 | 797.168 | 0.907 | 0.105500 | 0.111500 | 0.946 | 1.189 | 1.053 |
| 6 | 1125.205 | 971.441 | 0.863 | 0.097000 | 0.128500 | 0.755 | 1.522 | 1.283 |
| 12 | 546.417 | 574.080 | 1.051 | 0.182500 | 0.170000 | 1.074 | 0.739 | 0.758 |

## small_salt_polymer_box

| ntomp | plain-C ns/day | SIMD ns/day | wall speedup | plain-C Force s | SIMD Force s | force speedup | plain-C scaling | SIMD scaling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 586.300 | 651.197 | 1.111 | 0.129500 | 0.114000 | 1.136 | 1.000 | 1.000 |
| 2 | 709.554 | 670.056 | 0.944 | 0.115000 | 0.132000 | 0.871 | 1.210 | 1.029 |
| 6 | 906.091 | 691.504 | 0.763 | 0.127500 | 0.183500 | 0.695 | 1.545 | 1.062 |
| 12 | 473.466 | 493.214 | 1.042 | 0.209500 | 0.198000 | 1.058 | 0.808 | 0.757 |
