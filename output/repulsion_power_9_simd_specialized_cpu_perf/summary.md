# Repulsion-Power-9 Specialized SIMD CPU Benchmark

This file is host-local. It is not a cross-machine claim.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `../../build/bin/gmx`
- steps per run: `8000`
- repeats per point: `2`
- pin mode: `on`

## Measurement Notes

- `plain-C` forces the reference path with `GMX_DISABLE_SIMD_KERNELS=1`.
- `generic SIMD` keeps the admitted repulsion-power-9 SIMD path but disables the specialization with `GMX_DISABLE_REPULSION_POWER_9_SIMD_SPECIALIZATION=1`.
- `specialized SIMD` is the default admitted repulsion-power-9 path.
- `ns/day` is the wall-clock campaign metric from the mdrun performance line.
- `Force s` is the `REAL CYCLE AND TIME ACCOUNTING` wallcycle entry for `Force`.
- `Force s` is kernel-adjacent, not isolated nonbonded-only timing.

## small_oligomer

### Wall Throughput

| ntomp | plain-C ns/day | generic SIMD ns/day | specialized SIMD ns/day | generic/plain-C | specialized/generic | specialized/plain-C | plain-C scaling | generic scaling | specialized scaling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 836.245 | 800.563 | 662.459 | 0.957 | 0.827 | 0.792 | 1.000 | 1.000 | 1.000 |
| 2 | 848.809 | 775.103 | 988.711 | 0.913 | 1.276 | 1.165 | 1.015 | 0.968 | 1.492 |
| 6 | 1001.229 | 1063.742 | 852.943 | 1.062 | 0.802 | 0.852 | 1.197 | 1.329 | 1.288 |
| 12 | 512.827 | 485.624 | 517.734 | 0.947 | 1.066 | 1.010 | 0.613 | 0.607 | 0.782 |

### Force Proxy

| ntomp | plain-C Force s | generic SIMD Force s | specialized SIMD Force s | generic/plain-C | specialized/generic | specialized/plain-C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.083500 | 0.088000 | 0.113500 | 0.949 | 0.775 | 0.736 |
| 2 | 0.101500 | 0.115500 | 0.090500 | 0.879 | 1.276 | 1.122 |
| 6 | 0.123500 | 0.110000 | 0.153500 | 1.123 | 0.717 | 0.805 |
| 12 | 0.210500 | 0.223500 | 0.207000 | 0.942 | 1.080 | 1.017 |

## small_salt_polymer_box

### Wall Throughput

| ntomp | plain-C ns/day | generic SIMD ns/day | specialized SIMD ns/day | generic/plain-C | specialized/generic | specialized/plain-C | plain-C scaling | generic scaling | specialized scaling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 610.125 | 619.515 | 630.121 | 1.015 | 1.017 | 1.033 | 1.000 | 1.000 | 1.000 |
| 2 | 708.568 | 728.848 | 809.605 | 1.029 | 1.111 | 1.143 | 1.161 | 1.176 | 1.285 |
| 6 | 973.930 | 979.106 | 931.574 | 1.005 | 0.951 | 0.957 | 1.596 | 1.580 | 1.478 |
| 12 | 474.308 | 503.291 | 506.713 | 1.061 | 1.007 | 1.068 | 0.777 | 0.812 | 0.804 |

### Force Proxy

| ntomp | plain-C Force s | generic SIMD Force s | specialized SIMD Force s | generic/plain-C | specialized/generic | specialized/plain-C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.122000 | 0.120000 | 0.118500 | 1.017 | 1.013 | 1.030 |
| 2 | 0.124500 | 0.116500 | 0.104000 | 1.069 | 1.120 | 1.197 |
| 6 | 0.114000 | 0.116500 | 0.131500 | 0.979 | 0.886 | 0.867 |
| 12 | 0.211000 | 0.189000 | 0.190500 | 1.116 | 0.992 | 1.108 |
