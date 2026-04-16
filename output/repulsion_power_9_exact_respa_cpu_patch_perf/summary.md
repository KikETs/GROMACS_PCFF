# Repulsion-Power-9 Exact-r-RESPA CPU Patch Benchmark

This benchmark compares the generic and specialized exact-r-RESPA CPU pair-splitting patch path.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `/home/kiket/Desktop/test/GROMACS_PCFF/build/bin/gmx`
- steps per run: `200`
- repeats per point: `1`
- pin mode: `on`

## gate_h_dense_salt_polymer_2x2x2

| ntomp | generic ns/day | specialized ns/day | specialized/generic |
| --- | --- | --- | --- |
| 1 | 3.704 | 3.753 | 1.013 |
| 2 | 3.644 | 3.857 | 1.058 |
| 6 | 3.789 | 4.039 | 1.066 |

| ntomp | component | generic | specialized | generic/specialized time ratio |
| --- | --- | --- | --- | --- |
| 1 | neighbor_search_seconds | 0.060000 | 0.062000 | 0.968 |
| 1 | force_seconds | 2.024000 | 1.980000 | 1.022 |
| 1 | pme_mesh_seconds | 0.086000 | 0.093000 | 0.925 |
| 1 | update_seconds | 1.162000 | 1.140000 | 1.019 |
| 1 | total_wallcycle_seconds | 2.344000 | 2.314000 | 1.013 |
| 2 | neighbor_search_seconds | 0.040000 | 0.041000 | 0.976 |
| 2 | force_seconds | 2.107000 | 1.969000 | 1.070 |
| 2 | pme_mesh_seconds | 0.063000 | 0.063000 | 1.000 |
| 2 | update_seconds | 1.173000 | 1.110000 | 1.057 |
| 2 | total_wallcycle_seconds | 2.383000 | 2.251000 | 1.059 |
| 6 | neighbor_search_seconds | 0.017000 | 0.017000 | 1.000 |
| 6 | force_seconds | 2.087000 | 1.944000 | 1.074 |
| 6 | pme_mesh_seconds | 0.027000 | 0.026000 | 1.038 |
| 6 | update_seconds | 1.140000 | 1.059000 | 1.076 |
| 6 | total_wallcycle_seconds | 2.291000 | 2.150000 | 1.066 |

## small_oligomer

| ntomp | generic ns/day | specialized ns/day | specialized/generic |
| --- | --- | --- | --- |
| 1 | 464.221 | 473.128 | 1.019 |
| 2 | 717.268 | 609.335 | 0.850 |
| 6 | 683.930 | 673.061 | 0.984 |

| ntomp | component | generic | specialized | generic/specialized time ratio |
| --- | --- | --- | --- | --- |
| 1 | neighbor_search_seconds | 0.000000 | 0.000000 | n/a |
| 1 | force_seconds | 0.002000 | 0.003000 | 0.667 |
| 1 | pme_mesh_seconds | 0.006000 | 0.008000 | 0.750 |
| 1 | update_seconds | 0.004000 | 0.006000 | 0.667 |
| 1 | total_wallcycle_seconds | 0.019000 | 0.018000 | 1.056 |
| 2 | neighbor_search_seconds | 0.000000 | 0.000000 | n/a |
| 2 | force_seconds | 0.002000 | 0.003000 | 0.667 |
| 2 | pme_mesh_seconds | 0.004000 | 0.004000 | 1.000 |
| 2 | update_seconds | 0.003000 | 0.004000 | 0.750 |
| 2 | total_wallcycle_seconds | 0.012000 | 0.014000 | 0.857 |
| 6 | neighbor_search_seconds | 0.002000 | 0.000000 | n/a |
| 6 | force_seconds | 0.002000 | 0.002000 | 1.000 |
| 6 | pme_mesh_seconds | 0.002000 | 0.003000 | 0.667 |
| 6 | update_seconds | 0.003000 | 0.004000 | 0.750 |
| 6 | total_wallcycle_seconds | 0.013000 | 0.013000 | 1.000 |

## small_salt_polymer_box

| ntomp | generic ns/day | specialized ns/day | specialized/generic |
| --- | --- | --- | --- |
| 1 | 536.867 | 531.896 | 0.991 |
| 2 | 512.965 | 468.689 | 0.914 |
| 6 | 555.640 | 565.028 | 1.017 |

| ntomp | component | generic | specialized | generic/specialized time ratio |
| --- | --- | --- | --- | --- |
| 1 | neighbor_search_seconds | 0.000000 | 0.000000 | n/a |
| 1 | force_seconds | 0.003000 | 0.003000 | 1.000 |
| 1 | pme_mesh_seconds | 0.008000 | 0.008000 | 1.000 |
| 1 | update_seconds | 0.006000 | 0.006000 | 1.000 |
| 1 | total_wallcycle_seconds | 0.016000 | 0.016000 | 1.000 |
| 2 | neighbor_search_seconds | 0.000000 | 0.000000 | n/a |
| 2 | force_seconds | 0.002000 | 0.003000 | 0.667 |
| 2 | pme_mesh_seconds | 0.006000 | 0.007000 | 0.857 |
| 2 | update_seconds | 0.005000 | 0.006000 | 0.833 |
| 2 | total_wallcycle_seconds | 0.017000 | 0.019000 | 0.895 |
| 6 | neighbor_search_seconds | 0.000000 | 0.000000 | n/a |
| 6 | force_seconds | 0.003000 | 0.002000 | 1.500 |
| 6 | pme_mesh_seconds | 0.003000 | 0.003000 | 1.000 |
| 6 | update_seconds | 0.004000 | 0.004000 | 1.000 |
| 6 | total_wallcycle_seconds | 0.016000 | 0.015000 | 1.067 |
