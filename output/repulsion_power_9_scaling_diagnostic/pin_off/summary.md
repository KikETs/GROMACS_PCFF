# Repulsion-Power-9 Specialized SIMD CPU Benchmark

This file is host-local. It is not a cross-machine claim.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `/home/kiket/Desktop/test/GROMACS_PCFF/build/bin/gmx`
- steps per run: `8000`
- repeats per point: `2`
- pin mode: `off`
- modes: `generic, specialized`
- affinity reporting: `False`

## Measurement Notes

- `plain-C` forces the reference path with `GMX_DISABLE_SIMD_KERNELS=1` when `plainc` is included.
- `generic SIMD` keeps the admitted repulsion-power-9 SIMD path but disables the specialization with `GMX_DISABLE_REPULSION_POWER_9_SIMD_SPECIALIZATION=1`.
- `specialized SIMD` is the default admitted repulsion-power-9 path when `GMX_DISABLE_REPULSION_POWER_9_SIMD_SPECIALIZATION` is unset.
- `ns/day` is the wall-clock campaign metric from the mdrun performance line.
- `Force`, `PME mesh`, `NB X/F buffer ops.`, `Update`, and `Total` come from `REAL CYCLE AND TIME ACCOUNTING`.
- `Force s` is kernel-adjacent, not isolated nonbonded-only timing.
- Bonded-only wallcycle time is not reported separately by this stack.

## small_oligomer

### Wall Throughput

| ntomp | generic ns/day | specialized ns/day | specialized/generic | generic scaling | specialized scaling |
| --- | --- | --- | --- | --- | --- |
| 1 | 856.711 | 867.145 | 1.012 | 1.000 | 1.000 |
| 2 | 1060.624 | 1038.459 | 0.979 | 1.238 | 1.198 |
| 4 | 1119.218 | 1141.169 | 1.020 | 1.306 | 1.316 |
| 6 | 706.832 | 712.635 | 1.008 | 0.825 | 0.822 |
| 8 | 568.372 | 597.947 | 1.052 | 0.663 | 0.690 |
| 12 | 482.341 | 479.802 | 0.995 | 0.563 | 0.553 |

### Wallcycle Decomposition

| ntomp | component | generic | specialized | specialized/generic |
| --- | --- | --- | --- | --- |
| 1 | force_seconds | 0.080500 | 0.079000 | 1.019 |
| 1 | pme_mesh_seconds | 0.248000 | 0.247000 | 1.004 |
| 1 | nb_xf_buffer_ops_seconds | 0.004000 | 0.004000 | 1.000 |
| 1 | update_seconds | 0.191500 | 0.190000 | 1.008 |
| 1 | total_wallcycle_seconds | 0.403500 | 0.399000 | 1.011 |
| 2 | force_seconds | 0.080500 | 0.081500 | 0.988 |
| 2 | pme_mesh_seconds | 0.162500 | 0.168000 | 0.967 |
| 2 | nb_xf_buffer_ops_seconds | 0.004500 | 0.004500 | 1.000 |
| 2 | update_seconds | 0.151000 | 0.154500 | 0.977 |
| 2 | total_wallcycle_seconds | 0.326000 | 0.333000 | 0.979 |
| 4 | force_seconds | 0.090500 | 0.090000 | 1.006 |
| 4 | pme_mesh_seconds | 0.116500 | 0.116000 | 1.004 |
| 4 | nb_xf_buffer_ops_seconds | 0.007500 | 0.007000 | 1.071 |
| 4 | update_seconds | 0.139000 | 0.137500 | 1.011 |
| 4 | total_wallcycle_seconds | 0.309000 | 0.306500 | 1.008 |
| 6 | force_seconds | 0.128000 | 0.127000 | 1.008 |
| 6 | pme_mesh_seconds | 0.181500 | 0.180000 | 1.008 |
| 6 | nb_xf_buffer_ops_seconds | 0.018000 | 0.018000 | 1.000 |
| 6 | update_seconds | 0.212000 | 0.210500 | 1.007 |
| 6 | total_wallcycle_seconds | 0.489000 | 0.486000 | 1.006 |
| 8 | force_seconds | 0.141000 | 0.143500 | 0.983 |
| 8 | pme_mesh_seconds | 0.259000 | 0.223500 | 1.159 |
| 8 | nb_xf_buffer_ops_seconds | 0.023000 | 0.023500 | 0.979 |
| 8 | update_seconds | 0.266000 | 0.250000 | 1.064 |
| 8 | total_wallcycle_seconds | 0.608000 | 0.578000 | 1.052 |
| 12 | force_seconds | 0.168500 | 0.169500 | 0.994 |
| 12 | pme_mesh_seconds | 0.301000 | 0.299500 | 1.005 |
| 12 | nb_xf_buffer_ops_seconds | 0.030000 | 0.029000 | 1.034 |
| 12 | update_seconds | 0.313500 | 0.314000 | 0.998 |
| 12 | total_wallcycle_seconds | 0.717000 | 0.721000 | 0.994 |

## small_salt_polymer_box

### Wall Throughput

| ntomp | generic ns/day | specialized ns/day | specialized/generic | generic scaling | specialized scaling |
| --- | --- | --- | --- | --- | --- |
| 1 | 694.393 | 702.065 | 1.011 | 1.000 | 1.000 |
| 2 | 788.643 | 835.246 | 1.059 | 1.136 | 1.190 |
| 4 | 926.061 | 835.436 | 0.902 | 1.334 | 1.190 |
| 6 | 586.748 | 580.311 | 0.989 | 0.845 | 0.827 |
| 8 | 459.923 | 461.934 | 1.004 | 0.662 | 0.658 |
| 12 | 420.414 | 406.192 | 0.966 | 0.605 | 0.579 |

### Wallcycle Decomposition

| ntomp | component | generic | specialized | specialized/generic |
| --- | --- | --- | --- | --- |
| 1 | force_seconds | 0.103500 | 0.103000 | 1.005 |
| 1 | pme_mesh_seconds | 0.310500 | 0.307000 | 1.011 |
| 1 | nb_xf_buffer_ops_seconds | 0.004000 | 0.004000 | 1.000 |
| 1 | update_seconds | 0.237500 | 0.235000 | 1.011 |
| 1 | total_wallcycle_seconds | 0.497500 | 0.492500 | 1.010 |
| 2 | force_seconds | 0.103500 | 0.100000 | 1.035 |
| 2 | pme_mesh_seconds | 0.237500 | 0.225000 | 1.056 |
| 2 | nb_xf_buffer_ops_seconds | 0.005500 | 0.004500 | 1.222 |
| 2 | update_seconds | 0.205000 | 0.194000 | 1.057 |
| 2 | total_wallcycle_seconds | 0.438500 | 0.414000 | 1.059 |
| 4 | force_seconds | 0.108000 | 0.115500 | 0.935 |
| 4 | pme_mesh_seconds | 0.159500 | 0.182000 | 0.876 |
| 4 | nb_xf_buffer_ops_seconds | 0.007000 | 0.008000 | 0.875 |
| 4 | update_seconds | 0.170000 | 0.188500 | 0.902 |
| 4 | total_wallcycle_seconds | 0.373500 | 0.415000 | 0.900 |
| 6 | force_seconds | 0.154000 | 0.147500 | 1.044 |
| 6 | pme_mesh_seconds | 0.234000 | 0.259500 | 0.902 |
| 6 | nb_xf_buffer_ops_seconds | 0.020000 | 0.018500 | 1.081 |
| 6 | update_seconds | 0.259000 | 0.264000 | 0.981 |
| 6 | total_wallcycle_seconds | 0.589500 | 0.596000 | 0.989 |
| 8 | force_seconds | 0.166500 | 0.159000 | 1.047 |
| 8 | pme_mesh_seconds | 0.357500 | 0.371500 | 0.962 |
| 8 | nb_xf_buffer_ops_seconds | 0.024500 | 0.024000 | 1.021 |
| 8 | update_seconds | 0.333500 | 0.333500 | 1.000 |
| 8 | total_wallcycle_seconds | 0.752000 | 0.748500 | 1.005 |
| 12 | force_seconds | 0.180500 | 0.187000 | 0.965 |
| 12 | pme_mesh_seconds | 0.395500 | 0.408500 | 0.968 |
| 12 | nb_xf_buffer_ops_seconds | 0.028000 | 0.030000 | 0.933 |
| 12 | update_seconds | 0.365500 | 0.379000 | 0.964 |
| 12 | total_wallcycle_seconds | 0.822500 | 0.853000 | 0.964 |
