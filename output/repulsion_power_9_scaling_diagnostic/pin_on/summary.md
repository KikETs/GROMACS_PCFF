# Repulsion-Power-9 Specialized SIMD CPU Benchmark

This file is host-local. It is not a cross-machine claim.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `../../../build/bin/gmx`
- steps per run: `8000`
- repeats per point: `2`
- pin mode: `on`
- modes: `generic, specialized`
- affinity reporting: `True`

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
| 1 | 828.931 | 682.104 | 0.823 | 1.000 | 1.000 |
| 2 | 992.514 | 977.396 | 0.985 | 1.197 | 1.433 |
| 4 | 1007.429 | 1154.249 | 1.146 | 1.215 | 1.692 |
| 6 | 1150.045 | 1044.997 | 0.909 | 1.387 | 1.532 |
| 8 | 607.866 | 616.815 | 1.015 | 0.733 | 0.904 |
| 12 | 541.598 | 479.299 | 0.885 | 0.653 | 0.703 |

### Wallcycle Decomposition

| ntomp | component | generic | specialized | specialized/generic |
| --- | --- | --- | --- | --- |
| 1 | force_seconds | 0.085000 | 0.114500 | 0.742 |
| 1 | pme_mesh_seconds | 0.252500 | 0.305500 | 0.827 |
| 1 | nb_xf_buffer_ops_seconds | 0.004000 | 0.005000 | 0.800 |
| 1 | update_seconds | 0.197500 | 0.251500 | 0.785 |
| 1 | total_wallcycle_seconds | 0.417000 | 0.530000 | 0.787 |
| 2 | force_seconds | 0.090000 | 0.085000 | 1.059 |
| 2 | pme_mesh_seconds | 0.170000 | 0.182500 | 0.932 |
| 2 | nb_xf_buffer_ops_seconds | 0.004500 | 0.004000 | 1.125 |
| 2 | update_seconds | 0.162000 | 0.164500 | 0.985 |
| 2 | total_wallcycle_seconds | 0.349000 | 0.354500 | 0.984 |
| 4 | force_seconds | 0.119500 | 0.091500 | 1.306 |
| 4 | pme_mesh_seconds | 0.117500 | 0.111500 | 1.054 |
| 4 | nb_xf_buffer_ops_seconds | 0.006500 | 0.006500 | 1.000 |
| 4 | update_seconds | 0.163000 | 0.136000 | 1.199 |
| 4 | total_wallcycle_seconds | 0.356500 | 0.299500 | 1.190 |
| 6 | force_seconds | 0.100500 | 0.124000 | 0.810 |
| 6 | pme_mesh_seconds | 0.093500 | 0.092500 | 1.011 |
| 6 | nb_xf_buffer_ops_seconds | 0.007500 | 0.007500 | 1.000 |
| 6 | update_seconds | 0.134500 | 0.154500 | 0.871 |
| 6 | total_wallcycle_seconds | 0.300500 | 0.344000 | 0.874 |
| 8 | force_seconds | 0.162000 | 0.151000 | 1.073 |
| 8 | pme_mesh_seconds | 0.195500 | 0.199500 | 0.980 |
| 8 | nb_xf_buffer_ops_seconds | 0.022000 | 0.022500 | 0.978 |
| 8 | update_seconds | 0.249000 | 0.241500 | 1.031 |
| 8 | total_wallcycle_seconds | 0.572500 | 0.561000 | 1.020 |
| 12 | force_seconds | 0.198500 | 0.232500 | 0.854 |
| 12 | pme_mesh_seconds | 0.182000 | 0.190500 | 0.955 |
| 12 | nb_xf_buffer_ops_seconds | 0.029500 | 0.030000 | 0.983 |
| 12 | update_seconds | 0.274500 | 0.307000 | 0.894 |
| 12 | total_wallcycle_seconds | 0.646500 | 0.721000 | 0.897 |

## small_salt_polymer_box

### Wall Throughput

| ntomp | generic ns/day | specialized ns/day | specialized/generic | generic scaling | specialized scaling |
| --- | --- | --- | --- | --- | --- |
| 1 | 697.053 | 658.581 | 0.945 | 1.000 | 1.000 |
| 2 | 763.946 | 802.403 | 1.050 | 1.096 | 1.218 |
| 4 | 819.174 | 822.438 | 1.004 | 1.175 | 1.249 |
| 6 | 951.720 | 930.240 | 0.977 | 1.365 | 1.412 |
| 8 | 501.264 | 535.016 | 1.067 | 0.719 | 0.812 |
| 12 | 491.725 | 487.491 | 0.991 | 0.705 | 0.740 |

### Wallcycle Decomposition

| ntomp | component | generic | specialized | specialized/generic |
| --- | --- | --- | --- | --- |
| 1 | force_seconds | 0.105500 | 0.112000 | 0.942 |
| 1 | pme_mesh_seconds | 0.306000 | 0.325000 | 0.942 |
| 1 | nb_xf_buffer_ops_seconds | 0.004000 | 0.004000 | 1.000 |
| 1 | update_seconds | 0.237000 | 0.251500 | 0.942 |
| 1 | total_wallcycle_seconds | 0.496000 | 0.526500 | 0.942 |
| 2 | force_seconds | 0.114500 | 0.103000 | 1.112 |
| 2 | pme_mesh_seconds | 0.239500 | 0.235000 | 1.019 |
| 2 | nb_xf_buffer_ops_seconds | 0.004500 | 0.005000 | 0.900 |
| 2 | update_seconds | 0.213000 | 0.202000 | 1.054 |
| 2 | total_wallcycle_seconds | 0.453500 | 0.430500 | 1.053 |
| 4 | force_seconds | 0.133000 | 0.141000 | 0.943 |
| 4 | pme_mesh_seconds | 0.173500 | 0.165500 | 1.048 |
| 4 | nb_xf_buffer_ops_seconds | 0.006500 | 0.006500 | 1.000 |
| 4 | update_seconds | 0.197000 | 0.199000 | 0.990 |
| 4 | total_wallcycle_seconds | 0.425500 | 0.432500 | 0.984 |
| 6 | force_seconds | 0.121000 | 0.115000 | 1.052 |
| 6 | pme_mesh_seconds | 0.131000 | 0.144500 | 0.907 |
| 6 | nb_xf_buffer_ops_seconds | 0.007000 | 0.008000 | 0.875 |
| 6 | update_seconds | 0.166500 | 0.169500 | 0.982 |
| 6 | total_wallcycle_seconds | 0.363500 | 0.371500 | 0.978 |
| 8 | force_seconds | 0.190500 | 0.161000 | 1.183 |
| 8 | pme_mesh_seconds | 0.266000 | 0.261500 | 1.017 |
| 8 | nb_xf_buffer_ops_seconds | 0.023500 | 0.024500 | 0.959 |
| 8 | update_seconds | 0.305000 | 0.281000 | 1.085 |
| 8 | total_wallcycle_seconds | 0.693500 | 0.646000 | 1.074 |
| 12 | force_seconds | 0.189000 | 0.194000 | 0.974 |
| 12 | pme_mesh_seconds | 0.247000 | 0.241500 | 1.023 |
| 12 | nb_xf_buffer_ops_seconds | 0.031500 | 0.032000 | 0.984 |
| 12 | update_seconds | 0.299500 | 0.298500 | 1.003 |
| 12 | total_wallcycle_seconds | 0.703000 | 0.709000 | 0.992 |
