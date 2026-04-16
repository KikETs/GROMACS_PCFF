# Repulsion-Power-9 Specialized SIMD CPU Benchmark

This file is host-local. It is not a cross-machine claim.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `/home/kiket/Desktop/test/GROMACS_PCFF/build/bin/gmx`
- steps per run: `8000`
- repeats per point: `6`
- pin mode: `on`
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
| 2 | 890.534 | 936.687 | 1.052 | n/a | n/a |
| 6 | 1181.845 | 903.779 | 0.765 | n/a | n/a |

### Wallcycle Decomposition

| ntomp | component | generic | specialized | specialized/generic |
| --- | --- | --- | --- | --- |
| 2 | force_seconds | 0.101500 | 0.086000 | 1.180 |
| 2 | pme_mesh_seconds | 0.185000 | 0.190000 | 0.974 |
| 2 | nb_xf_buffer_ops_seconds | 0.005000 | 0.005000 | 1.000 |
| 2 | update_seconds | 0.180500 | 0.172500 | 1.046 |
| 2 | total_wallcycle_seconds | 0.388000 | 0.369000 | 1.051 |
| 6 | force_seconds | 0.098000 | 0.141000 | 0.695 |
| 6 | pme_mesh_seconds | 0.085000 | 0.098500 | 0.863 |
| 6 | nb_xf_buffer_ops_seconds | 0.007000 | 0.008000 | 0.875 |
| 6 | update_seconds | 0.130000 | 0.172500 | 0.754 |
| 6 | total_wallcycle_seconds | 0.292500 | 0.384000 | 0.762 |

## small_salt_polymer_box

### Wall Throughput

| ntomp | generic ns/day | specialized ns/day | specialized/generic | generic scaling | specialized scaling |
| --- | --- | --- | --- | --- | --- |
| 2 | 737.567 | 752.659 | 1.020 | n/a | n/a |
| 6 | 853.747 | 961.120 | 1.126 | n/a | n/a |

### Wallcycle Decomposition

| ntomp | component | generic | specialized | specialized/generic |
| --- | --- | --- | --- | --- |
| 2 | force_seconds | 0.105500 | 0.111500 | 0.946 |
| 2 | pme_mesh_seconds | 0.265000 | 0.256000 | 1.035 |
| 2 | nb_xf_buffer_ops_seconds | 0.004500 | 0.005000 | 0.900 |
| 2 | update_seconds | 0.220000 | 0.216000 | 1.019 |
| 2 | total_wallcycle_seconds | 0.469000 | 0.459000 | 1.022 |
| 6 | force_seconds | 0.148500 | 0.113500 | 1.308 |
| 6 | pme_mesh_seconds | 0.133500 | 0.132500 | 1.008 |
| 6 | nb_xf_buffer_ops_seconds | 0.007500 | 0.007000 | 1.071 |
| 6 | update_seconds | 0.188500 | 0.163000 | 1.156 |
| 6 | total_wallcycle_seconds | 0.415500 | 0.359500 | 1.156 |
