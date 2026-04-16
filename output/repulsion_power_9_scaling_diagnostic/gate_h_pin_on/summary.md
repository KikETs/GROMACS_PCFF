# Repulsion-Power-9 Specialized SIMD CPU Benchmark

This file is host-local. It is not a cross-machine claim.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `/home/kiket/Desktop/test/GROMACS_PCFF/build/bin/gmx`
- steps per run: `1000`
- repeats per point: `2`
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

## gate_h_dense_salt_polymer_2x2x2

### Wall Throughput

| ntomp | generic ns/day | specialized ns/day | specialized/generic | generic scaling | specialized scaling |
| --- | --- | --- | --- | --- | --- |
| 2 | 3.953 | 4.005 | 1.013 | n/a | n/a |
| 6 | 4.209 | 4.226 | 1.004 | n/a | n/a |

### Wallcycle Decomposition

| ntomp | component | generic | specialized | specialized/generic |
| --- | --- | --- | --- | --- |
| 2 | force_seconds | 9.597500 | 9.441500 | 1.017 |
| 2 | pme_mesh_seconds | 0.416000 | 0.430000 | 0.967 |
| 2 | nb_xf_buffer_ops_seconds | 0.003500 | 0.003000 | 1.167 |
| 2 | update_seconds | 5.444000 | 5.376500 | 1.013 |
| 2 | total_wallcycle_seconds | 10.943000 | 10.799000 | 1.013 |
| 6 | force_seconds | 9.400000 | 9.361000 | 1.004 |
| 6 | pme_mesh_seconds | 0.153500 | 0.150000 | 1.023 |
| 6 | nb_xf_buffer_ops_seconds | 0.002000 | 0.002000 | 1.000 |
| 6 | update_seconds | 5.126500 | 5.090500 | 1.007 |
| 6 | total_wallcycle_seconds | 10.274500 | 10.232500 | 1.004 |
