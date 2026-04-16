# Repulsion-Power-9 Short-MD CPU Layout Sweep

This benchmark compares pure-OpenMP and PME-split CPU layouts on the non-MTS short-MD shape.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `/home/kiket/Desktop/test/GROMACS_PCFF/build_subcounters/bin/gmx`
- steps per run: `10000`
- repeats per point: `3`
- pin mode: `on`
- DLB mode: `no`
- alternate mode order: `False`
- warmup cycles per layout: `1`
- modes: `specialized`

## Notes

- `real_wall_seconds` comes from the `Time:` line and is the metric to use for final speed claims.
- For layouts with separate PME ranks, `Force`, `PME mesh`, and related wallcycle rows overlap across ranks and are not additive wall shares.
- `NB F kernel` remains useful for PP-kernel comparison inside the same layout, but not as a total-wall decomposition term for PME-split layouts.

## gate_h_dense_salt_polymer_2x2x2

| layout | total threads | ntmpi | npme | ntomp | ntomp_pme | specialized ns/day | specialized/generic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| omp6 | 6 | 1 | 0 | 6 | 6 | 175.104 | n/a |
| split6_pp1_pme4 | 6 | 3 | 1 | 1 | 4 | 159.189 | n/a |
| split6_pp2_pme4 | 6 | 2 | 1 | 2 | 4 | 165.954 | n/a |
| split6_pp3_pme3 | 6 | 2 | 1 | 3 | 3 | 149.997 | n/a |

| layout | metric | specialized | generic/specialized time ratio |
| --- | --- | --- | --- |
| omp6 | real_wall_seconds | 2.467000 | n/a |
| omp6 | force_seconds | 0.799000 | n/a |
| omp6 | pme_mesh_seconds | 1.498000 | n/a |
| omp6 | update_seconds | 0.019000 | n/a |
| omp6 | nb_f_kernel_seconds | 0.464000 | n/a |
| omp6 | pme_spread_seconds | 0.149000 | n/a |
| omp6 | pme_gather_seconds | 0.348000 | n/a |
| omp6 | pme_3d_fft_seconds | 0.918000 | n/a |
| split6_pp1_pme4 | real_wall_seconds | 2.714000 | n/a |
| split6_pp1_pme4 | force_seconds | 2.274000 | n/a |
| split6_pp1_pme4 | pme_mesh_seconds | 2.154000 | n/a |
| split6_pp1_pme4 | update_seconds | 0.025000 | n/a |
| split6_pp1_pme4 | nb_f_kernel_seconds | 1.385000 | n/a |
| split6_pp1_pme4 | pme_spread_seconds | 0.192000 | n/a |
| split6_pp1_pme4 | pme_gather_seconds | 0.490000 | n/a |
| split6_pp1_pme4 | pme_3d_fft_seconds | 1.348000 | n/a |
| split6_pp2_pme4 | real_wall_seconds | 2.603000 | n/a |
| split6_pp2_pme4 | force_seconds | 2.245000 | n/a |
| split6_pp2_pme4 | pme_mesh_seconds | 2.132000 | n/a |
| split6_pp2_pme4 | update_seconds | 0.027000 | n/a |
| split6_pp2_pme4 | nb_f_kernel_seconds | 1.343000 | n/a |
| split6_pp2_pme4 | pme_spread_seconds | 0.199000 | n/a |
| split6_pp2_pme4 | pme_gather_seconds | 0.487000 | n/a |
| split6_pp2_pme4 | pme_3d_fft_seconds | 1.337000 | n/a |
| split6_pp3_pme3 | real_wall_seconds | 2.880000 | n/a |
| split6_pp3_pme3 | force_seconds | 1.529000 | n/a |
| split6_pp3_pme3 | pme_mesh_seconds | 2.709000 | n/a |
| split6_pp3_pme3 | update_seconds | 0.022000 | n/a |
| split6_pp3_pme3 | nb_f_kernel_seconds | 0.912000 | n/a |
| split6_pp3_pme3 | pme_spread_seconds | 0.222000 | n/a |
| split6_pp3_pme3 | pme_gather_seconds | 0.595000 | n/a |
| split6_pp3_pme3 | pme_3d_fft_seconds | 1.734000 | n/a |
