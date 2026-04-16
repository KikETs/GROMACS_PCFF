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
- alternate mode order: `True`
- warmup cycles per layout: `1`
- modes: `specialized`

## Notes

- `real_wall_seconds` comes from the `Time:` line and is the metric to use for final speed claims.
- For layouts with separate PME ranks, `Force`, `PME mesh`, and related wallcycle rows overlap across ranks and are not additive wall shares.
- `NB F kernel` remains useful for PP-kernel comparison inside the same layout, but not as a total-wall decomposition term for PME-split layouts.

## gate_h_dense_salt_polymer_2x2x2

| layout | total threads | ntmpi | npme | ntomp | ntomp_pme | specialized ns/day | specialized/generic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| omp12 | 12 | 1 | 0 | 12 | 12 | 166.339 | n/a |
| omp6 | 6 | 1 | 0 | 6 | 6 | 186.480 | n/a |
| split12_pp6_pme6 | 12 | 2 | 1 | 6 | 6 | 272.692 | n/a |

| layout | metric | specialized | generic/specialized time ratio |
| --- | --- | --- | --- |
| omp12 | real_wall_seconds | 2.597000 | n/a |
| omp12 | force_seconds | 0.551000 | n/a |
| omp12 | pme_mesh_seconds | 1.770000 | n/a |
| omp12 | update_seconds | 0.050000 | n/a |
| omp12 | nb_f_kernel_seconds | 0.287000 | n/a |
| omp12 | pme_spread_seconds | 0.333000 | n/a |
| omp12 | pme_gather_seconds | 0.302000 | n/a |
| omp12 | pme_3d_fft_seconds | 1.076000 | n/a |
| omp6 | real_wall_seconds | 2.317000 | n/a |
| omp6 | force_seconds | 0.808000 | n/a |
| omp6 | pme_mesh_seconds | 1.329000 | n/a |
| omp6 | update_seconds | 0.019000 | n/a |
| omp6 | nb_f_kernel_seconds | 0.467000 | n/a |
| omp6 | pme_spread_seconds | 0.141000 | n/a |
| omp6 | pme_gather_seconds | 0.163000 | n/a |
| omp6 | pme_3d_fft_seconds | 0.942000 | n/a |
| split12_pp6_pme6 | real_wall_seconds | 1.584000 | n/a |
| split12_pp6_pme6 | force_seconds | 0.834000 | n/a |
| split12_pp6_pme6 | pme_mesh_seconds | 1.394000 | n/a |
| split12_pp6_pme6 | update_seconds | 0.036000 | n/a |
| split12_pp6_pme6 | nb_f_kernel_seconds | 0.490000 | n/a |
| split12_pp6_pme6 | pme_spread_seconds | 0.160000 | n/a |
| split12_pp6_pme6 | pme_gather_seconds | 0.185000 | n/a |
| split12_pp6_pme6 | pme_3d_fft_seconds | 0.959000 | n/a |
