# Repulsion-Power-9 Short-MD CPU Layout Sweep

This benchmark compares pure-OpenMP and PME-split CPU layouts on the non-MTS short-MD shape.

## Host

- hostname: `user-Z690-AORUS-PRO`
- cpu: `12th Gen Intel(R) Core(TM) i9-12900K`
- gmx: `/home/user/tmp/gromacs_pcff_remotecheck/build_subcounters_remote/bin/gmx`
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
| omp12 | 12 | 1 | 0 | 12 | 12 | 77.844 | n/a |
| omp6 | 6 | 1 | 0 | 6 | 6 | 107.904 | n/a |
| split12_pp6_pme6 | 12 | 2 | 1 | 6 | 6 | 95.751 | n/a |

| layout | metric | specialized | generic/specialized time ratio |
| --- | --- | --- | --- |
| omp12 | real_wall_seconds | 5.550000 | n/a |
| omp12 | force_seconds | 1.741000 | n/a |
| omp12 | pme_mesh_seconds | 2.957000 | n/a |
| omp12 | update_seconds | 0.216000 | n/a |
| omp12 | nb_f_kernel_seconds | 1.066000 | n/a |
| omp12 | pme_spread_seconds | 0.664000 | n/a |
| omp12 | pme_gather_seconds | 0.639000 | n/a |
| omp12 | pme_3d_fft_seconds | 1.540000 | n/a |
| omp6 | real_wall_seconds | 4.004000 | n/a |
| omp6 | force_seconds | 1.313000 | n/a |
| omp6 | pme_mesh_seconds | 2.176000 | n/a |
| omp6 | update_seconds | 0.119000 | n/a |
| omp6 | nb_f_kernel_seconds | 0.775000 | n/a |
| omp6 | pme_spread_seconds | 0.392000 | n/a |
| omp6 | pme_gather_seconds | 0.336000 | n/a |
| omp6 | pme_3d_fft_seconds | 1.304000 | n/a |
| split12_pp6_pme6 | real_wall_seconds | 4.512000 | n/a |
| split12_pp6_pme6 | force_seconds | 1.373000 | n/a |
| split12_pp6_pme6 | pme_mesh_seconds | 4.045000 | n/a |
| split12_pp6_pme6 | update_seconds | 0.134000 | n/a |
| split12_pp6_pme6 | nb_f_kernel_seconds | 0.800000 | n/a |
| split12_pp6_pme6 | pme_spread_seconds | 0.605000 | n/a |
| split12_pp6_pme6 | pme_gather_seconds | 0.694000 | n/a |
| split12_pp6_pme6 | pme_3d_fft_seconds | 2.564000 | n/a |
