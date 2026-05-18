# Repulsion-Power-9 Short-MD CPU Layout Sweep

This benchmark compares pure-OpenMP and PME-split CPU layouts on the non-MTS short-MD shape.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `../../build_subcounters/bin/gmx`
- steps per run: `10000`
- repeats per point: `6`
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
| omp12 | 12 | 1 | 0 | 12 | 12 | 161.897 | n/a |
| split12_pp6_pme6 | 12 | 2 | 1 | 6 | 6 | 269.829 | n/a |

| layout | metric | specialized | generic/specialized time ratio |
| --- | --- | --- | --- |
| omp12 | real_wall_seconds | 2.668500 | n/a |
| omp12 | force_seconds | 0.565500 | n/a |
| omp12 | pme_mesh_seconds | 1.806500 | n/a |
| omp12 | update_seconds | 0.051500 | n/a |
| omp12 | nb_f_kernel_seconds | 0.287500 | n/a |
| omp12 | pme_spread_seconds | 0.338000 | n/a |
| omp12 | pme_gather_seconds | 0.285500 | n/a |
| omp12 | pme_3d_fft_seconds | 1.117000 | n/a |
| split12_pp6_pme6 | real_wall_seconds | 1.601000 | n/a |
| split12_pp6_pme6 | force_seconds | 0.845000 | n/a |
| split12_pp6_pme6 | pme_mesh_seconds | 1.405000 | n/a |
| split12_pp6_pme6 | update_seconds | 0.036000 | n/a |
| split12_pp6_pme6 | nb_f_kernel_seconds | 0.497000 | n/a |
| split12_pp6_pme6 | pme_spread_seconds | 0.162000 | n/a |
| split12_pp6_pme6 | pme_gather_seconds | 0.188500 | n/a |
| split12_pp6_pme6 | pme_3d_fft_seconds | 0.968000 | n/a |
