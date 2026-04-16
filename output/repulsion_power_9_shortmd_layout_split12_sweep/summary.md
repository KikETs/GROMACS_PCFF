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
| split12_pp4_pme8 | 12 | 2 | 1 | 4 | 8 | 167.128 | n/a |
| split12_pp5_pme7 | 12 | 2 | 1 | 5 | 7 | 181.011 | n/a |
| split12_pp6_pme6 | 12 | 2 | 1 | 6 | 6 | 239.346 | n/a |
| split12_pp7_pme5 | 12 | 2 | 1 | 7 | 5 | 205.901 | n/a |
| split12_pp8_pme4 | 12 | 2 | 1 | 8 | 4 | 170.607 | n/a |

| layout | metric | specialized | generic/specialized time ratio |
| --- | --- | --- | --- |
| split12_pp4_pme8 | real_wall_seconds | 2.585000 | n/a |
| split12_pp4_pme8 | force_seconds | 1.177000 | n/a |
| split12_pp4_pme8 | pme_mesh_seconds | 2.437000 | n/a |
| split12_pp4_pme8 | update_seconds | 0.020000 | n/a |
| split12_pp4_pme8 | nb_f_kernel_seconds | 0.702000 | n/a |
| split12_pp4_pme8 | pme_spread_seconds | 0.450000 | n/a |
| split12_pp4_pme8 | pme_gather_seconds | 0.550000 | n/a |
| split12_pp4_pme8 | pme_3d_fft_seconds | 1.357000 | n/a |
| split12_pp5_pme7 | real_wall_seconds | 2.387000 | n/a |
| split12_pp5_pme7 | force_seconds | 0.982000 | n/a |
| split12_pp5_pme7 | pme_mesh_seconds | 2.236000 | n/a |
| split12_pp5_pme7 | update_seconds | 0.019000 | n/a |
| split12_pp5_pme7 | nb_f_kernel_seconds | 0.581000 | n/a |
| split12_pp5_pme7 | pme_spread_seconds | 0.241000 | n/a |
| split12_pp5_pme7 | pme_gather_seconds | 0.460000 | n/a |
| split12_pp5_pme7 | pme_3d_fft_seconds | 1.458000 | n/a |
| split12_pp6_pme6 | real_wall_seconds | 1.805000 | n/a |
| split12_pp6_pme6 | force_seconds | 0.846000 | n/a |
| split12_pp6_pme6 | pme_mesh_seconds | 1.621000 | n/a |
| split12_pp6_pme6 | update_seconds | 0.037000 | n/a |
| split12_pp6_pme6 | nb_f_kernel_seconds | 0.501000 | n/a |
| split12_pp6_pme6 | pme_spread_seconds | 0.175000 | n/a |
| split12_pp6_pme6 | pme_gather_seconds | 0.387000 | n/a |
| split12_pp6_pme6 | pme_3d_fft_seconds | 0.975000 | n/a |
| split12_pp7_pme5 | real_wall_seconds | 2.098000 | n/a |
| split12_pp7_pme5 | force_seconds | 0.827000 | n/a |
| split12_pp7_pme5 | pme_mesh_seconds | 1.872000 | n/a |
| split12_pp7_pme5 | update_seconds | 0.050000 | n/a |
| split12_pp7_pme5 | nb_f_kernel_seconds | 0.472000 | n/a |
| split12_pp7_pme5 | pme_spread_seconds | 0.183000 | n/a |
| split12_pp7_pme5 | pme_gather_seconds | 0.446000 | n/a |
| split12_pp7_pme5 | pme_3d_fft_seconds | 1.143000 | n/a |
| split12_pp8_pme4 | real_wall_seconds | 2.532000 | n/a |
| split12_pp8_pme4 | force_seconds | 0.736000 | n/a |
| split12_pp8_pme4 | pme_mesh_seconds | 2.305000 | n/a |
| split12_pp8_pme4 | update_seconds | 0.050000 | n/a |
| split12_pp8_pme4 | nb_f_kernel_seconds | 0.412000 | n/a |
| split12_pp8_pme4 | pme_spread_seconds | 0.235000 | n/a |
| split12_pp8_pme4 | pme_gather_seconds | 0.541000 | n/a |
| split12_pp8_pme4 | pme_3d_fft_seconds | 1.407000 | n/a |
