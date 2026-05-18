# Repulsion-Power-9 Short-MD CPU Layout Sweep

This benchmark compares pure-OpenMP and PME-split CPU layouts on the non-MTS short-MD shape.

## Host

- hostname: `user-Z690-AORUS-PRO`
- cpu: `12th Gen Intel(R) Core(TM) i9-12900K`
- gmx: `tmp/gromacs_pcff_remotecheck/pcore_gmx.sh`
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
| omp6 | 6 | 1 | 0 | 6 | 6 | 109.401 | n/a |
| omp8 | 8 | 1 | 0 | 8 | 8 | 113.313 | n/a |
| split8_pp2_pme6 | 8 | 2 | 1 | 2 | 6 | 104.876 | n/a |
| split8_pp3_pme5 | 8 | 2 | 1 | 3 | 5 | 139.540 | n/a |
| split8_pp4_pme4 | 8 | 2 | 1 | 4 | 4 | 129.320 | n/a |
| split8_pp5_pme3 | 8 | 2 | 1 | 5 | 3 | 113.325 | n/a |
| split8_pp6_pme2 | 8 | 2 | 1 | 6 | 2 | 82.817 | n/a |

| layout | metric | specialized | generic/specialized time ratio |
| --- | --- | --- | --- |
| omp6 | real_wall_seconds | 3.949000 | n/a |
| omp6 | force_seconds | 1.297000 | n/a |
| omp6 | pme_mesh_seconds | 2.145000 | n/a |
| omp6 | update_seconds | 0.111000 | n/a |
| omp6 | nb_f_kernel_seconds | 0.764000 | n/a |
| omp6 | pme_spread_seconds | 0.382000 | n/a |
| omp6 | pme_gather_seconds | 0.335000 | n/a |
| omp6 | pme_3d_fft_seconds | 1.278000 | n/a |
| omp8 | real_wall_seconds | 3.813000 | n/a |
| omp8 | force_seconds | 1.040000 | n/a |
| omp8 | pme_mesh_seconds | 2.201000 | n/a |
| omp8 | update_seconds | 0.128000 | n/a |
| omp8 | nb_f_kernel_seconds | 0.554000 | n/a |
| omp8 | pme_spread_seconds | 0.571000 | n/a |
| omp8 | pme_gather_seconds | 0.379000 | n/a |
| omp8 | pme_3d_fft_seconds | 1.141000 | n/a |
| split8_pp2_pme6 | real_wall_seconds | 4.120000 | n/a |
| split8_pp2_pme6 | force_seconds | 3.284000 | n/a |
| split8_pp2_pme6 | pme_mesh_seconds | 2.276000 | n/a |
| split8_pp2_pme6 | update_seconds | 0.099000 | n/a |
| split8_pp2_pme6 | nb_f_kernel_seconds | 2.143000 | n/a |
| split8_pp2_pme6 | pme_spread_seconds | 0.443000 | n/a |
| split8_pp2_pme6 | pme_gather_seconds | 0.372000 | n/a |
| split8_pp2_pme6 | pme_3d_fft_seconds | 1.319000 | n/a |
| split8_pp3_pme5 | real_wall_seconds | 3.096000 | n/a |
| split8_pp3_pme5 | force_seconds | 2.367000 | n/a |
| split8_pp3_pme5 | pme_mesh_seconds | 2.399000 | n/a |
| split8_pp3_pme5 | update_seconds | 0.099000 | n/a |
| split8_pp3_pme5 | nb_f_kernel_seconds | 1.534000 | n/a |
| split8_pp3_pme5 | pme_spread_seconds | 0.392000 | n/a |
| split8_pp3_pme5 | pme_gather_seconds | 0.350000 | n/a |
| split8_pp3_pme5 | pme_3d_fft_seconds | 1.488000 | n/a |
| split8_pp4_pme4 | real_wall_seconds | 3.341000 | n/a |
| split8_pp4_pme4 | force_seconds | 1.769000 | n/a |
| split8_pp4_pme4 | pme_mesh_seconds | 2.882000 | n/a |
| split8_pp4_pme4 | update_seconds | 0.115000 | n/a |
| split8_pp4_pme4 | nb_f_kernel_seconds | 1.069000 | n/a |
| split8_pp4_pme4 | pme_spread_seconds | 0.482000 | n/a |
| split8_pp4_pme4 | pme_gather_seconds | 0.400000 | n/a |
| split8_pp4_pme4 | pme_3d_fft_seconds | 1.795000 | n/a |
| split8_pp5_pme3 | real_wall_seconds | 3.812000 | n/a |
| split8_pp5_pme3 | force_seconds | 1.506000 | n/a |
| split8_pp5_pme3 | pme_mesh_seconds | 3.330000 | n/a |
| split8_pp5_pme3 | update_seconds | 0.121000 | n/a |
| split8_pp5_pme3 | nb_f_kernel_seconds | 0.905000 | n/a |
| split8_pp5_pme3 | pme_spread_seconds | 0.455000 | n/a |
| split8_pp5_pme3 | pme_gather_seconds | 0.412000 | n/a |
| split8_pp5_pme3 | pme_3d_fft_seconds | 2.220000 | n/a |
| split8_pp6_pme2 | real_wall_seconds | 5.217000 | n/a |
| split8_pp6_pme2 | force_seconds | 1.457000 | n/a |
| split8_pp6_pme2 | pme_mesh_seconds | 4.580000 | n/a |
| split8_pp6_pme2 | update_seconds | 0.142000 | n/a |
| split8_pp6_pme2 | nb_f_kernel_seconds | 0.805000 | n/a |
| split8_pp6_pme2 | pme_spread_seconds | 0.614000 | n/a |
| split8_pp6_pme2 | pme_gather_seconds | 0.503000 | n/a |
| split8_pp6_pme2 | pme_3d_fft_seconds | 3.089000 | n/a |
