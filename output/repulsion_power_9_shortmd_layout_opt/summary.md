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
- modes: `generic, specialized`

## Notes

- `real_wall_seconds` comes from the `Time:` line and is the metric to use for final speed claims.
- For layouts with separate PME ranks, `Force`, `PME mesh`, and related wallcycle rows overlap across ranks and are not additive wall shares.
- `NB F kernel` remains useful for PP-kernel comparison inside the same layout, but not as a total-wall decomposition term for PME-split layouts.

## gate_h_dense_salt_polymer_2x2x2

| layout | total threads | ntmpi | npme | ntomp | ntomp_pme | generic ns/day | specialized ns/day | specialized/generic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| omp12 | 12 | 1 | 0 | 12 | 12 | 146.940 | 153.761 | 1.046 |
| omp2 | 2 | 1 | 0 | 2 | 2 | 60.793 | 68.103 | 1.120 |
| omp6 | 6 | 1 | 0 | 6 | 6 | 155.767 | 172.017 | 1.104 |
| split12_pp6_pme6 | 12 | 2 | 1 | 6 | 6 | 234.906 | 233.601 | 0.994 |

| layout | metric | generic | specialized | generic/specialized time ratio |
| --- | --- | --- | --- | --- |
| omp12 | real_wall_seconds | 2.940000 | 2.810000 | 1.046 |
| omp12 | force_seconds | 0.685000 | 0.560000 | 1.223 |
| omp12 | pme_mesh_seconds | 1.969000 | 1.960000 | 1.005 |
| omp12 | update_seconds | 0.050000 | 0.052000 | 0.962 |
| omp12 | nb_f_kernel_seconds | 0.417000 | 0.288000 | 1.448 |
| omp12 | pme_spread_seconds | 0.380000 | 0.384000 | 0.990 |
| omp12 | pme_gather_seconds | 0.405000 | 0.430000 | 0.942 |
| omp12 | pme_3d_fft_seconds | 1.118000 | 1.087000 | 1.029 |
| omp2 | real_wall_seconds | 7.107000 | 6.344000 | 1.120 |
| omp2 | force_seconds | 2.877000 | 2.147000 | 1.340 |
| omp2 | pme_mesh_seconds | 3.906000 | 3.880000 | 1.007 |
| omp2 | update_seconds | 0.026000 | 0.027000 | 0.963 |
| omp2 | nb_f_kernel_seconds | 2.012000 | 1.280000 | 1.572 |
| omp2 | pme_spread_seconds | 0.307000 | 0.307000 | 1.000 |
| omp2 | pme_gather_seconds | 0.907000 | 0.888000 | 1.021 |
| omp2 | pme_3d_fft_seconds | 2.471000 | 2.456000 | 1.006 |
| omp6 | real_wall_seconds | 2.774000 | 2.512000 | 1.104 |
| omp6 | force_seconds | 1.064000 | 0.806000 | 1.320 |
| omp6 | pme_mesh_seconds | 1.530000 | 1.527000 | 1.002 |
| omp6 | update_seconds | 0.019000 | 0.019000 | 1.000 |
| omp6 | nb_f_kernel_seconds | 0.727000 | 0.465000 | 1.563 |
| omp6 | pme_spread_seconds | 0.150000 | 0.152000 | 0.987 |
| omp6 | pme_gather_seconds | 0.378000 | 0.374000 | 1.011 |
| omp6 | pme_3d_fft_seconds | 0.918000 | 0.925000 | 0.992 |
| split12_pp6_pme6 | real_wall_seconds | 1.839000 | 1.849000 | 0.995 |
| split12_pp6_pme6 | force_seconds | 1.104000 | 0.846000 | 1.305 |
| split12_pp6_pme6 | pme_mesh_seconds | 1.649000 | 1.670000 | 0.987 |
| split12_pp6_pme6 | update_seconds | 0.036000 | 0.036000 | 1.000 |
| split12_pp6_pme6 | nb_f_kernel_seconds | 0.757000 | 0.498000 | 1.520 |
| split12_pp6_pme6 | pme_spread_seconds | 0.172000 | 0.173000 | 0.994 |
| split12_pp6_pme6 | pme_gather_seconds | 0.421000 | 0.430000 | 0.979 |
| split12_pp6_pme6 | pme_3d_fft_seconds | 0.970000 | 0.979000 | 0.991 |
