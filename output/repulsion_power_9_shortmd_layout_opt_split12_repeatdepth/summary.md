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
- modes: `generic, specialized`

## Notes

- `real_wall_seconds` comes from the `Time:` line and is the metric to use for final speed claims.
- For layouts with separate PME ranks, `Force`, `PME mesh`, and related wallcycle rows overlap across ranks and are not additive wall shares.
- `NB F kernel` remains useful for PP-kernel comparison inside the same layout, but not as a total-wall decomposition term for PME-split layouts.

## gate_h_dense_salt_polymer_2x2x2

| layout | total threads | ntmpi | npme | ntomp | ntomp_pme | generic ns/day | specialized ns/day | specialized/generic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split12_pp6_pme6 | 12 | 2 | 1 | 6 | 6 | 231.981 | 235.065 | 1.013 |

| layout | metric | generic | specialized | generic/specialized time ratio |
| --- | --- | --- | --- | --- |
| split12_pp6_pme6 | real_wall_seconds | 1.862500 | 1.838000 | 1.013 |
| split12_pp6_pme6 | force_seconds | 1.090500 | 0.838000 | 1.301 |
| split12_pp6_pme6 | pme_mesh_seconds | 1.679000 | 1.665000 | 1.008 |
| split12_pp6_pme6 | update_seconds | 0.037000 | 0.035500 | 1.042 |
| split12_pp6_pme6 | nb_f_kernel_seconds | 0.748500 | 0.496500 | 1.508 |
| split12_pp6_pme6 | pme_spread_seconds | 0.175500 | 0.174500 | 1.006 |
| split12_pp6_pme6 | pme_gather_seconds | 0.430500 | 0.421000 | 1.023 |
| split12_pp6_pme6 | pme_3d_fft_seconds | 0.986000 | 0.982000 | 1.004 |
