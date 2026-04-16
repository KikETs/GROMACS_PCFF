# Repulsion-Power-9 SIMD Short-MD CPU Benchmark

This benchmark is designed to exercise the admitted CPU short-range nonbonded kernel path without exact r-RESPA pair splitting.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `/home/kiket/Desktop/test/GROMACS_PCFF/build_subcounters/bin/gmx`
- steps per run: `200`
- repeats per point: `1`
- pin mode: `on`
- modes: `generic, specialized`

## Notes

- This is a non-MTS short-MD shape.
- `NB F kernel` comes from the wallcycle subcounter breakdown when the binary was built with `GMX_CYCLE_SUBCOUNTERS=ON`.
- If `NB F kernel` is `n/a`, the selected binary did not emit subcounters for that run.

## gate_h_dense_salt_polymer_2x2x2

| ntomp | generic ns/day | specialized ns/day | specialized/generic |
| --- | --- | --- | --- |
| 1 | 27.283 | 31.445 | 1.153 |
| 2 | 49.061 | 42.559 | 0.867 |
| 6 | 109.711 | 103.140 | 0.940 |

| ntomp | component | generic | specialized | generic/specialized time ratio |
| --- | --- | --- | --- | --- |
| 1 | force_seconds | 0.130000 | 0.095000 | 1.368 |
| 1 | nb_f_kernel_seconds | 0.094000 | 0.059000 | 1.593 |
| 1 | bonded_force_seconds | 0.036000 | 0.035000 | n/a |
| 1 | update_seconds | 0.001000 | 0.001000 | n/a |
| 1 | total_wallcycle_seconds | 0.318000 | 0.276000 | 1.152 |
| 2 | force_seconds | 0.067000 | 0.061000 | 1.098 |
| 2 | nb_f_kernel_seconds | 0.048000 | 0.038000 | 1.263 |
| 2 | bonded_force_seconds | 0.018000 | 0.023000 | n/a |
| 2 | update_seconds | 0.001000 | 0.001000 | n/a |
| 2 | total_wallcycle_seconds | 0.177000 | 0.204000 | 0.868 |
| 6 | force_seconds | 0.026000 | 0.021000 | 1.238 |
| 6 | nb_f_kernel_seconds | 0.018000 | 0.013000 | 1.385 |
| 6 | bonded_force_seconds | 0.007000 | 0.008000 | n/a |
| 6 | update_seconds | 0.000000 | 0.000000 | n/a |
| 6 | total_wallcycle_seconds | 0.079000 | 0.084000 | 0.940 |
