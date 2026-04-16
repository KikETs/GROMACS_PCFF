# Repulsion-Power-9 SIMD Short-MD CPU Benchmark

This benchmark is designed to exercise the admitted CPU short-range nonbonded kernel path without exact r-RESPA pair splitting.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `/home/kiket/Desktop/test/GROMACS_PCFF/build_subcounters/bin/gmx`
- steps per run: `4000`
- repeats per point: `5`
- pin mode: `on`
- DLB mode: `no`
- alternate mode order: `False`
- warmup cycles per ntomp: `0`
- modes: `generic, specialized`

## Notes

- This is a non-MTS short-MD shape.
- `NB F kernel` comes from the wallcycle subcounter breakdown when the binary was built with `GMX_CYCLE_SUBCOUNTERS=ON`.
- If `NB F kernel` is `n/a`, the selected binary did not emit subcounters for that run.

## gate_h_dense_salt_polymer_2x2x2

| ntomp | generic ns/day | specialized ns/day | specialized/generic |
| --- | --- | --- | --- |
| 1 | 15.277 | 17.232 | 1.128 |
| 2 | 11.535 | 16.752 | 1.452 |
| 6 | 3.941 | 6.032 | 1.531 |

| ntomp | component | generic | specialized | generic/specialized time ratio |
| --- | --- | --- | --- | --- |
| 1 | force_seconds | 4.681000 | 3.512000 | 1.333 |
| 1 | nb_f_kernel_seconds | 3.298000 | 2.102000 | 1.569 |
| 1 | bonded_force_seconds | 1.379000 | 1.395000 | n/a |
| 1 | update_seconds | 0.037000 | 0.049000 | n/a |
| 1 | total_wallcycle_seconds | 11.314000 | 10.030000 | 1.128 |
| 2 | force_seconds | 5.097000 | 2.904000 | 1.755 |
| 2 | nb_f_kernel_seconds | 3.386000 | 1.470000 | 2.303 |
| 2 | bonded_force_seconds | 1.175000 | 1.026000 | n/a |
| 2 | update_seconds | 0.349000 | 0.313000 | n/a |
| 2 | total_wallcycle_seconds | 14.985000 | 10.318000 | 1.452 |
| 6 | force_seconds | 9.673000 | 6.140000 | 1.575 |
| 6 | nb_f_kernel_seconds | 2.316000 | 1.356000 | 1.708 |
| 6 | bonded_force_seconds | 2.299000 | 1.741000 | n/a |
| 6 | update_seconds | 4.010000 | 2.472000 | n/a |
| 6 | total_wallcycle_seconds | 43.860000 | 28.654000 | 1.531 |
