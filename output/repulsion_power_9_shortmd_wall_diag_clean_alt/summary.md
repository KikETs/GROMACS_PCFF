# Repulsion-Power-9 SIMD Short-MD CPU Benchmark

This benchmark is designed to exercise the admitted CPU short-range nonbonded kernel path without exact r-RESPA pair splitting.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `/home/kiket/Desktop/test/GROMACS_PCFF/build_subcounters/bin/gmx`
- steps per run: `2000`
- repeats per point: `4`
- pin mode: `on`
- DLB mode: `no`
- alternate mode order: `True`
- modes: `generic, specialized`

## Notes

- This is a non-MTS short-MD shape.
- `NB F kernel` comes from the wallcycle subcounter breakdown when the binary was built with `GMX_CYCLE_SUBCOUNTERS=ON`.
- If `NB F kernel` is `n/a`, the selected binary did not emit subcounters for that run.

## gate_h_dense_salt_polymer_2x2x2

| ntomp | generic ns/day | specialized ns/day | specialized/generic |
| --- | --- | --- | --- |
| 2 | 46.822 | 35.267 | 0.753 |
| 6 | 122.441 | 137.860 | 1.126 |

| ntomp | component | generic | specialized | generic/specialized time ratio |
| --- | --- | --- | --- | --- |
| 2 | force_seconds | 0.693500 | 1.105500 | 0.627 |
| 2 | nb_f_kernel_seconds | 0.481500 | 0.604500 | 0.797 |
| 2 | bonded_force_seconds | 0.202500 | 0.409000 | n/a |
| 2 | update_seconds | 0.007000 | 0.096500 | n/a |
| 2 | total_wallcycle_seconds | 1.846000 | 3.818500 | 0.483 |
| 6 | force_seconds | 0.248000 | 0.186000 | 1.333 |
| 6 | nb_f_kernel_seconds | 0.169000 | 0.107500 | 1.572 |
| 6 | bonded_force_seconds | 0.072000 | 0.070500 | n/a |
| 6 | update_seconds | 0.004000 | 0.004000 | n/a |
| 6 | total_wallcycle_seconds | 0.706000 | 0.627000 | 1.126 |
