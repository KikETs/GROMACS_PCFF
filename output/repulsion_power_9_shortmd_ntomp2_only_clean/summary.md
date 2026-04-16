# Repulsion-Power-9 SIMD Short-MD CPU Benchmark

This benchmark is designed to exercise the admitted CPU short-range nonbonded kernel path without exact r-RESPA pair splitting.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `/home/kiket/Desktop/test/GROMACS_PCFF/build_subcounters/bin/gmx`
- steps per run: `2000`
- repeats per point: `6`
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
| 2 | 50.300 | 54.895 | 1.091 |

| ntomp | component | generic | specialized | generic/specialized time ratio |
| --- | --- | --- | --- | --- |
| 2 | force_seconds | 0.656000 | 0.497000 | 1.320 |
| 2 | nb_f_kernel_seconds | 0.456500 | 0.295000 | 1.547 |
| 2 | bonded_force_seconds | 0.190500 | 0.193000 | n/a |
| 2 | update_seconds | 0.006500 | 0.006500 | n/a |
| 2 | total_wallcycle_seconds | 1.718500 | 1.574500 | 1.091 |
