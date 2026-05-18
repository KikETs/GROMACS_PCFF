# Repulsion-Power-9 SIMD Short-MD CPU Benchmark

This benchmark is designed to exercise the admitted CPU short-range nonbonded kernel path without exact r-RESPA pair splitting.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `../../build_subcounters/bin/gmx`
- steps per run: `4000`
- repeats per point: `5`
- pin mode: `on`
- DLB mode: `auto`
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
| 1 | 15.280 | 17.225 | 1.127 |
| 2 | 11.560 | 16.747 | 1.449 |
| 6 | 3.942 | 6.037 | 1.531 |

| ntomp | component | generic | specialized | generic/specialized time ratio |
| --- | --- | --- | --- | --- |
| 1 | force_seconds | 4.678000 | 3.525000 | 1.327 |
| 1 | nb_f_kernel_seconds | 3.262000 | 2.112000 | 1.545 |
| 1 | bonded_force_seconds | 1.370000 | 1.379000 | n/a |
| 1 | update_seconds | 0.047000 | 0.045000 | n/a |
| 1 | total_wallcycle_seconds | 11.311000 | 10.035000 | 1.127 |
| 2 | force_seconds | 5.099000 | 2.863000 | 1.781 |
| 2 | nb_f_kernel_seconds | 3.379000 | 1.396000 | 2.420 |
| 2 | bonded_force_seconds | 1.154000 | 1.050000 | n/a |
| 2 | update_seconds | 0.388000 | 0.285000 | n/a |
| 2 | total_wallcycle_seconds | 14.952000 | 10.321000 | 1.449 |
| 6 | force_seconds | 9.687000 | 6.184000 | 1.566 |
| 6 | nb_f_kernel_seconds | 2.438000 | 1.418000 | 1.719 |
| 6 | bonded_force_seconds | 2.332000 | 1.724000 | n/a |
| 6 | update_seconds | 3.940000 | 2.471000 | n/a |
| 6 | total_wallcycle_seconds | 43.851000 | 28.631000 | 1.532 |
