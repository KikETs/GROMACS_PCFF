# Repulsion-Power-9 SIMD Short-MD CPU Benchmark

This benchmark is designed to exercise the admitted CPU short-range nonbonded kernel path without exact r-RESPA pair splitting.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `../../build_subcounters/bin/gmx`
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
| 6 | 135.975 | 144.188 | 1.060 |

| ntomp | component | generic | specialized | generic/specialized time ratio |
| --- | --- | --- | --- | --- |
| 6 | force_seconds | 0.233000 | 0.178500 | 1.305 |
| 6 | nb_f_kernel_seconds | 0.159000 | 0.104000 | 1.529 |
| 6 | bonded_force_seconds | 0.066500 | 0.067500 | n/a |
| 6 | update_seconds | 0.004000 | 0.004000 | n/a |
| 6 | total_wallcycle_seconds | 0.635500 | 0.600500 | 1.058 |
