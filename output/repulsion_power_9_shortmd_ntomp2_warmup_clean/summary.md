# Repulsion-Power-9 SIMD Short-MD CPU Benchmark

This benchmark is designed to exercise the admitted CPU short-range nonbonded kernel path without exact r-RESPA pair splitting.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `../../build_subcounters/bin/gmx`
- steps per run: `2000`
- repeats per point: `3`
- pin mode: `on`
- DLB mode: `no`
- alternate mode order: `True`
- warmup cycles per ntomp: `1`
- modes: `generic, specialized`

## Notes

- This is a non-MTS short-MD shape.
- `NB F kernel` comes from the wallcycle subcounter breakdown when the binary was built with `GMX_CYCLE_SUBCOUNTERS=ON`.
- If `NB F kernel` is `n/a`, the selected binary did not emit subcounters for that run.

## gate_h_dense_salt_polymer_2x2x2

| ntomp | generic ns/day | specialized ns/day | specialized/generic |
| --- | --- | --- | --- |
| 2 | 52.109 | 58.864 | 1.130 |

| ntomp | component | generic | specialized | generic/specialized time ratio |
| --- | --- | --- | --- | --- |
| 2 | force_seconds | 0.645000 | 0.478000 | 1.349 |
| 2 | nb_f_kernel_seconds | 0.451000 | 0.285000 | 1.582 |
| 2 | bonded_force_seconds | 0.187000 | 0.185000 | n/a |
| 2 | update_seconds | 0.006000 | 0.006000 | n/a |
| 2 | total_wallcycle_seconds | 1.659000 | 1.469000 | 1.129 |
