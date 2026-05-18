# Repulsion-Power-9 SIMD Short-MD CPU Benchmark

This benchmark is designed to exercise the admitted CPU short-range nonbonded kernel path without exact r-RESPA pair splitting.

## Host

- hostname: `KikET`
- cpu: `AMD Ryzen 9 9900X 12-Core Processor`
- gmx: `../../build_subcounters/bin/gmx`
- steps per run: `10000`
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
| 12 | 146.769 | 153.206 | 1.044 |

| ntomp | component | generic | specialized | generic/specialized time ratio |
| --- | --- | --- | --- | --- |
| 12 | force_seconds | 0.686000 | 0.559000 | 1.227 |
| 12 | nb_f_kernel_seconds | 0.416000 | 0.292000 | 1.425 |
| 12 | bonded_force_seconds | 0.192000 | 0.192000 | n/a |
| 12 | update_seconds | 0.052000 | 0.052000 | n/a |
| 12 | total_wallcycle_seconds | 2.944000 | 2.820000 | 1.044 |
