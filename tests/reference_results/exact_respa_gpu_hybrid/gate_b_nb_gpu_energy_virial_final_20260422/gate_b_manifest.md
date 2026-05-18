# Gate B Oracle Comparison

- Status: PASS
- Gate C allowed: True
- gmx: `../../../../build_gateb_cuda/bin/gmx`
- precision: `mixed`
- GPU support: `CUDA`
- ntmpi / ntomp: `1` / `2`
- DLB: `no`
- PME ranks: `0`
- Binary reproducibility supported: `False`

## Reproducibility Notes

- Repro note: Binary reproducibility (-reprod) is not enabled because GROMACS rejects -nb gpu together with -reprod.
- Repro note: Determinism is constrained with single-rank execution, the recorded OpenMP thread count, DLB disabled, and measured repeated-run GPU noise floors.

## Blocking Reasons

- None

## Systems

### small_oligomer

- Gate B assessment: `PASS`
- Main run return code: `0`
- First nonzero comparison field: `{'field': 'total_force_comparison', 'details': {'step': 0, 'highest_active_level': 2, 'expected_vector_sum': [-0.107147216796875, 0.1064453125, -0.014705657958984375], 'actual_vector_sum': [-0.107513427734375, 0.10585784912109375, -0.0146942138671875], 'component_abs_deltas': [0.0003662109375, 0.00058746337890625, 1.1444091796875e-05]}}`
- Artifact root: `./small_oligomer`
- Command script: `./small_oligomer/run_commands.sh`

### small_salt_polymer_box

- Gate B assessment: `PASS`
- Main run return code: `0`
- First nonzero comparison field: `{'field': 'total_force_comparison', 'details': {'step': 0, 'highest_active_level': 2, 'expected_vector_sum': [-0.07440376281738281, 1.0354995727539062, -1.04656982421875], 'actual_vector_sum': [-0.07490730285644531, 1.0343399047851562, -1.0447235107421875], 'component_abs_deltas': [0.0005035400390625, 0.00115966796875, 0.0018463134765625]}}`
- Artifact root: `./small_salt_polymer_box`
- Command script: `./small_salt_polymer_box/run_commands.sh`

