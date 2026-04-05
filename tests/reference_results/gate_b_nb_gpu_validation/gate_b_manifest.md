# Gate B Oracle Comparison

- Status: PASS
- Gate C allowed: True
- gmx: `/home/kiket/Desktop/test/GROMACS_PCFF/build_gateb_cuda/bin/gmx`
- precision: `mixed`
- GPU support: `CUDA`
- ntmpi / ntomp: `1` / `1`
- DLB: `no`
- PME ranks: `0`
- Binary reproducibility supported: `False`

## Reproducibility Notes

- Repro note: Binary reproducibility (-reprod) is not enabled because GROMACS rejects -nb gpu together with -reprod.
- Repro note: Determinism is constrained with single-rank execution, single OpenMP thread, DLB disabled, and measured repeated-run GPU noise floors.

## Blocking Reasons

- None

## Systems

### small_oligomer

- Gate B assessment: `PASS`
- Main run return code: `0`
- First nonzero comparison field: `{'field': 'total_force_comparison', 'details': {'step': 0, 'highest_active_level': 2, 'expected_vector_sum': [-0.107147216796875, 0.1064453125, -0.01470184326171875], 'actual_vector_sum': [-0.10711669921875, 0.10619354248046875, -0.014678955078125], 'component_abs_deltas': [3.0517578125e-05, 0.00025177001953125, 2.288818359375e-05]}}`
- Artifact root: `/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_b_nb_gpu_validation/small_oligomer`
- Command script: `/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_b_nb_gpu_validation/small_oligomer/run_commands.sh`

### small_salt_polymer_box

- Gate B assessment: `PASS`
- Main run return code: `0`
- First nonzero comparison field: `{'field': 'total_force_comparison', 'details': {'step': 0, 'highest_active_level': 2, 'expected_vector_sum': [-0.07440376281738281, 1.0354995727539062, -1.04656982421875], 'actual_vector_sum': [-0.07407569885253906, 1.0353164672851562, -1.0461769104003906], 'component_abs_deltas': [0.00032806396484375, 0.00018310546875, 0.000392913818359375]}}`
- Artifact root: `/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_b_nb_gpu_validation/small_salt_polymer_box`
- Command script: `/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_b_nb_gpu_validation/small_salt_polymer_box/run_commands.sh`

