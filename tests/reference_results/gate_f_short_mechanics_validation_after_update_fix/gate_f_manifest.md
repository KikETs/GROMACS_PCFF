# Gate F Short-Window Mechanics

- Status: PASS
- Gate G allowed: True
- gmx: `/home/kiket/Desktop/test/GROMACS_PCFF/build_gateb_cuda/bin/gmx`
- precision: `mixed`
- GPU support: `CUDA`
- Exact GPU bonded validation mode: `combined_kernel`
- ntmpi / ntomp: `1` / `1`
- DLB: `no`

## Blocking Reasons

- None

## Systems

### small_oligomer

- Status: `PASS`
- CPU total-energy drift envelope: `0.11199999999999477`
- GPU total-energy drift envelope: `0.11199999999999477`
- Envelope inflation: `0.0`
- Low-level noise resolution: `under_resolved_by_energy_terms`
- Force-store noise floor: `{'available': True, 'successful_run_count': 3, 'reference_run_id': 'gpu_full', 'max_abs_component_delta': 0.00048828125, 'max_abs_component_delta_by_bucket': {'final/0': 0.00048828125, 'initial/0': 0.00048828125, 'initial/1': 0.0, 'initial/2': 0.0, 'final/1': 0.0, 'final/2': 0.0}, 'worst_repeat': {'run_id': 'mdrun_gpu_repeat_1', 'max_abs_component_delta': 0.00048828125, 'first_nonzero_delta': {'key': [2, 'final', 0, 0], 'deltas': {'fx': -6.103515625e-05, 'fy': -1.52587890625e-05, 'fz': 3.814697265625e-06}, 'expected': {'fx': 850.9303588867188, 'fy': -35.6051025390625, 'fz': 12.117683410644531}, 'actual': {'fx': 850.9302978515625, 'fy': -35.60511779785156, 'fz': 12.117687225341797}}}}`
- Bonded reduction noise floor: `{'available': True, 'successful_run_count': 3, 'reference_run_id': 'gpu_full', 'max_abs_component_delta': 0.00048828125, 'max_abs_component_delta_by_bucket': {'after_reduce/0': 0.00048828125, 'before_reduce/0': 0.0, 'reduction_delta/0': 0.00048828125, 'nbat_output_buffer/0': 0.00048828125}, 'worst_repeat': {'run_id': 'mdrun_gpu_repeat_1', 'max_abs_component_delta': 0.00048828125, 'first_nonzero_delta': {'key': [2, 0, 'after_reduce', 0], 'deltas': {'fx': 0.0, 'fy': 1.52587890625e-05, 'fz': 1.9073486328125e-06}, 'expected': {'fx': 904.5487060546875, 'fy': -59.32267761230469, 'fz': 16.126388549804688}, 'actual': {'fx': 904.5487060546875, 'fy': -59.322662353515625, 'fz': 16.12639045715332}}}}`
- Restart comparison: `PASS`
- First failing observable: `None`
- Drift TSV: `/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_f_short_mechanics_validation_after_update_fix/small_oligomer/summaries/cpu_gpu_total_pressure_drift.tsv`
- Artifact root: `/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_f_short_mechanics_validation_after_update_fix/small_oligomer`

### small_salt_polymer_box

- Status: `PASS`
- CPU total-energy drift envelope: `0.61099999999999`
- GPU total-energy drift envelope: `0.61099999999999`
- Envelope inflation: `0.0`
- Low-level noise resolution: `consistent_with_energy_terms`
- Force-store noise floor: `{'available': True, 'successful_run_count': 3, 'reference_run_id': 'gpu_full', 'max_abs_component_delta': 0.08154296875, 'max_abs_component_delta_by_bucket': {'final/0': 0.0814208984375, 'initial/0': 0.08154296875, 'initial/1': 4.482269287109375e-05, 'initial/2': 0.0008087158203125, 'final/1': 4.482269287109375e-05, 'final/2': 0.0008087158203125}, 'worst_repeat': {'run_id': 'mdrun_gpu_repeat_1', 'max_abs_component_delta': 0.08154296875, 'first_nonzero_delta': {'key': [2, 'final', 0, 1], 'deltas': {'fx': 0.000244140625, 'fy': 0.0, 'fz': 0.0}, 'expected': {'fx': 2739.13134765625, 'fy': -2765.911376953125, 'fz': -895.2478637695312}, 'actual': {'fx': 2739.131591796875, 'fy': -2765.911376953125, 'fz': -895.2478637695312}}}}`
- Bonded reduction noise floor: `{'available': True, 'successful_run_count': 3, 'reference_run_id': 'gpu_full', 'max_abs_component_delta': 0.08154296875, 'max_abs_component_delta_by_bucket': {'after_reduce/0': 0.08154296875, 'before_reduce/0': 0.0, 'reduction_delta/0': 0.08154296875, 'nbat_output_buffer/0': 0.08154296875}, 'worst_repeat': {'run_id': 'mdrun_gpu_repeat_1', 'max_abs_component_delta': 0.08154296875, 'first_nonzero_delta': {'key': [4, 0, 'after_reduce', 0], 'deltas': {'fx': 0.0, 'fy': -1.52587890625e-05, 'fz': 0.0}, 'expected': {'fx': 1088.957763671875, 'fy': 169.47119140625, 'fz': 67.00653076171875}, 'actual': {'fx': 1088.957763671875, 'fy': 169.47117614746094, 'fz': 67.00653076171875}}}}`
- Restart comparison: `PASS`
- First failing observable: `None`
- Drift TSV: `/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_f_short_mechanics_validation_after_update_fix/small_salt_polymer_box/summaries/cpu_gpu_total_pressure_drift.tsv`
- Artifact root: `/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_f_short_mechanics_validation_after_update_fix/small_salt_polymer_box`

