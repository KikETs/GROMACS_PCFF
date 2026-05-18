# Gate E Oracle Comparison

- Status: PASS
- Gate F allowed: True
- gmx: `../../../build_gateb_cuda/bin/gmx`
- precision: `mixed`
- GPU support: `CUDA`
- ntmpi / ntomp: `1` / `1`
- npme flag used: `False`
- DLB: `no`

## Blocking Reasons

- None

## Systems

### small_oligomer

- Gate E assessment: `PASS`
- Main run return code: `0`
- Event order identical: `True`
- Restart continuity: `pass`
- State max coordinate delta vs Gate D: `0.0`
- State max velocity delta vs Gate D: `0.0`
- State max coordinate delta vs Gate A: `0.0`
- State max velocity delta vs Gate A: `2.000000000002e-06`
- Coul. recip. max abs delta: `0.0001000000000015433`
- Reciprocal force characterization: `within_roundoff_proxy`
- Reciprocal force roundoff proxy bound: `7.901409480837174e-05`
- FFT-backend arithmetic evidence available: `False`
- Layout report: `single-rank colocated PP+PME tasks on rank 0`
- First failure field: `None`
- Artifact root: `./small_oligomer`
- Command script: `./small_oligomer/run_commands.sh`

### small_salt_polymer_box

- Gate E assessment: `PASS`
- Main run return code: `0`
- Event order identical: `True`
- Restart continuity: `pass`
- State max coordinate delta vs Gate D: `0.0`
- State max velocity delta vs Gate D: `1.000000000001e-06`
- State max coordinate delta vs Gate A: `0.0`
- State max velocity delta vs Gate A: `1.0000000000509601e-05`
- Coul. recip. max abs delta: `0.0030000000000427463`
- Reciprocal force characterization: `fft_backend_arithmetic_chain`
- Reciprocal force roundoff proxy bound: `0.00028457870390852236`
- FFT-backend arithmetic evidence available: `True`
- Layout report: `single-rank colocated PP+PME tasks on rank 0`
- First failure field: `None`
- Artifact root: `./small_salt_polymer_box`
- Command script: `./small_salt_polymer_box/run_commands.sh`

