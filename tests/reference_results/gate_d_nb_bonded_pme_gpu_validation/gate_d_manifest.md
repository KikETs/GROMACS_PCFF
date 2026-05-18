# Gate D Oracle Comparison

- Status: PASS
- Gate E allowed: True
- gmx: `../../../build_gateb_cuda/bin/gmx`
- precision: `mixed`
- GPU support: `CUDA`
- ntmpi / ntomp: `1` / `1`
- npme flag used: `False`
- npme requested: `None`
- DLB: `no`
- PME ranks: `0`

## Blocking Reasons

- None

## Systems

### small_oligomer

- Gate D assessment: `PASS`
- Main run return code: `0`
- Event order identical: `True`
- Reciprocal force max abs component delta: `2.288818359375e-05`
- Reciprocal force characterization: `within_roundoff_proxy`
- Reciprocal force roundoff proxy bound: `7.901409480837174e-05`
- FFT-backend arithmetic evidence available: `False`
- Coul. recip. max abs delta: `0.0001000000000015433`
- CPU reciprocal/self/exclusion max abs delta: `1.33514404296875e-05`
- Layout report: `single-rank colocated PP+PME tasks on rank 0`
- First failure field: `None`
- Artifact root: `./small_oligomer`
- Command script: `./small_oligomer/run_commands.sh`

### small_salt_polymer_box

- Gate D assessment: `PASS`
- Main run return code: `0`
- Event order identical: `True`
- Reciprocal force max abs component delta: `0.0007291212677955627`
- Reciprocal force characterization: `fft_backend_arithmetic_chain`
- Reciprocal force roundoff proxy bound: `0.00028457870390852236`
- FFT-backend arithmetic evidence available: `True`
- Coul. recip. max abs delta: `0.0030000000000427463`
- CPU reciprocal/self/exclusion max abs delta: `2.288818359375e-05`
- Layout report: `single-rank colocated PP+PME tasks on rank 0`
- First failure field: `None`
- Artifact root: `./small_salt_polymer_box`
- Command script: `./small_salt_polymer_box/run_commands.sh`

