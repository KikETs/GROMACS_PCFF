# Gate C Oracle Comparison

- Status: PASS
- Gate D allowed: True
- gmx: `/home/kiket/Desktop/test/GROMACS_PCFF/build_gateb_cuda/bin/gmx`
- precision: `mixed`
- GPU support: `CUDA`
- ntmpi / ntomp: `1` / `1`
- DLB: `no`
- PME ranks: `0`

## Blocking Reasons

- None

## Systems

### small_oligomer

- Gate C assessment: `PASS`
- Main run return code: `0`
- Event order identical: `True`
- Total force max abs component delta: `0.000385284423828125`
- Per-level force max abs component delta: `0.001434326171875`
- First failure field: `None`
- First mismatching term: `None`
- Artifact root: `/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_c_nb_bonded_gpu_validation/small_oligomer`
- Command script: `/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_c_nb_bonded_gpu_validation/small_oligomer/run_commands.sh`

### small_salt_polymer_box

- Gate C assessment: `PASS`
- Main run return code: `0`
- Event order identical: `True`
- Total force max abs component delta: `0.00089263916015625`
- Per-level force max abs component delta: `0.0020904541015625`
- First failure field: `None`
- First mismatching term: `None`
- Artifact root: `/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_c_nb_bonded_gpu_validation/small_salt_polymer_box`
- Command script: `/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_c_nb_bonded_gpu_validation/small_salt_polymer_box/run_commands.sh`

