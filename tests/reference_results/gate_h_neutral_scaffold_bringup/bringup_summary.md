# Gate H Neutral Scaffold Bring-up

Date: `2026-04-02`

System:

- [gate_h_dense_oligomer_2x2x2 scaffold manifest](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_fixture_scaffold/gate_h_dense_oligomer_2x2x2/fixture_manifest.json)

Commands:

```bash
python3 /home/kiket/Desktop/test/GROMACS_PCFF/tools/pcff_respa_parity/bringup_gate_h_scaffold.py \
  --gmx /home/kiket/Desktop/test/GROMACS_PCFF/build_gateb_cuda/bin/gmx \
  --scaffold-manifest /home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_fixture_scaffold/gate_h_dense_oligomer_2x2x2/fixture_manifest.json \
  --out /home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_neutral_scaffold_bringup
```

Artifacts:

- [bringup_result.json](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_neutral_scaffold_bringup/summaries/bringup_result.json)
- [run_commands.sh](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_neutral_scaffold_bringup/run_commands.sh)
- [mdrun_cpu.stderr](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_neutral_scaffold_bringup/logs/mdrun_cpu.stderr)
- [mdrun_gpu.stderr](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_neutral_scaffold_bringup/logs/mdrun_gpu.stderr)

## Result

- status: `PASS`
- scope: `short exact-r-RESPA bring-up only`
- single-rank layout: `ntmpi=1`, `ntomp=1`, `DLB=no`
- GPU mapping: `PP:0,PME:0`
- GPU shape: `nb gpu / bonded gpu / pme gpu / update gpu`

## Surviving Claims

- `grompp` succeeds on the 3072-atom neutral scaffold.
- CPU and full-GPU exact-r-RESPA both execute successfully.
- CPU and GPU event order both match the reference exact-r-RESPA schedule.
- CPU and GPU event traces match each other exactly.

## Nonzero Deltas

- total-force max component delta: `0.004052162170410156`
- per-level force max component delta: `0.0070953369140625`
- energy comparison max abs delta: `1.5 kJ/mol`
- first energy mismatch field: `#Surf*SurfTen`
- main explicit deltas:
  - `Coulomb (SR)`: up to `1.5 kJ/mol`
  - `Potential`: up to `1.1 kJ/mol`
  - `Total Energy`: up to `1.2 kJ/mol`
  - `Pressure`: up to `0.02`
  - `Coul. recip.`: up to `0.005 kJ/mol`

## Interpretation

This is enough to say the large neutral scaffold is runnable on the standalone exact-r-RESPA full-GPU path.

This is not enough to claim long-horizon statistical parity yet.

The remaining deltas are aggregate CPU-vs-GPU differences on a much larger system, not event-order failures. The next gate should measure whether these stay within a stable uncertainty budget over replicated longer runs, rather than pretending short-trajectory identity still matters.
