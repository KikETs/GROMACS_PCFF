# CPU Exact r-RESPA Current-Binary Revalidation, 2026-04-24

## Scope

This artifact revalidates the current CPU-only exact-r-RESPA claim path after the CPU speedup/update
changes on branch `exact-respa-cpu-only-speedups-20260422`.

Claimable candidate runtime environment:

- `GMX_PCFF_EXACT_RESPA_NATIVE_MULTI=1`
- `GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_OWNER_STEP_FALLBACK=1`
- `GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_SPLIT_OWNER_OUTPUTS=0`
- `GMX_PCFF_EXACT_RESPA_UPDATE_OMP=1`
- `GMX_PCFF_EXACT_RESPA_UPDATE_DIRECT_FASTPATH=1`
- `GMX_PCFF_EXACT_RESPA_FUSED_INITIAL_DRIFT=0`

Baseline:

- per-contribution launch path
- update OpenMP disabled
- update direct fast path disabled
- fused initial drift disabled

## Results

| Check | Fixture | Steps | Verdict | Key deltas |
| --- | --- | ---: | --- | --- |
| Small deterministic parity | `small_oligomer_gate_a_current_cpu` | 2000 | PASS | total force `0.0`, per-level force `0.0`, energy `0.0`, GRO coord `0.0` |
| Medium NVT parity | `gate_h_dense_salt_polymer_2x2x2_medium_nvt_current_cpu` | 20000 | PASS | total force `0.0`, per-level force `0.0`, energy `0.0`, GRO coord `0.0` |
| Restart continuity | both fixtures | probe 4 base steps | PASS | same-coordinate probe force/energy/GRO deltas `0.0` |
| Virial / pressure tensor | both fixtures | all dumped EDR frames | PASS | max `Pressure`, `Vir-*`, `Pres-*` delta `0.0` |

## Artifacts

- `small_deterministic_native_multi_ownerfallback_update/report.json`
- `small_deterministic_native_multi_ownerfallback_update/report.tsv`
- `medium_nvt_dense_salt_polymer_ownerfallback_update/report.json`
- `medium_nvt_dense_salt_polymer_ownerfallback_update/report.tsv`

## Boundary

This supports the current-binary CPU exact-r-RESPA mechanical/runtime claim for the audited fixtures
and runtime shape. It does not claim conductivity-production readiness, GPU/hybrid readiness, or
LAMMPS-vs-GROMACS transport parity.
