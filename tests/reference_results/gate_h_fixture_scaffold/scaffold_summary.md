# Gate H Scaffold Summary

Date: `2026-04-01`

## What Was Generated

Command lines:

```bash
python3 /home/kiket/Desktop/test/GROMACS_PCFF/tools/pcff_respa_parity/scaffold_gate_h_fixture.py \
  --seed-system dense_oligomer \
  --system-id gate_h_dense_oligomer_2x2x2 \
  --replicate 2 2 2 \
  --out /home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_fixture_scaffold

python3 /home/kiket/Desktop/test/GROMACS_PCFF/tools/pcff_respa_parity/scaffold_gate_h_fixture.py \
  --seed-system dense_salt_polymer \
  --system-id gate_h_dense_salt_polymer_2x2x2 \
  --replicate 2 2 2 \
  --out /home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_fixture_scaffold
```

Generated artifacts:

- [gate_h_dense_oligomer_2x2x2 manifest](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_fixture_scaffold/gate_h_dense_oligomer_2x2x2/fixture_manifest.json)
- [gate_h_dense_oligomer_2x2x2 topology](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_fixture_scaffold/gate_h_dense_oligomer_2x2x2/generated/topol.top)
- [gate_h_dense_oligomer_2x2x2 GRO](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_fixture_scaffold/gate_h_dense_oligomer_2x2x2/generated/system.gro)
- [gate_h_dense_salt_polymer_2x2x2 manifest](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_fixture_scaffold/gate_h_dense_salt_polymer_2x2x2/fixture_manifest.json)
- [gate_h_dense_salt_polymer_2x2x2 topology](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_fixture_scaffold/gate_h_dense_salt_polymer_2x2x2/generated/topol.top)
- [gate_h_dense_salt_polymer_2x2x2 GRO](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_h_fixture_scaffold/gate_h_dense_salt_polymer_2x2x2/generated/system.gro)

## Result

- `gate_h_dense_oligomer_2x2x2`
  - `3072 atoms`
  - box `4.4 x 4.4 x 4.4 nm`
  - TP0 size fit: `true`
  - TP0 box fit: `true`
  - status: `best neutral Gate H scaffold`

- `gate_h_dense_salt_polymer_2x2x2`
  - `2160 atoms`
  - box `4.4 x 4.4 x 4.4 nm`
  - TP0 size fit: `true`
  - TP0 box fit: `true`
  - status: `scaffold only; do not use for Gate H runs until charged TP1 instability is resolved`

## Important Constraint

The charged scaffold is not a run approval.

Evidence:

- [validation_report_tp1.md](/home/kiket/Desktop/test/GROMACS_PCFF/docs/validation_report_tp1.md)
- [tp1_status.json](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/transport_protocol_metadata/tp1_status.json)

That thread is still physically blocked by thermal runaway on the smaller authoritative charged seed. Scaling an unstable seed is weak logic.

## Next Action

1. Start Gate H-neutral fixture bring-up from `gate_h_dense_oligomer_2x2x2`.
2. Freeze an exact-r-RESPA transport addendum before claiming TP0-compliant Gate H semantics, because TP0 currently marks RESPA as deferred.
3. Keep charged Gate H blocked until the `dense_salt_polymer` stability thread is fixed at seed scale.
