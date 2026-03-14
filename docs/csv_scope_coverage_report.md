# CSV Scope Coverage Report

## Status

현재 coverage baseline 결과는 다음과 같다.

- unique SMILES coverage: `0 / 6042`
- row-weighted coverage: `0 / 6270`
- release readiness: `not_ready`

이 결과는 중요하다. 현재는 일부 carbonyl subset이 parse와 atom typing을 통과해서 PT4까지 내려간다. 하지만 coverage는 여전히 `0%`다. 이유는 adapter 안정화가 덜 됐고, carbonyl family의 bonded Class2 term이 아직 비어 있기 때문이다.

## Machine-Readable Evidence

Coverage audit results:

- [coverage_audit_results.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/csv_scope_audit/coverage_audit_results.json)

Coverage summary:

- [coverage_audit_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/csv_scope_audit/coverage_audit_summary.json)

Scope manifests:

- [simulation_trajectory_aggregate_snapshot.json](/home/kiket/바탕화면/test/GROMACS_PCFF/data_manifests/simulation_trajectory_aggregate_snapshot.json)
- [simulation_trajectory_aggregate_unique_smiles.json](/home/kiket/바탕화면/test/GROMACS_PCFF/data_manifests/simulation_trajectory_aggregate_unique_smiles.json)
- [simulation_trajectory_aggregate_row_map.json](/home/kiket/바탕화면/test/GROMACS_PCFF/data_manifests/simulation_trajectory_aggregate_row_map.json)

## Observed Failure Distribution

Unique-SMILES 기준:

- `parse_failure`: `56`
- `chemical_perception_failure`: `0`
- `atom_typing_failure`: `5984`
- `parameter_assignment_failure`: `2`
- `nonbonded_assignment_failure`: `0`
- `emitter_export_failure`: `0`

Row-weighted 기준:

- `parse_failure`: `60`
- `atom_typing_failure`: `6206`
- `parameter_assignment_failure`: `4`
- 나머지 failure class: 모두 `0`

Observed failure codes:

- `csv_smiles_adapter_failure`: `56`
- `unsupported_carbonyl_chemistry`: `5915`
- `missing_parameter`: `2`
- `unsupported_element`: `55`
- `unsupported_component_family`: `11`
- `unsupported_aromatic_sp2_ring`: `2`
- `unsupported_resonance_encoding`: `1`

## What The Current Result Actually Means

현재 파이프라인 contract:

- PT1 parser supported input formats:
  - `mol_v2000`
  - `sdf`
  - `mol2`
  - `pdb`
- PT7 polymer workflow input:
  - `mol_v2000` only

근거:

- [typing_ir_spec.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/typing_ir_spec.md)
- [polymer_typing_workflow.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/polymer_typing_workflow.md)
- [formats.py](/home/kiket/바탕화면/test/GROMACS_PCFF/src/typing_ir/formats.py)
- [engine.py](/home/kiket/바탕화면/test/GROMACS_PCFF/src/polymer_workflow/engine.py)
- [worker.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_csv_scope_audit/worker.py)

이번 audit에서는 CSV의 pSMILES/SMILES를 `mol2`로 바꾸는 deterministic adapter를 추가로 사용했다.

따라서 현재 결론은 이렇게 바뀐다.

- CSV scope는 더 이상 전부 parse 단계에서 막히지 않는다.
- 현재 release blocker의 본질은 입력 부재가 아니라 **chemistry support 부족**이다.
- 특히 carbonyl-containing polymer chemistry가 overwhelming majority를 차지하며, 대부분은 여전히 PT3 typing rule scope 밖에 있다.

parse failure `56`건도 무시하면 안 된다. 현재는 `mixed formal_charge canonicalization` 버그는 제거됐지만, adapter 단계의 다른 구조 생성 실패가 남아 있다. 즉 parse 수치가 그대로라는 사실 자체가 “앞단 결함이 아직 구조적”이라는 증거다.

새로 확인된 사실:

- `csv_scope_smiles_000001` (`*C(=O)CCCC(=O)OC(C)CO*`)
- `csv_scope_smiles_000002` (`*C(=O)COCC(=O)OC(C)CO*`)

이 두 항목은 이제 `atom_typing_failure`가 아니라 `parameter_assignment_failure`로 내려간다.

즉:

- `csv_scope_pysoftk_aldehyde_polyester` 분류와 atom family assignment는 실제로 동작한다.
- 하지만 PT4는 `angle(c2|c=|o_1)` 또는 `angle(c=|c2|o_2)` 같은 carbonyl angle signature에서 멈춘다.
- 더 아래의 dihedral Class2 cross-term도 여전히 큰 구멍이다.

## What This Report Does Not Claim

이 문서는 다음을 주장하지 않는다.

- CSV snapshot chemistry가 support된다는 주장
- 일부 chemistry가 사실상 거의 다 될 것이라는 추측
- current PT7/PT8 supported examples가 CSV scope readiness를 대신한다는 주장

그 주장은 근거가 없다.

## Immediate Release-Target Gap

현재 release target과 구현 사이의 가장 작은 실질 격차는 이것이다.

- `csv_smiles_adapter_failure` 56건을 재현 가능하게 줄이거나 분류할 수 있을 만큼, `pysoftk` 입력 어댑터를 안정화해야 한다.
- carbonyl chemistry는 이제 일부가 PT4까지 내려가지만, `pcff.frc` 기반 reference에서 빠지는 bond-angle / dihedral cross-term을 어떻게 다룰지 source-of-truth를 먼저 고정해야 한다.

이 둘 중 하나라도 해결되지 않으면, downstream release target을 이 CSV snapshot으로 정의하면서 동시에 `typing/export coverage ready`라고 말할 수 없다.
