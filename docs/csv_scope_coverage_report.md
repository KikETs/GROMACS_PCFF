# CSV Scope Coverage Report

## Status

현재 coverage baseline 결과는 다음과 같다.

- unique SMILES coverage: `0 / 6042`
- row-weighted coverage: `0 / 6270`
- release readiness: `not_ready`

이 결과는 실패가 아니다. 현재 구현 범위의 정직한 측정 결과다.

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

- `parse_failure`: `6042`
- `chemical_perception_failure`: `0`
- `atom_typing_failure`: `0`
- `parameter_assignment_failure`: `0`
- `nonbonded_assignment_failure`: `0`
- `emitter_export_failure`: `0`

Row-weighted 기준:

- `parse_failure`: `6270`
- 나머지 failure class: 모두 `0`

Observed failure code:

- `unsupported_csv_smiles_input`: `6042`

## Why Everything Currently Stops At Parse

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

반면 이번 release target scope는 CSV의 SMILES text다. 현재 저장소에는 이 snapshot용 deterministic SMILES-to-structure adapter가 없고, CSV row를 PT1 typed IR 입력으로 바꾸는 공식 경로도 없다.

그래서 현재 baseline은 모든 unique SMILES를 `parse_failure`로 분류한다. 이건 silent skip이 아니라 explicit accounting이다.

## What This Report Does Not Claim

이 문서는 다음을 주장하지 않는다.

- CSV snapshot chemistry가 support된다는 주장
- 일부 chemistry가 사실상 동작할 것이라는 추측
- current PT7/PT8 supported examples가 CSV scope readiness를 대신한다는 주장

그 주장은 근거가 없다.

## Immediate Release-Target Gap

현재 release target과 구현 사이의 가장 작은 실질 격차는 이것이다.

- CSV snapshot의 SMILES/PSMILES 표현을 PT1 typed IR chain으로 deterministic하게 연결하는 공식 입력 경로가 없다.

이 격차가 해결되기 전에는, downstream release target을 이 CSV snapshot으로 정의하면서 동시에 `typing/export coverage ready`라고 말할 수 없다.
