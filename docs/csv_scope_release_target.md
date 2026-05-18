# CSV Scope Release Target

## Status

이 문서는 `simulation-trajectory-aggregate.csv` snapshot을 현재 release target chemistry scope로 고정한다.

중요:

- scope boundary는 이 CSV snapshot 자체다.
- support target은 이 파일에 들어 있는 모든 **unique SMILES** 다.
- row-level duplicate는 먼저 unique-SMILES manifest로 deduplicate한다.
- 최종 downstream 목표는 이 snapshot에 대해 `100% typing/export coverage`다.
- 이 작업은 scope baseline만 추가한다. CSV 밖의 chemistry support는 추가하지 않는다.

## Frozen Snapshot Identity

Snapshot filename:

- `simulation-trajectory-aggregate.csv`

Frozen snapshot hash:

- `a67a8f86f1842cd9d35ffe6cce2de8a3cf3577635aed19564b910242dd226fcf`

Observed counts:

- row count: `6270`
- unique SMILES count: `6042`
- duplicate rows removed by dedup manifest: `228`

Source path used to build the checked-in manifests:

- `MY_PAPER_RELATED/LAMMPS_BATCH/data/simulation-trajectory-aggregate.csv`

중요한 점:

- workspace에는 같은 해시를 가진 동명 CSV 복제본이 여러 개 있다.
- 따라서 **path보다 hash가 authoritative** 하다.
- checked-in snapshot manifest는 generation 당시 발견한 동일-hash path들을 함께 기록한다.

## Checked-In Scope Artifacts

Snapshot manifest:

- [simulation_trajectory_aggregate_snapshot.json](../data_manifests/simulation_trajectory_aggregate_snapshot.json)

Unique SMILES manifest:

- [simulation_trajectory_aggregate_unique_smiles.json](../data_manifests/simulation_trajectory_aggregate_unique_smiles.json)

Row-to-unique map:

- [simulation_trajectory_aggregate_row_map.json](../data_manifests/simulation_trajectory_aggregate_row_map.json)

Coverage audit outputs:

- [coverage_audit_results.json](../tests/reference_results/csv_scope_audit/coverage_audit_results.json)
- [coverage_audit_summary.json](../tests/reference_results/csv_scope_audit/coverage_audit_summary.json)

## Deterministic ID Policy

Unique SMILES ID는 다음 규칙으로 고정한다.

- exact SMILES string를 lexicographic order로 정렬
- 1-based contiguous ID 부여
- 형식: `csv_scope_smiles_%06d`

예:

- `csv_scope_smiles_000001`
- `csv_scope_smiles_006042`

이 규칙은 source row order와 독립적으로 unique manifest를 안정화하기 위한 것이다.

## Release-Target Interpretation

이번 task 이후의 release gate는 다음 질문으로 해석해야 한다.

- 현재 pipeline이 이 CSV snapshot의 unique SMILES `6042`개를 얼마나 처리하는가?

즉, 기존 PT7/PT8 supported subset만 보는 것으로는 충분하지 않다. 이번 scope baseline 이후에는 이 CSV snapshot 전체가 현재 release target이다.

## Audit Execution Contract

CSV scope audit는 기존 `GROMACS_PCFF` 파이프라인을 유지한 채, 입력 어댑터만 별도로 둔다.

- source-of-truth pipeline:
  - PT1 parse
  - PT2 perception
  - PT3 atom typing
  - PT4 bonded assignment
  - PT5 nonbonded assignment
  - PT6 GROMACS dry-run emitter
- CSV input adapter:
  - `conda` 환경 `MD`
  - RDKit fixed-seed embed
  - local `pysoftk` `proto_polymer`
  - placeholder `Br`
  - deterministic `mol2` 출력

중요:

- `pysoftk`는 구조 입력 어댑터일 뿐이다.
- `LUNAR atom_typing.py`나 외부 `pcff.frc` 기반 결과는 이 audit에 쓰지 않는다.
- adapter failure와 downstream pipeline failure는 분리해서 기록한다.

## Explicit Non-Goals Of This Task

이번 task는 다음을 하지 않는다.

- SMILES parser 추가
- chemistry support 확장
- CSV 밖의 새로운 polymer/salt family 추가
- ML 기반 typing 추가
- 실패를 숨기거나 reclassify해서 coverage를 높이는 일

현재 상태가 좋지 않아도 그대로 계측하는 것이 목적이다.
