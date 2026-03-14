# CSV Scope Audit Tooling

이 디렉터리는 `simulation-trajectory-aggregate.csv` snapshot을 현재 릴리스 chemistry scope로 동결하고, 현재 typing/export pipeline이 그 scope를 어디까지 처리하는지 계측하는 도구를 담는다.

중요:

- 이 도구는 support를 넓히지 않는다.
- CSV snapshot 밖의 chemistry는 취급하지 않는다.
- 기존 `GROMACS_PCFF` 파이프라인이 source-of-truth다.
- `pysoftk`는 CSV의 pSMILES/SMILES를 `mol2`로 바꾸는 검증용 입력 어댑터로만 쓴다.
- adapter 실패와 downstream pipeline 실패는 분리해서 기록한다.
- 이 경로는 `conda` 환경 `MD`를 사용한다.

명령:

```bash
PYTHONPATH=src python3 tools/run_csv_scope_audit/generate.py snapshot \
  --csv /path/to/simulation-trajectory-aggregate.csv
```

위 명령은 다음 manifest를 쓴다.

- `data_manifests/simulation_trajectory_aggregate_snapshot.json`
- `data_manifests/simulation_trajectory_aggregate_unique_smiles.json`
- `data_manifests/simulation_trajectory_aggregate_row_map.json`

Coverage audit:

```bash
PYTHONPATH=src python3 tools/run_csv_scope_audit/generate.py audit
```

필수 런타임:

- adapter python: `/home/kiket/anaconda3/envs/MD/bin/python`
- local pysoftk root: `/home/kiket/바탕화면/test/torch/pysoftk`

Audit는 내부적으로 다음 경로를 거친다.

- CSV manifest entry
- RDKit embed with fixed seed
- `pysoftk` `proto_polymer`
- placeholder 제거 후 `mol2` 출력
- PT1 parse
- PT2 perception
- PT3 typing
- PT4 bonded assignment
- PT5 nonbonded assignment
- PT6 GROMACS dry-run emitter

Reference validation:

```bash
PYTHONPATH=src python3 tools/run_csv_scope_audit/generate.py validate
```
