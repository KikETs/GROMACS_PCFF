# CSV Scope Audit Tooling

이 디렉터리는 `simulation-trajectory-aggregate.csv` snapshot을 현재 릴리스 chemistry scope로 동결하고, 현재 typing/export pipeline이 그 scope를 어디까지 처리하는지 계측하는 도구를 담는다.

중요:

- 이 도구는 support를 넓히지 않는다.
- CSV snapshot 밖의 chemistry는 취급하지 않는다.
- SMILES 입력이 현재 pipeline에서 미지원이면 그 사실을 `parse_failure`로 명시한다.

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

Reference validation:

```bash
PYTHONPATH=src python3 tools/run_csv_scope_audit/generate.py validate
```
