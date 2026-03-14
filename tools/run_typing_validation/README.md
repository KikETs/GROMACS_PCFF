# PT8 Typing Validation Runner

이 도구는 **PT8** 범위의 supported SPE workflow validation summary를 생성하거나, 생성 결과를 체크인된 golden reference와 비교한다.

## Commands

Generate checked-in reference summaries:

```bash
PYTHONPATH=src python3 tools/run_typing_validation/generate.py generate
```

Validate regenerated summaries against the checked-in reference:

```bash
PYTHONPATH=src python3 tools/run_typing_validation/generate.py validate
```

## Output files

- `per_case_results.json`
- `failure_probe_summary.json`
- `lammps_smoke_parity_summary.json`
- `validation_summary.json`

## Scope

- PT7 supported SPE cases only
- end-to-end `polymer_workflow` execution
- explicit failure-class probes for typing / parameter / emitter
- contract-level LAMMPS smoke parity only

이 도구는 direct force/energy parity를 주장하지 않는다. 현재 repository에는 PT7 SPE case와 trajectory-matched LAMMPS reference가 없기 때문이다.
