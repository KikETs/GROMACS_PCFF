# PT8 M10 Handoff Typing

## Purpose

이 문서는 PT8 validation 결과를 M10 downstream workflow에 넘길 때 필요한 정확한 integration path만 고정한다.

대상:

- GROMACS PCFF topology generation
- M10 r-RESPA method-validation workflow

비대상:

- ACPYPE/GAFF2 fallback
- unsupported polymer chemistry 자동 처리
- trajectory-level scientific conclusions

## Input Contract

M10이 받아야 하는 상위 입력은 PT7 polymer workflow spec이다.

Entry point:

- [src/polymer_workflow/engine.py](/home/kiket/바탕화면/test/GROMACS_PCFF/src/polymer_workflow/engine.py)
- CLI: `PYTHONPATH=src python3 -m polymer_workflow <spec.json> --out <dir>`

Validated supported examples:

- [testdata/polymer_workflow_golden/cases/monoglyme_litfsi_1to1/spec.json](/home/kiket/바탕화면/test/GROMACS_PCFF/testdata/polymer_workflow_golden/cases/monoglyme_litfsi_1to1/spec.json)
- [testdata/polymer_workflow_golden/cases/diglyme_litfsi_1to1/spec.json](/home/kiket/바탕화면/test/GROMACS_PCFF/testdata/polymer_workflow_golden/cases/diglyme_litfsi_1to1/spec.json)
- [testdata/polymer_workflow_golden/cases/triglyme_litfsi_2to2/spec.json](/home/kiket/바탕화면/test/GROMACS_PCFF/testdata/polymer_workflow_golden/cases/triglyme_litfsi_2to2/spec.json)

## Required Output Artifacts

M10은 다음 파일만 PCFF-qualified typing/export artifact로 받아야 한다.

- `forcefield_pcff.itp`
- `molecule_<component>.itp`
- `topol.top`
- `polymer_workflow_report.json`

중요:

- `polymer_workflow_report.json` 없이는 provenance chain을 검증할 수 없다.
- ACPYPE/GAFF2 산출물은 여기에 섞으면 안 된다.

## Provenance Gate

M10 stage가 topology를 수용하기 전에 최소한 다음을 검사해야 한다.

1. `polymer_workflow_report.json`의 `workflow.status`가 `written`인지 확인
2. 각 exportable component의 `source_chain`에 다음 hash가 모두 있는지 확인
   - `typed_ir_sha256`
   - `chem_perception_sha256`
   - `typing_report_sha256`
   - `bonded_assignment_sha256`
   - `nonbonded_assignment_sha256`
3. `assembly_checks`에서 다음이 모두 `pass`인지 확인
   - `charge_neutrality`
   - `salt_balance`
   - `fragment_consistency`
4. M10 run registry에 `spec_path`, `spec_sha256`, output file sha256를 같이 기록

이 네 단계를 생략하면, 다시 M10에서 provenance가 흐려진다. 그건 이미 [docs/validation_report_m10.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/validation_report_m10.md)가 막으려는 문제다.

## Failure Routing

M10은 PT8 validation에서 확인한 failure class를 그대로 사용해야 한다.

- `typing_failure`
  - examples: malformed input, unsupported component classification, unresolved atom families
- `parameter_failure`
  - examples: missing bonded parameters, missing nonbonded atom types, missing pair rules
- `emitter_failure`
  - examples: unsupported 1-4 scaling, rendered output mismatch, representability failure

Machine-readable source:

- [failure_probe_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/pt8_typing_validation/failure_probe_summary.json)

Recommendation:

- M10 stage result JSON에는 `failure_class`, `failure_code`, `message`, `source_artifact`를 필수 필드로 둔다.

## Integration Sequence

권장 순서:

1. PT7 spec를 입력으로 `polymer_workflow` 실행
2. `polymer_workflow_report.json` 생성 여부와 provenance chain 확인
3. emitted `topol.top`과 companion `.itp` 파일을 M10 GROMACS run directory로 복사
4. M10 artifact registry에 PT8 hash chain 기록
5. 이후에만 `grompp`/simulation stage 실행

잘못된 순서:

- ACPYPE 또는 다른 topology builder를 먼저 돌리고 나중에 PT8 provenance를 덮어씌우는 방식
- emitted topology만 복사하고 `polymer_workflow_report.json`을 버리는 방식

둘 다 provenance를 깨뜨린다.

## Validation Step Before M10 Use

M10에 넣기 전, repository 기준으로 다음을 먼저 통과시켜야 한다.

```bash
PYTHONPATH=src python3 tools/run_typing_validation/generate.py validate
```

이 명령은 다음 reference JSON과 byte-for-byte 일치하는지 확인한다.

- [tests/reference_results/pt8_typing_validation/validation_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/pt8_typing_validation/validation_summary.json)
- [tests/reference_results/pt8_typing_validation/per_case_results.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/pt8_typing_validation/per_case_results.json)
- [tests/reference_results/pt8_typing_validation/failure_probe_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/pt8_typing_validation/failure_probe_summary.json)
- [tests/reference_results/pt8_typing_validation/lammps_smoke_parity_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/pt8_typing_validation/lammps_smoke_parity_summary.json)

## Handoff Limits

이 handoff는 다음까지만 보장한다.

- PT7 supported SPE subset
- deterministic typing/export provenance
- M10 dependency insertion path

보장하지 않는 것:

- unsupported chemistry 자동 확장
- matched LAMMPS force parity
- publishable transport agreement
