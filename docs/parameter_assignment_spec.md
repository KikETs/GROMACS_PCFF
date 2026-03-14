# Parameter Assignment Specification (PT4)

## Status

이 문서는 **PT4** 범위만 고정합니다.

PT4는 다음만 구현합니다.

- PT1 `typed_system` IR, PT2 `chem_perception`, PT3 `pcff_atom_typing_report`를 입력으로 받는 bonded parameter assignment
- bond / angle / dihedral / improper class의 결정적 canonical signature 생성
- signature 기반의 deterministic parameter lookup
- 규칙 ID와 규칙 파일 provenance를 포함한 assignment record 생성
- 누락된 bonded parameter의 explicit diagnostic 보고

PT4는 다음을 구현하지 않습니다.

- nonbonded parameter assignment
- topology export
- raw input 파일을 다시 읽어서 별도로 chemistry를 재해석하는 ad hoc parameter assignment
- 외부 PCFF 데이터베이스 자동 병합

중요한 사실:

- 이 저장소에는 현재 외부 배포형 PCFF parameter library가 포함되어 있지 않습니다.
- 따라서 [rules/pcff_parameters.json](/home/kiket/바탕화면/test/GROMACS_PCFF/rules/pcff_parameters.json)은 **repository-local frozen Class2 coefficients**를 담습니다.
- provenance는 규칙 파일 경로, `ruleset_id`, `rule_id`, `canonical_signature`로 추적됩니다.
- CSV scope용으로 추가된 carbonyl subset 계수는 `LUNAR all2lmp + pcff.frc` 대표 추출값을 repository-local rule로 동결한 것입니다.
- 중요한 제약:
  - bond / angle / improper 일부만 동결했습니다.
  - dihedral cross-term은 여전히 비어 있는 경우가 많고, 이 저장소는 그 구멍을 0으로 메우지 않습니다.
  - 따라서 CSV carbonyl subset은 PT4에서 `missing_parameter`로 멈출 수 있습니다.

근거:

- [docs/pcff_respa_reference_spec.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/pcff_respa_reference_spec.md)
- [tools/pcff_fixture_bridge/common.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/pcff_fixture_bridge/common.py)
- [tools/pcff_short_md_parity/prepare_reference.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/pcff_short_md_parity/prepare_reference.py)

## Module Layout

소스 위치:

- [src/parameter_assignment](/home/kiket/바탕화면/test/GROMACS_PCFF/src/parameter_assignment)

핵심 진입점:

- `parameter_assignment.assign_ir(ir, typing_report=..., perception=...)`
- `parameter_assignment.assign_file(path, input_format=..., source_id=...)`

규칙 파일:

- [rules/pcff_parameters.json](/home/kiket/바탕화면/test/GROMACS_PCFF/rules/pcff_parameters.json)

## Input Contract

PT4는 다음 세 입력이 같은 source chain에 속한다고 검증합니다.

1. PT1 `typed_system`
2. PT2 `chem_perception`
3. PT3 `pcff_atom_typing_report`

검증 규칙:

- `typing_report.source.typed_ir_sha256`는 supplied IR과 일치해야 합니다.
- `typing_report.source.chem_perception_sha256`는 supplied perception report와 일치해야 합니다.
- `typing.status`는 반드시 `typed`여야 합니다.
- atom typing이 `unsupported`, `unresolved`, `ambiguous`이면 PT4는 진행하지 않고 명시적으로 실패합니다.

즉, PT4 출력은 raw parser 재실행 결과가 아니라 **typed IR chain** 위에서만 생성됩니다.

## Rules Schema

규칙 스키마:

- `schema_name = "pcff_bonded_parameter_rules"`
- `schema_version = 1`

최상위 필드:

- `ruleset_id`
- `term_model`
- `interaction_rules`

`interaction_rules`는 다음 네 키를 반드시 가집니다.

- `bond`
- `angle`
- `dihedral`
- `improper`

각 규칙 필드:

- `rule_id`
- `canonical_signature`
- `parameters`
- `provenance`

추가 제약:

- 같은 interaction kind 안에서 `canonical_signature`는 유일해야 합니다.
- lookup ambiguity는 허용하지 않습니다.
- 규칙이 없으면 fallback하지 않고 `missing_parameter` diagnostics를 냅니다.

## Canonical Signatures

### Bond

형식:

- `bond(<family_1>|<family_2>)`

규칙:

- 두 끝 atom family를 정방향/역방향으로 비교합니다.
- family tuple이 더 작은 쪽을 canonical orientation으로 선택합니다.
- family tuple이 같으면 atom index tuple로 tie-break 합니다.

### Angle

형식:

- `angle(<end_1>|<center>|<end_2>)`

규칙:

- 중심 atom family는 고정합니다.
- 양 끝만 정방향/역방향 비교로 canonicalize 합니다.

### Dihedral

형식:

- `dihedral(<a>|<b>|<c>|<d>)`

규칙:

- 4-atom path와 reversed path를 비교합니다.
- 더 작은 family tuple을 canonical orientation으로 채택합니다.
- family tuple이 같으면 atom index tuple로 tie-break 합니다.

### Improper

형식:

- `improper(<center>|<outer_1>|<outer_2>|<outer_3>)`

규칙:

- center는 고정합니다.
- outer atoms는 `(family, atom_index)` 정렬로 canonicalize 합니다.

현재 PT4에서 actual improper interaction 생성은 다음 경우만 합니다.

- PT2 `improper_center_candidate.kinds`에 `planar_trigonal`이 있고
- ordered neighbors가 정확히 3개인 경우

`tetrahedral_distinct_substituents`는 PT2 feature로는 유지되지만, PT4 bonded improper term으로 변환하지 않습니다. 4-center improper와 직접 1:1 대응되지 않기 때문입니다.

## Generated Interaction Instances

PT4는 component graph에서 다음을 생성합니다.

- bond: IR bond 1개당 1개
- angle: 각 중심 atom의 이웃 2개 조합
- dihedral: 각 중심 bond 양쪽 이웃들의 곱집합 경로
- improper: 위의 planar trigonal candidate만

출력 record 필드:

- `assignment_id`
- `interaction_kind`
- `atom_indices`
- `source_atom_indices`
- `source_atom_ids`
- `atom_families`
- `canonical_signature`
- `status`
- `parameter_rule_id`
- `parameters`
- `provenance`

## Parameter Payload Shape

Bond:

- `main`
  - `r0_angstrom`
  - `k2_kcal_mol_per_a2`
  - `k3_kcal_mol_per_a3`
  - `k4_kcal_mol_per_a4`

Angle:

- `main`
  - `theta0_deg`
  - `k2_kcal_mol`
  - `k3_kcal_mol`
  - `k4_kcal_mol`
- `bb`
  - `k_kcal_mol_per_a2`
  - `r1_angstrom`
  - `r2_angstrom`
- `ba`
  - `k1_kcal_mol_per_a`
  - `k2_kcal_mol_per_a`
  - `r1_angstrom`
  - `r2_angstrom`

Dihedral:

- `main`
  - `k1_kcal_mol`
  - `phi1_deg`
  - `k2_kcal_mol`
  - `phi2_deg`
  - `k3_kcal_mol`
  - `phi3_deg`
- `mbt`
- `ebt`
- `at`
- `aat`
- `bb13`

필드 이름은 기존 repository-local Class2 tooling과 맞춥니다.

근거:

- [tools/pcff_fixture_bridge/common.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/pcff_fixture_bridge/common.py)
- [tools/pcff_short_md_parity/prepare_reference.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/pcff_short_md_parity/prepare_reference.py)

## Output Contract

최상위 report:

- `schema_name = "pcff_parameter_assignment_report"`
- `schema_version = 1`
- `source`
- `parameter_assignment`
- `components`

`source`는 다음 hash chain을 포함합니다.

- `typed_ir_sha256`
- `chem_perception_sha256`
- `typing_report_sha256`
- `rules_sha256`

`parameter_assignment`:

- `status`
  - `assigned`
  - `missing_parameters`
- `ruleset_id`
- `term_model`

component payload:

- `interaction_counts`
- `interactions`
  - `bond`
  - `angle`
  - `dihedral`
  - `improper`
- `diagnostics`

## Malformed / Incomplete Input Behavior

PT4는 다음 상황에서 명시적으로 실패합니다.

- typing report가 supplied IR/perception hash chain과 맞지 않음
- typing report가 `typed` 상태가 아님
- typing report atom 중 미할당 atom이 존재함
- rules schema가 잘못됨

PT4는 다음 상황에서 report를 생성하지만 status를 낮춥니다.

- 특정 interaction signature에 대응 규칙이 없음
  - report status: `missing_parameters`
  - per-interaction status: `missing_parameter`
  - diagnostic code: `missing_parameter`

즉, malformed input과 missing rule은 구분합니다.

## Regression Scope

기본 ruleset은 현재 frozen golden supported systems만 커버합니다.

- `ethane_neutral`
- `dimethyl_ether_neutral`
- `lithium_cation`
- `tfsi_anion_explicit`

unsupported chemistry는 PT3에서 차단되며, PT4는 그 이후 단계만 담당합니다.
