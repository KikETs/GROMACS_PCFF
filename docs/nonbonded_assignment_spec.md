# Nonbonded Assignment Specification (PT5)

## Status

이 문서는 **PT5** 범위만 고정합니다.

PT5는 다음을 구현합니다.

- PT1 `typed_system`, PT3 `pcff_atom_typing_report`, PT4 bonded report를 입력으로 받는 deterministic nonbonded assignment
- atom family별 self nonbonded parameter assignment
- sixth-power mixing 기반 normal pair-class 해석
- explicit exclusions와 explicit `1-4` pair semantics 생성
- optional pair-specific override 적용
- later GROMACS / optional LAMMPS emitters에 필요한 export metadata 생성

PT5는 다음을 구현하지 않습니다.

- topology emitter
- nonbonded runtime execution
- long-range solver parameterization
- raw input 파일에서 `special_bonds`를 다시 파싱해서 semantics를 바꾸는 동적 경로

## Critical Scope Decision

현재 PT1 typed IR chain에는 `special_bonds` 명령이 없습니다.

따라서 PT5는 `special_bonds` semantics를 ruleset에 **명시적으로 동결**합니다.

기본 frozen profile:

- `lj/coul 0.0 0.0 1.0 angle no dihedral no`

근거:

- [docs/pcff_respa_reference_spec.md](./pcff_respa_reference_spec.md)
- [docs/validation_report_m4.md](./validation_report_m4.md)
- [tools/pcff_fixture_bridge/common.py](../tools/pcff_fixture_bridge/common.py)

이건 범위 축소가 아니라 현재 typed IR contract의 빈칸을 숨기지 않고 ruleset으로 고정한 것입니다.

## Module Layout

소스 위치:

- [src/nonbonded_assignment](../src/nonbonded_assignment)

핵심 진입점:

- `nonbonded_assignment.assign_ir(ir, typing_report=..., bonded_report=...)`
- `nonbonded_assignment.assign_file(path, input_format=..., source_id=...)`

규칙 파일:

- [rules/pcff_nonbonded.json](../rules/pcff_nonbonded.json)
- phase1 bridge lookup source: [frc_file/pcff.frc](../frc_file/pcff.frc)

중요한 제약:

- CSV-scoped carbonyl subset용 self nonbonded rule은 추가됐습니다.
- 하지만 PT5는 여전히 `parameter_assignment.status = assigned`를 요구합니다.
- 따라서 PT4에서 dihedral / angle gap이 남아 있으면 PT5까지 도달하지 못합니다.
- `LUNAR` 경로가 provenance에 남아 있어도 runtime lookup은 저장소 내부 frozen files만 사용합니다.

## Input Contract

PT5는 다음 source chain을 검증합니다.

1. PT1 `typed_system`
2. PT3 `pcff_atom_typing_report`
3. PT4 `pcff_parameter_assignment_report`

검증 규칙:

- PT3 report는 supplied IR hash와 일치해야 합니다.
- PT4 report는 supplied IR hash와 PT3 typing report hash와 일치해야 합니다.
- `typing.status`는 `typed`여야 합니다.
- PT4 bonded report는 `parameter_assignment.status = assigned`여야 합니다.

즉, PT5의 exclusions / `1-4`는 ad hoc graph walk가 아니라 PT4가 이미 생성한 bond / angle / dihedral interaction records를 기반으로 합니다.

## Rules Schema

규칙 스키마:

- `schema_name = "pcff_nonbonded_rules"`
- `schema_version = 1`

최상위 필드:

- `ruleset_id`
- `pair_model`
- `atom_type_rules`
- `pair_overrides`

### `pair_model`

필수 필드:

- `mixing_rule`
- `neutral_pair_style`
- `charged_pair_style`
- `repulsion_power`
- `dispersion_power`
- `special_bonds_profile`
- `charge_source_policy`

기본 PT5 값:

- `mixing_rule = sixthpower`
- neutral: `lj/class2`
- charged: `lj/class2/coul/long`
- `repulsion_power = 9`
- `dispersion_power = 6`

### `atom_type_rules`

각 atom family rule은 다음을 가집니다.

- `rule_id`
- `family`
- `nonbonded_type`
- `self_parameters`
  - `epsilon_kcal_mol`
  - `sigma_angstrom`
- `provenance`

### `pair_overrides`

각 override rule은 다음을 가집니다.

- `rule_id`
- `canonical_family_pair`
- `scope`
  - `normal`
  - `pair14`
  - `both`
- `parameters`
  - `epsilon_kcal_mol`
  - `sigma_angstrom`
- `provenance`

기본 ruleset은 override를 비워 두지만, 엔진은 override를 지원합니다.

## Charge Assignment

PT5는 charge를 추정하지 않습니다.

규칙:

- `partial_charge`가 있으면 그것을 사용
- 없으면 `formal_charge`를 사용
- 둘 다 없으면 explicit failure

이 정책은 `pair_model.charge_source_policy = prefer_partial_charge_else_formal_charge`로 기록됩니다.

## Mixing Semantics

기본 normal pair는 sixth-power mixing으로 계산합니다.

공식:

- `sigma_ij = ((sigma_i^6 + sigma_j^6) / 2)^(1/6)`
- `epsilon_ij = 2 * sqrt(epsilon_i * epsilon_j) * sigma_i^3 * sigma_j^3 / (sigma_i^6 + sigma_j^6)`

근거:

- [docs/pcff_respa_reference_spec.md](./pcff_respa_reference_spec.md)
- [docs/validation_report_m4.md](./validation_report_m4.md)

## Class2 Coefficient Forms

PT5 report는 human-readable sigma/epsilon뿐 아니라 later emitter를 위한 coefficient form도 남깁니다.

### Normal nonbonded

- `c6 = 18 * epsilon * sigma^6`
- `c9 = 18 * epsilon * sigma^9`

### Listed `1-4`

- `c6 = 3 * epsilon * sigma^6`
- `c9 = 2 * epsilon * sigma^9`

근거:

- [docs/validation_report_m4.md](./validation_report_m4.md)

## Exclusions and `1-4`

PT5는 PT4 interaction records에서 다음을 생성합니다.

- bond record의 끝 atom pair -> `1-2`
- angle record의 first/third atom pair -> `1-3`
- dihedral record의 first/fourth atom pair -> `1-4`

기본 special-bonds profile에서는:

- `1-2`: LJ 0.0, Coul 0.0
- `1-3`: LJ 0.0, Coul 0.0
- `1-4`: LJ 1.0, Coul 1.0

중요:

- `1-4`는 graph distance 추정이 아니라 **dihedral records에서 유도**됩니다.
- exclusion과 `1-4`가 동시에 주장되는 pair가 있으면 더 짧은 topological relation이 우선합니다.

## Output Contract

최상위 report:

- `schema_name = "pcff_nonbonded_assignment_report"`
- `schema_version = 1`
- `source`
- `nonbonded_assignment`
- `components`

`source`는 다음 hash chain을 포함합니다.

- `typed_ir_sha256`
- `typing_report_sha256`
- `bonded_assignment_sha256`
- `rules_sha256`

Per-component payload:

- `atoms`
- `pair_classes`
- `exclusions`
- `pair14`
- `export_metadata`
- `diagnostics`

### `atoms`

각 atom record는 다음을 가집니다.

- source/canonical identity
- `assigned_family`
- `charge_assignment`
- `nonbonded_type`
- `self_parameters`
- `provenance`

### `pair_classes`

component에 존재하는 unique family 조합마다 1개 생성합니다.

필드:

- `canonical_family_pair`
- `parameter_source`
  - `mixed`
  - `override`
- `parameters`
  - sigma/epsilon
  - normal coefficients
  - pair14 coefficients

### `exclusions`

필드:

- `atom_indices`
- `topological_relation`
- `lj_scale`
- `coul_scale`
- `source_assignment_ids`

### `pair14`

필드:

- `atom_indices`
- `canonical_family_pair`
- `lj_scale`
- `coul_scale`
- `source_dihedral_assignment_ids`
- `parameter_source`
- `parameters`
- `provenance`

## Export Metadata

### GROMACS

PT5는 later GROMACS emitter에 필요한 최소 metadata를 기록합니다.

- `combination_rule = sixthpower`
- `repulsion_power = 9`
- `dispersion_power = 6`
- explicit `[ pairs ]` required
- explicit `[ exclusions ]` required

### LAMMPS

PT5는 optional LAMMPS emitter용 metadata도 기록합니다.

- recommended `pair_style_kind`
  - charged component면 `lj/class2/coul/long`
  - 아니면 `lj/class2`
- `pair_modify mix sixthpower`
- frozen `special_bonds`
- self-only `pair_coeff` + optional explicit cross override policy

## Malformed / Missing Behavior

explicit failure:

- source hash chain mismatch
- typing report incomplete
- bonded report incomplete
- atom charge source 부재
- rules schema invalid

report with diagnostics:

- atom family용 nonbonded self rule 부재
- pair class 해석 실패
- `1-4` pair parameter 해석 실패

즉, malformed input과 missing rule은 구분합니다.
