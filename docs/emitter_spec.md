## PT6 Emitter Spec

### Scope

PT6 범위는 GROMACS topology emitter만 포함한다.

- 구현 대상: `src/emitters/gromacs/`
- 출력 파일:
  - `forcefield_pcff.itp`
  - `molecule.itp`
  - `topol.top`
- 선택 과제였던 `src/emitters/lammps/`는 이번 마일스톤 범위에 포함하지 않는다.

Emitter는 원시 입력 파일을 다시 해석하지 않는다. 출력은 반드시 아래의 결정적 source chain 위에서만 생성된다.

1. PT1 `typed_system` IR
2. PT3 atom typing report
3. PT4 bonded parameter assignment report
4. PT5 nonbonded assignment report

`emit_file(...)`는 편의 함수일 뿐이며, 내부에서 위 chain을 생성한 뒤 `emit_ir(...)`로 넘긴다. `emit_ir(...)`는 report hash와 `source_id`를 검증하고, chain이 맞지 않으면 즉시 실패한다.

### Determinism Rules

출력은 다음 규칙으로 고정한다.

- 파일명과 파일 순서는 항상 `forcefield_pcff.itp`, `molecule.itp`, `topol.top`
- 모든 파일은 LF line ending만 사용하고 마지막에 newline 하나로 끝난다
- float formatting은 고정 소수점 8자리
- atom order는 PT1 canonical atom index 순서
- interaction order는 PT4/PT5 report 순서
- `[ atomtypes ]`는 `nonbonded_type` 기준 정렬된 고유 레코드만 출력
- override section은 canonical family pair 순서로 정렬

같은 입력과 같은 ruleset이면 동일한 파일 바이트열이 재생성되어야 한다.

### Input Contract

`emit_ir(...)`는 다음 상태를 요구한다.

- `typing_report["typing"]["status"] == "typed"`
- `bonded_report["parameter_assignment"]["status"] == "assigned"`
- `nonbonded_report["nonbonded_assignment"]["status"] == "assigned"`

하나라도 충족하지 않으면 emitter는 출력하지 않고 명시적으로 실패한다.

### GROMACS Mapping

#### `forcefield_pcff.itp`

- `[ defaults ]`
  - 고정값: `1 4 yes 1.0 1.0 9.0`
  - 의미:
    - combination rule: GROMACS sixth-power (`comb-rule = 4`)
    - `gen-pairs = yes`
    - `fudgeLJ = 1.0`
    - `fudgeQQ = 1.0`
    - repulsion power: `9`
- `[ atomtypes ]`
  - PT5 atom self parameter를 nm / kJ mol 단위로 변환해 출력
  - 원소 질량은 emitter 내부의 고정 테이블을 사용
- `[ nonbond_params ]`
  - PT5 normal pair override가 있을 때만 출력
- `[ pairtypes ]`
  - PT5 `pair14` override가 있을 때만 출력

#### `molecule.itp`

- `[ moleculetype ]`
  - monoatomic component는 `nrexcl = 1`
  - 그 외는 `nrexcl = 3`
- `[ atoms ]`
  - PT1 canonical order 사용
  - type은 PT5 `nonbonded_type`
  - charge는 PT5 charge assignment 값
- `[ bonds ]`
  - PT4 bond coefficients를 GROMACS class2 bond 형식(`funct 11`)으로 변환
- `[ pairs ]`
  - PT5 `pair14` 레코드를 explicit `1-4` pair로 출력
  - 현재 emitter는 pair parameter scaling이 `1.0 / 1.0`인 경우만 허용한다
- `[ angles ]`
  - PT4 angle coefficients를 GROMACS class2 angle 형식(`funct 11`)으로 변환
- `[ dihedrals ]`
  - proper dihedral은 `funct 13`
  - improper는 `funct 12`
- `[ exclusions ]`
  - 현재 emitter는 별도 섹션을 쓰지 않는다
  - 대신 `nrexcl = 3`과 explicit `[ pairs ]` 조합으로 표현한다
  - 이 방식이 안전하려면 exclusion 레코드가 모두 zero-scaled `1-2` / `1-3` 관계여야 하며, emitter가 이를 검증한다

#### `topol.top`

- `forcefield_pcff.itp`, `molecule.itp`를 이 순서로 include
- `[ system ]`은 PT1 component name
- `[ molecules ]`는 PT3 classification family에서 유도한 고정 molecule name과 count `1`

### Dry Run And Validation

`emit_ir(...)`와 `emit_file(...)`는 두 가지 validation 모드를 지원한다.

- `dry_run=True`
  - bundle을 렌더링하고 검증하지만 파일은 쓰지 않는다
- `validate_existing=True`
  - `out_dir`에 이미 있는 파일이 새로 렌더링한 결과와 byte-identical인지 비교한다
  - 하나라도 다르면 `rendered_output_mismatch`로 실패한다

manifest는 다음을 기록한다.

- source chain hash
- emitter mode
- component summary
- 출력 파일별 `sha256`, byte size, line count

### Explicit Failure Conditions

Emitter는 다음 조건에서 즉시 실패한다.

- source chain hash mismatch
- 필요한 output directory 누락
- 기존 출력 검증 실패
- 지원되지 않는 원소 질량
- 같은 `nonbonded_type`에 대해 상충하는 atomtype 정의가 들어온 경우
- zero-scaled `1-2` / `1-3`로 축약할 수 없는 exclusion
- `1.0 / 1.0`이 아닌 `pair14` scaling

### Out Of Scope

이번 마일스톤에는 아래를 포함하지 않는다.

- LAMMPS emitter 구현
- multi-component system assembly
- coordinate emitter (`.gro`, `.pdb`) 생성
- 새로운 typing / parameter assignment 규칙 추가
