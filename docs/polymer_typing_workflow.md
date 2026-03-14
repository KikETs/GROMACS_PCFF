# PT7 Polymer Typing Workflow

## Status

이 문서는 **PT7** 범위만 고정한다.

PT7는 PT1-PT6 pipeline 위에 polymer-electrolyte용 workflow layer를 추가한다.

핵심 목표:

- repeat-unit template와 capped oligomer fragment를 같은 workflow에서 다룬다
- lithium / TFSI salt species를 polymer fragment와 함께 조립한다
- charge-neutrality와 fragment-consistency를 명시적으로 검사한다
- mixed-system GROMACS topology를 결정적으로 생성한다

PT7는 general polymer builder가 아니다.

## Source Layout

- workflow code: [src/polymer_workflow](/home/kiket/바탕화면/test/GROMACS_PCFF/src/polymer_workflow)
- PT7 golden inputs: [testdata/polymer_workflow_golden](/home/kiket/바탕화면/test/GROMACS_PCFF/testdata/polymer_workflow_golden)

주요 진입점:

- `polymer_workflow.run_file(spec_path, out_dir=..., dry_run=..., validate_existing=...)`
- `polymer_workflow.run_spec(spec, ...)`
- `python -m polymer_workflow <spec.json> --dry-run`

## Supported PT7 Chemistry

PT7 workflow가 직접 지원하는 exportable polymer fragment chemistry는 다음으로 제한된다.

- **linear methoxy-capped polyether oligomer**
  - `monoglyme`
  - `diglyme`
  - `triglyme`
- explicit `Li+`
- explicit TFSI-like sulfonimide anion

지원 조건:

- 모든 bond는 explicit single bond
- explicit hydrogens 필수
- polymer fragment는 ring이 없어야 함
- polymer fragment는 두 개의 methyl ether cap을 가져야 함
- polymer backbone carbon은 `C/H2/O/C` 환경이어야 함

현재 workflow는 `PEO` bulk chain, branched polyether, hydroxy end group, carbonate, carbonyl, aromatic spacer를 지원하지 않는다.

## Workflow Modes

### 1. `repeat_unit_template`

입력은 explicit placeholder를 가진 template fragment다.

현재 인식 evidence:

- `Du`
- `R<number>`
- `*`
- explicit `polymer_connection_label` annotation

PT7는 template에서 다음만 수행한다.

- PT1 parse
- PT2 polymer connection tag propagation
- 정확히 두 개의 connection point가 있는지 validation

template는 직접 typing/parameter/export되지 않는다.

### 2. `capped_oligomer`

입력은 실제로 export할 수 있는 capped fragment다.

PT7는 workflow-local augmented ruleset으로 다음을 수행한다.

- PT3 component classification
  - new family: `acyclic_polyether_oligomer`
- PT3 atom family assignment
  - reused:
    - `ether_alpha_carbon_sp3`
    - `ether_oxygen_sp3`
    - `hydrogen_on_ether_alpha_carbon`
  - new:
    - `polyether_backbone_methylene_sp3`
    - `hydrogen_on_polyether_backbone_methylene`
- PT4 bonded parameter assignment
- PT5 nonbonded assignment
- PT6-compatible GROMACS molecule topology rendering

중요:

- 기본 PT3/PT4/PT5 frozen rules 파일은 수정하지 않는다
- PT7는 [src/polymer_workflow/rules.py](/home/kiket/바탕화면/test/GROMACS_PCFF/src/polymer_workflow/rules.py)에서 workflow-local augmented ruleset을 구성해 기존 엔진에 주입한다

## End-Group Model

PT7의 capped oligomer end-group model은 고정이다.

- model id: `methyl_ether_caps`
- terminal cap atom family: `ether_alpha_carbon_sp3`
- terminal cap count: 반드시 `2`

`left` / `right` 같은 방향성 이름은 사용하지 않는다.

이유:

- symmetric glyme fragment에서 화학적 좌/우는 본질적으로 임의다
- canonical atom order 기반 `terminal_cap_atom_indices` 두 개만 고정하는 편이 더 결정적이다

`repeat_unit_count`는 다음으로 계산한다.

- `backbone_methylene_count / 2`

현재 PT7 golden cases에서는:

- monoglyme: `1`
- diglyme: `2`
- triglyme: `3`

## Spec Schema

Top-level spec:

- `schema_name = "pcff_polymer_workflow_spec"`
- `schema_version = 1`
- `system_id`
- `description`
- `components`

각 component 필드:

- `component_id`
- `role`
  - `polymer_template`
  - `polymer_fragment`
  - `salt_cation`
  - `salt_anion`
- `workflow_kind`
  - `repeat_unit_template`
  - `capped_oligomer`
  - `salt_species`
- `path`
- `input_format`
  - currently only `mol_v2000`
- `source_id`
- `count`
- optional `molecule_name`
- optional `residue_name`

role / workflow kind 조합은 고정이다.

- `polymer_template` -> `repeat_unit_template`
- `polymer_fragment` -> `capped_oligomer`
- `salt_cation`, `salt_anion` -> `salt_species`

## Mixed-System Checks

PT7 workflow는 다음을 명시적으로 검사한다.

### Charge Neutrality

- exportable component의 assigned charge를 모두 합산한다
- 총합이 `0`이 아니면 `charge_imbalance`로 즉시 실패한다

### Salt Balance

현재 지원 salt는 `Li+` / `TFSI-`뿐이다.

- cation count와 anion count가 다르면 `salt_stoichiometry_mismatch`로 실패한다

이 제약은 현재 salt charge magnitude가 `+1 / -1`로 고정되어 있기 때문에 의도적이다.

### Fragment Consistency

polymer fragment는 다음을 만족해야 한다.

- component family = `acyclic_polyether_oligomer`
- terminal methyl ether cap atom = exactly `2`
- backbone methylene count = positive even integer
- oxygen count = `repeat_unit_count + 1`
- each terminal cap has exactly one heavy neighbor and that neighbor is `ether_oxygen_sp3`
- each backbone methylene has heavy-neighbor families `ether_oxygen_sp3` + `polyether_backbone_methylene_sp3`

하나라도 깨지면 `invalid_polyether_fragment`로 실패한다.

## Export Contract

PT7는 mixed-system GROMACS bundle을 생성한다.

- `forcefield_pcff.itp`
- `molecule_<component_id>.itp`
- `topol.top`
- optional companion report: `polymer_workflow_report.json`

출력 규칙:

- deterministic file ordering
- LF line ending only
- PT6 formatting helpers 재사용
- shared forcefield는 exportable component들의 atomtype union으로 생성
- `[ molecules ]` section은 spec 순서를 유지한다

repeat-unit template component는 report에는 남지만 topology export에는 포함되지 않는다.

## Realistic PT7 Example Systems

현재 frozen end-to-end example:

- `monoglyme_litfsi_1to1`
- `diglyme_litfsi_1to1`
- `triglyme_litfsi_2to2`

이 예제들은 모두 다음을 동시에 검증한다.

- repeat-unit template validation
- capped oligomer typing
- bonded / nonbonded assignment
- salt species inclusion
- mixed-system GROMACS export

## Explicit Failure Modes

PT7 workflow는 다음에서 즉시 실패한다.

- invalid workflow spec
- missing input file
- unsupported component family for declared role
- invalid repeat-unit template
- invalid polyether fragment
- charge imbalance
- salt stoichiometry mismatch
- PT6-level GROMACS representability failure
- rendered output mismatch during `validate_existing`

## Out Of Scope

PT7는 다음을 구현하지 않는다.

- arbitrary PEO chain growth
- hydroxyl / acrylate / carbonate / aromatic end groups
- multi-anion or multi-cation chemistry beyond `Li+` / TFSI
- coordinate packing
- solvent box assembly
- concentration targeting
- direct LAMMPS mixed-system emitter
