# PT7 Polymer Workflow Golden Cases

이 디렉터리는 **PT7** 범위의 polymer electrolyte workflow golden input을 보관한다.

포함 범위:

- linear methoxy-capped polyether oligomer fragment
- explicit repeat-unit template with `Du` placeholders
- lithium cation
- explicit TFSI-like anion
- mixed-system assembly metadata for GROMACS export

현재 frozen PT7 example systems:

- `monoglyme_litfsi_1to1`
- `diglyme_litfsi_1to1`
- `triglyme_litfsi_2to2`

중요한 한계:

- PT7는 일반 PEO bulk polymer builder가 아니다.
- 지원되는 polymer fragment chemistry는 현재 `monoglyme`, `diglyme`, `triglyme` 형태의 **linear methoxy-capped polyether oligomer**로 제한된다.
- repeat-unit template는 PT2 polymer connection tag validation까지만 수행하고 직접 typing/export되지는 않는다.
