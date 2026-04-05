# Gate H Fixture Scan

Date: `2026-04-01`

## Bottom Line

Repo 안에는 Gate H production-signoff에 바로 쓸 수 있는 exact-r-RESPA transport fixture가 없다.

- neutral 쪽 최선 후보는 `dense_oligomer`다.
- charged 쪽은 현재 `dense_salt_polymer`밖에 없지만, 이건 물리적으로 이미 막혀 있다.
- 둘 다 TP0 frozen size floor를 못 맞춘다.

## TP0 Constraints Used

Source: [transport_protocol_freeze.md](/home/kiket/Desktop/test/GROMACS_PCFF/docs/transport_protocol_freeze.md), [transport_scope_matrix.md](/home/kiket/Desktop/test/GROMACS_PCFF/docs/transport_scope_matrix.md)

- minimum `1000 atoms`
- minimum box dimension `> 3.0 nm`
- neutral: `>= 2 ns` NPT equilibration + `>= 10 ns` production
- charged: `>= 5 ns` NPT equilibration + `>= 20 ns` production
- TP0 method scope says `RESPA are deferred`

## Candidate Assessment

### `dense_oligomer`

Evidence:
- [system.json](/home/kiket/Desktop/test/GROMACS_PCFF/testdata/lammps_golden/systems/dense_oligomer/system.json)
- [system.gro](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_3_dense_ensemble_gate/dense_oligomer/system.gro)
- [system.top](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_3_dense_ensemble_gate/dense_oligomer/system.top)
- [report.json](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_3_dense_ensemble_gate/dense_oligomer/report.json)
- [m10_3_dense_ensemble_parity.md](/home/kiket/Desktop/test/GROMACS_PCFF/docs/m10_3_dense_ensemble_parity.md)

Facts:
- `384 atoms`
- box `2.2 x 2.2 x 2.2 nm`
- molecules: `MOL1 64`
- `constraints = none`
- PME + class2 bonded + pair14 topology shape is present
- existing long-run evidence is `20 ps NPT equilibration + 100 ps NPT production`, `integrator = md`

Assessment:
- neutral transport seed로는 가장 낫다.
- 단, TP0 size floor와 duration floor를 둘 다 못 맞춘다.
- multi-molecule이라 `small_oligomer`처럼 COM-drift removal 때문에 MSD가 구조적으로 붕괴하지는 않는다.
- 하지만 이걸 그대로 Gate H pass evidence로 쓰면 논리가 약하다.

### `dense_salt_polymer`

Evidence:
- [system.json](/home/kiket/Desktop/test/GROMACS_PCFF/testdata/lammps_golden/systems/dense_salt_polymer/system.json)
- [system.gro](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_4_charged_ensemble_gate/dense_salt_polymer/system.gro)
- [system.top](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_4_charged_ensemble_gate/dense_salt_polymer/system.top)
- [report.json](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_4_charged_ensemble_gate/dense_salt_polymer/report.json)
- [validation_report_tp1.md](/home/kiket/Desktop/test/GROMACS_PCFF/docs/validation_report_tp1.md)
- [tp1_status.json](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/transport_protocol_metadata/tp1_status.json)

Facts:
- `270 atoms`
- box `2.2 x 2.2 x 2.2 nm`
- charged multi-species system
- `constraints = none`
- existing ensemble evidence is already caveated
- TP1 long-equilibration rerun failed at `3.017 ns` with thermal runaway

Assessment:
- charged Gate H 후보로는 지금 쓰면 안 된다.
- 문제는 단순 under-sampling이 아니라 물리적 stability failure다.
- 이걸 더 길게 돌려서 transport를 보겠다는 건 real bottleneck을 피하는 선택이다.

## Repo-Wide Scan Result

Source: [testdata/lammps_golden/systems](/home/kiket/Desktop/test/GROMACS_PCFF/testdata/lammps_golden/systems)

`system.json`이 있는 in-repo fixtures는 아래뿐이다.

- `angle_toy`
- `bond_toy`
- `coulomb_toy`
- `dense_oligomer`
- `dense_salt_polymer`
- `dihedral_toy`
- `exclusion_toy`
- `improper_toy`
- `lj96_toy`
- `mixing_toy`
- `small_oligomer`
- `small_salt_polymer_box`

즉, repo 안에는 TP0 size floor를 만족하는 larger transport fixture가 없다.

## Recommendation

1. neutral Gate H를 다시 열 거면 `dense_oligomer`를 seed로 써서 `>=1000 atoms`, `>3.0 nm`로 새 fixture를 만들어라.
2. charged Gate H는 `dense_salt_polymer` 270-atom stability thread를 먼저 해결해라. 안정화 전 scale-up은 시간 낭비다.
3. exact-r-RESPA transport를 정말 정식 gate로 유지할 거면, TP0의 `RESPA deferred` 상태와 충돌하므로 Gate H용 protocol addendum을 먼저 얼려라.
