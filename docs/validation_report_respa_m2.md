# R-RESPA M2 Current Status

## Scope

이 문서는 현재 구현 상태를 좁은 검증 범위 기준으로만 요약한다.

검증 범위:

- fixture: `dense_oligomer`
- integrator path: exact 3-level `mts-mode = lammps-respa`
- timestep: coarse `dt = 0.0005 ps`
- baseline comparison: plain Verlet
- primary analysis scope: step `0`

이 문서는 다음을 주장하지 않는다:

- full-system closure
- broad fixture closure
- long-trajectory closure
- production readiness

## Closed In Locked Scope

- event-669 LJ geometry branch는 locked scope에서 닫혔다.
  - semantic event identity는 plain/patch 양쪽에서 일치한다.
  - first arithmetic divergence는 `rsq/r`였고, upstream producer trace는 `dx/dy/dz` construction mismatch로 좁혀졌다.
  - engine fix는 exact path의 shifted-i coordinate producer shape를 plain reference와 맞춘 것이다.

- `Coulomb-(SR)` residual은 engine-side first cause가 아니라 comparator contract mismatch로 정리됐다.
  - plain native total과 patch total은 달랐다.
  - plain shadow replay in patch contract가 patch comparable total과 일치했다.
  - locked-scope comparator는 plain native `Coulomb-(SR)` 대신 plain patch-contract replay total을 기준으로 써야 한다.

- `LJ-(SR)` residual도 같은 종류의 comparator contract mismatch로 정리됐다.
  - plain native LJ total과 patch comparable total은 달랐다.
  - plain shadow replay in patch LJ contract가 patch pair-phase truth와 일치했다.
  - locked-scope comparator는 plain native `LJ-(SR)` 대신 plain patch-contract replay total을 기준으로 써야 한다.

- 현재 locked-scope baseline에서:
  - `Coulomb-(SR)` comparator delta는 `0.0`
  - `LJ-(SR)` comparator delta는 small numerical tail only
  - 남은 `Potential` residual은 component deltas로 설명된다

## Fixed In Engine

- [sim_util.cpp](../src/gromacs/mdlib/sim_util.cpp)
  - exact path geometry producer를 plain reference shape로 맞추는 shifted-i fix를 유지한다.
  - 핵심은 raw `coord_i - coord_j + shift` 대신 shifted-i coordinate를 먼저 materialize한 뒤 `dx/dy/dz`를 구성하는 것이다.

이 문서 기준으로 유지하는 engine fix는 이것뿐이다.

## Fixed In Comparator/Analysis

- [run_respa_m2.py](../tools/run_respa_m2_microfixtures/run_respa_m2.py)
  - locked-scope `Coulomb-(SR)` comparator는 plain native total이 아니라 plain patch-contract replay total을 plain reference로 사용한다.
  - locked-scope `LJ-(SR)` comparator도 같은 방식으로 plain patch-contract replay total을 plain reference로 사용한다.
  - artifact에는 native plain total과 replay plain total을 둘 다 남긴다.

## Still Open / Still Needs Validation

- 현재 closure는 locked scope only다.
  - `dense_oligomer`
  - coarse `dt = 0.0005 ps`
  - exact 3-level path
  - primarily step `0`

- `Potential` residual은 더 이상 standalone mystery가 아니다.
  - current locked-scope baseline에서는 remaining `LJ-(SR)` tail과 `Other-Terms` delta로 설명된다.
  - 다만 이것이 broader scope에서도 그대로 유지되는지는 아직 검증되지 않았다.

- 아직 필요한 검증:
  - same fixture, multiple steps
  - at least one additional harness fixture
  - same comparator-corrected baseline under a slightly broader validation matrix

## Working Summary

현재 상태를 한 줄로 줄이면 이렇다.

exact 3-level path의 locked-scope primary defects는 모두 engine-side physics bug로 남아 있지 않다. 남은 주요 차이는 comparator contract와 narrow residual accounting으로 설명되며, broader validation은 아직 열려 있다.
