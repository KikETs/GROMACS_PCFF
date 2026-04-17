# Exact r-RESPA NBNXM Output Contract Track

## Scope

이 트랙은 `kernel write semantics` 자체를 바로 바꾸는 작업이 아니다.
우선 목표는 `nbnxm outputBuffer -> exact r-RESPA contribution sink` 사이의 계약을 명시적으로 만드는 것이다.

현재 범위:

- contribution-aware output sink contract 추가
- `Inner/Middle/Outer`별 host sink 선택/검증 API 추가
- contribution-indexed `nbat` output storage 추가
- native multi-contribution storage에서 declared sink로 force/shift-force reduction
- CPU NBNXM dispatch가 exact contribution launch를 조용히 full-force로 처리하지 못하도록 fail-fast guard 추가

현재 비범위:

- GPU 또는 hybrid runtime 연결
- native kernel multi-write
- direct virial native accumulation
- CPU exact pair-splitting path의 native `nbnxm` migration
- native energy/virial multi-write

## Why

기존 `nbnxm output model`에는 exact contribution ownership 정보가 없어서,
후속 `outputBuffer` 확장 작업의 출발점이 없었다.

## Phase 1 change

새 구조는 `NbnxmOutputSink`를 추가하고, `nonbonded_verlet_t::atomdata_add_nbat_f_to_outputs()`
가 active contribution에 맞는 sink를 선택해 reduce하도록 만든다.

적용 경로:

- `src/gromacs/nbnxm/nbnxm.h`
- `src/gromacs/nbnxm/nbnxm.cpp`

현재 contract 제한:

- contribution당 sink는 정확히 1개만 허용
- runtime exact path에는 아직 연결하지 않음
- native kernel write 자체는 여전히 single-output 의미를 유지

## Current status

- `nbnxm-output-contract-test` build: PASS
- `NbnxmOutputContractTests`: PASS
- `MdtypesUnitTest`: PASS
- GPU runtime exactness fixture: out of scope for this CPU-core track
- native kernel multi-write: not implemented

즉 현재 증거는 output contract/storage/reduction/staging 단위 검증이다. native single-launch
multi-write나 성능 개선 claim은 없다.

## Phase 2 change

contract를 `force/shift`에서 한 단계 더 늘려서, energy ownership도 명시적으로 담도록 확장했다.

- `NbnxmEnergyOutput`
- `NbnxmOutputContract`
- energy owner contribution을 contract에 기록하고 검증

이 단계도 아직 bounded다.

- native kernel energy multi-write는 없음
- direct virial native accumulation 확장은 없음
- output model에 energy owner metadata만 추가

## Phase 3 change

계약 경계를 더 명시적으로 만들기 위해 virial ownership도 별도 metadata로 올렸다.

- `NbnxmOutputSink`에 `directVirialOutput` 포인터 추가
- `NbnxmVirialOutput` 추가
- `NbnxmEnergyOutput`에 owner contribution 추가
- `atomdata_add_nbat_f_to_outputs()`가 active contribution과 energy/virial owner contract의 일관성을 assert

이 단계도 bounded다.

- native virial accumulation 의미는 아직 확장하지 않음
- native energy multi-write도 아직 없음
- 바뀐 것은 `output contract`의 ownership 명시성과 validation뿐임

## Phase 4 change

baseline-broken `NbnxmTests`에 기대지 않는 전용 회귀 하네스를 추가했다.

- 새 테스트 타겟: `NbnxmOutputContractTests`
- 새 테스트 파일: `src/gromacs/nbnxm/tests/outputcontract.cpp`
- 검증 대상:
  - contribution별 force sink lookup
  - missing/duplicate sink rejection
  - outer/full-only energy/virial ownership boundary
  - shift-force virial owner와 direct-virial owner mismatch rejection

이 하네스는 GPU state나 `nonbonded_verlet_t` 인스턴스 없이 contract validation을 직접 검증한다.

## Phase 5 change

`NbnxmOutputContract`가 실행 모델을 명시적으로 구분하도록 확장했다.

- `NbnxmOutputContractKind::PerContributionLaunch`
- `NbnxmOutputContractKind::NativeMultiContribution`
- `NbnxmNativeMultiContributionOutput`
- `nbnxmOutputSinksForNativeMultiContribution()`

계약상 현재 실행 모델은 `PerContributionLaunch`다.
`NativeMultiContribution`은 output model에서 여러 contribution sink와 outer energy/virial owner를
표현하고 검증할 수 있게 만든 contract-level 확장이다.

아직 완료되지 않은 것:

- native NBNXM kernel multi-write
- direct-virial native kernel accumulation
- native energy multi-write
- wall-clock 성능 개선 claim

## Phase 6 change

native multi-contribution contract를 NBNXM boundary에 연결했다.

- `nonbonded_verlet_t::atomdata_add_nbat_f_to_native_multi_outputs()`
- native contract validation은 수행
- contribution-indexed NBNXM kernel output buffers가 없으면 즉시 `InternalError`

이건 의도적인 fail-fast 연결이다. 현재 `nbnxn_atomdata_t::outputBuffers_`는 contribution별 buffer가
아니라 thread/list output buffer이므로, 이 버퍼를 native multi-write 결과처럼 재사용하면 force routing이
틀린다.

Phase 7/8에서 이 fail-fast는 storage/reduction 구현으로 대체되었다. 단, kernel native multi-write는
여전히 미구현이다.

추가 small test:

- native contract를 per-contribution lookup으로 조회하면 실패
- per-contribution contract를 native lookup으로 조회하면 실패
- native contribution order와 outer direct-virial/energy ownership은 contract layer에서 검증

## Phase 7 change

`nbnxn_atomdata_t`에 native exact r-RESPA contribution-indexed output storage를 추가했다.

- `ensureNativeMultiContributionOutputBuffers()`
- `nativeMultiContributionOutputBuffers()`
- `numNativeMultiContributionOutputSets()`

버퍼 구조는 contribution index가 먼저 오고, 각 contribution 아래에 기존 thread/list output buffer index가
오는 형태다. 기존 `outputBuffers_`는 normal NBNXM kernel output 의미를 유지한다.

검증:

- contribution별 output buffer 수가 기존 output buffer 수와 일치
- force buffer resize가 native storage에도 적용
- contribution storage가 서로 독립
- 같은 contribution count로 재호출하면 storage를 재생성하지 않음
- contribution count 변경 시 range validation 수행

## Phase 8 change

native multi-contribution storage를 NBNXM reduction boundary에 실제 연결했다.

- `nbnxn_atomdata_t::reduceForceOutputBuffers()`
- `nbnxn_atomdata_add_output_fshift_to_fshift()`
- `nonbonded_verlet_t::atomdata_add_nbat_f_to_native_multi_outputs()`

이제 native contract가 들어오면:

1. contract를 validation한다.
2. contribution-indexed `nbat` output storage를 준비한다.
3. 각 contribution output buffer를 declared force sink로 reduce한다.
4. `ShiftForce` sink가 shift-force를 소유하면 해당 contribution의 `fshift`도 reduce한다.

중요한 한계:

- 이 단계는 kernel write semantics를 바꾼 것이 아니다.
- 현재 NBNXM kernels는 아직 contribution-indexed output buffers에 직접 쓰지 않는다.
- 따라서 이 단계는 `output storage + reduction contract` 완성이지, native single-launch multi-write 완성이 아니다.
- 성능 개선 claim은 없음.

검증:

- `NbnxmOutputContractTests`: 15 tests PASS
- boundary test가 native storage allocation을 확인
- seeded native contribution buffers가 declared force/shift-force sinks로 reduce되는 것을 확인

## Phase 9 scope decision

GPU runtime 연결은 이 CPU-core 트랙에서 제외했다.

유지하는 것:

- `nbnxn_atomdata_t::copyOutputBuffersToNativeMultiContributionOutputBuffers()`
- normal output buffer에서 contribution-indexed storage로 복사하는 단위 테스트

유지하지 않는 것:

- exact GPU narrow path의 native-staged runtime 연결
- GPU fixture parity claim
- GPU/hybrid 성능 claim

이 API는 CPU/native output model 검증용 host-side staging primitive로만 남긴다. GPU runtime 확장은 별도
트랙에서 다시 열어야 한다.

## Phase 10 CPU semantic guard

CPU NBNXM dispatch에 fail-fast guard를 추가했다.

- 파일: `src/gromacs/nbnxm/kerneldispatch.cpp`
- 조건: CPU NBNXM kernel type에서 `stepWork.nonbondedRespaContribution != Full`이면 즉시 assert
- 이유: 현재 CPU NBNXM kernels는 `Inner/Middle/Outer` contribution semantics를 계산하지 않는다.

이 guard가 없으면 향후 CPU exact-r-RESPA NBNXM 연결 과정에서 contribution launch가 full-force 계산으로
조용히 처리될 수 있다. 따라서 이 변경은 성능 확장이 아니라 semantic ambiguity closure다.

아직 완료되지 않은 것:

- CPU NBNXM kernel 내부의 exact contribution scaling
- CPU native multi-contribution single-launch write
- scalar exact CPU path 대비 CPU NBNXM contribution parity
- CPU NBNXM 기반 wall-clock speedup claim

## Phase 11 CPU narrow runtime migration

contract/output-storage/kernel semantics를 실제 exact runtime에 연결했다.

- `src/gromacs/mdlib/sim_util.cpp`
- `src/gromacs/mdtypes/interaction_const.cpp`
- `src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h`
- `src/gromacs/nbnxm/simd_kernel_inner.h`
- `src/gromacs/nbnxm/kerneldispatch.cpp`

현재 연결 방식:

- execution model은 여전히 `PerContributionLaunch`
- `do_force()`가 exact CPU narrow 조건을 만족하면 `computeExactRespaNonbondedCpuNbnxmNarrow()`를 선택
- launch마다 `StepWorkload::withExactNonbondedContribution()`로 `Inner/Middle/Outer`를 분리
- CPU NBNXM kernels는 launch guard가 심어준 `interaction_const_t::exactRespaCpuPairSplit` metadata를 읽어
  direct-weighted force만 계산
- outer launch만 Coulomb exclusion correction, energy, virial owner를 가진다

현재 admitted narrow boundary:

- single-rank CPU-only exact nonbonded
- no PP domain decomposition
- no active M2P trace directory
- CPU kernel type in `{Cpu1x1_PlainC, Cpu4x4_PlainC, Cpu4xN_Simd_4xN, Cpu4xN_Simd_2xNN}`
- real-space `LJ cut + no modifier`
- Coulomb `PME/Ewald + no modifier`

핵심 보완:

- `init_interaction_const()`가 exact pair-splitting geometry를 `interactionConst.exactRespaCpuPairSplit`에 고정
- plain-C와 SIMD kernel 둘 다 split direct force를 계산
- outer `ForceWithVirial` launch는 NBNXM `fshift`를 host에서 reduce해서 virial matrix를 재구성한 뒤
  `ForceWithVirial::addVirialContribution()`로 넘긴다

검증:

- `nbnxm-output-contract-test`: PASS
- `nbnxm-test --gtest_filter='*ExactRespa*:*PcffClass2NonbondedCurveTest*'`: PASS
- `mdtypes-test --gtest_filter='*ExactRespa*:*MultipleTimeStepping*'`: PASS
- small runtime smoke:
  - fixture: `tests/reference_results/r_respa_m2k_narrow_patch_proof/.../patch_shape_a/exact.tpr`
  - command shape: `gmx mdrun -nsteps 4 -ntmpi 1 -ntomp 1 -nb cpu -pme cpu -bonded cpu -update cpu -pin off -reprod`
  - evidence: runtime log contains `Exact r-RESPA CPU nonbonded will use the narrow per-contribution NBNXM path.`

중요한 한계:

- 이건 narrow CPU runtime migration이다. broad CPU completion claim이 아니다.
- native single-launch multi-contribution write는 여전히 미구현이다.
- PP-DD / multi-rank exact CPU NBNXM은 아직 미구현이다.
- wall-clock speedup claim은 아직 없다.
- baseline-broken generic `NbnxmTests`는 clean `HEAD`에서도 실패하므로, 이 트랙의 회귀 판정 근거로 쓰지 않는다.

## Next step

1. `do_force()` narrow CPU NBNXM path에 대한 checked-in runtime regression harness를 추가
2. scalar exact CPU path 대비 force/energy/virial parity를 runtime fixture에서도 자동 비교
3. 그 다음에야 native single-launch multi-contribution write 또는 wall-clock optimization을 열 것
