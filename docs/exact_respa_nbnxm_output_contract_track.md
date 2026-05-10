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
- native single-launch multi-contribution write는 이 시점 기준으로 force-only narrow step에는 미구현이었다.
- PP-DD / multi-rank exact CPU NBNXM은 아직 미구현이다.
- wall-clock speedup claim은 아직 없다.
- baseline-broken generic `NbnxmTests`는 clean `HEAD`에서도 실패하므로, 이 트랙의 회귀 판정 근거로 쓰지 않는다.

## Phase 12 force-only native multi-contribution single-launch

`PerContributionLaunch`를 force-only exact narrow steps에서 `NativeMultiContribution`
single-launch로 바꿨다.

- `src/gromacs/mdlib/sim_util.cpp`
- `src/gromacs/mdtypes/interaction_const.h`
- `src/gromacs/nbnxm/atomdata.h`
- `src/gromacs/nbnxm/atomdata.cpp`
- `src/gromacs/nbnxm/kernel_common.h`
- `src/gromacs/nbnxm/kerneldispatch.cpp`
- `src/gromacs/nbnxm/kernels_reference/kernel_ref_outer.h`
- `src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h`
- `src/gromacs/nbnxm/simd_kernel.h`
- `src/gromacs/nbnxm/simd_kernel_inner.h`
- `src/gromacs/nbnxm/tests/pcff_class2_nonbonded.cpp`

현재 연결 방식:

- `computeExactRespaNonbondedCpuNbnxmNarrow()`가 force-only / no-energy / no-virial narrow step에서
  `NbnxmOutputContractKind::NativeMultiContribution`를 만든다.
- `nonbonded_verlet_t::dispatchExactRespaCpuNativeMultiKernel()`가
  `Inner/Middle/Outer` contribution metadata를 한 번에 kernel launch guard에 심고
  CPU NBNXM kernel을 한 번만 실행한다.
- plain-C와 SIMD kernel 둘 다 현재 output-buffer slot에 대응하는 native contribution buffer를 찾아
  contribution별 `f/fshift`에 직접 쓴다.
- energy/virial owner semantics가 필요한 step은 여전히 `PerContributionLaunch` fallback을 탄다.
- `GMX_PCFF_EXACT_RESPA_NATIVE_MULTI=0`으로 기존 per-launch runtime을 강제로 유지할 수 있다.

검증:

- `nbnxm-test --gtest_filter='*ExactRespa*:*PcffClass2NonbondedCurveTest*'`: PASS
- 새 force-only unit tests:
  - `PcffClass2NonbondedCurveTest.ExactRespaNativeMultiForceOnlyMatchesPerContributionLaunch`
  - `PcffClass2NonbondedCurveTest.ExactRespaNativeMultiForceOnlyKeepsExcludedPmeCorrectionOuterOnly`
- `nbnxm-output-contract-test`: PASS
- short runtime smoke:
  - fixture: `output/repulsion_power_9_exact_respa_cpu_patch_perf/gate_h_dense_salt_polymer_2x2x2/exact_respa_cpu_patch_perf.tpr`
  - shape: `-nsteps 2000 -ntmpi 1 -ntomp 6 -dlb no -nb cpu -pme cpu -bonded cpu -update cpu -pin off -reprod`
  - report: `output/exact_respa_native_multi_probe/gate_h_ntomp6_20260417/report.json`
  - result: `per_launch 34.729 ns/day -> native_multi 37.148 ns/day`
  - result: `Force 0.853 s -> 0.640 s`, `Update 1.210 s -> 1.145 s`
- automated runtime parity harness:
  - script: `tools/pcff_respa_parity/validate_exact_respa_native_multi_runtime.py`
  - gate_h report: `output/exact_respa_native_multi_probe/gate_h_runtime_parity_20260417/report.json`
  - gate_i report: `output/exact_respa_native_multi_probe/gate_i_equil_runtime_parity_20260417/report.json`
  - both reports show runtime hash mismatch on the 2000-step short run, but step-0 total-force and per-level force deltas are `0.0`
- same-coordinate continuation probe:
  - method: reuse the original TPR, derive a short continuation TPR with `gmx convert-tpr`, and continue from the baseline checkpoint with `-cpi`
  - gate_h: same-coordinate `2000 -> 2004` probe gives `total/per-level force = 0.0`, `energy = 0.0`, `gro = 0.0`
  - gate_i: same-coordinate first frame at step `2000` gives `total/per-level force = 0.0`, `energy = 0.0`, `gro = 0.0`
  - gate_i still shows a later force-dump delta at recorded frame `2400`, but the same probe keeps `energy = 0.0` and final `gro = 0.0`, so this is not evidence of an immediate same-state semantic mismatch

현재 honest boundary:

- force-only narrow exact step에 한해서 native multi-contribution single-launch는 구현/연결/검증됐다.
- wall-clock gain은 audited short runtime에서 `gate_h ~1.15x`, `gate_i ~1.07x`다.
- 2000-step runtime `run.gro`, `run.edr`, `run.cpt` SHA256는 일치하지 않는다.
- 하지만 `same-state first-frame` force/energy/virial/gro parity는 `gate_h`와 `gate_i` 둘 다 닫혔다.
- 따라서 현재 증거는 `bitwise identical` claim이 아니라
  `force-only runtime integration + same-state first-frame parity + bounded observable parity + small host-local speedup`까지다.
- energy/virial owner step native migration은 아직 미구현이었다.

## Next step

1. gate_i / gate_h의 later whole-run force-dump divergence가 dump cadence 문제인지, reduction-order-driven trajectory divergence인지 더 좁힐 것
2. owner-step native migration 이후에도 남는 whole-run hash mismatch를 어떤 수준까지 public parity claim에 포함할지 정리할 것
3. 그 다음에야 broader dataflow migration 또는 더 큰 wall-clock optimization을 열 것

## Phase 13 owner-step native multi-contribution migration

force-only에 한정돼 있던 native multi single-launch를 exact narrow owner-step
energy/virial path까지 확장했다.

- `src/gromacs/mdlib/sim_util.cpp`
- `src/gromacs/nbnxm/nbnxm.h`
- `src/gromacs/nbnxm/kerneldispatch.cpp`
- `src/gromacs/nbnxm/tests/pcff_class2_nonbonded.cpp`

현재 연결 방식:

- `computeExactRespaNonbondedCpuNbnxmNarrow()`가 owner step에서도
  `NbnxmOutputContractKind::NativeMultiContribution`를 선택할 수 있다.
- `dispatchExactRespaCpuNativeMultiKernel()`는 더 이상 synthetic force-only
  workload를 쓰지 않고 실제 `StepWorkload`와 real-space energy sink를 받는다.
- native multi launch에서 force/shift force는 contribution-indexed output
  buffers로 쓰고, owner-step energy는 기존 kernel energy arrays/reduction을
  그대로 사용한다.
- direct virial owner는 base output buffer가 아니라 owner contribution의
  native `fshift` buffers를 reduce해서 virial matrix를 재구성한다.

검증:

- `nbnxm-output-contract-test --gtest_filter='*Native*:*ExactRespa*'`: PASS
- `nbnxm-test --gtest_filter='*ExactRespaNativeMultiForceOnlyMatchesPerContributionLaunch:*ExactRespaNativeMultiOwnerEnergyMatchesPerContributionLaunch:*ExactRespaNativeMultiForceOnlyKeepsExcludedPmeCorrectionOuterOnly'`: PASS
- runtime owner-step parity reports:
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_owner_probe/local_9900x_gate_h_owner_native_multi_runtime_report.json`
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_owner_probe/local_9900x_gate_i_owner_native_multi_runtime_report.json`
- same-coordinate continuation probe:
  - `gate_h`: final `gro` identical, `Potential/Total Energy` delta `0`,
    force delta and virial/pressure deltas are small
  - `gate_i`: final `gro` identical, `Potential/Total Energy` delta `0`,
    force delta and virial/pressure deltas are small
- full 2000-step runs still show later force/virial divergence after
  trajectories separate, so whole-run hash parity는 닫히지 않았다.

현재 honest boundary:

- native multi single-launch는 force-only narrow step뿐 아니라 owner-step
  energy/virial path까지 실제 runtime에 연결됐다.
- 같은 state에서의 first-frame / short continuation parity는 지지된다.
- 하지만 whole-run `gro/edr/cpt` hash parity와 full-trajectory identity는
  여전히 지지되지 않는다.
- 따라서 지금 가능한 가장 강한 문장은
  `owner-step runtime integration + bounded same-state observable parity + host-local speedup`
  까지다.

## Phase 14 divergence onset narrowing

owner-step native migration 이후 남아 있던 “later whole-run divergence”를
step-count scan으로 더 좁혔다.

- new harness:
  `tools/pcff_respa_parity/scan_exact_respa_native_multi_divergence_onset.py`
- canonical reports:
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_divergence_onset_probe/local_9900x_gate_h_native_multi_divergence_onset_report.json`
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_divergence_onset_probe/local_9900x_gate_i_native_multi_divergence_onset_report.json`

결과:

- `gate_h`
  - force delta stays at `5.4931640625e-04` through `64` steps
  - grows to `3.2586669921875e-01` at `256`
  - grows to `1.7891845703125` at `1024`
  - grows to `13.73583984375` at `2000`
- `gate_i`
  - force delta stays at `5.4931640625e-04` through `256` steps
  - grows to `2.182708740234375` at `1024`
  - grows to `6.3876953125` at `2000`
- both fixtures lose `edr` hash identity at `4` steps and `gro` hash identity
  at `64` steps, but the large force divergence appears substantially later

현재 해석:

- 이것은 dump cadence mismatch보다는 reduction-order-driven trajectory
  branching 설명과 더 잘 맞는다.
- 즉, immediate owner-step semantic failure 증거는 더 약해졌고,
  whole-run bitwise identity가 닫히지 않았다는 사실만 남았다.

## Phase 15 serial reduction probe

`GMX_PCFF_EXACT_RESPA_NBNXM_SERIAL_REDUCTION=1`를 추가해서 native-multi
output-buffer reduction 자체를 serial diagnostic mode로 고정한 뒤 같은
onset scan을 다시 돌렸다.

- reports:
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_serial_reduction_probe/local_9900x_gate_h_native_multi_serial_reduction_report.json`
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_serial_reduction_probe/local_9900x_gate_i_native_multi_serial_reduction_report.json`

결과:

- `gate_h`와 `gate_i` 모두 default onset curve와 serial-reduction onset
  curve가 수치적으로 동일하다.
- 따라서 final NBNXM output-buffer reduction은 later whole-run divergence의
  핵심 원인으로 지지되지 않는다.

현재 더 강한 해석:

- divergence source는 reduction tail보다 upstream native-multi kernel-side
  accumulation / arithmetic ordering 쪽에 있을 가능성이 더 높다.
- whole-run identity는 여전히 닫히지 않았지만, 원인 후보는 더 좁아졌다.

## Phase 16 ntomp=1 and plain-C falsification probes

남아 있던 whole-run divergence에 대해 두 가지 쉬운 핑계를 더 제거했다.

- canonical `ntomp=1` reports:
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_ntomp1_divergence_probe/local_9900x_gate_h_native_multi_ntomp1_divergence_report.json`
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_ntomp1_divergence_probe/local_9900x_gate_i_native_multi_ntomp1_divergence_report.json`
- canonical `GMX_DISABLE_SIMD_KERNELS=1`, `ntomp=1` plain-C reports:
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_plainc_divergence_probe/local_9900x_gate_h_native_multi_plainc_ntomp1_divergence_report.json`
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_plainc_divergence_probe/local_9900x_gate_i_native_multi_plainc_ntomp1_divergence_report.json`

결과:

- `ntomp=1`로 줄여도 divergence는 사라지지 않는다.
  - `gate_h`: step `2000` force delta `3.6497650146484375`
  - `gate_i`: step `2000` force delta `765.7132415771484`
- plain-C reference kernel로 강제해도 divergence는 사라지지 않는다.
  - `gate_h`: step `2000` force delta `14.1541748046875`
  - `gate_i`: step `2000` force delta `1278.444320678711`
- plain-C report는 `disable_simd_kernels_env=1`,
  `disable_simd_kernels_marker_seen=true`를 남긴다.

현재 더 강한 해석:

- remaining divergence를 “OpenMP thread fan-out 문제”로만 보는 건 근거가 약하다.
- remaining divergence를 “SIMD kernel 특이 문제”로만 보는 것도 근거가 약하다.
- 가장 강한 남은 후보는 native-multi single-launch 내부의 contribution
  interleaving / arithmetic grouping이다.

## Phase 17 owner-step fallback closure

dense force-dump를 다시 보니 예전 해석의 핵심 빈틈이 드러났다. owner-level
exact-r-RESPA step 중에는 `computeEnergy=0`, `computeVirial=0`인 force-only
pass도 있는데, 이전 fallback heuristic은 이를 owner-step으로 보지 못했다.

수정:

- `src/gromacs/mdrun/md.cpp`에 `GMX_EXACT_RESPA_FORCE_DUMP_INTERVAL`를 추가해
  dense dump가 `nstenergy` cadence에 묶이지 않게 했다.
- `src/gromacs/mdlib/sim_util.cpp`에서 owner-step 판정을
  `highestActiveLevel == exactRespaNonbondedOuterLevel(inputrec)`로 바꿨다.
- `GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_OWNER_STEP_FALLBACK`를 default-on으로
  전환했다. `0`을 주면 이전 divergent owner-native path를 다시 강제할 수 있다.

결과:

- middle-only fallback은 `gate_h/gate_i` 둘 다 step `0`부터 mismatch가 난다.
- owner-only fallback은 dense interval-1 scan에서 둘 다 `32`, `256`, `1024`
  steps까지 total/per-level force, energy, gro가 모두 exact다.
- default owner-fallback `2000`-step runtime도 둘 다 닫힌다.
  - `gro`/`edr` hash equal
  - total-force/per-level-force/energy/gro exact
  - `cpt` hash는 여전히 다름

canonical artifacts:

- `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_owner_fallback_probe/local_9900x_gate_h_owner_fallback_dense_1024_report.json`
- `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_owner_fallback_probe/local_9900x_gate_i_owner_fallback_dense_1024_report.json`
- `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_owner_fallback_probe/local_9900x_gate_h_default_owner_fallback_runtime_report.json`
- `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_owner_fallback_probe/local_9900x_gate_i_default_owner_fallback_runtime_report.json`
- `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_owner_fallback_probe/summary.tsv`

현재 가장 강한 해석:

- audited `gate_h/gate_i` local runtime에서는 “native-multi whole-run
  divergence”를 더 이상 general unresolved 문제로 두면 안 된다.
- 실제 주범은 owner-level native-multi launch였고, owner-step을 legacy
  per-contribution launch로 되돌리면 audited parity가 닫힌다.
- 다만 이것으로 restart-bitwise identity나 broad cross-host closure까지
  주장하면 과장이다.

## Phase 18 default safe runtime closure

Phase 17 이후 남은 native-multi ambiguity를 더 좁혔다. owner-step fallback만으로는
middle-level force-only step의 first-frame ULP drift가 남을 수 있으므로, 기본 실행
경로는 owner step과 middle step 모두 legacy per-contribution launch로 되돌린다.

수정:

- `src/gromacs/mdlib/sim_util.cpp`
  - `GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_OWNER_STEP_FALLBACK` default-on 유지
  - `GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_MIDDLE_STEP_FALLBACK` default-on 전환
  - decision trace가 owner/middle fallback과 native-multi eligibility를 기록
- `tools/pcff_respa_parity/validate_exact_respa_native_multi_runtime.py`
  - `--probe-steps 0` 지원 추가
- `src/gromacs/nbnxm/tests/pcff_class2_nonbonded.cpp`
  - dense native-multi unit fixture의 SIMD ULP envelope를 `Cpu4xN_Simd_4xN`과
    `Cpu4xN_Simd_2xNN` 모두에 대해 좁게 허용

검증:

- `cmake --build build -j8 --target gmx nbnxm-test`: PASS
- `nbnxm-test --gtest_filter='PcffClass2NonbondedCurveTest.*ExactRespa*'`: PASS
- default safe 10000-step runtime parity:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_default_safe_probe/local_9900x_gate_i_default_owner_middle_fallback_runtime_10000_report.json`
  - total force delta `0.0`
  - per-level force delta `0.0`
  - energy/virial/pressure tracked terms delta `0.0`
  - final GRO coordinate/box delta `0.0`
  - same-coordinate continuation probe total/per-level/energy/GRO delta `0.0`
  - performance is noise-level only: `61.237 -> 61.526 ns/day`
- fused initial-drift update probe:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_default_safe_probe/local_9900x_gate_i_default_owner_middle_fallback_fused_update_runtime_10000_report.json`
  - exactness delta remains `0.0`
  - performance is slightly worse: `62.490 -> 61.933 ns/day`
- forced owner-native negative controls:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_default_safe_probe/local_9900x_gate_i_forced_owner_native_runtime_fail_report.json`
  and
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_default_safe_probe/local_9900x_gate_i_forced_owner_native_plainc_runtime_fail_report.json`
  - forced owner-native remains invalid despite apparent speedup
  - plain-C forced owner-native still has nonzero first-frame force/energy delta
- split-owner sidecar retest at 10000 steps:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_default_safe_probe/local_9900x_gate_i_split_owner_middle_fallback_runtime_10000_fail_report.json`
  - this mode has a small apparent speedup, but fails long enough runtime parity
    on Gate I: total force delta `9181.2333984375`, per-level force delta
    `9455.427734375`, energy delta `3927.1699999999996`, and GRO coordinate
    delta `4.831 nm`

현재 honest boundary:

- 기본 exact-r-RESPA CPU runtime parity는 audited Gate I 10000-step 기준으로 닫혔다.
- 이 closure는 owner/middle fallback에 의해 성립한다. 즉 full native-multi
  single-launch completion claim이 아니다.
- `GMX_PCFF_EXACT_RESPA_FUSED_INITIAL_DRIFT=1`은 exact하지만 이 fixture에서는
  성능 이득이 없으므로 default-off로 유지한다.
- forced owner-native / full native-multi owner step은 여전히 장기 replica에 쓰면
  안 된다.
- split-owner sidecar도 2000-step evidence만으로 Gate I 장기 replica 후보가 될 수
  없다. 10000-step negative control에서 실패했다.
- 다음 Gate I density/volume replica는 default safe path로만 시작할 수 있다.
