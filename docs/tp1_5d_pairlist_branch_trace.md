# TP1.5d Pairlist Branch Trace

## Goal

TP1.5d의 목적은 TP1.5c에서 확인된 `(1,4, shift 21)` membership divergence를 source branch 수준으로 더 좁히는 것이다. 범위는 `dense_nonlisted` fixture의 `n1_r0909` 대 `n10_r0909` 비교와 step `170-173`만이다.

## Constraining Prior Evidence

- TP1.5c는 동일 fixture에서 cross-run membership divergence의 첫 step을 `171`로 기록했다.
- TP1.5c는 pruning이 아니라 membership divergence가 먼저라고 좁혔다.
- TP1.5d가 추가로 설명해야 하는 것은 왜 step `171`에서 `n1`에만 `(1,4, shift 21)`가 나타나는지다.

## Narrow Code Path Map

주요 경로는 [source_path_map.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/source_path_map.json)에 저장했다.

- [sim_util.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp)
  - `doPairSearch`: search step에서 rebuild를 스케줄하고 `constructPairlist`를 호출한다.
  - `do_nb_verlet`: TP1.5d trace를 기록하는 runtime entry point다.
- [pairlistsets.h](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlistsets.h)
  - `numStepsWithPairlist`, `isDynamicPruningStepCpu`: pairlist age와 prune step 판정을 노출한다.
- [pairlist.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlist.cpp)
  - `PairlistSets::construct`: outer-list creation step를 갱신한다.
  - `nbnxn_make_pairlist_part`: near-cutoff pair admission과 shift/image 선택을 담당한다.
  - `prepareListsForDynamicPruning`: pruning 활성화 시 inner/outer list 분리를 담당한다.
- [prunekerneldispatch.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/prunekerneldispatch.cpp)
  - `PairlistSet::dispatchPruneKernel`: pruning 분기다. 이번 rerun에서는 실행 증거가 없다.

## Instrumentation Strategy

`[sim_util.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp)`에 env-var gated debug trace를 추가했다. TP1.5d runner는 다음을 step별로 기록한다.

- rebuild 여부
- pairlist age
- dynamic pruning enabled 여부
- prune step 여부
- `rlist_outer`, `rlist_inner`
- pair `(1,4)`에 대해 요청한 shift `21`
- 실제 geometry orientation
- target shift 거리
- minimum-image shift index와 거리
- outer/inner active/excluded list에서의 pair 존재 여부

이 trace는 [raw_debug_n1.log](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/raw_debug_n1.log)와 [raw_debug_n10.log](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/raw_debug_n10.log)로 보존되고, 같은 내용을 CSV로 정규화한 결과가 [branch_trace_n1.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/branch_trace_n1.csv)와 [branch_trace_n10.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/branch_trace_n10.csv)다.

## Controlled Reruns

동일한 TP1.5b `dense_nonlisted` fixture를 그대로 사용했다.

- topology: 동일
- coordinates: 동일
- charges: 동일
- box: 동일
- `rep-pow = 9`: 동일
- `rcoulomb = 0.9`, `rvdw = 0.9`, `rlist = 0.909`: 동일
- `verlet-buffer-tolerance = -1`: 동일
- integrator family와 seed: 동일
- 달라지는 값: `nstlist`만 `1` 대 `10`

실행 명령은 [raw_commands.txt](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/raw_commands.txt), provenance는 [provenance_manifest.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/provenance_manifest.json)에 있다.

## Branch-Level Findings

### Confirmed Difference

- step `170`
  - 두 run 모두 `target_shift_distance_nm = 0.9095347524`
  - 두 run 모두 `rlist_outer_nm = 0.9089999795`
  - pair는 둘 다 absent
- step `171`
  - 두 run 모두 `target_shift_distance_nm = 0.9038842320`
  - shift `21`은 둘 다 minimum image
  - `n1_r0909`: `rebuild_this_step = true`, pair present in outer/inner active list
  - `n10_r0909`: `rebuild_this_step = false`, `pairlist_age = 1`, pair still absent

즉, 확인된 분기 차이는 refresh/pruning이 아니라 rebuild cadence다.

### Weakened Alternatives

- refresh/pruning logic
  - 두 run 모두 `dynamic_pruning_enabled = false`
  - 두 run 모두 `prune_step = false`
  - 이번 trace는 pruning path를 전혀 밟지 않는다
- shift/image selection bug
  - 두 run 모두 step `168-173`에서 `min_shift_index = 21`
  - traced pair의 실제 geometry orientation은 일관되게 `(4,1,21)`이다
  - 잘못된 shift 선택 증거는 없다

### Still Unresolved

- rebuild scheduling 또는 buffer policy가 production expectation과 어긋나는지
- `nstlist = 10`, `rlist = 0.909`, `verlet-buffer-tolerance = -1` 조합에서 이런 pair crossing 누락이 설계상 허용되는지
- 따라서 지금 단계에서는 source-level bug를 단정할 수 없다

## Patch Readiness

현재 증거만으로는 minimal production patch를 정당화할 수 없다.

이유:
- TP1.5d는 branch-level difference를 확인했지만, “잘못된 branch decision”을 증명하지는 못했다.
- pruning bug 가설은 이번 rerun에서 약화됐다.
- shift-selection bug 가설도 약화됐다.
- 남은 핵심 질문은 rebuild cadence와 buffer policy가 설계상 의도인지 여부다.

## Next Step

다음 단계는 patch가 아니라, step `170-171`의 rebuild scheduling criterion과 pairlist margin policy가 의도와 맞는지 source-level contract를 확인하는 것이다. 필요하면 그때 `doPairSearch`와 outer-list construction 경계에 더 좁은 trace를 추가하면 된다.
