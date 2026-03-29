# TP1.5d Validation Report

- milestone: `TP1.5d — Pairlist Rebuild / Refresh Decision Branch Trace`
- date: `2026-03-19`
- scope: TP1.5c의 `dense_nonlisted` 고정 fixture에서 `n1_r0909`와 `n10_r0909`의 `(1,4, shift 21)` 분기 차이를 step `170-173`에 한정해 추적
- verdict: `PASS`
- patching now justified: `NO`

## Outcome

TP1.5d는 분기 수준 차이를 실제로 추적했다. `(1,4, shift 21)` pair는 두 run 모두 step `170`에서는 `rlist_outer = 0.9089999795 nm` 바깥에 있고, step `171`에서 `0.9038842320 nm`로 내려오면서 cutoff 안으로 들어온다. 이때 `n1_r0909`는 step `171`에서 rebuild가 발생해 pair를 outer/inner active list에 포함시키고, `n10_r0909`는 rebuild가 없어 같은 step에서 pair를 계속 누락한다.

동시에 TP1.5d trace에서는 두 run 모두 `dynamic_pruning_enabled = false`, `prune_step = false`다. 따라서 이번 차이는 refresh/pruning 분기가 아니라 rebuild cadence 차이로 설명된다. 또한 shift `21`은 모든 추적 step에서 최소 이미지로 유지되어 shift 선택 오류 증거는 없다.

## Evidence Boundaries

- 확인된 사실:
  - step `171`이 첫 분기 시점이다.
  - `n1_r0909`만 step `171`에서 rebuild를 수행한다.
  - shift `21`은 두 run 모두에서 최소 이미지다.
  - pruning 분기는 이번 rerun에서 실행되지 않았다.
- 확인되지 않은 것:
  - 이 rebuild cadence 차이가 소스 수준 버그인지, 아니면 현재 `nstlist=10`, `rlist=0.909`, `verlet-buffer-tolerance=-1` 설정에서 기대 가능한 재사용 결과인지
  - list construction 내부의 어떤 세부 branch가 production patch를 요구하는지

## Key Artifacts

- branch trace: [branch_trace_n1.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/branch_trace_n1.csv), [branch_trace_n10.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/branch_trace_n10.csv)
- summary: [pair_1_4_decision_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/pair_1_4_decision_summary.json)
- source localization: [source_path_map.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/source_path_map.json)
- ranking: [tp1_5d_suspicion_ranking.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/tp1_5d_suspicion_ranking.json)
- raw debug: [raw_debug_n1.log](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/raw_debug_n1.log), [raw_debug_n10.log](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/raw_debug_n10.log)

## Reporting

- files changed
  - [src/gromacs/mdlib/sim_util.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp)
  - [tools/run_tp1_5d_pairlist_branch_audit/run_pairlist_branch_audit.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_5d_pairlist_branch_audit/run_pairlist_branch_audit.py)
  - [tools/run_tp1_5d_pairlist_branch_audit/README.md](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_5d_pairlist_branch_audit/README.md)
  - [docs/validation_report_tp1_5d.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/validation_report_tp1_5d.md)
  - [docs/tp1_5d_pairlist_branch_trace.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/tp1_5d_pairlist_branch_trace.md)
  - [tests/reference_results/tp1_5d_pairlist_branch_audit](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit)
- commands run
  - `git status --short`
  - `git rev-parse HEAD`
  - `build/bin/gmx --version`
  - `cmake --build build --target gmx -j4`
  - `python3 -m py_compile tools/run_tp1_5d_pairlist_branch_audit/run_pairlist_branch_audit.py`
  - `python3 tools/run_tp1_5d_pairlist_branch_audit/run_pairlist_branch_audit.py`
  - exact `gmx grompp` and `gmx mdrun` commands: [raw_commands.txt](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5d_pairlist_branch_audit/raw_commands.txt)
- fixtures executed
  - TP1.5b `dense_nonlisted` 4-atom periodic cut-off-only 9-6 fixture under `n1_r0909`
  - TP1.5b `dense_nonlisted` 4-atom periodic cut-off-only 9-6 fixture under `n10_r0909`
- strongest confirmed finding
  - step `171`에서 동일한 최소 이미지 pair `(1,4, shift 21)`가 cutoff 안으로 들어오자 `n1_r0909`는 rebuild로 pair를 포함하고 `n10_r0909`는 rebuild가 없어 pair를 계속 누락한다
- strongest unresolved uncertainty
  - 이 rebuild cadence 차이가 잘못된 production branch인지, 아니면 현재 pairlist margin 설정이 너무 타이트해서 생기는 기대 가능한 재사용 결과인지
- exact next step recommendation
  - `doPairSearch`와 fresh outer-list construction 사이에서 step `170-171`의 rebuild scheduling criterion과 buffer policy가 의도대로인지 먼저 검증하고, 그 전에는 production patch를 보류한다
- verdict
  - `PASS`
