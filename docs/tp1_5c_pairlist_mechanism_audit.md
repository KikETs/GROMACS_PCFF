# TP1.5c Pairlist Mechanism Audit

## Scope
TP1.5c stayed on the TP1.5b `dense_nonlisted` fixture and only compared `n1_r0909` versus `n10_r0909`. No transport work, PME work, or listed-route expansion was added.

## What Was Already Known
- TP1.5b reproduced the cut-off-only worsening direction and showed that `n1_r0909` stayed tight while `n10_r0909` widened the total-energy range.
- TP1.5 localized the runtime family to the plain-C CPU cut-off path.
- TP1.5b weakened blanket shift and listed-routing explanations, but it did not preserve per-step membership or force evidence.

## Instrumentation Strategy
TP1.5c added an env-var gated trace hook only.
- [sim_util.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp) writes per-step membership and force CSVs when `GMX_TP15C_*` variables are set.
- [pairlistset.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlistset.cpp) and [pairlistsets.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlistsets.cpp) expose active inner-list plain-pair extraction, so TP1.5c can distinguish current active membership from the outer list used for refreshes.
- The hook is inert when the env vars are unset.

## Executed Checks
The TP1.5c runner is [run_pairlist_trace_audit.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_5c_pairlist_trace_audit/run_pairlist_trace_audit.py). It executed:
- `n1_r0909`: `nstlist = 1`, `rlist = 0.909`, `verlet-buffer-tolerance = -1`
- `n10_r0909`: `nstlist = 10`, `rlist = 0.909`, `verlet-buffer-tolerance = -1`

Exact commands are in [raw_commands.txt](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/raw_commands.txt). Runtime logs confirming the same cut-off path are:
- [raw_n1_md.log](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/raw_n1_md.log)
- [raw_n10_md.log](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/raw_n10_md.log)

## Observed Results
The traced reruns matched TP1.5b exactly on the energy-level reproducer:
- `n1_r0909 total_energy_range_kj = 8.65774499999992`
- `n10_r0909 total_energy_range_kj = 12.57632499999994`

That is preserved in [membership_vs_force_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/membership_vs_force_summary.json).

The key separation results are:
- Against pruning inside an already-built outer list:
  - `n1_num_prune_steps = 0`
  - `n10_num_prune_steps = 0`
  - `n1_steps_with_outer_inner_difference = 0`
  - `n10_steps_with_outer_inner_difference = 0`
- For pairlist lifetime / stale membership:
  - The first cross-run inner-membership difference appears at step `171`
  - The first differing entry is `n1_only = [(1, 4, 21)]`
  - The first sensitive-atom force difference follows at step `173`
  - The near-cutoff `2-3` pair remains inner-active for `489` steps in `n10`, but only `214` in `n1`

Raw traces are:
- [per_step_pair_membership_n1.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/per_step_pair_membership_n1.csv)
- [per_step_pair_membership_n10.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/per_step_pair_membership_n10.csv)
- [per_step_force_trace_n1.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/per_step_force_trace_n1.csv)
- [per_step_force_trace_n10.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/per_step_force_trace_n10.csv)

## Interpretation
TP1.5c does not support “downstream force accumulation despite identical membership” as the leading story. The membership difference shows up first, and the force trace follows. That weakens a kernel-accumulation-first explanation.

TP1.5c also does not support a pruning bug inside an existing outer list. The traced runs never show outer-versus-inner divergence within a run, and no prune steps are recorded. The better fit is stale or long-lived membership caused by the list refresh cadence itself.

This is still not a final root cause. The result is: confirmed path issue at the pairlist lifetime / stale membership level, with exact source-level cause unresolved.

## Remaining Gaps
- Dirty-tree provenance remains; see [provenance_manifest.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/provenance_manifest.json)
- TP1.5c stayed on `dense_nonlisted` only, so listed-vs-nonlisted remains at TP1.5b’s deferred status
- The trace shows when membership diverges, not which branch in pairlist construction made the wrong keep/drop decision

## Narrow Next Step
Stay inside the pairlist path. Compare the list-construction decision for the first divergent `(1,4, shift 21)` entry across steps `170-173` between `n1_r0909` and `n10_r0909`, starting from pairlist rebuild/update code rather than kernel accumulation code.

files changed
- [src/gromacs/mdlib/sim_util.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp)
- [src/gromacs/nbnxm/nbnxm.h](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/nbnxm.h)
- [src/gromacs/nbnxm/nbnxm.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/nbnxm.cpp)
- [src/gromacs/nbnxm/pairlistset.h](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlistset.h)
- [src/gromacs/nbnxm/pairlistset.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlistset.cpp)
- [src/gromacs/nbnxm/pairlistsets.h](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlistsets.h)
- [src/gromacs/nbnxm/pairlistsets.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlistsets.cpp)
- [tools/run_tp1_5c_pairlist_trace_audit/run_pairlist_trace_audit.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_5c_pairlist_trace_audit/run_pairlist_trace_audit.py)
- [tests/reference_results/tp1_5c_pairlist_trace_audit](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit)

commands run
- `cmake --build build --target gmx -j4`
- `python3 -m py_compile tools/run_tp1_5c_pairlist_trace_audit/run_pairlist_trace_audit.py`
- `python3 tools/run_tp1_5c_pairlist_trace_audit/run_pairlist_trace_audit.py`

fixtures executed
- `dense_nonlisted` under `n1_r0909`
- `dense_nonlisted` under `n10_r0909`

strongest confirmed finding
- Cross-run active membership diverges before force divergence, which strengthens pairlist lifetime / stale membership over downstream kernel accumulation

strongest unresolved uncertainty
- The exact list-build or refresh branch that causes the stale `(1,4, shift 21)` membership is still not isolated

exact next step recommendation
- Add one more narrow trace at pairlist rebuild/update time around steps `170-173`, focused only on why `(1,4, shift 21)` is retained or dropped

verdict
- PASS
