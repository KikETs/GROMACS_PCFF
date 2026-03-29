# TP1.5c Validation Report

Verdict: PASS

TP1.5c reran the TP1.5b `dense_nonlisted` cut-off-only reproducer with per-step tracing enabled and preserved raw plus derived evidence under [tests/reference_results/tp1_5c_pairlist_trace_audit](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit). The traced reruns reproduced the TP1.5b energy behavior exactly: `n1_r0909` kept `total_energy_range_kj = 8.65774499999992`, while `n10_r0909` kept `12.57632499999994`; see [membership_vs_force_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/membership_vs_force_summary.json).

The strongest mechanism-level result is narrower than “kernel bug.” Within each run, outer and inner memberships stayed identical and no prune steps were recorded: `n1_num_prune_steps = 0`, `n10_num_prune_steps = 0`, and `steps_with_outer_inner_difference = 0` for both runs. That weakens a pruning-inside-existing-list story. Across runs, however, the active inner membership first diverged at step `171`, and the first sensitive-atom force divergence followed at step `173`; see [membership_vs_force_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/membership_vs_force_summary.json) and [tp1_5c_suspicion_ranking.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/tp1_5c_suspicion_ranking.json).

This is evidence for pairlist lifetime / stale membership sensitivity on the plain-C cut-off path, not proof of a final source-level root cause. The first traced membership difference is one extra active `(1,4, shift 21)` entry in `n1` at step `171`, while the traced near-cutoff `2-3` pair remains active much longer in `n10` (`489` inner-active steps versus `214` in `n1`). Those are concrete path-level differences, but TP1.5c does not yet prove which exact list-build or refresh branch is wrong.

Remaining limits:
- The tree and binary are dirty; provenance is recorded in [provenance_manifest.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/provenance_manifest.json).
- The tracing hook is env-var gated and was added only for TP1.5c.
- Listed-vs-nonlisted routing stays deferred to TP1.5b conclusions.

files changed
- [src/gromacs/mdlib/sim_util.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp)
- [src/gromacs/nbnxm/nbnxm.h](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/nbnxm.h)
- [src/gromacs/nbnxm/nbnxm.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/nbnxm.cpp)
- [src/gromacs/nbnxm/pairlistset.h](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlistset.h)
- [src/gromacs/nbnxm/pairlistset.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlistset.cpp)
- [src/gromacs/nbnxm/pairlistsets.h](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlistsets.h)
- [src/gromacs/nbnxm/pairlistsets.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlistsets.cpp)
- [tools/run_tp1_5c_pairlist_trace_audit/run_pairlist_trace_audit.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_5c_pairlist_trace_audit/run_pairlist_trace_audit.py)
- [docs/tp1_5c_pairlist_mechanism_audit.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/tp1_5c_pairlist_mechanism_audit.md)
- [tests/reference_results/tp1_5c_pairlist_trace_audit](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit)

commands run
- `cmake --build build --target gmx -j4`
- `python3 -m py_compile tools/run_tp1_5c_pairlist_trace_audit/run_pairlist_trace_audit.py`
- `python3 tools/run_tp1_5c_pairlist_trace_audit/run_pairlist_trace_audit.py`
- Exact executed subcommands are preserved in [raw_commands.txt](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/raw_commands.txt)

fixture executed
- TP1.5b `dense_nonlisted` 4-atom periodic cut-off-only 9-6 fixture, rerun under `n1_r0909` and `n10_r0909`; see [dense_fixture_reference.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit/dense_fixture_reference.json)

strongest confirmed finding
- Pairlist lifetime / stale membership is directly implicated: cross-run inner membership differs at step `171`, while force divergence on sensitive atoms follows at step `173`

strongest unresolved uncertainty
- The exact code site that lets stale membership persist under `nstlist=10` is still not isolated

exact next step recommendation
- Instrument pairlist rebuild / refresh code around the first divergent `(1,4, shift 21)` entry, and compare the list-construction decision at steps `170-173` for `n1_r0909` versus `n10_r0909` before attempting a production fix

verdict
- PASS
