TP1.5d reruns the TP1.5b `dense_nonlisted` cut-off-only fixture under `n1_r0909` and `n10_r0909`.

It uses the env-var-gated `GMX_TP15D_*` trace in [sim_util.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp) to capture:
- pairlist age and rebuild-vs-reuse state
- dynamic-pruning enablement and prune-step state
- pair `(1,4)` under shift `21`
- current shifted distance versus `rlist`
- outer/inner active/excluded membership for that pair

Run:

```bash
python3 tools/run_tp1_5d_pairlist_branch_audit/run_pairlist_branch_audit.py
```
