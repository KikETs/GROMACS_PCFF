TP1.5e reruns the TP1.5b `dense_nonlisted` fixture under `n1_r0909` and `n10_r0909`
using the existing TP1.5d branch-trace instrumentation, but extends the trace window
to steps `160-180`.

Purpose:
- reconstruct the last rebuild before the TP1.5d divergence
- compute pair `(1,4, shift 21)` distance margins against `cutoff` and `rlist`
- classify the omission as `BUG`, `ALLOWED-UNSAFE`, or `UNRESOLVED`

Run:

```bash
python3 tools/run_tp1_5e_pairlist_contract_audit/run_pairlist_contract_audit.py
```
