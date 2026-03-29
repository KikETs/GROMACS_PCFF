TP1.5c runner for the dense cut-off pairlist mechanism audit.

Entry point:
- [run_pairlist_trace_audit.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_5c_pairlist_trace_audit/run_pairlist_trace_audit.py)

What it does:
- rebuilds `gmx`
- reruns the TP1.5b `dense_nonlisted` fixture for `n1_r0909` and `n10_r0909`
- enables env-var gated TP1.5c tracing during `mdrun`
- writes raw logs plus machine-readable artifacts under [tests/reference_results/tp1_5c_pairlist_trace_audit](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5c_pairlist_trace_audit)

Scope limit:
- no transport calculations
- no PME work
- no listed-route expansion
