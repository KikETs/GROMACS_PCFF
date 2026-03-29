`run_dense_cutoff_audit.py` executes TP1.5b only.

It creates:
- one dense periodic nonlisted 4-atom cut-off-only 9-6 fixture
- one routed sister fixture with exclusions plus an explicit pair
- a pairlist-only sweep over `nstlist`, `rlist`, and auto-buffer controls
- static fixed-frame reruns to separate pairlist lifetime effects from fixed-frame kernel behavior

Outputs are written to [tests/reference_results/tp1_5b_dense_cutoff_audit](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5b_dense_cutoff_audit).
