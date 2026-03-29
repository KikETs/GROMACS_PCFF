# TP1.6 regression runner

Runs the focused TP1.6 regressions:

- existing TP1.4 isolated split scan (`tools/run_tp1_4_pme_proof/test_split.py`)
- new mixed-type 9-6 LJ-PME startup fixture that reproduced the pre-fix assert

Usage:

```bash
python3 tools/run_tp1_6_regressions/run_regressions.py
```

Artifacts are written to `tests/reference_results/tp1_6_regressions/`.
