TP1.8 reruns the TP1.7b authoritative charged-system setup with the short-range safe baseline fixed and changes only long-range-relevant controls:

- `safe_pme_n10_r0911`: TP1.7b-style safe baseline
- `safe_pme_tight_fs006_po6`: tighter Coulomb PME accuracy
- `safe_cutoff_n10_r0911`: no-reciprocal Coulomb cut-off isolation variant

Run with:

```bash
python3 tools/run_tp1_8_longrange_isolation/run_longrange_isolation.py
```

Outputs are written to `tests/reference_results/tp1_8_longrange_isolation/`.
