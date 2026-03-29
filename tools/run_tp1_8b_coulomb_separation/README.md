TP1.8b reuses the authoritative safe short-range baseline and runs narrower Coulomb-path variants:

- `safe_pme_shift_ref`: baseline Coulomb PME with potential shift
- `safe_pme_tight_mesh`: tighter PME reciprocal accuracy
- `safe_ewald_shift`: full Ewald without the PME mesh, keeping the Ewald-family direct-space Coulomb path
- `safe_pme_none`: Coulomb PME with `coulomb-modifier = None`

Run with:

```bash
python3 tools/run_tp1_8b_coulomb_separation/run_coulomb_separation.py
```

Outputs are written to `tests/reference_results/tp1_8b_coulomb_separation/`.
