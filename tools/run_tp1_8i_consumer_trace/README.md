TP1.8i runner for tracing the next active kinetic/temperature consumer after `Update::update_coords` under the fixed authoritative safe baseline.

Usage:

```bash
python3 tools/run_tp1_8i_consumer_trace/run_consumer_trace.py
```

What it does:

- reruns the authoritative `safe_pme_shift_ref` baseline and `safe_ewald_shift` variant
- enables `GMX_TP18I_TRACE_FILE` to capture `compute_globals` kinetic/temperature consumer rows
- preserves raw `grompp`, `mdrun`, `md.log`, `mdout.mdp`, energy outputs, and machine-readable summaries under `tests/reference_results/tp1_8i_consumer_trace/`

Trace scope:

- incoming velocity summary used by the next active kinetic consumer
- `calc_ke_part` aggregate tensor outputs
- `sum_ekin` aggregate `Temperature` / `KineticEnergy` outputs
- whether a rank-reduction path is actually active

Non-goals:

- no r-RESPA work
- no transport calculations
- no inactive pressure-control tracing under `pcoupl = no`
- no production logic patching
