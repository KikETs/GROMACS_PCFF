TP1.8h runner for tracing the active integration-state update path under the fixed authoritative safe baseline.

Usage:

```bash
python3 tools/run_tp1_8h_update_trace/run_update_trace.py
```

What it does:

- reruns the authoritative `safe_pme_shift_ref` baseline and `safe_ewald_shift` variant
- enables `GMX_TP18H_TRACE_FILE` to capture `Update::Impl::update_coords` aggregate trace rows
- preserves raw `grompp`, `mdrun`, `md.log`, `mdout.mdp`, energy outputs, and machine-readable summaries under `tests/reference_results/tp1_8h_update_trace/`

Trace scope:

- incoming force summary at the update boundary
- pre/post velocity summaries
- xprime increment summaries
- mass-weighted kinetic proxy before/after update

Non-goals:

- no r-RESPA work
- no transport calculations
- no pressure-control tracing under `pcoupl = no`
- no production logic patching
