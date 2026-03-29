TP1.8f runner for compute_globals / pressure-handoff tracing on the authoritative charged-system safe baseline.

Outputs:
- `tests/reference_results/tp1_8f_compute_globals_trace/compute_globals_trace_baseline.csv`
- `tests/reference_results/tp1_8f_compute_globals_trace/compute_globals_trace_variant.csv`
- machine-readable comparison JSON files and preserved raw logs

The runner reuses the TP1.8e authoritative safe baseline settings and compares:
- `safe_pme_shift_ref`
- `safe_ewald_shift`

Trace instrumentation is enabled with `GMX_TP18F_TRACE_FILE`.
