TP1.8g runner for immediate post-compute_globals consumer tracing on the authoritative charged-system safe baseline.

Outputs:
- `tests/reference_results/tp1_8g_consumer_trace/consumer_trace_baseline.csv`
- `tests/reference_results/tp1_8g_consumer_trace/consumer_trace_variant.csv`
- machine-readable comparison JSON files and preserved raw logs

The runner reuses the TP1.8f authoritative safe baseline settings and compares:
- `safe_pme_shift_ref`
- `safe_ewald_shift`

Trace instrumentation is enabled with `GMX_TP18G_TRACE_FILE`.
