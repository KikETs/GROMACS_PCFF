# TP1.8j Slice Trace Runner

Runs the authoritative `dense_salt_polymer` baseline and narrowed Ewald variant under an isolated post-update `compute_globals` trace.

Outputs:

- `tests/reference_results/tp1_8j_slice_trace/run_matrix.json`
- `tests/reference_results/tp1_8j_slice_trace/slice_trace_baseline.csv`
- `tests/reference_results/tp1_8j_slice_trace/slice_trace_variant.csv`
- `tests/reference_results/tp1_8j_slice_trace/slice_trace_summary.json`
- `tests/reference_results/tp1_8j_slice_trace/tp1_8j_recommendation.json`

Usage:

```bash
python3 tools/run_tp1_8j_slice_trace/run_slice_trace.py
```
