# TP1.8c Coulomb Accumulation Trace Runner

This tool reruns the authoritative safe baseline from TP1.8b with source-level
trace instrumentation enabled through `GMX_TP18C_TRACE_FILE`.

Runs:

- `safe_pme_shift_ref`
- `safe_ewald_shift`
- `safe_pme_none`

Outputs:

- `tests/reference_results/tp1_8c_coulomb_trace/trace_observables_baseline.csv`
- `tests/reference_results/tp1_8c_coulomb_trace/trace_observables_variant.csv`
- `tests/reference_results/tp1_8c_coulomb_trace/trace_observables_direct_modifier.csv`
- raw `grompp`, `mdrun`, `md.log`, `mdout.mdp`, `energy` artifacts
- machine-readable summaries and recommendation JSON files

The trace is aggregate and source-level. It does not provide per-pair direct-space
force decomposition.
