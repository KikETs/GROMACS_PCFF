# TP1.8j Validation Report

## Verdict
PASS

## Scope
TP1.8j isolated only the post-update `compute_globals` callsite under the same authoritative safe baseline from TP1.8i and compared it against the narrowed Ewald variant. It stayed inside `integrator = md`, `pcoupl = no`, `tcoupl = no`, and did not mix pre-step `compute_globals` invocations into the traced slice.

## Strongest Confirmed Finding
TP1.8j fixed the main TP1.8i limitation: the trace now contains only `callsite = post_update_compute_globals`, with no `step = -1` rows and no extra `compute_globals` callsites.

Isolation checks passed:

- `baseline_reuses_tp18i_safe_settings = true`
- `short_range_fields_fixed_across_runs = true`
- `baseline_isolated_post_update_only = true`
- `variant_isolated_post_update_only = true`
- `baseline_prestep_row_count = 0`
- `variant_prestep_row_count = 0`

At the isolated simulator-owned consumer output `after_sum_ekin` in the early window (`step <= 200`):

- `mean_abs_delta_kinetic_energy_kj = 0.7946533203125`
- `mean_abs_delta_temperature_k = 0.23687744140625`

against baseline early means:

- `mean_kinetic_energy_kj = 1072.8181396484374`
- `mean_temperature_k = 319.77784729003906`

So the isolated slice deltas are about `0.064%` of the baseline early kinetic-energy and temperature means. That is bounded carry-through, not runaway-relevant amplification. Basis: `tests/reference_results/tp1_8j_slice_trace/slice_trace_summary.json`.

## Strongest Unresolved Uncertainty
TP1.8j still does not show where the surviving PME-vs-Ewald difference becomes operationally decisive later. Both authoritative reruns still hit runaway onset at `0.2 ps`, and the isolated slice does not expose a localized fault site.

## Interpretation Boundary
What TP1.8j supports:

- TP1.8i’s multi-callsite `compute_globals` contamination is removed
- the isolated post-update `compute_globals -> calc_ke_part -> sum_ekin` slice is explicitly traced
- the PME-vs-Ewald difference survives into simulator-owned kinetic-energy and temperature terms there
- that difference remains small there

What TP1.8j does not support:

- PME dominance
- Ewald/direct dominance
- a defect localized to the isolated post-update `compute_globals` slice
- production patch readiness

## Files Changed
- `src/gromacs/mdlib/md_support.h`
- `src/gromacs/mdlib/md_support.cpp`
- `src/gromacs/mdrun/md.cpp`
- `tools/run_tp1_8j_slice_trace/run_slice_trace.py`
- `tools/run_tp1_8j_slice_trace/README.md`
- `docs/validation_report_tp1_8j.md`
- `docs/tp1_8j_post_update_compute_globals_slice.md`
- `tests/reference_results/tp1_8j_slice_trace/`

## Commands Run
- `git status --short`
- multiple `sed -n ...` and `rg -n ...` inspections over TP1.8h/TP1.8i artifacts and `src/gromacs/mdrun/md.cpp`, `src/gromacs/mdlib/md_support.cpp`, `src/gromacs/mdlib/md_support.h`, `src/gromacs/mdlib/tgroup.cpp`
- `python3 -m py_compile tools/run_tp1_8j_slice_trace/run_slice_trace.py`
- `cmake --build build --target gmx -j4`
- `build/bin/gmx --version | head -n 1`
- `python3 tools/run_tp1_8j_slice_trace/run_slice_trace.py`
- exact `gmx grompp`, `gmx mdrun`, and `gmx energy` invocations are preserved in `tests/reference_results/tp1_8j_slice_trace/raw_commands.txt`

## Fixtures Executed
- authoritative `dense_salt_polymer` under `safe_pme_shift_ref`
- authoritative `dense_salt_polymer` under `safe_ewald_shift`

## Exact Next Step Recommendation
Stop at unresolved unless there is a concrete, active consumer after the isolated post-update `compute_globals` slice that can be shown to transform the bounded PME-vs-Ewald difference. Do not start production patching from TP1.8j alone.
