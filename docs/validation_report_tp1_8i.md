# TP1.8i Validation Report

## Verdict
PASS

## Scope
TP1.8i traced the next active kinetic/temperature consumer after `Update::update_coords` under the fixed authoritative safe baseline from TP1.8h. It stayed inside `integrator = md`, `pcoupl = no`, `tcoupl = no`, and did not return to inactive pressure-control branches.

## Strongest Confirmed Finding
The next active consumer after `Update::update_coords` is the simulator-owned kinetic/temperature path inside `compute_globals`, specifically `calc_ke_part` followed by `sum_ekin`. The optional `global_stat` rank-reduction path is inactive in these authoritative reruns because the runs are effectively single-rank.

The baseline reuse and fairness checks both passed:

- `baseline_reuses_tp18h_safe_settings = true`
- `short_range_fields_fixed_across_runs = true`
- shared stages:
  - `before_calc_ke_part`
  - `after_calc_ke_part`
  - `after_gstat_block`
  - `after_sum_ekin`
- all shared stages had `step_mismatch_count = 0`
- all shared stages had `gstat_reduction_executed_mismatch_count = 0`

At the actual kinetic/temperature consumer output (`after_sum_ekin`), the early-window PME-vs-Ewald deltas stayed very small:

- `mean_abs_delta_v_l2_in = 0.0025264163421641803`
- `mean_abs_delta_ekind_ekin_trace = 0.6631673177083334`
- `mean_abs_delta_kinetic_energy_kj = 0.6631673177083334`
- `mean_abs_delta_temperature_k = 0.1976776123046875`

against baseline early means:

- `mean_v_l2_in = 12.363061687213394`
- `mean_ekind_ekin_trace = 1117.9584935506184`
- `mean_kinetic_energy_kj = 1117.958475748698`
- `mean_temperature_k = 333.2329584757487`

That is bounded carry-through, not update-to-consumer amplification. Basis: `tests/reference_results/tp1_8i_consumer_trace/consumer_trace_summary.json`.

## Strongest Unresolved Uncertainty
TP1.8i does not show where the surviving split becomes runaway-relevant later. Both authoritative reruns still hit runaway onset at `0.2 ps`, and TP1.8i does not isolate a localized source-level defect at the kinetic/temperature consumer boundary.

## Interpretation Boundary
What TP1.8i supports:

- the active consumer after `Update::update_coords` is `compute_globals -> calc_ke_part -> sum_ekin`
- that consumer is explicitly traced
- the incoming PME-vs-Ewald difference survives into simulator-owned kinetic/temperature terms
- the difference remains small there and does not show anomalous amplification

What TP1.8i does not support:

- PME dominance
- Ewald/direct dominance
- a defect localized to `compute_globals`, `calc_ke_part`, or `sum_ekin`
- production patch readiness

## Files Changed
- `src/gromacs/mdlib/md_support.cpp`
- `tools/run_tp1_8i_consumer_trace/run_consumer_trace.py`
- `tools/run_tp1_8i_consumer_trace/README.md`
- `docs/validation_report_tp1_8i.md`
- `docs/tp1_8i_kinetic_temperature_consumer_trace.md`
- `tests/reference_results/tp1_8i_consumer_trace/`

## Commands Run
- `git status --short`
- multiple `sed -n ...` and `rg -n ...` inspections over TP1.8d/TP1.8e/TP1.8f/TP1.8g/TP1.8h artifacts and `src/gromacs/mdlib/md_support.cpp`, `src/gromacs/mdlib/tgroup.cpp`, `src/gromacs/mdlib/stat.cpp`, `src/gromacs/mdrun/md.cpp`
- `python3 -m py_compile tools/run_tp1_8i_consumer_trace/run_consumer_trace.py`
- `cmake --build build --target gmx -j4`
- `build/bin/gmx --version | head -n 1`
- `python3 tools/run_tp1_8i_consumer_trace/run_consumer_trace.py`
- `pgrep -af 'gmx mdrun|gmx energy|run_consumer_trace.py|run_tp1_8i_consumer_trace'`
- exact `gmx grompp`, `gmx mdrun`, and `gmx energy` invocations are preserved in `tests/reference_results/tp1_8i_consumer_trace/raw_commands.txt`

## Fixtures Executed
- authoritative `dense_salt_polymer` under `safe_pme_shift_ref`
- authoritative `dense_salt_polymer` under `safe_ewald_shift`

## Exact Next Step Recommendation
If tracing continues at all, target only the next active kinetic/temperature consumer after `compute_globals`. If there is no concrete active consumer worth tracing, stop at unresolved rather than inventing a root cause. Do not start production patching from TP1.8i alone.
