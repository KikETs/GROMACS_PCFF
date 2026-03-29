# TP1.8j — Pure Post-Update compute_globals Slice Isolation

## Goal
Isolate only the post-update `compute_globals` invocation under the same authoritative safe baseline from TP1.8i and determine whether the surviving PME-vs-Ewald difference stays bounded there or becomes amplified in a way that is plausibly runaway-relevant.

## What TP1.8i Already Showed
TP1.8i traced the next active kinetic/temperature consumer path:

- `compute_globals`
- `calc_ke_part`
- `sum_ekin`

and showed bounded carry-through into simulator-owned `KineticEnergy` and `Temperature`.

But TP1.8i mixed multiple `compute_globals(...)` callsites together:

- pre-step initialization calls produced `step = -1` rows
- the post-update callsite was not isolated as a single slice

That weakened the statement “next active consumer after `Update::update_coords`,” because the trace contained more than the post-update boundary.

## Callsite Localization
TP1.8j isolates the exact callsite in:

- `src/gromacs/mdrun/md.cpp` `LegacySimulator::do_md`
  - the main-loop `compute_globals(...)` invocation immediately after `update_coords`

Excluded:

- the earlier pre-step `compute_globals(...)` initialization calls in the same function

The path map is recorded in `tests/reference_results/tp1_8j_slice_trace/callsite_path_map.json`.

## Instrumentation Strategy
TP1.8j adds the smallest trace-only scope marker needed to isolate a single callsite:

- `src/gromacs/mdlib/md_support.h`
  - `ScopedTp18jPostUpdateComputeGlobalsTrace`
- `src/gromacs/mdrun/md.cpp`
  - wraps only the post-update `compute_globals(...)` callsite
- `src/gromacs/mdlib/md_support.cpp`
  - writes TP1.8j trace rows only while that scope is active

Recorded stages:

- `before_calc_ke_part`
- `after_calc_ke_part`
- `after_gstat_block`
- `after_sum_ekin`

Recorded observables:

- callsite marker:
  - `callsite = post_update_compute_globals`
- simulator-owned kinetic/temperature outputs:
  - `ekind_ekin_trace`
  - `kinetic_energy_kj`
  - `temperature_k`
- kinetic-state inputs and intermediate aggregates:
  - `v_l2_in`
  - `tcstat_ekinh_trace_sum`
  - `tcstat_ekinh_old_trace_sum`
  - `tcstat_ekinf_trace_sum`
- auxiliary integration-state terms:
  - `total_energy_term_kj`
  - `conserved_energy_term_kj`

Exact per-atom decomposition was not added. TP1.8j uses aggregate simulator-owned quantities at the isolated slice only.

## Controlled Reruns
The two authoritative reruns were:

- `safe_pme_shift_ref`
- `safe_ewald_shift`

Fairness checks:

- `baseline_reuses_tp18i_safe_settings = true`
- `short_range_fields_fixed_across_runs = true`
- `nstlist = 10`
- `rlist = 0.911`
- `verlet-buffer-tolerance = -1`
- `vdw-type = Cut-off`
- `pcoupl = no`
- `tcoupl = no`

Runtime-distinct Coulomb axis:

- baseline raw log contains `Solve PME`
- variant raw log contains `Will do ordinary reciprocal space Ewald sum.`

Both reruns remained unstable:

- baseline runaway onset: `0.2 ps`
- variant runaway onset: `0.2 ps`

## Isolation Check
The isolation itself succeeded.

From `tests/reference_results/tp1_8j_slice_trace/slice_trace_summary.json`:

- `baseline_callsite_names = ["post_update_compute_globals"]`
- `variant_callsite_names = ["post_update_compute_globals"]`
- `baseline_prestep_row_count = 0`
- `variant_prestep_row_count = 0`
- `baseline row_count = 1604 = 401 * 4`
- `variant row_count = 1604 = 401 * 4`

So TP1.8j no longer mixes the pre-step `compute_globals` calls that produced `step = -1` rows in TP1.8i.

## Isolated-Slice Results
At `after_calc_ke_part` in the early window (`step <= 200`):

- `mean_abs_delta_kinetic_energy_kj = 0.532373046875`
- `mean_abs_delta_temperature_k = 0.158697509765625`

At the actual simulator-owned output `after_sum_ekin` in the same early window:

- `mean_abs_delta_kinetic_energy_kj = 0.7946533203125`
- `mean_abs_delta_temperature_k = 0.23687744140625`

Baseline early means at `after_sum_ekin`:

- `mean_kinetic_energy_kj = 1072.8181396484374`
- `mean_temperature_k = 319.77784729003906`

Derived TP1.8j ratios:

- `after_sum_ekin_kinetic_energy_delta_ratio_early = 0.0006405069617817236`
- `after_sum_ekin_temperature_delta_ratio_early = 0.0006405419622076281`
- `after_sum_vs_after_calc_kinetic_delta_ratio_early = 1.4926625699348803`

Interpretation:

- there is some internal reshaping inside the isolated slice
- but the output delta remains about `0.064%` of the baseline early kinetic-energy and temperature means
- that is still bounded carry-through, not a sharp runaway-relevant amplification step

The auxiliary global terms do not provide a stronger localization here:

- `conserved_energy_term_kj` stays `0`
- `total_energy_term_kj` differs between runs, but TP1.8j does not treat it as the primary mechanism signal because the isolated slice is focused on simulator-owned kinetic/temperature accumulation

## Conservative Interpretation
What TP1.8j narrows:

- TP1.8i’s multi-callsite limitation is removed
- the isolated post-update `compute_globals` slice is real and auditable
- the PME-vs-Ewald difference survives into simulator-owned kinetic-energy and temperature outputs there
- the difference remains bounded there

What TP1.8j still does not isolate:

- a unique runaway-relevant transformation at the isolated slice
- PME dominance
- Ewald/direct dominance
- a patch-ready source-level defect

Conservative classification:

- `bounded carry-through only` at the isolated slice
- overall `still_unresolved`

## Patch Readiness
Source patching remains unjustified.

Basis:

- `tests/reference_results/tp1_8j_slice_trace/tp1_8j_recommendation.json`
- isolated callsite tracing passed
- no post-update-slice amplification stronger than bounded carry-through
- both authoritative reruns still go runaway at `0.2 ps`

## Exact Next Step
Stop at unresolved unless there is a concrete, active consumer after the isolated post-update `compute_globals` slice that can be shown to transform the bounded PME-vs-Ewald difference. Otherwise TP1.8j is already at sharply diminishing returns.
