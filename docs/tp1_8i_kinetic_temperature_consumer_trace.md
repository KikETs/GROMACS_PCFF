# TP1.8i — Active Kinetic/Temperature Consumer Trace

## Goal
Trace the next active kinetic/temperature-related consumer after `Update::update_coords` under the same authoritative safe baseline from TP1.8h and determine whether the surviving PME-vs-Ewald difference becomes operationally relevant there.

## Constraining Prior Evidence
- TP1.8d and TP1.8e showed that a PME-vs-Ewald force-side difference survives through post-selection and later handoff boundaries.
- TP1.8f showed `compute_globals` preserves and linearly re-expresses incoming virial/pressure differences rather than creating a new local pressure anomaly.
- TP1.8g showed the immediate post-`compute_globals` pressure-control consumers are inactive under `pcoupl = no`.
- TP1.8h showed the active update path is fixed and the surviving force-side difference carries into `v`, `xprime`, and a kinetic proxy without update-local amplification.

That left one active path worth tracing directly: the simulator-owned kinetic/temperature accumulation inside `compute_globals`.

## Active Consumer Path
TP1.8i localized the next active consumer after `Update::update_coords` to:

- `src/gromacs/mdrun/md.cpp` `LegacySimulator::do_md`
- `src/gromacs/mdlib/md_support.cpp` `compute_globals`
- `src/gromacs/mdlib/md_support.cpp` `calc_ke_part / calc_ke_part_normal`
- `src/gromacs/mdlib/tgroup.cpp` `sum_ekin`

Inactive for this setup:

- `src/gromacs/mdlib/stat.cpp` `global_stat` rank-reduction path
  - reason: the TP1.8i reruns are effectively single-rank, and `gstat_reduction_executed_any = false`

Basis: `tests/reference_results/tp1_8i_consumer_trace/consumer_path_map.json`, `tests/reference_results/tp1_8i_consumer_trace/consumer_trace_summary.json`.

## Instrumentation
Trace-only instrumentation was added in `src/gromacs/mdlib/md_support.cpp` under `GMX_TP18I_TRACE_FILE`.

Recorded stages:

- `before_calc_ke_part`
- `after_calc_ke_part`
- `after_gstat_block`
- `after_sum_ekin`

Recorded observables:

- incoming velocity summary:
  - `v_l2_in`
  - `v_max_abs_in`
- kinetic tensor aggregates:
  - `tcstat_ekinh_trace_sum`
  - `tcstat_ekinh_old_trace_sum`
  - `tcstat_ekinf_trace_sum`
  - corresponding `*_l2_sum`
- simulator-owned aggregate outputs:
  - `ekind_ekin_trace`
  - `ekind_ekin_l2`
  - `kinetic_energy_kj`
  - `temperature_k`
- path-state flags:
  - `compute_ekin`
  - `b_temperature`
  - `have_leapfrog`
  - `have_ekinh_old`
  - `mpi_parallel`
  - `gstat_reduction_executed`

Exact per-atom decomposition was not added. TP1.8i uses aggregate source-level observables only.

## Controlled Reruns
The two authoritative reruns were:

- `safe_pme_shift_ref`
- `safe_ewald_shift`

Fairness checks:

- TP1.8h baseline reuse: `true`
- short-range fields fixed across runs: `true`
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
- baseline `max_temperature_k = 779.376526`
- variant `max_temperature_k = 788.437805`

Basis: `tests/reference_results/tp1_8i_consumer_trace/consumer_trace_summary.json`, `tests/reference_results/tp1_8i_consumer_trace/raw_safe_pme_shift_ref_mdout.mdp`, `tests/reference_results/tp1_8i_consumer_trace/raw_safe_ewald_shift_mdout.mdp`.

## Strongest Consumer-Level Facts
The active consumer path did not branch between runs:

- all shared stages matched exactly
- all shared stages had `step_mismatch_count = 0`
- all shared stages had `gstat_reduction_executed_mismatch_count = 0`
- `mpi_parallel_any = false`
- `gstat_reduction_executed_any = false`

So TP1.8i is not explaining the surviving split by a hidden MPI reduction or a consumer-path switch.

## Comparative Results
At `after_calc_ke_part` in the early window (`step <= 200`):

- `mean_abs_delta_v_l2_in = 0.0025264163421641803`
- `mean_abs_delta_tcstat_ekinh_trace_sum = 0.6557871500651041`
- `mean_abs_delta_kinetic_energy_kj = 0.444091796875`
- `mean_abs_delta_temperature_k = 0.13238016764322916`

with baseline early means:

- `mean_v_l2_in = 12.363061687213394`
- `mean_tcstat_ekinh_trace_sum = 1202.8804626464844`
- `mean_kinetic_energy_kj = 894.0146687825521`
- `mean_temperature_k = 266.48140716552734`

At the actual consumer output `after_sum_ekin` in the same early window:

- `mean_abs_delta_ekind_ekin_trace = 0.6631673177083334`
- `mean_abs_delta_kinetic_energy_kj = 0.6631673177083334`
- `mean_abs_delta_temperature_k = 0.1976776123046875`

with baseline early means:

- `mean_ekind_ekin_trace = 1117.9584935506184`
- `mean_kinetic_energy_kj = 1117.958475748698`
- `mean_temperature_k = 333.2329584757487`

So the PME-vs-Ewald difference is present in simulator-owned kinetic/temperature state, but it stays tiny there. The early `after_sum_ekin` temperature delta is about `0.06%` of the baseline early mean temperature, and the early kinetic-energy delta is about `0.06%` of the baseline early mean kinetic energy.

TP1.8h context:

- `early_mean_abs_delta_kinetic_proxy_after = 0.5817528137743532`
- `early_mean_abs_delta_delta_kinetic_proxy = 0.11146624350452901`

TP1.8i therefore does not reveal a new amplification step beyond what TP1.8h already showed as bounded carry-through.

## Interpretation
What TP1.8i narrows:

- the next active consumer after `Update::update_coords` is the `compute_globals` kinetic/temperature path
- the optional rank-reduction path is inactive here
- the PME-vs-Ewald split reaches simulator-owned `Temperature` and `KineticEnergy`
- the split remains bounded there rather than becoming sharply amplified

What TP1.8i does not isolate:

- a specific kinetic/temperature consumer defect
- a unique runaway-relevant transformation at this boundary
- PME or Ewald dominance

Conservative classification:

- `still_unresolved`
- stronger support for `bounded carry-through only` at this consumer boundary

## Patch Readiness
Source patching remains unjustified.

Basis:

- `tests/reference_results/tp1_8i_consumer_trace/tp1_8i_recommendation.json`
- no active consumer-path switch
- no active MPI reduction path
- no anomalous kinetic/temperature amplification
- no stabilization improvement in same-build authoritative reruns

## Exact Next Step
If tracing continues at all, target only the next active kinetic/temperature consumer after `compute_globals`. If there is no concrete active consumer worth tracing, stop at unresolved and do not start production patching.
