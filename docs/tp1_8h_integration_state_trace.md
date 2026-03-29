# TP1.8h — Integration-State Update Trace

## Goal
Trace the active post-force integration update path under the same authoritative safe baseline from TP1.8g and test whether the surviving PME-vs-Ewald force-side difference becomes operationally relevant through velocity, xprime, or kinetic-energy-relevant updates.

## Constraining Prior Evidence
- TP1.8d/TP1.8e showed that a force-side PME-vs-Ewald difference survives through immediate post-selection and later handoff boundaries.
- TP1.8f showed `compute_globals` preserves and linearly re-expresses incoming virial/pressure differences rather than creating a new local anomaly.
- TP1.8g showed the immediate post-`compute_globals` pressure-control consumers are inactive under `pcoupl = no`; only reporting/output consumers are active there.

That left one active path worth tracing: the actual update path that consumes force into `v` and `xprime`.

## Active Update Path
TP1.8h localized the active update path to:

- `src/gromacs/mdrun/md.cpp` `LegacySimulator::do_md`
- `src/gromacs/mdlib/update.cpp` `Update::Impl::update_coords`
- `src/gromacs/mdlib/update.cpp` `do_update_md`
- `src/gromacs/mdlib/update.cpp` `updateMDLeapfrogSimple / updateMDLeapfrogSimpleSimd`

Inactive for this setup:

- `updateMDLeapfrogGeneral`
- `doUpdateMDDoNotUpdateVelocities`
- velocity-Verlet paths

Basis: `tests/reference_results/tp1_8h_update_trace/update_path_map.json`.

## Instrumentation
Trace-only instrumentation was added at the `Update::Impl::update_coords` boundary under `GMX_TP18H_TRACE_FILE`.

Captured observables:

- incoming `force_l2_in`, `force_max_abs_in`
- `v_l2_before`, `v_l2_after`, `delta_v_l2`
- `xprime_l2_after`, `delta_xprime_from_x_l2`
- mass-weighted `kinetic_proxy_before`, `kinetic_proxy_after`, `delta_kinetic_proxy`
- active path metadata:
  - `integrator`
  - `update_part`
  - `helper_path`
  - `pcoupl_is_no`
  - `tcoupl_is_no`
  - `do_temp_couple`
  - `do_nose_hoover`
  - `have_constraints`

Exact per-atom decomposition was not added. TP1.8h uses aggregate source-level surrogates only.

## Controlled Reruns
The two authoritative reruns were:

- `safe_pme_shift_ref`
- `safe_ewald_shift`

Fairness checks:

- TP1.8g baseline reuse: `true`
- short-range fields fixed across runs: `true`
- `nstlist = 10`
- `rlist = 0.911`
- `verlet-buffer-tolerance = -1`
- `vdw-type = Cut-off`
- `pcoupl = no`
- `tcoupl = no`

Basis: `tests/reference_results/tp1_8h_update_trace/update_trace_summary.json`, `tests/reference_results/tp1_8h_update_trace/raw_safe_pme_shift_ref_mdout.mdp`, `tests/reference_results/tp1_8h_update_trace/raw_safe_ewald_shift_mdout.mdp`.

Runtime-distinct Coulomb axis:

- baseline raw log contains `Solve PME`
- variant raw log contains `Will do ordinary reciprocal space Ewald sum.`

## Strongest Update-Path Facts
The update path did not change between runs.

From all `20001` traced calls in both runs:

- `integrator = md`
- `update_part = position`
- `helper_path = md_leapfrog_simple_simd`
- `using_simd_path = 1`
- `pcoupl_is_no = 1`
- `tcoupl_is_no = 1`
- `do_temp_couple = 0`
- `do_nose_hoover = 0`
- `have_partially_frozen_atoms = 0`
- `have_constraints = 0`

Basis: `tests/reference_results/tp1_8h_update_trace/update_trace_baseline.csv`, `tests/reference_results/tp1_8h_update_trace/update_trace_variant.csv`, `tests/reference_results/tp1_8h_update_trace/update_trace_summary.json`.

## Comparative Results
Both reruns remained unstable:

- baseline runaway onset: `0.2 ps`
- variant runaway onset: `0.2 ps`

Energy/temperature summary:

- baseline `max_temperature_k = 780.858887`
- variant `max_temperature_k = 788.437805`
- baseline `total_energy_range_kj = 5.501952999999048`
- variant `total_energy_range_kj = 5.011718999998266`

Update-boundary differences in the early window (`step <= 200`):

- `mean_abs_delta_force_l2_in = 16.830066013439065`
- `mean_abs_delta_delta_v_l2 = 0.0005572939693862765`
- `mean_abs_delta_delta_xprime_from_x_l2 = 2.067596288322491e-06`
- `mean_abs_delta_kinetic_proxy_after = 0.5817528137743532`
- `mean_abs_delta_delta_kinetic_proxy = 0.11146624350452901`

Context from baseline early-window magnitudes:

- baseline `mean_abs_force_l2_in = 12942.69689311459`
- baseline `mean_abs_delta_v_l2 = 0.8975991587845586`
- baseline `mean_abs_delta_xprime_from_x_l2 = 0.011790836647643892`
- baseline `mean_abs_kinetic_proxy_after = 1116.4450060792014`
- baseline `mean_abs_delta_kinetic_proxy = 8.655720527802227`

So the cross-run difference is visible in update-state quantities, but it stays small at this boundary and does not show a new update-stage amplification pattern.

## Interpretation
What TP1.8h narrows:

- the surviving force-side difference is not being created by a path switch in update
- it does carry into velocity, xprime, and kinetic-proxy updates
- that carry-through looks proportional and bounded at this boundary

What TP1.8h does not isolate:

- a specific temperature/kinetic consumer that makes the difference runaway-relevant
- a unique source-level defect inside `update.cpp`
- PME or Ewald dominance

Conservative classification:

- `still_unresolved`

## Patch Readiness
Source patching remains unjustified.

Basis:

- `tests/reference_results/tp1_8h_update_trace/tp1_8h_recommendation.json`
- no update-path switch
- no update-local anomaly stronger than bounded carry-through
- no stabilization improvement in same-build authoritative reruns

## Exact Next Step
If tracing continues at all, follow the next active kinetic/temperature consumer after `Update::update_coords`. If there is no concrete reason to continue, stop at unresolved rather than inventing a root cause.
