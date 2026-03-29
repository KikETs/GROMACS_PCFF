# TP1.8g Post-compute_globals Consumer Trace

## Phase 1 Summary
- TP1.8c already showed a real PME-vs-Ewald split at `CpuPpLongRangeNonbondeds::calculate`.
- TP1.8d and TP1.8e showed that a force-side difference survives beyond `postProcessForces` and into later handoff state.
- TP1.8f showed that `compute_globals` itself does not create a new pressure anomaly: `m_add` preserves the incoming virial split and `calc_pres` linearly re-expresses it.
- TP1.8g therefore had to trace only the immediate consumers after `compute_globals`, not reopen earlier boundaries.

## Trace Strategy
- keep the authoritative TP1.8f safe baseline fixed
- rerun `safe_pme_shift_ref` and `safe_ewald_shift`
- trace four immediate post-`compute_globals` stages from [md.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdrun/md.cpp):
  - `after_update_pcouple`
  - `after_energy_add`
  - `after_energy_print`
  - `after_pressure_prev_handoff`
- capture only aggregate consumer-boundary observables:
  - `total_vir_trace`
  - `pres_trace`
  - scalar pressure
  - booleans that prove whether a control-state consumer actually acted

## Active Path Map
See [consumer_path_map.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/consumer_path_map.json).

The critical path statuses are:
- `update_pcouple_after_coordinates`: active call site, effectively inert here because `pcoupl = no`
- `copy_mat(pres, state_->pres_prev)`: inactive because `PressurePrevious` is not allocated for this setup
- `EnergyOutput::addDataAtEnergyStep`: active reporting/global accumulation consumer
- `EnergyOutput::printStepToEnergyFile`: active reporting/output consumer

## Runtime Verification
Both runs preserve the same short-range baseline:
- `nstlist = 10`
- `rlist = 0.911`
- `verlet-buffer-tolerance = -1`
- runtime pairlist line: `updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm`

The only intended path change is Coulomb solver selection:
- baseline [raw_safe_pme_shift_ref_md.log](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/raw_safe_pme_shift_ref_md.log) contains `Solve PME`
- variant [raw_safe_ewald_shift_md.log](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/raw_safe_ewald_shift_md.log) contains `Will do ordinary reciprocal space Ewald sum.`

## Consumer-Level Result
The immediate post-`compute_globals` boundary does not localize a new operational fault site.

Facts:
- `after_update_pcouple.pressure_coupling_consumer_active_any = false` in both runs
- `after_pressure_prev_handoff.has_pressure_previous_any = false` in both runs
- `after_pressure_prev_handoff.pressure_previous_copy_executed_any = false` in both runs
- `after_energy_add.energy_add_called_any = true` in both runs
- `after_energy_print.energy_print_called_any = true` in both runs

This means the first live consumers of the split are reporting/output consumers, not pressure-control state consumers; basis: [consumer_trace_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/consumer_trace_summary.json).

## Why The Boundary Is Reporting-Level Only
At `step = 0` the baseline and variant rows differ in `total_vir_trace`, `pres_trace`, and scalar pressure, but the stage booleans show:
- no pressure-coupling action
- no `pres_prev` state write
- yes energy accumulation
- yes energy/log output

Example rows:
- [consumer_trace_baseline.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/consumer_trace_baseline.csv)
- [consumer_trace_variant.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/consumer_trace_variant.csv)

The same early-window deltas persist at every traced stage:
- `total_vir_trace`: `7.081266944088153`
- `pres_trace`: `21.135772268570477`
- scalar pressure: `7.0452753323227615`

That pattern matters. If a post-`compute_globals` operational consumer were transforming the split here, the stage deltas would change. They do not.

## Boundary Honesty
- TP1.8g does not prove a root cause.
- TP1.8g does not prove force-side dominance.
- TP1.8g does not prove virial/pressure-side dominance.
- TP1.8g only supports a narrower claim: at the immediate post-`compute_globals` boundary, the split remains aggregate/reporting-level and does not become an active pressure-control handoff.

## Strongest Confirmed Finding
The immediate post-`compute_globals` control-state consumers are inactive in the authoritative setup, while the reporting/output consumers are active and forward the preserved split.

## Strongest Unresolved Uncertainty
The later shared downstream site that makes the preserved split physically runaway-relevant is still not localized.

## Exact Next Step Recommendation
Stop at unresolved unless there is a concrete reason to trace beyond the immediate post-`compute_globals` output consumers. If tracing continues, it should be justified by a specific later consumer that is active in this authoritative configuration, not by generic suspicion.
