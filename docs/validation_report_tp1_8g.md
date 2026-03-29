# TP1.8g Validation Report

## Verdict
PASS

## Scope
TP1.8g traces the immediate consumers after `compute_globals` on the authoritative `dense_salt_polymer` safe baseline. It does not reopen `compute_globals` internals, does not touch r-RESPA, and does not patch production logic.

## Goal
- Reuse the TP1.8f authoritative safe baseline and narrowed Ewald variant on the current build.
- Trace the first post-`compute_globals` consumers of `pres`, `total_vir`, and scalar pressure.
- Decide whether the surviving PME-vs-Ewald split becomes operational at this boundary or remains reporting-level only.

## Files Changed
- [md.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdrun/md.cpp)
- [run_consumer_trace.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_8g_consumer_trace/run_consumer_trace.py)
- [README.md](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_8g_consumer_trace/README.md)
- [consumer_path_map.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/consumer_path_map.json)
- [run_matrix.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/run_matrix.json)
- [consumer_trace_baseline.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/consumer_trace_baseline.csv)
- [consumer_trace_variant.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/consumer_trace_variant.csv)
- [consumer_trace_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/consumer_trace_summary.json)
- [tp1_8g_recommendation.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/tp1_8g_recommendation.json)

## Commands Run
- `git status --short`
- multiple `sed -n ...` and `rg -n ...` inspections over TP1.8d/TP1.8e/TP1.8f artifacts and `src/gromacs/mdrun/md.cpp`, `src/gromacs/mdlib/coupling.cpp`, `src/gromacs/mdlib/energyoutput.cpp`, `src/gromacs/mdlib/md_support.cpp`
- `python3 -m py_compile tools/run_tp1_8g_consumer_trace/run_consumer_trace.py`
- `cmake --build build --target gmx -j4`
- `python3 tools/run_tp1_8g_consumer_trace/run_consumer_trace.py`
- exact `gmx grompp`, `gmx mdrun`, `gmx energy` invocations are preserved in [raw_commands.txt](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/raw_commands.txt)

## Fixtures Executed
- authoritative `dense_salt_polymer` under `safe_pme_shift_ref`
- authoritative `dense_salt_polymer` under `safe_ewald_shift`

## Setup Held Fixed
- `nstlist = 10`
- `rlist = 0.911`
- `verlet-buffer-tolerance = -1`
- `vdw-type = Cut-off`
- `coulomb-modifier = Potential-shift-Verlet`
- `dt = 0.001`
- `nsteps = 20000`
- `pcoupl = no`
- same topology, start structure, seed, ensemble family

The only intended A/B change remains `coulombtype = PME` versus `Ewald`; basis: [run_matrix.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/run_matrix.json), [raw_safe_pme_shift_ref_mdout.mdp](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/raw_safe_pme_shift_ref_mdout.mdp), [raw_safe_ewald_shift_mdout.mdp](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/raw_safe_ewald_shift_mdout.mdp).

## Consumer Path Findings
- `update_pcouple_after_coordinates` is called immediately after `compute_globals`, but under this authoritative setup it is inert because `pcoupl = no`; source basis: [coupling.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdlib/coupling.cpp#L218).
- `state_->pres_prev` handoff is inactive because `StateEntry::PressurePrevious` is not allocated for `pcoupl = no`; source basis: [md_support.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdlib/md_support.cpp#L910), [md.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdrun/md.cpp#L3019).
- The active immediate consumers are reporting/output paths: [md.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdrun/md.cpp#L2822), [md.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdrun/md.cpp#L2884), and [energyoutput.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdlib/energyoutput.cpp#L841).

## Strongest Confirmed Finding
The surviving PME-vs-Ewald split does not become an operational pressure-control or `pres_prev` state handoff at the immediate post-`compute_globals` boundary. In both runs:
- `pressure_coupling_is_no_all = true`
- `pressure_coupling_consumer_active_any = false`
- `has_pressure_previous_any = false`
- `pressure_previous_copy_executed_any = false`

At the same time, the reporting consumers are active:
- `after_energy_add.energy_add_called_any = true`
- `after_energy_print.energy_print_called_any = true`

Basis: [consumer_trace_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/consumer_trace_summary.json).

## Consumer-Boundary Comparison
At all four traced stages, the early-window A/B deltas are the same:
- `mean_abs_delta_total_vir_trace_early = 7.081266944088153`
- `mean_abs_delta_pres_trace_early = 21.135772268570477`
- `mean_abs_delta_pressure_scalar_early = 7.0452753323227615`

This is the relevant negative result. The split survives through the boundary unchanged, but not as a newly activated control-state consumer. It is forwarded only into the active reporting/output consumers at this level; basis: [consumer_trace_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/consumer_trace_summary.json), [consumer_trace_baseline.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/consumer_trace_baseline.csv), [consumer_trace_variant.csv](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/consumer_trace_variant.csv).

## Runaway Status
The authoritative instability still persists:
- baseline onset `0.2 ps`, `max_temperature_k = 753.839478`
- Ewald variant onset `0.2 ps`, `max_temperature_k = 788.437805`

Basis: [consumer_trace_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_8g_consumer_trace/consumer_trace_summary.json).

## Strongest Unresolved Uncertainty
TP1.8g narrows the immediate post-`compute_globals` consumer story, but it still does not isolate which later shared downstream path makes the preserved split runaway-relevant. The immediate control-state consumers are inactive here, and the active consumers are reporting-level only. That weakens a localized fault-site story at this boundary, but it does not identify the real later operational site.

## Recommendation
- `source_patching_now_justified = false`
- `plain_safe_baseline_acceptable_for_later_non_rrespa_validation = PARTIAL`
- consumer-boundary classification: `aggregate_or_reporting_level_only`
- overall classification: `still_unresolved`

## Exact Next Step Recommendation
Stop at unresolved unless there is a concrete reason to trace beyond the immediate post-`compute_globals` output consumers. Under `pcoupl = no`, this boundary does not activate pressure-control state consumption, so production patching is still not justified.
