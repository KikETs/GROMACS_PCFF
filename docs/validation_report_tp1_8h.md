# TP1.8h Validation Report

## Verdict
PASS

## Scope
TP1.8h traced the active integration-state update path under the fixed authoritative safe baseline from TP1.8g. It stayed within `integrator = md`, `pcoupl = no`, `tcoupl = no`, and did not revisit inactive pressure-control branches.

## Strongest Confirmed Finding
The authoritative PME baseline and narrowed Ewald variant both reused the same safe short-range baseline and both took the same active update path: `integrator = md`, `update_part = position`, `helper_path = md_leapfrog_simple_simd`, `pcoupl_is_no = 1`, `tcoupl_is_no = 1`, `do_temp_couple = 0`, `do_nose_hoover = 0`, `have_constraints = 0`. Basis: `tests/reference_results/tp1_8h_update_trace/update_trace_baseline.csv`, `tests/reference_results/tp1_8h_update_trace/update_trace_variant.csv`, `tests/reference_results/tp1_8h_update_trace/update_trace_summary.json`.

Under that fixed path, the surviving PME-vs-Ewald force-side difference does carry into update-state quantities, but without evidence of anomalous update-stage amplification. Early-window mean absolute cross-run deltas were:

- `force_l2_in = 16.830066013439065`
- `delta_v_l2 = 0.0005572939693862765`
- `delta_xprime_from_x_l2 = 2.067596288322491e-06`
- `kinetic_proxy_after = 0.5817528137743532`
- `delta_kinetic_proxy = 0.11146624350452901`

while both runs still hit runaway onset at `0.2 ps`. Basis: `tests/reference_results/tp1_8h_update_trace/update_trace_summary.json`.

## Strongest Unresolved Uncertainty
TP1.8h shows proportional carry-through into the active update boundary, but it still does not localize where that preserved difference becomes runaway-relevant later. No source-level root cause is isolated here.

## Interpretation Boundary
What TP1.8h supports:

- the surviving force-side split is present on entry to the active update path
- the active path is the same in both runs
- the split is reflected in velocity, xprime, and kinetic-proxy increments
- no new update-local amplification or path-switch evidence appears here

What TP1.8h does not support:

- PME dominance
- Ewald/direct dominance
- a source-level defect in `update.cpp`
- production patch readiness

## Files Changed
- `src/gromacs/mdlib/update.cpp`
- `tools/run_tp1_8h_update_trace/run_update_trace.py`
- `tools/run_tp1_8h_update_trace/README.md`
- `docs/validation_report_tp1_8h.md`
- `docs/tp1_8h_integration_state_trace.md`
- `tests/reference_results/tp1_8h_update_trace/`

## Commands Run
- `git status --short`
- multiple `sed -n ...` and `rg -n ...` inspections over TP1.8d/TP1.8e/TP1.8f/TP1.8g artifacts and `src/gromacs/mdlib/update.cpp`, `src/gromacs/mdrun/md.cpp`
- `python3 -m py_compile tools/run_tp1_8h_update_trace/run_update_trace.py`
- `cmake --build build --target gmx -j4`
- `build/bin/gmx --version | head -n 1`
- `python3 tools/run_tp1_8h_update_trace/run_update_trace.py`
- `pgrep -af 'gmx mdrun|run_update_trace.py|run_tp1_8h_update_trace'`

## Fixtures Executed
- authoritative `dense_salt_polymer` under `safe_pme_shift_ref`
- authoritative `dense_salt_polymer` under `safe_ewald_shift`

## Exact Next Step Recommendation
If one final localization step is still required, trace the next active kinetic/temperature consumer after `Update::update_coords`. Do not return to inactive pressure-control branches, and do not start production patching from TP1.8h alone.
