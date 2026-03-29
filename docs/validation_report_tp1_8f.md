# TP1.8f Validation Report

## Scope
- Milestone: `TP1.8f — compute_globals / Pressure-Handoff Trace`
- Status: `PASS`
- Boundary kept: no r-RESPA work, no transport work, no production-fix claim

## Before Execution
- Goal: reuse the TP1.8e authoritative safe baseline, trace `compute_globals` and the immediate `calc_pres` handoff, and compare the PME baseline against the narrowed Ewald variant without changing the short-range baseline.
- Files/functions inspected: `src/gromacs/mdlib/md_support.cpp`, `src/gromacs/mdlib/coupling.cpp`, plus TP1.8c / TP1.8d / TP1.8e artifacts.
- Trace strategy: add one trace-only CSV row per `compute_globals` call when `GMX_TP18F_TRACE_FILE` is set.
- Captured observables: incoming `force_vir` / `shake_vir`, `total_vir` before and after `m_add`, `pres` before and after `calc_pres`, `pressure_scalar`, `pressure_formula_residual`, and aggregate energy / temperature terms.
- Fairness rule: keep topology, start structure, `nstlist = 10`, `rlist = 0.911`, `verlet-buffer-tolerance = -1`, `vdw-type = Cut-off`, timestep, seed, and ensemble family fixed.

## Executed Runs
- Baseline: `safe_pme_shift_ref`
- Narrowed Coulomb variant: `safe_ewald_shift`

Raw outputs and machine-readable artifacts are under `tests/reference_results/tp1_8f_compute_globals_trace/`.

## Strongest Confirmed Finding
- `compute_globals` does not create a new virial split before pressure is formed.
- Early-window comparison:
  - mean absolute delta in `force_vir_trace_in`: `3.163600376674107`
  - mean absolute delta in `total_vir_trace_after`: `3.163600376674107`
- The equality is expected here because `shake_vir` stays zero and `m_add(force_vir, shake_vir, total_vir)` simply carries the incoming virial difference forward.

## Strongest Pressure-Handoff Finding
- The pressure handoff follows the `calc_pres` formula to near floating-point noise rather than introducing a new anomaly.
- Residual checks:
  - baseline max abs `pressure_formula_residual_full`: `0.0005656929309841314`
  - variant max abs `pressure_formula_residual_full`: `0.0004889883524262353`
  - baseline max abs `pressure_scalar_residual_full`: `0.000244140625`
  - variant max abs `pressure_scalar_residual_full`: `0.000244140625`
- Example at the first pressure-producing call (`step = 0`):
  - delta in `force_vir_trace_in`: `+7.538909912109375`
  - delta in `pres_trace_after`: `-23.51504135131836`
  - constant `pressure_fac`: `3.1189687354611904`
- That is consistent with linear re-expression of the incoming virial difference, not a new compute_globals-local amplification branch.

## Strongest Negative Result
- The authoritative instability still persists in both runs:
  - baseline `RUNAWAY`, onset `0.2 ps`
  - Ewald variant `RUNAWAY`, onset `0.2 ps`
- TP1.8f therefore does not justify a source-level defect claim.

## Conservative Interpretation
- TP1.8f supports:
  - the PME-versus-Ewald difference still enters `compute_globals` on the virial side
  - `compute_globals` carries that difference into `total_vir`
  - `calc_pres` then re-expresses it linearly into `pres` and scalar pressure
- TP1.8f does **not** support:
  - a compute_globals-specific pressure bug
  - pressure/virial dominance as a proven runaway root cause
  - a single proven force-side defect

## Strongest Unresolved Uncertainty
- The later pressure observables still differ strongly over the full window, but TP1.8f only proves that `compute_globals` re-expresses the incoming virial split consistently. It does not yet show whether the runaway relevance lies in the incoming force/virial state, a later pressure consumer, or a broader mixed electrostatics path.

## Exact Next Step Recommendation
- Keep the authoritative safe baseline fixed and, only if another isolation step is required, trace the immediate consumers after `compute_globals` that use `pres`, `total_vir`, or the derived pressure term. Do not patch production logic yet.

## Reporting Footer
- Files changed: `src/gromacs/mdlib/md_support.cpp`, `tools/run_tp1_8f_compute_globals_trace/run_compute_globals_trace.py`, `tools/run_tp1_8f_compute_globals_trace/README.md`, `docs/validation_report_tp1_8f.md`, `docs/tp1_8f_compute_globals_trace.md`, artifacts under `tests/reference_results/tp1_8f_compute_globals_trace/`
- Commands run: `cmake --build build --target gmx -j4`; `python3 -m py_compile tools/run_tp1_8f_compute_globals_trace/run_compute_globals_trace.py`; `python3 tools/run_tp1_8f_compute_globals_trace/run_compute_globals_trace.py`; supporting `sed` / `rg` / `python3 - <<'PY' ...` inspections over TP1.8c/TP1.8d/TP1.8e artifacts and TP1.8f trace outputs
- Systems executed: `dense_salt_polymer` under `safe_pme_shift_ref`, `safe_ewald_shift`
- Strongest confirmed finding: `compute_globals` preserves the incoming virial split and `calc_pres` re-expresses it linearly, rather than introducing a new pressure-handoff anomaly
- Strongest unresolved uncertainty: whether the surviving runaway relevance lies upstream of `compute_globals`, in later pressure consumers, or in a broader mixed electrostatics path
- Exact next step recommendation: keep the baseline fixed and trace immediate post-`compute_globals` pressure consumers only if more isolation is required
- Verdict: `PASS`
