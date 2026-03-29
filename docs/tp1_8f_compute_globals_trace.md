# TP1.8f compute_globals / Pressure-Handoff Trace

## Why TP1.8f Exists
- `TP1.8c` showed a large PME-versus-Ewald split at `CpuPpLongRangeNonbondeds::calculate`.
- `TP1.8d` and `TP1.8e` showed that a force-side difference survives through `postProcessForces` and into the later integration-relevant handoff.
- That still left one specific ambiguity: does `compute_globals` or the immediate pressure handoff turn that incoming difference into a new pressure-side anomaly, or is it mostly just re-expression?

## What Was Already Known
- `TP1.8c`
  - PME and Ewald split strongly at the long-range producer boundary.
  - runaway onset stayed `0.2 ps`.
- `TP1.8d`
  - the producer-boundary reciprocal split was heavily transformed before final potential bookkeeping.
  - a force-side difference survived through `postProcessForces`.
- `TP1.8e`
  - the force-side difference survived unchanged to the `update_coords` handoff.
  - the later `compute_globals` caller boundary still showed a large pressure/virial aggregate split.
- `TP1.4` LJ-PME path remains inactive here because the authoritative setup still uses `vdw-type = Cut-off`.

## What TP1.8f Traces
### Upstream reference
- `src/gromacs/mdrun/md.cpp`
  - `LegacySimulator::do_md`
  - caller-side entry into `compute_globals` was already traced in TP1.8e

### Internal pressure handoff
- `src/gromacs/mdlib/md_support.cpp`
  - `compute_globals`
  - traced around:
    - incoming `force_vir`
    - incoming `shake_vir`
    - `total_vir` before `m_add`
    - `total_vir` after `m_add`
    - `pres` before `calc_pres`
    - `pres` after `calc_pres`
    - scalar pressure term written to `enerd`
- `src/gromacs/mdlib/coupling.cpp`
  - `calc_pres`
  - used as the formula reference for interpreting the traced output

### Non-active side path
- `src/gromacs/mdlib/stat.cpp`
  - `global_stat`
  - not relevant for explaining the difference in these TP1.8f runs because the reruns are single-rank and do not use MPI reduction here

## Instrumentation
- Trace-only instrumentation was added in `src/gromacs/mdlib/md_support.cpp`.
- It writes one CSV row per `compute_globals` call when `GMX_TP18F_TRACE_FILE` is set.
- Captured fields:
  - `ekin_trace`, `ekin_l2`
  - `force_vir_trace_in`, `force_vir_l2_in`
  - `shake_vir_trace_in`, `shake_vir_l2_in`
  - `total_vir_trace_before`, `total_vir_l2_before`
  - `pres_trace_before`, `pres_l2_before`
  - `total_vir_trace_after`, `total_vir_l2_after`
  - `pressure_fac`
  - `pres_trace_after`, `pres_l2_after`
  - `pressure_scalar`
  - `pressure_formula_residual`
  - `pressure_scalar_residual`
  - aggregate energy / temperature terms

제한된 정보입니다. This trace is aggregate. It does not provide per-pair or per-component force/virial decomposition.

## Controlled Runs
- `safe_pme_shift_ref`
  - same authoritative safe baseline settings as TP1.8e
- `safe_ewald_shift`
  - same short-range baseline, same start structure, same topology, same timestep, same seed
  - only the Coulomb solver side changes from `PME` to `Ewald`

## Verified Runtime Fairness
- Both runs keep:
  - `nstlist = 10`
  - `rlist = 0.911`
  - `verlet-buffer-tolerance = -1`
  - `vdw-type = Cut-off`
  - `coulomb-modifier = Potential-shift-Verlet`
  - `pme-order = 4`
  - `fourierspacing = 0.12`
- Runtime logs keep the same short-range kernel and pairlist line:
  - `Using plain-C-4x4 4x4 nonbonded short-range kernels`
  - `updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm`
- The solver-side runtime distinction is real:
  - PME baseline has `Solve PME ...`
  - Ewald variant has `Will do ordinary reciprocal space Ewald sum.` and no `Solve PME`

## compute_globals Trace Results
### 1. The incoming virial split survives unchanged through `m_add`
- There are `403` shared traced `compute_globals` calls, with `401` calls entering the pressure block in both runs.
- Early-window mean absolute deltas:
  - `force_vir_trace_in`: `3.163600376674107`
  - `total_vir_trace_after`: `3.163600376674107`
- Full-window mean absolute deltas:
  - `force_vir_trace_in`: `616.7227922444308`
  - `total_vir_trace_after`: `616.7227922444308`
- Since `shake_vir` stays zero here, `m_add(force_vir, shake_vir, total_vir)` is only carrying the incoming virial split forward. TP1.8f does not show a new compute_globals-local virial branch.

### 2. The pressure handoff is algebraically consistent with `calc_pres`
- `pressure_fac` stays constant at `3.1189687354611904`.
- Residual checks stay near floating-point noise:
  - baseline max abs `pressure_formula_residual_full`: `0.0005656929309841314`
  - variant max abs `pressure_formula_residual_full`: `0.0004889883524262353`
  - baseline max abs `pressure_scalar_residual_full`: `0.000244140625`
  - variant max abs `pressure_scalar_residual_full`: `0.000244140625`
- Example at the first pressure-producing call (`call_index = 2`, `step = 0`):
  - baseline `force_vir_trace_in`: `2128.7225341796875`
  - variant `force_vir_trace_in`: `2136.2614440917969`
  - delta in `force_vir_trace_in`: `+7.538909912109375`
  - baseline `pres_trace_after`: `-3497.0739974975586`
  - variant `pres_trace_after`: `-3520.589038848877`
  - delta in `pres_trace_after`: `-23.51504135131836`
  - residuals remain around `1e-05`
- That matches linear re-expression through `calc_pres`, not a new pressure-handoff amplification branch.

### 3. Pressure/virial differences remain visible, but not as a new `compute_globals` defect
- Early-window mean absolute deltas:
  - `pres_trace_after`: `11.055914197649274`
  - `pressure_scalar`: `3.685302734375`
- Full-window mean absolute deltas:
  - `pres_trace_after`: `2108.869350692031`
  - `pressure_scalar`: `702.9564571303706`
- The pressure-side split is real, but TP1.8f shows it as a consistent representation of incoming virial/ekin differences. It is not direct evidence that `compute_globals` itself is mis-accumulating pressure.

### 4. Runaway still persists
- baseline `RUNAWAY`, onset `0.2 ps`
- variant `RUNAWAY`, onset `0.2 ps`
- TP1.8f therefore does not justify a source-level defect claim.

## What TP1.8f Actually Narrowed
- It did **not** isolate a single root cause.
- It did narrow the compute_globals-level picture:
  - the incoming virial split is already present when `compute_globals` begins its pressure work
  - `m_add` preserves that split
  - `calc_pres` then re-expresses it linearly into `pres` and scalar pressure
- That weakens a compute_globals-specific pressure-handoff bug story.

## What TP1.8f Did Not Prove
- It did not prove the runaway is purely force-side.
- It did not prove the runaway is purely pressure/virial-side.
- It did not prove the later pressure difference is only symptomatic.
- It did not isolate which later consumer, if any, makes the preserved split runaway-relevant.

## Conservative Interpretation
- Best supported reading:
  - the PME-versus-Ewald difference entering `compute_globals` is already present on the virial side
  - `compute_globals` preserves that difference and re-expresses it consistently into pressure
  - the pressure-hand-off logic itself is not where a new anomaly appears in this trace
- That is narrower than TP1.8e, but still below root-cause proof.

## Patch Boundary
- Production patching is still not justified.
- The next step, if needed, is deeper trace-only work on immediate post-`compute_globals` consumers of `pres` / `total_vir`, not a fix-first change.

## Reporting Footer
- Files changed: `src/gromacs/mdlib/md_support.cpp`, `tools/run_tp1_8f_compute_globals_trace/run_compute_globals_trace.py`, `tools/run_tp1_8f_compute_globals_trace/README.md`, `docs/validation_report_tp1_8f.md`, `docs/tp1_8f_compute_globals_trace.md`, artifacts under `tests/reference_results/tp1_8f_compute_globals_trace/`
- Commands run: `cmake --build build --target gmx -j4`; `python3 -m py_compile tools/run_tp1_8f_compute_globals_trace/run_compute_globals_trace.py`; `python3 tools/run_tp1_8f_compute_globals_trace/run_compute_globals_trace.py`; supporting `sed` / `rg` / `python3 - <<'PY' ...` inspections
- Systems executed: `dense_salt_polymer` under `safe_pme_shift_ref`, `safe_ewald_shift`
- Strongest confirmed finding: `compute_globals` preserves the incoming virial split and `calc_pres` re-expresses it linearly, rather than introducing a new pressure-hand-off anomaly
- Strongest unresolved uncertainty: whether the surviving runaway relevance lies upstream of `compute_globals`, in later pressure consumers, or in a broader mixed electrostatics path
- Exact next step recommendation: keep the baseline fixed and trace immediate post-`compute_globals` pressure consumers only if another isolation step is required
- Verdict: `PASS`
