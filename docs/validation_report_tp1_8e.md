# TP1.8e Validation Report

## Scope
- Milestone: `TP1.8e — Post-Postprocess Force/Virial Handoff Trace`
- Status: `PASS`
- Boundary kept: no r-RESPA work, no transport work, no production-fix claim

## Before Execution
- Goal: reuse the TP1.8d authoritative safe baseline, trace one level deeper after `postProcessForces`, and compare the PME baseline against the narrowed Ewald variant at the next force/virial/integration-relevant handoff.
- Files/functions inspected: `src/gromacs/mdrun/md.cpp`, `src/gromacs/mdlib/update.cpp`, `src/gromacs/mdlib/md_support.cpp`, plus TP1.8c / TP1.8d artifacts.
- Post-postprocess trace strategy: add trace-only snapshots at four points:
  - `after_do_force_return`
  - `before_update_coords`
  - `after_update_coords`
  - `after_compute_globals`
- Captured handoff observables: final force-buffer norms, current state `x`/`v` norms, `xprime` norms after update, `force_vir` / `shake_vir` / `total_vir` traces, pressure trace, and aggregate energy/temperature terms.
- Fairness rule: keep topology, start structure, `nstlist = 10`, `rlist = 0.911`, `verlet-buffer-tolerance = -1`, `vdw-type = Cut-off`, timestep, seed, and ensemble family fixed.

## Executed Runs
- Baseline: `safe_pme_shift_ref`
- Narrowed Coulomb variant: `safe_ewald_shift`

Raw outputs and machine-readable artifacts are under `tests/reference_results/tp1_8e_handoff_trace/`.

## Strongest Confirmed Finding
- The PME-versus-Ewald force-side difference survives intact into the next force handoff used by integration.
- Full-window comparison:
  - `after_do_force_return` mean absolute delta in `force_l2`: `1012.5172460553654`
  - `before_update_coords` mean absolute delta in `force_l2`: `1012.5172460553654`
  - `after_update_coords` mean absolute delta in `force_l2`: `1012.5172460553654`
- That flat carry-through means the surviving TP1.8d force-side difference is preserved at the later handoff boundary, not attenuated before `Update::update_coords`.

## Strongest Later-Handoff Attenuation Signal
- The immediate integration-relevant state divergence remains small at this next handoff.
- Full-window comparison:
  - `after_update_coords` mean absolute delta in `state_v_l2`: `0.37599275745973754`
  - `after_update_coords` mean absolute delta in `xprime_l2`: `0.8964216798462992`
- Those are much smaller than the carried force-buffer delta at the same stage.

## Strongest Virial/Pressure Survival Signal
- The later virial/pressure handoff shows a larger aggregate split than the immediate state update.
- `after_compute_globals` full-window comparison:
  - mean absolute delta in `total_vir_trace`: `644.3893600045296`
  - mean absolute delta in `pres_trace`: `2162.748007006553`
- This is a real downstream pressure-side survival signal, but it is still aggregate and not root-cause proof.

## Strongest Negative Result
- The authoritative instability still persists in both runs:
  - baseline `RUNAWAY`, onset `0.2 ps`
  - Ewald variant `RUNAWAY`, onset `0.2 ps`
- TP1.8e therefore does not justify a source-level defect claim.

## Conservative Interpretation
- TP1.8e supports:
  - the TP1.8d force-side delta survives unchanged into the `update_coords` handoff
  - the immediate one-step integration state divergence is small at this traced boundary
  - the later virial/pressure handoff shows a larger aggregate split after `compute_globals`
- TP1.8e does **not** support:
  - PME dominance
  - Ewald/direct dominance
  - a single proven handoff defect

## Strongest Unresolved Uncertainty
- The pressure/virial-side signal is stronger than the immediate one-step state-update signal, but this trace is still aggregate. It does not prove whether that later pressure/virial divergence is causal for runaway or only a downstream symptom of a broader mixed electrostatics issue.

## Exact Next Step Recommendation
- Keep the authoritative safe baseline fixed and, only if another isolation step is required, trace one level deeper into the later `compute_globals` pressure/virial consumers or other state-transfer boundaries. Do not patch production logic yet.

## Reporting Footer
- Files changed: `src/gromacs/mdrun/md.cpp`, `tools/run_tp1_8e_handoff_trace/run_handoff_trace.py`, `tools/run_tp1_8e_handoff_trace/README.md`, `docs/validation_report_tp1_8e.md`, `docs/tp1_8e_post_postprocess_handoff_trace.md`, artifacts under `tests/reference_results/tp1_8e_handoff_trace/`
- Commands run: `cmake --build build --target gmx -j4`; `python3 -m py_compile tools/run_tp1_8e_handoff_trace/run_handoff_trace.py`; `python3 tools/run_tp1_8e_handoff_trace/run_handoff_trace.py`; supporting `sed` / `rg` / `python3 - <<'PY' ...` inspections over TP1.8c/TP1.8d artifacts and TP1.8e handoff outputs
- Systems executed: `dense_salt_polymer` under `safe_pme_shift_ref`, `safe_ewald_shift`
- Verdict: `PASS`
