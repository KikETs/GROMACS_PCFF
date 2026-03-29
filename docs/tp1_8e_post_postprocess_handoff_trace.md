# TP1.8e Post-Postprocess Handoff Trace

## Why TP1.8e Exists
- TP1.8d showed that the PME-versus-Ewald difference survives through `postProcessForces`.
- That still did not answer whether the surviving force-side delta matters at the next handoff actually used by integration or pressure reporting.
- TP1.8e therefore traces one level deeper, at the caller side after `postProcessForces`.

## What Was Already Known
- `TP1.8c`:
  - PME and Ewald split strongly at `CpuPpLongRangeNonbondeds::calculate`
  - runaway onset stayed `0.2 ps`
- `TP1.8d`:
  - the producer-boundary reciprocal-energy split was heavily transformed before final potential bookkeeping
  - a force-side difference survived through `postProcessForces`
  - root cause was still unresolved
- `TP1.4` LJ-PME path remains inactive here because the authoritative setup still uses `vdw-type = Cut-off`.

## What TP1.8e Traces
### Upstream reference
- `src/gromacs/mdlib/sim_util.cpp`
  - `postProcessForces`
  - immediate downstream consumer already traced in TP1.8d

### Later handoff consumers
- `src/gromacs/mdrun/md.cpp`
  - `LegacySimulator::do_md`
  - traced at:
    - `after_do_force_return`
    - `before_update_coords`
    - `after_update_coords`
    - `after_compute_globals`
- `src/gromacs/mdlib/update.cpp`
  - `Update::update_coords`
  - traced at caller boundary using the force buffer handed into the update and the immediate `xprime` / velocity state after update
- `src/gromacs/mdlib/md_support.cpp`
  - `compute_globals`
  - traced at caller boundary after `total_vir` and `pres` are finalized

## Instrumentation
- Trace-only instrumentation was added in `src/gromacs/mdrun/md.cpp`.
- It writes one CSV row per traced stage when `GMX_TP18E_TRACE_FILE` is set.
- Captured fields:
  - `force_l2`
  - `force_max_abs`
  - `state_x_l2`
  - `state_x_max_abs`
  - `state_v_l2`
  - `state_v_max_abs`
  - `xprime_l2`
  - `xprime_max_abs`
  - `force_vir_trace`
  - `shake_vir_trace`
  - `total_vir_trace`
  - `pres_trace`
  - aggregate energy / temperature terms

제한된 정보입니다. This trace is aggregate. It does not provide per-pair or per-component force decomposition.

## Controlled Runs
- `safe_pme_shift_ref`
  - same authoritative safe baseline settings as TP1.8d
- `safe_ewald_shift`
  - same short-range baseline, same start structure, same topology, same timestep, same seed
  - only the Coulomb solver side changes from `PME` to `Ewald`

## Verified Runtime Fairness
- Both runs keep:
  - `nstlist = 10`
  - `rlist = 0.911`
  - `verlet-buffer-tolerance = -1`
  - `vdw-type = Cut-off`
  - `pme-order = 4`
  - `fourierspacing = 0.12`
- Runtime logs keep the same short-range kernel and pairlist line:
  - `Using plain-C-4x4 4x4 nonbonded short-range kernels`
  - `updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm`
- The solver-side runtime distinction is real:
  - PME baseline has `Solve PME ...`
  - Ewald variant has `Will do ordinary reciprocal space Ewald sum.` and no `Solve PME`

## Handoff Trace Results
### 1. The later force handoff preserves the TP1.8d force-side delta
- Full-window mean absolute delta in `force_l2`:
  - `after_do_force_return`: `1012.5172460553654`
  - `before_update_coords`: `1012.5172460553654`
  - `after_update_coords`: `1012.5172460553654`
- The preserved equality means the later force handoff does not attenuate the surviving PME-versus-Ewald force-side difference before integration uses it.

### 2. Immediate one-step state divergence is small
- Full-window mean absolute deltas after `update_coords`:
  - `state_v_l2`: `0.37599275745973754`
  - `xprime_l2`: `0.8964216798462992`
- That is much smaller than the carried force-buffer delta at the same stage.

### 3. The later virial/pressure handoff is a stronger aggregate signal
- `after_compute_globals` full-window mean absolute deltas:
  - `total_vir_trace`: `644.3893600045296`
  - `pres_trace`: `2162.748007006553`
- So the later pressure-side handoff shows a stronger aggregate survival signal than the immediate one-step `v` / `xprime` update.

### 4. Runaway still persists
- baseline `RUNAWAY`, onset `0.2 ps`
- variant `RUNAWAY`, onset `0.2 ps`
- TP1.8e therefore does not justify a defect claim.

## What TP1.8e Actually Narrowed
- It did **not** isolate a single root cause.
- It did narrow the later handoff picture:
  - the surviving force-side difference from TP1.8d is preserved at the integration force handoff
  - the immediate one-step integrated state divergence remains small
  - the later virial/pressure handoff shows a stronger aggregate split

## What TP1.8e Did Not Prove
- It did not prove the runaway is purely reciprocal-space / PME.
- It did not prove the runaway is purely Ewald/direct-space.
- It did not prove that the later pressure/virial split is the causal runaway driver.

## Conservative Interpretation
- Best supported reading:
  - the force-side delta survives to the later handoff used by integration
  - but it is not immediately amplified into a large one-step `v` / `xprime` divergence
  - the later pressure/virial handoff is a stronger aggregate signal than the immediate one-step integration state
- That is narrower than TP1.8d, but still below root-cause proof.

## Patch Boundary
- Production patching is still not justified.
- The next step, if needed, is deeper trace-only work on the later virial/pressure or state-transfer consumers, not a fix-first change.

## Reporting Footer
- Files changed: `src/gromacs/mdrun/md.cpp`, `tools/run_tp1_8e_handoff_trace/run_handoff_trace.py`, `tools/run_tp1_8e_handoff_trace/README.md`, `docs/validation_report_tp1_8e.md`, `docs/tp1_8e_post_postprocess_handoff_trace.md`, artifacts under `tests/reference_results/tp1_8e_handoff_trace/`
- Commands run: `cmake --build build --target gmx -j4`; `python3 -m py_compile tools/run_tp1_8e_handoff_trace/run_handoff_trace.py`; `python3 tools/run_tp1_8e_handoff_trace/run_handoff_trace.py`; supporting `sed` / `rg` / `python3 - <<'PY' ...` inspections
- Systems executed: `dense_salt_polymer` under `safe_pme_shift_ref`, `safe_ewald_shift`
- Strongest confirmed finding: the surviving PME-versus-Ewald force-side delta is preserved at the later integration force handoff, while the later virial/pressure handoff shows a larger aggregate split than the immediate one-step state update
- Strongest unresolved uncertainty: whether the later pressure/virial split is causally important for runaway or only correlated with a broader mixed electrostatics issue
- Exact next step recommendation: keep the baseline fixed and trace one level deeper only on later virial/pressure or state-transfer consumers if another isolation step is required
- Verdict: `PASS`
