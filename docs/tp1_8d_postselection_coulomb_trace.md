# TP1.8d Post-Selection Coulomb Trace

## Why TP1.8d Exists
- TP1.8c showed that PME and Ewald populate the aggregate reciprocal outputs very differently at `CpuPpLongRangeNonbondeds::calculate`.
- That did **not** answer whether the difference survives downstream in any runaway-relevant form.
- TP1.8d therefore traces the first consumer boundary after `CpuPpLongRangeNonbondeds::calculate`.

## What Was Already Known
- `TP1.8c` producer-boundary result:
  - baseline PME and Ewald differ strongly in `Vlr_q`, reciprocal-energy bookkeeping, and virial traces at `CpuPpLongRangeNonbondeds::calculate`
  - runaway onset remained `0.2 ps`
- `TP1.4` LJ-PME path remains inactive here because the authoritative setup still uses `vdw-type = Cut-off`.

## What TP1.8d Traces
### Upstream reference
- `src/gromacs/mdlib/force.cpp`
  - `CpuPpLongRangeNonbondeds::calculate`
  - producer boundary already traced in TP1.8c

### Immediate downstream consumers
- `src/gromacs/mdlib/sim_util.cpp`
  - `do_force`
  - traced at `after_longrange` and `after_accumulate_energy`
- `src/gromacs/mdlib/sim_util.cpp`
  - `postProcessForces`
  - traced at `before_postprocess` and `after_postprocess`
- `src/gromacs/mdlib/enerdata_utils.cpp`
  - `accumulatePotentialEnergies`
  - traced at caller boundary after the final potential bookkeeping step
- `src/gromacs/mdtypes/forceoutput.h`
  - `ForceWithVirial`
  - traced indirectly through:
    - `forceWithVirial.force_`
    - `forceWithVirial.getVirial()`

## Instrumentation
- Trace-only instrumentation was added in `src/gromacs/mdlib/sim_util.cpp`.
- It writes one CSV row per step and per traced stage when `GMX_TP18D_TRACE_FILE` is set.
- Captured fields:
  - `coulomb_recip_term_kj`
  - `potential_energy_kj`
  - `force_with_virial_l2`
  - `force_with_virial_max_abs`
  - `final_force_l2`
  - `final_force_max_abs`
  - `direct_virial_trace`
  - `vir_force_trace`
  - buffer-sharing flags and step-work flags

제한된 정보입니다. This trace is aggregate. It does not provide per-pair or per-component force decomposition.

## Controlled Runs
- `safe_pme_shift_ref`
  - same authoritative safe baseline settings as TP1.8c
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

## Downstream Trace Results
### 1. Producer-boundary difference is huge
- `after_longrange`
  - early mean absolute delta in `coulomb_recip_term_kj`: `567.1987245474289`

### 2. Final potential bookkeeping does not preserve that magnitude
- `after_accumulate_energy`
  - early mean absolute delta in `potential_energy_kj`: `1.35546875`
- That is a strong downstream transformation/cancellation relative to the producer-boundary reciprocal-energy difference.

### 3. Force-side difference survives through the immediate consumer
- `after_longrange`
  - full mean absolute delta in `final_force_l2`: `1012.5654315746223`
- `after_postprocess`
  - full mean absolute delta in `final_force_l2`: `1012.517246055365`
- The near-equality means the force-side difference is preserved through `postProcessForces`, not cancelled there.

### 4. Direct-virial-only explanation is weakened
- `after_longrange`
  - early mean absolute delta in `direct_virial_trace`: `0.10660594731421359`
- `after_postprocess`
  - early mean absolute delta in `vir_force_trace`: `0.11010742187497512`
- This is small compared with the force-side downstream delta.

## What TP1.8d Actually Narrowed
- It did **not** isolate a single root cause.
- It did narrow the downstream picture:
  - the huge reciprocal-energy bookkeeping split from TP1.8c is mostly attenuated before final potential bookkeeping
  - some difference still survives into the final force consumer
  - the immediate consumer does not erase that force-side difference

## What TP1.8d Did Not Prove
- It did not prove the surviving blocker is purely reciprocal-space / PME.
- It did not prove the surviving blocker is purely Ewald/direct-space.
- It did not prove that the surviving downstream force delta is, by itself, the runaway trigger.

## Conservative Interpretation
- Best supported reading:
  - downstream handling is mixed
  - the PME-versus-Ewald difference is transformed strongly in energy bookkeeping
  - but not cancelled in final force accumulation
- That is narrower than TP1.8c, but still below root-cause proof.

## Patch Boundary
- Production patching is still not justified.
- The next step, if needed, is deeper trace-only work on the shared force-consumer path, not a fix-first change.

## Reporting Footer
- Files changed: `src/gromacs/mdlib/sim_util.cpp`, `tools/run_tp1_8d_coulomb_consumer_trace/run_coulomb_consumer_trace.py`, `tools/run_tp1_8d_coulomb_consumer_trace/README.md`, `docs/validation_report_tp1_8d.md`, `docs/tp1_8d_postselection_coulomb_trace.md`, artifacts under `tests/reference_results/tp1_8d_coulomb_consumer_trace/`
- Commands run: `cmake --build build --target gmx -j4`; `python3 -m py_compile tools/run_tp1_8d_coulomb_consumer_trace/run_coulomb_consumer_trace.py`; `python3 tools/run_tp1_8d_coulomb_consumer_trace/run_coulomb_consumer_trace.py`; supporting `sed` / `rg` / `python3 - <<'PY' ...` inspections
- Systems executed: `dense_salt_polymer` under `safe_pme_shift_ref`, `safe_ewald_shift`
- Strongest confirmed finding: the PME-versus-Ewald difference is heavily transformed before final potential bookkeeping, but preserved through the immediate force consumer
- Strongest unresolved uncertainty: whether the surviving downstream force delta is causally important for runaway or only correlated with a broader mixed electrostatics issue
- Exact next step recommendation: keep the baseline fixed and trace one level deeper only on the shared post-selection force-consumer path if another isolation step is required
- Verdict: `PASS`
