# TP1.8d Validation Report

## Scope
- Milestone: `TP1.8d — Post-Selection Coulomb Accumulation Consumer Trace`
- Status: `PASS`
- Boundary kept: no r-RESPA work, no transport work, no production-fix claim

## Before Execution
- Goal: reuse the TP1.8c authoritative safe baseline, trace the immediate downstream consumers after `CpuPpLongRangeNonbondeds::calculate`, and compare the PME baseline against the narrowed Ewald variant.
- Files/functions inspected: `src/gromacs/mdlib/force.cpp`, `src/gromacs/mdlib/sim_util.cpp`, `src/gromacs/mdtypes/forceoutput.h`, `src/gromacs/mdlib/enerdata_utils.cpp`, `src/gromacs/mdlib/forcerec.cpp`, plus TP1.8b / TP1.8c artifacts.
- Downstream trace strategy: add trace-only snapshots at four points:
  - `after_longrange`
  - `before_postprocess`
  - `after_postprocess`
  - `after_accumulate_energy`
- Captured downstream observables: `CoulombReciprocalSpace`, `PotentialEnergy`, `ForceWithVirial` force-buffer norms, final force-buffer norms, direct virial trace, and `vir_force` trace.
- Fairness rule: keep topology, start structure, `nstlist = 10`, `rlist = 0.911`, `verlet-buffer-tolerance = -1`, `vdw-type = Cut-off`, timestep, seed, and ensemble family fixed.

## Executed Runs
- Baseline: `safe_pme_shift_ref`
- Narrowed Coulomb variant: `safe_ewald_shift`

Raw outputs and machine-readable artifacts are under `tests/reference_results/tp1_8d_coulomb_consumer_trace/`.

## Strongest Confirmed Finding
- The large PME-versus-Ewald producer-boundary reciprocal-energy difference is **not** preserved at the same magnitude downstream.
- Early-window (`step <= 200`) comparison:
  - `after_longrange` mean absolute delta in `coulomb_recip_term_kj`: `567.1987245474289`
  - `after_postprocess` mean absolute delta in `final_force_l2`: `16.82640800359558`
  - `after_accumulate_energy` mean absolute delta in `potential_energy_kj`: `1.35546875`
- This is a real downstream transformation: the huge producer-boundary reciprocal-energy bookkeeping difference is strongly attenuated at the immediate force/energy consumer boundaries.

## Strongest Downstream Survival Signal
- The difference is **not** fully cancelled. It survives into final force accumulation.
- Full-window comparison:
  - `after_longrange` mean absolute delta in `final_force_l2`: `1012.5654315746223`
  - `after_postprocess` mean absolute delta in `final_force_l2`: `1012.517246055365`
- That near-equality shows the force-side difference survives through the immediate `postProcessForces` consumer essentially unchanged, not cancelled there.

## Strongest Negative Result
- Even with that downstream survival, the authoritative instability still persists:
  - baseline `RUNAWAY`, onset `0.2 ps`
  - Ewald variant `RUNAWAY`, onset `0.2 ps`
- So TP1.8d does not justify a source-level root-cause claim.

## Conservative Interpretation
- TP1.8d supports:
  - the PME-versus-Ewald difference survives into downstream force consumers
  - the same difference is much smaller at the final potential-energy consumer than at the producer boundary
  - a simple direct-virial-only story is weakened, because the direct virial trace delta stays small:
    - `after_longrange` early `direct_virial_trace` delta: `0.10660594731421359`
    - `after_postprocess` early `vir_force_trace` delta: `0.11010742187497512`
- TP1.8d does **not** support:
  - PME dominance
  - Ewald/direct-space dominance
  - a single proven downstream defect

## Strongest Unresolved Uncertainty
- The surviving force-side delta may be physically relevant, but this trace is still aggregate. It does not isolate which exact post-selection force consumer or later integration-relevant handoff turns that surviving difference into runaway.

## Exact Next Step Recommendation
- Keep the authoritative safe baseline fixed and trace one level deeper into the shared post-selection force-consumer path only if needed, centered on the force accumulation handoff after `postProcessForces`, before any production patching.

## Reporting Footer
- Files changed: `src/gromacs/mdlib/sim_util.cpp`, `tools/run_tp1_8d_coulomb_consumer_trace/run_coulomb_consumer_trace.py`, `tools/run_tp1_8d_coulomb_consumer_trace/README.md`, `docs/validation_report_tp1_8d.md`, `docs/tp1_8d_postselection_coulomb_trace.md`, artifacts under `tests/reference_results/tp1_8d_coulomb_consumer_trace/`
- Commands run: `cmake --build build --target gmx -j4`; `python3 -m py_compile tools/run_tp1_8d_coulomb_consumer_trace/run_coulomb_consumer_trace.py`; `python3 tools/run_tp1_8d_coulomb_consumer_trace/run_coulomb_consumer_trace.py`; supporting `sed` / `rg` / `python3 - <<'PY' ...` inspections over TP1.8b/TP1.8c artifacts and downstream trace outputs
- Systems executed: `dense_salt_polymer` under `safe_pme_shift_ref`, `safe_ewald_shift`
- Verdict: `PASS`
