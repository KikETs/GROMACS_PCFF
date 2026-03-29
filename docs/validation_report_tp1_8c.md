# TP1.8c Validation Report

## Scope
- Milestone: `TP1.8c — Source-Level Coulomb Accumulation Trace on Authoritative Safe Baseline`
- Status: `PASS`
- Boundary kept: no r-RESPA work, no transport work, no production-fix claim

## Before Execution
- Goal: reuse the TP1.8b authoritative safe baseline, trace `CpuPpLongRangeNonbondeds::calculate`, and compare baseline PME against a narrower Ewald variant under the same short-range baseline.
- Files/functions inspected: `src/gromacs/mdlib/force.cpp`, `src/gromacs/mdlib/force.h`, `src/gromacs/mdlib/forcerec.cpp`, `src/gromacs/ewald/pme.cpp`, `src/gromacs/ewald/ewald.cpp`, TP1.8 / TP1.8b artifacts under `tests/reference_results/tp1_8_longrange_isolation/` and `tests/reference_results/tp1_8b_coulomb_separation/`.
- Instrumentation strategy: add trace-only CSV emission inside `CpuPpLongRangeNonbondeds::calculate` gated by `GMX_TP18C_TRACE_FILE`; do not alter force logic.
- Captured observables: call-path identity, `pme_do_called`, `ewald_called`, `Vlr_q`, `Vcorr_q`, `InteractionFunction::CoulombReciprocalSpace`, virial traces, short-range family labels, and step-work flags.
- Fairness rule: keep topology, start structure, `nstlist = 10`, `rlist = 0.911`, `verlet-buffer-tolerance = -1`, `vdw-type = Cut-off`, timestep, seed, and ensemble family fixed.

## Executed Runs
- Baseline: `safe_pme_shift_ref`
- Narrow reciprocal variant: `safe_ewald_shift`
- Optional direct-modifier control: `safe_pme_none`

Raw outputs and machine-readable artifacts are under `tests/reference_results/tp1_8c_coulomb_trace/`.

## Strongest Confirmed Finding
- The traced source-level accumulation path is real and runtime-distinct. In `src/gromacs/mdlib/force.cpp`, `CpuPpLongRangeNonbondeds::calculate` calls `gmx_pme_do` for the baseline and `do_ewald` for the Ewald variant. The trace shows a strong aggregate difference:
  - baseline PME: `coulomb_recip_term_kj` is non-zero on `201 / 20001` calls
  - Ewald variant: `coulomb_recip_term_kj` is non-zero on `20001 / 20001` calls
- Source basis for that difference:
  - `src/gromacs/ewald/pme.cpp`: `computeEnergyAndVirial = (stepWork.computeEnergy || stepWork.computeVirial)` and `*energy_q` / virial outputs are filled only inside that gate
  - `src/gromacs/ewald/ewald.cpp`: `do_ewald(...)` returns reciprocal energy every call and accumulates virial directly

## Strongest Negative Result
- That strong traced difference did **not** rescue the authoritative instability. All traced runs still show `RUNAWAY` with onset `0.2 ps`.
- Therefore TP1.8c weakens a simple “PME mesh reciprocal accumulation alone is the blocker” story.

## Conservative Interpretation
- `TP1.4` LJ-PME / SixthPower path remains inactive here because all TP1.8c runs keep `vdw-type = Cut-off`.
- The optional `safe_pme_none` control produced an identical long-range trace to the baseline at this call site, so a simple Coulomb-modifier explanation is weakened.
- The surviving blocker is narrowed to `still_unresolved`, with the best live hypothesis being a shared Ewald-family / broader Coulomb accumulation path outside this simple solver-versus-modifier distinction.

## Strongest Unresolved Uncertainty
- The current trace is aggregate, not per-pair or per-force-component decomposition. It shows how this call site populates reciprocal energy / virial observables, but it does not isolate whether the surviving runaway comes from:
  - a shared force accumulation path after PME/Ewald selection
  - a broader mixed electrostatics issue outside the traced aggregate terms

## Exact Next Step Recommendation
- Keep the TP1.8b safe baseline fixed and trace one level deeper inside the shared Coulomb accumulation flow after solver selection, centered on the PME-versus-Ewald accumulation outputs and their immediate consumers, before any production patching.

## Reporting Footer
- Files changed: `src/gromacs/mdlib/force.cpp`, `src/gromacs/mdlib/force.h`, `tools/run_tp1_8c_coulomb_trace/run_coulomb_trace.py`, `docs/validation_report_tp1_8c.md`, `docs/tp1_8c_coulomb_accumulation_trace.md`, artifacts under `tests/reference_results/tp1_8c_coulomb_trace/`
- Commands run: `build/bin/gmx --version`; `python3 -m py_compile tools/run_tp1_8c_coulomb_trace/run_coulomb_trace.py`; `python3 tools/run_tp1_8c_coulomb_trace/run_coulomb_trace.py`; supporting `sed` / `rg` inspections over TP1.8 / TP1.8b artifacts and relevant source files
- Systems executed: `dense_salt_polymer` under `safe_pme_shift_ref`, `safe_ewald_shift`, `safe_pme_none`
- Verdict: `PASS`
