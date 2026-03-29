# TP1.8c Coulomb Accumulation Trace

## Why TP1.8c Exists
- TP1.8b narrowed the authoritative blocker along Coulomb-path lines, but it still ended at `mixed_or_still_unresolved`.
- The missing evidence was source-level trace data around `CpuPpLongRangeNonbondeds::calculate`, specifically the PME-versus-Ewald accumulation split and what that split actually writes into aggregate long-range outputs.

## What Was Already Ruled Out
- `TP1.4` LJ-PME path is inactive in the authoritative setup: `vdw-type = Cut-off`, not `PME`.
- The TP1.8b safe baseline is real and reused here unchanged:
  - `nstlist = 10`
  - `rlist = 0.911`
  - `verlet-buffer-tolerance = -1`
  - `coulombtype = PME`
  - `coulomb-modifier = Potential-shift-Verlet`
  - `vdw-type = Cut-off`

## Trace Localization
### Active traced path
- `src/gromacs/mdlib/force.cpp`
  - `CpuPpLongRangeNonbondeds::calculate`
  - role: selects PME versus Ewald long-range work, accumulates `Vlr_q`, `Vcorr_q`, virial, and writes `InteractionFunction::CoulombReciprocalSpace`

### Solver split
- `src/gromacs/mdlib/force.cpp`
  - `gmx_pme_do(...)` call site
  - active for `safe_pme_shift_ref`, inactive for `safe_ewald_shift`
- `src/gromacs/mdlib/force.cpp`
  - `do_ewald(...)` call site
  - inactive for `safe_pme_shift_ref`, active for `safe_ewald_shift`

### Shared direct-space family
- `src/gromacs/mdlib/forcerec.cpp`
  - PME and Ewald both translate to the same Ewald-family direct-space kernel family
  - TP1.8c therefore keeps the short-range direct-space family fixed while changing the reciprocal solver

### Source basis for the traced aggregate difference
- `src/gromacs/ewald/pme.cpp`
  - `computeEnergyAndVirial = (stepWork.computeEnergy || stepWork.computeVirial)`
  - `*energy_q` and virial outputs are only written inside that gate
- `src/gromacs/ewald/ewald.cpp`
  - `do_ewald(...)` computes reciprocal energy and virial directly and returns energy every call

## Instrumentation
- Trace-only instrumentation was added to `CpuPpLongRangeNonbondeds::calculate`.
- It writes one CSV row per call when `GMX_TP18C_TRACE_FILE` is set.
- Captured fields include:
  - call index
  - `coulomb_type`
  - `coulomb_modifier`
  - `direct_space_family`
  - `compute_pme_on_cpu`
  - `pme_do_called`
  - `ewald_called`
  - `vlr_q_kj`
  - `vcorr_q_kj`
  - `coulomb_recip_term_kj`
  - `virial_q_trace`
  - step-work flags

제한된 정보입니다. This trace does not provide per-pair direct-space decomposition. It only captures the strongest available aggregate/source-level surrogate at this call site.

## Controlled Runs
- `safe_pme_shift_ref`
  - reused authoritative safe baseline
- `safe_ewald_shift`
  - changed only the reciprocal solver from PME to Ewald while keeping the short-range baseline fixed
- `safe_pme_none`
  - optional direct-modifier control with PME kept active

All three runs kept:
- same topology
- same start structure
- same `nstlist`, `rlist`, `verlet-buffer-tolerance`
- same `vdw-type = Cut-off`
- same timestep
- same seed
- same no-thermostat / no-barostat ensemble family

## Verified Runtime Differences
### Baseline PME
- `raw_safe_pme_shift_ref_md.log`
  - `Solve PME ...`
  - `updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm`
  - `Using plain-C-4x4 4x4 nonbonded short-range kernels`

### Ewald variant
- `raw_safe_ewald_shift_md.log`
  - `Will do ordinary reciprocal space Ewald sum.`
  - no `Solve PME` line
  - same pairlist line: `updated every 10 steps, buffer 0.011 nm, rlist 0.911 nm`
  - same short-range kernel line

### Direct-modifier control
- `raw_safe_pme_none_md.log`
  - still `Solve PME ...`
  - same pairlist line
  - only the Coulomb modifier changed

## Trace Results
### Baseline vs Ewald
- `trace_observables_baseline.csv`
  - `pme_do_called = 1` on all traced calls
  - `coulomb_recip_term_kj != 0` on `201 / 20001` calls
- `trace_observables_variant.csv`
  - `ewald_called = 1` on all traced calls
  - `coulomb_recip_term_kj != 0` on `20001 / 20001` calls
- Mean absolute early-call delta:
  - `coulomb_recip_term_kj`: `567.1987297096063`
  - `virial_q_trace`: `603.9684569705184`

This is the strongest verified source-level difference in TP1.8c.

### Baseline vs direct-modifier control
- `trace_observables_direct_modifier.csv`
  - same `pme_do_called` pattern as the baseline
  - same `coulomb_recip_term_kj` pattern as the baseline
- Mean absolute early-call delta:
  - `coulomb_recip_term_kj`: `0.0`
  - `virial_q_trace`: `0.0`

This weakens a simple Coulomb-modifier explanation at the traced call site.

## Stability Outcome
- `safe_pme_shift_ref`
  - `RUNAWAY`, onset `0.2 ps`
- `safe_ewald_shift`
  - `RUNAWAY`, onset `0.2 ps`
- `safe_pme_none`
  - `RUNAWAY`, onset `0.2 ps`

The Ewald variant changes the traced aggregate accumulation behavior strongly, but that does not materially weaken the authoritative runaway. That is why TP1.8c stops at `still_unresolved`.

## Conservative Interpretation
- What TP1.8c supports:
  - the active source-level Coulomb accumulation split is real and traced
  - the PME-versus-Ewald solver split materially changes aggregate reciprocal energy / virial observables at this call site
  - the direct-modifier control does not change those aggregate observables here
- What TP1.8c does **not** support:
  - PME dominance
  - Ewald/direct-space dominance
  - a single proven source-level defect

## Patch Boundary
- Production patching is still not justified.
- The remaining work is trace-depth, not fix-first refactoring.

## Reporting Footer
- Files changed: `src/gromacs/mdlib/force.cpp`, `src/gromacs/mdlib/force.h`, `tools/run_tp1_8c_coulomb_trace/run_coulomb_trace.py`, `docs/validation_report_tp1_8c.md`, `docs/tp1_8c_coulomb_accumulation_trace.md`, artifacts under `tests/reference_results/tp1_8c_coulomb_trace/`
- Commands run: `build/bin/gmx --version`; `python3 -m py_compile tools/run_tp1_8c_coulomb_trace/run_coulomb_trace.py`; `python3 tools/run_tp1_8c_coulomb_trace/run_coulomb_trace.py`; supporting `sed` / `rg` inspections over TP1.8 / TP1.8b artifacts and the traced source files
- Systems executed: `dense_salt_polymer` under `safe_pme_shift_ref`, `safe_ewald_shift`, `safe_pme_none`
- Strongest confirmed finding: PME and Ewald variants populate the traced reciprocal accumulation outputs very differently at `CpuPpLongRangeNonbondeds::calculate`, but that does not materially change runaway onset
- Strongest unresolved uncertainty: whether the surviving blocker lives in a shared post-selection Coulomb force path or in a broader mixed electrostatics path not isolated by the current aggregate trace
- Exact next step recommendation: trace one level deeper into the shared Coulomb accumulation consumers after solver selection before any patching
- Verdict: `PASS`
