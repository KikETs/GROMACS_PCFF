# PCFF implementation paper: frozen scope

Decision recorded on 2026-09-05: stop feature development and transport expansion.
The paper concerns implementation correctness and measured performance. New code
changes are limited to defects exposed by validation. Existing trajectories and
historical installations remain preserved.

## Claim

Implementation of PCFF Class2 bonded terms and cross terms, LJ 9-6 nonbonded
interactions, and supported exact r-RESPA execution in GROMACS, validated against
analytic/finite-difference checks and LAMMPS on specified configurations, with
CPU/GPU performance measured on a declared host and runtime profile.

This is a bounded implementation claim. It does not claim universal PCFF/PCFF+
parameter coverage, unrestricted automatic atom typing, every GROMACS feature,
or converged transport coefficients. Parameter provenance remains essential:
LAMMPS data-file conversion and molecular typing from SMILES are distinct tasks.
The chemistry limits in `current_status_note.md` remain in force unless a
specific newer fixture supplies direct evidence.

## Numerical contract

- Reference engine: LAMMPS `lj/class2/coul/long`, Class2 bonded styles,
  sixth-power mixing and the explicitly recorded exclusion/1-4 convention.
- Compare coordinates, box, charges, topology and coefficients before comparing
  forces. Separate the chosen electrostatic approximation from functional-form
  correctness. PME and PPPM are both present; matching nominal tolerances alone
  does not match the Ewald beta or reciprocal mesh error.
- Record `GMX_PCFF_EWALD_BETA_INV_A` per state and PME/PPPM grids. Do not reuse
  a beta from a different system or NPT box without checking it.
- LAMMPS tail matching is opt-in:
  `GMX_PCFF_LAMMPS_DISPERSION_CORRECTION=1`, with `DispCorr=AllEnerPres`.
  The option is restricted to the guarded PCFF 9-6 cutoff contract.
- Periodic COM removal needs both input and runtime settings. In the tested
  0.5 fs profile, `comm-mode=Linear`, `comm-grps=System`, `nstcomm=400`, and
  `GMX_PCFF_EXACT_RESPA_ALLOW_LINEAR_COM_REMOVAL=1` request removal every 0.2 ps.
  For input generation use `GROMACS_BATCH_NSTCOMM=400`. The legacy generator
  default of 1e9 steps does not provide periodic removal over these runs.
- The frozen evidence covers single-rank CPU and specified hybrid GPU layouts,
  unconstrained periodic systems, and declared two-/three-level schedules.
  It does not grant blanket support to constraints, virtual sites, arbitrary
  domain decomposition, GPU bitwise reproducibility, or arbitrary restart times.

## Evidence organization

The final evidence directory on the validation host is:

`/home/kiket/Desktop/test/GROMACS_PCFF/output/implementation_closeout_20260905/`

`final_validation/` contains final-revision runs and their inputs, EDR/TRR files,
commands, runtime environments, checks and binary/input hashes. Earlier attempts
are retained outside that directory and must not be pooled into the final table.
`REPORT.md` provides the acceptance/limitation boundary and final artifact links.

Reproduction entrypoint:

```bash
/home/kiket/anaconda3/envs/MD/bin/python tools/pcff_respa_parity/validate_implementation_freeze.py micro
/home/kiket/anaconda3/envs/MD/bin/python tools/pcff_respa_parity/validate_implementation_freeze.py static
/home/kiket/anaconda3/envs/MD/bin/python tools/pcff_respa_parity/validate_implementation_freeze.py dynamics
/home/kiket/anaconda3/envs/MD/bin/python tools/pcff_respa_parity/validate_implementation_freeze.py benchmark
```

The driver uses the preserved campaign input/reference paths on that host;
these commands are not a standalone data distribution. `--out`, `--cpu`, and
`--gpu` select separate destinations/installations. Completed runs are reused
only after checking their input and executable/library signatures. Each CLI
simulation starts in a fresh process. Performance runs should be serial.

## Defects closed during finalization

- The earlier exact-r-RESPA COM omission, scalar complete-pairlist setup, and
  optional LAMMPS tail-weighting changes are included in the source freeze.
- A three-level forced-scalar run asserted when exclusion corrections were
  visited on an inner/middle-only step. Those corrections belong to the outer
  contribution; its accumulator is legitimately absent on that substep. Both
  optimized scalar branches now use their existing null-aware accumulation.
  Fresh-process checks exercise specialized and generic scalar branches, compare
  against NBNXM, and independently check total/per-level force closure.
- The GPU test compilation order was repaired. The force-dump test parser now
  reads the writer's global-atom-index column. The GPU usage assertion now checks
  the implemented Class2 bonded report rather than obsolete pair14-only wording.

## Explicit limitations

- Existing in-process diagnostic tests change environment variables between
  mdrun calls, while several diagnostic getters cache the first value. They can
  miss or combine trace files, and are not an all-green release suite. Preserve
  their failure logs. Fresh-process CLI comparisons supply separate numerical
  evidence; they do not claim to repair the embedding/diagnostic API behavior.
- Small-fixture GPU restart checks and large-system strict restart checks are
  different evidence. A failure of the latter must not be relabeled as passing
  merely because identical-input GPU repeats also differ. GPU restart equality
  remains restricted to the actual checks reported in `REPORT.md`.
- Five-ps NVE controls measure finite energy excursions and timestep sensitivity.
  They do not establish asymptotic second-order convergence or long-time ensemble
  equivalence. Short NPT controls do not establish density equilibration.
- Single-host timing does not establish multi-host scaling or a universal speedup.
  CPU double and GPU mixed precision must not be presented as equal-precision
  acceleration; use the same mixed binary's CPU execution as an additional control.
- NE/cNE0 results are archived as exploratory evidence. The 50 ns analyses remain
  window-dependent; neither a 50 ns convergence threshold nor a unique cause of
  all transport discrepancies has been demonstrated. GK is outside the paper.

Suggested scope statement:

> This work validates the implemented PCFF interactions, bounded numerical
> execution paths and computational performance. Convergence and cross-engine
> statistical equivalence of long-time transport coefficients are not established.
