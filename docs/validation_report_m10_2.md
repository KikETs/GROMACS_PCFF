# M10.2 — Medium-Scale Ensemble Gate Validation Report

## Scope
M10.2 is no longer treated as a short cross-engine NPT parity claim.
It is also not exact `r-RESPA` evidence: the checked-in runner writes plain `integrator = md` inputs, so this report is diagnostic-only.

The gate is now split into:
- **Required:** medium-scale **NVT thermal parity**
- **Diagnostic only:** longer short-horizon **NPT stability / trend**

This change is explicit in the machine-readable decision artifact:
- [m10_2_gate_decision.json](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_2_ensemble_gate/m10_2_gate_decision.json)

## Execution Prerequisite
- runner: [run_m10_2.py](/home/kiket/Desktop/test/GROMACS_PCFF/tools/run_m10_2_ensemble_gate/run_m10_2.py)
- required interpreter: `/home/kiket/anaconda3/envs/MD/bin/python3`
- reason: the runner imports `numpy`; validation was performed in the `MD` conda environment

## Result
- `small_oligomer_medium.nvt_parity`: `pass`
- `small_oligomer_medium.npt_stability`: `partial`
- overall `M10.2` gate: `pass`

## Required Pass: Medium-Scale NVT Parity
System:
- `small_oligomer_medium`
- 64 replicated oligomers
- 384 atoms

Protocol:
- shared initial velocities injected into both engines
- 2 ps equilibration + 8 ps production
- fixed volume / NVT

Observed results:
- GROMACS mean temperature: `301.13 K`
- LAMMPS mean temperature: `298.50 K`
- temperature delta: `2.63 K`
- recomputed density difference: `2.19e-16`

Interpretation:
- medium-scale thermal behavior is aligned when barostat effects are removed
- the density comparison is performed from a common mass/volume formula, not engine-reported density alone

## Non-Blocking Diagnostic: Short NPT Stability
Protocol:
- shared initial velocities injected into both engines
- 5 ps equilibration + 15 ps production
- isotropic NPT

Observed results:
- GROMACS NPT stability: `partial`
- LAMMPS NPT stability: `partial`
- strongest residual issue: LAMMPS density drift remains non-negligible over the short horizon

Key values:
- GROMACS density drift (recomputed): `0.00030`
- LAMMPS density drift (recomputed): `0.14667`

Interpretation:
- this short NPT window is useful as a stability/trend diagnostic
- it is not strong enough to certify ensemble convergence
- full convergence remains owned by `M10.2.1` or later

## What This Report Does Not Prove
- It does **not** prove short-horizon NPT parity.
- It does **not** certify conductivity production readiness.
- It does **not** replace longer convergence checks for density/volume relaxation.

## Artifacts
- runner: [run_m10_2.py](/home/kiket/Desktop/test/GROMACS_PCFF/tools/run_m10_2_ensemble_gate/run_m10_2.py)
- summary: [m10_2_summary.json](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_2_ensemble_gate/m10_2_summary.json)
- decision: [m10_2_gate_decision.json](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_2_ensemble_gate/m10_2_gate_decision.json)
- detailed report: [report.json](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_2_ensemble_gate/small_oligomer_medium/report.json)

## Conclusion
`M10.2` is closed only as a **plain-`md` medium-scale NVT parity + NPT stability handoff gate**.

It may be cited as evidence that:
- the medium-scale bridge is thermally consistent under fixed volume in this diagnostic setup
- short NPT runs remain finite and diagnostically usable

It must **not** be cited as proof that the system is ready for conductivity production. That requires the longer convergence path.
It must **not** be cited as exact `r-RESPA` medium-scale evidence.
