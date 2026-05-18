# M10.2 — Medium-Scale Ensemble Gate Findings

## Overview
The original short-horizon `NPT vs NPT` parity interpretation was too weak.

This gate is also not exact `r-RESPA` evidence.
The checked-in runner writes plain `integrator = md` inputs, so `M10.2` is a medium-scale diagnostic gate only.

The updated gate separates two different questions:
- **Can both engines maintain medium-scale thermal behavior under the same fixed-volume setup?**
- **Do short NPT runs remain numerically stable enough to justify longer convergence work?**

Only the first question is blocking for `M10.2`.

## Execution Prerequisite
- runner: [run_m10_2.py](../tools/run_m10_2_ensemble_gate/run_m10_2.py)
- required interpreter: `python3`
- reason: this runner imports `numpy` and was validated in the `MD` conda environment, not bare system `python3`

## System
- `small_oligomer_medium`
- 64 replicated oligomers
- 384 atoms
- total mass: `4995.456 amu`

## NVT Parity Findings
Protocol:
- shared initial velocities generated once in LAMMPS and injected into both engines
- 2 ps equilibration + 8 ps production
- fixed volume

Results:
- GROMACS temperature mean: `301.13 K`
- LAMMPS temperature mean: `298.50 K`
- temperature delta: `2.63 K`
- GROMACS recomputed density: `16.2015 kg/m^3`
- LAMMPS recomputed density: `16.2015 kg/m^3`
- parity status: `pass`

Meaning:
- once barostat-driven volume relaxation is removed, the medium-scale bridge behaves consistently enough for a thermal parity gate

## NPT Stability Findings
Protocol:
- shared initial velocities
- 5 ps equilibration + 15 ps production
- isotropic NPT

Results:
- GROMACS NPT stability: `partial`
- LAMMPS NPT stability: `partial`
- GROMACS recomputed density drift: `0.00030`
- LAMMPS recomputed density drift: `0.14667`

Meaning:
- the short NPT runs do not blow up
- but they are still relaxing, especially on the LAMMPS side
- this is a stability/trend diagnostic, not convergence proof

## Gate Decision
Blocking:
- `small_oligomer_medium.nvt_parity`

Non-blocking:
- `small_oligomer_medium.npt_stability`

Final machine-readable decision:
- [m10_2_gate_decision.json](../tests/reference_results/m10_2_ensemble_gate/m10_2_gate_decision.json)
- overall status: `pass`

## Limits
- This gate does not justify conductivity production by itself.
- Density/volume convergence still belongs to `M10.2.1` or a later long-horizon convergence gate.
- Absolute cross-engine potential energy remains diagnostic only because the bridge already showed a constant-offset problem in earlier parity audits.

## Practical Interpretation
`M10.2` now means:
- medium-scale NVT parity passes in this plain-`md` diagnostic setup
- short NPT runs are stable enough to continue diagnostics

It does **not** mean:
- exact `r-RESPA` medium-scale closure exists
- NPT parity is solved
- any transport-production readiness claim follows
