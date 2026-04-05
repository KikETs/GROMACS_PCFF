# M10.2.1 — Longer NPT Convergence Findings

## Overview
The original `M10.2.1` interpretation was too optimistic.

After aligning the runner with the current `M10.2` semantics:
- shared initial velocities
- molecule-local atom naming
- checkpoint-based GROMACS restart
- common density recomputation from mass and volume

the longer 100 ps NPT run still does **not** support conductivity handoff.

## System & Runtime
- `small_oligomer_medium_100ps`
- 384 atoms
- 10 ps NPT equilibration
- 100 ps NPT production
- `dt = 2 fs`

## Main Findings
### Temperature
- GROMACS mean: `299.92 K`
- LAMMPS mean: `298.64 K`
- delta: `1.28 K`
- both engines are thermally well behaved

### Density / Volume
- GROMACS density mean: `16.41 kg/m^3`
- LAMMPS density mean: `11.53 kg/m^3`
- cross-engine density gap: `29.74%`
- GROMACS density drift: `0.00170`
- LAMMPS density drift: `1.91332`

This is the real blocker.

## Interpretation
The 100 ps run changed the diagnosis:
- it did **not** reveal an exact r-RESPA runtime failure
- it did **not** reveal a temperature-control failure
- it **did** show that density/volume relaxation is still not converged strongly enough

So the method is not ready for conductivity production.

## Machine-Readable Artifacts
- [m10_2_1_summary.json](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_2_1_convergence_gate/m10_2_1_summary.json)
- [m10_2_1_gate_decision.json](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_2_1_convergence_gate/m10_2_1_gate_decision.json)
- [report.json](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_2_1_convergence_gate/small_oligomer_medium_100ps/report.json)

## Bottom Line
`M10.2.1` is currently a `partial` result.

That is enough to say:
- do not start conductivity production yet
- the next work should target density/volume relaxation, not transport analysis itself
