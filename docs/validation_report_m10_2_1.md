# M10.2.1 — Longer NPT Convergence Validation Report

## Scope
M10.2.1 extends the medium-scale neutral oligomer gate to a longer `100 ps` NPT production horizon.

Its purpose is narrower than a transport-production claim:
- check whether longer-horizon NPT temperature remains well behaved
- check whether density/volume relaxation is actually converging
- decide whether the system is ready to advance toward conductivity production

## Result
- overall status: `partial`
- artifact: [m10_2_1_summary.json](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_2_1_convergence_gate/m10_2_1_summary.json)
- gate decision: [m10_2_1_gate_decision.json](/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/m10_2_1_convergence_gate/m10_2_1_gate_decision.json)

## Protocol
- system: `small_oligomer_medium_100ps`
- 64 replicated oligomers, 384 atoms
- shared initial velocities injected into both engines
- GROMACS production restart uses the equilibration checkpoint
- 10 ps NPT equilibration + 100 ps NPT production
- timestep: `2 fs`

## Key Outcomes
### Temperature
- GROMACS mean: `299.92 K`
- LAMMPS mean: `298.64 K`
- cross-engine delta: `1.28 K`
- both engines: `pass`

### Density
- GROMACS mean density: `16.41 kg/m^3`
- LAMMPS mean density: `11.53 kg/m^3`
- cross-engine density diff: `29.74%`
- GROMACS density drift: `0.00170`
- LAMMPS density drift: `1.91332`
- GROMACS density status: `pass`
- LAMMPS density status: `fail`

## Interpretation
- Temperature control is no longer the bottleneck.
- Density/volume relaxation is still the bottleneck.
- The longer run did not fix the NPT convergence problem on the LAMMPS side.
- Therefore `M10.2.1` does **not** justify conductivity production.

## What This Report Proves
- the longer NPT path remains numerically stable
- shared-initial-condition handling and restart semantics are now aligned better than before
- temperature convergence is acceptable over 100 ps

## What This Report Does Not Prove
- density convergence
- NPT ensemble agreement strong enough for transport production
- conductivity readiness

## Conclusion
`M10.2.1` is **not a pass for production handoff**.

It is a longer-horizon diagnostic that narrows the bottleneck:
- exact/runtime path is not the current blocker
- temperature control is not the current blocker
- density/volume convergence remains the blocker before conductivity work
