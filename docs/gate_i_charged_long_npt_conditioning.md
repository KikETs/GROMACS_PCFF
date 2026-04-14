# Gate I Charged Long-NPT Conditioning

Gate I is the concrete next gate for the remaining CPU-only exact `r-RESPA` blocker.

Its purpose is narrow:

- close or fail the charged large/medium long-`NPT` density/volume conditioning blocker on `gate_h_dense_salt_polymer_2x2x2`
- freeze predeclared density/volume convergence criteria before any later TP0-scale transport campaign
- produce a conditioned-state handoff artifact without implying transport readiness

It is not a transport gate.
It is not a conductivity-production gate.
It is not a LAMMPS-vs-GROMACS transport parity gate.

## Scope

Gate I is frozen to:

- single-rank
- CPU-only
- standalone exact `r-RESPA`
- charged `gate_h_dense_salt_polymer_2x2x2`
- long-`NPT`

Primary machine-readable artifacts:

- [Gate I contract](../tests/reference_results/gate_i_charged_long_npt_conditioning/gate_i_contract.json)
- [Gate I manifest](../tests/reference_results/gate_i_charged_long_npt_conditioning/gate_i_manifest.json)

The checked-in manifest is currently a pending-execution artifact.
That is intentional.
It proves the gate is predeclared and bounded, but not yet passed.

## Frozen Criteria

Gate I freezes these primary pass metrics:

- density mean relative block drift
- density worst-replica relative block drift
- density cross-replica relative span
- volume mean relative block drift
- volume worst-replica relative block drift
- volume cross-replica relative span

Temperature is a supporting stability check, not the blocker itself.
Pressure, potential, and box dimensions are supporting diagnostics.

The public contract also freezes conditioned-state handoff rules:

- select the replica whose density and volume means are closest to the aggregate conditioning center
- require checked-in `prod.gro`, `prod.cpt`, `prod.tpr`, and `replica_summary.json`

## Runtime Cost

The current cost is not trivial.

The checked-in charged large-scaffold CPU exact run in [gate_h_transport_validation_large_medium/gate_h_dense_salt_polymer_2x2x2/replica_01/cpu/prod.log](../tests/reference_results/gate_h_transport_validation_large_medium/gate_h_dense_salt_polymer_2x2x2/replica_01/cpu/prod.log) reports about `2.647 ns/day` or `9.068 hour/ns`.

Using the current Gate I default horizon:

- equilibration: `250 ps`
- production: `1000 ps`
- replicas: `3`

The rough serial wall-clock estimate is about `34` hours.
That estimate is operational, not scientific evidence.
It explains why the repository can honestly freeze the gate before checking in a finished campaign.

## Non-Claims

Even if Gate I passes, it still does not imply:

- TP0-scale production length
- conductivity-production readiness
- transport uncertainty closure
- transport linearity closure
- LAMMPS-vs-GROMACS transport parity

Gate I only addresses the density/volume conditioning blocker.
