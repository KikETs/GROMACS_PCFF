# Gate I Charged Long-NPT Conditioning

Gate I was the concrete gate for the former CPU-only exact `r-RESPA`
charged long-NPT density/volume conditioning blocker. It now has a dated PASS
artifact.

Its purpose is narrow:

- record the charged large/medium long-`NPT` density/volume conditioning result on `gate_h_dense_salt_polymer_2x2x2`
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
- [Gate I PASS contract, 3000 ps + 1000 ps x 3 replicas](../tests/reference_results/gate_i_charged_long_npt_conditioning_eq3000_prod1000_ntmpi1_ntomp12_pinstride2_owner_fallback_updatefastpath_ldseed84001_20260421/gate_i_contract.json)
- [Gate I PASS manifest, 3000 ps + 1000 ps x 3 replicas](../tests/reference_results/gate_i_charged_long_npt_conditioning_eq3000_prod1000_ntmpi1_ntomp12_pinstride2_owner_fallback_updatefastpath_ldseed84001_20260421/gate_i_manifest.json)

The generic checked-in manifest remains the frozen predeclaration artifact.
The dated `eq3000_prod1000_ntmpi1_ntomp12_pinstride2_owner_fallback_updatefastpath_ldseed84001_20260421`
artifact is the completed CPU-only exact-r-RESPA Gate I PASS evidence.

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

The completed local 9900X Gate I PASS campaign used:

- equilibration: `3000 ps`
- production: `1000 ps`
- replicas: `3`
- runtime shape: `ntmpi 1`, `ntomp 12`, `pin on`, `pinstride 2`, CPU-only PP/PME/bonded/update

Observed throughput was about `63.3-63.6 ns/day` during equilibration and
`65.6-67.0 ns/day` during production on the local host. This is performance
evidence for this audited host/runtime shape only.

## Completed Gate Result

The dated PASS campaign closes the Gate I density/volume conditioning blocker
within the predeclared criteria:

- density mean relative block drift: `0.008912910771277587`
- density worst-replica relative block drift: `0.012652646434178126`
- density cross-replica relative span: `0.023858714505661974`
- volume mean relative block drift: `0.008848600187149268`
- volume worst-replica relative block drift: `0.012740374517488726`
- volume cross-replica relative span: `0.023728645886153563`
- temperature mean absolute error: `11.661190940078654 K`

The selected conditioned-state handoff replica is `replica_03`.

## Non-Claims

Even if Gate I passes, it still does not imply:

- TP0-scale production length
- conductivity-production readiness
- transport uncertainty closure
- transport linearity closure
- LAMMPS-vs-GROMACS transport parity

Gate I only addresses the density/volume conditioning blocker.
