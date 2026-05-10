# Gate I Charged Long-NPT Conditioning

- Status: `DECLARED_PENDING_EXECUTION`
- System: `gate_h_dense_salt_polymer_2x2x2`
- Scope: CPU-only single-rank exact-r-RESPA charged large/medium long-NPT conditioning gate
- Contract: `/home/kiket/Desktop/test/GROMACS_PCFF/output/exact_respa_update_probe/gate_i_contract_pin_fastpath_prepare_20260421/gate_i_contract.json`
- Replicas / horizon: `3` / `3000.0 ps + 1000.0 ps`

## Non-Claims
- A declared Gate I contract is not a passed gate.
- A Gate I PASS still does not imply conductivity-production readiness.
- A Gate I PASS would still not imply LAMMPS-vs-GROMACS transport parity.
- A Gate I PASS would still not imply TP0-scale production length or uncertainty closure.
- A common preconditioned starting structure is a conditioning input only; it is not production handoff approval.

## Failure Reasons
- Checked-in repository now freezes the Gate I contract, but no completed CPU-only exact long-NPT campaign has been checked in yet.

