# Gate I Charged Long-NPT Conditioning

- Status: `PASS`
- System: `gate_h_dense_salt_polymer_2x2x2`
- Scope: CPU-only single-rank exact-r-RESPA charged large/medium long-NPT conditioning gate
- Contract: `/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_i_charged_long_npt_conditioning_eq3000_prod1000_ntmpi1_ntomp12_pinstride2_owner_fallback_updatefastpath_ldseed84001_20260421/gate_i_contract.json`
- Replicas / horizon: `3` / `3000.0 ps + 1000.0 ps`

## Non-Claims
- A declared Gate I contract is not a passed gate.
- A Gate I PASS still does not imply conductivity-production readiness.
- A Gate I PASS would still not imply LAMMPS-vs-GROMACS transport parity.
- A Gate I PASS would still not imply TP0-scale production length or uncertainty closure.
- A common preconditioned starting structure is a conditioning input only; it is not production handoff approval.

## Failure Reasons
- None

