# Gate I Charged Long-NPT Conditioning

- Status: `FAIL`
- System: `gate_h_dense_salt_polymer_2x2x2`
- Scope: CPU-only single-rank exact-r-RESPA charged large/medium long-NPT conditioning gate
- Contract: `/home/kiket/Desktop/test/GROMACS_PCFF/tests/reference_results/gate_i_charged_long_npt_conditioning/gate_i_contract.json`
- Replicas / horizon: `3` / `250.0 ps + 1000.0 ps`

## Non-Claims
- A declared Gate I contract is not a passed gate.
- A Gate I PASS still does not imply conductivity-production readiness.
- A Gate I PASS would still not imply LAMMPS-vs-GROMACS transport parity.
- A Gate I PASS would still not imply TP0-scale production length or uncertainty closure.

## Failure Reasons
- Density mean relative block drift 0.051429 exceeds 0.050000.
- Density worst-replica relative block drift 0.103388 exceeds 0.080000.
- Volume mean relative block drift 0.053646 exceeds 0.050000.
- Volume worst-replica relative block drift 0.108671 exceeds 0.080000.

