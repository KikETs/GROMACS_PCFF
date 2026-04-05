# Gate H Transport Validation

- Verdict: `FAIL`
- Production recommendation: `NO-GO`
- Replica count per layout: `2`
- Equilibration / production: `100.0 ps / 500.0 ps`
- Protocol caveat: Gate H reuses the mechanically validated exact-r-RESPA NVT path; small mode is out of TP0 transport scope by size/box/duration, while large mode fixes size/box but still requires TP0-scale production length and charged-side long NPT density conditioning.

## Systems
- `gate_h_dense_oligomer_2x2x2`: `FAIL`
  First failing observable: `protocol_scope`
- `gate_h_dense_salt_polymer_2x2x2`: `FAIL`
  First failing observable: `protocol_scope`
