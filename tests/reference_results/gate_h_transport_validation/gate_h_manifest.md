# Gate H Transport Validation

- Verdict: `FAIL`
- Production recommendation: `NO-GO`
- Replica count per layout: `3`
- Equilibration / production: `200.0 ps / 500.0 ps`
- Protocol caveat: Gate H reuses the mechanically validated exact-r-RESPA NVT path, but the current small fixtures sit outside the frozen TP0 transport scope (size/box/duration, and charged transport also lacks TP0-style long NPT conditioning).

## Systems
- `small_oligomer`: `FAIL`
  First failing observable: `oligomer_diffusivity_cm2_s`
- `small_salt_polymer_box`: `FAIL`
  First failing observable: `finite_sampling`
