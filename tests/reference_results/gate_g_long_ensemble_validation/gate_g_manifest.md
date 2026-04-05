# Gate G Long-Horizon Ensemble Validation

- Verdict: `PASS`
- Gate H allowed: `True`
- Long-run baseline: `Controlled CPU exact-r-RESPA long-run baselines derived from the Gate A mechanical path; no trajectory-identity claim is used.`
- Replica count per layout: `3`
- Equilibration / production: `20.0 ps / 40.0 ps`
- Sampling limitation note: Production windows are long enough for temperature/pressure/box/potential summaries, but too short and too small-box to support a defensible MSD/diffusion claim.

## Systems
- `small_oligomer` `nvt`: `PASS`
- `small_salt_polymer_box` `npt`: `PASS`
