# Gate H-Neutral Transport Validation

- Verdict: `PASS`
- Scope: Official Gate H-neutral only: large-neutral scaffold self-diffusion on the standalone exact r-RESPA full-GPU path.
- Scope-limited recommendation: `GO`
- Overall transport production recommendation: `NO-GO`
- Replica count per layout: `2`
- Equilibration / production: `10.0 ps / 100.0 ps`

## Observable
- `oligomer_diffusivity_cm2_s`: `stochastic` / passes=`True`
- CPU mean: `0.0001154565` cm^2/s
- GPU mean: `0.000113118` cm^2/s
- Mean diff: `-2.3385e-06` cm^2/s
- Combined uncertainty: `1.136651454e-05` cm^2/s

## Boundaries
- Full official Gate H remains out of scope: `FAIL`
- Charged conductivity/cNE are not covered by this manifest.
