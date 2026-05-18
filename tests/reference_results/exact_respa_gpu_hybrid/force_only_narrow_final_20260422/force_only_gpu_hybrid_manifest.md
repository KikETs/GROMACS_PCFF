# exact r-RESPA GPU Hybrid Force-Only Smoke

- Status: PASS
- gmx: `../../../../build_gateb_cuda/bin/gmx`
- GPU support: `CUDA`
- ntmpi / ntomp: `1` / `2`
- force tolerance: `0.001`
- aggregate force tolerance: `per-system force_tol * atom_count`

## Systems

- small_oligomer: `PASS`
- small_salt_polymer_box: `PASS`

## Claim Boundary

- This validates only the admitted force-only nonbonded GPU shape if status is PASS.
- It does not validate energy, virial, density, volume, transport, GPU update, or production readiness.
