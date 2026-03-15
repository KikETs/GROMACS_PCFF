# M10.5 — Troubleshooting Guide

## 1. Density/Volume Drift
**Symptom:** Mean density is not reaching a plateau after 500 ps.
- **Cause:** Large ionic clusters or polymer chains taking long to relax.
- **Solution:**
  1. Extend equilibration to 1-2 ns.
  2. Increase `tau-p` to 5.0 or 10.0 to dampen pressure fluctuations.
  3. Ensure `compressibility` matches the physical state (default 4.5e-5 for liquids).

## 2. Unstable Charged Equilibration
**Symptom:** System blows up (NaN) or box expands infinitely.
- **Cause:** Large initial forces from ionic clashes or PME grid artifacts.
- **Solution:**
  1. Perform more rigorous minimization (`emtol = 10.0`).
  2. Use a smaller timestep (`dt = 0.0005`) for the first 50 ps of equilibration.
  3. Verify that net charge is exactly zero in `system.top`.

## 3. Potential Energy Disagreement
**Symptom:** PE differs from LAMMPS by > 5% for a neutral system.
- **Cause:** Incorrect mapping of cross-terms or 1-4 scaling.
- **Solution:**
  1. Check `[ defaults ]` in `system.top` for `comb-rule 4` and `gen-pairs yes`.
  2. Verify that `rep-pow 9.0` is present.
  3. Run a single-point energy check on a small fixture to isolate the term.

## 4. Restart/Checkpoint Handling
**Symptom:** Discontinuity in energy or pressure after restarting from a `.cpt` file.
- **Cause:** Mismatched `nstlist` or integrator state.
- **Solution:**
  1. Always use `gmx mdrun -cpi state.cpt` for exact restarts.
  2. Do not change `dt` or `cutoff-scheme` between restarts.
