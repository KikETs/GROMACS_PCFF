# Known Limitations (v1.0.0-rc1)

This document lists the known technical and physical limitations of the GROMACS-PCFF bridge as of the `rc1` release.

## 1. Charged System Density Parity
**Issue:** Mean density in charged systems can deviate by > 10% from LAMMPS references.
**Reason:** Differences in reciprocal-space virial stress implementations (PME in GROMACS vs PPPM in LAMMPS) are amplified in small, high-charge boxes.
**Mitigation:** 
- Use larger simulation boxes (> 4 nm) where practical.
- Use high electrostatics accuracy (`ewald-rtol = 1e-5`).
- Perform manual density drift analysis over at least 500 ps.

## 2. Transport Property Validation
**Issue:** Diffusion coefficients, ionic conductivity, and transference numbers are NOT validated.
**Reason:** These properties require multi-nanosecond sampling and explicit correlation analysis, which were outside the scope of the v1 ensemble-gate milestones.
**Status:** Out-of-Scope for `rc1`. Use for exploratory research only.

## 3. Chemistry Scope
**Issue:** The bridge only supports chemistries present in the `lammps_golden` validation corpus.
**Reason:** PCFF typing rules are deterministic and restricted to verified functional groups to ensure 100% parameter assignment accuracy.
**Status:** If your system fails to type, it is likely outside the validated chemical scope.

## 4. Performance Scaling
**Issue:** GPU acceleration for certain Class2 cross-terms may have limited support or different precision profiles.
**Reason:** PCFF-specific GROMACS kernels are optimized for accuracy first.
**Mitigation:** Use CPU-based `mdrun` for high-precision validation before moving to GPU-accelerated production.
