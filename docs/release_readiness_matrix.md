# Release Readiness Matrix (v1.0.0-rc1)

This matrix defines the validated operational boundaries for the GROMACS-PCFF bridge as of release candidate 1.

| Component / System Class | Readiness Class | Validation Reference | Operational Constraint |
| :--- | :--- | :--- | :--- |
| **Topology Generation** | Production-Ready | M10.handoff_typing | Automatic parity with LAMMPS |
| **Bonded Parity (Class2)** | Production-Ready | PT8.2, PT8.3 | Verified Force/Energy identity |
| **Non-bonded Parity (LJ 9-6)** | Production-Ready | PT8.4 | Verified SixthPower mixing |
| **Neutral Dense Ensemble** | Production-Ready | M10.3 | Sub-1% density agreement |
| **Charged/Salt Ensemble** | **Qualified-Ready** | M10.4 | **Mandatory** density drift review |
| **Transport Properties** | **Out-of-Scope** | N/A | Not yet validated |
| **Conductivity / Transference** | **Out-of-Scope** | N/A | Use at own risk |
| **Unknown Chemistries** | **Out-of-Scope** | N/A | Restricted to `lammps_golden` |

## Readiness Definitions

### Production-Ready
- Extensive statistical evidence of parity with LAMMPS.
- Stable under nanosecond-class sampling.
- Recommended for routine production use.

### Qualified-Ready
- Force field mapping is correct (verified by Potential Energy parity).
- Ensemble observables (e.g., Density) show engine-specific numerical sensitivities.
- Requires manual expert review of equilibration and convergence before use.

### Out-of-Scope
- No formal validation performed.
- Substantive physical results cannot be guaranteed.
- Performance or correctness defects are expected.
