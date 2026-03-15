# M10.1 — Trajectory Parity, Integration Convergence & NPT Stability Validation Report

## Overview
This report documents the validation of trajectory-level consistency, integration convergence, and barostat stability for the GROMACS-PCFF bridge.

## Validated Outcomes
1.  **Timestep Convergence:**
    - Established that end-state parity between LAMMPS and GROMACS improves as the timestep is reduced (from 0.5 fs to 0.1 fs).
    - Status: PASS
2.  **Short-Time Trajectory Consistency:**
    - Verified stable energy trends for NVE and NVT protocols on neutral systems.
    - Status: PASS
3.  **NPT Stability Gate:**
    - Completed a 0.1 ps NPT sanity run without numerical blow-up or fatal errors.
    - Status: PASS
4.  **Charged System Dynamical Sanity:**
    - Verified stable NVT execution for a salt-containing polymer box using PME.
    - Status: PASS

## Technical Details
- **NVE Protocol:** Deterministic comparison without thermostats.
- **NVT Protocol:** V-rescale thermostat (GMX) vs. Nose-Hoover (LAMMPS).
- **NPT Protocol:** Berendsen barostat (GMX) for short-time stability.
- **Observables:** Potential Energy, Temperature, Volume.

## Remaining Gaps / Not Validated
- **Long-Timescale Equivalence:** Thermodynamic averages (e.g. density, specific heat) require much longer runs.
- **Barostat Parity:** Quantitative agreement between GROMACS Parrinello-Rahman and LAMMPS barostats was not tested.
- **System Size Scale-up:** Only small systems (6-10 atoms) were used.

## Artifacts Produced
- `tools/run_m10_1_trajectory_gate/run_m10_1.py`: Trajectory parity runner.
- `tests/reference_results/m10_1_trajectory_gate/`: Directory containing per-protocol comparison reports and logs.

## Conclusion
Milestone M10.1 is successfully completed. The GROMACS-PCFF bridge is dynamically consistent and stable under pressure coupling for small supported systems. PT8.3 is officially considered PASS.
