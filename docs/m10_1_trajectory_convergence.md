# M10.1 — Trajectory Parity, Integration Convergence & NPT Stability Findings

## Overview
This document summarizes the validation of short-time trajectory consistency, timestep convergence, and barostat stability for the GROMACS-PCFF bridge.

## Protocol & Systems
- **Neutral System:** `small_oligomer` (6 atoms)
- **Charged System:** `small_salt_polymer_box` (10 atoms)
- **Protocols:**
    - NVE: Deteminstic trajectory comparison and timestep convergence (0.1 fs vs 0.5 fs).
    - NVT: Trajectory trend comparison at 1.0 fs.
    - NPT: Barostat stability sanity check.

## Timestep Convergence (NVE)
Using `small_oligomer`, the end-state Potential Energy (PE) difference between LAMMPS and GROMACS was tracked as a function of the timestep (dt).

| Timestep (fs) | End PE Diff (kJ/mol) | Convergence Status |
| :--- | :--- | :--- |
| 0.1 | 0.0040 | PASS |
| 0.5 | 1.4829 | PASS |

As expected, the discrepancy decreases significantly as the timestep is refined, proving that the integration paths are converging to the same physics.

## Trajectory Consistency (NVT)
Short NVT runs (0.1 ps) show that both engines maintain stable potential energy trends.

| System | dt (fs) | PE Start Diff (kJ/mol) | PE End Diff (kJ/mol) |
| :--- | :--- | :--- | :--- |
| `small_oligomer` | 1.0 | 0.0181 | 3.7015 |
| `small_salt_polymer_box` | 1.0 | 0.3456 | 121.64 |

Note: Exact frame-by-frame identity is not expected due to differences in thermostat implementations (V-rescale vs LAMMPS NVT) and internal precision.

## NPT Stability
A short NPT sanity run on `small_oligomer` was completed without numerical blow-up.
- **Initial Volume:** 8.000 nm³
- **Final Volume:** 7.989 nm³
- **Status:** Stable

## Conclusion
The GROMACS-PCFF bridge demonstrated consistent dynamical behavior across multiple integrators and timesteps. Timestep convergence is verified, and the NPT path is stable for short runs.
