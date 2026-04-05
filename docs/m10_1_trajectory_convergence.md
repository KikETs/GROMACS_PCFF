# M10.1 — Deterministic NVE Gate and Short-Time Ensemble Diagnostics

## Overview
M10.1 is not a general ensemble-parity milestone.

Its corrected scope is:
- deterministic NVE parity
- timestep convergence on deterministic paths
- short-time NVT/NPT diagnostics

Ensemble-level parity remains owned by M10.2.

## Protocol & Systems
- **Neutral system:** `small_oligomer` (6 atoms)
- **Charged system:** `small_salt_polymer_box` (10 atoms)
- **Protocols:**
  - `NVE`: deterministic comparison and timestep convergence
  - `NVT`: short-time thermostat diagnostic only
  - `NPT`: short-time barostat diagnostic only

## Deterministic NVE Gate
M10.1 now uses PE and pressure deltas rather than absolute PE offsets.

| System | Timestep (fs) | PE Delta Diff (kJ/mol) | Pressure Delta Diff | Status |
| :--- | :--- | :--- | :--- | :--- |
| `small_oligomer` | 0.1 | 2.0723 | 43.4710 | PASS |
| `small_oligomer` | 0.5 | 0.8554 | 95.5157 | PASS |
| `small_salt_polymer_box` | 0.1 | 3.2372 | 56.0051 | PASS |

These are the only blocking gates in M10.1.

## NVT/NPT Diagnostics
These runs are useful for triage, but they are not blocking parity gates because the engines use different thermostat/barostat families.

| System | Protocol | Current status |
| :--- | :--- | :--- |
| `small_oligomer` | NVT 1.0 fs | diagnostic fail |
| `small_oligomer` | NPT 1.0 fs | diagnostic fail |
| `small_salt_polymer_box` | NVT 1.0 fs | diagnostic fail |

The charged-system NVT path remains the strongest non-deterministic blocker before conductivity handoff.

## Interpretation
- M10.1 can justify deterministic dynamic handoff only.
- M10.1 cannot justify NVT/NPT ensemble parity.
- Conductivity production should not start until the charged ensemble path is acceptable in M10.2 or a later gate.

## Conclusion
M10.1 should be read as `deterministic NVE pass, ensemble diagnostics pending`.
Any stronger conclusion is not supported by the current evidence.
