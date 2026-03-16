# TP1.3 Root-Cause Analysis Report

## 1. Executive Summary
Milestone TP1.3 performed a structured diagnostic analysis of the thermal runaway observed in the `dense_salt_polymer` (270-atom Na/Cl) system. Despite multiple stabilization attempts—including reducing the timestep to 0.5 fs, using NVT-only ensembles, and applying a very strong thermostat ($\tau_t = 0.01$ ps)—all trials exhibited systematic heating to $> 700$ K.

## 2. Baseline Capture (TRL-0)
- **Settings:** 1.0 fs, NPT, V-rescale (300K).
- **Behavior:** Temperature increased from 300K to 700K+ within 500 ps.
- **PE/Coulomb:** Reciprocal Coulomb energy showed significant instability.

## 3. Diagnostic Trial Results

| Trial | Variable | Outcome | Max Temp | Stability Improvement? |
| :--- | :--- | :--- | :--- | :--- |
| **TRL-1** | 0.5 fs timestep | **FAILED** | 814 K | None |
| **TRL-2** | NVT Ensemble | **FAILED** | 805 K | None |
| **TRL-3** | NVT + 0.5 fs | **FAILED** | 802 K | None |
| **TRL-5** | Cut-off (No PME) | **FAILED** | 1070 K | **Regressed** (Severe blow-up) |
| **TRL-6** | Strong Thermostat | **FAILED** | 796 K | Minimal |

## 4. Root-Cause Synthesis
The dominant instability driver is **NOT** the timestep, the barostat, or the PME implementation in isolation. 

### 4.1 Implementation implementation
The systematic heating across all diagnostic axes points to a fundamental issue in the custom GROMACS implementation of PCFF/Class2 function types (11, 13). Specifically:
- **Energy-Force Consistency:** The thermostat cannot counteract the energy injection, suggesting a possible sign error or scaling issue in the force/virial calculation.
- **Non-bonded Interactions:** The severe regression in TRL-5 (Cut-off only) suggests that the real-space non-bonded interactions (Function Type 11 for bonds/angles/LJ) are generating unphysical forces.

## 5. Final Decision
**Status: STILL BLOCKED.**
The authoritative charged system is not equilibratable under the current implementation. TP1 cannot proceed until the bridge implementation or the GROMACS custom kernel is repaired.
