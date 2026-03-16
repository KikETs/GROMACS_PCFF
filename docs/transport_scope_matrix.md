# TP0 — Transport Scope Matrix

## 1. Overview
This matrix defines the boundaries of transport validation. Items marked "Deferred" or "Out-of-Scope" are not covered by the current TP0 protocol and must be re-evaluated in later milestones.

## 2. Validation Scope Matrix

| System / Observable | Diffusion ($D$) | Conductivity ($\sigma$) | Transference ($t_+$) | Viscosity ($\eta$) |
| :--- | :--- | :--- | :--- | :--- |
| **Neutral Melt** | **VALIDATED** | N/A | N/A | **DEFERRED** |
| **Charged Salt-in-Polymer** | **VALIDATED** | **VALIDATED (cNE)** | **VALIDATED (cNE)** | **DEFERRED** |
| **Infinite Dilution** | **VALIDATED** | **DEFERRED** | **DEFERRED** | **DEFERRED** |
| **Net-Charged Box** | **OUT-OF-SCOPE** | **OUT-OF-SCOPE** | **OUT-OF-SCOPE** | **OUT-OF-SCOPE** |
| **Ionic Liquids** | **DEFERRED** | **DEFERRED** | **DEFERRED** | **DEFERRED** |

## 3. Method Scope

| Method / Estimator | Status | Reason |
| :--- | :--- | :--- |
| **Einstein (MSD)** | **FROZEN** | Primary estimator for $D$ and $\sigma_{cNE}$. |
| **Green-Kubo (GK)** | **DEFERRED** | High sensitivity to sampling length and noise. |
| **Einstein-Helfand (EH)** | **DEFERRED** | Requires long trajectories for collective terms. |
| **NEMD (Wall)** | **OUT-OF-SCOPE** | Not part of current equilibrium validation. |
| **Periodic Perturbation** | **OUT-OF-SCOPE** | Requires specific code paths in GROMACS. |

## 4. Parameter Sensitivity
The protocol freezes the following sensitivities for subsequent analysis:
- **PME Tolerance:** $10^{-5}$ is the baseline. Sensitivities at $10^{-4}$ or $10^{-6}$ are **Deferred**.
- **Timestep:** 1 fs is the baseline. 2 fs or RESPA are **Deferred**.
- **Cutoff:** 0.9 nm is the baseline. 1.0 nm or 1.2 nm are **Deferred**.
- **Ensemble:** NVT is the baseline. NPT production is **Deferred**.
