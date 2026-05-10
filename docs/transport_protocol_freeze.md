# TP0 — Transport Property Protocol Freeze

## 1. Introduction
This document defines the frozen simulation and analysis protocol for the Transport Property Parity validation phase. All subsequent transport-related milestones must adhere to these specifications to ensure comparability and auditability.

## 2. Supported System Classes
Only the following system classes are in-scope for TP0-frozen validation:
- **Neutral Dense Systems:** Pure liquids and neutral polymer/oligomer melts (net charge = 0.0).
- **Charged/Salt Systems:** Polymer electrolytes with dissolved salts (e.g., LiTFSI in PEO/PC).
- **Size Class:** Minimum 1,000 atoms; minimum box dimension > 3.0 nm to minimize finite-size effects on diffusion.

## 3. Workflow Conditions
The following conditions are frozen as the mandatory minimum for transport validation:

### 3.1 Simulation Stages
1.  **Minimization:** Steepest descent (GROMACS `steep`) until $F_{max} < 100$ kJ/mol/nm.
2.  **Equilibration (NPT):** 
    - Duration: $\ge 2$ ns (Neutral), $\ge 5$ ns (Charged).
    - Thermostat: V-rescale ($\tau_t = 0.5$ ps).
    - Barostat: Berendsen ($\tau_p = 5.0$ ps) followed by Parrinello-Rahman ($\tau_p = 10.0$ ps).
3.  **Production (NVT or NPT):**
    - **Default:** NVT (using average volume from the last 1 ns of NPT).
    - **Duration:** $\ge 10$ ns (Neutral), $\ge 20$ ns (Charged).
    - **Timestep:** 1.0 fs (fixed).
    - **Thermostat:** V-rescale ($\tau_t = 0.5$ ps).

### 3.2 Trajectory Output
- **Energy Stride:** $\le 1$ ps.
- **Coordinate Stride:** $\le 10$ ps (must be frequent enough for MSD slope resolution).
- **Format:** `.trr` (full precision) preferred for transport analysis to avoid quantization noise.

## 4. Electrostatics & Reciprocal Space
- **Method:** PME (GROMACS) vs PPPM (LAMMPS).
- **Real-space Cutoff:** 0.9 nm (fixed).
- **PME Tolerance:** $10^{-5}$ (fixed).
- **Grid Spacing:** $\le 0.12$ nm.

## 5. Observable Definitions (Primary)
- **Self-Diffusion Coefficient ($D$):** Calculated via Einstein relation from Mean Squared Displacement (MSD).
- **Ionic Conductivity ($\sigma$):** 
    - **Primary:** Nernst-Einstein (cNE) approximation (ignores ion-ion correlations).
    - **Secondary (Deferred):** Green-Kubo (GK) or Einstein-Helfand (EH) for true collective conductivity.
- **Transference Number ($t_+$):** Based on self-diffusion coefficients ($t_+ = \frac{D_+}{D_+ + D_-}$).

## 6. Analysis Policy
- **Unwrap:** Mandatory coordinate unwrapping before MSD calculation.
- **COM Drift:** Removal of center-of-mass motion is mandatory.
- **Fitting Window:** 20% to 80% of the production trajectory (excluding initial 2 ns to ensure relaxation from equilibration-to-production transition).
- **Uncertainty:** 5-block averaging over the production window.

## 6.1 Current PolyGen Screening Exception

The 2026-05-10 PolyGen exact r-RESPA CPU/GPU analysis is a 10 ns GROMACS-only screening run, not a TP0 charged transport sign-off.

It may be used to discuss:

- CPU/GPU production artifact readiness
- CPU/GPU stage-metric screening parity
- 10 ns NE screening consistency

It must not be used to claim:

- charged transport readiness
- LAMMPS-vs-GROMACS charged transport parity
- production cNE0 parity

Reason: the charged production duration requirement above is `>= 20 ns`, and the HTP-MD-style cNE0 endpoint estimator is not stable over the 10 ns screening trajectory.

## 7. Out-of-Scope
- Multi-microsecond dynamics.
- Non-equilibrium molecular dynamics (NEMD) for viscosity or conductivity.
- Systems with net non-zero total charge.
- Dielectric constant validation.
