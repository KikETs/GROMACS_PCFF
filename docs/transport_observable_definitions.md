# TP0 — Transport Observable Definitions

## 1. Self-Diffusion Coefficient ($D$)

The self-diffusion coefficient for species $i$ is calculated using the Einstein relation in 3D:

$$D_i = \lim_{t \to \infty} \frac{1}{6t} \langle | \mathbf{r}_i(t) - \mathbf{r}_i(0) |^2 \rangle$$

where:
- $\langle \dots \rangle$ denotes an ensemble average over multiple time origins.
- $\mathbf{r}_i(t)$ are the **unwrapped** coordinates of the center-of-mass of species $i$ at time $t$.
- For polymers, $i$ refers to the entire chain's center-of-mass.
- For ions, $i$ refers to the individual ion.

### Analysis Requirements:
- **Unwrapping:** Essential for displacement calculation across periodic boundaries.
- **COM Drift Removal:** Mandatory to remove fictitious motion in the simulation box.
- **Fitting:** Performed on the Mean Squared Displacement (MSD) vs time, using a linear fit in the specified window (typically 20% to 80% of total time).

## 2. Ionic Conductivity ($\sigma_{cNE}$)

The primary frozen estimator for ionic conductivity is the Nernst-Einstein (cNE) approximation:

$$\sigma_{cNE} = \frac{e^2}{V k_B T} \sum_{j} n_j z_j^2 D_j$$

where:
- $e$ is the elementary charge.
- $V$ is the average system volume.
- $k_B$ is the Boltzmann constant.
- $T$ is the target temperature.
- $n_j$ is the number of ions of species $j$.
- $z_j$ is the charge (valence) of species $j$.
- $D_j$ is the self-diffusion coefficient of species $j$ (calculated via Einstein MSD).

**Note:** This estimator neglects cross-correlations between ions (distinct diffusion terms). It is used as the first validation pass for simplicity and reproducibility.

## 3. Transference Number ($t_{+, NE}$)

The transference number for the cation (e.g., $Li^+$) is defined using the self-diffusion coefficients:

$$t_{+, NE} = \frac{D_+}{D_+ + D_-}$$

where:
- $D_+$ is the cation diffusion coefficient.
- $D_-$ is the anion diffusion coefficient.

**Note:** Similar to $\sigma_{cNE}$, this is an approximation based on self-diffusion and serves as the baseline for transport-parity comparisons.
