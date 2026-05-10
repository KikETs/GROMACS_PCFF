# TP0 — Transport Observable Definitions

## Current Implementation Note

The current PolyGen analysis script is
`tools/pcff_respa_parity/analyze_polygen_transport.py`.

For the 2026-05-10 PolyGen exact r-RESPA screening report, the script computes:

- drift-removed molecular COM MSD diffusion
- Nernst-Einstein conductivity and transference from the MSD-fit diffusion
- HTP-MD-style `cNE0` using atom-level cluster populations and endpoint raw atom diffusivity
- diagnostic lifetime-tracked cNE
- diagnostic collective Einstein conductivity

Only the MSD-fit NE values are currently stable enough for CPU/GPU screening language on the 10 ns run. The HTP-MD-style `cNE0` endpoint estimator is reported, but the 2026-05-10 screening run shows that it is not production-stable over 10 ns.

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

## 2. Nernst-Einstein Conductivity (`sigma_NE`)

The primary screening estimator for ionic conductivity is the Nernst-Einstein approximation:

$$\sigma_{NE} = \frac{e^2}{V k_B T} \sum_{j} n_j z_j^2 D_j$$

where:
- $e$ is the elementary charge.
- $V$ is the average system volume.
- $k_B$ is the Boltzmann constant.
- $T$ is the target temperature.
- $n_j$ is the number of ions of species $j$.
- $z_j$ is the charge (valence) of species $j$.
- $D_j$ is the self-diffusion coefficient of species $j$ (calculated via Einstein MSD).

**Note:** This estimator neglects cross-correlations between ions (distinct diffusion terms). It is used as the first validation pass for simplicity and reproducibility.

In `analyze_polygen_transport.py`, this is reported under `NE_msd_fit` and is computed from drift-removed cation/anion molecular COM MSD fits using the 20-80% lag-time window.

## 3. HTP-MD-Style `cNE0`

The local HTP-MD-style `cNE0` estimator in
`tools/pcff_respa_parity/analyze_polygen_transport.py` is not the same as the MSD-fit NE estimator.

Implementation basis:

- cluster population matrix from wrapped ion atom coordinates
- cluster cutoff: 3.4 A in the current PolyGen screening command
- cation trace atom: type `90`
- anion reference atom: type `93`
- anion trace atom types: `93`, `94`, `95`
- endpoint raw atom diffusivity from the first and last frames
- `max_cluster=10` in the current PolyGen screening command

Because it uses endpoint diffusivity, this estimator is sensitive to finite trajectory length. The 2026-05-10 PolyGen CPU/GPU 10 ns screening reports a `cNE0_htp_sigma_S_cm` CPU/GPU relative delta of `182.67%`, so it must not be used as production cNE0 parity evidence without longer trajectories and block analysis.

## 4. Transference Number ($t_{+, NE}$)

The transference number for the cation (e.g., $Li^+$) is defined using the self-diffusion coefficients:

$$t_{+, NE} = \frac{D_+}{D_+ + D_-}$$

where:
- $D_+$ is the cation diffusion coefficient.
- $D_-$ is the anion diffusion coefficient.

**Note:** Similar to $\sigma_{NE}$, this is an approximation based on self-diffusion and serves as the baseline for transport-parity comparisons.
