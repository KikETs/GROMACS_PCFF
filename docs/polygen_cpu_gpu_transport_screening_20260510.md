# PolyGen Exact r-RESPA CPU/GPU Transport Screening, 2026-05-10

## Scope

This note records the latest GROMACS-only CPU/GPU parity and transport screening on the PolyGen PCFF exact r-RESPA full-run outputs.

This is not a LAMMPS-vs-GROMACS transport parity claim. The LAMMPS production dump is used only as the topology source for atom ids, molecule ids, atom types, masses, and charges in the transport analyzer.

Source outputs:

- Run root: `output/polygen_pcff_gromacs_initial_em_notebook`
- CPU lane: `gromacs_cpu_openmp`
- GPU lane: `gromacs_gpu_hybrid_strict_pme5`
- Analysis output: `output/polygen_cpu_gpu_parity_transport_20260510`

## Commands

Stage metric audit:

```bash
python tools/pcff_respa_parity/polygen_stage_metric_audit.py \
  --out-root output/polygen_pcff_gromacs_initial_em_notebook \
  --audit-out output/polygen_cpu_gpu_parity_transport_20260510/stage_metric_audit \
  --gmx build_gateb_double_cpu/bin/gmx_d \
  --lanes gromacs_cpu_openmp gromacs_gpu_hybrid_strict_pme5 \
  --include-all-stages \
  --relax-nonphysics-signature
```

Transport-ready artifact audit:

```bash
python tools/pcff_respa_parity/audit_polygen_transport_readiness.py \
  --out-root output/polygen_pcff_gromacs_initial_em_notebook \
  --report-dir output/polygen_cpu_gpu_parity_transport_20260510/transport_readiness \
  --expected-prod-chunks 50 \
  --expected-lammps-dump-stride 1000 \
  --expected-gmx-xtc-stride 4000 \
  --gromacs-lane gromacs_cpu_strict=gromacs_cpu_openmp \
  --gromacs-lane gromacs_gpu_strict=gromacs_gpu_hybrid_strict_pme5
```

Transport analysis:

```bash
python tools/pcff_respa_parity/analyze_polygen_transport.py \
  --root output/polygen_pcff_gromacs_initial_em_notebook \
  --outdir output/polygen_cpu_gpu_parity_transport_20260510/transport_analysis \
  --lanes cpu,gpu \
  --stride-ps 2.0 \
  --temperature 353.0 \
  --z 1.0 \
  --max-cluster 10 \
  --cluster-cutoff-angstrom 3.4 \
  --cluster-sample-stride 1 \
  --gmx-cpu build_gateb_double_cpu/bin/gmx_d \
  --gmx-gpu build_gateb_cuda/bin/gmx
```

## Artifact Readiness

`audit_polygen_transport_readiness.py` reports `PASS`.

Evidence:

- LAMMPS production dump manifest exists and records `stride_steps=1000`, `stride_ps=2.0`.
- LAMMPS dump fields include `id`, `mol`, `type`, `mass`, `q`, `x`, `y`, `z`, `ix`, `iy`, `iz`.
- CPU and GPU GROMACS lanes each have 50 nonempty production XTC chunks and 50 TPR chunks.
- GROMACS production MDP contract includes `exact-respa-levels=2`, `exact-respa-level2-factor=4`, `exact-respa-pair-level=2`, `exact-respa-kspace-level=2`, and `nstxout-compressed=4000`.

## CPU/GPU Stage-Metric Parity

The stage audit found 160 current GROMACS runtime records:

- CPU lane records: 80
- GPU lane records: 80
- Stale or missing records: 0
- Metrics blocked by runtime freshness: 0

Production mean comparison over the 50 production chunks:

| Metric | CPU mean | GPU mean | GPU - CPU | Relative vs CPU |
|---|---:|---:|---:|---:|
| `volume_nm3_mean` | 76.099968 | 76.099968 | 0 | 0% |
| `density_g_cm3_mean` | 1.3716504 | 1.3716504 | 0 | 0% |
| `temperature_k_mean` | 352.85034 | 352.94605 | 0.095719 | 0.0271% |
| `pressure_bar_mean` | 37.13048 | 0.437309 | -36.69317 | diagnostic only |
| `potential_kj_mol_mean` | -18233.292 | -18069.256 | 164.036 | 0.900% |
| `kinetic_energy_kj_mol_mean` | 31130.137 | 31138.582 | 8.44487 | 0.0271% |
| `total_energy_kj_mol_mean` | 12896.845 | 13069.326 | 172.481 | 1.337% |

Interpretation:

- Fixed-volume production density and volume match exactly between CPU and GPU because both production lanes use the same relaxed production box.
- Temperature and kinetic energy are close at the production-window mean level.
- Pressure is a noisy NVT diagnostic here. The absolute CPU/GPU mean difference is about 36.7 bar; the relative percentage is not useful because the GPU mean is near zero and the instantaneous pressure fluctuations are large.
- Energy endpoints should not be interpreted as bitwise parity. CPU double and GPU mixed precision trajectories diverge chaotically.

## Transport Screening

Transport analysis settings:

- Production trajectory length: 10 ns
- Frame count per GROMACS lane: 5001
- Stride: 2 ps
- Temperature: 353 K
- Formal charge magnitude: `z=1.0`
- Cluster cutoff for HTP-MD-style cNE0: 3.4 A
- `max_cluster=10`
- Ion topology: 7075 atoms, 100 cations, 100 anions

CPU/GPU transport deltas:

| Observable | CPU | GPU | GPU - CPU | Relative vs CPU |
|---|---:|---:|---:|---:|
| `D_cation_fit_cm2_s` | 9.62247e-08 | 9.14669e-08 | -4.75777e-09 | -4.94% |
| `D_anion_fit_cm2_s` | 1.13101e-07 | 1.23862e-07 | 1.07606e-08 | 9.51% |
| `NE_sigma_S_cm` | 0.00144877 | 0.00149032 | 4.15464e-05 | 2.87% |
| `NE_t_plus` | 0.459689 | 0.424779 | -0.034910 | -7.59% |
| `cNE0_htp_sigma_S_cm` | 0.00215901 | 0.00610284 | 0.00394382 | 182.67% |
| `cNE0_htp_t_plus` | 0.295760 | 0.281707 | -0.014053 | -4.75% |
| `cNE_lifetime_sigma_S_cm` | 0.00917098 | 0.00877588 | -0.00039510 | -4.31% |
| `Einstein_sigma_S_cm` | 0.000980011 | 0.000213741 | -0.000766270 | -78.19% |

MSD fit quality:

- CPU cation MSD fit `R^2=0.998081`; CPU anion MSD fit `R^2=0.991931`.
- GPU cation MSD fit `R^2=0.995625`; GPU anion MSD fit `R^2=0.998630`.

Interpretation:

- NE conductivity is CPU/GPU consistent at screening level over this 10 ns run: GPU is 2.87% above CPU.
- NE transference differs by 0.0349 absolute, inside the existing charged transport acceptance target of +/-0.05, but this is still only a GROMACS CPU/GPU screening comparison.
- HTP-MD-style cNE0 conductivity is not CPU/GPU stable over 10 ns. The endpoint diffusivity used by this estimator is more than 2x the MSD-fit diffusivity in both lanes, and the CPU/GPU cNE0 conductivity delta is 182.67%.
- The lifetime-tracked cNE and collective Einstein numbers are diagnostics from the same trajectory, not production estimators for the current claim boundary.

## Claim Boundary

Supported by this run:

- Latest GROMACS CPU/GPU production artifact readiness for the PolyGen exact r-RESPA lanes.
- Latest GROMACS CPU/GPU stage-metric screening parity for production density, volume, temperature, and mean energy behavior.
- Latest GROMACS CPU/GPU NE transport screening over 10 ns.

Not supported by this run:

- LAMMPS-vs-GROMACS charged transport parity.
- Production cNE0 parity.
- Final charged transport readiness. `docs/transport_protocol_freeze.md` requires charged transport production of at least 20 ns, while this screening run is 10 ns.
- Bitwise CPU/GPU trajectory identity.
