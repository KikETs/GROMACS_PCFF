# PolyGen 20 ns parity and transport analysis

Date: 2026-05-11 KST

## Scope

- Source run root: `output/polygen_pcff_gromacs_initial_em_notebook`
- Analysis output: `output/polygen_20ns_parity_transport_20260511`
- Lanes: LAMMPS OpenMP, GROMACS CPU OpenMP, GROMACS CPU+GPU strict PME5
- Transport settings: 20 ns, 2 ps frame stride, `z=1.0`, `max_cluster=10`, cluster cutoff `3.4 A`, temperature `353 K`

## Artifact and runtime status

- Transport readiness: `PASS`
- Runtime records: `260`; stale/missing runtime records: `0`
- Production chunks found: LAMMPS `100`, GROMACS CPU `100`, GROMACS GPU `100`

## Production parity means

| lane                           |   chunks |   volume_nm3_mean |   density_g_cm3_mean |   temperature_k_mean |   pressure_bar_mean |   potential_kj_mol_mean |   kinetic_energy_kj_mol_mean |   total_energy_kj_mol_mean |
|:-------------------------------|---------:|------------------:|---------------------:|---------------------:|--------------------:|------------------------:|-----------------------------:|---------------------------:|
| lammps_openmp                  |      100 |           76.1002 |              1.37165 |              353.261 |           -13.4636  |                -18133.1 |                      31166.4 |                    13033.3 |
| gromacs_cpu_openmp             |      100 |           76.1    |              1.37165 |              353.04  |            -5.75485 |                -18199.7 |                      31146.9 |                    12947.2 |
| gromacs_gpu_hybrid_strict_pme5 |      100 |           76.1    |              1.37165 |              352.93  |           -21.3985  |                -18158   |                      31137.1 |                    12979.1 |

Key interpretation: volume and density are matched to about `0.0003%`. Temperature means are within `0.094%` of LAMMPS. Mean potential/total energy differences are below `0.7%` versus LAMMPS. NVT pressure mean is noisy and near zero, so percent pressure deltas are not meaningful; use absolute bar-scale deltas instead.

## Production mean deltas

| comparison                                              | metric                     |    reference |        value |         delta |   pct_delta_vs_abs_reference |
|:--------------------------------------------------------|:---------------------------|-------------:|-------------:|--------------:|-----------------------------:|
| gromacs_cpu_openmp_minus_lammps_openmp                  | volume_nm3_mean            |     76.1002  |     76.1     |  -0.000200644 |                 -0.000263658 |
| gromacs_cpu_openmp_minus_lammps_openmp                  | density_g_cm3_mean         |      1.37165 |      1.37165 |   2.66408e-06 |                  0.000194225 |
| gromacs_cpu_openmp_minus_lammps_openmp                  | temperature_k_mean         |    353.261   |    353.04    |  -0.221223    |                 -0.0626231   |
| gromacs_cpu_openmp_minus_lammps_openmp                  | pressure_bar_mean          |    -13.4636  |     -5.75485 |   7.70871     |                 57.2561      |
| gromacs_cpu_openmp_minus_lammps_openmp                  | potential_kj_mol_mean      | -18133.1     | -18199.7     | -66.5677      |                 -0.367105    |
| gromacs_cpu_openmp_minus_lammps_openmp                  | kinetic_energy_kj_mol_mean |  31166.4     |  31146.9     | -19.5556      |                 -0.0627458   |
| gromacs_cpu_openmp_minus_lammps_openmp                  | total_energy_kj_mol_mean   |  13033.3     |  12947.2     | -86.1233      |                 -0.660795    |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | volume_nm3_mean            |     76.1002  |     76.1     |  -0.000200644 |                 -0.000263658 |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | density_g_cm3_mean         |      1.37165 |      1.37165 |   2.66408e-06 |                  0.000194225 |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | temperature_k_mean         |    353.261   |    352.93    |  -0.331437    |                 -0.0938222   |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | pressure_bar_mean          |    -13.4636  |    -21.3985  |  -7.93498     |                -58.9367      |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | potential_kj_mol_mean      | -18133.1     | -18158       | -24.8942      |                 -0.137286    |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | kinetic_energy_kj_mol_mean |  31166.4     |  31137.1     | -29.2793      |                 -0.0939449   |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | total_energy_kj_mol_mean   |  13033.3     |  12979.1     | -54.1734      |                 -0.415654    |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | volume_nm3_mean            |     76.1     |     76.1     |   0           |                  0           |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | density_g_cm3_mean         |      1.37165 |      1.37165 |   0           |                  0           |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | temperature_k_mean         |    353.04    |    352.93    |  -0.110214    |                 -0.0312187   |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | pressure_bar_mean          |     -5.75485 |    -21.3985  | -15.6437      |               -271.835       |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | potential_kj_mol_mean      | -18199.7     | -18158       |  41.6735      |                  0.228979    |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | kinetic_energy_kj_mol_mean |  31146.9     |  31137.1     |  -9.72362     |                 -0.0312186   |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | total_energy_kj_mol_mean   |  12947.2     |  12979.1     |  31.9499      |                  0.246771    |

## Transport summary

| lane                           |   frames |   time_ns |   D_cation_fit_cm2_s |   D_anion_fit_cm2_s |   NE_sigma_S_cm |   NE_t_plus |   cNE0_htp_sigma_S_cm |   cNE0_htp_t_plus |   Einstein_sigma_S_cm |
|:-------------------------------|---------:|----------:|---------------------:|--------------------:|----------------:|------------:|----------------------:|------------------:|----------------------:|
| lammps_openmp                  |    10001 |        20 |          6.17432e-08 |         1.13047e-07 |      0.00120975 |    0.353242 |           0.000758102 |          0.174194 |           0.000955514 |
| gromacs_cpu_openmp             |    10001 |        20 |          8.21361e-08 |         1.46191e-07 |      0.00158028 |    0.359729 |           0.00251169  |          0.289326 |           0.00220567  |
| gromacs_gpu_hybrid_strict_pme5 |    10001 |        20 |          8.52538e-08 |         1.26706e-07 |      0.001467   |    0.402217 |           0.0199661   |          0.349769 |           0.00057955  |

## cNE0 estimator sensitivity

The HTP-MD-style raw endpoint `cNE0_htp_*` values are retained as a diagnostic, but the production comparison should use a separate MSD-fit estimator because the raw endpoint estimator absorbs residual center-of-mass drift. Two MSD-fit variants were computed:

- `cNE0_pop_molecular_MSDfit_*`: HTP-MD-style cluster population matrix, but molecular COM MSD-fit diffusivities, matching the NE diffusion basis.
- `cNE0_pop_atom90_93_MSDfit_*`: HTP-MD-style cluster population matrix, but type-90/type-93 atom MSD-fit diffusivities after ion-mass drift removal. This isolates endpoint sensitivity while preserving the HTP atom selection.

| lane                           |   raw_endpoint_cNE0_S_cm |   raw_endpoint_t_plus |   molecular_MSDfit_cNE0_S_cm |   molecular_MSDfit_t_plus |   atom90_93_MSDfit_cNE0_S_cm |   atom90_93_MSDfit_t_plus |
|:-------------------------------|-------------------------:|----------------------:|------------------------------:|--------------------------:|------------------------------:|--------------------------:|
| lammps_openmp                  |              0.000758102 |              0.174194 |                   0.000647542 |                  0.166429 |                   0.000647853 |                  0.166210 |
| gromacs_cpu_openmp             |              0.00251169  |              0.289326 |                   0.000842304 |                  0.191184 |                   0.000842127 |                  0.191281 |
| gromacs_gpu_hybrid_strict_pme5 |              0.0199661   |              0.349769 |                   0.000747041 |                  0.219217 |                   0.000746337 |                  0.219727 |

| lane                           |   D_mol_cat_MSDfit_cm2_s |   D_mol_an_MSDfit_cm2_s |   D_atom90_MSDfit_drift_removed_cm2_s |   D_atom93_MSDfit_drift_removed_cm2_s |   D_atom90_MSDfit_R2 |   D_atom93_MSDfit_R2 |
|:-------------------------------|-------------------------:|------------------------:|--------------------------------------:|--------------------------------------:|---------------------:|---------------------:|
| lammps_openmp                  |              6.17432e-08 |             1.13047e-07 |                           6.17432e-08 |                           1.13129e-07 |             0.996175 |             0.999239 |
| gromacs_cpu_openmp             |              8.21361e-08 |             1.46191e-07 |                           8.21361e-08 |                           1.46144e-07 |             0.997116 |             0.999076 |
| gromacs_gpu_hybrid_strict_pme5 |              8.52538e-08 |             1.26706e-07 |                           8.52538e-08 |                           1.26511e-07 |             0.996105 |             0.997586 |

MSD-fit `cNE0_pop_molecular_MSDfit_sigma_S_cm` deltas:

| comparison                                              |   reference |       value |       delta |   pct_delta_vs_abs_reference |
|:--------------------------------------------------------|------------:|------------:|------------:|-----------------------------:|
| gromacs_cpu_openmp_minus_lammps_openmp                  | 0.000647542 | 0.000842304 | 0.000194762 |                      30.0771 |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | 0.000647542 | 0.000747041 | 9.94985e-05 |                      15.3656 |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | 0.000842304 | 0.000747041 | -9.52637e-05 |                     -11.3099 |

MSD-fit `cNE0_pop_atom90_93_MSDfit_sigma_S_cm` deltas:

| comparison                                              |   reference |       value |       delta |   pct_delta_vs_abs_reference |
|:--------------------------------------------------------|------------:|------------:|------------:|-----------------------------:|
| gromacs_cpu_openmp_minus_lammps_openmp                  | 0.000647853 | 0.000842127 | 0.000194274 |                      29.9873 |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | 0.000647853 | 0.000746337 | 9.84839e-05 |                      15.2016 |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | 0.000842127 | 0.000746337 | -9.57898e-05 |                     -11.3747 |

Interpretation: with the same HTP-MD-style population matrix, replacing raw endpoint diffusivity by MSD-fit diffusivity collapses the GPU `cNE0` from `0.0199661 S/cm` to about `0.000747 S/cm`. The remaining MSD-fit `cNE0` difference is then comparable to the NE difference, not a GPU-only blow-up.

MSD fit quality:

| lane                           |   frames |   time_ns |   cation_msd_fit_r2 |   anion_msd_fit_r2 |
|:-------------------------------|---------:|----------:|--------------------:|-------------------:|
| lammps_openmp                  |    10001 |        20 |            0.996175 |           0.999096 |
| gromacs_cpu_openmp             |    10001 |        20 |            0.997116 |           0.999003 |
| gromacs_gpu_hybrid_strict_pme5 |    10001 |        20 |            0.996105 |           0.997813 |

## Transport deltas

| comparison                                              | metric              |   reference |       value |        delta |   pct_delta_vs_abs_reference |
|:--------------------------------------------------------|:--------------------|------------:|------------:|-------------:|-----------------------------:|
| gromacs_cpu_openmp_minus_lammps_openmp                  | D_cation_fit_cm2_s  | 6.17432e-08 | 8.21361e-08 |  2.03928e-08 |                     33.0285  |
| gromacs_cpu_openmp_minus_lammps_openmp                  | D_anion_fit_cm2_s   | 1.13047e-07 | 1.46191e-07 |  3.31444e-08 |                     29.3192  |
| gromacs_cpu_openmp_minus_lammps_openmp                  | NE_sigma_S_cm       | 0.00120975  | 0.00158028  |  0.000370539 |                     30.6295  |
| gromacs_cpu_openmp_minus_lammps_openmp                  | NE_t_plus           | 0.353242    | 0.359729    |  0.00648728  |                      1.8365  |
| gromacs_cpu_openmp_minus_lammps_openmp                  | cNE0_htp_sigma_S_cm | 0.000758102 | 0.00251169  |  0.00175359  |                    231.313   |
| gromacs_cpu_openmp_minus_lammps_openmp                  | cNE0_htp_t_plus     | 0.174194    | 0.289326    |  0.115132    |                     66.0939  |
| gromacs_cpu_openmp_minus_lammps_openmp                  | Einstein_sigma_S_cm | 0.000955514 | 0.00220567  |  0.00125016  |                    130.836   |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | D_cation_fit_cm2_s  | 6.17432e-08 | 8.52538e-08 |  2.35106e-08 |                     38.078   |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | D_anion_fit_cm2_s   | 1.13047e-07 | 1.26706e-07 |  1.36593e-08 |                     12.0828  |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | NE_sigma_S_cm       | 0.00120975  | 0.001467    |  0.000257257 |                     21.2654  |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | NE_t_plus           | 0.353242    | 0.402217    |  0.0489745   |                     13.8643  |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | cNE0_htp_sigma_S_cm | 0.000758102 | 0.0199661   |  0.0192079   |                   2533.69    |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | cNE0_htp_t_plus     | 0.174194    | 0.349769    |  0.175575    |                    100.793   |
| gromacs_gpu_hybrid_strict_pme5_minus_lammps_openmp      | Einstein_sigma_S_cm | 0.000955514 | 0.00057955  | -0.000375963 |                    -39.3467  |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | D_cation_fit_cm2_s  | 8.21361e-08 | 8.52538e-08 |  3.11773e-09 |                      3.79581 |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | D_anion_fit_cm2_s   | 1.46191e-07 | 1.26706e-07 | -1.94852e-08 |                    -13.3285  |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | NE_sigma_S_cm       | 0.00158028  | 0.001467    | -0.000113281 |                     -7.16841 |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | NE_t_plus           | 0.359729    | 0.402217    |  0.0424872   |                     11.8109  |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | cNE0_htp_sigma_S_cm | 0.00251169  | 0.0199661   |  0.0174544   |                    694.924   |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | cNE0_htp_t_plus     | 0.289326    | 0.349769    |  0.0604433   |                     20.8911  |
| gromacs_gpu_hybrid_strict_pme5_minus_gromacs_cpu_openmp | Einstein_sigma_S_cm | 0.00220567  | 0.00057955  | -0.00162612  |                    -73.7246  |

## cNE0 endpoint drift diagnostic

| lane                           |   ion_mass_drift_endpoint_norm_nm |   li_type90_mean_endpoint_norm_nm |   anion_type93_mean_endpoint_norm_nm |   ion_mass_drift_x_nm |   ion_mass_drift_y_nm |   ion_mass_drift_z_nm |
|:-------------------------------|----------------------------------:|----------------------------------:|-------------------------------------:|----------------------:|----------------------:|----------------------:|
| lammps_openmp                  |                         0.0804788 |                         0.0791184 |                            0.0779532 |              0.078361 |            -0.0124349 |            -0.0134822 |
| gromacs_cpu_openmp             |                         1.69782   |                         1.57501   |                            1.68873   |             -1.62256  |            -0.266064  |             0.42319   |
| gromacs_gpu_hybrid_strict_pme5 |                         5.7212    |                         5.74323   |                            5.73133   |             -4.43034  |            -3.3417    |             1.39184   |

Interpretation: `NE_*` uses drift-removed molecular COM MSD and has good linear-fit R^2. `cNE0_htp_*` follows the HTP-MD-style raw endpoint estimator requested here, so it is highly sensitive to whole-trajectory drift. The GROMACS GPU lane has about `5.72 nm` ion-mass endpoint drift over 20 ns, which dominates its raw endpoint diffusivity and makes the GPU `cNE0_htp_sigma_S_cm` much larger than LAMMPS or GROMACS CPU. Do not use the current `cNE0_htp_*` numbers as parity evidence without either matching COM-removal semantics or reporting a drift-corrected sensitivity analysis alongside the HTP-MD raw value.

## Evidence files

- `output/polygen_20ns_parity_transport_20260511/transport_readiness/transport_readiness_report.md`
- `output/polygen_20ns_parity_transport_20260511/stage_metric_audit/summary.md`
- `output/polygen_20ns_parity_transport_20260511/stage_metric_audit/stage_parity_rollup.csv`
- `output/polygen_20ns_parity_transport_20260511/production_mean_parity_summary.csv`
- `output/polygen_20ns_parity_transport_20260511/production_mean_parity_delta.csv`
- `output/polygen_20ns_parity_transport_20260511/transport_analysis/transport_report.md`
- `output/polygen_20ns_parity_transport_20260511/transport_analysis/transport_summary.csv`
- `output/polygen_20ns_parity_transport_20260511/transport_analysis/transport_delta_20ns.csv`
- `output/polygen_20ns_parity_transport_20260511/transport_analysis/cne0_msd_fit_sensitivity_20ns.csv`
- `output/polygen_20ns_parity_transport_20260511/transport_analysis/msd_fit_quality_20ns.csv`
- `output/polygen_20ns_parity_transport_20260511/transport_analysis/endpoint_drift_diagnostic_20ns.csv`
