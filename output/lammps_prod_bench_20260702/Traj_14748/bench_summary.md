# Traj_14748 LAMMPS Prod-Start Benchmark

## Input

- Source data: `/home/user/바탕화면/DL/LAMMPS_BATCH/batch_runs/Traj_14748/MD/relaxed.lmp`
- LAMMPS production input: `/home/user/바탕화면/DL/LAMMPS_BATCH/batch_runs/Traj_14748/MD/production.in`
- Converted with: `tools/pcff_fixture_bridge/lammps_data_bridge.py`
- Counts: 12,730 atoms, 12,500 bonds, 23,290 angles, 27,240 dihedrals, 14,560 impropers
- Production-derived settings: `lj/class2/coul/long 9.5`, `pppm 0.0001`, `special_bonds lj/coul 0.0 0.0 1.0`, neighbor skin 0.300 nm

This is a prod-start performance benchmark. It is not a LAMMPS-vs-GROMACS transport or ensemble-parity claim. The benchmark disables trajectory/energy output and uses NVE because the exact r-RESPA fast path is benchmarked most cleanly without thermostat/barostat noise.

## TPRs

- Native: `native_grompp/native.tpr`, `dt = 0.002 ps`, `nsteps = 5000`
- Exact r-RESPA: `exact_grompp/exact.tpr`, `dt = 0.0005 ps`, `nsteps = 20000`, levels 1/2/4

Both represent 10 ps of nominal benchmark time, but exact r-RESPA uses base steps. Compare wall-cycle components as well as `ns/day`.

## Results

| mode | offload | OMP | wall s | ns/day | ms/step | log |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| native | CPU | 8 | 11.166 | 77.392 | 2.233 | `runs/native_cpu_omp8/run.log` |
| native | CPU | 16 | 23.044 | 37.501 | 4.608 | `runs/native_cpu_omp16/run.log` |
| native | NB+PME GPU | 8 | 3.544 | 243.823 | 0.709 | `runs/native_gpu_omp8/run.log` |
| native | NB+PME GPU | 16 | 6.615 | 130.634 | 1.323 | `runs/native_gpu_omp16/run.log` |
| exact r-RESPA | CPU | 8 | 201.750 | 4.283 | 10.087 | `runs/exact_cpu_omp8/run.log` |
| exact r-RESPA | CPU | 16 | 385.993 | 2.238 | 19.299 | `runs/exact_cpu_omp16/run.log` |
| exact r-RESPA | NB GPU, PME CPU, bonded CPU | 8 | 31.698 | 27.259 | 1.585 | `runs/exact_nb_gpu_omp8/run.log` |
| exact r-RESPA | NB GPU, PME CPU, bonded CPU | 16 | 72.878 | 11.856 | 3.644 | `runs/exact_nb_gpu_omp16/run.log` |
| exact r-RESPA | NB+PME+PCFF bonded GPU | 8 | 18.424 | 46.897 | 0.921 | `runs/exact_all_gpu_omp8/run.log` |
| exact r-RESPA | NB+PME+PCFF bonded GPU | 16 | 49.979 | 17.288 | 2.499 | `runs/exact_all_gpu_omp16/run.log` |

## Bottleneck Notes

- OMP16 is slower in every tested path on this system. This is not limited to GPU offload or exact r-RESPA.
- Native GPU OMP16 regression is dominated by CPU-side force/update overhead growth: Force 2.183 -> 3.359 s, Update 0.047 -> 0.828 s.
- Exact all-GPU OMP16 regression is dominated by CPU-side accumulation/buffer overhead: `NB X/F buffer ops` 0.739 -> 12.291 s, `eR GPU add F` 0.438 -> 6.507 s, `eR bond add F` 0.247 -> 4.291 s, Update 0.151 -> 3.516 s.
- Exact CPU OMP16 regression is also accumulation/setup heavy: `NB X/F buffer ops` 0.290 -> 27.014 s, `PME mesh` 2.777 -> 29.692 s, `eR CPU listed` 17.105 -> 35.490 s.
- The unsupported exact layout `nb gpu + pme gpu + bonded cpu` fails at startup by design; use either `nb gpu` only or `nb+bonded+pme gpu`.

## Patch Direction

- Do not chase raw GPU kernel time first. In prod-start benchmarks, OMP16 losses come from CPU accumulation, update, PME CPU scaling, and repeated exact r-RESPA buffer operations.
- For exact r-RESPA OMP16, the next concrete targets are `NB X/F buffer ops`, `eR GPU add F`, `eR bond add F`, and CPU PME/long-range scaling.
- For native OMP16, test MPI-thread/OpenMP splits before changing kernels; 1 rank x 16 OpenMP is clearly worse than 1 rank x 8 OpenMP here.
