# Exact r-RESPA PCFF Kernel Optimization Handoff, 2026-05-10

## Scope

This handoff freezes the current GROMACS PCFF exact r-RESPA implementation state before the next kernel-optimization pass.

The next pass must optimize CPU/GPU kernels without changing the physical run protocol. Runtime sweep knobs may be used for diagnosis, but the target is source-level kernel/runtime overhead reduction, not changing the equilibration or production schedule.

## Current Branch State

- Repository: `/home/kiket/Desktop/test/GROMACS_PCFF`
- Branch: `exact-respa-cpu-only-speedups-20260422`
- Base commit before this handoff: `d927afa16f Add CPU exact r-RESPA current binary revalidation`
- Main notebook: `/home/kiket/Desktop/test/GROMACS_PCFF/output/jupyter-notebook/polygen_pcff_rrespa_lammps_gromacs_benchmark.ipynb`
- Main run root: `/home/kiket/Desktop/test/GROMACS_PCFF/output/polygen_pcff_gromacs_initial_em_notebook`

## Current Measured Performance

Measured from the existing full PolyGen-style output logs in the run root:

- GROMACS CPU OpenMP prod: about `41.994 ns/day`
- GROMACS GPU hybrid strict prod: about `154.071 ns/day`
- LAMMPS OpenMP prod reference: about `39.435 ns/day`

Representative CPU timing evidence:

- Log: `output/polygen_pcff_gromacs_initial_em_notebook/gromacs_cpu_openmp/14_prod01_nvt_10000ps_chunk0010.log`
- Dominant timer: `eR CPU listed`, about `51.6%` of wall time
- Overall performance in that chunk: `41.615 ns/day`

Representative GPU timing evidence:

- Log: `output/polygen_pcff_gromacs_initial_em_notebook/gromacs_gpu_hybrid_strict_pme5/14_prod01_nvt_10000ps_chunk0010.log`
- Dominant timers:
  - `PME wait for PP`, about `79.8%`
  - `Wait GPU NB local`, about `25.8%`
  - `eR bond*` timers together, about `30%`
- Overall performance in that chunk: `153.152 ns/day`

## Kernel Bottleneck Interpretation

CPU optimization target:

- Primary path: `src/gromacs/listed_forces/*`
- Driver path: `src/gromacs/mdrun/exactrespastepper.cpp`
- Dominant cost is exact r-RESPA listed/bonded work, not trajectory I/O.

GPU optimization target:

- Bonded GPU path: `src/gromacs/listed_forces/listed_forces_gpu_impl*.{cpp,h}`
- Exact r-RESPA nonbonded GPU path: `src/gromacs/mdlib/exactrespa_nonbonded_gpu*.{cpp,cu,h}`
- Scheduling/workload path: `src/gromacs/taskassignment/decidegpuusage.cpp`, `src/gromacs/taskassignment/decidesimulationworkload.cpp`
- Dominant remaining cost is not merely "bonded on/off"; it is GPU/CPU synchronization, coordinate/force transfer, force add, and exact r-RESPA force residency.

## Optimization Priorities

1. Reduce CPU `eR CPU listed` cost.
   - Predecode/prepack PCFF class2 listed interactions where safe.
   - Avoid repeated generic bonded dispatch in exact r-RESPA inner steps.
   - Reduce force scatter contention.

2. Reduce GPU bonded transfer/add/wait overhead.
   - Keep listed-force data resident where possible.
   - Avoid avoidable H2D/D2H and CPU-side force add in exact r-RESPA inner loops.
   - Fuse or specialize PCFF class2 bonded kernels only after correctness gates are in place.

3. Reduce exact r-RESPA NB GPU wait/copyback overhead.
   - Inspect `exactrespa_nonbonded_gpu.cpp` and `exactrespa_nonbonded_gpu_internal.cu`.
   - Avoid force copyback to host unless required by update, virial, or energy accounting.

4. Keep physical parity checks active.
   - Do not replace the PolyGen-style schedule with a faster but non-equivalent schedule.
   - Final verification must run GROMACS CPU/GPU full equilibration and production only.
   - LAMMPS should be disabled in the notebook for the final verification pass.

## Local Generated Artifacts Not Committed

The following local artifacts are intentionally not part of this handoff commit because they are generated run outputs or oversized local scratch data:

- `tests/reference_results/` new local result directories, about `2.1G`
- `tmp/`, about `1019M`
- Root-level transient logs and outputs such as `grompp.log`, `grompp.stdout.log`, `mdrun.stdout.log`, `energy.xvg`, `mdout.mdp`, `log.lammps`

These paths remain useful as local evidence, but should not be pushed as repository source unless a small, curated fixture is extracted.

## Final Verification Plan After Kernel Optimization

1. Set notebook execution so LAMMPS is not run.
2. Run GROMACS CPU OpenMP full `equil + prod`.
3. Run GROMACS GPU hybrid full `equil + prod`.
4. Compare stage-level density, volume, pressure, thermal, and energy behavior against the saved reference summaries.
5. Report CPU/GPU prod speed from `mdrun` logs and wallcycle timers.
