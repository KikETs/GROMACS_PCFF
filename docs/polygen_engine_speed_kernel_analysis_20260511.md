# PolyGen PCFF engine speed and kernel-level analysis

Date: 2026-05-11 KST

Scope: current PolyGen PCFF strict production path after LAMMPS/GROMACS schedule alignment. The reference production style is `run_style respa 2 4`, `lj/class2/coul/long 9.5`, `pppm 0.0001`, 2 fs outer timestep, transport-ready trajectory output, and the current GROMACS strict exact-rRESPA mapping.

## Executive conclusion

The current speed order is:

1. GROMACS CPU+GPU strict hybrid: about `123 ns/day` over the 10->20 ns extension.
2. GROMACS CPU double OpenMP: about `44 ns/day` over the 10->20 ns extension.
3. LAMMPS CPU OpenMP: about `40 ns/day` over the 10->20 ns extension.
4. LAMMPS CPU+KOKKOS GPU: about `10.6 ns/day` in the matched 10k-step smoke.

The main reason is not I/O. The dominant differences are kernel ownership and synchronization:

- LAMMPS CPU stays fully CPU-bound and pays large `Pair + Bond + Kspace + Comm + Modify` costs.
- LAMMPS KOKKOS does move `lj/class2/coul/long/kk` and class2 styles to KOKKOS, but r-RESPA disables KOKKOS communication fast path and the run becomes dominated by `Comm + Modify + Kspace`, not pair compute.
- GROMACS CPU uses the custom exact-rRESPA CPU/NBNXM path, but the run is still dominated by PCFF listed/class2 work.
- GROMACS GPU hybrid accelerates NBNXM short-range and exact-rRESPA GPU listed/bonded work, but PME and update remain on CPU; GPU wait time is now the limiter.

## Speed snapshot

| Lane | Precision / execution | Evidence | Speed |
|---|---:|---|---:|
| LAMMPS CPU OpenMP | double, `-sf omp -pk omp 16`, `taskset 0-23` | `output/polygen_pcff_gromacs_initial_em_notebook/lammps_openmp/prod_extend_10_to_20.stdout.log` | `39.824 ns/day` 10->20 ns extension average |
| LAMMPS CPU OpenMP last chunk | same | same log, chunk0100 | `40.034 ns/day` |
| LAMMPS CPU+KOKKOS GPU | double KOKKOS CUDA, `-k on g 1 t 12 -sf kk`, 10k-step smoke from current chunk0050 restart | `output/perf_kernel_lammps_kokkos_current_20260511/kokkos_respa24_smoke_bound.log` | `10.588 ns/day` |
| GROMACS CPU OpenMP | double, `-nb cpu -pme cpu -bonded cpu -update cpu`, `ntomp=20` | `output/polygen_pcff_gromacs_initial_em_notebook/gromacs_cpu_openmp/14_prod01_nvt_10000ps_chunk0100.log` and extension run output | `44.249 ns/day` 10->20 ns extension average |
| GROMACS CPU OpenMP last chunk | same | chunk0100 log | `41.697 ns/day` |
| GROMACS CPU+GPU strict | mixed/single CUDA, `-nb gpu -pme cpu -bonded gpu -update cpu`, `ntomp=12` | `output/polygen_pcff_gromacs_initial_em_notebook/gromacs_gpu_hybrid_strict_pme5/14_prod01_nvt_10000ps_chunk0100.log` and extension run output | `123.363 ns/day` 10->20 ns extension average |
| GROMACS CPU+GPU strict last chunk | same | chunk0100 log | `125.118 ns/day` |

Ratio versus LAMMPS CPU extension average:

- LAMMPS KOKKOS smoke: `0.27x`
- GROMACS CPU extension: `1.11x`
- GROMACS GPU strict extension: `3.10x`

The KOKKOS number is a matched short smoke, not a full 20 ns production average. It is still useful for kernel diagnosis because the log exposes where the time goes.

## Kernel breakdown

### LAMMPS CPU OpenMP

Representative chunk: `prod_extend_10_to_20.stdout.log`, chunk0100.

| Timer | Wall s | % total |
|---|---:|---:|
| Pair | `116.28` | `26.94` |
| Bond | `102.28` | `23.70` |
| Kspace | `66.01` | `15.29` |
| Comm | `65.73` | `15.23` |
| Modify | `57.60` | `13.34` |
| Neigh | `18.15` | `4.20` |
| Output | `0.52` | `0.12` |
| Total loop | `431.64` | `100` |

Interpretation:

- LAMMPS CPU is balanced across expensive PCFF pair, class2 bonded, PPPM, communication, and Nose-Hoover/modify work.
- Output is not the speed limiter in the current transport-ready dump setting: `0.12%` in the representative chunk.
- `Pair + Bond + Kspace` alone is about `65.9%`; adding `Comm + Modify` reaches about `94.5%`.

### LAMMPS CPU+KOKKOS GPU

Matched smoke: `output/perf_kernel_lammps_kokkos_current_20260511/kokkos_respa24_smoke_bound.log`.

This was run from the current `prod_chunk0050.restart` with the same force-field and `run_style respa 2 4` production semantics:

```bash
/home/kiket/src/lammps/build-kokkos/lmp \
  -nonbuf -k on g 1 t 12 -sf kk \
  -pk kokkos neigh full newton off \
  -log kokkos_respa24_smoke_bound.log \
  -in kokkos_respa24_smoke.in
```

Observed setup:

- `lj/class2/coul/long/kk`, `class2/kk` bond/angle/dihedral/improper were restored.
- The pair neighbor list attributes were `full, newton off, kokkos_device`.
- LAMMPS printed:
  - `WARNING: Fix RESPA not compatible with sending data in Kokkos communication`
  - `WARNING: Fix with atom-based arrays not compatible with sending data in Kokkos communication, switching to legacy exchange/border communication`

Breakdown:

| Timer | Wall s | % total |
|---|---:|---:|
| Pair | `2.41` | `1.47` |
| Bond | `20.48` | `12.55` |
| Kspace | `36.39` | `22.30` |
| Comm | `52.00` | `31.86` |
| Modify | `49.89` | `30.57` |
| Output | `0.04` | `0.02` |
| Total loop | `163.21` for 10k steps | `100` |

Interpretation:

- KOKKOS makes the pair timer tiny (`1.47%`), so pair offload itself is working.
- The run is slow because the remaining dominant work is not pair compute:
  - `Comm + Modify = 62.43%`
  - `Kspace = 22.30%`
  - `Bond = 12.55%`
- r-RESPA is the key incompatibility. The log explicitly says KOKKOS communication is downgraded to legacy exchange/border communication.
- For this 7075-atom system, KOKKOS GPU offload does not amortize the device/host and KOKKOS fix/communication overhead.

The LAMMPS KOKKOS documentation matches this behavior: GPU+OpenMP hybrid use is supported, but CPU/GPU overlap has conditions, and unsupported fixes/styles force host-device movement. See local source docs:

- `/home/kiket/src/lammps/doc/src/Speed_kokkos.rst`
- `/home/kiket/src/lammps/doc/src/package.rst`

### GROMACS CPU OpenMP

Representative chunk: `gromacs_cpu_openmp/14_prod01_nvt_10000ps_chunk0100.log`.

| Timer | Wall s | % total |
|---|---:|---:|
| Force | `337.08` | `81.3` |
| eR CPU listed | `200.32` | `48.3` |
| PME mesh | `34.99` | `8.4` |
| eR long range | `35.01` | `8.4` |
| NB X/F buffer ops | `8.66` | `2.1` |
| Update | `7.67` | `1.9` |
| Neighbor search | `5.19` | `1.3` |
| Total | `414.42` | `100` |

Interpretation:

- GROMACS CPU is only slightly faster than LAMMPS CPU in the final production extension because it is still dominated by exact PCFF listed/class2 work.
- The largest single identified sub-timer is `eR CPU listed` at `48.3%`.
- PME is not the main CPU bottleneck here; exact listed/class2 and force bookkeeping dominate.

Relevant local source/code evidence:

- CPU exact-rRESPA/NBNXM contract code: `src/gromacs/nbnxm/nbnxm.cpp`, `src/gromacs/nbnxm/kernel_common.h`, `src/gromacs/mdlib/sim_util.cpp`.
- Exact-rRESPA timing labels: `src/gromacs/timing/include/gromacs/timing/wallcycle.h`.

### GROMACS CPU+GPU strict hybrid

Representative chunk: `gromacs_gpu_hybrid_strict_pme5/14_prod01_nvt_10000ps_chunk0100.log`.

Run command:

```bash
gmx mdrun -ntmpi 1 -ntomp 12 -pin off -dlb no -notunepme \
  -nb gpu -pme cpu -bonded gpu -update cpu
```

Observed setup:

- `PP tasks will do non-perturbed short-ranged and Lennard-Jones 1-4 listed-pair interactions on the GPU`
- `Using GPU 8x4 nonbonded short-range kernels`
- PME is CPU-side.
- Coordinate update is CPU-side.
- Exact-rRESPA GPU bonded/listed wide mode is enabled by:
  - `GMX_PCFF_EXACT_RESPA_GPU_BONDED_FTYPES=all`
  - `GMX_PCFF_EXACT_RESPA_GPU_BONDED_CPU_LISTED_OVERLAP=1`
  - `GMX_PCFF_EXACT_RESPA_GPU_BONDED_LIST_CACHE=1`
  - `GMX_PCFF_GPU_BONDED_THREADS_PER_BLOCK=256`

Breakdown:

| Timer | Wall s | % total |
|---|---:|---:|
| Force | `114.78` | `83.1` |
| PME mesh | `22.94` | `16.6` |
| PME wait for PP | `115.18` | `83.4` |
| Wait GPU NB local | `55.19` | `40.0` |
| eR GPU wait NB | `11.54` | `8.4` |
| eR bond wait | `43.69` | `31.6` |
| eR long range | `22.95` | `16.6` |
| NB X/F buffer ops | `6.11` | `4.4` |
| Update | `4.57` | `3.3` |
| Total | `138.11` | `100` |

Interpretation:

- This lane is faster because the expensive short-range NBNXM and exact-rRESPA listed/bonded work are mostly offloaded.
- It does not reach the generic GAFF/leap-frog GPU speed class because strict PCFF exact-rRESPA has more synchronization:
  - CPU PME remains on the critical path.
  - CPU update remains on the critical path.
  - exact-rRESPA GPU bonded/listed requires H2D/D2H force traffic and per-level synchronization.
  - `eR bond wait` and `Wait GPU NB local` together are a large part of the wall time.

Relevant local source/code evidence:

- Exact-rRESPA nonbonded GPU support requires mixed/single precision and exact pair splitting: `src/gromacs/mdlib/exactrespa_nonbonded_gpu.cpp`.
- Wide exact-rRESPA bonded/listed GPU admission is controlled by `GMX_PCFF_EXACT_RESPA_GPU_BONDED_FTYPES`: `src/gromacs/listed_forces/listed_forces_gpu_impl.cpp`, `src/gromacs/listed_forces/manage_threading.cpp`.
- The timing terms `eR GPU ...` and `eR bond ...` are explicit wallcycle labels in `src/gromacs/timing/include/gromacs/timing/wallcycle.h`.

## Why the GPU run became slower after the recent patch/run

Evidence from the current logs does not support "more algorithmic work" as the main reason.

Comparison: GPU strict chunk0050 versus chunk0100.

| Metric | chunk0050 | chunk0100 | Change |
|---|---:|---:|---:|
| Performance | `156.637 ns/day` | `125.118 ns/day` | `-20.1%` |
| Total wall | `110.319 s` | `138.110 s` | `+27.791 s` |
| M-Flops total | `72.205 G` | `72.160 G` | effectively same |
| PME mesh | `22.598 s` | `22.935 s` | `+0.337 s` |
| Force | `87.318 s` | `114.778 s` | `+27.460 s` |
| Wait GPU NB local | `28.548 s` | `55.187 s` | `+26.639 s` |
| eR bond wait | `20.825 s` | `43.690 s` | `+22.865 s` |

The MDP and command-side settings are unchanged in the compared logs:

- `nsteps = 400000`
- `nstlist = 80`
- `rlist = 1.04`
- `nstcalcenergy = 40000`
- `nstenergy = 40000`
- `nstxout-compressed = 4000`
- same Ewald beta override: `0.237144 A^-1`
- same GPU: RTX 5070 Ti
- same `-nb gpu -pme cpu -bonded gpu -update cpu`

Therefore, at log level, the slowdown localizes to GPU wait timers:

- `Wait GPU NB local`
- `eR bond wait`

PME did not slow down meaningfully, and the total flop count did not increase. The current evidence says the later GPU run waited longer for GPU kernels to complete.

잘 모르겠습니다. The exact external cause of the increased GPU wait cannot be proven from the existing mdrun logs alone because they do not record GPU clocks, power cap, temperature, graphics/desktop contention, or CUDA scheduling state during chunk0050 and chunk0100.

What would prove it:

1. Re-run chunk0050 and chunk0100 back-to-back from their existing checkpoints with the same binary and command.
2. Log `nvidia-smi dmon` or equivalent during the run: SM %, memory %, power, graphics/SM clocks, temperature, throttle reason.
3. Compare wallcycle timers again. If M-Flops remain constant and only GPU wait changes with clock/throttle state, this is runtime/hardware state, not kernel math.

## Why the four engines differ

### LAMMPS CPU vs GROMACS CPU

These are close because both are still CPU-bound strict PCFF exact-rRESPA runs. GROMACS CPU is modestly faster in the current extension (`44.249` vs `39.824 ns/day`) because the CPU nonbonded/force path uses the GROMACS exact-rRESPA/NBNXM machinery while LAMMPS spends more time spread across Pair, Bond, Kspace, Comm, and Modify.

This is not a large architectural win. The CPU strict path is still limited by listed/class2 work and exact-rRESPA force bookkeeping.

### LAMMPS CPU vs LAMMPS KOKKOS GPU

KOKKOS is slower here because the accelerated pair kernel is not the bottleneck after offload.

The matched smoke shows:

- Pair drops to `1.47%`.
- But `Comm + Modify + Kspace` becomes `84.73%`.
- r-RESPA forces KOKKOS communication fallback:
  - `Fix RESPA not compatible with sending data in Kokkos communication`
  - `switching to legacy exchange/border communication`

So LAMMPS KOKKOS is paying GPU/KOKKOS overhead while the actual bottleneck moves to communication, fix/thermostat/update machinery, and PPPM. For this 7075-atom system, the GPU has too little useful pair work to amortize that overhead.

### GROMACS CPU vs GROMACS GPU

GROMACS GPU is about `2.8x` faster than GROMACS CPU over the 10->20 ns extension because it moves the dominant short-range and listed/bonded work onto CUDA. It is not `5x` or `300 ns/day` because:

- PME remains CPU-side.
- update remains CPU-side.
- exact-rRESPA GPU path requires additional H2D/D2H and force-add work.
- `eR bond wait` is large in the current strict GPU implementation.

The current strict GPU bottleneck is not raw nonbonded math. It is GPU wait/synchronization around exact-rRESPA NB and bonded/listed kernels.

### LAMMPS KOKKOS GPU vs GROMACS GPU

Both use GPU, but they are not equivalent GPU execution models.

LAMMPS KOKKOS:

- Uses `/kk` styles for pair/bonded/PPPM.
- r-RESPA disables the KOKKOS communication fast path.
- The matched smoke becomes `Comm + Modify + Kspace` dominated.

GROMACS GPU:

- Uses CUDA NBNXM for short-range.
- Uses a custom exact-rRESPA GPU path for nonbonded and listed/bonded kernels.
- Keeps PME/update on CPU but overlaps enough work to reach about `123 ns/day`.

This is why "GPU used" alone is not the deciding variable. The deciding variable is whether the current bottleneck is actually resident in efficient GPU kernels without forced synchronization or host fallback.

## Current optimization implications

High-value targets:

1. Reduce `eR bond wait`.
   - This is the largest strict-GPU-specific timer after `Wait GPU NB local`.
   - Candidate work: reduce per-step bonded launch count, fuse class2 listed kernels where possible, or keep force accumulation on device longer.

2. Reduce `Wait GPU NB local`.
   - The same algorithmic work got slower between chunk0050 and chunk0100 without more flops.
   - First isolate runtime state with GPU telemetry before changing math.

3. Move more PME/update work off CPU only if it does not break strict PCFF parity.
   - Current strict lane intentionally uses `-pme cpu -update cpu`.
   - Moving update or PME can improve throughput, but it changes the claim boundary and must be revalidated.

4. Do not spend time optimizing LAMMPS KOKKOS for this exact small strict r-RESPA system unless the goal is a separate KOKKOS paper path.
   - The matched smoke shows KOKKOS pair offload is already fast.
   - The dominant time is in r-RESPA-incompatible communication/fix overhead, not pair math.

## Evidence files

- LAMMPS CPU full extension:
  - `output/polygen_pcff_gromacs_initial_em_notebook/lammps_openmp/prod_extend_10_to_20.stdout.log`
- LAMMPS KOKKOS matched smoke:
  - `output/perf_kernel_lammps_kokkos_current_20260511/kokkos_respa24_smoke_bound.log`
  - `output/perf_kernel_lammps_kokkos_current_20260511/kokkos_respa24_smoke.in`
- GROMACS CPU:
  - `output/polygen_pcff_gromacs_initial_em_notebook/gromacs_cpu_openmp/14_prod01_nvt_10000ps_chunk0100.log`
- GROMACS GPU:
  - `output/polygen_pcff_gromacs_initial_em_notebook/gromacs_gpu_hybrid_strict_pme5/14_prod01_nvt_10000ps_chunk0050.log`
  - `output/polygen_pcff_gromacs_initial_em_notebook/gromacs_gpu_hybrid_strict_pme5/14_prod01_nvt_10000ps_chunk0100.log`
- GROMACS source:
  - `src/gromacs/mdlib/exactrespa_nonbonded_gpu.cpp`
  - `src/gromacs/listed_forces/listed_forces_gpu_impl.cpp`
  - `src/gromacs/listed_forces/manage_threading.cpp`
  - `src/gromacs/timing/include/gromacs/timing/wallcycle.h`
- LAMMPS KOKKOS source/docs:
  - `/home/kiket/src/lammps/doc/src/Speed_kokkos.rst`
  - `/home/kiket/src/lammps/doc/src/package.rst`
