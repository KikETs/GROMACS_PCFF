# Repulsion-Power-9 Short-MD Low-Level Diagnostic

## Scope

This note answers the remaining low-level question for the audited short-MD CPU shape:

- why pure OpenMP `ntomp=12` loses to `ntomp=6`
- whether that loss is mainly FFT, cache/bandwidth pressure, or generic thread-efficiency collapse
- why the `-ntmpi 2 -npme 1 -ntomp 6 -ntomp_pme 6` layout avoids most of that loss

This is not an exact-`r-RESPA` note.

## Measurement Basis

Representative specialized-layout runs were profiled on the same `gate_h_dense_salt_polymer_2x2x2`
TPR with:

- `omp6`: `-ntmpi 1 -ntomp 6`
- `omp12`: `-ntmpi 1 -ntomp 12`
- `split12_pp6_pme6`: `-ntmpi 2 -npme 1 -ntomp 6 -ntomp_pme 6`

Perf stack:

- `perf stat` base events:
  - `task-clock`
  - `cycles`
  - `ref-cycles`
  - `instructions`
  - `cache-references`
  - `cache-misses`
  - `context-switches`
  - `cpu-migrations`
  - `page-faults`
- `perf stat` metrics:
  - `backend_bound`
  - `backend_bound_by_memory`
- `perf record` + `perf report` for hotspot attribution

Summary evidence:

- [`output/repulsion_power_9_shortmd_lowlevel_profile/summary.md`](../output/repulsion_power_9_shortmd_lowlevel_profile/summary.md)
- [`output/repulsion_power_9_shortmd_lowlevel_profile_post_pmegather/summary.md`](../output/repulsion_power_9_shortmd_lowlevel_profile_post_pmegather/summary.md)

Limits:

- `nmi_watchdog` could not be disabled in this session, so some `L1`/`dTLB`/branch events stayed unavailable
- kernel symbol resolution was restricted, so kernel-side attribution is incomplete
- `libgomp` was stripped, but `objdump -d /lib/x86_64-linux-gnu/libgomp.so.1.0.0` shows the sampled
  `0x256c0` and `0x258a0` sites are tight `pause` spin loops

## Main Counter Evidence

Representative perf results:

| layout | ns/day | wall s | IPC | cache miss rate | cache MPKI | backend bound | memory-bound backend |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `omp6` | `172.847` | `2.500` | `3.27` | `8.32%` | `3.388` | `39.2%` | `20.8%` |
| `omp12` | `155.522` | `2.778` | `1.55` | `18.12%` | `7.087` | `55.1%` | `41.7%` |
| `split12_pp6_pme6` | `240.140` | `1.799` | `2.29` | `8.20%` | `3.318` | `44.7%` | `27.6%` |

What this rules out:

- no meaningful clock collapse:
  - `cycles/ref-cycles` stays near constant
  - `omp6`: about `1.181`
  - `omp12`: about `1.185`
  - `split12`: about `1.160`

What this supports:

- pure OpenMP `12` becomes much more memory-bound than pure OpenMP `6`
- the cache miss rate more than doubles
- MPKI roughly doubles
- IPC collapses from `3.27` to `1.55`
- the split layout recovers most of that loss and brings cache behavior back close to `omp6`

## FFT Evidence

PME wallcycle subcomponents:

| layout | PME 3D-FFT s | PME spread s | PME gather s |
| --- | ---: | ---: | ---: |
| `omp6` | `0.921` | `0.151` | `0.359` |
| `omp12` | `1.086` | `0.375` | `0.424` |
| `split12_pp6_pme6` | `0.969` | `0.172` | `0.396` |

Interpretation:

- the pure OpenMP `12` regression is not only FFT
- `PME 3D-FFT` does get worse at `12`
- but `spread` and `gather` also get worse sharply
- the split layout improves all three PME subcomponents relative to pure OpenMP `12`

## Hotspot Evidence

Approximate sampled cycle attribution by DSO:

| layout | libgromacs | libfftw3f | libgomp | libc |
| --- | ---: | ---: | ---: | ---: |
| `omp6` | `40.81 Gcycles` | `22.38 Gcycles` | `6.65 Gcycles` | `8.45 Gcycles` |
| `omp12` | `61.43 Gcycles` | `46.13 Gcycles` | `51.49 Gcycles` | `13.57 Gcycles` |
| `split12_pp6_pme6` | `47.40 Gcycles` | `23.81 Gcycles` | `34.67 Gcycles` | `8.36 Gcycles` |

This matters more than percentage alone.

What changes from `omp6` to `omp12`:

- sampled `libfftw3f` cycles roughly double
- sampled `libgomp` cycles jump by almost an order of magnitude

What changes from `omp12` to `split12`:

- sampled `libfftw3f` cycles drop back near the `omp6` level
- sampled `libgomp` cycles also drop, though they remain significant

Representative hot symbols:

- `omp6`
  - `18.76%` `nbnxmKernelSimd`
  - `9.21%` `fft5d_execute`
- `omp12`
  - `19.48%` `libgomp` pause-loop site `0x258a0`
  - `8.56%` `libgomp` pause-loop site `0x256c0`
  - `8.33%` `fft5d_execute`
  - `2.90%` `spread_on_grid` OpenMP clone
- `split12_pp6_pme6`
  - `23.77%` `libgomp` pause-loop site `0x256c0`
  - `5.68%` `fft5d_execute`
  - `5.63%` `tMPI_Event_wait`

Interpretation:

- `omp12` has both FFT-side cost growth and thread-wait growth
- the split layout does not eliminate waiting, but it reduces FFT-side cost enough to win decisively
- therefore the root cause is not “threads are waiting” alone

## Final Diagnosis

Current ranked explanation for the pure OpenMP `12` collapse:

1. PME-side memory pressure rises sharply at `12`
2. that pressure shows up as worse FFT, spread, and gather times
3. the higher-memory-pressure regime also coincides with heavy OpenMP spin-wait overhead
4. the PP kernel is not the cause; it continues to improve locally

In short:

- `FFT` is a real part of the problem
- `cache/bandwidth pressure` is the stronger low-level explanation
- `thread efficiency collapse` is real, but it is coupled to the memory-heavy PME regime, not a standalone scheduler story

## Current Optimization Direction

The next low-level optimization target should be CPU PME-side work, in this order:

1. `fft5d_execute` / FFTW-side efficiency
2. PME spread/gather memory behavior
3. thread-wait reduction after PME memory pressure is reduced

The repulsion-power-9 specialized PP kernel is no longer the right place to spend time first.

## PME Gather Hot-Path Follow-Up

One secondary PME-side cost was worth fixing immediately because it was not physics work at all:

- [`src/gromacs/ewald/pme_gather.cpp`](../src/gromacs/ewald/pme_gather.cpp)
- [`src/gromacs/ewald/pme.cpp`](../src/gromacs/ewald/pme.cpp)

Applied change:

- cache PME trace env lookups once per process instead of calling `getenv` in the hot path
- compute the current-step trace gate once per `gather_f_bsplines()` call
- precompute the traced global atom index and per-atom trace decision inside `do_fspline`

Post-change representative profile:

| layout | ns/day | wall s | PME gather s | `getenv` sample share |
| --- | ---: | ---: | ---: | ---: |
| `omp6` | `187.258` | `2.307` | `0.162` | `0.18%` |
| `omp12` | `160.828` | `2.686` | `0.305` | `0.08%` |
| `split12_pp6_pme6` | `263.032` | `1.643` | `0.195` | `0.12%` |

Previous baseline for the same measurement stack:

- `omp6`: `PME gather 0.359 s`, `getenv 7.56%`
- `omp12`: `PME gather 0.424 s`, `getenv 3.78%`
- `split12_pp6_pme6`: `PME gather 0.396 s`, `getenv 4.81%`

What changed:

- the PME gather wall time dropped sharply on all three audited layouts
- `getenv` effectively disappeared from the hotspot list
- final wall speed improved:
  - `omp6`: `172.847 -> 187.258 ns/day`
  - `omp12`: `155.522 -> 160.828 ns/day`
  - `split12_pp6_pme6`: `240.140 -> 263.032 ns/day`

What did not change:

- pure OpenMP `12` is still much more memory-bound than pure OpenMP `6`
- `fft5d_execute` and `libgomp` wait sites remain major hotspots
- the remaining primary ceiling is still PME-side memory pressure and FFT-side work, not the PP kernel

Updated priority after this cleanup:

1. FFT / FFTW-side efficiency
2. PME spread/gather memory behavior beyond the removed trace-control overhead
3. thread-wait reduction after PME memory pressure is reduced

## PME Spread Thread-Merge Follow-Up

The next PME-side follow-up targeted `spread_on_grid()` itself:

- [`src/gromacs/ewald/pme_spread.cpp`](../src/gromacs/ewald/pme_spread.cpp)

Applied change:

- for the threaded CPU spread path, merge the spread/copy phase and the overlap-reduction phase into a
  single OpenMP team
- keep the old path as fallback when the threaded-grid assumptions are not met
- this removes one extra OpenMP team launch and one extra inter-team synchronization from the hot path

Post-change layout sweep:

- [`output/repulsion_power_9_shortmd_layout_post_pmespread_merge/summary.md`](../output/repulsion_power_9_shortmd_layout_post_pmespread_merge/summary.md)
- [`output/repulsion_power_9_shortmd_layout_post_pmespread_merge_repeatdepth/summary.md`](../output/repulsion_power_9_shortmd_layout_post_pmespread_merge_repeatdepth/summary.md)

Representative impact:

| layout | basis | ns/day | wall s | PME spread s | PME gather s | PME 3D-FFT s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `omp6` | 3-repeat sweep | `186.480` | `2.317` | `0.141` | `0.163` | `0.942` |
| `omp12` | 6-repeat confirmation | `161.897` | `2.6685` | `0.338` | `0.2855` | `1.117` |
| `split12_pp6_pme6` | 6-repeat confirmation | `269.829` | `1.601` | `0.162` | `0.1885` | `0.968` |

Compared to the post-PME-gather baseline:

- `split12_pp6_pme6`
  - `264.066 -> 269.829 ns/day`
  - `1.636 -> 1.601 s`
  - `PME spread 0.175 -> 0.162 s`
  - `PME gather 0.195 -> 0.1885 s`
- `omp12`
  - the 3-repeat sweep looked better, but the 6-repeat confirmation fell back near the old baseline
  - this is not stable evidence for a real pure-OpenMP-12 fix

What this means:

- the spread-thread-merge cleanup is a real improvement for the best audited split layout
- it is not a general cure for the single-rank `omp12` collapse
- after this change, `PME 3D-FFT` remains much larger than `PME spread`

Updated priority after this cleanup:

1. FFT / FFTW-side efficiency
2. deeper PME memory behavior around spread/gather, not OpenMP team-launch overhead
3. thread-wait reduction after PME memory pressure is reduced

## Deep PMU Follow-Up

After lowering the host restrictions to allow broader PMU access, the same three representative
specialized layouts were profiled again with additional branch, L1, and TLB events:

- [`output/repulsion_power_9_shortmd_deep_profile_post_pmegather/summary.md`](../output/repulsion_power_9_shortmd_deep_profile_post_pmegather/summary.md)

Representative results:

| layout | IPC | branch miss | cache miss | L1 miss | dTLB miss | iTLB miss | mem-bound backend | libfftw3f | libgomp | affinity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `omp6` | `3.03` | `0.41%` | `8.37%` | `5.71%` | `0.31%` | `0.09%` | `21.8%` | `30.37%` | `9.73%` | `0-5` |
| `omp12` | `1.42` | `0.34%` | `18.30%` | `6.16%` | `0.90%` | `0.29%` | `43.6%` | `28.40%` | `31.07%` | `0-11` |
| `split12_pp6_pme6` | `2.32` | `0.24%` | `7.83%` | `5.66%` | `0.07%` | `0.47%` | `28.3%` | `22.04%` | `28.88%` | `0-5 / 6-11` |

What this rules out more clearly:

- the `omp12` failure is not a branch-misprediction story
  - branch miss rate actually drops: `0.41% -> 0.34%`
- it is not mainly an L1 miss story
  - L1 miss rate only moves modestly: `5.71% -> 6.16%`

What becomes stronger:

- deep-cache / memory pressure is the main backend explanation
  - cache miss rate jumps: `8.37% -> 18.30%`
  - memory-bound backend doubles: `21.8% -> 43.6%`
- TLB pressure also worsens materially
  - dTLB miss rate rises about `3x`: `0.31% -> 0.90%`
  - iTLB miss rate rises about `3.4x`: `0.09% -> 0.29%`

## Host-Local Topology Interpretation

The audited `Ryzen 9 9900X` host reports:

- `L3 cache: 64 MiB (2 instances)`
- CPUs `0-5` share `L3:0`
- CPUs `6-11` share `L3:1`

Observed affinity placement:

- pure OpenMP `omp6`: `0-5`
- pure OpenMP `omp12`: `0-11`
- split `2 ranks + 1 PME rank`: rank 0 on `0-5`, rank 1 on `6-11`

That means:

- pure OpenMP `12` spans both 32 MiB L3 groups in one shared OpenMP team
- the split layout keeps the two MPI ranks separated across the two L3 groups

Current host-local explanation is therefore narrower and stronger than before:

1. pure OpenMP `12` pushes the PME-heavy single-rank path across both L3 groups
2. that coincides with much higher deep-cache miss cost, higher TLB miss cost, and much worse memory-bound backend pressure
3. FFT and PME spread/gather still grow, but they are symptoms inside that memory-heavy regime
4. the split layout wins because it restores locality and reduces cross-group contention by pinning PP and PME work to separate 6-core / 32 MiB-L3 clusters

This does not prove a universal multi-CCD rule for all hosts.
It does close the audited host-local question more tightly: on this machine, `omp12` loses because
the PME-heavy single-rank CPU path scales poorly once it crosses the two-L3-group boundary.

Operational boundary:

- do not treat this as a universal OpenMP recommendation
- CPU layout must be remeasured per host
- at minimum, sweep pure OpenMP and PME-split layouts across the available L3/core topology
- hybrid-core CPUs need a separate P-core/E-core affinity sweep before accepting any `ntomp` result
