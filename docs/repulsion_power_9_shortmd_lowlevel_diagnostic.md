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

- [`output/repulsion_power_9_shortmd_lowlevel_profile/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_shortmd_lowlevel_profile/summary.md)

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
