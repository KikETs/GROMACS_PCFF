# Repulsion-Power-9 Short-MD OpenMP Scaling Decomposition

## Scope

This note answers the current scaling question for the cleaned CPU short-MD benchmark on the audited
host:

- why scaling from `ntomp=2` to `ntomp=6` is positive but not ideal
- why scaling from `ntomp=6` to `ntomp=12` regresses

This is not an exact-`r-RESPA` note.

## Measurement Basis

Benchmark driver:

- [`tools/pcff_respa_parity/bench_repulsion_power_9_simd_shortmd_cpu.py`](../tools/pcff_respa_parity/bench_repulsion_power_9_simd_shortmd_cpu.py)

Clean benchmark controls:

- fixed `ntomp` per run set
- `pin=on`
- `dlb=no`
- alternating mode order
- one warmup cycle per `ntomp`
- multi-repeat median

Sequential long-run evidence:

- [`output/repulsion_power_9_shortmd_ntomp2_long_seq/summary.md`](../output/repulsion_power_9_shortmd_ntomp2_long_seq/summary.md)
- [`output/repulsion_power_9_shortmd_ntomp6_long_seq/summary.md`](../output/repulsion_power_9_shortmd_ntomp6_long_seq/summary.md)
- [`output/repulsion_power_9_shortmd_ntomp12_long_seq/summary.md`](../output/repulsion_power_9_shortmd_ntomp12_long_seq/summary.md)

Representative low-level runs:

- `perf stat` on one warmed run per `ntomp`
- available counters:
  - `task-clock`
  - `cycles`
  - `instructions`
  - `branches`
  - `branch-misses`
  - `context-switches`
  - `cpu-migrations`
  - `page-faults`

Host constraints:

- PMU access is available after lowering `perf_event_paranoid`
- cache-miss counters were still not counted on this host in the current configuration

## Current Result

The premise needs correction:

- `ntomp=6` is faster than `ntomp=2`
- `ntomp=12` is slower than `ntomp=6`

Specialized path, cleaned sequential long runs:

| ntomp | ns/day | wall s | speedup vs `ntomp=2` | parallel efficiency vs `ntomp=2` |
| --- | ---: | ---: | ---: | ---: |
| 2 | 68.399 | 6.316 | 1.000x | 1.000 |
| 6 | 173.872 | 2.485 | 2.542x | 0.847 |
| 12 | 153.206 | 2.820 | 2.240x | 0.373 |

Generic path shows the same shape:

- `2 -> 6`: good scaling, but sublinear
- `6 -> 12`: regression

So the current question is not “why `6` is not faster than `2`.”
The real question is “why `12` loses to `6`, and why scaling saturates before `12`.”

## PME Wall-Share Decomposition

Specialized path:

| ntomp | wall s | Force share | PME share | Update share | NB F kernel share |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 6.316 | 33.9% | 61.1% | 0.4% | 20.2% |
| 6 | 2.485 | 32.2% | 60.8% | 0.8% | 18.6% |
| 12 | 2.820 | 19.8% | 69.9% | 1.8% | 10.4% |

Interpretation:

- `2 -> 6`: PP and PME both get faster
- `6 -> 12`: PP still gets faster, but PME gets slower and becomes the dominant wall component

Specialized path, median component speedups:

| transition | total wall | Force | PME mesh | NB F kernel | Update |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2 -> 6` | `2.542x` | `2.672x` | `2.554x` | `2.749x` | `1.421x` |
| `6 -> 12` | `0.881x` | `1.433x` | `0.767x` | `1.586x` | `0.365x` |

The decisive fact is this:

- from `6` to `12`, `NB F kernel` still improves
- from `6` to `12`, `PME mesh` gets worse

That means the current scaling ceiling is not the repulsion-power-9 specialized nonbonded kernel.

## PME Internal Breakdown

Representative specialized runs:

| ntomp | PME spread s | PME gather s | PME 3D-FFT s | PME solve s | PME mesh total s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.306 | 0.872 | 2.460 | 0.219 | 3.863 |
| 6 | 0.151 | 0.359 | 0.934 | 0.078 | 1.528 |
| 12 | 0.381 | 0.422 | 1.068 | 0.059 | 1.945 |

The largest PME term at every thread count is `PME 3D-FFT`.

What changes at `12`:

- `PME spread` gets much worse than at `6`
- `PME gather` gets worse than at `6`
- `PME 3D-FFT` also gets worse than at `6`

So the `12`-thread regression is not a single tiny bookkeeping issue. It is a broader CPU PME/FFT-side
efficiency collapse.

## Low-Level CPU Evidence

Representative `perf stat` runs:

| ntomp | mode | affinity | IPC | branch miss rate | context switches | migrations |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2 | generic | `CPUs: 0,1` | 3.457 | 0.241% | 132 | 60 |
| 2 | specialized | `CPUs: 0,1` | 3.700 | 0.243% | 130 | 55 |
| 6 | generic | `CPUs: 0-5` | 3.113 | 0.218% | 158 | 71 |
| 6 | specialized | `CPUs: 0-5` | 3.323 | 0.209% | 152 | 66 |
| 12 | generic | `CPUs: 0-11` | 1.550 | 0.244% | 231 | 80 |
| 12 | specialized | `CPUs: 0-11` | 1.557 | 0.241% | 220 | 79 |

What this supports:

- specialized keeps a small IPC advantage over generic at each tested `ntomp`
- branch-miss rate is nearly unchanged
- context switches and migrations do not explode at `12`
- both modes suffer the same large IPC collapse at `12`

So the current evidence does **not** support:

- “the specialized kernel stops scaling at `12`”
- “OpenMP scheduler noise is the dominant cause”

It **does** support:

- CPU-side execution efficiency collapses at `12` for both modes
- the collapse coincides with PME-side slowdown, especially FFT/spread/gather

## Diagnosis

Current ranked causes for poor `2 -> 6 -> 12` scaling:

1. CPU PME-side scaling collapse at `12`, dominated by `PME 3D-FFT` plus worse spread/gather
2. generalized per-thread efficiency collapse at `12` shown by the IPC drop from about `3.3` to about `1.56`
3. modest growth in `Update` and buffer-op overhead at `12`
4. the nonbonded kernel itself, which is **not** the current limit because it still improves from `6` to `12`

## Current/Future Recommendation

The next optimization target should be CPU PME/FFT scaling, not the repulsion-power-9 specialized
nonbonded kernel.

Current honest host-local guidance:

- use `ntomp=6` as the current sweet spot on this audited host for this cleaned short-MD shape
- do not expect `ntomp=12` to beat `ntomp=6` until CPU PME scaling is improved
- treat further nonbonded-kernel work as secondary until PME-side scaling is revisited
