# Repulsion-Power-9 Short-MD Layout Optimization

## Scope

This note answers the current host-local performance question for the non-MTS short-MD benchmark on
the audited `AMD Ryzen 9 9900X` host:

- what the best current CPU runtime layout is
- whether more OpenMP threads alone are enough
- whether CPU PME-side work or the specialized PP kernel is the dominant final-speed lever

This is not an exact-`r-RESPA` note.

## Measurement Basis

Two measurement stacks were used.

Pure-OpenMP scaling basis:

- [`docs/repulsion_power_9_shortmd_omp_scaling_decomposition.md`](./repulsion_power_9_shortmd_omp_scaling_decomposition.md)
- cleaned single-rank `ntomp=2/6/12` runs with fixed `pin=on`, `dlb=no`, alternating mode order,
  one warmup cycle, and median-of-repeats accounting

Runtime layout basis:

- runner: [`tools/pcff_respa_parity/bench_repulsion_power_9_shortmd_layout_cpu.py`](../tools/pcff_respa_parity/bench_repulsion_power_9_shortmd_layout_cpu.py)
- 3-repeat layout sweep:
  [`output/repulsion_power_9_shortmd_layout_opt/summary.md`](../output/repulsion_power_9_shortmd_layout_opt/summary.md)
- 3-repeat 12-thread PP/PME split sweep:
  [`output/repulsion_power_9_shortmd_layout_split12_sweep/summary.md`](../output/repulsion_power_9_shortmd_layout_split12_sweep/summary.md)
- 3-repeat 6-thread layout sweep:
  [`output/repulsion_power_9_shortmd_layout_6t_sweep/summary.md`](../output/repulsion_power_9_shortmd_layout_6t_sweep/summary.md)
- 6-repeat confirmation of the best 12-thread layout:
  [`output/repulsion_power_9_shortmd_layout_opt_split12_repeatdepth/summary.md`](../output/repulsion_power_9_shortmd_layout_opt_split12_repeatdepth/summary.md)
- post-PME-gather-cleanup 3-repeat layout sweep:
  [`output/repulsion_power_9_shortmd_layout_post_pmegather_opt/summary.md`](../output/repulsion_power_9_shortmd_layout_post_pmegather_opt/summary.md)
- post-PME-gather-cleanup 6-repeat confirmation of the best 12-thread layout:
  [`output/repulsion_power_9_shortmd_layout_post_pmegather_split12_repeatdepth/summary.md`](../output/repulsion_power_9_shortmd_layout_post_pmegather_split12_repeatdepth/summary.md)
- post-PME-spread-thread-merge 3-repeat layout sweep:
  [`output/repulsion_power_9_shortmd_layout_post_pmespread_merge/summary.md`](../output/repulsion_power_9_shortmd_layout_post_pmespread_merge/summary.md)
- post-PME-spread-thread-merge 6-repeat confirmation:
  [`output/repulsion_power_9_shortmd_layout_post_pmespread_merge_repeatdepth/summary.md`](../output/repulsion_power_9_shortmd_layout_post_pmespread_merge_repeatdepth/summary.md)

Important interpretation rule:

- with separate PME ranks, the `Force`, `PME mesh`, and related wallcycle lines overlap across ranks
  and are not additive wall shares
- final speed claims therefore use `Performance` and `Time:` wall seconds

## Current Best Result After PME Gather Cleanup And PME Spread Thread-Merge

After the PME gather hot-path cleanup, a second PME-side cleanup merged the threaded spread/copy and
overlap-reduction phases into a single OpenMP team for the threaded spread path. The layout ordering
still did not change, but the best split layout moved up again.

Post-spread-merge specialized results:

| layout | basis | real wall s | ns/day |
| --- | --- | ---: | ---: |
| pure OpenMP `-ntmpi 1 -ntomp 6` | 3-repeat sweep | `2.317` | `186.480` |
| pure OpenMP `-ntmpi 1 -ntomp 12` | 6-repeat confirmation | `2.6685` | `161.897` |
| split `-ntmpi 2 -npme 1 -ntomp 6 -ntomp_pme 6` | 6-repeat confirmation | `1.601` | `269.829` |

Important honesty boundary:

- the split `6 / 6` gain is stable in the 6-repeat confirmation: `264.066 -> 269.829 ns/day`
- the pure OpenMP `12` point improved in the 3-repeat sweep, but that did not hold as a stable gain
  in the 6-repeat confirmation
- this means the spread-thread-merge change is a real best-layout improvement, but not a proven
  pure-OpenMP-`12` fix

The current best confirmed point is:

- specialized path
- `-ntmpi 2 -npme 1 -ntomp 6 -ntomp_pme 6`
- `269.829 ns/day`
- `1.601 s` real wall time

## What OpenMP Alone Gets You

Current cleaned single-rank specialized results:

| layout | real wall s | ns/day |
| --- | ---: | ---: |
| `ntmpi=1 ntomp=2` | `6.344` | `68.103` |
| `ntmpi=1 ntomp=6` | `2.512` | `172.017` |
| `ntmpi=1 ntomp=12` | `2.810` | `153.761` |

Implications:

- `ntomp=6` is faster than `ntomp=2`
- `ntomp=12` is slower than `ntomp=6`
- the pure-OpenMP ceiling is therefore already reached before `12` on this host/shape

The reason is the PME side, not the repulsion-power-9 PP kernel:

- at `ntomp=12`, `PME mesh` is worse than at `ntomp=6`
- the largest PME sub-term remains `PME 3D-FFT`
- `PME spread` and `PME gather` also worsen at `12`

Basis:

- [`repulsion_power_9_shortmd_omp_scaling_decomposition.md`](./repulsion_power_9_shortmd_omp_scaling_decomposition.md:75)

## What Actually Improves Final Speed

The tested 12-thread runtime layouts were:

| layout | runtime flags | specialized ns/day |
| --- | --- | ---: |
| pure OpenMP | `-ntmpi 1 -ntomp 12` | `161.897` in the post-spread-merge 6-repeat confirmation |
| split 2-way | `-ntmpi 2 -npme 1 -ntomp 6 -ntomp_pme 6` | `272.692` in the post-spread-merge 3-repeat sweep, `269.829` in the post-spread-merge 6-repeat confirmation |

The stable recommendation is the simpler 2-rank split:

- `-ntmpi 2 -npme 1 -ntomp 6 -ntomp_pme 6`

Reason:

- it matches the best observed performance
- it is also the simplest tested split topology

## PME Thread Split Tuning Inside the Best 12-Thread Shape

For the `2-rank + 1 PME rank` family, the tested PP/PME thread partitions were:

| PP/PME threads | specialized ns/day |
| --- | ---: |
| `4 / 8` | `167.128` |
| `5 / 7` | `181.011` |
| `6 / 6` | `239.346` in the original sweep, `264.066` after the PME-gather cleanup, `269.829` after the PME-spread thread-merge cleanup |
| `7 / 5` | `205.901` |
| `8 / 4` | `170.607` |

`6 / 6` is not a cosmetic choice. It is the clear optimum among tested splits.

Interpretation:

- giving PME fewer than 6 threads leaves the PME side too slow
- giving PME more than 6 threads starves PP too much
- on this host/shape, `6 / 6` is the balanced point

## Final Speed

Best confirmed result:

- specialized path
- `-ntmpi 2 -npme 1 -ntomp 6 -ntomp_pme 6`
- `269.829 ns/day`
- `1.601 s` real wall time

Compared with the current pure-OpenMP baselines:

| baseline | baseline ns/day | best-layout ns/day | speedup |
| --- | ---: | ---: | ---: |
| pure OpenMP `ntomp=6` specialized | `186.480` | `269.829` | `1.447x` |
| pure OpenMP `ntomp=12` specialized | `161.897` | `269.829` | `1.667x` |

This is the current host-local final-speed answer.

## Where Priorities 1, 2, 3 Landed

Priority 1: CPU PME / FFT scaling

- this is the dominant final-speed lever
- the practical fix was not another PP-kernel change
- the working fix was to split PME onto a dedicated rank with 6 threads

Priority 2: PME spread / gather

- these also improve under the `6 / 6` PME split, and one more spread-side cleanup moved the split
  layout further
- example, specialized path:
  - pure OpenMP `ntomp=12` after the spread-thread-merge cleanup: `PME spread 0.338 s`,
    `PME gather 0.2855 s`
  - split `6 / 6` after the spread-thread-merge cleanup: `PME spread 0.162 s`,
    `PME gather 0.1885 s`
- the biggest absolute PME term is still `PME 3D-FFT`

Priority 3: Update / buffer overhead

- after PME splitting, `Update` stays tiny and does not drive final wall time
- `NB F kernel` still benefits from specialization, but it is no longer the wall limiter

## What This Means For The Specialized PP Kernel

Under the best final-speed layout, specialized still helps, but only slightly:

- the last audited generic-vs-specialized point under the best split layout remained small:
  - `231.981 -> 235.065 ns/day`
  - `1.013x`
- that comparison was not rerun after the PME-gather cleanup because this follow-up targeted PME control overhead, not PP microkernel math
- the practical conclusion is unchanged: whole-run speed is currently dominated by runtime layout and PME-side work more than by PP microkernel differences

## Current Recommendation

For this audited host and this cleaned short-MD shape:

- if you only use 6 total CPU threads, stay with pure OpenMP `-ntmpi 1 -ntomp 6`
- dedicated 6-thread layout sweep result: pure OpenMP `175.104 ns/day`
- tested 6-thread PME-split layouts were all slower: `165.954`, `159.189`, `149.997 ns/day`
- if you use 12 total CPU threads, do not use pure OpenMP `-ntmpi 1 -ntomp 12`
- use `-ntmpi 2 -npme 1 -ntomp 6 -ntomp_pme 6`

This is not a universal CPU setting. The fastest layout has to be tuned per CPU topology and workload.
At minimum, re-sweep rank/thread/PME-rank layouts when the CPU L3 topology, hybrid P/E-core layout, FFT
library, PME grid, or system size changes. A fixed `ntomp` value from this host should not be carried to
another CPU without measurement.

## Future Recommendation

If this needs productization beyond a host-local benchmark script:

1. Do not change global GROMACS thread/rank heuristics from this single-host result alone.
2. Add a bounded runtime-layout benchmark or recommendation path for PME-heavy charged PCFF short-MD CPU jobs,
   because the fastest setting is CPU-specific and must be found by measurement.
3. If core-code optimization resumes, target CPU PME/FFT first; the spread-side orchestration cleanup is
   no longer the dominant remaining lever.
