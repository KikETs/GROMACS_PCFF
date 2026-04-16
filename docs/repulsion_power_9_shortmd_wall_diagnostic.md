# Repulsion-Power-9 Short-MD Wall-Clock Diagnostic

## Question

Why did the original valid short-MD benchmark show a faster `NB F kernel` but a mixed or negative
whole-run wall result at `ntomp=2` and `ntomp=6` on the audited host?

## Measurement Basis

Benchmark driver:

- [`tools/pcff_respa_parity/bench_repulsion_power_9_simd_shortmd_cpu.py`](/home/kiket/Desktop/test/GROMACS_PCFF/tools/pcff_respa_parity/bench_repulsion_power_9_simd_shortmd_cpu.py)

Key controls now supported by the driver:

- explicit `--dlb {auto,no,yes}`
- `--alternate-mode-order` to remove fixed `generic -> specialized` ordering
- `--warmup-cycles-per-ntomp` to discard cold-transition runs after changing thread count

Metrics used:

- `ns/day`
- wallcycle `Force`
- wallcycle `PME mesh`
- wallcycle `Update`
- wallcycle subcounter `NB F kernel`
- wallcycle subcounter `NB F buffer ops.`

The benchmark shape is CPU-only non-MTS short MD and does execute the admitted short-range nonbonded
kernel path. This is not an exact-`r-RESPA` pair-splitting benchmark.

## Experiment Matrix

### Historical short run, now treated as insufficient

- [`output/repulsion_power_9_simd_shortmd_cpu_perf/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_simd_shortmd_cpu_perf/summary.md)
- `steps=200`, `repeats=1`, `pin=on`, `dlb=auto`, fixed mode order

This run is still valid for proving the kernel is reachable, but it is too short and too noisy for
wall-clock interpretation at `ntomp=2` and `ntomp=6`.

### DLB control sweep

- [`output/repulsion_power_9_shortmd_wall_diag_auto/`](</home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_shortmd_wall_diag_auto>)
- [`output/repulsion_power_9_shortmd_wall_diag_dlb_no/`](</home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_shortmd_wall_diag_dlb_no>)
- `steps=4000`, `repeats=5`, `pin=on`, completed evidence set `ntomp=1,2,6`

These runs show that `dlb=auto` versus `dlb=no` does not materially change the medians. DLB is not
the dominant cause of the earlier mixed wall result.

### Clean fixed-`ntomp` runs

- [`output/repulsion_power_9_shortmd_ntomp2_only_clean/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_shortmd_ntomp2_only_clean/summary.md)
- [`output/repulsion_power_9_shortmd_ntomp6_only_clean/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_shortmd_ntomp6_only_clean/summary.md)
- `steps=2000`, `repeats=6`, `pin=on`, `dlb=no`, alternating mode order

### Clean fixed-`ntomp` run with explicit warmup

- [`output/repulsion_power_9_shortmd_ntomp2_warmup_clean/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_shortmd_ntomp2_warmup_clean/summary.md)
- `steps=2000`, `repeats=3`, `pin=on`, `dlb=no`, alternating mode order, `warmup-cycles-per-ntomp=1`

## Findings

### 1. The old `ntomp=2/6` wall regression signal was a benchmark artifact

The old short benchmark used `steps=200` and a single repeat. That left the wall result vulnerable to
cold-start and transition effects that were larger than the specialized-vs-generic kernel delta.

### 2. DLB is not the dominant cause

The `steps=4000`, `repeats=5` sweeps with `dlb=auto` and `dlb=no` produced nearly identical medians
for `ntomp=1`, `2`, and `6`. Blaming DLB would be speculation.

### 3. The real artifact is the cross-`ntomp` transition, especially at `ntomp=2`

The mixed run that swept `ntomp=1 -> 2 -> 6` showed abnormal slow repeats immediately after the
transition to `ntomp=2`, with simultaneous inflation of:

- `Force`
- `PME mesh`
- `Update`
- `NB F kernel`
- `NB F buffer ops.`

That is not the signature of a specialized microkernel defect. It is a cold-transition artifact in the
benchmark harness.

Evidence:

- In the mixed fixed-order clean run, `ntomp=2` had anomalously slow early repeats while later repeats
  stabilized:
  - [`output/repulsion_power_9_shortmd_wall_diag_clean_alt/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_shortmd_wall_diag_clean_alt/summary.md)
- In the fixed-`ntomp=2` run, the same system was stable and the specialized path won:
  - wall `1.091x`
  - `NB F kernel` `1.547x`
- With one explicit warmup cycle per `ntomp`, the fixed-`ntomp=2` result stayed stable and improved
  slightly further:
  - wall `1.130x`
  - `NB F kernel` `1.582x`

### 4. On clean fixed-`ntomp` runs, the specialized path does improve wall time

Measured on the audited `gate_h_dense_salt_polymer_2x2x2` short-MD host-local benchmark:

- `ntomp=2`
  - wall `1.091x`
  - `NB F kernel` `1.547x`
- `ntomp=6`
  - wall `1.060x`
  - `NB F kernel` `1.529x`

The specialized path still does not justify a broad CPU-wide claim. But the earlier host-local
`ntomp=2/6` regression claim is no longer defensible on the cleaned-up measurement basis.

## Ranked Cause Analysis

1. benchmark harness transition artifact after changing `ntomp`
2. too-short run length and insufficient repeats in the original wall benchmark
3. fixed mode ordering without warmup, which allowed cold-start effects to contaminate one mode more than the other
4. DLB

`PME mesh` remains a large share of wall time, but it is not the root cause of the old regression
story. The old story was invalid because the benchmark protocol was weak.

## Recommendation

Use the short-MD benchmark only with these controls for wall-clock interpretation:

- fixed `ntomp` per run set
- alternating mode order
- at least one warmup cycle per `ntomp`
- multi-repeat medians
- `NB F kernel` and total wall reported together

Without those controls, the benchmark should be treated as kernel-reachability evidence only, not as
wall-scaling evidence.

## Claim Boundary

Allowed:

- `On the audited Ryzen 9 9900X host, the specialized repulsion-power-9 SIMD path improves NB F kernel time and also improves wall time on cleaned fixed-ntomp short-MD runs at ntomp=2 and ntomp=6.`

Not allowed:

- `the old 200-step single-repeat benchmark proved a real ntomp=2/6 regression`
- `DLB was shown to be the dominant cause`
- `the specialized path is broadly faster on CPU across hosts or chemistries`
