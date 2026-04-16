# Repulsion-Power-9 Short-MD Remote 12900K Check

## Scope

This note answers one narrow question:

- does the same `high-thread slowdown` observed on the audited Ryzen 9 9900X host also appear on
  the remote `i9-12900K` host
- and if it does, is it the same root cause

This is not a new capability note.

## Remote Host

- hostname: `user-Z690-AORUS-PRO`
- CPU: `12th Gen Intel(R) Core(TM) i9-12900K`
- L3 cache: `30 MiB (1 instance)`

Important topology difference from the audited 9900X host:

- the 9900X host has `64 MiB` L3 reported as `2 instances`
- the 12900K host reports a single `30 MiB` L3

So the 9900X explanation that depends on crossing two L3 groups cannot simply be copied here.

## Measurement Basis

Remote clean worktree:

- base code: `61fefe8f86`
- local unpushed PME follow-up was overlaid for:
  - [`src/gromacs/ewald/pme.cpp`](/home/kiket/Desktop/test/GROMACS_PCFF/src/gromacs/ewald/pme.cpp)
  - [`src/gromacs/ewald/pme_gather.cpp`](/home/kiket/Desktop/test/GROMACS_PCFF/src/gromacs/ewald/pme_gather.cpp)

Remote summaries copied back locally:

- mixed-core check:
  [`output/remote_shortmd_layout_check_12900k/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/remote_shortmd_layout_check_12900k/summary.md)
- P-core-only sweep:
  [`output/remote_shortmd_layout_check_12900k_pcore_only/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/remote_shortmd_layout_check_12900k_pcore_only/summary.md)

Benchmark shape:

- system: `gate_h_dense_salt_polymer_2x2x2`
- `steps=10000`
- `repeats=3`
- `pin=on`
- `dlb=no`
- alternating mode order
- one warmup cycle per layout

## Mixed-Core Check

Using the host's default CPU set:

| layout | ns/day | wall s |
| --- | ---: | ---: |
| `omp6` | `107.904` | `4.004` |
| `omp12` | `77.844` | `5.550` |
| `split12_pp6_pme6` | `95.751` | `4.512` |

Affinity evidence:

- `omp6`: `0-10:2`
- `omp12`: `0-22:2`
- `split12_pp6_pme6`:
  - rank 0: `0-10:2`
  - rank 1: `12-22:2`

Interpretation:

- `omp12` is not using a homogeneous 12-P-core team
- it spans `0,2,4,...,22`, which includes both P-core and E-core CPUs on this host
- the `split12` layout is also contaminated:
  - rank 1 lands on `12,14,16,18,20,22`
  - that set includes only two P-core CPUs (`12,14`) plus four E-core CPUs (`16,18,20,22`)

This immediately weakens any comparison that treats `omp12` here as “12 comparable CPU workers.”

## P-Core-Only Sweep

To remove the hybrid-core ambiguity, the remote benchmark was rerun under:

- `taskset -c 0,2,4,6,8,10,12,14`

That forces execution onto the eight P-core hardware threads only.

Results:

| layout | ns/day | wall s |
| --- | ---: | ---: |
| `omp6` | `109.401` | `3.949` |
| `omp8` | `113.313` | `3.813` |
| `split8_pp2_pme6` | `104.876` | `4.120` |
| `split8_pp3_pme5` | `139.540` | `3.096` |
| `split8_pp4_pme4` | `129.320` | `3.341` |
| `split8_pp5_pme3` | `113.325` | `3.812` |
| `split8_pp6_pme2` | `82.817` | `5.217` |

Best remote P-core-only point:

- `-ntmpi 2 -npme 1 -ntomp 3 -ntomp_pme 5`
- `139.540 ns/day`
- `3.096 s`

P-core-only affinity evidence:

- `omp8`: `0-14:2`
- `split8_pp3_pme5`:
  - rank 0: `0-4:2`
  - rank 1: `6-14:2`

That is a clean P-core-only placement.

## What This Proves

Two separate facts are now supported.

1. The remote host also shows a high-thread slowdown if you use the mixed `omp12` shape.
2. That slowdown is not the same root cause as the 9900X host-local issue.

The 12900K explanation is simpler:

- mixed `omp12` and mixed `split12` are contaminated by E-core placement
- once the run is restricted to P-cores only, the ranking changes
- the best remote result is not `omp12` and not `split12_pp6_pme6`
- the best remote result is a P-core-only split layout with `3 PP / 5 PME`

## Claim Boundary

Allowed conclusion:

- the 9900X and 12900K hosts do not support the same runtime recommendation
- the 9900X best current layout remains the audited `6/6` split on a `2x32 MiB L3` host
- the 12900K requires a separate P-core-only tuning envelope

Disallowed conclusion:

- “the same problem reproduced on the remote host”
- “the same fix should be used on both hosts”

## Current Remote Recommendation

For the remote `i9-12900K` host, on this short-MD PCFF shape:

1. do not use mixed-core `omp12` as a tuning reference
2. restrict the run to P-cores before comparing layouts
3. current best tested point is:
   - `taskset -c 0,2,4,6,8,10,12,14`
   - `-ntmpi 2 -npme 1 -ntomp 3 -ntomp_pme 5`

This is still host-local. It should not be generalized beyond this machine without separate evidence.
