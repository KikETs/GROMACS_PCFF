# Repulsion-Power-9 `ntomp=6` Regression Analysis

## Executive Decision

`BOUND`, not `FIX`, is the evidence-based next step.

The audited `ntomp=6` regression is not supported as a stable specialized-microkernel defect. The strongest supported explanation is more fundamental: the audited exact-`r-RESPA` pair-splitting CPU performance fixtures do not execute the admitted specialized SIMD short-range nonbonded kernel in the timed real-space force path. They route through the exact pair-splitting CPU patch path instead, so the earlier `generic` vs `specialized` throughput deltas were not valid evidence about specialized-microkernel scaling.

## Experiment Matrix

- Small fixtures, `pin=on`, `ntomp=1,2,4,6,8,12`, `repeats=2`:
  - [`output/repulsion_power_9_scaling_diagnostic/pin_on/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/pin_on/summary.md)
- Small fixtures, `pin=off`, `ntomp=1,2,4,6,8,12`, `repeats=2`:
  - [`output/repulsion_power_9_scaling_diagnostic/pin_off/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/pin_off/summary.md)
- Small fixtures, repeat-depth probe, `pin=on`, `ntomp=2,6`, `repeats=6`:
  - [`output/repulsion_power_9_scaling_diagnostic/pin_on_repeatdepth_2_6/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/pin_on_repeatdepth_2_6/summary.md)
- Larger charged relevance check, `pin=on`, `ntomp=2,6`, `repeats=2`:
  - [`output/repulsion_power_9_scaling_diagnostic/gate_h_pin_on/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/gate_h_pin_on/summary.md)

## Key Findings

1. The audited performance fixtures are all exact `r-RESPA` pair-splitting runs.

- The performance `.mdp` files all set:
  - `mts = yes`
  - `mts-mode = lammps-respa`
  - `mts-levels = 3`
  - `mts-respa-inner/middle/outer-*`
- Basis:
  - [`output/repulsion_power_9_scaling_diagnostic/pin_on/small_oligomer/exact_respa_specialized_perf.mdp`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/pin_on/small_oligomer/exact_respa_specialized_perf.mdp:25)
  - [`output/repulsion_power_9_scaling_diagnostic/pin_on/small_salt_polymer_box/exact_respa_specialized_perf.mdp`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/pin_on/small_salt_polymer_box/exact_respa_specialized_perf.mdp:25)
  - [`output/repulsion_power_9_scaling_diagnostic/gate_h_pin_on/gate_h_dense_salt_polymer_2x2x2/exact_respa_specialized_perf.mdp`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/gate_h_pin_on/gate_h_dense_salt_polymer_2x2x2/exact_respa_specialized_perf.mdp:25)

2. Exact `r-RESPA` pair splitting on CPU bypasses `dispatchNonbondedKernel()` and calls the exact CPU patch path directly.

- In `do_force()`, when exact `r-RESPA` pair splitting is active and GPU nonbonded offload is not used, the code calls `computeExactRespaNonbondedCpu(...)`.
- That is a different path from the admitted SIMD short-range kernel dispatch.
- Basis:
  - [`src/gromacs/mdlib/sim_util.cpp`](/home/kiket/Desktop/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp:8189)

3. The exact CPU patch path uses a plain pairlist and scalar nonbonded math, including `std::pow(rinv, repulsionPower)` for non-12 repulsion.

- `computeExactRespaNonbondedCpu(...)` asserts the presence of a plain pairlist and then iterates the pair entries directly.
- Its repulsion-power handling is scalar:
  - `repulsionPower == 12 ? rinvsix * rinvsix : std::pow(rinv, repulsionPower)`
- This is not the admitted specialized SIMD short-range kernel path.
- Basis:
  - [`src/gromacs/mdlib/sim_util.cpp`](/home/kiket/Desktop/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp:3546)
  - [`src/gromacs/mdlib/sim_util.cpp`](/home/kiket/Desktop/test/GROMACS_PCFF/src/gromacs/mdlib/sim_util.cpp:4325)

4. The subcounter-enabled probe is consistent with that bypass: `NB F kernel` does not appear, even on the larger charged fixture.

- In all audited subcounter runs, the `Breakdown of PP / PME activities` section shows:
  - `Bonded F`
  - `Listed buffer ops.`
  - `NB X buffer ops.`
  - `Clear force buffer`
- But it does not show:
  - `NB F kernel`
  - `NB F buffer ops.`
- This is exactly what you would expect if the timed real-space exact-`r-RESPA` force work is not flowing through `dispatchNonbondedKernel()`.
- Basis:
  - [`output/repulsion_power_9_subcounter_probe/small_oligomer/ntomp6/generic/run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_subcounter_probe/small_oligomer/ntomp6/generic/run.log:1014)
  - [`output/repulsion_power_9_subcounter_probe/small_salt_polymer_box/ntomp6/generic/run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_subcounter_probe/small_salt_polymer_box/ntomp6/generic/run.log:1014)
  - [`output/repulsion_power_9_subcounter_probe/gate_h_dense_salt_polymer_2x2x2/ntomp6/generic/run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_subcounter_probe/gate_h_dense_salt_polymer_2x2x2/ntomp6/generic/run.log:626)

5. The original `pin=on` small-fixture matrix does show an audited `ntomp=6` regression on one tiny fixture.

- `small_oligomer`, `pin=on`, `ntomp=6`:
  - wall throughput `0.909x` specialized/generic
  - `Force` proxy `0.810x`
  - `Update` `0.871x`
  - `PME mesh` `1.011x`
- Basis:
  - [`pin_on/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/pin_on/summary.md:43)

6. The admission markers in the logs are real setup markers, but they do not prove specialized-kernel execution inside exact `r-RESPA` pair splitting.

- The logs still report:
  - `Using SIMD2xMM 4x8 nonbonded short-range kernels`
  - `Keeping the admitted generic CPU SIMD repulsion-power-9 path for baseline comparison.`
- Those messages describe setup and admission state.
- They do not override the actual exact-`r-RESPA` force-dispatch code path above.
- Basis:
  - [`output/repulsion_power_9_subcounter_probe/gate_h_dense_salt_polymer_2x2x2/ntomp6/generic/run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_subcounter_probe/gate_h_dense_salt_polymer_2x2x2/ntomp6/generic/run.log:439)
  - [`output/repulsion_power_9_subcounter_probe/gate_h_dense_salt_polymer_2x2x2/ntomp6/generic/run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_subcounter_probe/gate_h_dense_salt_polymer_2x2x2/ntomp6/generic/run.log:449)

7. `PME` is not the dominant cause of the small-fixture `ntomp=6` regression signal.

- In the strongest regression case above, `PME mesh` is effectively unchanged while `Force` and `Update` worsen materially.
- This rules out a simple “PME dominates and hides everything” explanation.

8. Cross-L3 locality loss is not the dominant cause for the pinned `ntomp=6` case.

- The pinned `ntomp=6` probe binds both generic and specialized runs to CPUs `0-5`.
- On this host, CPUs `0-5` sit within the same L3 instance.
- Basis:
  - [`output/ntomp6_affinity_probe/generic/run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/ntomp6_affinity_probe/generic/run.log:470)
  - [`output/ntomp6_affinity_probe/specialized/run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/ntomp6_affinity_probe/specialized/run.log:469)
  - host `lscpu` / sysfs cache-id checks recorded during diagnosis

9. Removing pinning collapses absolute throughput for both modes and removes the strongest relative regression signal.

- `small_oligomer`, `pin=off`, `ntomp=6` becomes near-parity:
  - wall throughput `1.008x`
  - `Force` proxy `1.008x`
  - `Update` `1.007x`
- But both modes scale much worse in absolute throughput than the pinned best cases.
- This means “turn pinning off” is not a credible fix. It only proves the regression is pinning-sensitive and host-local.
- Basis:
  - [`pin_off/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/pin_off/summary.md:43)

10. The small audited fixtures are too small to support a stable OpenMP scaling story on their own.

- `small_oligomer` has `6` atoms.
- `small_salt_polymer_box` has `10` atoms.
- Basis:
  - [`pin_on/small_oligomer/.../run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/pin_on/small_oligomer/ntomp6/generic/repeat1/run.log:467)
  - [`pin_on/small_salt_polymer_box/.../run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/pin_on/small_salt_polymer_box/ntomp6/generic/repeat1/run.log:467)

11. Repeat-depth data shows large run-to-run variance at the disputed points.

- `small_oligomer`, `pin=on`, `ntomp=6`, `repeats=6`:
  - generic mean `1139.280 ns/day`, CV `0.134`
  - specialized mean `978.774 ns/day`, CV `0.180`
- `small_salt_polymer_box`, `pin=on`, `ntomp=6`, `repeats=6`:
  - generic mean `857.771 ns/day`, CV `0.189`
  - specialized mean `937.215 ns/day`, CV `0.129`
- That is not the signature of a clean, deterministic microkernel regression. It is the signature of tiny-threaded host-local instability.
- Basis:
  - [`pin_on_repeatdepth_2_6/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/pin_on_repeatdepth_2_6/summary.md:25)

12. The larger charged relevance fixture does not reproduce the `ntomp=6` regression pattern, and its subcounter probe also lacks `NB F kernel`.

- `gate_h_dense_salt_polymer_2x2x2` has `2160` atoms.
- `pin=on`, `ntomp=2`: specialized/generic `1.013x`
- `pin=on`, `ntomp=6`: specialized/generic `1.004x`
- `Force` proxy is also near-parity or slightly positive for specialized at both thread counts.
- This shows the small-fixture regression is not a broadly stable large-fixture behavior on the audited host.
- Combined with the path audit above, it also shows the larger charged relevance run is still not a valid specialized-kernel timing probe under exact `r-RESPA` pair splitting.
- Basis:
  - [`gate_h_pin_on/summary.md`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/gate_h_pin_on/summary.md:25)
  - [`gate_h ... run.log`](/home/kiket/Desktop/test/GROMACS_PCFF/output/repulsion_power_9_scaling_diagnostic/gate_h_pin_on/gate_h_dense_salt_polymer_2x2x2/ntomp6/generic/repeat1/run.log:464)

## Ranked Cause Analysis

1. Most supported: the audited exact-`r-RESPA` pair-splitting CPU benchmark path bypasses the specialized SIMD short-range kernel, so the observed `generic` vs `specialized` deltas are not valid specialized-microkernel scaling evidence.

- Supported by:
  - exact `r-RESPA` performance `.mdp` settings
  - `do_force()` calling `computeExactRespaNonbondedCpu(...)`
  - scalar `std::pow(rinv, repulsionPower)` inside that exact CPU patch path
  - zero printed `NB F kernel` subcounter even in the larger charged probe

2. Strongly supported but secondary: tiny-fixture thread overhead and host-local run-to-run noise dominate the residual differences that remain after the path mismatch.

- Supported by:
  - `6` and `10` atom fixtures
  - high repeat-depth CV at `ntomp=6`
  - larger charged fixture removing the regression pattern

3. Secondary and fixture-specific: pinned-thread `Force`/`Update` interaction is where the visible small-fixture regression signal appears.

- Supported by:
  - `small_oligomer`, `pin=on`, `ntomp=6`
  - regression concentrated in `Force` and `Update`, not `PME`

4. Down-ranked: PME contribution can move the total result, but it is not the dominant cross-case explanation.

- Down-ranked because the strongest audited regression case has essentially neutral `PME mesh`.

5. Down-ranked: L3 locality loss across chiplets / cache groups.

- Down-ranked because pinned `ntomp=6` stays inside one L3 slice on this host.

6. Unproven: force-buffer false sharing or reduction-path pathology specific to the exact CPU patch path.

- `NB X buffer ops.` deltas are small.
- PMU counters were unavailable, so there is no direct cache-line or reduction-stall evidence.

## Fix-Or-Bound Recommendation

Choose `BOUND`.

Do not start a new specialized-microkernel rewrite based on this `ntomp=6` result. That would optimize the wrong code path. The evidence says the audited exact-`r-RESPA` pair-splitting CPU benchmark is not timing the specialized short-range SIMD kernel in the first place.

If a follow-up optimization track is opened later, it should target one of these instead:

- a valid CPU benchmark shape that actually executes `dispatchNonbondedKernel()` and produces non-zero `NB F kernel`
- a separate optimization track for `computeExactRespaNonbondedCpu(...)` if exact `r-RESPA` pair splitting is the real target runtime
- a deeper timing split inside the exact `r-RESPA` runtime around the CPU patch path and `Update`
- profiler access with PMU counters enabled

## Updated Claim Boundary

Allowed:

- “The previously observed exact-`r-RESPA` `ntomp=6` regression is not valid evidence about the specialized repulsion-power-9 SIMD microkernel.”
- “On the audited host, the exact-`r-RESPA` pair-splitting CPU benchmark routes through a separate exact CPU patch path, so its `generic` vs `specialized` timing deltas should remain out of scope for specialized-kernel scaling claims.”

Not allowed:

- “The specialized path has a generic `ntomp=6` CPU scaling defect.”
- “The audited exact-`r-RESPA` `ntomp=6` regression proves anything about specialized-kernel OpenMP scaling.”
- “The specialized path broadly improves or broadly harms multi-thread CPU performance.”
- any GPU, hybrid, or multi-host implication
