# Exact r-RESPA CPU-Only Speedup Notes, 2026-04-22

## Scope

This note covers only the CPU-only exact r-RESPA Gate-I-shaped runtime:

- `-ntmpi 1`
- `-nb cpu -pme cpu -bonded cpu -update cpu`
- `GMX_PCFF_EXACT_RESPA_UPDATE_OMP=1`
- no GPU or hybrid claim
- no transport or production-readiness claim

The validation target remains mechanical/density-volume readiness only. Transport use remains outside this note.

## Source Changes

- `tools/pcff_respa_parity/validate_gate_g_long_ensemble.py`
  now allows `exact_respa_common_mdp(..., nstlist=...)` while preserving the historical default `nstlist=4`.
- `tools/pcff_respa_parity/validate_gate_i_charged_long_npt_conditioning.py`
  now exposes `--nstlist` and records `nstlist_base_steps` in the contract.
- `tools/pcff_respa_parity/validate_gate_i_charged_long_npt_conditioning.py`
  now blocks `full_owner_native` and `split_owner_sidecar` by default. These modes require
  `--allow-experimental-native-multi-probe` and are not accepted as Gate I exactness evidence.
- `tools/pcff_respa_parity/validate_gate_i_charged_long_npt_conditioning.py`
  now rejects `--exact-respa-fused-initial-drift-mode on` for claimable Gate I runs unless
  `--allow-experimental-update-probe` is explicitly set. This matches the existing boundary that
  fused initial drift is default-off probe code, not replica-ready evidence.
- `tools/pcff_respa_parity/validate_gate_i_charged_long_npt_conditioning.py`
  now supports `--common-precondition-ps`, which generates a provenance-recorded common exact-NPT
  preconditioned start before independent replica equilibration. This is input preparation only;
  Gate I acceptance still depends on independent production density/volume observables.
- `src/gromacs/mdrun/exactrespastepper.cpp`
  reduces repeated velocity stores in the 2-kick and 3-kick direct update fast paths while preserving
  the original operation order.

## Evidence

Build:

- `cmake --build build --target gmx -j 12` passed.
- `python3 -m py_compile tools/pcff_respa_parity/validate_gate_g_long_ensemble.py tools/pcff_respa_parity/validate_gate_i_charged_long_npt_conditioning.py` passed.
- Fused initial drift rejection probe passed: a prepare-only Gate I invocation with
  `--exact-respa-fused-initial-drift-mode on` exits before writing claimable evidence unless
  `--allow-experimental-update-probe` is supplied.
- Common-preconditioning smoke run passed locally: a 4 ps precondition + 4 ps equilibration +
  4 ps production run generated precondition, replica summaries, and equilibration diagnostics.
  This is code-path smoke evidence only, not density/volume convergence evidence.

Update fast path parity:

- Report: `output/exact_respa_cpu_only_speedups_20260422/update_direct_store_rewrite_parity_steps1000/report.json`
- `direct_on` vs `direct_off` force, per-level force, energy, and GRO deltas were all `0.0`.
- Performance was not improved in this probe: `direct_on/direct_off = 0.9985x`.

Update thread-cap probe:

- Report: `output/exact_respa_cpu_only_speedups_20260422/update_thread_cap_confirm_ntomp12_pinoff_steps10000/report.json`
- With `ntomp=12`, cap `12` was faster than cap `1`: `67.556 ns/day` vs `63.441 ns/day`.
- Lower caps are not a supported speedup on this host/runtime shape.

Native multi no-fallback:

- Report: `output/exact_respa_cpu_only_speedups_20260422/full_native_no_fallback_reprobe_ntomp12_steps1000_noprobe/report.json`
- It is not exact enough for Gate I evidence: total-force max component delta `1.2072906494140625`, per-level-force max component delta `1.20941162109375`, energy max delta `0.6600000000000819`, GRO mismatch.
- Serial reduction did not fix the delta:
  `output/exact_respa_cpu_only_speedups_20260422/full_native_no_fallback_serialred_ntomp12_steps1000/report.json`.
- Plain-C/ntomp1 still produced nonzero force deltas:
  `output/exact_respa_cpu_only_speedups_20260422/full_native_no_fallback_plainc_ntomp1_steps200/report.json`.

Owner/middle fallback:

- Report: `output/exact_respa_cpu_only_speedups_20260422/owner_and_middle_fallback_parity_ntomp12_steps1000/report.json`
- Force, per-level force, energy, and GRO deltas were `0.0`.
- This is the claimable exactness path for the current Gate I validation run.

`nstlist` short performance probe, owner-fallback exactness path:

- `nstlist=4`: mean prod throughput `74.525 ns/day`
- `nstlist=8`: mean prod throughput `76.364 ns/day`
- `nstlist=12`: mean prod throughput `76.673 ns/day`
- `nstlist=20`: mean prod throughput `77.226 ns/day`

These are short probes only. They all fail the Gate I temperature criterion because the run is intentionally too short
(`2 ps` equilibration and `4 ps` production). They are performance-screening evidence, not density/volume convergence evidence.

## Failed Long Validation

The long validation run below completed, but it is not claimable Gate I closure:

- Output:
  `tests/reference_results/gate_i_charged_long_npt_conditioning_eq750_prod10000_ntmpi1_ntomp12_pinoff_ownerfallback_updatefastpath_fused_nst20_cpuonly_20260422`
- Launch log:
  `output/exact_respa_cpu_only_speedups_20260422/launch_logs/gate_i_nst20_ownerfallback_cpuonly_20260422.stdout.txt`
- Exit-code file:
  `output/exact_respa_cpu_only_speedups_20260422/launch_logs/gate_i_nst20_ownerfallback_cpuonly_20260422.exit`
- PID file:
  `output/exact_respa_cpu_only_speedups_20260422/launch_logs/gate_i_nst20_ownerfallback_cpuonly_20260422.pid`

Settings:

- `equil-ps = 750`
- `prod-ps = 10000`
- `replicas = 3`
- `ntmpi = 1`
- `ntomp = 12`
- `pin = off`
- `native-multi-owner-mode = owner_fallback`
- `exact-respa-fused-initial-drift-mode = on`
- `nstlist = 20`

Result:

- Status: `FAIL`
- Density cross-replica relative span: `0.13348446164476266`
- Volume cross-replica relative span: `0.1386428182212153`
- Replica density means: `1280.9077134163317`, `1467.6754991798564`, `1448.9342391480372`
- Replica volume means: `45.52199006593868`, `39.722945847163054`, `40.23673619607608`
- Temperature cross-replica relative span: `0.0012181330046100419`

Root-cause boundary:

- This is not a temperature-control failure.
- `replica_01` remains in a lower-density / higher-volume basin across the production blocks; this is
  not fixable by discarding only the first production block.
- The run also used fused initial drift, which was already documented as default-off probe code and
  should not have been accepted as claimable Gate I evidence. The validator now rejects that mode unless
  the run is explicitly marked as experimental.

## Completed Claimable Gate I Validation

The conservative claimable runtime path subsequently passed Gate I:

- `native-multi-owner-mode = owner_fallback`
- `exact-respa-fused-initial-drift-mode = off`
- `exact-respa-update-omp-mode = on`
- `exact-respa-update-direct-fastpath-mode = on`
- `nstlist = 20`
- `ntmpi = 1`
- `ntomp = 12`
- `pin = off`
- `common-precondition-ps = 3000`
- `equil-ps = 750`
- `prod-ps = 10000`
- `replicas = 3`
- no `--allow-experimental-native-multi-probe`
- no `--allow-experimental-update-probe`

Output:

- `tests/reference_results/gate_i_charged_long_npt_conditioning_commonpre3000_eq750_prod10000_ntmpi1_ntomp12_pinoff_ownerfallback_updatefastpath_nst20_cpuonly_20260423`

Result:

- Status: `PASS`
- Density cross-replica relative span: `0.0193330420978208`
- Density max replica absolute block drift relative: `0.018262018483332174`
- Density mean absolute block drift relative: `0.007775726335315165`
- Volume cross-replica relative span: `0.01935822991481781`
- Volume max replica absolute block drift relative: `0.01835889789722278`
- Volume mean absolute block drift relative: `0.007818995993356963`
- Temperature mean absolute error: `1.4744566379606 K`
- Conditioned-state handoff candidate: `replica_02`

This closes the density/volume conditioning blocker for this CPU-only exact-r-RESPA Gate I shape. It
does not imply transport production readiness, TP0 uncertainty closure, or LAMMPS-vs-GROMACS transport
parity.

## Claim Boundary

Supported now:

- The direct update store rewrite is parity-safe.
- Update thread cap reduction is not beneficial on the measured host/runtime.
- Full native multi and split-owner sidecar remain non-claimable for Gate I exactness.
- Owner/middle fallback remains the claimable exactness path.
- `nstlist=20` is now part of the passed CPU-only Gate I conditioning shape.
- Common preconditioning is an auditable input-preparation mechanism; acceptance still depends on the
  independent replica production density/volume observables.
- CPU-only Gate I density/volume conditioning is closed for the recorded host-local shape.

Not supported:

- CPU-only production readiness from these probes alone.
- Transport-readiness wording.
- Full native multi exact-r-RESPA completion.
- Broad CPU scaling guidance from this single host and fixture.
