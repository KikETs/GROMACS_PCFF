# Exact r-RESPA Pair-Loop OpenMP Speedup Evidence

This directory records the first host-local evidence for the experimental
`GMX_PCFF_EXACT_RESPA_PAIRLOOP_OMP=1` fast path.

The evidence supports only a narrow claim:

- host: `KikET`, AMD Ryzen 9 9900X
- runtime: exact r-RESPA CPU path, `computeExactRespaNonbondedCpu()`
- fixture: Gate I charged NPT equilibration TPR, replica 01
- shape: `-ntmpi 1`, `-nb cpu -pme cpu -bonded cpu -update cpu`, `-pin off`, `-reprod`
- tracked fixture TPR: `tests/reference_results/gate_i_charged_long_npt_conditioning_eq750_prod10000_split12_pp6_pme6_local/cpu/replica_01/equil.tpr`
- fixture TPR SHA256: `54dab590e7e0d07828572c92618eb7dc574c0cfa80722a0f0aac8ca725ecd935`
- optimization: no-trace/no-energy/no-virial pair-loop OpenMP fast path using thread-private force/shift buffers and deterministic fixed-order reduction
- vector mode: `GMX_PCFF_EXACT_RESPA_PAIRLOOP_VECTOR=1`, fixed-width compiler-SIMD batch math followed by deterministic scatter into the same force/shift accumulation strategy

Unsupported:

- GPU or hybrid implications
- broad CPU scaling guidance
- production transport readiness
- density/volume blocker closure

Supported with limitations:

- vector-only speedup exists on the audited host/fixture, but is smaller than the OpenMP pair-loop speedup
- combined OpenMP+vector gives negligible additional whole-run gain over OpenMP alone in the first 2000-step Gate I sweep
- this is not a broad SIMD claim and does not imply other architectures or fixtures

Validation limit:

- Legacy exact r-RESPA force-dump parity tests exercise the diagnostic serial
  fallback when trace/dump outputs are enabled.
- Fast-path direct force-component evidence now comes from
  `tools/pcff_respa_parity/validate_exact_respa_pairloop_force_delta.py`, which
  dumps pair-loop force-buffer deltas without entering the legacy trace flags
  that disable the fast path.
- The accepted Gate I `ntomp=6` force-delta criterion is bounded
  single-precision parity: pass if `abs_delta <= 1e-2` or
  `rel_delta <= 5e-5`.
- The current force-delta evidence compares 28 fast-path snapshots per
  candidate mode and 285120 force components per mode across `pairloop_omp`,
  `pairloop_vector`, and `combined`.
- Machine-readable local force-delta report:
  `local_9900x_gate_i_pairloop_force_delta_report.json`.
- Machine-readable remote Z690 force-delta report:
  `remote_z690_gate_i_pairloop_force_delta_report.json`.
- Maximum observed absolute pair-loop force-delta difference is `0.00604248046875`
  on the local 9900X run and `0.0048828125` on the remote Z690/i9-12900K run;
  the largest relative differences occur only on near-zero force components
  with absolute differences below `4e-5`.
- Final `.gro` output is byte-identical across baseline and all candidate modes
  within each 20-step force-delta harness.

Remote Z690 staged performance:

- Machine-readable report: `remote_z690_gate_i_pairloop_omp_report.json`.
- Host: `user-Z690-AORUS-PRO`, Intel i9-12900K.
- Runtime: same Gate I exact-r-RESPA CPU TPR, `-ntmpi 1`, `-pin off`,
  `-reprod`, 2000 steps, one repeat.
- Best same-thread remote OpenMP result: `pairloop_omp` `5.893 ns/day` at
  `ntomp=6`, `1.666x` versus the `ntomp=6` baseline.
- Best remote vector-only result: `4.409 ns/day` at `ntomp=6`, `1.246x` versus
  baseline.
- Combined OpenMP+vector remains effectively no better than OpenMP alone on the
  remote host: `5.857 ns/day` at `ntomp=6`, below the OpenMP-only row.

Remaining validation limit:

- This does not validate a parallel energy or virial fast path. Energy/virial
  pair-loop calls still deliberately use the serial path.
- Two measured hosts still do not justify broad CPU scaling guidance or
  production/transport readiness.

Experimental sparse-reduction / update-OMP probe:

- Machine-readable force-delta report:
  `local_9900x_gate_i_update_sparse_force_delta_report.json`.
- Machine-readable `ntomp=6` performance report:
  `local_9900x_gate_i_update_sparse_ntomp6_report.json`.
- `GMX_PCFF_EXACT_RESPA_UPDATE_OMP=1` parallelizes exact-r-RESPA VV half-kick
  and drift atom loops.
- `GMX_PCFF_EXACT_RESPA_PAIRLOOP_SPARSE_REDUCTION=1` enables guarded touched-atom
  pair-loop reduction, but dense pairlists fall back to the full reduction.
- Current local Gate I `ntomp=6` result: `pairloop_omp` `6.568 ns/day`,
  `pairloop_omp_update` `6.614 ns/day`, and `pairloop_sparse_update`
  `6.589 ns/day`.
- Verdict: Update OMP is exactness-clean in the short harness but only marginal
  for wall time here; sparse reduction is not useful for this dense Gate I
  pairlist.

Experimental block-reduction / tile probe:

- Machine-readable force-delta report:
  `local_9900x_gate_i_block_tile_force_delta_report.json`.
- Machine-readable `ntomp=6` performance report:
  `local_9900x_gate_i_block_tile_ntomp6_report.json`.
- The pair-loop fast path now uses separate audited kernels for
  `plainPairlist.pairs` and `plainPairlist.excludedPairs`.
- `GMX_PCFF_EXACT_RESPA_PAIRLOOP_BLOCK_REDUCTION=1` replaces the dense
  atom-by-atom post-reduction with a block-owned reduction sweep.
- `GMX_PCFF_EXACT_RESPA_PAIRLOOP_TILE=1` adds an experimental tile-local
  force scatter path for the standard `pairs` list only; `excludedPairs`
  remain on the non-tiled specialized path.
- Current local Gate I `ntomp=6`, 2000-step result: baseline
  `4.086 ns/day`, `pairloop_omp` `6.938 ns/day`, `pairloop_block`
  `6.977 ns/day`, and `pairloop_tile` `6.836 ns/day`.
- Maximum observed absolute pair-loop force-delta difference remains
  `0.00604248046875`, and final `run.gro` SHA256 is identical across
  baseline, `pairloop_block`, and `pairloop_tile`:
  `a00186d56c84f8105820521f8f0b7937630fc34226a0254f3691393c6672a313`.
- Verdict: block reduction is a small audited host-local gain over
  `pairloop_omp`; the current tile backend is exactness-clean but lands below
  the block-only and `pairloop_omp` rows on this dense Gate I fixture.

Block-indexed tile retry:

- Machine-readable force-delta report:
  `local_9900x_gate_i_blockindexed_tile_force_delta_report.json`.
- Machine-readable `ntomp=6` performance report:
  `local_9900x_gate_i_blockindexed_tile_ntomp6_report.json`.
- The tile cache was reworked to use a block-indexed cache plus per-thread
  cache reuse.
- Current local Gate I `ntomp=6`, 2000-step result: baseline
  `4.012 ns/day`, `pairloop_omp` `6.927 ns/day`, `pairloop_block`
  `6.942 ns/day`, and `pairloop_tile` `6.698 ns/day`.
- Verdict: the block-indexed cache did not rescue the tile backend; tile
  remains slower than the simpler `pairloop_block` path on this fixture.

Experimental nbnxm-style 4x4 prototype:

- Machine-readable force-delta report:
  `local_9900x_gate_i_nbnxm4x4_force_delta_report.json`.
- Machine-readable `ntomp=6` performance report:
  `local_9900x_gate_i_nbnxm4x4_ntomp6_report.json`.
- `GMX_PCFF_EXACT_RESPA_PAIRLOOP_NBNXM4X4=1` enables a bounded `standard pairs`
  4x4 cluster prototype. It rebuilds 4x4 clusters every eligible call and keeps
  `excludedPairs` on the existing specialized path.
- Current local Gate I `ntomp=6`, 2000-step result: baseline
  `4.058 ns/day`, `pairloop_omp` `7.012 ns/day`, `pairloop_block`
  `7.003 ns/day`, and `pairloop_nbnxm4x4` `4.284 ns/day`.
- The short exactness harness still passes. The maximum observed absolute
  pair-loop force-delta difference for `pairloop_nbnxm4x4` is
  `0.00531005859375`.
- Verdict: this bounded nbnxm-style prototype is mechanically acceptable in the
  short harness, but it is not a viable performance path. It barely clears the
  scalar baseline and loses badly against the current OpenMP fast path.

Direct CPU-list consumer backend:

- Machine-readable force-delta report at `ntomp=6`:
  `local_9900x_gate_i_direct_cpulist_force_delta_ntomp6_report.json`.
- Machine-readable force-delta report at `ntomp=12`:
  `local_9900x_gate_i_direct_cpulist_force_delta_ntomp12_report.json`.
- Machine-readable `ntomp=6` performance report:
  `local_9900x_gate_i_direct_cpulist_ntomp6_report.json`.
- Machine-readable `ntomp=2,12` performance report:
  `local_9900x_gate_i_direct_cpulist_ntomp2_12_report.json`.
- Machine-readable `ntomp=12` direct-vs-block report:
  `local_9900x_gate_i_direct_cpulist_ntomp12_direct_vs_block_report.json`.
- `GMX_PCFF_EXACT_RESPA_PAIRLOOP_DIRECT_CPULIST=1` switches the exact CPU
  runtime from `plainPairlist` materialization to direct `PairlistSet::cpuLists()`
  consumption when the existing pair-loop fast-path contract is active.
- This is not a full nbnxm kernel migration. Pair admission is matched to
  `appendPlainPairlistCpu()` using `nbat` coordinates and `nbat.shift_vec`,
  while the admitted-pair force geometry still uses the exact runtime
  `coordinates` and `fr->shift_vec`.
- Exactness currently passes at the validated thread counts:
  `ntomp=6` max abs force-delta `0.00604248046875`,
  `ntomp=12` max abs force-delta `0.0006103515625`.
- Current local Gate I performance summary:
  `ntomp=1` baseline `3.763 ns/day`;
  `ntomp=2` baseline `3.947`, `pairloop_block` `5.800`,
  `pairloop_direct_block` `5.770`;
  `ntomp=6` baseline `4.103`, `pairloop_omp` `6.971`,
  `pairloop_block` `6.988`, `pairloop_direct` `9.750`,
  `pairloop_direct_block` `9.774`;
  `ntomp=12` baseline `4.055`, `pairloop_block` `6.951`,
  `pairloop_direct` `11.221`, `pairloop_direct_block` `11.289`.
- Verdict: on this audited host and fixture, the direct CPU-list backend gives
  a real exact-runtime host-local gain at `ntomp=6` and `ntomp=12`. It is not
  a universal win at every OpenMP point, and it is not evidence for a broad
  full-nbnxm migration claim.

- Follow-up refinement reports:
  `local_9900x_gate_i_direct_flags_fullmask_partition_force_delta_ntomp2_report.json`,
  `local_9900x_gate_i_direct_flags_fullmask_partition_force_delta_ntomp6_report.json`,
  `local_9900x_gate_i_direct_flags_fullmask_partition_force_delta_ntomp12_report.json`,
  and `local_9900x_gate_i_direct_flags_fullmask_partition_ntomp2_6_12_report.json`.
- The refinement keeps the same direct CPU-list scope, but adds three bounded
  optimizations inside the exact runtime:
  `ci.shift` flag consumption, full-mask suffix fast-path use, and direct
  `cpuLists()` partition consumption without flattening each `ci` into a
  separate work item.
- Exactness passes at the newly audited thread counts:
  `ntomp=2` max abs force-delta `0.00054931640625`,
  `ntomp=6` `0.00604248046875`,
  `ntomp=12` `0.00189971923828125`.
- Refined local Gate I performance summary:
  `ntomp=2` baseline `3.941`, `pairloop_block` `5.744`,
  `pairloop_direct` `7.138`, `pairloop_direct_block` `7.137`;
  `ntomp=6` baseline `4.063`, `pairloop_block` `6.901`,
  `pairloop_direct` `11.331`, `pairloop_direct_block` `11.510`;
  `ntomp=12` baseline `3.995`, `pairloop_block` `6.872`,
  `pairloop_direct` `12.593`, `pairloop_direct_block` `12.423`.
- Verdict: on this audited host and fixture, the refined direct CPU-list
  backend now has measured wins over `pairloop_block` at `ntomp=2`, `6`, and
  `12`. This still does not justify a broad CPU scaling claim or a full nbnxm
  migration claim.

- Packed-cluster and bounded-dispatch follow-up reports:
  `local_9900x_gate_i_direct_packed_dispatch_force_delta_ntomp2_report.json`,
  `local_9900x_gate_i_direct_packed_dispatch_force_delta_ntomp6_report.json`,
  `local_9900x_gate_i_direct_packed_dispatch_force_delta_ntomp12_report.json`,
  and `local_9900x_gate_i_direct_packed_dispatch_ntomp2_6_12_report.json`.
- This follow-up keeps the same exact runtime scope, but adds:
  packed cluster-force accumulation, bounded layout dispatch on the native CPU
  list shape, and a full-mask standard-pairs cluster microkernel path.
- Exactness passes at the audited thread counts:
  `ntomp=2`, `6`, and `12` max abs force-delta
  `0.00189971923828125`.
- Packed-dispatch local Gate I performance summary:
  `ntomp=2` baseline `3.933`, `pairloop_block` `5.825`,
  `pairloop_direct` `7.189`, `pairloop_direct_block` `7.226`;
  `ntomp=6` baseline `4.101`, `pairloop_block` `6.979`,
  `pairloop_direct` `11.623`, `pairloop_direct_block` `11.570`;
  `ntomp=12` baseline `4.038`, `pairloop_block` `6.916`,
  `pairloop_direct` `12.484`, `pairloop_direct_block` `12.575`.
- Verdict: on this audited host and fixture, the packed-cluster /
  bounded-dispatch follow-up improves the best direct CPU-list path again. This
  is still not a broad CPU scaling claim and still not a full nbnxm migration
  claim.

- Update-mode probe report:
  `update_mode_probe/local_9900x_gate_i_update_modes_ntomp6_report.json`
  and `update_mode_probe/local_9900x_gate_i_update_modes_ntomp6_report.tsv`.
- This probe targets exact-r-RESPA CPU update work in
  `src/gromacs/mdrun/exactrespastepper.cpp`, not the pair-loop itself.
- `GMX_PCFF_EXACT_RESPA_UPDATE_OMP` now supports three audited states:
  `baseline=0`, `update_auto=<unset>`, and `update_omp=1`.
- Exactness evidence for this update-only probe is `run.gro` and `run.edr`
  SHA256 identity across all three modes, not pair-loop force-delta snapshots.
- Current local Gate I `ntomp=6`, 2000-step summary:
  `baseline` `25.516 ns/day`, `update_auto` `26.752 ns/day`,
  `update_omp` `26.795 ns/day`.
- Verdict: the exact-r-RESPA update path now has a small but real host-local
  gain over forced-serial update, and `auto` is close enough to forced `on` to
  serve as the defensible default. This still does not support broader CPU,
  restart-bitwise, density, or transport claims.

- Complete-pairlist contract report:
  `complete_pairlist_contract_probe/local_9900x_gate_i_complete_pairlist_contract_report.json`
  and `complete_pairlist_contract_probe/local_9900x_gate_i_complete_pairlist_contract_report.tsv`.
- This probe targets exact-r-RESPA pair-search / pair-loop contract wiring in
  `runner.cpp`, `forcerec.h`, `sim_util.cpp`, and `exactrespa_nonbonded_gpu.cpp`.
- `plainPairlistRange` now means MD-module signal/materialization range, while
  `completePairlistRange` means the pair-search contract that requires
  `includeAllPairs`.
- Exact CPU/GPU r-RESPA now consume `completePairlistRange`, so exact pair
  admission no longer depends on the presence of an MD-module plain-pairlist
  range.
- Benchmark and force-delta scripts explicitly set
  `GMX_PCFF_EXACT_RESPA_PAIRLOOP_OMP=0` and
  `GMX_PCFF_EXACT_RESPA_PAIRLOOP_DIRECT_CPULIST=0` for baseline and non-direct
  modes so staged reports remain honest.
- `GMX_PCFF_EXACT_RESPA_EAGER_PLAIN_PAIRLIST=1` still forces legacy eager
  materialization for exact-only runs by materializing at the complete-pairlist
  range when no MD-module signal range exists.
- Audited local Gate I `ntomp=6`, `2000 steps x 2 repeats` averages:
  `default_auto` `85.5245 ns/day`, explicit `on` `84.7455`,
  forced-off baseline `85.3810`, forced eager-materialization `29.8505`.
- Exactness evidence for this contract probe is `run.gro` / `run.edr` SHA256
  identity across all audited modes and repeats.
- `run.cpt` hashes differ, so restart-bitwise identity is not claimed.
- Verdict: this closes the contract ambiguity and proves that unnecessary eager
  plain-pairlist materialization is very expensive, but it does not support a
  stable short-run speedup claim for `default_auto` over forced-off baseline on
  this audited shape.

- Owner-step native-multi runtime reports:
  `native_multi_owner_probe/local_9900x_gate_h_owner_native_multi_runtime_report.json`,
  `native_multi_owner_probe/local_9900x_gate_h_owner_native_multi_runtime_report.tsv`,
  `native_multi_owner_probe/local_9900x_gate_i_owner_native_multi_runtime_report.json`,
  and
  `native_multi_owner_probe/local_9900x_gate_i_owner_native_multi_runtime_report.tsv`.
- This pass extends native multi from force-only exact narrow steps to owner
  steps that also request real-space energy and direct virial.
- Owner-step energy stays on the existing kernel energy arrays, while
  direct-virial reconstruction now reads the owning native contribution's
  `fshift` buffers after contribution-indexed force reduction.
- Unit evidence:
  `nbnxm-output-contract-test --gtest_filter='*Native*:*ExactRespa*'` PASS and
  `nbnxm-test --gtest_filter='*ExactRespaNativeMultiForceOnlyMatchesPerContributionLaunch:*ExactRespaNativeMultiOwnerEnergyMatchesPerContributionLaunch:*ExactRespaNativeMultiForceOnlyKeepsExcludedPmeCorrectionOuterOnly'` PASS.
- Runtime evidence:
  `gate_h` `52.591 -> 58.553 ns/day` (`1.113x`),
  `gate_i` `52.638 -> 64.535 ns/day` (`1.226x`).
- Honest boundary:
  2000-step `gro/edr/cpt` hashes still diverge, so there is no whole-run
  identity claim.
  The defensible claim is same-state continuation parity with small
  force/virial/pressure deltas plus a host-local owner-step runtime gain.

- Native-multi divergence-onset reports:
  `native_multi_divergence_onset_probe/local_9900x_gate_h_native_multi_divergence_onset_report.json`,
  `native_multi_divergence_onset_probe/local_9900x_gate_h_native_multi_divergence_onset_report.tsv`,
  `native_multi_divergence_onset_probe/local_9900x_gate_i_native_multi_divergence_onset_report.json`,
  and
  `native_multi_divergence_onset_probe/local_9900x_gate_i_native_multi_divergence_onset_report.tsv`.
- The onset harness reruns the same TPR from step `0` at increasing lengths and
  compares per-launch vs native-multi without rebuilding an alternate runtime
  shape.
- `gate_h` onset summary:
  force delta stays at `5.4931640625e-04` through `64` steps, rises to
  `3.2586669921875e-01` at `256`, `1.7891845703125` at `1024`, and
  `13.73583984375` at `2000`.
- `gate_i` onset summary:
  force delta stays at `5.4931640625e-04` through `256` steps, rises to
  `2.182708740234375` at `1024`, and `6.3876953125` at `2000`.
- Both fixtures already lose `edr` hash identity by `4` steps and `gro` hash
  identity by `64` steps, but the large force deltas appear only later.
- Honest reading:
  this looks like gradual reduction-order trajectory branching, not an
  immediate owner-step semantic failure.

- Pair-loop force-dump decoupling:
  `src/gromacs/mdlib/sim_util.cpp` no longer forces `PlainPairlist`
  materialization merely because `GMX_PCFF_EXACT_RESPA_PAIRLOOP_FORCE_DUMP_DIR`
  is set.
- Force-dump headers now report `plain_pair_count_available=true/false` so the
  dump does not silently claim empty-list counts when the runtime stayed on the
  direct `cpuLists()` path.

- Serial-reduction probe reports:
  `native_multi_serial_reduction_probe/local_9900x_gate_h_native_multi_serial_reduction_report.json`,
  `native_multi_serial_reduction_probe/local_9900x_gate_h_native_multi_serial_reduction_report.tsv`,
  `native_multi_serial_reduction_probe/local_9900x_gate_i_native_multi_serial_reduction_report.json`,
  and
  `native_multi_serial_reduction_probe/local_9900x_gate_i_native_multi_serial_reduction_report.tsv`.
- These runs set `GMX_PCFF_EXACT_RESPA_NBNXM_SERIAL_REDUCTION=1` and repeat the
  native-multi onset scan.
- Honest result:
  gate_h/gate_i onset curves are unchanged versus the default scans, so the
  final NBNXM output-buffer reduction is not the driver of the later
  whole-run divergence.

- Scalar pair-loop force-delta harness restore:
  `validate_exact_respa_pairloop_force_delta.py` and
  `bench_exact_respa_pairloop_omp.py` now export
  `GMX_PCFF_EXACT_RESPA_DISABLE_NBNXM_NARROW=1` so old scalar pair-loop probes
  cannot silently run on the narrow NBNXM runtime.
- Restored report:
  `scalar_pairloop_force_delta_probe/local_9900x_gate_h_pairloop_direct_force_delta_ntomp6_report.json`
- That restored scalar report passes.
- Snapshot-level decoupling evidence:
  at `pairloop_direct` dump `ordinal 2` / `step 1`,
  `pair_fast_path_used=true`, `excluded_pair_fast_path_used=true`,
  `compute_pair_energies=false`, `compute_virial=false`, and
  `plain_pair_count_available=false`.
- This is the direct proof that pair-loop force-dump no longer forces
  `PlainPairlist` materialization on the active direct fast path.

- Native-multi `ntomp=1` divergence probes:
  `native_multi_ntomp1_divergence_probe/local_9900x_gate_h_native_multi_ntomp1_divergence_report.json`,
  `native_multi_ntomp1_divergence_probe/local_9900x_gate_h_native_multi_ntomp1_divergence_report.tsv`,
  `native_multi_ntomp1_divergence_probe/local_9900x_gate_i_native_multi_ntomp1_divergence_report.json`,
  and
  `native_multi_ntomp1_divergence_probe/local_9900x_gate_i_native_multi_ntomp1_divergence_report.tsv`.
- These runs repeat the onset scan at `ntomp=1` without changing the audited
  runtime shape otherwise.
- Honest result:
  `ntomp=1` does not close the later divergence.
  `gate_h` step-`2000` force delta improves only to `3.6497650146484375`,
  while `gate_i` worsens dramatically to `765.7132415771484`.

- Native-multi plain-C reference-kernel probes:
  `native_multi_plainc_divergence_probe/local_9900x_gate_h_native_multi_plainc_ntomp1_divergence_report.json`,
  `native_multi_plainc_divergence_probe/local_9900x_gate_h_native_multi_plainc_ntomp1_divergence_report.tsv`,
  `native_multi_plainc_divergence_probe/local_9900x_gate_i_native_multi_plainc_ntomp1_divergence_report.json`,
  and
  `native_multi_plainc_divergence_probe/local_9900x_gate_i_native_multi_plainc_ntomp1_divergence_report.tsv`.
- These runs set `GMX_DISABLE_SIMD_KERNELS=1` and confirm the log marker
  before comparing per-launch vs native-multi.
- Honest result:
  disabling SIMD kernels also does not close the divergence.
  `gate_h` step-`2000` force delta becomes `14.1541748046875`,
  `gate_i` becomes `1278.444320678711`, and throughput collapses to about
  `1.2 ns/day`.
- Therefore the remaining divergence cannot honestly be blamed only on OpenMP
  thread fan-out or only on SIMD kernels.
  The strongest remaining source hypothesis is native-multi contribution
  interleaving / arithmetic grouping itself.

- Dense-dump owner-step closure:
  `native_multi_owner_fallback_probe/local_9900x_gate_h_owner_fallback_dense_1024_report.json`
  and
  `native_multi_owner_fallback_probe/local_9900x_gate_i_owner_fallback_dense_1024_report.json`.
- These reports use the corrected owner-level detection plus
  `GMX_EXACT_RESPA_FORCE_DUMP_INTERVAL=1`.
- Honest result:
  owner-only fallback closes dense total/per-level force, energy, and gro
  through `1024` steps for both audited fixtures.
  Middle-only fallback does not; it mismatches immediately from step `0`.

- Default owner-fallback runtime closure:
  `native_multi_owner_fallback_probe/local_9900x_gate_h_default_owner_fallback_runtime_report.json`,
  `native_multi_owner_fallback_probe/local_9900x_gate_i_default_owner_fallback_runtime_report.json`,
  and
  `native_multi_owner_fallback_probe/summary.tsv`.
- `GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_OWNER_STEP_FALLBACK` is now default-on in
  `src/gromacs/mdlib/sim_util.cpp`; `0` restores the older owner-native path.
- Honest result on the audited `2000`-step runtime with no special env:
  `gate_h` and `gate_i` both close total-force, per-level-force, energy, and
  gro exactly, with `gro`/`edr` hash equality preserved.
  `cpt` hashes still differ, so restart-bitwise identity is still not claimed.
- Performance stays positive after the closure fix:
  `gate_h` speedup `1.0966x`, `gate_i` speedup `1.1374x`.

- Split-owner sidecar bounded gain:
  `native_multi_split_owner_probe/local_9900x_gate_i_split_owner_runtime_report.json`,
  `native_multi_split_owner_probe/local_9900x_gate_i_full_owner_native_runtime_report.json`,
  `native_multi_split_owner_probe/local_9900x_gate_i_owner_fallback_runtime_refresh_report.json`,
  `native_multi_split_owner_probe/local_9900x_gate_i_split_owner_vs_owner_fallback_ntomp1_2_6_12_summary.json`,
  and
  `native_multi_split_owner_probe/local_9900x_gate_i_split_owner_vs_owner_fallback_ntomp1_2_6_12_summary.tsv`.
- These reports keep owner-step native-multi enabled for non-owner outputs, but
  move the owner contribution back to a sidecar per-contribution launch via
  `GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_SPLIT_OWNER_OUTPUTS=1` with
  `GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_OWNER_STEP_FALLBACK=0`.
- Honest result:
  the audited `gate_i` `2000`-step runtime closes total-force, per-level-force,
  energy, gro, and same-coordinate continuation exactly for the split-owner
  mode, while `gro`/`edr` hashes stay equal and `cpt` hashes still differ.
- The refreshed full owner-native report remains a failure artifact, not a
  success artifact:
  total-force delta `5.338623046875`, per-level-force delta
  `5.38525390625`, energy delta `1.25`, and gro max coordinate delta
  `0.001 nm`.
- Host-local speedup is real but bounded:
  split-owner speedup vs per-launch is `1.2140x`, `1.1968x`, `1.1767x`, and
  `1.0701x` at `ntomp=1/2/6/12`, respectively.
- Relative speedup is stronger than the current default owner-fallback at all
  audited `ntomp` points, but the evidence is still one host and one audited
  runtime shape. This is not yet a broad default-path claim.

- AVX2 kernel-family portability fix:
  `avx2_kernel_family_fix_probe/remote_z690_gate_i_forced_2xnn_split_owner_report.json`,
  `avx2_kernel_family_fix_probe/remote_z690_gate_i_default_split_owner_after_2xnn_patch_report.json`,
  and
  `avx2_kernel_family_fix_probe/remote_z690_gate_i_default_owner_fallback_after_2xnn_patch_report.json`.
- Root-cause isolation:
  on the remote AVX2 host, the same audited exact-r-RESPA runtime diverged when
  the default SIMD path selected `SIMD4xM 4x8`, even when the local audited TPR
  was copied over directly.
  Forcing `GMX_NBNXN_SIMD_2XNN=1` closed total-force, per-level-force, energy,
  gro, and same-coordinate continuation exactly on that host.
- Corrective patch:
  `src/gromacs/nbnxm/nbnxm_setup.cpp` now prefers `Cpu4xN_Simd_2xNN` whenever
  the input uses exact-r-RESPA pair splitting and both SIMD kernel families are
  available.
- Honest result after the patch on the remote AVX2 host:
  default exact-r-RESPA runs now select `SIMD2xMM 4x4` kernels and both
  `split_owner_sidecar` and `owner_fallback` close exactly through the audited
  `2000`-step runtime.
- This is still a bounded portability fix for the audited exact-r-RESPA CPU
  path. It is not evidence that generic non-exact-respa CPU kernel selection
  should change globally.

- Default safe native-multi boundary before Gate I replicas:
  `native_multi_default_safe_probe/local_9900x_gate_i_default_owner_middle_fallback_runtime_10000_report.json`,
  `native_multi_default_safe_probe/local_9900x_gate_i_default_owner_middle_fallback_fused_update_runtime_10000_report.json`,
  `native_multi_default_safe_probe/local_9900x_gate_i_forced_owner_native_runtime_fail_report.json`,
  `native_multi_default_safe_probe/local_9900x_gate_i_forced_owner_native_plainc_runtime_fail_report.json`,
  and
  `native_multi_default_safe_probe/local_9900x_gate_i_split_owner_middle_fallback_runtime_10000_fail_report.json`.
- `GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_MIDDLE_STEP_FALLBACK` is now default-on
  alongside the existing owner-step fallback. This means the replica-ready
  default path keeps owner-level and middle-level multi-output steps on the
  legacy per-contribution launch unless explicitly overridden.
- Honest 10000-step Gate I result:
  default owner+middle fallback closes total-force, per-level-force, energy,
  final `gro`, and same-coordinate continuation with `0.0` deltas.
- Fused initial-drift update is exact but not faster in this probe:
  `62.490 -> 61.933 ns/day`, so it remains default-off.
- Forced owner-native remains a failure artifact despite apparent speedup:
  total-force delta `4.97509765625`, per-level-force delta `5.13720703125`,
  energy delta `1.1599999999998545`, and final `gro` coordinate delta
  `0.001 nm`.
- The older split-owner sidecar result is not enough for long Gate I use.
  The 10000-step retest fails with total-force delta `9181.2333984375`,
  per-level-force delta `9455.427734375`, energy delta
  `3927.1699999999996`, and final `gro` coordinate delta `4.831 nm`.
- Therefore the only replica-ready mode from this evidence set is the default
  owner+middle fallback safe path. This is exactness closure, not a
  native-multi performance-win claim.
