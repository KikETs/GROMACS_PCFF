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
