# Exact r-RESPA Pair-Loop OpenMP Speedup Track

## P1 Scope Freeze

This track targets only the exact r-RESPA CPU pair-splitting runtime path:

- Function: `computeExactRespaNonbondedCpu()` in `src/gromacs/mdlib/sim_util.cpp`.
- Inner loop: the local `processPairlist()` loop over `plainPairlist.pairs` and `plainPairlist.excludedPairs`.
- Work type: short-range nonbonded-adjacent exact pair work, including PME real-space excluded-pair correction bookkeeping.
- Not targeted: the listed-force pair14 kernels. Pair14 placement can feed exact r-RESPA scheduling, but the optimized loop here is the plain pairlist/excluded-pair CPU consumer.
- Consumed exact levels: inner, optional middle, and outer exact r-RESPA nonbonded force outputs selected by `exactRespaNonbondedInnerLevel()`, `exactRespaNonbondedMiddleLevel()`, and `exactRespaNonbondedOuterLevel()`.
- Initial audited runtime shape: CPU-only, `-ntmpi 1`, `-nb cpu -pme cpu -bonded cpu -update cpu`, exact r-RESPA pair splitting enabled, repulsion-power-9 specialized path enabled unless explicitly disabled for a control run.

Out of scope for this track unless separately opened:

- GPU or hybrid execution implications.
- LJ-PME SIMD or generic PME SIMD claims.
- Broad chemistry expansion beyond the already-audited PCFF exact r-RESPA fixtures.
- Generic CPU-wide speedup claims outside `computeExactRespaNonbondedCpu()`.
- Conductivity, production, or transport-readiness wording.

## P2 Measurement Protocol

All performance rows must record:

- fixture identity and `.tpr` path
- git commit and dirty state
- binary path and `gmx --version`
- mode: `baseline`, `pairloop_omp`, `pairloop_vector`, or `combined`
- `ntmpi`, `ntomp`, pinning mode, and relevant environment toggles
- run length, repeat count, and command line
- throughput from `Performance`
- wallcycle `Force`, `Neighbor search`, `PME mesh`, `Update`, and `Total` rows when present

Required staged comparisons:

1. Baseline exact r-RESPA CPU path.
2. Same path with pair-loop OpenMP enabled.
3. Same path with pair-loop vectorization enabled.
4. Combined path if both changes coexist.

The benchmark path must not switch to normal short-MD, `exact-respa = no`, benchmark-only NVE shortcuts, or a non-pair-splitting runtime while being compared against exact r-RESPA pair-loop data.

## P3 Parallelization Design

The first parallel implementation must be default-off and selected by an explicit environment toggle. It must not alter the serial path used by existing claims.

Safe first eligibility window:

- no trace-only diagnostic replay
- no debug/trace/probe dumps
- no pair-energy accumulation on that call
- no direct virial accumulation on that call
- no excluded-correction dump side effect
- at least two OpenMP threads
- sufficiently large pair list

Force-buffer strategy:

- each OpenMP thread writes to thread-private force buffers for each active exact nonbonded contribution
- each OpenMP thread writes to thread-private shift-force buffers
- a deterministic post-loop reduction accumulates thread buffers into the existing `ForceOutputs` buffers in fixed thread order
- energy and virial calls remain serial until their reductions are explicitly implemented and separately validated

This preserves physics but may change floating-point summation order relative to the serial pair traversal. Therefore bitwise equivalence to `ntomp=1` is not required for the parallel fast path. Required validation is bounded numerical equivalence under predeclared tolerances, plus deterministic reproducibility for the same thread count and affinity shape.

## P4 Vectorization Design

Vectorization is subordinate to correctness and profiling. The first vector path is an explicit
default-off batch path selected by `GMX_PCFF_EXACT_RESPA_PAIRLOOP_VECTOR=1`.

Current implemented vectorization form:

- fixed-width lane batching inside the same no-trace/no-energy/no-virial pair-loop fast path
- `#pragma omp simd` over the per-lane distance, split-weight, repulsion-power-9, and PME real-space force scalar math
- deterministic scatter after the vectorized math stage into the same thread-private force/shift buffers used by the OpenMP path

Still allowed as follow-up work:

- narrower branch-specialized kernels when profiling shows branch dispatch dominates
- more explicit SIMD for the fixed repulsion-power-9 scalar math shape if the compiler-vectorized batch path is insufficient

Forbidden:

- approximation of PME table interpolation
- relaxed cutoffs or changed switching weights
- hidden work skipping
- vector-only microbenchmark claims without real exact-r-RESPA runtime evidence

Observed first-stage result:

- vector-only produces a real exact-runtime improvement on the audited host/fixture, but it is smaller than the OpenMP pair-loop gain
- combined OpenMP+vector has negligible additional whole-run benefit over OpenMP alone in the first Gate I sweep
- therefore vectorization is a limited capability, not a broad SIMD-performance claim

## P5/P6 Exactness and Integration Gates

Minimum exactness checks after each stage:

- same `.tpr`, same command shape, serial mode versus candidate mode
- short deterministic fixture first
- medium charged PCFF scaffold second
- compare final coordinates, energies, box, and pressure/virial-relevant outputs where available
- restart continuity check for accepted stages
- exact r-RESPA runtime event/order tests remain passing

Trace/debug observable dumps remain serial until explicitly made parallel-safe.
The legacy force-dump parity tests are diagnostic-contract tests because those
trace paths intentionally disable the new fast path. Fast-path force-component
parity is covered by the separate pair-loop force-delta harness:

- tool: `tools/pcff_respa_parity/validate_exact_respa_pairloop_force_delta.py`
- instrumentation env:
  `GMX_PCFF_EXACT_RESPA_PAIRLOOP_FORCE_DUMP_DIR`,
  `GMX_PCFF_EXACT_RESPA_PAIRLOOP_FORCE_DUMP_LABEL`, and
  `GMX_PCFF_EXACT_RESPA_PAIRLOOP_FORCE_DUMP_MAX`
- measured object: per-active-level force-buffer delta across
  `plainPairlist.pairs` and `plainPairlist.excludedPairs`
- eligibility: the dump env is not included in `computePairEnergies` and does
  not disable the no-energy/no-virial fast path
- acceptance: bounded single-precision parity, not bitwise pair-loop parity;
  a component passes if `abs_delta <= 1e-2` or `rel_delta <= 5e-5`

Current Gate I 20-step `ntomp=6` force-delta evidence:

- report: `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_pairloop_force_delta_report.json`
- second-host report: `tests/reference_results/exact_respa_pairloop_omp_speedup/remote_z690_gate_i_pairloop_force_delta_report.json`
- compared fast-path snapshots: 28 per candidate mode
- compared components: 285120 per candidate mode
- modes: `pairloop_omp`, `pairloop_vector`, `combined`
- local 9900X max absolute pair-loop force-delta difference: `0.00604248046875`
- remote Z690/i9-12900K max absolute pair-loop force-delta difference: `0.0048828125`
- max relative difference occurs only on near-zero components with absolute
  differences below `4e-5`
- final `.gro` output is byte-identical across baseline and all three
  candidate modes in each host-local run

This closes the previous "fast-path force-component dump" evidence gap only for
the audited Gate I fixture and the no-energy/no-virial fast-path window on the
two measured hosts. It does not make virial/energy fast paths validated, because
those calls still deliberately fall back to the serial pair loop.

## P7/P8 Claim Decision

At the end of each stage, the only allowed outcomes are:

- real host-local exact-r-RESPA OpenMP speedup
- pair-loop-local or Force-proxy gain without meaningful whole-run gain
- no worthwhile gain

Any public claim must name the host, fixture, runtime shape, thread counts, and whether the gain is whole-run, Force-proxy, or pair-loop-local.
The first accepted wording must state that vectorization is implemented only as
the audited batch path, and that the measured combined benefit over OpenMP alone
is marginal on the first Gate I host-local fixture.

Second-host staged performance evidence:

- report: `tests/reference_results/exact_respa_pairloop_omp_speedup/remote_z690_gate_i_pairloop_omp_report.json`
- host: `user-Z690-AORUS-PRO`, Intel i9-12900K
- fixture/runtime: same Gate I exact-r-RESPA CPU TPR, `-ntmpi 1`, `-pin off`,
  `-reprod`, 2000 steps, one repeat
- best remote baseline: `3.538 ns/day` at `ntomp=6`
- best remote `pairloop_omp`: `5.893 ns/day` at `ntomp=6`, `1.666x` versus
  same-thread baseline
- best remote `pairloop_vector`: `4.409 ns/day` at `ntomp=6`, `1.246x` versus
  same-thread baseline
- best remote `combined`: `5.857 ns/day` at `ntomp=6`, `1.655x` versus
  same-thread baseline

The second-host result supports a bounded, audited-host statement that the
OpenMP pair-loop fast path improves this exact-r-RESPA runtime shape on both
measured hosts. It still does not support broad CPU guidance, production
transport readiness, or a claim that vectorization adds material benefit on top
of the OpenMP path.

## P10 Optimization Priority

Before deeper SIMD/OpenMP surgery:

1. Add isolated nonbonded-only timing or PMU-style profiling where practical.
2. Decompose pair-loop, force-buffer reduction, accumulation, and update overhead.
3. Only then pursue tighter dispatch, branch reduction, fixed-shape kernel variants, or more aggressive vectorization.

## P11 Sparse Reduction And Update OMP Probe

Two additional default-off knobs were added after the first two-host OpenMP
evidence:

- `GMX_PCFF_EXACT_RESPA_PAIRLOOP_SPARSE_REDUCTION=1` enables touched-atom
  tracking for pair-loop reduction only when a pairlist is sparse enough to
  justify the bookkeeping. Dense pairlists fall back to the existing full
  reduction to avoid the measured slowdown from tracking nearly all atoms.
- `GMX_PCFF_EXACT_RESPA_PAIRLOOP_TIMING_DIR=<dir>` writes per-call
  clear/pair/reduce timing rows for the exact r-RESPA pair-loop fast path.
- `GMX_PCFF_EXACT_RESPA_UPDATE_OMP=1` parallelizes the exact r-RESPA
  velocity-half-kick and position-drift atom loops in
  `src/gromacs/mdrun/exactrespastepper.cpp`.

Current local Gate I evidence:

- force-delta report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_update_sparse_force_delta_report.json`
- `ntomp=6` performance report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_update_sparse_ntomp6_report.json`
- `pairloop_omp_update` and `pairloop_sparse_update` both pass the same
  bounded force-delta criterion as the earlier fast-path modes.
- final `.gro` output is byte-identical across baseline,
  `pairloop_omp_update`, and `pairloop_sparse_update` in the 20-step harness.
- `ntomp=6`, 2000-step local performance:
  baseline `3.820 ns/day`, `pairloop_omp` `6.568 ns/day`,
  `pairloop_omp_update` `6.614 ns/day`, and `pairloop_sparse_update`
  `6.589 ns/day`.

Claim boundary:

- Update OMP is a safe experimental knob for this audited exact-r-RESPA VV path,
  but the current Gate I evidence shows only marginal whole-run improvement over
  `pairloop_omp`.
- Sparse reduction is not a useful Gate I density-run optimization because the
  pairlist is dense; it is retained only as a guarded path for genuinely sparse
  pairlists.
- Neither knob supports stronger production, transport, density/volume, or
  broad CPU scaling claims.

## P12 Specialized Pair Kernels, Block Reduction, And Tile Probe

The next exact-r-RESPA CPU performance pass split the fast path by audited
runtime shape instead of continuing to route both lists through the same
generic lambda:

- `plainPairlist.pairs` now uses the standard exact-r-RESPA short-range
  kernel with fixed physical semantics: LJ + bare Coulomb + excluded-pair
  PME real-space correction split across the active exact levels.
- `plainPairlist.excludedPairs` now uses a correction-only kernel that skips
  the standard-pair branch shape and writes only the outer excluded-pair
  correction semantics.

Two additional default-off knobs were then audited on top of that split:

- `GMX_PCFF_EXACT_RESPA_PAIRLOOP_BLOCK_REDUCTION=1` replaces the dense
  atom-by-atom reduction with a block-owned reduction sweep.
- `GMX_PCFF_EXACT_RESPA_PAIRLOOP_TILE=1` adds an experimental tile-local
  force scatter path, but only for `plainPairlist.pairs`; the correction-only
  `excludedPairs` path stays on the specialized non-tiled kernel because
  tiling there measured as overhead rather than useful work reduction.

Current local Gate I evidence:

- force-delta report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_block_tile_force_delta_report.json`
- `ntomp=6` performance report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_block_tile_ntomp6_report.json`
- `pairloop_block` and `pairloop_tile` both pass the same bounded force-delta
  criterion as the earlier fast-path modes.
- final `run.gro` SHA256 is identical across baseline, `pairloop_block`, and
  `pairloop_tile` in the 20-step harness:
  `a00186d56c84f8105820521f8f0b7937630fc34226a0254f3691393c6672a313`
- `ntomp=6`, 2000-step local performance:
  baseline `4.086 ns/day`, `pairloop_omp` `6.938 ns/day`,
  `pairloop_block` `6.977 ns/day`, and `pairloop_tile` `6.836 ns/day`

Claim boundary:

- The specialized `pairs` versus `excludedPairs` split is now part of the
  audited exact-r-RESPA fast path implementation, not a separate public
  performance claim.
- Block reduction supports only a narrow audited-host statement:
  it is a small whole-run improvement over the earlier `pairloop_omp` path on
  the local Gate I `ntomp=6` fixture.
- The current tile backend is exactness-clean but does not outperform the
  block-only or `pairloop_omp` path on this dense Gate I fixture, so it
  remains experimental and does not expand the public claim boundary.

## P13 Block-Indexed Tile Retry And NBNXM-Style 4x4 Prototype

The tile path was retried with a block-indexed cache and per-thread cache reuse.
The goal was to remove the obvious atom-linear lookup weakness before deciding
whether the tile idea should be retired.

Current local Gate I evidence for the block-indexed retry:

- force-delta report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_blockindexed_tile_force_delta_report.json`
- `ntomp=6` performance report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_blockindexed_tile_ntomp6_report.json`
- `ntomp=6`, 2000-step local performance:
  baseline `4.012 ns/day`, `pairloop_omp` `6.927 ns/day`,
  `pairloop_block` `6.942 ns/day`, and `pairloop_tile` `6.698 ns/day`

Verdict for the block-indexed retry:

- The block-indexed cache does not change the claim boundary.
- Tile still loses to the simpler `pairloop_block` path, so the tile line is
  not worth pushing further on this audited fixture.

After that retry, a bounded nbnxm-style `4x4` prototype was tested for the
standard `pairs` path.

Prototype scope:

- env gate: `GMX_PCFF_EXACT_RESPA_PAIRLOOP_NBNXM4X4=1`
- runtime scope: `plainPairlist.pairs` only
- `excludedPairs` stay on the existing specialized exact correction path
- clusterization is rebuilt on every eligible call to avoid stale pairlist
  content across neighbor-list updates

Current local Gate I evidence for the bounded prototype:

- force-delta report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_nbnxm4x4_force_delta_report.json`
- `ntomp=6` performance report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_nbnxm4x4_ntomp6_report.json`
- short-harness maximum absolute pair-loop force-delta difference:
  `0.00531005859375`
- `ntomp=6`, 2000-step local performance:
  baseline `4.058 ns/day`, `pairloop_omp` `7.012 ns/day`,
  `pairloop_block` `7.003 ns/day`, and `pairloop_nbnxm4x4` `4.284 ns/day`

Claim boundary:

- This is not a successful nbnxm migration result.
- The bounded `4x4` prototype is mechanically acceptable in the short harness,
  but it is not a worthwhile performance direction in its current form.
- The evidence supports keeping `pairloop_block` as the best audited host-local
  CPU path so far.

## P14 Direct CPU-List Consumer Backend

Scope:

- env gate: `GMX_PCFF_EXACT_RESPA_PAIRLOOP_DIRECT_CPULIST=1`
- runtime scope: exact CPU `PairlistSet::cpuLists()` consumption inside
  `computeExactRespaNonbondedCpu()`
- this is not full nbnxm kernel reuse
- the backend still uses the exact CPU scalar force semantics; it only skips
  materializing `plainPairlist` when the fast-path contract is active

Design boundary:

- pair admission follows the same source used by `appendPlainPairlistCpu()`:
  `getCoordinate(nbat, ...)` plus `nbat.shift_vec`
- admitted-pair force geometry still uses the exact runtime source:
  `coordinates[...]` plus `fr->shift_vec`
- this split is required; the first direct attempt used the wrong admission
  source and failed parity at neighbor-list update boundaries

Exactness evidence:

- `ntomp=6` force-delta report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_direct_cpulist_force_delta_ntomp6_report.json`
- `ntomp=12` force-delta report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_direct_cpulist_force_delta_ntomp12_report.json`
- both reports pass the bounded short-harness parity check
- `ntomp=6` maximum absolute force-delta: `0.00604248046875`
- `ntomp=12` maximum absolute force-delta: `0.0006103515625`
- final `run.gro` SHA256 matches baseline for both validated thread counts

Performance evidence on the audited 9900X Gate I fixture:

- `ntomp=6` staged report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_direct_cpulist_ntomp6_report.json`
- `ntomp=2,12` staged report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_direct_cpulist_ntomp2_12_report.json`
- `ntomp=12` direct-vs-block split:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_direct_cpulist_ntomp12_direct_vs_block_report.json`
- baseline anchor for `ntomp=1`:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_direct_cpulist_ntomp1_baseline_report.json`

Current local throughput summary:

- `ntomp=1`: baseline `3.763 ns/day`
- `ntomp=2`: baseline `3.947 ns/day`, `pairloop_block` `5.800 ns/day`,
  `pairloop_direct_block` `5.770 ns/day`
- `ntomp=6`: baseline `4.103 ns/day`, `pairloop_omp` `6.971 ns/day`,
  `pairloop_block` `6.988 ns/day`, `pairloop_direct` `9.750 ns/day`,
  `pairloop_direct_block` `9.774 ns/day`
- `ntomp=12`: baseline `4.055 ns/day`, `pairloop_block` `6.951 ns/day`,
  `pairloop_direct` `11.221 ns/day`, `pairloop_direct_block` `11.289 ns/day`

Claim boundary:

- the evidence supports a real host-local exact-r-RESPA CPU gain on this
  audited host and fixture for `ntomp=6` and `ntomp=12`
- the evidence does not support a blanket statement that the direct CPU-list
  backend wins at every OpenMP point; at `ntomp=2` it is effectively tied with,
  or slightly below, `pairloop_block`
- `ntomp=1` is out of scope for this fast path because the current OpenMP gate
  only enables it when more than one OpenMP thread is active
- this is still not a broad CPU scaling claim, a GPU claim, or a successful
  full nbnxm kernel migration claim

## P15 Direct CPU-List Flag and Full-Mask Refinement

Scope:

- runtime scope stays the same as P14:
  `GMX_PCFF_EXACT_RESPA_PAIRLOOP_DIRECT_CPULIST=1` inside
  `computeExactRespaNonbondedCpu()`
- three bounded changes were added on top of the P14 backend:
  `ci.shift` flag consumption, full-mask suffix fast-path use, and direct
  `cpuLists()` partition consumption without flattening each `ci` into a
  separate work item
- this is still not full nbnxm kernel reuse; the exact CPU runtime still owns
  the force math and exact-r-RESPA accumulation semantics

Design delta:

- standard-pair scalar evaluation now consumes `NBNXN_CI_DO_LJ(0)`,
  `NBNXN_CI_HALF_LJ(0)`, and `NBNXN_CI_DO_COUL(0)` from `ci.shift`
- excluded-pair correction work now skips `ci` entries that advertise no
  Coulomb work
- sorted `cj` entries now use the existing `NBNXN_INTERACTION_MASK_ALL` suffix
  contract to avoid per-pair mask checks in the full-mask suffix
- the backend now iterates `PairlistSet::cpuLists()` directly and processes
  each list in-place instead of flattening every `ci` entry into a separate job

Exactness evidence:

- `ntomp=2` force-delta report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_direct_flags_fullmask_partition_force_delta_ntomp2_report.json`
- `ntomp=6` force-delta report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_direct_flags_fullmask_partition_force_delta_ntomp6_report.json`
- `ntomp=12` force-delta report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_direct_flags_fullmask_partition_force_delta_ntomp12_report.json`
- all three reports pass the bounded short-harness parity check
- maximum absolute force-delta:
  `ntomp=2` `0.00054931640625`,
  `ntomp=6` `0.00604248046875`,
  `ntomp=12` `0.00189971923828125`

Performance evidence on the audited 9900X Gate I fixture:

- staged report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_direct_flags_fullmask_partition_ntomp2_6_12_report.json`

Current local throughput summary:

- `ntomp=2`: baseline `3.941 ns/day`, `pairloop_block` `5.744 ns/day`,
  `pairloop_direct` `7.138 ns/day`, `pairloop_direct_block` `7.137 ns/day`
- `ntomp=6`: baseline `4.063 ns/day`, `pairloop_block` `6.901 ns/day`,
  `pairloop_direct` `11.331 ns/day`, `pairloop_direct_block` `11.510 ns/day`
- `ntomp=12`: baseline `3.995 ns/day`, `pairloop_block` `6.872 ns/day`,
  `pairloop_direct` `12.593 ns/day`, `pairloop_direct_block` `12.423 ns/day`

Claim boundary:

- the refinement survives exactness checks at `ntomp=2`, `6`, and `12`
- the direct CPU-list backend now has audited host-local wins over
  `pairloop_block` at all three measured OpenMP points on this host and fixture
- block reduction is no longer the universal best companion mode; after the
  refinement it is marginal and `ntomp`-dependent
- this remains a host-bounded exact-r-RESPA CPU claim, not a broad CPU scaling
  claim and not a full nbnxm migration claim

## P16 Packed Cluster Path and Bounded Layout Dispatch

Scope:

- runtime scope stays inside the exact CPU direct CPU-list backend in
  `computeExactRespaNonbondedCpu()`
- three follow-on implementation goals were attempted together:
  a full-mask standard-pairs cluster microkernel, local packed cluster-force
  accumulation to reduce scatter pressure, and bounded layout dispatch keyed by
  the native CPU list shape
- this is still not `kerneldispatch.cpp` reuse and still not a full nbnxm
  kernel migration

Design delta:

- standard-pairs direct CPU-list work now accumulates per-`iEntry` packed
  cluster forces and only flushes to thread scratch at cluster boundaries
- excluded-pair correction work now uses the same packed cluster flush pattern
- the bounded full-mask standard-pairs path dispatches on native CPU list
  layout, currently `na_ci == 4` and `na_cj in {4, 8}`, before taking the
  compiler-vectorizable cluster microkernel path
- the original `#pragma omp simd` array-reduction attempt hit a GCC 13 internal
  compiler error; the landed path keeps the cluster-microkernel structure but
  uses per-lane temporary buffers instead of the crashing reduction form

Exactness evidence:

- `ntomp=2` force-delta report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_direct_packed_dispatch_force_delta_ntomp2_report.json`
- `ntomp=6` force-delta report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_direct_packed_dispatch_force_delta_ntomp6_report.json`
- `ntomp=12` force-delta report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_direct_packed_dispatch_force_delta_ntomp12_report.json`
- all three reports pass the bounded short-harness parity check
- maximum absolute force-delta:
  `ntomp=2` `0.00189971923828125`,
  `ntomp=6` `0.00189971923828125`,
  `ntomp=12` `0.00189971923828125`

Performance evidence on the audited 9900X Gate I fixture:

- staged report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/local_9900x_gate_i_direct_packed_dispatch_ntomp2_6_12_report.json`

Current local throughput summary:

- `ntomp=2`: baseline `3.933 ns/day`, `pairloop_block` `5.825 ns/day`,
  `pairloop_direct` `7.189 ns/day`, `pairloop_direct_block` `7.226 ns/day`
- `ntomp=6`: baseline `4.101 ns/day`, `pairloop_block` `6.979 ns/day`,
  `pairloop_direct` `11.623 ns/day`, `pairloop_direct_block` `11.570 ns/day`
- `ntomp=12`: baseline `4.038 ns/day`, `pairloop_block` `6.916 ns/day`,
  `pairloop_direct` `12.484 ns/day`, `pairloop_direct_block` `12.575 ns/day`

Claim boundary:

- the packed-cluster and bounded-dispatch follow-on survives exactness at
  `ntomp=2`, `6`, and `12`
- on this audited host and fixture it improves the best measured direct
  CPU-list path again, but the best companion mode still depends on `ntomp`
- the bounded layout dispatch is evidence for a useful exact-runtime
  nbnxm-adjacent improvement, not for broad `kerneldispatch` reuse and not for
  a completed full nbnxm migration claim

## P17 Update Default-Auto Integration

Scope:

- runtime scope stays inside exact r-RESPA CPU update work in
  `src/gromacs/mdrun/exactrespastepper.cpp`
- this pass does not change force semantics, pair-loop semantics, PME
  semantics, or the narrow exact-r-RESPA claim boundary
- the audited question is whether exact-r-RESPA update work should remain
  default-off or move to a product-style default-auto thread policy

Design delta:

- `GMX_PCFF_EXACT_RESPA_UPDATE_OMP` now has three meanings:
  unset = `Auto`, `0` = `Off`, nonzero = `On`
- `Auto` delegates to
  `gmx_omp_nthreads_get_simple_rvec_task(ModuleMultiThread::Update, numAtoms)`
  instead of silently falling back to serial update loops
- exact-r-RESPA half-kick application now fuses multiple kick levels into one
  atom loop when the audited base-step schedule applies more than one kick
- benchmark scripts now freeze update semantics explicitly:
  benchmark `baseline` means `Off`, and `Auto` is a separate recorded mode

Exactness evidence:

- machine-readable update-mode report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/update_mode_probe/local_9900x_gate_i_update_modes_ntomp6_report.json`
- paired TSV summary:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/update_mode_probe/local_9900x_gate_i_update_modes_ntomp6_report.tsv`
- audited runtime shape: local 9900X, Gate I `equil.tpr`, `ntomp=6`, `-reprod`,
  2000 steps
- `run.gro` SHA256 is identical across `baseline`, `update_auto`, and
  `update_omp`:
  `3a7fa2e6fc7ff73b44894d02325f427e1233a31196902f05a96095bcf7a09a71`
- `run.edr` SHA256 is identical across the same three modes:
  `6fb84be422368691d77f544f54938f46d19edd3fb67a77463b740337d995c4d7`
- `run.cpt` hashes differ, so restart-bitwise identity is not claimed from this
  probe

Performance evidence:

- `baseline` (`Off`): `25.516 ns/day`, `Update 1.655 s`, `Total 3.388 s`
- `update_auto` (`Auto`): `26.752 ns/day`, `Update 1.565 s`, `Total 3.231 s`
- `update_omp` (`On`): `26.795 ns/day`, `Update 1.563 s`, `Total 3.226 s`

Claim boundary:

- this supports a narrow statement: exact-r-RESPA CPU update work no longer
  needs to remain default-serial on this audited host and runtime shape
- `Auto` is now a defensible product default because it preserves audited
  `gro/edr` outputs and lands within the same performance band as forced `On`
- this is still only a host-bounded update-path result; it does not justify a
  broad CPU scaling claim, a restart-bitwise claim, or any density/transport
  readiness claim
- pair-loop force-delta tooling remains the wrong validator for update-only
  modes, because update-only runs do not emit fast-path pair-loop snapshots

## P18 Complete-Pairlist Contract Separation and Lazy Plain-Pairlist Materialization

Scope:

- runtime scope stays inside exact r-RESPA pair-search / pair-loop contract
  wiring in `src/gromacs/mdrun/runner.cpp`, `src/gromacs/mdtypes/forcerec.h`,
  `src/gromacs/mdlib/sim_util.cpp`, and `src/gromacs/mdlib/exactrespa_nonbonded_gpu.cpp`
- this pass does not change force math, contribution ownership, or the narrow
  exact-r-RESPA claim boundary
- the audited questions are:
  1. whether the exact pair-admission contract can be separated from
     `plainPairlistRange`
  2. whether eager plain-pairlist materialization is still necessary when there
     are no `MDModulesPairlistConstructedSignal` subscribers
  3. whether the default-auto pair-loop path shows a stable short-run win over
     a forced-off baseline on this audited shape

Design delta:

- `t_forcerec` now separates:
  `plainPairlistRange` = MD-module signal/materialization range,
  `completePairlistRange` = pair-search range that requires `includeAllPairs`
- pair-search now keys `constructPairlist(... includeAllPairs ...)` off
  `completePairlistRange`, not `plainPairlistRange`
- exact CPU and exact GPU r-RESPA paths now consume `completePairlistRange`
  for their internal complete-pairlist contract
- benchmark and force-delta scripts now freeze staged meanings explicitly:
  baseline and non-direct modes set both env vars to `0` unless they are
  intentionally testing the direct path
- pair-search no longer eagerly materializes `plainPairlist` just because
  `plainPairlistRange` is present; eager materialization only happens when a
  real `MDModulesPairlistConstructedSignal` subscriber exists
- `GMX_PCFF_EXACT_RESPA_EAGER_PLAIN_PAIRLIST=1` still forces legacy eager
  materialization for exact-only runs by materializing at
  `completePairlistRange` when no MD-module signal range exists

Exactness evidence:

- machine-readable report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/complete_pairlist_contract_probe/local_9900x_gate_i_complete_pairlist_contract_report.json`
- paired TSV summary:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/complete_pairlist_contract_probe/local_9900x_gate_i_complete_pairlist_contract_report.tsv`
- audited runtime shape: local 9900X, Gate I `equil.tpr`, `ntomp=6`, `-reprod`,
  `2000 steps x 2 repeats`
- all audited modes and repeats produce the same `run.gro` SHA256:
  `ac3dc3ecc4728f0123e7a686d7468baa5e37d6c515af14f76ef67fb16aec0599`
- all audited modes and repeats also produce the same `run.edr` SHA256:
  `ca8088ffe00cadcd7d97fedd6e2d4920901bc193fa6ad7d90d573b75a9325f68`
- `run.cpt` hashes differ across compared modes, so restart-bitwise identity is
  not claimed from this pass

Performance evidence:

- repeat-averaged `default_auto`: `85.5245 ns/day`,
  `Neighbor search 0.153 s`, `Force 0.5875 s`, `Update 0.4355 s`
- repeat-averaged explicit `On`: `84.7455 ns/day`,
  `Neighbor search 0.1535 s`, `Force 0.5875 s`, `Update 0.4400 s`
- repeat-averaged forced-off baseline: `85.3810 ns/day`,
  `Neighbor search 0.1525 s`, `Force 0.5885 s`, `Update 0.4350 s`
- repeat-averaged forced eager plain-pairlist materialization: `29.8505 ns/day`,
  `Neighbor search 0.1655 s`, `Force 0.5775 s`, `Update 1.3795 s`

Claim boundary:

- this pass closes a real semantic ambiguity:
  exact pair admission no longer depends on the presence of an MD-module
  plain-pairlist signal range
- `default_auto` and explicit `On` are mechanically equivalent at the
  `gro/edr` level and land in the same short-run performance band on this
  audited shape
- the evidence does not support claiming a stable whole-run speed gain of
  `default_auto` over a forced-off baseline from this short probe;
  the repeat-averaged ratio is effectively `1.00x`
- the large `2.87x` difference only shows that reintroducing unnecessary eager
  plain-pairlist materialization is catastrophically expensive on this runtime
  shape while leaving `gro/edr` unchanged
- this is still not a broad CPU scaling claim, a restart-bitwise claim, or a
  full leap-frog / full-nbnxm migration claim

## P19 Native Multi Owner-Step Energy/Virial Migration

Scope:

- runtime scope stays inside exact narrow CPU NBNXM owner steps in
  `src/gromacs/mdlib/sim_util.cpp`, `src/gromacs/nbnxm/kerneldispatch.cpp`,
  `src/gromacs/nbnxm/nbnxm.h`, and `src/gromacs/nbnxm/tests/pcff_class2_nonbonded.cpp`
- this pass extends native multi from force-only steps to owner steps that also
  request energy and direct virial
- the change is still bounded to the audited single-rank CPU narrow runtime;
  it is not a broad CPU completion claim

Design delta:

- `dispatchExactRespaCpuNativeMultiKernel()` now accepts the real
  `StepWorkload` and audited real-space energy sinks instead of forcing a
  synthetic force-only workload
- CPU NBNXM native multi clearing now keys off native-multi force routing
  itself, not the old `force-only` guard, so owner steps no longer leave
  contribution-indexed force/shift buffers stale
- owner-step energy continues to use the existing kernel energy arrays and
  reductions, but now under the native-multi launch path
- owner-step direct virial now reconstructs from the owning native
  contribution's `fshift` buffers after contribution-indexed force reduction,
  instead of incorrectly assuming base output buffers still own the virial
  source
- new PCFF unit coverage validates native owner-step energy ownership against
  the existing per-contribution launch model

Exactness evidence:

- native contract/unit coverage still passes:
  `./build/bin/nbnxm-output-contract-test --gtest_filter='*Native*:*ExactRespa*'`
- owner-step PCFF coverage now passes:
  `./build/bin/nbnxm-test --gtest_filter='*ExactRespaNativeMultiForceOnlyMatchesPerContributionLaunch:*ExactRespaNativeMultiOwnerEnergyMatchesPerContributionLaunch:*ExactRespaNativeMultiForceOnlyKeepsExcludedPmeCorrectionOuterOnly'`
- new machine-readable runtime reports:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_owner_probe/local_9900x_gate_h_owner_native_multi_runtime_report.json`
  and
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_owner_probe/local_9900x_gate_i_owner_native_multi_runtime_report.json`
- short 2000-step runtime hashes still diverge for `gro/edr/cpt`, so
  restart-bitwise or whole-trajectory identity is not claimed
- same-coordinate continuation probes remain the honest semantic check:
  - `gate_h` `2000 -> 2004`: `gro` identical, `Potential/Total Energy` delta `0`,
    virial/pressure terms remain within printed-dump noise, total/per-level
    force max abs delta `7.32421875e-04`
  - `gate_i` `2000 -> 2004`: `gro` identical, `Potential/Total Energy` delta `0`,
    virial/pressure terms remain within printed-dump noise, total/per-level
    force max abs delta `7.32421875e-04`
- full 2000-step force/virial drift still appears after trajectories separate,
  so this pass does not close whole-run hash parity

Performance evidence:

- `gate_h` owner-step runtime:
  `52.591 -> 58.553 ns/day`, `speedup 1.113x`,
  `Force 0.920 -> 0.730 s`, `Update 0.749 -> 0.683 s`
- `gate_i` owner-step runtime:
  `52.638 -> 64.535 ns/day`, `speedup 1.226x`,
  `Force 0.905 -> 0.667 s`, `Update 0.753 -> 0.611 s`

Claim boundary:

- this pass supports a narrow runtime statement:
  exact narrow CPU native multi now covers owner steps with energy ownership
  and direct-virial reconstruction
- same-state continuation evidence supports bounded observable parity, not
  bitwise identity and not whole-trajectory identity
- the measured speedup is host-local and audited only on these two fixtures
- this still does not justify broad CPU scaling language, PP-DD claims, or any
  density/transport readiness claim

## P20 Native-Multi Divergence Onset And Pairloop Force-Dump Decoupling

Scope:

- this pass does not widen the physics/runtime claim
- it does two bounded things:
  diagnose when owner-step native-multi trajectories begin to separate on real
  runtime fixtures, and remove one remaining `PlainPairlist` coupling from the
  pair-loop force-dump path

Design delta:

- new onset-scan harness:
  `tools/pcff_respa_parity/scan_exact_respa_native_multi_divergence_onset.py`
- the harness reuses the real runtime comparator helpers from
  `validate_exact_respa_native_multi_runtime.py`, but reruns from step `0` at
  increasing step counts to show when divergence leaves the tiny-noise regime
- `src/gromacs/mdlib/sim_util.cpp` no longer forces `needPlainPairlist=true`
  merely because `GMX_PCFF_EXACT_RESPA_PAIRLOOP_FORCE_DUMP_DIR` is set
- force-dump headers now report whether plain-pair counts are actually
  available instead of silently printing counts from an empty placeholder list

Evidence:

- onset reports:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_divergence_onset_probe/local_9900x_gate_h_native_multi_divergence_onset_report.json`
  and
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_divergence_onset_probe/local_9900x_gate_i_native_multi_divergence_onset_report.json`
- `gate_h` onset curve:
  step `4/16/64` force delta stays `5.4931640625e-04`; step `256` rises to
  `3.2586669921875e-01`; step `1024` to `1.7891845703125`; step `2000` to
  `13.73583984375`
- `gate_i` onset curve:
  step `4/16/64/256` force delta stays `5.4931640625e-04`; step `1024` rises
  to `2.182708740234375`; step `2000` to `6.3876953125`
- both fixtures show:
  `edr` hash mismatch from `4` steps, `gro` hash mismatch from `64` steps, but
  large force-growth appears only later
- this is stronger evidence for reduction-order-driven trajectory branching than
  for an immediate owner-step semantic failure
- validation status for the pair-loop force-dump decoupling is narrower:
  `gmx` rebuild and normal audited runtime probes pass, but the old
  `validate_exact_respa_pairloop_force_delta.py` harness did not emit snapshots
  on the runtime shapes tried here, so that old dump harness is not currently a
  reliable validator for this coupling change

Claim boundary:

- the honest reading is now:
  owner-step native multi has bounded same-state parity, and the later whole-run
  divergence behaves like gradual trajectory branching rather than an immediate
  semantic break
- this still does not close whole-run hash parity or bitwise identity
- the pair-loop force-dump decoupling is a dataflow cleanup, not a new speedup
  claim

## P21 Serial-Reduction Probe And Scalar Force-Dump Harness Restore

Scope:

- this pass does two bounded things:
  - test whether the later native-multi divergence is caused by the final
    NBNXM output-buffer reduction stage
  - restore the old scalar pair-loop force-delta harness so it explicitly runs
    on the scalar exact CPU path instead of silently falling onto narrow NBNXM

Design delta:

- `src/gromacs/nbnxm/atomdata.cpp` now honors
  `GMX_PCFF_EXACT_RESPA_NBNXM_SERIAL_REDUCTION=1` as a diagnostic-only switch
  that forces the multi-buffer force reduction to use one worker over the block
  range
- `src/gromacs/mdlib/sim_util.cpp` now honors
  `GMX_PCFF_EXACT_RESPA_DISABLE_NBNXM_NARROW=1`, which forces the scalar exact
  CPU path for pair-loop diagnostics and leaves a log message at step `0`
- `tools/pcff_respa_parity/validate_exact_respa_pairloop_force_delta.py` and
  `tools/pcff_respa_parity/bench_exact_respa_pairloop_omp.py` now export
  `GMX_PCFF_EXACT_RESPA_DISABLE_NBNXM_NARROW=1` so the old scalar pair-loop
  track cannot silently measure the wrong runtime path

Evidence:

- serial-reduction onset reports:
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_serial_reduction_probe/local_9900x_gate_h_native_multi_serial_reduction_report.json`
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_serial_reduction_probe/local_9900x_gate_i_native_multi_serial_reduction_report.json`
- gate_h and gate_i onset curves are numerically unchanged versus the default
  scans:
  - `gate_h`: `4/64/256/1024/2000` force deltas remain
    `5.4931640625e-04 / 5.4931640625e-04 / 3.2586669921875e-01 / 1.7891845703125 / 13.73583984375`
  - `gate_i`: `4/64/256/1024/2000` force deltas remain
    `5.4931640625e-04 / 5.4931640625e-04 / 5.4931640625e-04 / 2.182708740234375 / 6.3876953125`
- this means the final output-buffer reduction stage is not the driver of the
  later divergence
- restored scalar force-delta report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/scalar_pairloop_force_delta_probe/local_9900x_gate_h_pairloop_direct_force_delta_ntomp6_report.json`
- that scalar harness now passes again on the intended runtime shape
  (`passed=true`)
- on a fast-path snapshot (`ordinal 2`, `step 1`) the candidate dump reports:
  `pair_fast_path_used=true`, `excluded_pair_fast_path_used=true`,
  `compute_pair_energies=false`, `compute_virial=false`,
  `plain_pair_count_available=false`
- this is the missing snapshot-level proof that pair-loop force-dump no longer
  forces `PlainPairlist` materialization when the direct fast path is active

Claim boundary:

- the serial-reduction probe narrows the likely source of whole-run divergence:
  the final NBNXM force-buffer reduction is not the culprit
- the remaining suspicion shifts upstream to native-multi kernel-side
  accumulation / arithmetic ordering
- scalar pair-loop diagnostics are now honest again because the scripts pin
  themselves to the scalar runtime they claim to audit

## P22 Ntomp1 And Plain-C Divergence Probes

Scope:

- this pass does not claim a fix
- it tests two narrower falsification questions for the remaining
  native-multi whole-run divergence:
  - does the divergence disappear when the runtime is reduced to `ntomp=1`
  - does the divergence disappear when SIMD kernels are disabled and the
    plain-C reference kernels are forced

Design delta:

- no runtime physics or kernel math was changed in this pass
- `tools/pcff_respa_parity/scan_exact_respa_native_multi_divergence_onset.py`
  now records whether `GMX_DISABLE_SIMD_KERNELS` was set
- `tools/pcff_respa_parity/validate_exact_respa_native_multi_runtime.py`
  now records the `GMX_DISABLE_SIMD_KERNELS` environment state and checks the
  step-0 log for a plain-C marker

Evidence:

- canonical `ntomp=1` onset reports:
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_ntomp1_divergence_probe/local_9900x_gate_h_native_multi_ntomp1_divergence_report.json`
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_ntomp1_divergence_probe/local_9900x_gate_i_native_multi_ntomp1_divergence_report.json`
- canonical `ntomp=1 + GMX_DISABLE_SIMD_KERNELS=1` plain-C reports:
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_plainc_divergence_probe/local_9900x_gate_h_native_multi_plainc_ntomp1_divergence_report.json`
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_plainc_divergence_probe/local_9900x_gate_i_native_multi_plainc_ntomp1_divergence_report.json`
- `ntomp=1` does not close the divergence:
  - `gate_h` step-`2000` force delta improves from `13.73583984375` at
    `ntomp=6` to `3.6497650146484375` at `ntomp=1`, but remains large
  - `gate_i` step-`2000` force delta becomes much worse:
    `6.3876953125 -> 765.7132415771484`
- plain-C reference kernels also do not close the divergence:
  - `gate_h` step-`2000` force delta becomes `14.1541748046875`
  - `gate_i` step-`2000` force delta becomes `1278.444320678711`
  - both plain-C reports record `disable_simd_kernels_env=1` and
    `disable_simd_kernels_marker_seen=true`
- plain-C speed is catastrophically lower on these audited shapes:
  candidate throughput stays around `1.2 ns/day`, so this probe is diagnostic
  evidence only and not a usable runtime path

Claim boundary:

- `ntomp=1` alone is not a real closure for the native-multi divergence
- forcing plain-C reference kernels is also not a closure; it can worsen the
  later divergence while destroying performance
- therefore the honest remaining suspect is not simply OpenMP thread fan-out
  and not specifically the SIMD kernels
- the strongest remaining source hypothesis is native-multi contribution
  interleaving / kernel-side arithmetic grouping itself

## P23 Owner-Step Fallback Closure

Scope:

- this pass fixes the audited native-multi divergence instead of probing around
  it
- the target is owner-level exact-r-RESPA dispatch inside
  `computeExactRespaNonbondedCpuNbnxmNarrow()`

Design delta:

- `src/gromacs/mdrun/md.cpp` now honors
  `GMX_EXACT_RESPA_FORCE_DUMP_INTERVAL`, so dense force-dump scans can observe
  real per-step onset instead of aliasing to `nstenergy`
- `src/gromacs/mdlib/sim_util.cpp` now detects owner-level steps from
  `highestActiveLevel == exactRespaNonbondedOuterLevel(inputrec)` instead of
  the weaker `computeEnergy || computeVirial` heuristic
- `GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_OWNER_STEP_FALLBACK` is default-on;
  setting it to `0` restores the older divergent owner-level native-multi path

Evidence:

- dense interval-1 validation isolates the culprit:
  - owner-only fallback is exact through `32`, `256`, and `1024` steps for
    both `gate_h` and `gate_i`
  - middle-only fallback fails immediately at step `0`
- canonical dense reports:
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_owner_fallback_probe/local_9900x_gate_h_owner_fallback_dense_1024_report.json`
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_owner_fallback_probe/local_9900x_gate_i_owner_fallback_dense_1024_report.json`
- canonical default-owner-fallback runtime reports:
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_owner_fallback_probe/local_9900x_gate_h_default_owner_fallback_runtime_report.json`
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_owner_fallback_probe/local_9900x_gate_i_default_owner_fallback_runtime_report.json`
  - `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_owner_fallback_probe/summary.tsv`
- on the audited `2000`-step runtime with no special env override:
  - `gate_h`: total-force/per-level-force/energy/gro all exact, `gro` and
    `edr` hashes equal, speedup `1.0966x`
  - `gate_i`: total-force/per-level-force/energy/gro all exact, `gro` and
    `edr` hashes equal, speedup `1.1374x`
- `cpt` hashes still differ, so restart-bitwise identity is not claimed

Claim boundary:

- for the audited local host and audited `gate_h/gate_i` runtime shapes, the
  native-multi divergence closes when owner-level steps stay on the legacy
  per-contribution launch
- this is still not a broad claim about every host, every chemistry, or
  restart-bitwise identity

## P24 Split-Owner Sidecar Runtime Closure

Scope:

- this pass audits a narrower owner-step recovery path instead of reopening the
  divergent full owner-native path
- the target remains
  `computeExactRespaNonbondedCpuNbnxmNarrow()` in
  `src/gromacs/mdlib/sim_util.cpp`
- only the owner contribution is peeled back into a sidecar
  per-contribution launch; non-owner contributions stay on the native-multi
  output contract

Design delta:

- no new runtime semantics were introduced in the kernel path for this pass;
  the main code change is measurement hardening in
  `tools/pcff_respa_parity/validate_exact_respa_native_multi_runtime.py`
- the runtime harness now accepts explicit mode names plus bounded
  `--baseline-env` / `--candidate-env` overrides, so
  `per_launch`, `owner_fallback`, `full_owner_native`, and
  `split_owner_sidecar` can be compared on the same audited runtime path

Evidence:

- canonical split-owner runtime exactness report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_split_owner_probe/local_9900x_gate_i_split_owner_runtime_report.json`
- canonical refreshed full owner-native failure report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_split_owner_probe/local_9900x_gate_i_full_owner_native_runtime_report.json`
- canonical sweep summary:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_split_owner_probe/local_9900x_gate_i_split_owner_vs_owner_fallback_ntomp1_2_6_12_summary.json`
  and
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_split_owner_probe/local_9900x_gate_i_split_owner_vs_owner_fallback_ntomp1_2_6_12_summary.tsv`
- audited `gate_i` findings:
  - split-owner closes total-force, per-level-force, energy, gro, and
    same-coordinate continuation exactly on the audited `2000`-step runtime
  - `gro` and `edr` hashes stay equal; `cpt` hashes still differ
  - full owner-native still fails with total-force delta
    `5.338623046875`, per-level-force delta `5.38525390625`, energy delta
    `1.25`, and gro max coordinate delta `0.001 nm`
  - split-owner speedup vs per-launch is `1.2140x`, `1.1968x`, `1.1767x`,
    and `1.0701x` at `ntomp=1`, `2`, `6`, and `12`
  - the current default owner-fallback remains exact, but its relative
    speedup is smaller at all audited `ntomp` points:
    `1.0946x`, `1.0962x`, `1.1095x`, and `1.0436x`

Claim boundary:

- on the audited local host and audited `gate_i` runtime shape, split-owner
  sidecar is the strongest currently validated owner-step recovery mode:
  it preserves audited runtime exactness while improving host-local throughput
  more than the default owner-fallback
- this is still not a broad default-path claim for every host, every fixture,
  or restart-bitwise identity
- the old full owner-native path remains outside the honest claim boundary

## P25 AVX2 Kernel-Family Portability Fix

Scope:

- this pass investigates why the audited remote AVX2 host still failed exact
  runtime closure after source synchronization and rebuild
- the target remains the exact-r-RESPA CPU narrow/NBNXM path, not generic
  leap-frog or non-exact-respa kernel selection

Root-cause isolation:

- stale source alone does not explain the failure:
  key exact-r-RESPA and NBNXM source files were synchronized byte-for-byte to
  the remote host before rebuilding
- TPR mismatch alone does not explain the failure:
  copying the local audited `gate_i` `equil.tpr` to the remote host still
  produced divergence on the default remote path
- the decisive difference is kernel family:
  - local audited host selected `SIMD2xMM 4x4`
  - remote AVX2 host selected `SIMD4xM 4x8`
  - remote divergence began at step `0` with tiny force/pressure differences and
    then amplified over time
- plain-C and forced-2xNN diagnostics narrow the culprit further:
  - remote plain-C runs reduced the force mismatch from order `1e2-1e3` down to
    order `1e-4`, but at unusable performance
  - forcing `GMX_NBNXN_SIMD_2XNN=1` on the remote AVX2 host restored exact
    runtime closure for the audited exact-r-RESPA path

Design delta:

- `src/gromacs/nbnxm/nbnxm_setup.cpp` now prefers
  `NbnxmKernelType::Cpu4xN_Simd_2xNN` whenever the input uses exact-r-RESPA and
  pair splitting is active, provided both SIMD kernel families are available
- the generic non-exact-respa heuristic is left unchanged

Evidence:

- forced remote 2xNN closure before the code change:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/avx2_kernel_family_fix_probe/remote_z690_gate_i_forced_2xnn_split_owner_report.json`
- canonical remote default split-owner report after the patch:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/avx2_kernel_family_fix_probe/remote_z690_gate_i_default_split_owner_after_2xnn_patch_report.json`
- canonical remote default owner-fallback report after the patch:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/avx2_kernel_family_fix_probe/remote_z690_gate_i_default_owner_fallback_after_2xnn_patch_report.json`
- post-patch remote logs show the corrected default kernel family:
  `Using SIMD2xMM 4x4 nonbonded short-range kernels`
- audited remote results after the patch:
  - default `split_owner_sidecar` closes total-force, per-level-force, energy,
    gro, and same-coordinate continuation exactly, with speedup `1.0145x`
  - default `owner_fallback` closes total-force, per-level-force, energy, gro,
    and same-coordinate continuation exactly, with speedup `1.0238x`
  - `gro` and `edr` hashes are equal in both cases; `cpt` still differs

Claim boundary:

- the supported claim is now stronger but still narrow:
  the audited exact-r-RESPA CPU narrow path closes on both the local AVX-512
  host and the audited remote AVX2 host after the exact-r-RESPA-scoped
  `2xNN` kernel-family override
- this is not a claim that the AVX2 `SIMD4xM 4x8` exact-r-RESPA path is fixed
  or validated; it is a bounded portability correction that routes audited
  exact-r-RESPA work onto the validated `2xNN` family
- this is also not a broad claim about generic non-exact-respa CPU kernel
  selection on AVX2 hosts

## P26 Default Safe Native-Multi Boundary Before Gate I Replicas

Scope:

- this pass audits which native-multi recovery mode is safe enough to use
  before starting longer Gate I density/volume replica experiments
- the target remains the exact-r-RESPA CPU narrow runtime in
  `src/gromacs/mdlib/sim_util.cpp`
- this pass does not reopen conductivity, transport, GPU, or broad CPU-scaling
  claims

Design delta:

- `GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_MIDDLE_STEP_FALLBACK` is now default-on
  for force-only middle-level steps
- owner-step fallback remains default-on through
  `GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_OWNER_STEP_FALLBACK`
- the default path therefore keeps owner-level and middle-level multi-output
  steps on the legacy per-contribution launch unless explicitly overridden
- `tools/pcff_respa_parity/validate_exact_respa_native_multi_runtime.py` now
  accepts `--probe-steps 0` so negative controls can skip unavailable
  same-coordinate probes explicitly
- `GMX_PCFF_EXACT_RESPA_FUSED_INITIAL_DRIFT=1` remains a separate default-off
  update experiment

Evidence:

- default owner+middle fallback 10000-step Gate I report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_default_safe_probe/local_9900x_gate_i_default_owner_middle_fallback_runtime_10000_report.json`
- fused initial-drift update report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_default_safe_probe/local_9900x_gate_i_default_owner_middle_fallback_fused_update_runtime_10000_report.json`
- forced owner-native failure reports:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_default_safe_probe/local_9900x_gate_i_forced_owner_native_runtime_fail_report.json`
  and
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_default_safe_probe/local_9900x_gate_i_forced_owner_native_plainc_runtime_fail_report.json`
- split-owner sidecar 10000-step failure report:
  `tests/reference_results/exact_respa_pairloop_omp_speedup/native_multi_default_safe_probe/local_9900x_gate_i_split_owner_middle_fallback_runtime_10000_fail_report.json`
- unit/build coverage:
  `cmake --build build -j8 --target gmx nbnxm-test` and
  `./build/bin/nbnxm-test --gtest_filter='PcffClass2NonbondedCurveTest.*ExactRespa*'`

Observed results:

- default owner+middle fallback closes total force, per-level force, energy,
  virial/pressure tracked terms, final GRO, and same-coordinate continuation
  with `0.0` deltas on the audited 10000-step Gate I shape
- default owner+middle fallback performance is noise-level versus per-launch:
  `61.237 -> 61.526 ns/day`
- fused initial-drift update is exact but slower on this probe:
  `62.490 -> 61.933 ns/day`
- forced owner-native still fails despite apparent speedup:
  total force delta `4.97509765625`, per-level force delta `5.13720703125`,
  energy delta `1.1599999999998545`, and GRO coordinate delta `0.001 nm`
- forced owner-native plain-C also fails at first-frame scale:
  total force delta `0.00030517578125`, per-level force delta
  `0.000244140625`, and energy delta `0.019999999999527063`
- split-owner sidecar is not safe for longer Gate I runs despite the older
  2000-step closure:
  total force delta `9181.2333984375`, per-level force delta
  `9455.427734375`, energy delta `3927.1699999999996`, and final GRO
  coordinate delta `4.831 nm`

Claim boundary:

- the only Gate I replica-ready mode from this pass is the default owner+middle
  fallback safe path
- this is exactness closure, not a native-multi performance win
- split-owner sidecar and forced owner-native cannot be used for Gate I
  density/volume replicas based on the current evidence
- fused initial-drift update should stay default-off because it does not
  improve this audited runtime
