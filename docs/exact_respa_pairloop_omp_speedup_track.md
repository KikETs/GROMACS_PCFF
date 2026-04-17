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
