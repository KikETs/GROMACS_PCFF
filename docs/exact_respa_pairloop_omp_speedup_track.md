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
- compared fast-path snapshots: 28 per candidate mode
- compared components: 285120 per candidate mode
- modes: `pairloop_omp`, `pairloop_vector`, `combined`
- max absolute pair-loop force-delta difference: `0.00604248046875`
- max relative difference occurs only on near-zero components with absolute
  differences below `4e-5`
- final `.gro` output is byte-identical across baseline and all three
  candidate modes

This closes the previous "fast-path force-component dump" evidence gap only for
the audited Gate I host-local fixture and the no-energy/no-virial fast-path
window. It does not make virial/energy fast paths validated, because those calls
still deliberately fall back to the serial pair loop.

## P7/P8 Claim Decision

At the end of each stage, the only allowed outcomes are:

- real host-local exact-r-RESPA OpenMP speedup
- pair-loop-local or Force-proxy gain without meaningful whole-run gain
- no worthwhile gain

Any public claim must name the host, fixture, runtime shape, thread counts, and whether the gain is whole-run, Force-proxy, or pair-loop-local.
The first accepted wording must state that vectorization is implemented only as
the audited batch path, and that the measured combined benefit over OpenMP alone
is marginal on the first Gate I host-local fixture.

## P10 Optimization Priority

Before deeper SIMD/OpenMP surgery:

1. Add isolated nonbonded-only timing or PMU-style profiling where practical.
2. Decompose pair-loop, force-buffer reduction, accumulation, and update overhead.
3. Only then pursue tighter dispatch, branch reduction, fixed-shape kernel variants, or more aggressive vectorization.
