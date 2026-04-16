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

Vectorization is subordinate to correctness and profiling. It may be introduced only after the parallel path has a passing exactness suite.

Allowed vectorization forms:

- compiler-vectorizable math-only helper extraction
- explicit SIMD for the fixed repulsion-power-9 scalar math shape
- narrower branch-specialized kernels when profiling shows branch dispatch dominates

Forbidden:

- approximation of PME table interpolation
- relaxed cutoffs or changed switching weights
- hidden work skipping
- vector-only microbenchmark claims without real exact-r-RESPA runtime evidence

## P5/P6 Exactness and Integration Gates

Minimum exactness checks after each stage:

- same `.tpr`, same command shape, serial mode versus candidate mode
- short deterministic fixture first
- medium charged PCFF scaffold second
- compare final coordinates, energies, box, and pressure/virial-relevant outputs where available
- restart continuity check for accepted stages
- exact r-RESPA runtime event/order tests remain passing

Trace/debug observable dumps remain serial until explicitly made parallel-safe.
The existing force-dump parity tests are therefore diagnostic-contract tests, not
proof that the new fast path itself emitted force-component dumps. Fast-path
force-component parity requires separate instrumentation that does not change
the fast-path eligibility window.

## P7/P8 Claim Decision

At the end of each stage, the only allowed outcomes are:

- real host-local exact-r-RESPA OpenMP speedup
- pair-loop-local or Force-proxy gain without meaningful whole-run gain
- no worthwhile gain

Any public claim must name the host, fixture, runtime shape, thread counts, and whether the gain is whole-run, Force-proxy, or pair-loop-local.
The first accepted wording must also state that vectorization is not implemented
until a real `pairloop_vector` mode exists and passes the same exactness gates.

## P10 Optimization Priority

Before deeper SIMD/OpenMP surgery:

1. Add isolated nonbonded-only timing or PMU-style profiling where practical.
2. Decompose pair-loop, force-buffer reduction, accumulation, and update overhead.
3. Only then pursue tighter dispatch, branch reduction, fixed-shape kernel variants, or more aggressive vectorization.
