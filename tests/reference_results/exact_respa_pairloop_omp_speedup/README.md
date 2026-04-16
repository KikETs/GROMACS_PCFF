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

- Existing exact r-RESPA force-dump parity tests exercise the diagnostic serial
  fallback when trace/dump outputs are enabled.
- The fast paths themselves are supported here by final-state bounded parity,
  printed energy agreement, restart continuity, and exact-runtime performance rows.
- Direct force-component dump parity for the fast path remains a follow-up gate
  because enabling the current dump instrumentation intentionally disables the
  fast path.
