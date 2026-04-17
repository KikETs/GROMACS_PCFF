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
