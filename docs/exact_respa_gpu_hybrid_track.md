# exact r-RESPA GPU Hybrid Track

## Supersession Note, 2026-05-11

This file began as the early GPU-hybrid admission track. Parts of the original
scope are now historical.

Current PolyGen evidence has widened the local strict production path to:

- single-rank exact r-RESPA
- `-nb gpu -pme cpu -bonded gpu -update cpu`
- strict PME order 5
- 50 production chunks / 10 ns completed on the audited host

See [Exact r-RESPA Kernel Optimization Handoff](exact_respa_kernel_optimization_handoff_20260510.md)
and [PolyGen CPU/GPU Transport Screening, 2026-05-10](polygen_cpu_gpu_transport_screening_20260510.md).

This does not imply broad GPU bonded readiness, PME-GPU readiness, update-GPU
readiness, multi-rank readiness, or charged transport readiness.

## Current Scope

This track starts from commit `53781aa0a5` on top of the CPU exact r-RESPA
Gate I closure work.  The CPU Gate I density/volume result does not imply any
GPU or transport readiness.

The first GPU target is the existing exact r-RESPA nonbonded offload path:

- Runtime selector: `simulationWork.useGpuNonbonded` inside `do_force()`.
- Dispatch site: `src/gromacs/mdlib/sim_util.cpp`, `useExactLammpsRespaGpuNonbonded`.
- Active implementation: `computeExactRespaNonbondedGpuNarrow()`.
- Kernel family: regular NBNXM GPU dispatch, launched once per active exact
  r-RESPA nonbonded contribution.
- Supported initial shape: single-rank exact r-RESPA, `-nb gpu`, CPU bonded
  work, CPU update, and CPU PME unless a later gate explicitly widens scope.

The standalone CUDA pair kernel in
`src/gromacs/mdlib/exactrespa_nonbonded_gpu.cpp` and
`src/gromacs/mdlib/exactrespa_nonbonded_gpu_internal.cu` is not the initial
claim target unless the runtime is explicitly switched to it and revalidated.

## Out Of Scope

- Transport or conductivity production claims.
- Broad GPU bonded PCFF/class2 readiness outside the current strict PolyGen lane.
- GPU update readiness for exact r-RESPA unless a Gate E-style chain passes.
- Multi-rank or domain-decomposed exact r-RESPA GPU execution.
- Broad GPU performance guidance from one host or one fixture.
- Claiming speedup from normal leap-frog GPU paths or non-exact benchmarks.

## Required Evidence Before Any GPU Claim

The minimum evidence must be collected in this order:

1. CUDA build provenance: `gmx --version`, GPU support, CUDA runtime, GPU
   inventory, precision, and git SHA.
2. CPU oracle linkage: a passed Gate A CPU exact r-RESPA oracle manifest.
3. Nonbonded GPU exactness: Gate B-style comparison for event order, per-level
   force ownership, energy terms, total force dumps, virial where present, and
   restart continuity.
4. Runtime-path proof: logs must show `-nb gpu` and exact r-RESPA, not normal
   non-MTS short-MD.
5. Noise-floor handling: repeated GPU runs must bound expected GPU nondeterminism.
6. Performance separation: report local GPU offload timing, force proxy timing,
   and whole-run wall clock separately.

## Initial Validation Commands

The current audited GPU admission guard allows only the force-only narrow
shape.  Use the force-only smoke gate first:

```bash
python tools/pcff_respa_parity/validate_exact_respa_gpu_hybrid_force_only.py \
  --gmx build_gateb_cuda/bin/gmx \
  --build-dir build_gateb_cuda \
  --skip-build \
  --out tests/reference_results/exact_respa_gpu_hybrid/force_only_narrow_final_20260422 \
  --ntmpi 1 \
  --ntomp 2 \
  --outer-steps 5
```

Then run the stricter Gate B energy/virial oracle:

```bash
python tools/pcff_respa_parity/validate_gate_b_nb_gpu.py \
  --gmx build_gateb_cuda/bin/gmx \
  --build-dir build_gateb_cuda \
  --skip-build \
  --out tests/reference_results/exact_respa_gpu_hybrid/gate_b_nb_gpu_energy_virial_final_20260422 \
  --ntmpi 1 \
  --ntomp 2 \
  --outer-steps 5 \
  --gpu-repeats 3
```

Both commands are correctness gates, not performance claims.

## Current Force-Only Evidence

Artifact:
`tests/reference_results/exact_respa_gpu_hybrid/force_only_narrow_final_20260422/force_only_gpu_hybrid_manifest.json`

Status: `PASS`.

This pass covers only the admitted force-only smoke shape:

- `-ntmpi 1 -ntomp 2`
- `-nb gpu -pme cpu -bonded cpu -update cpu`
- exact r-RESPA enabled
- small fixtures: `small_oligomer`, `small_salt_polymer_box`
- event trace parity against CPU
- per-atom total force parity within `1e-3`
- per-level aggregate merge-trace parity using the recorded per-system aggregate
  tolerance policy

## Current Gate B Evidence

Artifact:
`tests/reference_results/exact_respa_gpu_hybrid/gate_b_nb_gpu_energy_virial_final_20260422/gate_b_manifest.json`

Status: `PASS`.

This pass covers only the audited Gate B shape:

- `-ntmpi 1 -ntomp 2`
- `-nb gpu -pme cpu -bonded cpu -update cpu`
- exact r-RESPA nonbonded pair splitting enabled
- small fixtures: `small_oligomer`, `small_salt_polymer_box`
- event/order validation
- force comparisons assessed against the recorded GPU roundoff/noise policy
- offloaded short-range energy parity within display-resolution policy
- virial/pressure diagnostics recorded and bounded by Gate B assessment
- restart continuity and repeated GPU noise-floor checks as implemented by Gate B

This historical Gate B evidence does not support Gate I density/volume,
transport, production handoff, GPU update, PME GPU, broad GPU bonded
PCFF/class2, multi-rank, or broad performance claims. Later PolyGen-specific
strict GPU production evidence is documented separately in the 2026-05-10
handoff and screening notes.

## Host-Local Experiment Default

The audited validation default remains `-ntomp 2 -pin off`. That is an
exactness gate default, not a performance default.

For host-local charged-scaffold experiments on the audited CUDA host, the
runtime-selection evidence currently supports this separate experiment default:

- `-ntmpi 1 -ntomp 12`
- `-nb gpu -pme cpu -bonded cpu -update cpu`
- `-pin on -pinstride 1`

Evidence basis:

- Charged scaffold thread sweep:
  `output/exact_respa_gpu_hybrid_gate_h_dense_ntomp_sweep_pinstride1_20260422/summary.json`
- Host-local Gate-I-shaped calibration launcher:
  `tools/pcff_respa_parity/bench_exact_respa_gpu_hybrid_gate_i.py`
- Host-local Gate-I-shaped calibration result:
  `output/exact_respa_gpu_hybrid_gate_i_hostlocal_calibration_20260422/summary.tsv`

This default is host-bounded runtime guidance only. It does not widen the
audited exactness claim beyond the currently passed GPU manifests.

## Claim Boundary

No GPU hybrid claim is allowed until the relevant validation manifest or
PolyGen-specific screening report is `PASS`. A force-only smoke pass by itself
would support only this narrow claim:

> On the audited CUDA host and small fixtures, exact r-RESPA single-rank
> force-only nonbonded GPU offload preserves CPU event order and force dumps
> within the declared tolerance for the admitted narrow runtime shape.

It would not support density/volume Gate I GPU readiness, transport readiness,
GPU production handoff, energy/virial GPU exactness, or general GPU speedup.
The later strict PolyGen production run supports only the narrower
PolyGen-specific CPU/GPU screening boundary recorded in the 2026-05-10 docs.

## Next Gate

For the historical small-fixture track, after the Gate B pass, the next strict
gate was a short exact-rRESPA runtime performance probe on the same fixture
family:

- CPU-only exact r-RESPA baseline.
- `-nb gpu -pme cpu -bonded cpu -update cpu` hybrid.
- Same MDP, same rank/thread shape, same number of outer steps.
- Report wall time, Force/GPU-NB wallcycle rows, CPU-GPU copy behavior, and
  restart continuity.

The current active GPU issue is no longer "can any GPU hybrid path run"; it is
whether strict PolyGen GPU production can be made faster without changing the
physics. See [Current Active Issues](current_active_issues.md).
