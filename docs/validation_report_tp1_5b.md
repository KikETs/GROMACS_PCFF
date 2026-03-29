# Validation Report — TP1.5b Dense Cut-off Reproducer and Pairlist-vs-Kernel Separation Audit

## 1. Executive Summary

**Milestone TP1.5b Result: PASS**

TP1.5b produced a real dense periodic cut-off-only reproducer on the current dirty build and narrowed the live defect family further than TP1.5. A 4-atom periodic 9-6 cut-off fixture with one near-cutoff attractive cross-pair reproduced a worsening direction when pairlist lifetime was loosened, while fixed-frame reruns remained invariant.

Bounded interpretation:
- TP1.4 remains `PARTIAL`
- TP1.5 as a broader milestone remains `PARTIAL`
- TP1.5b itself passes because it achieved the requested fault-isolation milestone without overclaiming a single root cause

## 2. What Was Reproduced

Primary dense fixture:
- `dense_nonlisted`
- 4 charged atoms in a `2.5 x 2.5 x 2.5 nm` periodic box
- `rep-pow = 9`, `coulombtype = Cut-off`, `vdwtype = Cut-off`
- near-cutoff attractive cross-pair `A2-A3 = 0.905 nm`

Reference and sweep results from `tests/reference_results/tp1_5b_dense_cutoff_audit/pairlist_sweep_results.csv`:

| Run | Pairlist setup | Total-energy range (kJ/mol) | Final drift (kJ/mol) | Interpretation |
| :-- | :-- | --: | --: | :-- |
| `tight_ref_n1_r1200` | `nstlist=1`, `rlist=1.200` | `8.657745` | `+0.541504` | tight reference |
| `n1_r0909` | `nstlist=1`, `rlist=0.909` | `8.657745` | `+0.541504` | matches tight reference exactly |
| `n10_r0909` | `nstlist=10`, `rlist=0.909` | `12.576325` | `-4.373841` | worsened |
| `n20_r0909` | `nstlist=20`, `rlist=0.909` | `12.576325` | `-4.375183` | worsened, same as `n10` |
| `n10_r0900` | `nstlist=10`, `rlist=0.900` | `13.749512` | `-4.762085` | worst explicit no-buffer case |
| `auto_buffer_n10_vbt0005` | auto buffer, `rlist≈0.911` | `8.657745` | `+0.539795` | matches tight reference |

This is enough to call the worsening direction reproduced on a minimal dense cut-off-only fixture:
- loosening pairlist lifetime from `nstlist=1` to `nstlist=10` at the same `rlist=0.909` widens the total-energy range by `1.45x`
- reverting to auto buffer removes that worsening on the same fixture

## 3. Pairlist vs Kernel Separation

### Confirmed pairlist sensitivity

The strongest new evidence is the `nstlist` separation:
- `n1_r0909` and `tight_ref_n1_r1200` match exactly
- `n10_r0909` and `n20_r0909` both worsen materially

That weakens a story based on `rlist=0.909` alone. The sensitive axis on this fixture is pairlist lifetime / no-buffer operation on the dense cut-off path.

### Fixed-frame kernel miscompute weakened

Static reruns at `nsteps=0` on the same dense nonlisted frame are invariant:
- `rlist=0.900` vs `0.909`: potential diff `0.0 kJ/mol`
- max force-component diff `0.0`

See `tests/reference_results/tp1_5b_dense_cutoff_audit/listed_vs_nonlisted_checks.csv` plus the raw dumps:
- `raw_dense_nonlisted__static_r0900_force_dump.txt`
- `raw_dense_nonlisted__static_r0909_force_dump.txt`

This does **not** prove the cut-off kernel is clean in all dynamic contexts. It does show the reproduced worsening is not explained by a simple fixed-frame off-cutoff inclusion error.

## 4. Listed-vs-Nonlisted Separation

Executed sister fixture:
- `dense_routed_sister`
- same dense geometry
- introduces `nrexcl = 1`, two zero-force bonds, and one explicit pair

Observed results:
- fixed-frame `r0900` vs `r0909` is invariant in both potential and force
- routed `tight_ref` vs routed `n10_r0909` shows `energy_range_ratio = 1.0`

Interpretation:
- listed/nonlisted routing remains relevant as a topology family
- but it is **not** the strongest explanation for the pairlist-sensitive worsening reproduced on the primary dense fixture

## 5. Runtime Family Localization

The executed fixtures confirm the same runtime family localized in TP1.5:
- `Detected LJ repulsion power 9.`
- `Using plain-C-4x4 4x4 nonbonded short-range kernels`

Executed runtime evidence is preserved in:
- `tests/reference_results/tp1_5b_dense_cutoff_audit/runtime_path_trace.json`
- `tests/reference_results/tp1_5b_dense_cutoff_audit/raw_dense_nonlisted__n10_r0909_md.log`
- `tests/reference_results/tp1_5b_dense_cutoff_audit/raw_dense_routed_sister__n10_r0909_md.log`

The narrow path remains:
- `src/gromacs/mdlib/forcerec.cpp::init_forcerec`
- `src/gromacs/nbnxm/nbnxm_setup.cpp::chooseLJCombinationRule`
- `src/gromacs/nbnxm/nbnxm_setup.cpp::init_nb_verlet`
- `src/gromacs/nbnxm/pairlist_tuning.cpp::setupDynamicPairlistPruning`
- `src/gromacs/nbnxm/kerneldispatch.cpp::getCoulombKernelType`
- `src/gromacs/nbnxm/kerneldispatch.cpp::getVdwKernelType`
- `src/gromacs/nbnxm/kerneldispatch.cpp::nbnxn_kernel_cpu`
- `src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h`

## 6. What TP1.5b Did Not Prove

TP1.5b did **not** prove:
- one exact faulty line in the plain-C kernel
- that TP1.3 is fully explained
- that listed-vs-nonlisted routing is irrelevant in all dense charged topologies
- that TP1.4 is resolved

Those claims would overreach this evidence.

## 7. Exact Next Step Recommendation

Use TP1.5b as the gate into TP1.6:
1. keep the `dense_nonlisted` fixture fixed
2. instrument the pairlist lifetime path only
3. compare per-step pair inclusion and force accumulation between `n1_r0909` and `n10_r0909`
4. do **not** patch broad kernel math before a per-step inclusion mismatch or lifetime/routing error is directly observed

## 8. Required Milestone Summary

- files changed
  - `docs/validation_report_tp1_5b.md`
  - `docs/tp1_5b_dense_cutoff_reproducer.md`
  - `tools/run_tp1_5b_dense_cutoff_audit/README.md`
  - `tools/run_tp1_5b_dense_cutoff_audit/run_dense_cutoff_audit.py`
  - `tests/reference_results/tp1_5b_dense_cutoff_audit/*`
- commands run
  - captured exactly in `tests/reference_results/tp1_5b_dense_cutoff_audit/raw_commands.txt`
- fixtures executed
  - `dense_nonlisted`
  - `dense_routed_sister`
- strongest confirmed finding
  - dense cut-off worsening on the minimal primary fixture is pairlist-sensitive: `n10_r0909` widens total-energy range by `1.45x` over tight reference while `n1_r0909` matches tight reference exactly
- strongest unresolved uncertainty
  - TP1.5b does not yet identify the exact per-step inclusion or accumulation error inside the plain-C cut-off runtime family
- exact next step recommendation
  - instrument pairlist lifetime behavior on the TP1.5b `dense_nonlisted` fixture before patching
- verdict
  - `PASS`
