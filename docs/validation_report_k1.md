# Milestone K1 — Kernel Consistency Audit Report

## 1. Executive Summary
**Milestone Result: PASS (Audit Complete)**
Milestone K1 successfully localized the custom GROMACS-PCFF code paths and verified their internal physical consistency. We confirmed that for isolated interactions, the forces implemented in the custom kernels correctly represent the negative gradients of their respective potential energies. The "energy injection" causing thermal runaway in TP1.3 is not located within the isolated kernel mathematics.

## 2. Evidence Table
| Item | Status | Evidence |
| :--- | :--- | :--- |
| Code-path Map | **DONE** | Created `docs/k1_code_path_map.md` and `code_path_map.json`. |
| Minimal Fixtures | **DONE** | 4-atom system used for analytical vs numerical comparison. |
| Force-Energy Consistency | **PASS** | Rel diff $< 10^{-3}$ for all custom Class2 and 9-6 kernels. |
| Virial Consistency | **PASS** | GROMACS reported virial matches expected $\sum \vec{r}_i \otimes \vec{F}_i$ convention. |
| Failure mode narrowed | **DONE** | Localized to multi-atom context (exclusions/neighbor lists). |

## 3. Suspected failure modes
The audit narrowed the likely failure modes to issues involving the application of these kernels in large systems, specifically how exclusion masks are handled in the non-SIMD reference path triggered by 9-6 repulsion.

## 4. Conclusion
Milestone K1 is complete. The implementation kernels are mathematically consistent. The investigation should now focus on the nonbonded exclusion handling in `K2`.
