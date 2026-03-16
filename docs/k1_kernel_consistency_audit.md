# Milestone K1 — Kernel Force-Energy/Virial Consistency Audit

## 1. Executive Summary
Milestone K1 performed a structured audit of the custom GROMACS-PCFF kernels. Using minimal 4-atom fixtures and finite-difference numerical gradients, we confirmed that all custom kernels (Bond, Angle, Dihedral, Improper, and Nonbonded 9-6) are **internally consistent** regarding force-energy mapping. The systematic heating observed in TP1.3 is therefore unlikely to be caused by simple force-derivative errors in isolated interactions.

## 2. Confirmation Evidence
| Interaction | Status | Max Rel Diff (Force vs Gradient) |
| :--- | :--- | :--- |
| Bond Class2 | **PASS** | $1.1 \cdot 10^{-5}$ |
| Angle Class2 | **PASS** | $2.8 \cdot 10^{-4}$ |
| Dihedral AAT | **PASS** | $2.4 \cdot 10^{-4}$ |
| Improper Class2 | **PASS** | $2.4 \cdot 10^{-4}$ |
| Nonbonded 9-6 | **PASS** | $2.4 \cdot 10^{-4}$ (with charges) |

## 3. Localization Findings
- **Listed Forces:** Analytical kernels in `bonded.cpp` match numerical gradients within acceptable single-precision noise.
- **Nonbonded Pairs:** Use tabulated repulsion. Interpolation error is $\sim 1\%$ for small force magnitudes, but decreases relatively for large forces.
- **Virial:** Baseline virial calculation in GROMACS matches the expected $\Xi = - \frac{1}{2} \sum \vec{r}_{ij} \otimes \vec{F}_{ij}$ convention.

## 4. Suspected Failure Modes (Ranked)
1. **Exclusion Mask Bugs:** The 9-6 potential forces the use of plain-C reference kernels (`kernel_ref_inner.h`). These kernels might have bugs in applying exclusion masks for dense systems with complex molecule topologies.
2. **Double Counting:** Interactions might be added twice by the topology bridge.
3. **PME/Real-space Split:** Inconsistency in how 9-6 real-space interacts with reciprocal space (though TP1.3 showed Cut-off alone is also unstable).

## 5. Next Step Recommendation
Proceed to **K2 — Neighbor List & Exclusion Mask Audit**. Since kernels are consistent in isolation, the audit must move to how these kernels are applied in the multi-atom context of the full `dense_salt_polymer` system.
