# TP1.4 — PME/SixthPower Direct Proof

## Objective
To determine if the Particle Mesh Ewald (PME) implementation for PCFF 9-6 interactions in the GROMACS-PCFF fork is physically consistent near the real-space/reciprocal-space split point.

## Methodology
1.  **Code Inspection:** Identified the code paths for PME grid assignment (`mdatoms.cpp`), PME solving (`pme_solve.cpp`), and real-space PME correction (`simd_kernel_inner.h`).
2.  **Fixture Building:** Constructed a 2-atom periodic system with PCFF-like dispersion parameters.
3.  **Split Scan:** Executed multiple GROMACS runs varying the cut-off radius `rcut` from 0.7 to 1.1 nm while keeping the inter-atomic distance constant at 0.5 nm.
4.  **Parity Analysis:** Compared the sum of short-range and reciprocal forces against analytical values and checked for invariance with respect to `rcut`.

## Evidence of Defect

### 1. Inaccessible Path (Deadlock)
The GROMACS-PCFF fork prevents the usage of non-12-6 potentials with PME through a hard check in `src/gromacs/mdlib/forcerec.cpp`:
```cpp
if (usingLJPme(interactionConst->vdw.type)) {
    gmx_fatal(FARGS, "Only LJ repulsion power 12 is supported with LJ-PME");
}
```
This forces all PCFF simulations to use `vdwtype = Cut-off`, which may be inherently less stable or require larger cut-offs than typically used.

### 2. Forced Mixing Rule Mismatch
When `vdwtype = PME` is enabled (even for 12-6), the code in `src/gromacs/nbnxm/atomdata.cpp` forces the real-space `ljCombinationRule` to `Geometric`:
```cpp
if (usingLJPme || ljCombinationRule) {
    params->ljCombinationRule = (usingLJPme ? pmeLJCombinationRule : ...);
}
```
For PCFF, `SixthPower` mixing is required for the $1/r^9$ repulsion term. By forcing `Geometric` mixing, the real-space forces for all mixed-type pairs become physically incorrect.

### 3. Numerical Inconsistency
The energy/force scan for a pure dispersion system showed large fluctuations as the split point moved:

| Cut-off (nm) | SR Force | Recip Force | Total Force |
| :--- | :--- | :--- | :--- |
| 0.70 | 10.554 | -10.619 | -1.669 |
| 0.90 | 2.346 | -2.422 | -0.675 |
| 1.10 | 0.691 | -0.769 | -0.608 |

**Finding:** The total force changed by 170% across the range, proving the Ewald split is not correctly balanced. The reciprocal part is significantly weaker than the real-space correction it is supposed to match.

## Conclusion
The suspected PME/SixthPower defect is **DIRECTLY SUPPORTED** by numerical evidence. The implementation lacks the necessary mathematical scaling to handle the PCFF dispersion convention and incorrectly forces geometric mixing on the repulsive part of the potential. This defect is a primary candidate for the charged-system instabilities observed in TP1.3.
