# TP1.5 — Cut-off Only Regression / Exclusion-Reference Path Audit

## Objective
To isolate the specific code path and mechanism causing the cut-off-only simulations to regress more severely than PME-based simulations in the `dense_salt_polymer` system.

## Methodology
1.  **Log Analysis:** Identified that 9-6 repulsion forces a fallback to plain-C reference kernels (`forcerec.cpp`).
2.  **Kernel Inspection:** Analyzed `kernel_ref_inner.h` for exclusion and masking logic.
3.  **Fixture Testing:** Created a 2-atom excluded pair fixture (`repro_excl.py`) to measure energy and forces in the fallback path.
4.  **Sensitivity Scan:** Varied distance and repulsion power to confirm the presence of distance-invariant energy offsets for excluded pairs.

## Fault Isolation Results

### Implicated Path
- **Kernel:** `NbnxnKernelCpu1x1_PlainC` (and 4x4 variants).
- **Subsystem:** Non-bonded inner loop.
- **Trigger:** Any simulation using 9-6 repulsion where SIMD is disabled.

### Confirmed Mechanism: Ghost Energy
In `src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h`, the Coulomb energy for shifted potentials is calculated as:
```cpp
const real qq = skipmask * qi[i] * q[aj];
real vcoul = qq * (interact * rinv + reactionFieldCoefficient * rsq - reactionFieldShift);
```
**The Defect:**
- `skipmask` is 1.0 for all pairs in the neighbor list.
- `interact` is 0.0 for excluded pairs.
- Even when `interact == 0`, `vcoul` becomes `qq * (reactionFieldCoefficient * rsq - reactionFieldShift)`.
- This adds a large, spurious energy term for every excluded bond, angle, and dihedral in the system.

### Ranking of Suspicions
1.  **Reference Kernel Ghost Energy (CONFIRMED):** Measured directly. Magnitude is sufficient to explain the 826K thermal runaway.
2.  **Reference Kernel Ghost Force (LIKELY):** The `k_rf2` force term in the RF path also lacks proper `interact` gating.
3.  **PME Split Mismatch (CONFIRMED in TP1.4):** Mathematical inconsistency in $C_6$ scaling.

## Impact on TP1.3 Explanation
The initial hypothesis that PME was the sole cause of instability is **broadened**. While PME is mathematically incorrect, the fallback reference kernels used in "cut-off only" mode contain a logical error in exclusion handling that is physically more destructive to simulation stability.

## Action Plan
- Correct the masking logic in `kernel_ref_inner.h` to ensure `interact` gates all energy and force terms, not just the $1/r$ part.
- Align SIMD prefactors as identified in TP1.4.
