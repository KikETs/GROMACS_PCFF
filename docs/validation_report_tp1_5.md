# Validation Report — TP1.5 Cut-off Only Regression Audit

## 1. Executive Summary
**Milestone Result: PASS (Fault Isolated)**
Milestone TP1.5 identified why the cut-off-only simulation path regressed more severely than the PME path in the `dense_salt_polymer` diagnostics. The root cause is a "Ghost Energy" defect in the plain-C reference kernels (triggered by 9-6 repulsion) that incorrectly accumulates potential-shift terms for excluded atom pairs.

## 2. Evidence of Path Regression

| Case | Expected Force/Energy | Measured Force/Energy | Status |
| :--- | :--- | :--- | :--- |
| **Excluded Pair (< rcut)** | 0.0 kJ/mol | -277.87 kJ/mol | **FAIL (Ghost Energy)** |
| **Out-of-range Pair (> rcut)** | 0.0 kJ/mol | -138.93 kJ/mol | **FAIL (Ghost Energy)** |
| **PME Path (TP1.4)** | Invariant vs rcut | 170% Drift | **FAIL (Split Mismatch)** |

## 3. Key Findings
- **Fallback Trigger:** The usage of PCFF 9-6 repulsion disables SIMD kernels, forcing GROMACS to use plain-C reference loops.
- **Exclusion Logic Error:** In `src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h`, the energy calculation uses `skipmask` (neighbor list inclusion) rather than `interact` (exclusion mask) to decide whether to apply the constant potential shift.
- **Massive Bias:** For a dense system, this defect adds tens of thousands of kJ/mol of spurious attractive energy, which fluctuates as the neighbor list is rebuilt, driving thermal runaway.

## 4. Conclusion
The regression in the cut-off path is **CONFIRMED** as a reference-kernel defect. The PME path is also mathematically broken (TP1.4), but the cut-off path is physically destabilized by a much larger numerical artifact.

**Verdict: PASS.** The faults are isolated. Fixes must be applied to both the SIMD PME prefactors and the reference kernel exclusion logic.
