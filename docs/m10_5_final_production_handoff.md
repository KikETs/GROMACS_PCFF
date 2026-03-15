# M10.5 — Final Production Workflow Handoff

## 1. Introduction
This document establishes the final production handoff package for the GROMACS-PCFF bridge. It consolidates validation evidence from milestones M10.1 through M10.4 and provides operational guidance for deploying the bridge in production environments.

## 2. Readiness Classification

### 2.1 Production-Ready: Neutral Systems
Neutral dense/liquid-phase systems are classified as **Production-Ready**.
- **Evidence:** Sub-1% density parity (0.71%) and ~2% potential energy parity achieved over 100 ps production sampling.
- **Scope:** Supported PCFF chemistry (polymers, oligomers, neutral small molecules).
- **Engine Logic:** Verified use of `comb-rule 4` (SixthPower) and `rep-pow 9.0`.

### 2.2 Qualified-Ready: Charged/Salt Systems
Charged or salt-containing systems are classified as **Qualified-Ready**.
- **Evidence:** Exceptional potential energy parity (0.09%) confirms correct force mapping.
- **Caveats:** Significant density discrepancies (up to 55%) observed in small test boxes due to reciprocal-space virial sensitivities (PME vs PPPM) and slow volume relaxation.
- **Requirement:** Users MUST follow the [Charged Workflow Template](m10_5_charged_workflow_template.md) and perform mandatory density drift monitoring.

### 2.3 Not Yet Validated
The following areas are **OUT OF SCOPE** for the current handoff:
- **Transport Properties:** Diffusion, conductivity, and transference coefficients.
- **Multi-microsecond Dynamics:** Long-range chain entanglement or slow ionic clustering beyond the validated ~100 ps window.
- **Unsupported Chemistry:** Any atom types or functional groups not present in the `lammps_golden` corpus.

## 3. Operational Guidance

### 3.1 Workflow Overview
1.  **Topology Generation:** Use the PCFF emitter to produce GROMACS `.top` and `.gro` files.
2.  **Minimization:** Mandatory steepest descent to resolve initial clashes.
3.  **Equilibration:** Multi-stage NPT (Berendsen preferred for initial stability).
4.  **Production:** NPT or NVT (V-rescale/Parrinello-Rahman recommended for production).

### 3.2 Monitoring Rules
| Observable | Neutral Rule | Charged Rule |
| :--- | :--- | :--- |
| **Temperature** | Stable +/- 2K | Stable +/- 5K |
| **Potential Energy** | Fluctuating around mean | Fluctuating around mean |
| **Density/Volume** | 5-block drift < 1% | **Mandatory** block-wise drift check |
| **Blow-up (NaN)** | Immediate escalation | Immediate escalation |

## 4. Handoff Artifacts
- **Neutral Template:** [docs/m10_5_neutral_workflow_template.md](m10_5_neutral_workflow_template.md)
- **Charged Template:** [docs/m10_5_charged_workflow_template.md](m10_5_charged_workflow_template.md)
- **Troubleshooting:** [docs/m10_5_troubleshooting.md](m10_5_troubleshooting.md)
- **Metadata:** [tests/reference_results/m10_5_handoff_metadata/readiness.json](tests/reference_results/m10_5_handoff_metadata/readiness.json)

## 5. Conclusion
The GROMACS-PCFF bridge is ready for deployment. Neutral systems provide high-fidelity parity with LAMMPS, while charged systems require qualified oversight to manage engine-specific electrostatic sensitivities.
