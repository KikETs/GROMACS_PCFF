# M10.4 — Charged Dense-Box Recovery Findings

## Current Readout

This file now records a narrower conclusion than the original wording.

The 100 ps `dense_salt_polymer` run remains useful as a short-horizon charged diagnostic, but it does not support a charged readiness claim.

## What The Artifact Shows

| Observable | GROMACS | LAMMPS | Current Interpretation |
| :--- | :--- | :--- | :--- |
| Mean potential energy | close | close | short-horizon diagnostic only |
| Mean temperature | close | close | short-horizon diagnostic only |
| Volume | unstable / drifting | converged enough | not parity-supporting |
| Density | unstable / drifting | converged enough | not parity-supporting |

Direct machine-readable basis:

- [M10.4 summary](../tests/reference_results/m10_4_charged_ensemble_gate/m10_4_summary.json)
  - `parity_status = partial`
  - `density_parity_rel_diff = 0.5536`
  - `volume_parity_rel_diff = 0.3749`
  - GROMACS density / volume status = `failed / unstable`

## Current Boundary

What survives:

- a short-horizon dense charged-box energy / temperature sanity artifact

What does not survive:

- dense charged ensemble parity from this M10.4 artifact
- charged density readiness
- charged production or transport entry

## Superseding Evidence

Later evidence supersedes only parts of the old blocker:

- [M11.1 Charged Subset Expansion](validation_report_m11_1_pcff_charged_subset.md) closes density / volume parity only for the explicit `gate_h_dense_salt_polymer_2x2x2` subset.
- [M11.2 Strict Charged M4 Validation](validation_report_m11_2_pcff_charged_m4.md) closes separated M4 validation only for that same subset.
- [TP1 exact recovery audit](../tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/tp1_exact_recovery_audit.json) resolves the exact TP1 thermal-runaway blocker only for the corrected 5 ns rerun.

None of those later results turns this M10.4 artifact into a `PASS (with caveats)` claim.
