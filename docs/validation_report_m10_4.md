# M10.4 — Charged Dense-Box Ensemble Diagnostic

## Current Interpretation

This file is retained because the raw M10.4 artifact bundle still matters.

What changed is the claim boundary:

- the earlier wording overstated M10.4 as a charged-readiness pass
- the checked-in machine-readable summary already says the result is only `partial`
- later M11.1/M11.2 evidence closes density/volume and separated M4 validation only for the explicit `gate_h_dense_salt_polymer_2x2x2` subset; it does not convert this M10.4 artifact into a pass
- corrected TP1 later resolved the exact thermal-runaway blocker only; it did not close M10.4 density/volume parity or charged transport readiness

## What M10.4 Still Supports

On the frozen `dense_salt_polymer` fixture only:

- mean potential energy is close over a 100 ps window
- mean temperature is close over the same short window
- the run is useful as a short-horizon diagnostic artifact

Primary evidence:

- [M10.4 summary](../tests/reference_results/m10_4_charged_ensemble_gate/m10_4_summary.json)
- [Per-system report](../tests/reference_results/m10_4_charged_ensemble_gate/dense_salt_polymer/report.json)

## What M10.4 Does Not Support

- charged ensemble readiness
- dense charged density or volume parity from this M10.4 artifact
- charged production entry
- charged transport readiness

Direct reasons:

- density parity differs by about `55.36%`
- volume parity differs by about `37.49%`
- GROMACS density / volume status is `failed / unstable`
- overall parity status is `partial`

## Current Verdict

M10.4 is a charged dense-box diagnostic, not a charged-support pass.
