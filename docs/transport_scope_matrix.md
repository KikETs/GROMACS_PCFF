# Current Charged Transport Status

The older TP0 wording overstated the present charged path and is withdrawn.

This file now freezes only the current charged-transport boundary. It is not a neutral transport sign-off.

Source of truth:

- [Current Status Note](current_status_note.md)
- [Machine-Readable Support Matrix](../tests/reference_results/pcff_ion_narrow_claim/support_matrix.json)

## Charged Salt-in-Polymer Status

| Observable / Claim | Current Status | Basis |
| :--- | :--- | :--- |
| Diffusion ($D$) | `unsupported` | M10 method-readiness remains provenance-blocked |
| Conductivity ($\sigma$) | `unsupported` | No provenance-qualified charged transport claim survives |
| Transference ($t_+$) | `unsupported` | M11.2 adds only short-horizon CPU/GPU observable consistency, not transport readiness |
| cNE-style charged transport language | `unsupported` | Do not promote M11.2 transport-facing outputs to readiness claims |
| Long NPT conditioning for charged transport entry | `unsupported` | Corrected TP1 5 ns rerun resolves the thermal-runaway blocker only; its final box/cutoff margin fails endpoint continuation safety |

## Estimator Status

No charged transport estimator currently has a defensible PCFF-readiness claim in this repository, including:

- Einstein / MSD
- cNE-derived conductivity or transference
- Green-Kubo
- Einstein-Helfand

## What Survives

Only a narrower charged mechanics claim survives:

- frozen small-fixture charged mechanics
- frozen topology/semantics contract preservation
- corrected TP1 exact-system thermal-runaway recovery only
- M11.2 short-horizon transport-facing CPU/GPU observable consistency only on the strict `gate_h_dense_salt_polymer_2x2x2` subset

That survival path does not close charged transport.
