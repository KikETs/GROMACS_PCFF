# GROMACS-PCFF Bridge

This repository tracks an evidence-backed PCFF/Class2 bridge between frozen LAMMPS fixtures and a specialized GROMACS fork. Broad PCFF chemistry, generic charged dense-box readiness, and charged transport readiness are still not claimed. One explicit charged dense-box subset is validated on a strict-PCFF-qualified paired artifact path; see the M11.1 and M11.2 reports below. M11.3 adds one workflow-level chemistry expansion with an acyclic alkane neutral additive in a charged Li/TFSI assembly. M11.4 upgrades M2 dense charged parity from the old one-system 10 ps result to a two-system, 100 ps / final 50 ps high-pressure `250 bar` campaign. M11.5 records diagnostic root-cause evidence for the still-unresolved direct ambient `1 bar` dense-parity failure. M11.6 adds a pressure-preconditioned staged `250 bar -> 1 bar` dense-parity pass, but it is not an ambient `1 bar` equilibrium PASS. The historical TP1 `dense_salt_polymer` thermal-runaway blocker is separately superseded by a corrected 5 ns exact-system NPT rerun, but that endpoint is not transport-entry-ready because the final box is smaller than twice the 0.9 nm cutoff.

## Current Status
The baseline implementation boundary is documented in [Current Status Note](docs/current_status_note.md). The explicit charged dense-box subset expansion is documented in [M11.1 Charged Subset Expansion](docs/validation_report_m11_1_pcff_charged_subset.md), [M11.2 Strict Charged M4 Validation](docs/validation_report_m11_2_pcff_charged_m4.md), [M11.3 M5 Chemistry-Scope Expansion](docs/validation_report_m11_3_pcff_charged_m5.md), [M11.4 M2 Broader Dense Charged Parity](docs/validation_report_m11_4_pcff_charged_m2_broad.md), [M11.5 M2 Ambient 1 Bar Root-Cause Note](docs/validation_report_m11_5_pcff_charged_m2_1bar_root_cause.md), and [M11.6 M2 Staged 1 Bar Protocol](docs/validation_report_m11_6_pcff_charged_m2_staged_1bar_protocol.md).

Current evidence supports this present-tense claim:

> The bridge can deterministically type and export the frozen PT8 supported SPE subset, and it preserves charged Class2/LJ 9-6/long-range Coulomb mechanics on frozen small fixtures. In addition, one strict-PCFF-qualified charged dense-box subset is validated on the derived `gate_h_dense_salt_polymer_2x2x2` pair, M4 separated strict validation passes for that pair, M11.4 broadens M2 dense charged parity to two strict-PCFF-qualified dense charged pairs over a predeclared `250 bar`, 100 ps target / final 50 ps campaign, and M11.6 adds pressure-preconditioned staged `250 bar -> 1 bar` dense parity for the same two pairs over 100 ps precondition / 100 ps target / final 50 ps target analysis. M5 adds one workflow-level charged assembly containing an acyclic alkane neutral additive: `monoglyme_ethane_litfsi_1to1`. The exact TP1 `dense_salt_polymer` thermal-runaway blocker is also superseded by a corrected 5 ns NPT rerun with the intended `tcoupl`, `pcoupl`, and `gen-vel` keys applied. Broad PCFF chemistry, direct ambient 1 bar equilibrium dense charged parity, generic charged dense-box readiness, LAMMPS-vs-GROMACS charged transport parity, endpoint continuation safety from the TP1 final coordinates, and charged transport readiness are still not claimed.

## Quick Start
1.  **Scope First:** Read the [Current Status Note](docs/current_status_note.md) before using the emitter or workflow templates.
2.  **Topology:** Use the PCFF emitter only within the supported chemistry scope documented there.
3.  **Workflow References:** Use the [Neutral Workflow Template](docs/m10_5_neutral_workflow_template.md) as an operational example, and treat the [Charged Workflow Template](docs/m10_5_charged_workflow_template.md) as a diagnostic-only reference, not as a readiness guarantee.
4.  **Status / Validation:** Start with the [Current Status Note](docs/current_status_note.md) for the default support boundary, then review [M11.1 Charged Subset Expansion](docs/validation_report_m11_1_pcff_charged_subset.md), [M11.2 Strict Charged M4 Validation](docs/validation_report_m11_2_pcff_charged_m4.md), [M11.3 M5 Chemistry-Scope Expansion](docs/validation_report_m11_3_pcff_charged_m5.md), [M11.4 M2 Broader Dense Charged Parity](docs/validation_report_m11_4_pcff_charged_m2_broad.md), [M11.5 M2 Ambient 1 Bar Root-Cause Note](docs/validation_report_m11_5_pcff_charged_m2_1bar_root_cause.md), and [M11.6 M2 Staged 1 Bar Protocol](docs/validation_report_m11_6_pcff_charged_m2_staged_1bar_protocol.md) for explicit expansions and unresolved ambient diagnostics.

## Key Documentation
- [Current Status Note](docs/current_status_note.md)
- [M11.1 Charged Subset Expansion](docs/validation_report_m11_1_pcff_charged_subset.md)
- [M11.2 Strict Charged M4 Validation](docs/validation_report_m11_2_pcff_charged_m4.md)
- [M11.3 M5 Chemistry-Scope Expansion](docs/validation_report_m11_3_pcff_charged_m5.md)
- [M11.4 M2 Broader Dense Charged Parity](docs/validation_report_m11_4_pcff_charged_m2_broad.md)
- [M11.5 M2 Ambient 1 Bar Root-Cause Note](docs/validation_report_m11_5_pcff_charged_m2_1bar_root_cause.md)
- [M11.6 M2 Staged 1 Bar Protocol](docs/validation_report_m11_6_pcff_charged_m2_staged_1bar_protocol.md)
- [TP1 Charged Long-Equilibration Recovery](docs/validation_report_tp1.md)
- [Machine-Readable Support Matrix](tests/reference_results/pcff_ion_narrow_claim/support_matrix.json)
- [Known Limitations](docs/known_limitations.md)
- [Troubleshooting Guide](docs/m10_5_troubleshooting.md)

## License
Distributed under the LGPL v2.1. See `COPYING` for details.
