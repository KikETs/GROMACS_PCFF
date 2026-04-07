# M10.5 — Historical Handoff Reevaluation

## Current Interpretation

The original M10.5 handoff language is no longer valid as a present-tense readiness claim.

Why:

- charged dense-box density parity was not closed by M10.5; it is closed only later for the explicit M11.1/M11.2 `gate_h_dense_salt_polymer_2x2x2` subset
- the historical TP1 thermal-runaway failure is superseded only by a corrected 5 ns exact-system rerun, and that endpoint still fails cutoff/box continuation-safety audit
- M10 remains provenance-blocked for transport claims; M11.2 adds only short-horizon transport-facing CPU/GPU observable consistency on one strict-PCFF subset

## What Still Remains Useful

- the workflow templates remain useful as operational examples
- the troubleshooting guide remains useful
- the historical handoff packaging still explains how the old bundle was assembled
- the current corrected TP1 recovery is documented separately and does not revive M10.5 readiness language
- the M11.2 M4 validation report is the current source for the explicit strict charged subset rerun, not this M10.5 handoff
- the M11.3 M5 report is the current source for the explicit workflow-level chemistry expansion, not this M10.5 handoff

## What Is Withdrawn

- any claim of formal production handoff
- any split of `neutral = ready` and `charged = qualified` as current status
- any machine-readable metadata that implies charged readiness

## Current Source Of Truth

- [Current Status Note](current_status_note.md)
- [M11.2 Strict Charged M4 Validation](validation_report_m11_2_pcff_charged_m4.md)
- [M11.3 M5 Chemistry-Scope Expansion](validation_report_m11_3_pcff_charged_m5.md)
- [Narrow claim summary](../tests/reference_results/pcff_ion_narrow_claim/narrow_claim_summary.json)
- [Support matrix](../tests/reference_results/pcff_ion_narrow_claim/support_matrix.json)
