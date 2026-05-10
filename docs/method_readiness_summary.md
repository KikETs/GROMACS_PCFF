# Method Readiness Summary

## Bottom Line

The current fork is **not** scientifically ready for PCFF-based polymer-electrolyte transport claims.

M11.2 adds one strict-PCFF-qualified charged subset for mechanical, structural / density, and short-horizon transport-facing CPU/GPU observable validation. M11.3 adds one workflow-level acyclic-alkane neutral-additive chemistry subset. Neither closes LAMMPS-vs-GROMACS charged transport parity.

The 2026-05-10 PolyGen exact r-RESPA GROMACS-only CPU/GPU analysis adds a useful screening result, not a readiness claim: 10 ns NE conductivity differs by `2.87%` between CPU and GPU, while HTP-MD-style cNE0 conductivity differs by `182.67%`. The run is below the frozen charged transport duration requirement and does not compare against LAMMPS transport observables.

## Why It Is Blocked

Direct evidence:

- [pcff_paired_provenance_gate.csv](../tests/reference_results/m10/pcff_paired_provenance_gate.csv)
  - `14748` is rejected because the paired GROMACS topology is ACPYPE/GAFF2, not PCFF
  - `27670` is rejected because the paired GROMACS topology is missing and the preserved typing attempt is ACPYPE/GAFF2
- [strict_parity_summary.json](../tests/reference_results/m10/strict_parity_summary.json)
  - candidate paired systems exist
  - retained PCFF-qualified paired systems: `0`
  - strict status: `blocked_by_pcff_provenance`
- [comparison_summary.json](../tests/reference_results/m10/comparison_summary.json)
  - overall readiness: `pcff_provenance_blocked`
  - screening usefulness: `not_pcff_qualified`
- [M11.2 Strict Charged M4 Validation](validation_report_m11_2_pcff_charged_m4.md)
  - one strict-PCFF-qualified charged subset now passes M4 separated validation
  - the transport-facing branch is short-horizon CPU/GPU observable consistency, not LAMMPS-vs-GROMACS transport parity
- [M11.3 M5 Chemistry-Scope Expansion](validation_report_m11_3_pcff_charged_m5.md)
  - one charged assembly with an acyclic alkane neutral additive passes workflow-level typing/export and GROMACS smoke validation
  - it is not dense ensemble or transport evidence
- [PolyGen CPU/GPU Transport Screening, 2026-05-10](polygen_cpu_gpu_transport_screening_20260510.md)
  - latest GROMACS-only CPU/GPU exact r-RESPA 10 ns NE screening is recorded
  - HTP-MD-style cNE0 remains diagnostic-only for this run
  - this is not LAMMPS-vs-GROMACS charged transport parity

## What The Current M10 Data Can Still Be Used For

- provenance debugging
- workflow validation
- exploratory ranking diagnostics with strong caveats
- identifying which artifacts are missing or mismatched

## What It Cannot Be Used For

- PCFF-vs-LAMMPS transport readiness claims
- conductivity or transference publication claims
- chain-size parity claims
- final candidate selection based on the current screening outputs

## Transport Protocol (TP) Status

TP1 exact thermal-runaway status is no longer the blocker for the corrected 5 ns rerun, but transport entry is still blocked.

### TP1 - Charged Long-Equilibration Recovery
- **Status:** thermal-runaway blocker `PASS` only for the corrected 5 ns `dense_salt_polymer` NPT rerun.
- **Remaining audit finding:** the corrected final box is smaller than twice the 0.9 nm cutoff, so endpoint continuation / transport entry remains unsupported.
- **Basis:** [TP1 exact recovery audit](../tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/tp1_exact_recovery_audit.json).

## Remaining Blocking Gaps

- no M10 screening paired system currently passes the GROMACS PCFF provenance gate
- M11.2 has only one explicit strict-PCFF-qualified subset and no LAMMPS-vs-GROMACS transport parity
- M11.3 has only one workflow-level chemistry expansion and no dense/transport validation for that chemistry
- screening cohort is prepared with ACPYPE/GAFF2 rather than PCFF
- paired density provenance is unresolved
- paired GROMACS chain-size artifacts are unavailable
- paired raw production artifacts are missing while the registry still reports completed analysis

## Immediate Next Steps

1. extend the strict M11.2 pair, or build a second strict PCFF pair, to a predeclared transport-grade protocol
2. exclude ACPYPE/GAFF2-prepared M10 screening systems from any strict PCFF transport claim
3. only after that, rerun density/RDF/conductivity/transference strict comparisons with LAMMPS-vs-GROMACS transport-facing evidence
