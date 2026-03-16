# Method Readiness Summary

## Bottom Line

The current fork is **not** scientifically ready for PCFF-based polymer-electrolyte transport claims.

This is now blocked first by provenance, not by missing summary tables.

## Why It Is Blocked

Direct evidence:

- [pcff_paired_provenance_gate.csv](/home/user/바탕화면/gromacs/tests/reference_results/m10/pcff_paired_provenance_gate.csv)
  - `14748` is rejected because the paired GROMACS topology is ACPYPE/GAFF2, not PCFF
  - `27670` is rejected because the paired GROMACS topology is missing and the preserved typing attempt is ACPYPE/GAFF2
- [strict_parity_summary.json](/home/user/바탕화면/gromacs/tests/reference_results/m10/strict_parity_summary.json)
  - candidate paired systems exist
  - retained PCFF-qualified paired systems: `0`
  - strict status: `blocked_by_pcff_provenance`
- [comparison_summary.json](/home/user/바탕화면/gromacs/tests/reference_results/m10/comparison_summary.json)
  - overall readiness: `pcff_provenance_blocked`
  - screening usefulness: `not_pcff_qualified`

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

The TP milestone set is currently blocked by the audit failure of TP1.

### TP1 — Charged Long-Equilibration Recovery
- **Status:** **FAIL / NOT VERIFIED** (Milestone TP1.1 in progress for record repair).
- **Audit Findings:**
  - Raw logs and energy files are missing for the reported 5 ns run.
  - System identity mismatch: claimed 2,500-atom LiTFSI system, but actual system is 270-atom Na/Cl polymer.
  - No runner script found in the repository.
- **Requirement:** A full TP1.2 rerun is mandatory to restore the integrity of the transport-validation thread.

## Remaining Blocking Gaps

- no paired system currently passes the GROMACS PCFF provenance gate
- screening cohort is prepared with ACPYPE/GAFF2 rather than PCFF
- paired density provenance is unresolved
- paired GROMACS chain-size artifacts are unavailable
- paired raw production artifacts are missing while the registry still reports completed analysis

## Immediate Next Steps

1. rebuild at least one paired GROMACS system with verified PCFF provenance
2. remove ACPYPE/GAFF2-prepared systems from the strict paired set until rebuilt
3. only after that, rerun density/RDF/conductivity/transference strict comparisons
