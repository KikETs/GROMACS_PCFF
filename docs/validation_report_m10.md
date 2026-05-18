# M10 Validation Report

## Milestone Result

M10 now has a reproducible workflow and a machine-readable provenance gate, but the current evidence does **not** support a PCFF scientific-readiness claim.

The decisive point is not the transport error itself. The decisive point is provenance. The current paired GROMACS systems are not PCFF-qualified, so they cannot be used as a strict PCFF-vs-LAMMPS validation set.

Primary outputs:

- [comparison_summary.json](../tests/reference_results/m10/comparison_summary.json)
- [strict_parity_summary.json](../tests/reference_results/m10/strict_parity_summary.json)
- [screening_usefulness_summary.json](../tests/reference_results/m10/screening_usefulness_summary.json)
- [method_readiness_summary.json](../tests/reference_results/m10/method_readiness_summary.json)
- [pcff_paired_provenance_gate.csv](../tests/reference_results/m10/pcff_paired_provenance_gate.csv)

## Provenance Gate Result

The strict paired set candidate IDs are still `14748` and `27670`, but the retained strict paired set is now empty.

- candidate paired systems: `14748`, `27670`
- retained paired systems after PCFF gate: none
- strict parity status: `blocked_by_pcff_provenance`

Machine-readable evidence:

- [strict_parity_summary.json](../tests/reference_results/m10/strict_parity_summary.json)
- [pcff_paired_provenance_gate.csv](../tests/reference_results/m10/pcff_paired_provenance_gate.csv)

Per-system classification:

- `14748`
  - rejected
  - status: `acpype_gaff2_topology`
  - reason: GROMACS paired topology was generated with ACPYPE/GAFF2, not PCFF
  - direct evidence: [atomtyping_attempt1.log](../DL/gromacs/eval_top10_bottom10_stratified100/runs/Traj_14748/atomtyping_attempt1.log), [acpype.log](../DL/gromacs/eval_top10_bottom10_stratified100/runs/Traj_14748/topology/polymer.acpype/acpype.log), [polymer_GMX.itp](../DL/gromacs/eval_top10_bottom10_stratified100/runs/Traj_14748/topology/polymer_GMX.itp)
- `27670`
  - rejected
  - status: `acpype_gaff2_atomtyping_failed`
  - reason: paired GROMACS topology is missing and the preserved typing attempt is ACPYPE/GAFF2
  - direct evidence: [atomtyping_attempt3.log](../DL/gromacs/eval_top10_bottom10_stratified100/runs/Traj_27670/atomtyping_attempt3.log), [acpype.log](../DL/gromacs/eval_top10_bottom10_stratified100/runs/Traj_27670/topology/polymer.acpype/acpype.log)

Reference side remains PCFF/class2:

- [production.in](../DL/LAMMPS_NEW/Traj_14748/MD/production.in)
- [production.in](../DL/LAMMPS_NEW/Traj_27670/MD/production.in)

## Strict Parity

Strict parity is blocked, not passed.

- strict paired-system metric rows: `0`
- strict aggregates: all `null` or `0 compared`
- missing strict metric now explicitly includes `pcff_provenance_gate = blocked`

This is intentional. The workflow no longer pretends that ACPYPE/GAFF2-prepared GROMACS systems are valid PCFF strict-parity evidence.

## Screening Usefulness

Screening summaries still exist, but they are now explicitly marked `not_pcff_qualified`.

- completed GROMACS screening runs: `107 / 120`
- conductivity ranking: weak
  - Spearman `rho = 0.1761`
  - top-10 overlap `0/10`
- transference ranking: poor
  - Spearman `rho = -0.1814`
  - top-10 overlap `0/10`
- local density subset: relatively better
  - mean absolute error `0.0261 g/cm^3`
  - Spearman `rho = 0.7186`

These numbers are still useful as workflow diagnostics. They are **not** valid PCFF method-readiness evidence because the local screening cohort is prepared with ACPYPE/GAFF2 rather than PCFF class2.

Global preparation-path evidence:

- [gromacs_new_phase_atomtyping.py](../DL/gromacs/eval_top10_bottom10_stratified100/phase_scripts/gromacs_new_phase_atomtyping.py)
  - uses ACPYPE
  - includes Li fallback handling

## Other Provenance Diagnostics

These diagnostics remain important, but they are now secondary to the PCFF provenance gate.

- [paired_density_provenance.csv](../tests/reference_results/m10/paired_density_provenance.csv)
  - `14748`: `inconsistent`
  - `27670`: `unavailable`
- [paired_topology_recovery.csv](../tests/reference_results/m10/paired_topology_recovery.csv)
  - `27670` can be dry-run recovered with donor topology `14768`
  - this is topology recovery only, not production recovery
- [paired_artifact_registry_audit.csv](../tests/reference_results/m10/paired_artifact_registry_audit.csv)
  - both paired systems: `derived_metrics_without_raw_artifacts`
- [chain_size_artifact_status.csv](../tests/reference_results/m10/chain_size_artifact_status.csv)
  - both paired systems: `unavailable`

## Transport Mismatch Diagnostics

The transport decomposition still exists:

- [transport_decomposition.csv](../tests/reference_results/m10/transport_decomposition.csv)

It remains useful for debugging, but it is not a valid PCFF scientific comparison until the provenance problem is fixed.

Current diagnostic pattern:

- `sigma_NE` is closer than `sigma_cNE`
- heuristic driver order is `electrostatics > LJ > volume`

This is a debugging hint, not a publishable conclusion.

## Honest Readout

Current M10 status is:

- workflow exists: yes
- machine-readable outputs exist: yes
- strict PCFF paired validation set exists: no
- current screening cohort is PCFF-qualified: no
- scientifically usable for polymer-electrolyte transport claims: no

## Next Actions

1. rebuild the paired GROMACS validation systems with actual PCFF provenance
2. exclude ACPYPE/GAFF2-prepared systems from strict parity until they are rebuilt
3. rerun strict parity only after a PCFF-qualified paired set exists
4. treat all current transport/ranking outputs as provenance-debugging artifacts, not method-readiness evidence
