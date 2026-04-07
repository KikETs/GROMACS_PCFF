# M11.1 — Strict PCFF Charged Subset Expansion

## Scope

This report does not claim broad charged PCFF readiness.

The M4 rerun on this same qualified subset is documented separately in [M11.2 Strict Charged M4 Validation](validation_report_m11_2_pcff_charged_m4.md).

It documents one explicit capability expansion beyond the prior PT8-only / small-fixture boundary:

- strict-PCFF-qualified paired charged dense-box validation for the derived `gate_h_dense_salt_polymer_2x2x2` system
- dense charged density / volume parity on that paired set
- extended GROMACS continuation stability from the paired target endpoint

The separate exact-system TP1 `dense_salt_polymer` thermal-runaway recovery is documented in [TP1 Charged Long-Equilibration Recovery](validation_report_tp1.md).

## Artifact Inventory

Primary artifacts:

- [Qualified pair manifest](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/qualified_pair_manifest.json)
- [M1-M3 summary](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m1_m3_summary.json)
- [Dense NPT parity report](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/paired_npt/dense_npt_parity_report.json)
- [Initial 20 ps NVT stability report](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/long_nvt_stability/long_nvt_stability_report.json)
- [Extended 50 ps M3 recovery report](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/long_nvt_stability_50ps/m3_recovery_report.json)
- [Extended 50 ps M3 recovery protocol](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/long_nvt_stability_50ps/m3_recovery_protocol.json)
- [Exact TP1 recovery audit](../tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/tp1_exact_recovery_audit.json)
- [M4 strict validation inventory](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation/m4_strict_validation_inventory.json)

Raw run bundles:

- [Paired GROMACS NPT root](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/paired_npt/gromacs)
- [Paired LAMMPS NPT root](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/paired_npt/lammps)
- [Initial 20 ps GROMACS NVT continuation root](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/long_nvt_stability)
- [Extended 50 ps GROMACS NVT continuation root](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/long_nvt_stability_50ps)
- [M4 strict validation root](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation)

## Protocol

Paired-system provenance:

- source fixture is derived from repository-owned raw `dense_salt_polymer` artifacts
- GROMACS topology is generated directly from the repository PCFF/Class2 fixture path
- ACPYPE / GAFF2 / surrogate topology preparation is not used

NPT parity protocol:

- system: `gate_h_dense_salt_polymer_2x2x2`
- warmup: 5 ps GROMACS-only `md` + `v-rescale` + `Berendsen`
- target window: 10 ps paired NPT
- GROMACS target: `md-vv` + `nose-hoover` + `MTTK`
- LAMMPS target: `fix npt temp 300 300 100 iso 1 1 1000`
- analysis window: final 5 ps
- thresholds: density relative difference <= `0.05`, volume relative difference <= `0.05`

NVT stability protocol:

- continuation source: paired GROMACS NPT endpoint above
- integrator / thermostat: `md-vv` + `nose-hoover`
- duration: 50 ps
- extension path: copied the initial 20 ps NVT bundle, then used `gmx convert-tpr -extend 30` and `gmx mdrun -cpi nvt.cpt -append`
- analysis window: final 25 ps
- thresholds: mean temperature `300 +/- 20 K`, max temperature <= `400 K`

## Results

Dense NPT parity:

- density relative difference: `0.04540`
- volume relative difference: `0.04807`
- density parity status: `PASS`
- volume parity status: `PASS`

GROMACS NVT continuation:

- mean temperature over final 25 ps: `300.38 K`
- max temperature over final 25 ps: `312.88 K`
- stability status: `PASS`

## Capability Delta

Previous boundary:

- PT8-supported frozen SPE typing/export only
- frozen small charged mechanics only
- no strict-PCFF-qualified charged dense-box pair
- no dense charged density / volume parity pass
- no surviving charged continuation on a qualified dense pair

New boundary added by this report:

- one strict-PCFF-qualified charged dense-box pair survives audit: `gate_h_dense_salt_polymer_2x2x2`
- one dense charged parity path now passes the predeclared 5% density / volume thresholds
- one extended charged continuation path now survives 50 ps GROMACS NVT stability from the paired target endpoint
- one M4 strict charged validation path now separates and passes mechanical parity, structural / density parity, and short-horizon transport-facing CPU/GPU observable parity on the same qualified subset

## Non-Claims

This report still does not justify:

- broad PCFF charged support across chemistries
- generic charged-system production readiness
- charged transport readiness
- LAMMPS-vs-GROMACS charged transport parity
- any claim broader than this one explicit subset
- endpoint continuation safety from the corrected TP1 final coordinates

## Public Claim

Use this sentence when describing the expanded subset:

> Current evidence now supports one explicit strict-PCFF-qualified charged dense-box subset beyond the prior PT8-only boundary: the derived `gate_h_dense_salt_polymer_2x2x2` pair passes density and volume parity within 5% over a 10 ps target NPT window after a 5 ps GROMACS-only Berendsen warmup, a 50 ps GROMACS NVT continuation from that paired target endpoint remains thermally stable, and M4 separated strict validation passes mechanical parity, structural / density parity, and short-horizon transport-facing CPU/GPU observable parity. Separately, the exact TP1 `dense_salt_polymer` thermal-runaway blocker is superseded by a corrected 5 ns NPT rerun. This does not establish broad charged PCFF readiness, endpoint continuation safety from the TP1 final coordinates, LAMMPS-vs-GROMACS charged transport parity, or charged transport readiness.
