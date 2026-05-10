# Current Status Note

## Update

This note still describes the default broad-scope boundary outside any explicitly expanded subset.

For the new strict-PCFF-qualified charged dense-box subset that now survives M1-M3 evidence review, see [M11.1 Charged Subset Expansion](validation_report_m11_1_pcff_charged_subset.md).

For the strict M4 rerun on that qualified subset, see [M11.2 Strict Charged M4 Validation](validation_report_m11_2_pcff_charged_m4.md).

For the M5 workflow-level chemistry expansion, see [M11.3 M5 Chemistry-Scope Expansion](validation_report_m11_3_pcff_charged_m5.md).

For the broader high-pressure M2 dense charged parity campaign, see [M11.4 M2 Broader Dense Charged Parity](validation_report_m11_4_pcff_charged_m2_broad.md).

For the ambient `1 bar` M2 failure root-cause diagnostics, see [M11.5 M2 Ambient 1 Bar Root-Cause Note](validation_report_m11_5_pcff_charged_m2_1bar_root_cause.md).

For the staged `250 bar -> 1 bar` follow-up protocol, see [M11.6 M2 Staged 1 Bar Protocol](validation_report_m11_6_pcff_charged_m2_staged_1bar_protocol.md). This is now a pressure-preconditioned staged `1 bar` PASS, not an ambient `1 bar` equilibrium PASS.

For the exact TP1 `dense_salt_polymer` thermal-runaway recovery, see [TP1 Charged Long-Equilibration Recovery](validation_report_tp1.md).

For the latest GROMACS-only PolyGen exact r-RESPA CPU/GPU screening, see [PolyGen CPU/GPU Transport Screening, 2026-05-10](polygen_cpu_gpu_transport_screening_20260510.md). This records CPU/GPU stage-metric parity and 10 ns NE screening only; it is not a LAMMPS-vs-GROMACS transport parity claim.

For the active unresolved work list after the Markdown audit, see [Current Active Issues](current_active_issues.md).

## Bottom Line

Current evidence supports a bounded PCFF / ion-compatible claim:

> The bridge can deterministically type and export the frozen PT8 supported SPE subset, and it preserves charged Class2/LJ 9-6/long-range Coulomb mechanics on frozen small fixtures. One explicit strict-PCFF-qualified charged dense-box subset (`gate_h_dense_salt_polymer_2x2x2`) passes M4 separated mechanical, structural / density, and short-horizon transport-facing CPU/GPU observable validation. M11.4 broadens M2 dense charged parity to two strict-PCFF-qualified dense charged pairs over a predeclared `250 bar`, 100 ps target / final 50 ps campaign: `gate_h_dense_salt_polymer_2x2x2` and `monoglyme_ethane_litfsi_1to1_dense18`. M11.6 adds pressure-preconditioned staged `250 bar -> 1 bar` dense parity for those same two pairs over 100 ps precondition / 100 ps target / final 50 ps target analysis. M5 adds one workflow-level charged assembly containing an acyclic alkane neutral additive: `monoglyme_ethane_litfsi_1to1`. The historical TP1 `dense_salt_polymer` thermal-runaway blocker is superseded only for the corrected 5 ns NPT rerun. The project still does not support broad PCFF chemistry coverage, direct ambient 1 bar equilibrium dense charged parity, generic dense charged ensemble readiness, LAMMPS-vs-GROMACS charged transport parity, endpoint continuation safety from the corrected TP1 final coordinates, or charged transport readiness.

Machine-readable sources of truth:

- [narrow_claim_summary.json](../tests/reference_results/pcff_ion_narrow_claim/narrow_claim_summary.json)
- [support_matrix.json](../tests/reference_results/pcff_ion_narrow_claim/support_matrix.json)

## Chemistry Scope Statement

Baseline deterministic typing/export is validated for three frozen, net-neutral SPE cases:

- `monoglyme_litfsi_1to1`
- `diglyme_litfsi_1to1`
- `triglyme_litfsi_2to2`

That subset is limited to:

- linear methoxy-capped acyclic polyether oligomers
- explicit `Li+`
- explicit TFSI-like sulfonimide anions

What this means:

- PT8-supported typing/export is exact only for that frozen subset
- M5 adds exactly one workflow-level `acyclic_alkane` neutral-additive assembly: `monoglyme_ethane_litfsi_1to1`
- broader PCFF chemistry is not covered
- the CSV-snapshot release target remains unsupported: `0 / 6042` unique SMILES currently pass end-to-end coverage

Primary evidence:

- [PT8 typing validation summary](../tests/reference_results/pt8_typing_validation/validation_summary.json)
- [PT8 per-case results](../tests/reference_results/pt8_typing_validation/per_case_results.json)
- [M5 chemistry expansion report](../tests/reference_results/pcff_charged_expansion/m5_monoglyme_ethane_litfsi_1to1/m5_chemistry_expansion_report.json)
- [M11.4 M2 broad high-pressure campaign summary](../tests/reference_results/pcff_charged_expansion/m2_broad_v3_250bar_mttk_skin4_lmp1/m2_broad_campaign_summary.json)
- [M11.5 ambient 1 bar root-cause summary](../tests/reference_results/pcff_charged_expansion/m2_1bar_root_cause/m2_1bar_root_cause_summary.json)
- [M11.6 staged 1 bar campaign summary](../tests/reference_results/pcff_charged_expansion/m2_broad_v4_staged_250bar_to_1bar/m2_staged_1bar_campaign_summary.json)
- [CSV scope audit summary](../tests/reference_results/csv_scope_audit/coverage_audit_summary.json)

## Charged Semantic Scope Statement

Charged support is exact only for the following frozen semantics:

- emitted charged topology contract uses `lj/class2/coul/long`
- sixth-power LJ mixing is preserved
- `special_bonds lj/coul 0.0 0.0 1.0 angle no dihedral no` is preserved
- salt systems keep the frozen long-range electrostatics requirement
- small charged fixtures preserve combined energy/force mechanics
- the explicit `gate_h_dense_salt_polymer_2x2x2` subset preserves M4 separated mechanical, structural / density, and short-horizon transport-facing CPU/GPU observable parity on a strict-PCFF-qualified pair
- the M11.4 high-pressure M2 campaign preserves dense charged density/volume parity on two strict-PCFF-qualified dense charged pairs over 100 ps target / final 50 ps at `250 bar`
- the M11.6 pressure-preconditioned staged M2 campaign preserves dense charged density/volume parity on those two pairs over 100 ps `250 bar` precondition followed by 100 ps `1 bar` target / final 50 ps target analysis
- the explicit `monoglyme_ethane_litfsi_1to1` M5 workflow path types, exports, and GROMACS-smoke-validates one acyclic alkane neutral additive in a charged Li/TFSI assembly
- the corrected 5 ns TP1 `dense_salt_polymer` NPT rerun resolves the historical thermal-runaway blocker for that exact system/protocol

Charged support is only approximate for:

- one 100 ps dense charged-box diagnostic where mean potential energy and temperature look close, but the run remains only `partial`

Charged support is unvalidated for:

- explicit charged cross-pair overrides
- an exact charged short-time trajectory gate, because the checked-in M10.1 charged NVE artifact bundle is missing

Charged support is unsupported for:

- direct ambient 1 bar equilibrium dense charged density/volume parity
- dense charged density/volume parity outside the explicit M11.1/M11.2/M11.4/M11.6 subsets
- dense ensemble or transport support for the M5 `monoglyme_ethane_litfsi_1to1` chemistry outside the explicit M11.4 `250 bar` and M11.6 staged `1 bar` dense18 parity campaigns
- endpoint continuation safety from the corrected TP1 final coordinates, because the final box is smaller than twice the 0.9 nm cutoff
- LAMMPS-vs-GROMACS charged transport parity or charged transport readiness
- broad provenance-qualified strict PCFF charged parity beyond the explicit `gate_h_dense_salt_polymer_2x2x2` subset
- arbitrary neutral co-solvent or broad alkane coverage beyond the explicit M5 `ETHANE` additive

Primary evidence:

- [support_matrix.json](../tests/reference_results/pcff_ion_narrow_claim/support_matrix.json)
- [PT8 LAMMPS smoke parity summary](../tests/reference_results/pt8_typing_validation/lammps_smoke_parity_summary.json)
- [PT8.4.1 mixing parity summary](../tests/reference_results/pt8_4_1_mixing_parity/mixing_parity_summary.json)
- [PT8.4 nonbonded parity summary](../tests/reference_results/pt8_4_nonbonded_parity/nonbonded_parity_summary.json)
- [PT8.5 combined parity summary](../tests/reference_results/pt8_5_combined_parity/combined_parity_summary.json)
- [M10.4 charged ensemble summary](../tests/reference_results/m10_4_charged_ensemble_gate/m10_4_summary.json)
- [TP1 recovery summary](../tests/reference_results/tp1_charged_recovery/dense_salt_polymer/recovery_summary.json)
- [TP1 exact recovery audit](../tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/tp1_exact_recovery_audit.json)
- [M4 strict charged validation inventory](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation/m4_strict_validation_inventory.json)
- [M5 chemistry expansion report](../tests/reference_results/pcff_charged_expansion/m5_monoglyme_ethane_litfsi_1to1/m5_chemistry_expansion_report.json)
- [M11.4 M2 broad high-pressure campaign summary](../tests/reference_results/pcff_charged_expansion/m2_broad_v3_250bar_mttk_skin4_lmp1/m2_broad_campaign_summary.json)
- [M11.5 ambient 1 bar root-cause summary](../tests/reference_results/pcff_charged_expansion/m2_1bar_root_cause/m2_1bar_root_cause_summary.json)
- [M11.6 staged 1 bar campaign summary](../tests/reference_results/pcff_charged_expansion/m2_broad_v4_staged_250bar_to_1bar/m2_staged_1bar_campaign_summary.json)
- [M10 method readiness summary](../tests/reference_results/m10/method_readiness_summary.json)

## Non-Claims

The repository does not currently justify any of the following present-tense claims:

- full PCFF readiness across all chemistries
- charged polymer-electrolyte transport readiness
- direct ambient 1 bar equilibrium dense charged parity
- generic dense charged ensemble parity outside the explicit M11.1/M11.2/M11.4/M11.6 subsets
- parser/emitter success as a stand-in for scientifically usable charged support
- ACPYPE/GAFF2-prepared artifacts as strict PCFF parity evidence

## Public Claim To Reuse

Use this sentence when describing current scope:

> Current evidence supports a bounded PCFF / ion-compatible claim: the bridge can deterministically type and export the frozen PT8 supported SPE subset, it preserves charged Class2/LJ 9-6/long-range Coulomb mechanics on frozen small fixtures, the exact TP1 `dense_salt_polymer` thermal-runaway blocker is superseded only for the corrected 5 ns NPT rerun, one explicit strict-PCFF-qualified charged dense-box subset (`gate_h_dense_salt_polymer_2x2x2`) passes M4 separated mechanical, structural / density, and short-horizon transport-facing CPU/GPU observable validation, M11.4 broadens M2 dense charged parity to two strict-PCFF-qualified dense charged pairs at `250 bar` over 100 ps target / final 50 ps, M11.6 adds pressure-preconditioned staged `250 bar -> 1 bar` dense parity for those same pairs over 100 ps precondition / 100 ps target / final 50 ps target analysis, and one M5 workflow-level charged assembly (`monoglyme_ethane_litfsi_1to1`) adds an acyclic alkane neutral additive. It does not support broad PCFF chemistry coverage, direct ambient 1 bar equilibrium dense charged parity, generic dense charged ensemble readiness, LAMMPS-vs-GROMACS charged transport parity, endpoint continuation safety from the corrected TP1 final coordinates, arbitrary neutral co-solvents, or charged transport readiness.
