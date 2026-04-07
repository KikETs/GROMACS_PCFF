#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "tests" / "reference_results" / "pcff_ion_narrow_claim"


def load_json(relative_path: str):
    path = REPO_ROOT / relative_path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def find_case(case_results: dict, case_id: str) -> dict:
    for entry in case_results["cases"]:
        if entry["case_id"] == case_id:
            return entry
    raise KeyError(f"missing PT8 case: {case_id}")


def find_summary_row(rows: list[dict], system_id: str) -> dict:
    for row in rows:
        if row["system_id"] == system_id:
            return row
    raise KeyError(f"missing summary row: {system_id}")


def main() -> None:
    pt8_validation_path = "tests/reference_results/pt8_typing_validation/validation_summary.json"
    pt8_cases_path = "tests/reference_results/pt8_typing_validation/per_case_results.json"
    smoke_path = "tests/reference_results/pt8_typing_validation/lammps_smoke_parity_summary.json"
    mixing_path = "tests/reference_results/pt8_4_1_mixing_parity/mixing_parity_summary.json"
    nonbonded_path = "tests/reference_results/pt8_4_nonbonded_parity/nonbonded_parity_summary.json"
    combined_path = "tests/reference_results/pt8_5_combined_parity/combined_parity_summary.json"
    m10_4_path = "tests/reference_results/m10_4_charged_ensemble_gate/m10_4_summary.json"
    tp1_path = "tests/reference_results/tp1_charged_recovery/dense_salt_polymer/recovery_summary.json"
    tp1_exact_path = "tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/tp1_exact_recovery_audit.json"
    m11_npt_path = "tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/paired_npt/dense_npt_parity_report.json"
    m4_inventory_path = "tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation/m4_strict_validation_inventory.json"
    m4_mechanical_path = "tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation/mechanical_parity/mechanical_parity_report.json"
    m4_structural_path = "tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation/structural_density_parity/structural_density_parity_report.json"
    m4_transport_path = "tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation/transport_facing_parity/transport_facing_parity_report.json"
    m5_report_path = "tests/reference_results/pcff_charged_expansion/m5_monoglyme_ethane_litfsi_1to1/m5_chemistry_expansion_report.json"
    m5_manifest_path = "tests/reference_results/pcff_charged_expansion/m5_monoglyme_ethane_litfsi_1to1/m5_chemistry_scope_manifest.json"
    m11_4_summary_path = "tests/reference_results/pcff_charged_expansion/m2_broad_v3_250bar_mttk_skin4_lmp1/m2_broad_campaign_summary.json"
    m11_4_gate_h_path = "tests/reference_results/pcff_charged_expansion/m2_broad_v3_250bar_mttk_skin4_lmp1/systems/gate_h_dense_salt_polymer_2x2x2/paired_npt/dense_npt_parity_report.json"
    m11_4_m5_dense_path = "tests/reference_results/pcff_charged_expansion/m2_broad_v3_250bar_mttk_skin4_lmp1/systems/monoglyme_ethane_litfsi_1to1_dense18/paired_npt/dense_npt_parity_report.json"
    m11_5_root_cause_path = "tests/reference_results/pcff_charged_expansion/m2_1bar_root_cause/m2_1bar_root_cause_summary.json"
    m11_6_summary_path = "tests/reference_results/pcff_charged_expansion/m2_broad_v4_staged_250bar_to_1bar/m2_staged_1bar_campaign_summary.json"
    m11_6_gate_h_target_path = "tests/reference_results/pcff_charged_expansion/m2_broad_v4_staged_250bar_to_1bar/systems/gate_h_dense_salt_polymer_2x2x2/target_1bar/staged_1bar_parity_report.json"
    m11_6_m5_target_path = "tests/reference_results/pcff_charged_expansion/m2_broad_v4_staged_250bar_to_1bar/systems/monoglyme_ethane_litfsi_1to1_dense18/target_1bar/staged_1bar_parity_report.json"
    m10_readiness_path = "tests/reference_results/m10/method_readiness_summary.json"
    csv_scope_path = "tests/reference_results/csv_scope_audit/coverage_audit_summary.json"

    pt8_validation = load_json(pt8_validation_path)
    pt8_cases = load_json(pt8_cases_path)
    smoke = load_json(smoke_path)
    mixing_rows = load_json(mixing_path)
    nonbonded_rows = load_json(nonbonded_path)
    combined_rows = load_json(combined_path)
    m10_4_rows = load_json(m10_4_path)
    tp1 = load_json(tp1_path)
    tp1_exact = load_json(tp1_exact_path)
    m11_npt = load_json(m11_npt_path)
    m4_inventory = load_json(m4_inventory_path)
    m4_mechanical = load_json(m4_mechanical_path)
    m4_structural = load_json(m4_structural_path)
    m4_transport = load_json(m4_transport_path)
    m5_report = load_json(m5_report_path)
    m5_manifest = load_json(m5_manifest_path)
    m11_4_summary = load_json(m11_4_summary_path)
    m11_4_gate_h = load_json(m11_4_gate_h_path)
    m11_4_m5_dense = load_json(m11_4_m5_dense_path)
    m11_5_root_cause = load_json(m11_5_root_cause_path)
    m11_6_summary = load_json(m11_6_summary_path)
    m11_6_gate_h_target = load_json(m11_6_gate_h_target_path)
    m11_6_m5_target = load_json(m11_6_m5_target_path)
    m10_readiness = load_json(m10_readiness_path)
    csv_scope = load_json(csv_scope_path)

    supported_case_ids = list(pt8_validation["supported_scope"]["passed_case_ids"])
    supported_case_details = [
        {
            "case_id": case_id,
            "display_name": find_case(pt8_cases, case_id)["display_name"],
            "description": find_case(pt8_cases, case_id)["description"],
        }
        for case_id in supported_case_ids
    ]
    component_families = sorted(
        {
            component["classification_family"]
            for case in pt8_cases["cases"]
            for component in case["components"]
            if component.get("classification_family")
        }
    )

    smoke_reference = smoke["reference_system"]
    mixing_row = find_summary_row(mixing_rows, "mixing_toy")
    exclusion_row = find_summary_row(nonbonded_rows, "exclusion_toy")
    combined_charged_row = find_summary_row(combined_rows, "small_salt_polymer_box")
    m10_4_row = find_summary_row(m10_4_rows, "dense_salt_polymer")

    charged_nve_dir = REPO_ROOT / "tests/reference_results/m10_1_trajectory_gate/small_salt_polymer_box_nve_dt0.0001"
    charged_nve_gate = REPO_ROOT / "tests/reference_results/m10_1_trajectory_gate/m10_1_gate_decision.json"

    density_rel_percent = round(m10_4_row["density_parity_rel_diff"] * 100.0, 2)
    volume_rel_percent = round(m10_4_row["volume_parity_rel_diff"] * 100.0, 2)
    tp1_max_block_temp = max(block["temperature_mean"] for block in tp1["block_analysis"]["blocks"])

    public_claim = (
        "Current evidence supports a bounded PCFF / ion-compatible claim: "
        "the bridge can deterministically type and export the frozen PT8 supported SPE subset, "
        "and it preserves charged Class2/LJ 9-6/long-range Coulomb mechanics on frozen small fixtures. "
        "Beyond that baseline, one strict-PCFF-qualified charged dense-box pair (`gate_h_dense_salt_polymer_2x2x2`) has "
        "mechanical parity and short-horizon transport-facing CPU/GPU observable parity evidence, and M11.4 broadens dense "
        "density/volume parity to two strict-PCFF-qualified dense charged pairs at 250 bar over a 100 ps target / final 50 ps window: "
        "`gate_h_dense_salt_polymer_2x2x2` and `monoglyme_ethane_litfsi_1to1_dense18`. "
        "M11.6 adds pressure-preconditioned staged 1 bar dense parity for those same pairs over a 100 ps 250 bar precondition, "
        "100 ps 1 bar target, and final 50 ps target window. "
        "M5 also adds one workflow-level charged assembly containing an acyclic alkane neutral additive "
        "(`monoglyme_ethane_litfsi_1to1`). "
        "It does not support broad PCFF chemistry coverage, direct ambient 1 bar equilibrium dense charged parity, "
        "generic dense charged ensemble readiness, or charged transport readiness."
    )
    chemistry_scope_statement = (
        "Baseline deterministic typing/export is validated for three frozen, net-neutral SPE cases "
        "(monoglyme_litfsi_1to1, diglyme_litfsi_1to1, triglyme_litfsi_2to2): "
        "linear methoxy-capped acyclic polyether oligomers with explicit Li+ and explicit TFSI-like sulfonimide. "
        "M5 adds one workflow-level charged assembly with an acyclic alkane neutral additive: `monoglyme_ethane_litfsi_1to1`. "
        "Broader chemistry is not supported; the CSV-snapshot release target still covers 0 of 6042 unique SMILES."
    )
    charged_scope_statement = (
        "Charged support is exact only at the topology/semantics/mechanics level on frozen small fixtures and emitted PT8 SPE cases: "
        "`lj/class2/coul/long`, sixth-power mixing, `special_bonds lj/coul 0.0 0.0 1.0 angle no dihedral no`, "
        "k-space-required salt semantics, and small-fixture combined force/energy parity. "
        "For the explicit `gate_h_dense_salt_polymer_2x2x2` expansion subset, strict PCFF-qualified M4 evidence now separates "
        "GROMACS-vs-LAMMPS run-0 mechanical parity, structural density/volume parity, and short-horizon transport-facing CPU/GPU observable parity. "
        "M11.4 adds high-pressure dense charged density/volume parity across two strict-PCFF-qualified dense charged pairs at 250 bar over a 100 ps target / final 50 ps window. "
        "M11.6 adds pressure-preconditioned staged 1 bar dense charged density/volume parity across those same pairs after a 100 ps 250 bar precondition, "
        "100 ps 1 bar target, and final 50 ps target window. "
        "M5 adds workflow-level support for one acyclic alkane neutral-additive charged assembly: `monoglyme_ethane_litfsi_1to1`. "
        "The historical dense_salt_polymer TP1 thermal-runaway blocker is superseded only for the corrected 5 ns NPT rerun. "
        "Direct ambient 1 bar equilibrium dense charged parity and generic dense charged ensemble behavior remain unclaimed, the corrected TP1 endpoint has a cutoff/box caveat, "
        "and no charged transport readiness claim survives."
    )

    status_definitions = {
        "exact": "Reproducible checked-in artifact directly supports the statement within the frozen scope.",
        "approximate": "Evidence exists, but only as a short-horizon or caveated diagnostic; it does not justify readiness language.",
        "unsupported": "Direct evidence or explicit project scope says the item must not be claimed as supported now.",
        "unvalidated": "The repository exposes the path or mentions the feature, but the checked-in evidence is absent or insufficient.",
    }

    support_matrix = {
        "schema_name": "pcff_ion_support_matrix",
        "schema_version": 1,
        "claim_status_as_of": "2026-04-07",
        "status_definitions": status_definitions,
        "top_level_claim": public_claim,
        "items": [
            {
                "id": "chemistry.pt8_supported_spe_typing_export",
                "domain": "chemistry",
                "status": "exact",
                "supported_scope": {
                    "case_count": len(supported_case_ids),
                    "case_ids": supported_case_ids,
                    "component_families": component_families,
                },
                "claimable_statement": "Deterministic typing/export is validated for the three frozen PT8 SPE cases only.",
                "non_claimable_statement": "This is not chemistry-complete PCFF support for arbitrary polymer electrolytes.",
                "evidence": [
                    {
                        "path": pt8_validation_path,
                        "key": "overall_status",
                        "value": pt8_validation["overall_status"],
                    },
                    {
                        "path": pt8_cases_path,
                        "key": "case_count",
                        "value": pt8_cases["case_count"],
                    },
                ],
            },
            {
                "id": "chemistry.csv_snapshot_release_target",
                "domain": "chemistry",
                "status": "unsupported",
                "claimable_statement": "The broader CSV-snapshot chemistry target is not release-ready.",
                "non_claimable_statement": "Do not describe the current PCFF path as broadly covering the CSV snapshot or general polymer chemistry.",
                "evidence": [
                    {
                        "path": csv_scope_path,
                        "key": "release_readiness.status",
                        "value": csv_scope["release_readiness"]["status"],
                    },
                    {
                        "path": csv_scope_path,
                        "key": "totals.supported_unique_smiles_count",
                        "value": csv_scope["totals"]["supported_unique_smiles_count"],
                    },
                    {
                        "path": csv_scope_path,
                        "key": "totals.unique_smiles_count",
                        "value": csv_scope["totals"]["unique_smiles_count"],
                    },
                ],
            },
            {
                "id": "chemistry.m5_acyclic_alkane_neutral_additive_charged_assembly",
                "domain": "chemistry",
                "status": "exact",
                "supported_scope": {
                    "system_id": m5_report["system_id"],
                    "new_component_family": m5_report["chemistry_delta"]["new_component_family"],
                    "new_role": m5_report["chemistry_delta"]["new_role"],
                    "charged_context": m5_report["chemistry_delta"]["charged_context"],
                },
                "claimable_statement": "M5 validates one acyclic-alkane neutral-additive charged assembly: monoglyme + ethane + Li/TFSI.",
                "non_claimable_statement": "This is not broad alkane, arbitrary co-solvent, dense ensemble, or transport readiness support.",
                "evidence": [
                    {
                        "path": m5_report_path,
                        "key": "status",
                        "value": m5_report["status"],
                    },
                    {
                        "path": m5_report_path,
                        "key": "family_counts",
                        "value": m5_report["family_counts"],
                    },
                    {
                        "path": m5_report_path,
                        "key": "workflow_status.existing_output_matches_rendered",
                        "value": m5_report["workflow_status"]["existing_output_matches_rendered"],
                    },
                    {
                        "path": m5_report_path,
                        "key": "gromacs_smoke.status",
                        "value": m5_report["gromacs_smoke"]["status"],
                    },
                    {
                        "path": m5_report_path,
                        "key": "gromacs_smoke.grompp_warning_count",
                        "value": m5_report["gromacs_smoke"]["grompp_warning_count"],
                    },
                    {
                        "path": m5_manifest_path,
                        "key": "status",
                        "value": m5_manifest["status"],
                    },
                ],
            },
            {
                "id": "charged_semantics.long_range_pair_style_contract",
                "domain": "charged_semantics",
                "status": "exact",
                "claimable_statement": "Charged emitted topologies preserve the `lj/class2/coul/long` pair-style contract on the frozen PT8 SPE cases.",
                "non_claimable_statement": "Contract preservation alone does not prove dense-box ensemble or transport validity.",
                "evidence": [
                    {
                        "path": smoke_path,
                        "key": "reference_system.pair_style",
                        "value": smoke_reference["pair_style"],
                    }
                ],
            },
            {
                "id": "charged_semantics.sixthpower_mixing",
                "domain": "charged_semantics",
                "status": "exact",
                "claimable_statement": "Sixth-power mixing is numerically and contractually preserved on the frozen charged path.",
                "non_claimable_statement": "This does not imply broad pair-override coverage or dense-system readiness.",
                "evidence": [
                    {
                        "path": mixing_path,
                        "key": "mixing_toy.status",
                        "value": mixing_row["status"],
                    },
                    {
                        "path": smoke_path,
                        "key": "reference_system.relies_on_mixing",
                        "value": True,
                    },
                ],
            },
            {
                "id": "charged_semantics.special_bonds_and_14",
                "domain": "charged_semantics",
                "status": "exact",
                "claimable_statement": "The `special_bonds lj/coul 0.0 0.0 1.0 angle no dihedral no` exclusion/1-4 contract is numerically preserved on the frozen path.",
                "non_claimable_statement": "This is not evidence for every charged topology outside the frozen fixtures.",
                "evidence": [
                    {
                        "path": nonbonded_path,
                        "key": "exclusion_toy.status",
                        "value": exclusion_row["status"],
                    },
                    {
                        "path": smoke_path,
                        "key": "reference_system.special_bonds",
                        "value": smoke_reference["special_bonds"],
                    },
                ],
            },
            {
                "id": "charged_semantics.kspace_required_for_salt_systems",
                "domain": "charged_semantics",
                "status": "exact",
                "claimable_statement": "The frozen charged path explicitly requires long-range electrostatics semantics for salt systems.",
                "non_claimable_statement": "This is not a claim that PME/PPPM ensemble outcomes already match at dense-box scale.",
                "evidence": [
                    {
                        "path": smoke_path,
                        "key": "checks[*].checks.requires_kspace_for_salt_system",
                        "value": True,
                    }
                ],
            },
            {
                "id": "charged_semantics.self_only_pair_coeff_policy",
                "domain": "charged_semantics",
                "status": "exact",
                "claimable_statement": "The emitted charged path preserves the self-only pair-coefficient contract expected by the frozen charged reference.",
                "non_claimable_statement": "Explicit cross-pair override behavior is not covered by this statement.",
                "evidence": [
                    {
                        "path": smoke_path,
                        "key": "reference_system.pair_coeff_source",
                        "value": smoke_reference["pair_coeff_source"],
                    }
                ],
            },
            {
                "id": "charged_semantics.explicit_pair_overrides",
                "domain": "charged_semantics",
                "status": "unvalidated",
                "claimable_statement": "No present-tense support claim is made for explicit charged cross-pair overrides.",
                "non_claimable_statement": "Do not imply that arbitrary charged `[ pairtypes ]` or LAMMPS `pair_coeff` overrides are validated.",
                "evidence": [
                    {
                        "path": "docs/validation_report_pt8_4.md",
                        "key": "remaining_gaps",
                        "value": "Pair overrides were not tested in PT8.4.",
                    }
                ],
            },
            {
                "id": "charged_semantics.small_fixture_combined_mechanics",
                "domain": "charged_semantics",
                "status": "exact",
                "claimable_statement": "Combined charged mechanics are numerically consistent on the frozen `small_salt_polymer_box` fixture.",
                "non_claimable_statement": "This is a frozen small-fixture mechanics claim, not broad chemistry readiness or transport readiness.",
                "evidence": [
                    {
                        "path": combined_path,
                        "key": "small_salt_polymer_box.status",
                        "value": combined_charged_row["status"],
                    },
                    {
                        "path": combined_path,
                        "key": "small_salt_polymer_box.relative_force_diff",
                        "value": combined_charged_row["relative_force_diff"],
                    },
                ],
            },
            {
                "id": "charged_semantics.short_time_trajectory_gate",
                "domain": "charged_semantics",
                "status": "unvalidated",
                "claimable_statement": "No exact charged short-time trajectory gate claim survives without a checked-in charged NVE artifact bundle.",
                "non_claimable_statement": "Do not cite M10.1 as an exact charged trajectory gate until the missing charged NVE artifacts are restored.",
                "evidence": [
                    {
                        "path": "docs/validation_report_m10_1.md",
                        "key": "validated_outcomes",
                        "value": "Document cites `small_salt_polymer_box_nve_dt0.0001`, but the checked-in artifact bundle is absent.",
                    },
                    {
                        "path": repo_rel(charged_nve_dir),
                        "key": "exists",
                        "value": charged_nve_dir.exists(),
                    },
                    {
                        "path": repo_rel(charged_nve_gate),
                        "key": "exists",
                        "value": charged_nve_gate.exists(),
                    },
                ],
            },
            {
                "id": "charged_semantics.dense_box_short_horizon_energy_temperature",
                "domain": "charged_semantics",
                "status": "approximate",
                "claimable_statement": "A single 100 ps dense charged-box run showed close mean potential energy and temperature, but only as a partial, caveated diagnostic.",
                "non_claimable_statement": "Do not promote this to charged ensemble readiness, density parity, or transport readiness.",
                "evidence": [
                    {
                        "path": m10_4_path,
                        "key": "dense_salt_polymer.parity_status",
                        "value": m10_4_row["parity_status"],
                    },
                    {
                        "path": m10_4_path,
                        "key": "dense_salt_polymer.gmx.potential_energy.status",
                        "value": m10_4_row["gmx"]["potential_energy"]["status"],
                    },
                    {
                        "path": m10_4_path,
                        "key": "dense_salt_polymer.gmx.temperature.status",
                        "value": m10_4_row["gmx"]["temperature"]["status"],
                    },
                ],
            },
            {
                "id": "charged_semantics.m11_dense_box_density_volume_parity",
                "domain": "charged_semantics",
                "status": "exact",
                "claimable_statement": "The explicit strict-PCFF-qualified `gate_h_dense_salt_polymer_2x2x2` pair passes dense charged density/volume parity within the predeclared 5% thresholds.",
                "non_claimable_statement": "This is one explicit subset; do not generalize it to generic dense charged ensemble readiness.",
                "evidence": [
                    {
                        "path": m11_npt_path,
                        "key": "status",
                        "value": m11_npt["status"],
                    },
                    {
                        "path": m11_npt_path,
                        "key": "parity_metrics.density_rel_diff",
                        "value": m11_npt["parity_metrics"]["density_rel_diff"],
                    },
                    {
                        "path": m11_npt_path,
                        "key": "parity_metrics.volume_rel_diff",
                        "value": m11_npt["parity_metrics"]["volume_rel_diff"],
                    },
                ],
            },
            {
                "id": "charged_semantics.m11_4_broader_high_pressure_dense_parity",
                "domain": "charged_semantics",
                "status": "exact",
                "claimable_statement": "M11.4 closes broader M2 dense charged density/volume parity for two strict-PCFF-qualified dense charged pairs at 250 bar over a 100 ps target / final 50 ps window.",
                "non_claimable_statement": "This is not ambient 1 bar broader dense parity, generic dense charged ensemble readiness, or charged transport readiness.",
                "evidence": [
                    {
                        "path": m11_4_summary_path,
                        "key": "status",
                        "value": m11_4_summary["status"],
                    },
                    {
                        "path": m11_4_summary_path,
                        "key": "systems",
                        "value": m11_4_summary["systems"],
                    },
                    {
                        "path": m11_4_gate_h_path,
                        "key": "parity_metrics.density_rel_diff",
                        "value": m11_4_gate_h["parity_metrics"]["density_rel_diff"],
                    },
                    {
                        "path": m11_4_gate_h_path,
                        "key": "parity_metrics.volume_rel_diff",
                        "value": m11_4_gate_h["parity_metrics"]["volume_rel_diff"],
                    },
                    {
                        "path": m11_4_m5_dense_path,
                        "key": "parity_metrics.density_rel_diff",
                        "value": m11_4_m5_dense["parity_metrics"]["density_rel_diff"],
                    },
                    {
                        "path": m11_4_m5_dense_path,
                        "key": "parity_metrics.volume_rel_diff",
                        "value": m11_4_m5_dense["parity_metrics"]["volume_rel_diff"],
                    },
                ],
            },
            {
                "id": "charged_semantics.m11_6_pressure_preconditioned_staged_1bar_dense_parity",
                "domain": "charged_semantics",
                "status": "exact",
                "claimable_statement": "M11.6 closes pressure-preconditioned staged 1 bar dense charged density/volume parity for two strict-PCFF-qualified dense charged pairs over a 100 ps 250 bar precondition followed by a 100 ps 1 bar target / final 50 ps target window.",
                "non_claimable_statement": "This is not ambient 1 bar equilibrium dense parity, generic dense charged ensemble readiness, or charged transport readiness.",
                "evidence": [
                    {
                        "path": m11_6_summary_path,
                        "key": "status",
                        "value": m11_6_summary["status"],
                    },
                    {
                        "path": m11_6_summary_path,
                        "key": "systems",
                        "value": m11_6_summary["systems"],
                    },
                    {
                        "path": m11_6_gate_h_target_path,
                        "key": "parity_metrics.density_rel_diff",
                        "value": m11_6_gate_h_target["parity_metrics"]["density_rel_diff"],
                    },
                    {
                        "path": m11_6_gate_h_target_path,
                        "key": "parity_metrics.volume_rel_diff",
                        "value": m11_6_gate_h_target["parity_metrics"]["volume_rel_diff"],
                    },
                    {
                        "path": m11_6_m5_target_path,
                        "key": "parity_metrics.density_rel_diff",
                        "value": m11_6_m5_target["parity_metrics"]["density_rel_diff"],
                    },
                    {
                        "path": m11_6_m5_target_path,
                        "key": "parity_metrics.volume_rel_diff",
                        "value": m11_6_m5_target["parity_metrics"]["volume_rel_diff"],
                    },
                ],
            },
            {
                "id": "charged_semantics.legacy_m10_dense_box_density_volume_parity",
                "domain": "charged_semantics",
                "status": "unsupported",
                "claimable_statement": "The legacy M10 `dense_salt_polymer` density/volume parity path is not supportable.",
                "non_claimable_statement": "Do not cite the legacy M10 density/volume mismatch as a passing dense charged parity path; use only explicit M11 subset evidence for those subsets.",
                "evidence": [
                    {
                        "path": m10_4_path,
                        "key": "dense_salt_polymer.density_parity_rel_diff",
                        "value": m10_4_row["density_parity_rel_diff"],
                    },
                    {
                        "path": m10_4_path,
                        "key": "dense_salt_polymer.volume_parity_rel_diff",
                        "value": m10_4_row["volume_parity_rel_diff"],
                    },
                    {
                        "path": m10_4_path,
                        "key": "dense_salt_polymer.gmx.density.status",
                        "value": m10_4_row["gmx"]["density"]["status"],
                    },
                ],
            },
            {
                "id": "charged_semantics.m4_strict_charged_validation",
                "domain": "charged_semantics",
                "status": "exact",
                "claimable_statement": "M4 strict charged validation passes on the explicit `gate_h_dense_salt_polymer_2x2x2` strict-PCFF-qualified pair.",
                "non_claimable_statement": "This does not establish broad chemistry, LAMMPS-vs-GROMACS transport parity, charged transport readiness, or production readiness.",
                "evidence": [
                    {
                        "path": m4_inventory_path,
                        "key": "status",
                        "value": m4_inventory["status"],
                    },
                    {
                        "path": m4_inventory_path,
                        "key": "component_status",
                        "value": m4_inventory["component_status"],
                    },
                    {
                        "path": m4_mechanical_path,
                        "key": "metrics.energy_rel_diff",
                        "value": m4_mechanical["metrics"]["energy_rel_diff"],
                    },
                    {
                        "path": m4_mechanical_path,
                        "key": "metrics.force_rms_rel_diff",
                        "value": m4_mechanical["metrics"]["force_rms_rel_diff"],
                    },
                    {
                        "path": m4_structural_path,
                        "key": "parity_metrics.density_rel_diff",
                        "value": m4_structural["parity_metrics"]["density_rel_diff"],
                    },
                    {
                        "path": m4_transport_path,
                        "key": "fresh_m4_rerun",
                        "value": m4_transport["fresh_m4_rerun"],
                    },
                    {
                        "path": m4_transport_path,
                        "key": "primary_observables",
                        "value": m4_transport["primary_observables"],
                    },
                ],
            },
            {
                "id": "charged_semantics.long_horizon_stability",
                "domain": "charged_semantics",
                "status": "exact",
                "claimable_statement": "The corrected 5 ns TP1 dense_salt_polymer NPT rerun resolves the historical thermal-runaway blocker for that exact system/protocol.",
                "non_claimable_statement": "Do not promote this to dense parity, endpoint continuation safety, charged transport readiness, or production readiness.",
                "evidence": [
                    {
                        "path": tp1_path,
                        "key": "recovery_status.overall",
                        "value": tp1["recovery_status"]["overall"],
                    },
                    {
                        "path": tp1_path,
                        "key": "equilibration_duration_ns",
                        "value": tp1["equilibration_duration_ns"],
                    },
                    {
                        "path": tp1_path,
                        "key": "max_block_temperature_mean_K",
                        "value": tp1_max_block_temp,
                    },
                    {
                        "path": tp1_exact_path,
                        "key": "verdicts.thermal_runaway_exact_blocker",
                        "value": tp1_exact["verdicts"]["thermal_runaway_exact_blocker"],
                    },
                    {
                        "path": tp1_exact_path,
                        "key": "duration_completed_ps",
                        "value": tp1_exact["duration_completed_ps"],
                    },
                    {
                        "path": tp1_exact_path,
                        "key": "temperature_k.mean",
                        "value": tp1_exact["temperature_k"]["mean"],
                    },
                    {
                        "path": tp1_exact_path,
                        "key": "temperature_k.max",
                        "value": tp1_exact["temperature_k"]["max"],
                    },
                ],
            },
            {
                "id": "charged_semantics.endpoint_cutoff_margin_after_tp1_exact",
                "domain": "charged_semantics",
                "status": "unsupported",
                "claimable_statement": "Endpoint continuation safety from the corrected TP1 final coordinates is not supportable.",
                "non_claimable_statement": "Do not use the corrected TP1 endpoint as transport-entry or continuation-ready evidence until the cutoff/box margin is resolved.",
                "evidence": [
                    {
                        "path": tp1_exact_path,
                        "key": "verdicts.endpoint_cutoff_margin",
                        "value": tp1_exact["verdicts"]["endpoint_cutoff_margin"],
                    },
                    {
                        "path": tp1_exact_path,
                        "key": "final_box_nm",
                        "value": tp1_exact["final_box_nm"],
                    },
                    {
                        "path": tp1_exact_path,
                        "key": "half_box_margin_nm",
                        "value": tp1_exact["half_box_margin_nm"],
                    },
                ],
            },
            {
                "id": "charged_semantics.transport_observables",
                "domain": "charged_semantics",
                "status": "unsupported",
                "claimable_statement": "Charged diffusion, conductivity, and transference are not currently supportable as PCFF claims.",
                "non_claimable_statement": "Do not claim charged transport readiness, cNE readiness, or publication-grade transport validity.",
                "evidence": [
                    {
                        "path": m10_readiness_path,
                        "key": "overall_status",
                        "value": m10_readiness["overall_status"],
                    },
                    {
                        "path": m10_readiness_path,
                        "key": "recommended_use[-1]",
                        "value": m10_readiness["recommended_use"][-1],
                    },
                ],
            },
            {
                "id": "charged_semantics.provenance_qualified_strict_pcff_parity",
                "domain": "charged_semantics",
                "status": "exact",
                "claimable_statement": "One provenance-qualified strict PCFF charged validation path now survives for the explicit `gate_h_dense_salt_polymer_2x2x2` pair.",
                "non_claimable_statement": "ACPYPE/GAFF2-prepared charged artifacts remain disqualified; do not generalize this one strict PCFF pair to broad charged support.",
                "evidence": [
                    {
                        "path": m4_inventory_path,
                        "key": "status",
                        "value": m4_inventory["status"],
                    },
                    {
                        "path": m4_inventory_path,
                        "key": "component_status",
                        "value": m4_inventory["component_status"],
                    },
                    {
                        "path": m10_readiness_path,
                        "key": "strict_parity_readiness.status",
                        "value": m10_readiness["strict_parity_readiness"]["status"],
                    },
                    {
                        "path": m10_readiness_path,
                        "key": "pcff_provenance_gate_status_counts",
                        "value": m10_readiness["pcff_provenance_gate_status_counts"],
                    },
                ],
            },
        ],
    }

    summary = {
        "schema_name": "pcff_ion_narrow_claim_summary",
        "schema_version": 1,
        "claim_status_as_of": "2026-04-07",
        "public_facing_claim": public_claim,
        "chemistry_scope_statement": chemistry_scope_statement,
        "charged_semantic_scope_statement": charged_scope_statement,
        "supported_case_details": supported_case_details,
        "surviving_support_paths": [
            {
                "id": "pt8_supported_spe_typing_export",
                "status": "exact",
                "summary": "Three frozen glyme + Li/TFSI SPE cases pass deterministic typing/export validation.",
                "evidence_paths": [
                    pt8_validation_path,
                    pt8_cases_path,
                ],
            },
            {
                "id": "small_fixture_charged_mechanics",
                "status": "exact",
                "summary": "Frozen small charged fixtures preserve long-range charged semantics and combined energy/force parity.",
                "evidence_paths": [
                    smoke_path,
                    mixing_path,
                    nonbonded_path,
                    combined_path,
                ],
            },
            {
                "id": "m11_m4_strict_charged_subset_validation",
                "status": "exact",
                "summary": "The explicit strict-PCFF-qualified `gate_h_dense_salt_polymer_2x2x2` subset passes M4 separated mechanical, structural/density, and short-horizon transport-facing validation.",
                "evidence_paths": [
                    m4_inventory_path,
                    m4_mechanical_path,
                    m4_structural_path,
                    m4_transport_path,
                ],
            },
            {
                "id": "m11_4_broader_high_pressure_dense_parity",
                "status": "exact",
                "summary": "M11.4 broadens M2 dense charged parity to two strict-PCFF-qualified dense charged pairs at 250 bar over a 100 ps target / final 50 ps window.",
                "evidence_paths": [
                    m11_4_summary_path,
                    m11_4_gate_h_path,
                    m11_4_m5_dense_path,
                ],
            },
            {
                "id": "m11_6_pressure_preconditioned_staged_1bar_dense_parity",
                "status": "exact",
                "summary": "M11.6 closes pressure-preconditioned staged 1 bar dense charged parity for two strict-PCFF-qualified dense charged pairs over 100 ps precondition / 100 ps target / final 50 ps target window.",
                "evidence_paths": [
                    m11_6_summary_path,
                    m11_6_gate_h_target_path,
                    m11_6_m5_target_path,
                ],
            },
            {
                "id": "m5_acyclic_alkane_neutral_additive_charged_assembly",
                "status": "exact",
                "summary": "M5 adds one validated charged assembly containing an acyclic alkane neutral additive: monoglyme + ethane + Li/TFSI.",
                "evidence_paths": [
                    m5_report_path,
                    m5_manifest_path,
                ],
            },
            {
                "id": "tp1_corrected_5ns_thermal_stability",
                "status": "exact",
                "summary": "The authoritative dense_salt_polymer TP1 thermal-runaway blocker is superseded by a corrected 5 ns NPT rerun with tcoupl/pcoupl/gen-vel applied.",
                "evidence_paths": [
                    tp1_exact_path,
                ],
            },
        ],
        "blocked_paths": [
            {
                "id": "broad_pcff_chemistry",
                "status": "unsupported",
                "summary": "The broader CSV-snapshot chemistry target remains uncovered.",
                "evidence_paths": [csv_scope_path],
            },
            {
                "id": "legacy_m10_dense_charged_ensemble_parity",
                "status": "unsupported",
                "summary": f"The legacy M10 dense charged-box path still fails density parity by about {density_rel_percent}% and volume parity by about {volume_rel_percent}%; only explicit M11 subsets have passing density/volume evidence.",
                "evidence_paths": [m10_4_path],
            },
            {
                "id": "ambient_1bar_broader_dense_parity",
                "status": "unsupported",
                "summary": f"Ambient 1 bar equilibrium dense charged parity remains unsupported; M11.6 only supports pressure-preconditioned staged 1 bar parity and M11.5 still marks direct ambient 1 bar behavior as {m11_5_root_cause['interpretation']['ambient_1bar_broader_m2_status'].lower()}.",
                "evidence_paths": [m11_4_summary_path, m11_5_root_cause_path, m11_6_summary_path],
            },
            {
                "id": "tp1_corrected_endpoint_continuation_safety",
                "status": "unsupported",
                "summary": "The corrected TP1 5 ns endpoint is not continuation/transport-entry safe as-is because the final box is smaller than twice the 0.9 nm cutoff.",
                "evidence_paths": [tp1_exact_path],
            },
            {
                "id": "charged_transport_readiness",
                "status": "unsupported",
                "summary": "M4 adds a short-horizon transport-facing CPU/GPU observable check, but it is not LAMMPS-vs-GROMACS transport parity or charged transport readiness.",
                "evidence_paths": [m10_readiness_path, m4_transport_path],
            },
        ],
        "support_matrix_path": repo_rel(OUT_ROOT / "support_matrix.json"),
    }

    write_json(OUT_ROOT / "support_matrix.json", support_matrix)
    write_json(OUT_ROOT / "narrow_claim_summary.json", summary)


if __name__ == "__main__":
    main()
