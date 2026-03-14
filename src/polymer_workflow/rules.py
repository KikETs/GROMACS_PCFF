from __future__ import annotations

import copy

from atom_typing.rules import load_rules as load_typing_rules
from nonbonded_assignment.rules import load_rules as load_nonbonded_rules
from parameter_assignment.rules import load_rules as load_parameter_rules


POLYETHER_COMPONENT_FAMILY = "acyclic_polyether_oligomer"
POLYETHER_BACKBONE_CARBON = "polyether_backbone_methylene_sp3"
POLYETHER_BACKBONE_HYDROGEN = "hydrogen_on_polyether_backbone_methylene"

PROVENANCE = {
    "note": (
        "PT7 polymer_workflow specialization for linear methoxy-capped polyether oligomers. "
        "Coefficients are repository-local frozen extensions derived deterministically from the "
        "existing ether and alkane regression families."
    ),
    "source_file": "src/polymer_workflow/rules.py",
    "source_kind": "repository_local_frozen_polymer_workflow_rule",
}


def build_typing_ruleset() -> dict:
    ruleset = load_typing_rules()
    ruleset = copy.deepcopy(ruleset)
    ruleset["ruleset_id"] = f"{ruleset['ruleset_id']}_polymer_workflow_v1"

    support_rule = {
        "family": POLYETHER_COMPONENT_FAMILY,
        "kind": "support",
        "precedence": 115,
        "predicate": {
            "builtin": "component_is_acyclic_polyether_oligomer",
        },
        "rule_id": "classify_acyclic_polyether_oligomer",
    }
    fallback_index = next(
        index
        for index, rule in enumerate(ruleset["component_rules"])
        if rule["rule_id"] == "reject_unsupported_component_family"
    )
    ruleset["component_rules"].insert(fallback_index, support_rule)

    ruleset["atom_type_rules"].extend(
        [
            {
                "component_family": POLYETHER_COMPONENT_FAMILY,
                "family": "ether_alpha_carbon_sp3",
                "match": {
                    "conditions": [
                        {"equals": "C", "path": "element"},
                        {"equals": 0, "path": "formal_charge"},
                        {"equals": 4, "path": "valence.inferred_valence"},
                        {"equals": {"H": 3, "O": 1}, "path": "neighbor_element_counts"},
                        {"equals": 4, "path": "coordination.coordination_number"},
                    ]
                },
                "precedence": 100,
                "rule_id": "atom_polyether_end_carbon_sp3",
            },
            {
                "component_family": POLYETHER_COMPONENT_FAMILY,
                "family": "ether_oxygen_sp3",
                "match": {
                    "conditions": [
                        {"equals": "O", "path": "element"},
                        {"equals": 0, "path": "formal_charge"},
                        {"equals": 2, "path": "valence.inferred_valence"},
                        {"equals": 2, "path": "coordination.coordination_number"},
                        {"equals": {"C": 2}, "path": "neighbor_element_counts"},
                    ]
                },
                "precedence": 110,
                "rule_id": "atom_polyether_oxygen_sp3",
            },
            {
                "component_family": POLYETHER_COMPONENT_FAMILY,
                "family": POLYETHER_BACKBONE_CARBON,
                "match": {
                    "conditions": [
                        {"equals": "C", "path": "element"},
                        {"equals": 0, "path": "formal_charge"},
                        {"equals": 4, "path": "valence.inferred_valence"},
                        {"equals": {"C": 1, "H": 2, "O": 1}, "path": "neighbor_element_counts"},
                        {"equals": 4, "path": "coordination.coordination_number"},
                    ]
                },
                "precedence": 120,
                "rule_id": "atom_polyether_backbone_methylene_sp3",
            },
            {
                "component_family": POLYETHER_COMPONENT_FAMILY,
                "family": "hydrogen_on_ether_alpha_carbon",
                "match": {
                    "conditions": [
                        {"equals": "H", "path": "element"},
                        {"equals": "C", "path": "attached_atom.element"},
                        {"equals": {"H": 3, "O": 1}, "path": "attached_atom.neighbor_element_counts"},
                        {"equals": 4, "path": "attached_atom.valence.inferred_valence"},
                    ]
                },
                "precedence": 130,
                "rule_id": "atom_hydrogen_on_polyether_end_carbon",
            },
            {
                "component_family": POLYETHER_COMPONENT_FAMILY,
                "family": POLYETHER_BACKBONE_HYDROGEN,
                "match": {
                    "conditions": [
                        {"equals": "H", "path": "element"},
                        {"equals": "C", "path": "attached_atom.element"},
                        {"equals": {"C": 1, "H": 2, "O": 1}, "path": "attached_atom.neighbor_element_counts"},
                        {"equals": 4, "path": "attached_atom.valence.inferred_valence"},
                    ]
                },
                "precedence": 140,
                "rule_id": "atom_hydrogen_on_polyether_backbone_methylene",
            },
        ]
    )
    return ruleset


def build_parameter_ruleset() -> dict:
    ruleset = load_parameter_rules()
    ruleset = copy.deepcopy(ruleset)
    ruleset["ruleset_id"] = f"{ruleset['ruleset_id']}_polymer_workflow_v1"

    interaction_rules = ruleset["interaction_rules"]
    interaction_rules["bond"].extend(
        [
            _bond_rule(
                "param_polyether_backbone_c_o",
                f"bond(ether_oxygen_sp3|{POLYETHER_BACKBONE_CARBON})",
                r0_angstrom=1.43,
                k2=320.0,
                k3=-42.0,
                k4=5.5,
            ),
            _bond_rule(
                "param_polyether_backbone_c_h",
                f"bond({POLYETHER_BACKBONE_HYDROGEN}|{POLYETHER_BACKBONE_CARBON})",
                r0_angstrom=1.101,
                k2=338.0,
                k3=-44.0,
                k4=6.2,
            ),
            _bond_rule(
                "param_polyether_backbone_c_c",
                f"bond({POLYETHER_BACKBONE_CARBON}|{POLYETHER_BACKBONE_CARBON})",
                r0_angstrom=1.53,
                k2=250.0,
                k3=-35.0,
                k4=8.0,
            ),
        ]
    )
    interaction_rules["angle"].extend(
        [
            _angle_rule(
                "param_polyether_end_o_backbone",
                f"angle(ether_alpha_carbon_sp3|ether_oxygen_sp3|{POLYETHER_BACKBONE_CARBON})",
                theta0_deg=112.1,
                k2=41.0,
                k3=-5.3,
                k4=1.2,
                bb_k=5.8,
                r1=1.43,
                r2=1.43,
                ba_k1=1.75,
                ba_k2=1.75,
            ),
            _angle_rule(
                "param_polyether_backbone_o_backbone",
                f"angle({POLYETHER_BACKBONE_CARBON}|ether_oxygen_sp3|{POLYETHER_BACKBONE_CARBON})",
                theta0_deg=112.1,
                k2=41.0,
                k3=-5.3,
                k4=1.2,
                bb_k=5.8,
                r1=1.43,
                r2=1.43,
                ba_k1=1.75,
                ba_k2=1.75,
            ),
            _angle_rule(
                "param_polyether_o_backbone_h",
                f"angle(ether_oxygen_sp3|{POLYETHER_BACKBONE_CARBON}|{POLYETHER_BACKBONE_HYDROGEN})",
                theta0_deg=109.4,
                k2=37.1,
                k3=-4.1,
                k4=0.9,
                bb_k=4.9,
                r1=1.43,
                r2=1.101,
                ba_k1=1.42,
                ba_k2=1.28,
            ),
            _angle_rule(
                "param_polyether_h_backbone_h",
                f"angle({POLYETHER_BACKBONE_HYDROGEN}|{POLYETHER_BACKBONE_CARBON}|{POLYETHER_BACKBONE_HYDROGEN})",
                theta0_deg=107.6,
                k2=35.0,
                k3=-4.4,
                k4=1.0,
                bb_k=5.0,
                r1=1.101,
                r2=1.101,
                ba_k1=1.08,
                ba_k2=1.08,
            ),
            _angle_rule(
                "param_polyether_o_backbone_backbone",
                f"angle(ether_oxygen_sp3|{POLYETHER_BACKBONE_CARBON}|{POLYETHER_BACKBONE_CARBON})",
                theta0_deg=110.4,
                k2=36.0,
                k3=-4.2,
                k4=0.95,
                bb_k=4.75,
                r1=1.43,
                r2=1.53,
                ba_k1=1.42,
                ba_k2=1.35,
            ),
            _angle_rule(
                "param_polyether_h_backbone_backbone",
                f"angle({POLYETHER_BACKBONE_HYDROGEN}|{POLYETHER_BACKBONE_CARBON}|{POLYETHER_BACKBONE_CARBON})",
                theta0_deg=110.7,
                k2=34.5,
                k3=-4.2,
                k4=0.9,
                bb_k=4.6,
                r1=1.101,
                r2=1.53,
                ba_k1=1.1,
                ba_k2=1.35,
            ),
        ]
    )
    interaction_rules["dihedral"].extend(
        [
            _dihedral_rule(
                "param_polyether_h_end_c_o_backbone",
                f"dihedral(hydrogen_on_ether_alpha_carbon|ether_alpha_carbon_sp3|ether_oxygen_sp3|{POLYETHER_BACKBONE_CARBON})",
                theta0_1=112.1,
                theta0_2=109.4,
                r0_1=1.43,
                r0_2=1.101,
            ),
            _dihedral_rule(
                "param_polyether_end_o_backbone_h",
                f"dihedral(ether_alpha_carbon_sp3|ether_oxygen_sp3|{POLYETHER_BACKBONE_CARBON}|{POLYETHER_BACKBONE_HYDROGEN})",
                theta0_1=112.1,
                theta0_2=109.4,
                r0_1=1.43,
                r0_2=1.101,
            ),
            _dihedral_rule(
                "param_polyether_end_o_backbone_backbone",
                f"dihedral(ether_alpha_carbon_sp3|ether_oxygen_sp3|{POLYETHER_BACKBONE_CARBON}|{POLYETHER_BACKBONE_CARBON})",
                theta0_1=112.1,
                theta0_2=110.4,
                r0_1=1.43,
                r0_2=1.53,
            ),
            _dihedral_rule(
                "param_polyether_h_backbone_backbone_o",
                f"dihedral({POLYETHER_BACKBONE_HYDROGEN}|{POLYETHER_BACKBONE_CARBON}|ether_oxygen_sp3|{POLYETHER_BACKBONE_CARBON})",
                theta0_1=109.4,
                theta0_2=109.4,
                r0_1=1.101,
                r0_2=1.101,
            ),
            _dihedral_rule(
                "param_polyether_o_backbone_backbone_h",
                f"dihedral(ether_oxygen_sp3|{POLYETHER_BACKBONE_CARBON}|{POLYETHER_BACKBONE_CARBON}|{POLYETHER_BACKBONE_HYDROGEN})",
                theta0_1=110.4,
                theta0_2=109.4,
                r0_1=1.53,
                r0_2=1.101,
            ),
            _dihedral_rule(
                "param_polyether_o_backbone_backbone_o",
                f"dihedral(ether_oxygen_sp3|{POLYETHER_BACKBONE_CARBON}|{POLYETHER_BACKBONE_CARBON}|ether_oxygen_sp3)",
                theta0_1=110.4,
                theta0_2=110.4,
                r0_1=1.53,
                r0_2=1.53,
            ),
            _dihedral_rule(
                "param_polyether_backbone_o_backbone_backbone",
                f"dihedral({POLYETHER_BACKBONE_CARBON}|ether_oxygen_sp3|{POLYETHER_BACKBONE_CARBON}|{POLYETHER_BACKBONE_CARBON})",
                theta0_1=112.1,
                theta0_2=110.4,
                r0_1=1.43,
                r0_2=1.53,
            ),
            _dihedral_rule(
                "param_polyether_h_backbone_backbone_h",
                f"dihedral({POLYETHER_BACKBONE_HYDROGEN}|{POLYETHER_BACKBONE_CARBON}|{POLYETHER_BACKBONE_CARBON}|{POLYETHER_BACKBONE_HYDROGEN})",
                theta0_1=110.7,
                theta0_2=110.7,
                r0_1=1.101,
                r0_2=1.101,
                main_k1=0.85,
                main_k2=0.6,
                main_k3=0.35,
                mbt_f1=0.11,
                mbt_f2=-0.07,
                mbt_f3=0.03,
                ebt_f1_1=0.09,
                ebt_f2_1=-0.045,
                ebt_f3_1=0.02,
                ebt_f1_2=0.09,
                ebt_f2_2=-0.045,
                ebt_f3_2=0.02,
                at_f1_1=0.05,
                at_f2_1=-0.025,
                at_f3_1=0.012,
                at_f1_2=0.05,
                at_f2_2=-0.025,
                at_f3_2=0.012,
                aat_k=0.2,
                bb13_k=0.12,
                bb13_r1=1.101,
                bb13_r3=1.101,
            ),
        ]
    )
    return ruleset


def build_nonbonded_ruleset() -> dict:
    ruleset = load_nonbonded_rules()
    ruleset = copy.deepcopy(ruleset)
    ruleset["ruleset_id"] = f"{ruleset['ruleset_id']}_polymer_workflow_v1"
    ruleset["atom_type_rules"].extend(
        [
            {
                "family": POLYETHER_BACKBONE_CARBON,
                "nonbonded_type": "nb_polyether_c_backbone",
                "provenance": copy.deepcopy(PROVENANCE),
                "rule_id": "nb_polyether_backbone_methylene_sp3",
                "self_parameters": {
                    "epsilon_kcal_mol": 0.08,
                    "sigma_angstrom": 3.4,
                },
            },
            {
                "family": POLYETHER_BACKBONE_HYDROGEN,
                "nonbonded_type": "nb_h_polyether_backbone",
                "provenance": copy.deepcopy(PROVENANCE),
                "rule_id": "nb_hydrogen_on_polyether_backbone_methylene",
                "self_parameters": {
                    "epsilon_kcal_mol": 0.018,
                    "sigma_angstrom": 2.45,
                },
            },
        ]
    )
    return ruleset


def _bond_rule(rule_id: str, signature: str, *, r0_angstrom: float, k2: float, k3: float, k4: float) -> dict:
    return {
        "rule_id": rule_id,
        "canonical_signature": signature,
        "parameters": {
            "main": {
                "r0_angstrom": r0_angstrom,
                "k2_kcal_mol_per_a2": k2,
                "k3_kcal_mol_per_a3": k3,
                "k4_kcal_mol_per_a4": k4,
            }
        },
        "provenance": copy.deepcopy(PROVENANCE),
    }


def _angle_rule(
    rule_id: str,
    signature: str,
    *,
    theta0_deg: float,
    k2: float,
    k3: float,
    k4: float,
    bb_k: float,
    r1: float,
    r2: float,
    ba_k1: float,
    ba_k2: float,
) -> dict:
    return {
        "rule_id": rule_id,
        "canonical_signature": signature,
        "parameters": {
            "main": {
                "theta0_deg": theta0_deg,
                "k2_kcal_mol": k2,
                "k3_kcal_mol": k3,
                "k4_kcal_mol": k4,
            },
            "bb": {
                "k_kcal_mol_per_a2": bb_k,
                "r1_angstrom": r1,
                "r2_angstrom": r2,
            },
            "ba": {
                "k1_kcal_mol_per_a": ba_k1,
                "k2_kcal_mol_per_a": ba_k2,
                "r1_angstrom": r1,
                "r2_angstrom": r2,
            },
        },
        "provenance": copy.deepcopy(PROVENANCE),
    }


def _dihedral_rule(
    rule_id: str,
    signature: str,
    *,
    theta0_1: float,
    theta0_2: float,
    r0_1: float,
    r0_2: float,
    main_k1: float = 0.72,
    main_k2: float = 0.45,
    main_k3: float = 0.18,
    mbt_f1: float = 0.1,
    mbt_f2: float = -0.06,
    mbt_f3: float = 0.03,
    ebt_f1_1: float = 0.08,
    ebt_f2_1: float = -0.04,
    ebt_f3_1: float = 0.02,
    ebt_f1_2: float = 0.07,
    ebt_f2_2: float = -0.03,
    ebt_f3_2: float = 0.015,
    at_f1_1: float = 0.04,
    at_f2_1: float = -0.02,
    at_f3_1: float = 0.01,
    at_f1_2: float = 0.03,
    at_f2_2: float = -0.015,
    at_f3_2: float = 0.008,
    aat_k: float = 0.17,
    bb13_k: float = 0.11,
    bb13_r1: float | None = None,
    bb13_r3: float | None = None,
) -> dict:
    return {
        "rule_id": rule_id,
        "canonical_signature": signature,
        "parameters": {
            "main": {
                "k1_kcal_mol": main_k1,
                "phi1_deg": 0.0,
                "k2_kcal_mol": main_k2,
                "phi2_deg": 180.0,
                "k3_kcal_mol": main_k3,
                "phi3_deg": 0.0,
            },
            "mbt": {
                "f1_kcal_mol_per_a": mbt_f1,
                "f2_kcal_mol_per_a": mbt_f2,
                "f3_kcal_mol_per_a": mbt_f3,
                "r0_angstrom": r0_1,
            },
            "ebt": {
                "f1_1_kcal_mol_per_a": ebt_f1_1,
                "f2_1_kcal_mol_per_a": ebt_f2_1,
                "f3_1_kcal_mol_per_a": ebt_f3_1,
                "f1_2_kcal_mol_per_a": ebt_f1_2,
                "f2_2_kcal_mol_per_a": ebt_f2_2,
                "f3_2_kcal_mol_per_a": ebt_f3_2,
                "r0_1_angstrom": r0_1,
                "r0_2_angstrom": r0_2,
            },
            "at": {
                "f1_1_kcal_mol": at_f1_1,
                "f2_1_kcal_mol": at_f2_1,
                "f3_1_kcal_mol": at_f3_1,
                "f1_2_kcal_mol": at_f1_2,
                "f2_2_kcal_mol": at_f2_2,
                "f3_2_kcal_mol": at_f3_2,
                "theta0_1_deg": theta0_1,
                "theta0_2_deg": theta0_2,
            },
            "aat": {
                "k_kcal_mol": aat_k,
                "theta0_1_deg": theta0_1,
                "theta0_2_deg": theta0_2,
            },
            "bb13": {
                "k_kcal_mol_per_a2": bb13_k,
                "r1_angstrom": r0_1 if bb13_r1 is None else bb13_r1,
                "r3_angstrom": r0_2 if bb13_r3 is None else bb13_r3,
            },
        },
        "provenance": copy.deepcopy(PROVENANCE),
    }
