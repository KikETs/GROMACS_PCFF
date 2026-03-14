from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from chem_perception import dumps_report as dumps_perception_report
from chem_perception import perceive_ir, validate_report as validate_perception_report
from typing_ir import dumps_ir, parse_file, validate_ir

from .errors import TypingReportError
from .rules import DEFAULT_RULES_PATH, load_rules, validate_rules


SCHEMA_NAME = "pcff_atom_typing_report"
SCHEMA_VERSION = 1


def type_file(
    path: str | Path,
    *,
    input_format: str | None = None,
    source_id: str | None = None,
    rules_path: str | Path | None = None,
    ruleset: dict | None = None,
) -> dict:
    ir = parse_file(path, input_format=input_format, source_id=source_id)
    return type_ir(ir, rules_path=rules_path, ruleset=ruleset)


def type_ir(
    ir: dict,
    *,
    perception: dict | None = None,
    rules_path: str | Path | None = None,
    ruleset: dict | None = None,
) -> dict:
    validate_ir(ir)
    perception_report = perceive_ir(ir) if perception is None else perception
    validate_perception_report(perception_report)
    active_ruleset = load_rules(rules_path) if ruleset is None else copy.deepcopy(ruleset)
    validate_rules(active_ruleset)

    ir_component = ir["components"][0]
    perception_component = perception_report["components"][0]
    atom_contexts = _build_atom_contexts(ir_component, perception_component)
    component_view = _build_component_view(ir_component, perception_component, atom_contexts)
    classification = _classify_component(component_view, active_ruleset)

    atom_records = []
    explanations = []
    diagnostics = []
    overall_status = "typed"

    if classification["status"] != "supported":
        overall_status = classification["status"]
        diagnostics.extend(classification["diagnostics"])
        atom_records = _skipped_atom_records(atom_contexts, overall_status)
    else:
        component_family = classification["family"]
        atom_records, explanations, diagnostics = _assign_atom_types(
            atom_contexts,
            component_family=component_family,
            component_rule_id=classification["rule_id"],
            ruleset=active_ruleset,
        )
        if any(record["status"] == "ambiguous" for record in atom_records):
            overall_status = "ambiguous"
        elif any(record["status"] == "unresolved" for record in atom_records):
            overall_status = "unresolved"

    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "typed_ir_sha256": hashlib.sha256(dumps_ir(ir).encode("utf-8")).hexdigest(),
            "chem_perception_sha256": hashlib.sha256(
                dumps_perception_report(perception_report).encode("utf-8")
            ).hexdigest(),
            "rules_sha256": hashlib.sha256(
                json.dumps(active_ruleset, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "source_id": ir["source"]["source_id"],
            "input_format": ir["source"]["input_format"],
            "ruleset_id": active_ruleset["ruleset_id"],
            "rules_path": _display_rules_path(DEFAULT_RULES_PATH if rules_path is None else Path(rules_path)),
        },
        "typing": {
            "status": overall_status,
            "ruleset_id": active_ruleset["ruleset_id"],
        },
        "components": [
            {
                "component_id": ir_component["component_id"],
                "name": ir_component["name"],
                "atom_count": ir_component["atom_count"],
                "classification": classification,
                "atoms": atom_records,
                "atom_type_explanations": explanations,
                "diagnostics": diagnostics,
            }
        ],
    }
    validate_typing_report(report)
    return report


def dumps_typing_report(report: dict) -> str:
    validate_typing_report(report)
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def loads_typing_report(text: str) -> dict:
    report = json.loads(text)
    validate_typing_report(report)
    return report


def write_typing_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_typing_report(report), encoding="utf-8")


def validate_typing_report(report: dict) -> None:
    if report.get("schema_name") != SCHEMA_NAME:
        raise TypingReportError("invalid_typing_report", "schema_name must be 'pcff_atom_typing_report'")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise TypingReportError("invalid_typing_report", "Unsupported schema_version")
    source = report.get("source")
    if not isinstance(source, dict):
        raise TypingReportError("invalid_typing_report", "source must be a mapping")
    for key in {
        "typed_ir_sha256",
        "chem_perception_sha256",
        "rules_sha256",
        "source_id",
        "input_format",
        "ruleset_id",
        "rules_path",
    }:
        if key not in source:
            raise TypingReportError("invalid_typing_report", f"source.{key} is required")
    components = report.get("components")
    if not isinstance(components, list) or len(components) != 1:
        raise TypingReportError("invalid_typing_report", "typing report must contain exactly one component")


def _skipped_atom_records(atom_contexts: list[dict], status: str) -> list[dict]:
    return [
        {
            "canonical_index": context["canonical_index"],
            "source_index": context["source_index"],
            "source_atom_id": context["source_atom_id"],
            "element": context["element"],
            "status": f"skipped_{status}",
            "assigned_family": None,
            "matched_rule_id": None,
            "precedence": None,
        }
        for context in atom_contexts
    ]


def _build_atom_contexts(ir_component: dict, perception_component: dict) -> list[dict]:
    ir_atoms = {atom["canonical_index"]: atom for atom in ir_component["atoms"]}
    ir_bonds = {bond["canonical_index"]: bond for bond in ir_component["bonds"]}
    perception_atoms = {atom["canonical_index"]: atom for atom in perception_component["atoms"]}
    pair_to_bond_index = {
        tuple(sorted(bond["atom_indices"])): bond["canonical_index"]
        for bond in ir_component["bonds"]
    }

    contexts = []
    for canonical_index in sorted(perception_atoms):
        perception_atom = perception_atoms[canonical_index]
        ir_atom = ir_atoms[canonical_index]
        neighbor_indices = perception_atom["neighbor_indices"]
        attached_atom = None
        attached_bond = None
        if len(neighbor_indices) == 1:
            attached_index = neighbor_indices[0]
            attached_atom = perception_atoms[attached_index]
            bond_index = pair_to_bond_index[tuple(sorted((canonical_index, attached_index)))]
            attached_bond = ir_bonds[bond_index]

        contexts.append(
            {
                "canonical_index": canonical_index,
                "source_index": ir_atom["source_index"],
                "source_atom_id": ir_atom["source_atom_id"],
                "neighbor_source_indices": [ir_atoms[index]["source_index"] for index in neighbor_indices],
                "element": perception_atom["element"],
                "formal_charge": ir_atom["formal_charge"],
                "neighbor_indices": neighbor_indices,
                "neighbor_element_counts": perception_atom["neighbor_element_counts"],
                "valence": perception_atom["valence"],
                "ring": perception_atom["ring"],
                "aromaticity": perception_atom["aromaticity"],
                "coordination": perception_atom["coordination"],
                "improper_center_candidate": perception_atom["improper_center_candidate"],
                "polymer_connection": perception_atom["polymer_connection"],
                "attached_atom": (
                    {
                        "canonical_index": attached_atom["canonical_index"],
                        "source_index": ir_atoms[attached_index]["source_index"],
                        "source_atom_id": ir_atoms[attached_index]["source_atom_id"],
                        "element": attached_atom["element"],
                        "formal_charge": ir_atoms[attached_index]["formal_charge"],
                        "neighbor_element_counts": attached_atom["neighbor_element_counts"],
                        "valence": attached_atom["valence"],
                        "aromaticity": attached_atom["aromaticity"],
                        "coordination": attached_atom["coordination"],
                    }
                    if attached_atom is not None
                    else None
                ),
                "attached_bond": (
                    {
                        "canonical_index": attached_bond["canonical_index"],
                        "order": attached_bond["order"],
                        "bond_code": attached_bond["bond_code"],
                    }
                    if attached_bond is not None
                    else None
                ),
            }
        )
    return contexts


def _build_component_view(ir_component: dict, perception_component: dict, atom_contexts: list[dict]) -> dict:
    element_counts = dict(sorted(ir_component["element_counts"].items()))
    aromatic_ring_ids = [
        ring["ring_id"]
        for ring in perception_component["rings"]
        if ring["aromaticity"]["status"] == "aromatic"
    ]
    indeterminate_ring_ids = [
        ring["ring_id"]
        for ring in perception_component["rings"]
        if ring["aromaticity"]["status"] == "indeterminate"
    ]
    return {
        "name": ir_component["name"],
        "atom_count": ir_component["atom_count"],
        "bond_count": ir_component["bond_count"],
        "net_formal_charge": ir_component["net_formal_charge"],
        "element_counts": element_counts,
        "rings": perception_component["rings"],
        "bonds": ir_component["bonds"],
        "atoms": atom_contexts,
        "aromatic_ring_ids": aromatic_ring_ids,
        "indeterminate_ring_ids": indeterminate_ring_ids,
    }


def _classify_component(component_view: dict, ruleset: dict) -> dict:
    matches = []
    for order, rule in enumerate(ruleset["component_rules"]):
        matched, evidence = _evaluate_component_rule(component_view, rule, supported_elements=ruleset["supported_elements"])
        if matched:
            matches.append(
                {
                    "rule": rule,
                    "order": order,
                    "evidence": evidence,
                }
            )

    if not matches:
        raise TypingReportError("component_classification_failed", "No component rule matched and no fallback rule is present")

    best_precedence = min(match["rule"]["precedence"] for match in matches)
    best_matches = [match for match in matches if match["rule"]["precedence"] == best_precedence]
    if len(best_matches) > 1:
        candidate_rule_ids = [match["rule"]["rule_id"] for match in best_matches]
        diagnostic = {
            "scope": "component",
            "code": "ambiguous_component_classification",
            "message": f"Multiple component rules matched at precedence {best_precedence}: {', '.join(candidate_rule_ids)}",
            "candidate_rule_ids": candidate_rule_ids,
        }
        return {
            "status": "ambiguous",
            "family": None,
            "rule_id": None,
            "precedence": best_precedence,
            "diagnostics": [diagnostic],
        }

    winner = best_matches[0]
    rule = winner["rule"]
    if rule["kind"] == "reject":
        message = _reject_message(rule["failure_code"], winner["evidence"]["summary"])
        diagnostic = {
            "scope": "component",
            "code": rule["failure_code"],
            "message": message,
            "rule_id": rule["rule_id"],
            "evidence": winner["evidence"],
        }
        return {
            "status": "unsupported",
            "family": None,
            "rule_id": rule["rule_id"],
            "precedence": rule["precedence"],
            "failure_code": rule["failure_code"],
            "diagnostics": [diagnostic],
            "explanation": winner["evidence"],
        }

    return {
        "status": "supported",
        "family": rule["family"],
        "rule_id": rule["rule_id"],
        "precedence": rule["precedence"],
        "diagnostics": [],
        "explanation": winner["evidence"],
    }


def _evaluate_component_rule(component_view: dict, rule: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    predicate_name = rule["predicate"]["builtin"]
    predicate = COMPONENT_PREDICATES[predicate_name]
    return predicate(component_view, supported_elements=supported_elements)


def _assign_atom_types(
    atom_contexts: list[dict],
    *,
    component_family: str,
    component_rule_id: str,
    ruleset: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    atom_records = []
    explanations = []
    diagnostics = []
    rules = [rule for rule in ruleset["atom_type_rules"] if rule["component_family"] == component_family]

    for context in atom_contexts:
        matches = []
        enriched_context = dict(context)
        enriched_context["component_family"] = component_family
        for order, rule in enumerate(rules):
            matched, evidence = _evaluate_atom_rule(enriched_context, rule)
            if matched:
                matches.append(
                    {
                        "rule": rule,
                        "order": order,
                        "evidence": evidence,
                    }
                )

        if not matches:
            atom_records.append(
                {
                    "canonical_index": context["canonical_index"],
                    "source_index": context["source_index"],
                    "source_atom_id": context["source_atom_id"],
                    "element": context["element"],
                    "status": "unresolved",
                    "assigned_family": None,
                    "matched_rule_id": None,
                    "precedence": None,
                }
            )
            diagnostics.append(
                {
                    "scope": "atom",
                    "atom_index": context["canonical_index"],
                    "source_index": context["source_index"],
                    "code": "unresolved_atom_type",
                    "message": f"No atom typing rule matched atom {context['canonical_index']} ({context['element']}) in component family {component_family}",
                }
            )
            continue

        best_precedence = min(match["rule"]["precedence"] for match in matches)
        best_matches = [match for match in matches if match["rule"]["precedence"] == best_precedence]
        if len(best_matches) > 1:
            candidate_rule_ids = [match["rule"]["rule_id"] for match in best_matches]
            atom_records.append(
                {
                    "canonical_index": context["canonical_index"],
                    "source_index": context["source_index"],
                    "source_atom_id": context["source_atom_id"],
                    "element": context["element"],
                    "status": "ambiguous",
                    "assigned_family": None,
                    "matched_rule_id": None,
                    "precedence": best_precedence,
                }
            )
            diagnostics.append(
                {
                    "scope": "atom",
                    "atom_index": context["canonical_index"],
                    "source_index": context["source_index"],
                    "code": "ambiguous_atom_type_match",
                    "message": f"Multiple atom typing rules matched atom {context['canonical_index']} ({context['element']}) at precedence {best_precedence}: {', '.join(candidate_rule_ids)}",
                    "candidate_rule_ids": candidate_rule_ids,
                }
            )
            continue

        winner = best_matches[0]
        rule = winner["rule"]
        explanation = _build_explanation_record(
            context,
            component_family=component_family,
            component_rule_id=component_rule_id,
            rule=rule,
            evidence=winner["evidence"],
        )
        explanations.append(explanation)
        atom_records.append(
            {
                "canonical_index": context["canonical_index"],
                "source_index": context["source_index"],
                "source_atom_id": context["source_atom_id"],
                "element": context["element"],
                "status": "assigned",
                "assigned_family": rule["family"],
                "matched_rule_id": rule["rule_id"],
                "precedence": rule["precedence"],
                "explanation_id": explanation["explanation_id"],
            }
        )

    return atom_records, explanations, diagnostics


def _evaluate_atom_rule(atom_context: dict, rule: dict) -> tuple[bool, dict]:
    matched_conditions = []
    for condition in rule["match"]["conditions"]:
        actual = _resolve_path(atom_context, condition["path"])
        operator, expected = _condition_operator(condition)
        if not _condition_matches(actual, operator, expected):
            return False, {}
        matched_conditions.append(
            {
                "path": condition["path"],
                "operator": operator,
                "expected": expected,
                "actual": actual,
            }
        )
    return True, {"matched_conditions": matched_conditions}


def _build_explanation_record(
    atom_context: dict,
    *,
    component_family: str,
    component_rule_id: str,
    rule: dict,
    evidence: dict,
) -> dict:
    canonical_source_pairs = [(atom_context["canonical_index"], atom_context["source_index"])]
    for neighbor_index in atom_context["neighbor_indices"]:
        canonical_source_pairs.append((neighbor_index, None))
    return {
        "explanation_id": f"atom_{atom_context['canonical_index']}",
        "atom_index": atom_context["canonical_index"],
        "source_index": atom_context["source_index"],
        "source_atom_id": atom_context["source_atom_id"],
        "element": atom_context["element"],
        "assigned_family": rule["family"],
        "rule_id": rule["rule_id"],
        "precedence": rule["precedence"],
        "component_family": component_family,
        "component_rule_id": component_rule_id,
        "canonical_atom_indices": sorted({atom_context["canonical_index"], *atom_context["neighbor_indices"]}),
        "source_atom_indices": sorted({atom_context["source_index"], *atom_context["neighbor_source_indices"]}),
        "evidence": evidence,
    }


def _resolve_path(payload: dict | None, path: str):
    current = payload
    for token in path.split("."):
        if current is None:
            return None
        if not isinstance(current, dict) or token not in current:
            return None
        current = current[token]
    return current


def _condition_operator(condition: dict) -> tuple[str, object]:
    for operator in ("equals", "in", "contains", "contains_all"):
        if operator in condition:
            return operator, condition[operator]
    raise TypingReportError("invalid_rule_condition", f"Unsupported rule condition {condition}")


def _condition_matches(actual, operator: str, expected) -> bool:
    if operator == "equals":
        return actual == expected
    if operator == "in":
        return actual in expected if actual is not None else False
    if operator == "contains":
        return actual is not None and expected in actual
    if operator == "contains_all":
        return actual is not None and all(item in actual for item in expected)
    raise TypingReportError("invalid_rule_condition", f"Unsupported operator {operator!r}")


def _component_has_unsupported_element(component_view: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    unsupported = sorted(
        element for element in component_view["element_counts"] if element not in set(supported_elements)
    )
    if not unsupported:
        return False, {}
    return True, {
        "summary": f"unsupported elements present: {', '.join(unsupported)}",
        "unsupported_elements": unsupported,
    }


def _component_has_bound_lithium(component_view: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    lithium_atoms = [
        atom for atom in component_view["atoms"] if atom["element"] == "Li" and atom["coordination"]["coordination_number"] > 0
    ]
    if not lithium_atoms:
        return False, {}
    return True, {
        "summary": "lithium coordination is outside the PT0/PT3 supported scope",
        "source_indices": [atom["source_index"] for atom in lithium_atoms],
    }


def _component_has_aromatic_ring(component_view: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    if not component_view["aromatic_ring_ids"]:
        return False, {}
    return True, {
        "summary": f"aromatic rings detected: {', '.join(component_view['aromatic_ring_ids'])}",
        "ring_ids": component_view["aromatic_ring_ids"],
    }


def _component_has_indeterminate_ring_aromaticity(
    component_view: dict,
    *,
    supported_elements: list[str],
) -> tuple[bool, dict]:
    if not component_view["indeterminate_ring_ids"]:
        return False, {}
    return True, {
        "summary": f"ring aromaticity is indeterminate for: {', '.join(component_view['indeterminate_ring_ids'])}",
        "ring_ids": component_view["indeterminate_ring_ids"],
    }


def _component_has_carbonyl_bond(component_view: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    atom_elements = {atom["canonical_index"]: atom["element"] for atom in component_view["atoms"]}
    carbonyl_bonds = []
    for bond in component_view["bonds"]:
        if bond["order"] != 2:
            continue
        left, right = bond["atom_indices"]
        pair = {atom_elements[left], atom_elements[right]}
        if pair == {"C", "O"}:
            carbonyl_bonds.append(bond["canonical_index"])
    if not carbonyl_bonds:
        return False, {}
    return True, {
        "summary": f"carbonyl-like C=O bonds detected: {', '.join(str(index) for index in carbonyl_bonds)}",
        "bond_indices": carbonyl_bonds,
    }


def _component_is_lithium_cation(component_view: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    if component_view["atom_count"] != 1:
        return False, {}
    atom = component_view["atoms"][0]
    if atom["element"] != "Li" or atom["formal_charge"] != 1 or atom["coordination"]["coordination_number"] != 0:
        return False, {}
    return True, {
        "summary": "single monatomic lithium cation",
        "source_indices": [atom["source_index"]],
    }


def _component_is_acyclic_alkane(component_view: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    if component_view["rings"]:
        return False, {}
    if set(component_view["element_counts"]) != {"C", "H"}:
        return False, {}
    if any(bond["order"] != 1 for bond in component_view["bonds"]):
        return False, {}
    carbons = [atom for atom in component_view["atoms"] if atom["element"] == "C"]
    hydrogens = [atom for atom in component_view["atoms"] if atom["element"] == "H"]
    if not carbons or not hydrogens:
        return False, {}
    if any(atom["valence"]["inferred_valence"] != 4 for atom in carbons):
        return False, {}
    if any(atom["coordination"]["coordination_number"] != 4 for atom in carbons):
        return False, {}
    if any(atom["valence"]["inferred_valence"] != 1 for atom in hydrogens):
        return False, {}
    return True, {
        "summary": "acyclic saturated hydrocarbon component",
        "carbon_source_indices": [atom["source_index"] for atom in carbons],
    }


def _component_is_acyclic_ether(component_view: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    if component_view["rings"]:
        return False, {}
    if set(component_view["element_counts"]) - {"C", "H", "O"}:
        return False, {}
    if any(bond["order"] != 1 for bond in component_view["bonds"]):
        return False, {}
    oxygens = [atom for atom in component_view["atoms"] if atom["element"] == "O"]
    carbons = [atom for atom in component_view["atoms"] if atom["element"] == "C"]
    if not oxygens or not carbons:
        return False, {}
    for oxygen in oxygens:
        if oxygen["coordination"]["coordination_number"] != 2:
            return False, {}
        if oxygen["neighbor_element_counts"] != {"C": 2}:
            return False, {}
    for carbon in carbons:
        if carbon["valence"]["inferred_valence"] != 4:
            return False, {}
        if "O" not in carbon["neighbor_element_counts"]:
            return False, {}
    return True, {
        "summary": "acyclic ether component with explicit alpha-carbon environments",
        "oxygen_source_indices": [atom["source_index"] for atom in oxygens],
    }


def _component_is_acyclic_polyether_oligomer(component_view: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    if component_view["rings"]:
        return False, {}
    if set(component_view["element_counts"]) - {"C", "H", "O"}:
        return False, {}
    if any(bond["order"] != 1 for bond in component_view["bonds"]):
        return False, {}

    oxygens = [atom for atom in component_view["atoms"] if atom["element"] == "O"]
    carbons = [atom for atom in component_view["atoms"] if atom["element"] == "C"]
    if len(oxygens) < 2 or len(carbons) < 4:
        return False, {}
    for oxygen in oxygens:
        if oxygen["coordination"]["coordination_number"] != 2:
            return False, {}
        if oxygen["neighbor_element_counts"] != {"C": 2}:
            return False, {}
    for carbon in carbons:
        if carbon["valence"]["inferred_valence"] != 4 or carbon["coordination"]["coordination_number"] != 4:
            return False, {}

    end_carbons = [atom for atom in carbons if atom["neighbor_element_counts"] == {"H": 3, "O": 1}]
    backbone_carbons = [atom for atom in carbons if atom["neighbor_element_counts"] == {"C": 1, "H": 2, "O": 1}]
    if len(end_carbons) != 2 or not backbone_carbons:
        return False, {}
    if len(end_carbons) + len(backbone_carbons) != len(carbons):
        return False, {}
    if len(backbone_carbons) % 2 != 0:
        return False, {}
    if len(oxygens) != (len(backbone_carbons) // 2) + 1:
        return False, {}

    return True, {
        "summary": "linear methoxy-capped polyether oligomer with explicit repeat-unit backbone methylenes",
        "repeat_unit_count": len(backbone_carbons) // 2,
        "end_carbon_source_indices": [atom["source_index"] for atom in end_carbons],
        "backbone_carbon_source_indices": [atom["source_index"] for atom in backbone_carbons],
        "oxygen_source_indices": [atom["source_index"] for atom in oxygens],
    }


def _component_is_tfsi_like_sulfonimide(component_view: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    atoms = component_view["atoms"]
    if set(component_view["element_counts"]) != {"C", "F", "N", "O", "S"}:
        return False, {}
    if component_view["net_formal_charge"] != -1:
        return False, {}
    if component_view["element_counts"] != {"C": 2, "F": 6, "N": 1, "O": 4, "S": 2}:
        return False, {}

    nitrogens = [atom for atom in atoms if atom["element"] == "N"]
    sulfurs = [atom for atom in atoms if atom["element"] == "S"]
    carbons = [atom for atom in atoms if atom["element"] == "C"]
    oxygens = [atom for atom in atoms if atom["element"] == "O"]
    fluorines = [atom for atom in atoms if atom["element"] == "F"]

    if len(nitrogens) != 1 or nitrogens[0]["formal_charge"] != -1 or nitrogens[0]["neighbor_element_counts"] != {"S": 2}:
        return False, {}
    if any(atom["neighbor_element_counts"] != {"C": 1, "N": 1, "O": 2} or atom["valence"]["inferred_valence"] != 6 for atom in sulfurs):
        return False, {}
    if any(atom["neighbor_element_counts"] != {"F": 3, "S": 1} for atom in carbons):
        return False, {}
    if any(atom["neighbor_element_counts"] != {"S": 1} for atom in oxygens):
        return False, {}
    if any(atom["attached_bond"] is None or atom["attached_bond"]["order"] != 2 for atom in oxygens):
        return False, {}
    if any(atom["attached_atom"] is None or atom["attached_atom"]["neighbor_element_counts"] != {"F": 3, "S": 1} for atom in fluorines):
        return False, {}

    return True, {
        "summary": "explicit TFSI-like sulfonimide anion pattern",
        "nitrogen_source_index": nitrogens[0]["source_index"],
        "sulfur_source_indices": [atom["source_index"] for atom in sulfurs],
    }


def _always_true(component_view: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    return True, {"summary": "no supported PT3 component-family rule matched"}


COMPONENT_PREDICATES = {
    "always_true": _always_true,
    "component_has_aromatic_ring": _component_has_aromatic_ring,
    "component_has_bound_lithium": _component_has_bound_lithium,
    "component_has_carbonyl_bond": _component_has_carbonyl_bond,
    "component_has_indeterminate_ring_aromaticity": _component_has_indeterminate_ring_aromaticity,
    "component_has_unsupported_element": _component_has_unsupported_element,
    "component_is_acyclic_alkane": _component_is_acyclic_alkane,
    "component_is_acyclic_ether": _component_is_acyclic_ether,
    "component_is_acyclic_polyether_oligomer": _component_is_acyclic_polyether_oligomer,
    "component_is_lithium_cation": _component_is_lithium_cation,
    "component_is_tfsi_like_sulfonimide": _component_is_tfsi_like_sulfonimide,
}


def _reject_message(failure_code: str, summary: str) -> str:
    if failure_code == "unsupported_aromatic_sp2_ring":
        return f"unsupported PT0 aromatic sp2 ring: {summary}"
    if failure_code == "unsupported_carbonyl_chemistry":
        return f"unsupported PT0 carbonyl chemistry: {summary}"
    if failure_code == "unsupported_component_family":
        return f"unsupported PT0 component family: {summary}"
    if failure_code == "unsupported_resonance_encoding":
        return f"unsupported PT0 resonance encoding: {summary}"
    return f"{failure_code}: {summary}"


def _display_rules_path(path: Path) -> str:
    resolved = path.resolve()
    repo_root = DEFAULT_RULES_PATH.parents[1]
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return str(resolved)
