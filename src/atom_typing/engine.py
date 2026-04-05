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

PHASE1_VIRTUAL_COMPONENT_RULES = (
    {
        "family": "carbonate_like",
        "kind": "support",
        "precedence": 38,
        "predicate": {
            "builtin": "component_is_phase1_carbonate_like",
        },
        "rule_id": "classify_phase1_carbonate_like",
    },
    {
        "family": "amide_like",
        "kind": "support",
        "precedence": 38,
        "predicate": {
            "builtin": "component_is_phase1_amide_like",
        },
        "rule_id": "classify_phase1_amide_like",
    },
)


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
    component_features = detect_component_features(
        component_view,
        supported_elements=active_ruleset["supported_elements"],
    )
    classification = classify_component_family(component_features, ruleset=active_ruleset)

    atom_records = []
    explanations = []
    diagnostics = []
    overall_status = "typed"

    if classification["status"] == "supported":
        dispatch = dispatch_family_typing_rules(
            classification["family"],
            {"atom_contexts": atom_contexts},
            ruleset=active_ruleset,
            component_rule_id=classification["rule_id"],
        )
        atom_records = dispatch["atom_records"]
        explanations = dispatch["explanations"]
        diagnostics.extend(classification["diagnostics"])
        diagnostics.extend(dispatch["diagnostics"])
        overall_status = dispatch["status"]
    else:
        dispatch = _skipped_dispatch(classification, atom_contexts)
        atom_records = dispatch["atom_records"]
        diagnostics.extend(classification["diagnostics"])
        diagnostics.extend(dispatch["diagnostics"])
        overall_status = "unsupported" if classification["status"] == "unclassified" else classification["status"]

    typing_trace = emit_typing_trace(
        component_view,
        features=component_features,
        classification=classification,
        dispatch=dispatch,
    )

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
                "typing_trace": typing_trace,
            }
        ],
    }
    validate_typing_report(report)
    return report


def detect_component_features(component: dict, *, supported_elements: list[str] | None = None) -> dict:
    atom_by_index = {atom["canonical_index"]: atom for atom in component["atoms"]}
    adjacency = {index: [] for index in atom_by_index}
    carbonyl_bond_indices = []
    supported_element_set = None if supported_elements is None else set(supported_elements)

    for bond in component["bonds"]:
        left, right = bond["atom_indices"]
        adjacency[left].append((right, bond))
        adjacency[right].append((left, bond))
        pair = {atom_by_index[left]["element"], atom_by_index[right]["element"]}
        if bond["order"] == 2 and pair == {"C", "O"}:
            carbonyl_bond_indices.append(bond["canonical_index"])

    unsupported_elements = sorted(
        element
        for element in component["element_counts"]
        if supported_element_set is not None and element not in supported_element_set
    )
    bound_lithium_source_indices = sorted(
        atom["source_index"]
        for atom in component["atoms"]
        if atom["element"] == "Li" and atom["coordination"]["coordination_number"] > 0
    )

    carbonyl_site_records = []
    carbonate_like_sites = []
    amide_like_sites = []
    aldehyde_carbonyl_source_indices = []
    ester_carbonyl_source_indices = []
    carbonyl_oxygen_source_indices = []
    bridging_oxygen_source_indices = []
    hydroxyl_oxygen_source_indices = []
    hydroxyl_hydrogen_source_indices = []

    for atom in component["atoms"]:
        if atom["element"] == "O":
            bond_orders = [bond["order"] for _, bond in adjacency[atom["canonical_index"]]]
            if (
                atom["coordination"]["coordination_number"] == 1
                and atom["attached_atom"] is not None
                and atom["attached_atom"]["element"] == "C"
                and atom["attached_bond"] is not None
                and atom["attached_bond"]["order"] == 2
            ):
                carbonyl_oxygen_source_indices.append(atom["source_index"])
            elif atom["neighbor_element_counts"] == {"C": 2} and all(order == 1 for order in bond_orders):
                bridging_oxygen_source_indices.append(atom["source_index"])
            elif atom["neighbor_element_counts"] == {"C": 1, "H": 1} and all(order == 1 for order in bond_orders):
                hydroxyl_oxygen_source_indices.append(atom["source_index"])
        elif (
            atom["element"] == "H"
            and atom["attached_atom"] is not None
            and atom["attached_atom"]["element"] == "O"
            and atom["attached_atom"]["neighbor_element_counts"] == {"C": 1, "H": 1}
        ):
            hydroxyl_hydrogen_source_indices.append(atom["source_index"])

        carbonyl_site = _carbonyl_site_record(atom, atom_by_index, adjacency)
        if carbonyl_site is None:
            continue
        carbonyl_site_records.append(carbonyl_site)

        if (
            carbonyl_site["single_bond_oxygen_neighbor_count"] >= 1
            and carbonyl_site["single_bond_nitrogen_neighbor_count"] == 0
            and carbonyl_site["single_bond_sulfur_neighbor_count"] == 0
        ):
            carbonate_like_sites.append(carbonyl_site)
        if (
            carbonyl_site["single_bond_nitrogen_neighbor_count"] == 1
            and carbonyl_site["single_bond_oxygen_neighbor_count"] == 0
            and carbonyl_site["single_bond_sulfur_neighbor_count"] == 0
        ):
            amide_like_sites.append(carbonyl_site)
        if (
            carbonyl_site["single_bond_oxygen_neighbor_count"] == 1
            and carbonyl_site["single_bond_nitrogen_neighbor_count"] == 0
            and carbonyl_site["single_bond_sulfur_neighbor_count"] == 0
            and carbonyl_site["single_bond_carbon_neighbor_count"] >= 1
        ):
            ester_carbonyl_source_indices.append(carbonyl_site["source_index"])
        if (
            carbonyl_site["single_bond_hydrogen_neighbor_count"] == 1
            and carbonyl_site["single_bond_carbon_neighbor_count"] == 1
            and carbonyl_site["single_bond_oxygen_neighbor_count"] == 0
            and carbonyl_site["single_bond_nitrogen_neighbor_count"] == 0
            and carbonyl_site["single_bond_sulfur_neighbor_count"] == 0
        ):
            aldehyde_carbonyl_source_indices.append(carbonyl_site["source_index"])

    return {
        **component,
        "supported_elements": None if supported_elements is None else sorted(supported_element_set),
        "unsupported_elements": unsupported_elements,
        "bound_lithium_source_indices": bound_lithium_source_indices,
        "carbonyl_bond_indices": sorted(carbonyl_bond_indices),
        "carbonyl_site_records": carbonyl_site_records,
        "phase1_carbonate_like_sites": carbonate_like_sites,
        "phase1_amide_like_sites": amide_like_sites,
        "csv_scope_signature": {
            "aldehyde_carbonyl_source_indices": sorted(aldehyde_carbonyl_source_indices),
            "ester_carbonyl_source_indices": sorted(ester_carbonyl_source_indices),
            "carbonyl_oxygen_source_indices": sorted(carbonyl_oxygen_source_indices),
            "bridging_oxygen_source_indices": sorted(bridging_oxygen_source_indices),
            "hydroxyl_oxygen_source_indices": sorted(hydroxyl_oxygen_source_indices),
            "hydroxyl_hydrogen_source_indices": sorted(hydroxyl_hydrogen_source_indices),
        },
    }


def classify_component_family(features: dict, *, ruleset: dict | None = None, rules_path: str | Path | None = None) -> dict:
    active_ruleset = load_rules(rules_path) if ruleset is None else copy.deepcopy(ruleset)
    validate_rules(active_ruleset)

    evaluations = []
    matches = []
    for order, rule in enumerate(_component_rule_plan(active_ruleset)):
        matched, evidence = _evaluate_component_rule(
            features,
            rule,
            supported_elements=active_ruleset["supported_elements"],
        )
        evaluations.append(
            {
                "rule_id": rule["rule_id"],
                "kind": rule["kind"],
                "family": rule.get("family"),
                "failure_code": rule.get("failure_code"),
                "precedence": rule["precedence"],
                "matched": matched,
                "evidence": evidence,
            }
        )
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
    trace = {
        "evaluations": evaluations,
        "matched_rule_ids": [match["rule"]["rule_id"] for match in matches],
        "feature_summary": _feature_summary(features),
    }
    if len(best_matches) > 1:
        candidate_rule_ids = [match["rule"]["rule_id"] for match in best_matches]
        candidate_families = sorted(
            rule["family"]
            for rule in (match["rule"] for match in best_matches)
            if rule["kind"] == "support"
        )
        diagnostic = {
            "scope": "component",
            "code": "ambiguous_component_classification",
            "message": f"Multiple component-family rules matched at precedence {best_precedence}: {', '.join(candidate_rule_ids)}",
            "candidate_rule_ids": candidate_rule_ids,
            "candidate_families": candidate_families,
        }
        return {
            "status": "ambiguous",
            "family": None,
            "candidate_families": candidate_families,
            "rule_id": None,
            "precedence": best_precedence,
            "diagnostics": [diagnostic],
            "trace": trace,
        }

    winner = best_matches[0]
    rule = winner["rule"]
    if rule["kind"] == "reject":
        status = "unclassified" if rule["failure_code"] == "unsupported_component_family" else "unsupported"
        diagnostic = {
            "scope": "component",
            "code": rule["failure_code"],
            "message": _reject_message(rule["failure_code"], winner["evidence"]["summary"]),
            "rule_id": rule["rule_id"],
            "evidence": winner["evidence"],
        }
        return {
            "status": status,
            "family": None,
            "candidate_families": [],
            "rule_id": rule["rule_id"],
            "precedence": rule["precedence"],
            "failure_code": rule["failure_code"],
            "diagnostics": [diagnostic],
            "explanation": winner["evidence"],
            "trace": trace,
        }

    return {
        "status": "supported",
        "family": rule["family"],
        "candidate_families": [rule["family"]],
        "rule_id": rule["rule_id"],
        "precedence": rule["precedence"],
        "diagnostics": [],
        "explanation": winner["evidence"],
        "trace": trace,
    }


def dispatch_family_typing_rules(
    family: str | None,
    component: dict,
    *,
    ruleset: dict | None = None,
    rules_path: str | Path | None = None,
    component_rule_id: str | None = None,
) -> dict:
    atom_contexts = component.get("atom_contexts", component.get("atoms", []))
    active_ruleset = load_rules(rules_path) if ruleset is None else copy.deepcopy(ruleset)
    validate_rules(active_ruleset)

    if family is None:
        atom_records = _skipped_atom_records(atom_contexts, "unsupported")
        return {
            "status": "unsupported",
            "atom_records": atom_records,
            "explanations": [],
            "diagnostics": [
                {
                    "scope": "component",
                    "code": "missing_dispatch_family",
                    "message": "family dispatch requested without a classified family",
                }
            ],
            "trace": {
                "dispatcher": "missing_family_dispatch",
                "family": None,
                "component_rule_id": component_rule_id,
                "rule_count": 0,
                "atom_outcome_summary": _summarize_atom_outcomes(atom_records),
                "used_family_dispatch": True,
            },
        }

    rule_count = sum(1 for rule in active_ruleset["atom_type_rules"] if rule["component_family"] == family)
    if rule_count == 0:
        atom_records = _skipped_atom_records(atom_contexts, "unsupported")
        return {
            "status": "unsupported",
            "atom_records": atom_records,
            "explanations": [],
            "diagnostics": [
                {
                    "scope": "component",
                    "code": "unsupported_family_dispatch_bundle",
                    "message": f"No atom typing rule bundle is registered for component family {family!r}",
                    "family": family,
                }
            ],
            "trace": {
                "dispatcher": "missing_family_bundle",
                "family": family,
                "component_rule_id": component_rule_id,
                "rule_count": 0,
                "atom_outcome_summary": _summarize_atom_outcomes(atom_records),
                "used_family_dispatch": True,
            },
        }

    atom_records, explanations, diagnostics = _assign_atom_types(
        atom_contexts,
        component_family=family,
        component_rule_id=component_rule_id or "family_dispatch_direct",
        ruleset=active_ruleset,
    )
    status = "typed"
    if any(record["status"] == "ambiguous" for record in atom_records):
        status = "ambiguous"
    elif any(record["status"] == "unresolved" for record in atom_records):
        status = "unresolved"
    return {
        "status": status,
        "atom_records": atom_records,
        "explanations": explanations,
        "diagnostics": diagnostics,
        "trace": {
            "dispatcher": "ruleset_component_family_bundle",
            "family": family,
            "component_rule_id": component_rule_id,
            "rule_count": rule_count,
            "atom_outcome_summary": _summarize_atom_outcomes(atom_records),
            "used_family_dispatch": True,
        },
    }


def emit_typing_trace(component: dict, *, features: dict, classification: dict, dispatch: dict) -> dict:
    return {
        "component_name": component["name"],
        "component_atom_count": component["atom_count"],
        "feature_summary": _feature_summary(features),
        "classification": copy.deepcopy(classification["trace"]),
        "dispatch": copy.deepcopy(dispatch["trace"]),
    }


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
            "typing_outcome": "failed",
            "assigned_family": None,
            "assigned_atom_type": None,
            "matched_rule_id": None,
            "precedence": None,
            "fallback_behavior": None,
            "failure_reason": f"dispatch_skipped_{status}",
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
    adjacency = {index: [] for index in perception_atoms}
    for bond in ir_component["bonds"]:
        left, right = bond["atom_indices"]
        adjacency[left].append((right, bond))
        adjacency[right].append((left, bond))

    carbonyl_carbon_indices = set()
    carbonyl_oxygen_indices = set()
    nitrile_carbon_indices = set()
    nitrile_nitrogen_indices = set()
    for bond in ir_component["bonds"]:
        left, right = bond["atom_indices"]
        left_element = perception_atoms[left]["element"]
        right_element = perception_atoms[right]["element"]
        pair = {left_element, right_element}
        if bond["order"] == 2 and pair == {"C", "O"}:
            carbonyl_carbon_indices.update(
                index for index, element in ((left, left_element), (right, right_element)) if element == "C"
            )
            carbonyl_oxygen_indices.update(
                index for index, element in ((left, left_element), (right, right_element)) if element == "O"
            )
        if bond["order"] == 3 and pair == {"C", "N"}:
            nitrile_carbon_indices.update(
                index for index, element in ((left, left_element), (right, right_element)) if element == "C"
            )
            nitrile_nitrogen_indices.update(
                index for index, element in ((left, left_element), (right, right_element)) if element == "N"
            )

    contexts = []
    for canonical_index in sorted(perception_atoms):
        perception_atom = perception_atoms[canonical_index]
        ir_atom = ir_atoms[canonical_index]
        neighbor_indices = perception_atom["neighbor_indices"]
        bond_order_counts = _bond_order_histogram(canonical_index, adjacency)
        single_bond_neighbor_element_counts = _neighbor_element_histogram_for_bond_order(
            canonical_index,
            adjacency,
            perception_atoms,
            bond_order=1,
        )
        double_bond_neighbor_element_counts = _neighbor_element_histogram_for_bond_order(
            canonical_index,
            adjacency,
            perception_atoms,
            bond_order=2,
        )
        triple_bond_neighbor_element_counts = _neighbor_element_histogram_for_bond_order(
            canonical_index,
            adjacency,
            perception_atoms,
            bond_order=3,
        )
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
                "single_bond_neighbor_element_counts": single_bond_neighbor_element_counts,
                "double_bond_neighbor_element_counts": double_bond_neighbor_element_counts,
                "triple_bond_neighbor_element_counts": triple_bond_neighbor_element_counts,
                "bond_order_counts": bond_order_counts,
                "is_carbonyl_carbon": canonical_index in carbonyl_carbon_indices,
                "is_carbonyl_oxygen": canonical_index in carbonyl_oxygen_indices,
                "is_nitrile_carbon": canonical_index in nitrile_carbon_indices,
                "is_nitrile_nitrogen": canonical_index in nitrile_nitrogen_indices,
                "single_bond_neighbor_carbonyl_carbon_count": _count_neighbors_in_index_set(
                    canonical_index,
                    adjacency,
                    carbonyl_carbon_indices,
                    bond_order=1,
                ),
                "single_bond_neighbor_nitrile_carbon_count": _count_neighbors_in_index_set(
                    canonical_index,
                    adjacency,
                    nitrile_carbon_indices,
                    bond_order=1,
                ),
                "valence": perception_atom["valence"],
                "ring": perception_atom["ring"],
                "aromaticity": perception_atom["aromaticity"],
                "coordination": perception_atom["coordination"],
                "improper_center_candidate": perception_atom["improper_center_candidate"],
                "polymer_connection": perception_atom["polymer_connection"],
                "_attached_index": neighbor_indices[0] if len(neighbor_indices) == 1 else None,
                "_attached_bond_index": (
                    pair_to_bond_index[tuple(sorted((canonical_index, neighbor_indices[0])))]
                    if len(neighbor_indices) == 1
                    else None
                ),
            }
        )

    context_by_index = {context["canonical_index"]: context for context in contexts}
    for context in contexts:
        attached_index = context.pop("_attached_index")
        attached_bond_index = context.pop("_attached_bond_index")
        if attached_index is None:
            context["attached_atom"] = None
            context["attached_bond"] = None
            continue
        context["attached_atom"] = _attached_atom_view(context_by_index[attached_index])
        attached_bond = ir_bonds[attached_bond_index]
        context["attached_bond"] = {
            "canonical_index": attached_bond["canonical_index"],
            "order": attached_bond["order"],
            "bond_code": attached_bond["bond_code"],
        }
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


def _bond_order_histogram(canonical_index: int, adjacency: dict[int, list[tuple[int, dict]]]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for _, bond in adjacency[canonical_index]:
        key = "unknown" if bond["order"] is None else str(bond["order"])
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items()))


def _neighbor_element_histogram_for_bond_order(
    canonical_index: int,
    adjacency: dict[int, list[tuple[int, dict]]],
    atom_by_index: dict[int, dict],
    *,
    bond_order: int,
) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for neighbor_index, bond in adjacency[canonical_index]:
        if bond["order"] != bond_order:
            continue
        element = atom_by_index[neighbor_index]["element"]
        histogram[element] = histogram.get(element, 0) + 1
    return dict(sorted(histogram.items()))


def _count_neighbors_in_index_set(
    canonical_index: int,
    adjacency: dict[int, list[tuple[int, dict]]],
    target_indices: set[int],
    *,
    bond_order: int | None = None,
) -> int:
    count = 0
    for neighbor_index, bond in adjacency[canonical_index]:
        if bond_order is not None and bond["order"] != bond_order:
            continue
        if neighbor_index in target_indices:
            count += 1
    return count


def _attached_atom_view(atom_context: dict) -> dict:
    return {
        "canonical_index": atom_context["canonical_index"],
        "source_index": atom_context["source_index"],
        "source_atom_id": atom_context["source_atom_id"],
        "element": atom_context["element"],
        "formal_charge": atom_context["formal_charge"],
        "neighbor_element_counts": atom_context["neighbor_element_counts"],
        "single_bond_neighbor_element_counts": atom_context["single_bond_neighbor_element_counts"],
        "double_bond_neighbor_element_counts": atom_context["double_bond_neighbor_element_counts"],
        "triple_bond_neighbor_element_counts": atom_context["triple_bond_neighbor_element_counts"],
        "bond_order_counts": atom_context["bond_order_counts"],
        "is_carbonyl_carbon": atom_context["is_carbonyl_carbon"],
        "is_carbonyl_oxygen": atom_context["is_carbonyl_oxygen"],
        "is_nitrile_carbon": atom_context["is_nitrile_carbon"],
        "is_nitrile_nitrogen": atom_context["is_nitrile_nitrogen"],
        "single_bond_neighbor_carbonyl_carbon_count": atom_context["single_bond_neighbor_carbonyl_carbon_count"],
        "single_bond_neighbor_nitrile_carbon_count": atom_context["single_bond_neighbor_nitrile_carbon_count"],
        "valence": atom_context["valence"],
        "aromaticity": atom_context["aromaticity"],
        "coordination": atom_context["coordination"],
    }


def _evaluate_component_rule(component_view: dict, rule: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    predicate_name = rule["predicate"]["builtin"]
    predicate = COMPONENT_PREDICATES[predicate_name]
    return predicate(component_view, supported_elements=supported_elements)


def _component_rule_plan(ruleset: dict) -> list[dict]:
    rules = [copy.deepcopy(rule) for rule in ruleset["component_rules"]]
    existing_rule_ids = {rule["rule_id"] for rule in rules}
    for rule in PHASE1_VIRTUAL_COMPONENT_RULES:
        if rule["rule_id"] not in existing_rule_ids:
            rules.append(copy.deepcopy(rule))
    rules.sort(key=lambda rule: (rule["precedence"], rule["rule_id"]))
    return rules


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
                    "typing_outcome": "failed",
                    "assigned_family": None,
                    "assigned_atom_type": None,
                    "matched_rule_id": None,
                    "precedence": None,
                    "fallback_behavior": None,
                    "failure_reason": "phase1_rule_bundle_exhausted",
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
                    "typing_outcome": "failed",
                    "assigned_family": None,
                    "assigned_atom_type": None,
                    "matched_rule_id": None,
                    "precedence": best_precedence,
                    "fallback_behavior": None,
                    "failure_reason": "ambiguous_phase1_rule_match",
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
        typing_outcome = _rule_typing_outcome(rule)
        atom_records.append(
            {
                "canonical_index": context["canonical_index"],
                "source_index": context["source_index"],
                "source_atom_id": context["source_atom_id"],
                "element": context["element"],
                "status": "assigned",
                "typing_outcome": typing_outcome,
                "assigned_family": rule["family"],
                "assigned_atom_type": rule["family"],
                "matched_rule_id": rule["rule_id"],
                "precedence": rule["precedence"],
                "explanation_id": explanation["explanation_id"],
                "fallback_behavior": rule.get("fallback_behavior"),
                "failure_reason": rule.get("failure_reason"),
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
    return {
        "explanation_id": f"atom_{atom_context['canonical_index']}",
        "atom_index": atom_context["canonical_index"],
        "source_index": atom_context["source_index"],
        "source_atom_id": atom_context["source_atom_id"],
        "element": atom_context["element"],
        "assigned_family": rule["family"],
        "assigned_atom_type": rule["family"],
        "rule_id": rule["rule_id"],
        "precedence": rule["precedence"],
        "typing_outcome": _rule_typing_outcome(rule),
        "fallback_behavior": rule.get("fallback_behavior"),
        "failure_reason": rule.get("failure_reason"),
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


def _rule_typing_outcome(rule: dict) -> str:
    outcome = rule.get("typing_mode", "exact")
    if outcome not in {"exact", "fallback", "assumed"}:
        raise TypingReportError("invalid_rule_condition", f"Unsupported typing_mode {outcome!r}")
    return outcome


def _summarize_atom_outcomes(atom_records: list[dict]) -> dict:
    summary = {
        "exact_typed": 0,
        "fallback_typed": 0,
        "assumed_typed": 0,
        "failed": 0,
    }
    for record in atom_records:
        outcome = record.get("typing_outcome")
        if outcome == "exact":
            summary["exact_typed"] += 1
        elif outcome == "fallback":
            summary["fallback_typed"] += 1
        elif outcome == "assumed":
            summary["assumed_typed"] += 1
        else:
            summary["failed"] += 1
    return summary


def _feature_summary(features: dict) -> dict:
    return {
        "atom_count": features["atom_count"],
        "bond_count": features["bond_count"],
        "net_formal_charge": features["net_formal_charge"],
        "element_counts": copy.deepcopy(features["element_counts"]),
        "unsupported_elements": copy.deepcopy(features["unsupported_elements"]),
        "bound_lithium_source_indices": copy.deepcopy(features["bound_lithium_source_indices"]),
        "aromatic_ring_ids": copy.deepcopy(features["aromatic_ring_ids"]),
        "indeterminate_ring_ids": copy.deepcopy(features["indeterminate_ring_ids"]),
        "carbonyl_bond_indices": copy.deepcopy(features["carbonyl_bond_indices"]),
        "phase1_carbonate_like_source_indices": [
            record["source_index"] for record in features["phase1_carbonate_like_sites"]
        ],
        "phase1_amide_like_source_indices": [
            record["source_index"] for record in features["phase1_amide_like_sites"]
        ],
        "csv_scope_signature": copy.deepcopy(features["csv_scope_signature"]),
    }


def _skipped_dispatch(classification: dict, atom_contexts: list[dict]) -> dict:
    atom_records = _skipped_atom_records(atom_contexts, classification["status"])
    return {
        "status": classification["status"],
        "atom_records": atom_records,
        "explanations": [],
        "diagnostics": [],
        "trace": {
            "dispatcher": "not_invoked",
            "family": classification.get("family"),
            "component_rule_id": classification.get("rule_id"),
            "classification_status": classification["status"],
            "atom_outcome_summary": _summarize_atom_outcomes(atom_records),
            "used_family_dispatch": True,
        },
    }


def _carbonyl_site_record(atom: dict, atom_by_index: dict[int, dict], adjacency: dict[int, list[tuple[int, dict]]]) -> dict | None:
    if atom["element"] != "C":
        return None

    counts = {
        "single_bond_oxygen_neighbor_count": 0,
        "single_bond_nitrogen_neighbor_count": 0,
        "single_bond_sulfur_neighbor_count": 0,
        "single_bond_carbon_neighbor_count": 0,
        "single_bond_hydrogen_neighbor_count": 0,
    }
    double_bond_oxygen_source_indices = []
    for neighbor_index, bond in adjacency[atom["canonical_index"]]:
        neighbor = atom_by_index[neighbor_index]
        if bond["order"] == 2 and neighbor["element"] == "O":
            double_bond_oxygen_source_indices.append(neighbor["source_index"])
            continue
        if bond["order"] != 1:
            continue
        element = neighbor["element"]
        if element == "O":
            counts["single_bond_oxygen_neighbor_count"] += 1
        elif element == "N":
            counts["single_bond_nitrogen_neighbor_count"] += 1
        elif element == "S":
            counts["single_bond_sulfur_neighbor_count"] += 1
        elif element == "C":
            counts["single_bond_carbon_neighbor_count"] += 1
        elif element == "H":
            counts["single_bond_hydrogen_neighbor_count"] += 1

    if len(double_bond_oxygen_source_indices) != 1:
        return None
    return {
        "canonical_index": atom["canonical_index"],
        "source_index": atom["source_index"],
        "double_bond_oxygen_source_indices": sorted(double_bond_oxygen_source_indices),
        **counts,
    }


def _component_has_unsupported_element(component_view: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    unsupported = component_view["unsupported_elements"]
    if not unsupported:
        return False, {}
    return True, {
        "summary": f"unsupported elements present: {', '.join(unsupported)}",
        "unsupported_elements": unsupported,
    }


def _component_has_bound_lithium(component_view: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    if not component_view["bound_lithium_source_indices"]:
        return False, {}
    return True, {
        "summary": "lithium coordination is outside the PT0/PT3 supported scope",
        "source_indices": component_view["bound_lithium_source_indices"],
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
    carbonyl_bonds = component_view["carbonyl_bond_indices"]
    if not carbonyl_bonds:
        return False, {}
    return True, {
        "summary": f"carbonyl-like C=O bonds detected: {', '.join(str(index) for index in carbonyl_bonds)}",
        "bond_indices": carbonyl_bonds,
    }


def _component_is_csv_scope_pysoftk_aldehyde_polyester(
    component_view: dict,
    *,
    supported_elements: list[str],
) -> tuple[bool, dict]:
    if component_view["rings"]:
        return False, {}
    if set(component_view["element_counts"]) - {"C", "H", "O"}:
        return False, {}

    for bond in component_view["bonds"]:
        if bond["order"] not in {1, 2}:
            return False, {}
        if bond["order"] == 2:
            left, right = bond["atom_indices"]
            pair = {
                component_view["atoms"][left - 1]["element"],
                component_view["atoms"][right - 1]["element"],
            }
            if pair != {"C", "O"}:
                return False, {}

    signature = component_view["csv_scope_signature"]
    required_counts = {
        "aldehyde_carbonyl_source_indices": signature["aldehyde_carbonyl_source_indices"],
        "ester_carbonyl_source_indices": signature["ester_carbonyl_source_indices"],
        "carbonyl_oxygen_source_indices": signature["carbonyl_oxygen_source_indices"],
        "bridging_oxygen_source_indices": signature["bridging_oxygen_source_indices"],
        "hydroxyl_oxygen_source_indices": signature["hydroxyl_oxygen_source_indices"],
        "hydroxyl_hydrogen_source_indices": signature["hydroxyl_hydrogen_source_indices"],
    }
    if any(not values for values in required_counts.values()):
        return False, {}

    return True, {
        "summary": "csv-scope pysoftk aldehyde polyester oligomer with ester backbone and terminal alcohol signature",
        "signature": copy.deepcopy(required_counts),
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


def _component_is_phase1_carbonate_like(component_view: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    sites = component_view["phase1_carbonate_like_sites"]
    if not sites:
        return False, {}
    return True, {
        "summary": "phase1 carbonate-like oxygen-substituted carbonyl motif detected",
        "source_indices": [record["source_index"] for record in sites],
    }


def _component_is_phase1_amide_like(component_view: dict, *, supported_elements: list[str]) -> tuple[bool, dict]:
    sites = component_view["phase1_amide_like_sites"]
    if not sites:
        return False, {}
    return True, {
        "summary": "phase1 amide-like carbonyl-bound nitrogen motif detected",
        "source_indices": [record["source_index"] for record in sites],
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
    "component_is_csv_scope_pysoftk_aldehyde_polyester": _component_is_csv_scope_pysoftk_aldehyde_polyester,
    "component_is_lithium_cation": _component_is_lithium_cation,
    "component_is_phase1_amide_like": _component_is_phase1_amide_like,
    "component_is_phase1_carbonate_like": _component_is_phase1_carbonate_like,
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
