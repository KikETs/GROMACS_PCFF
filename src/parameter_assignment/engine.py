from __future__ import annotations

import copy
import hashlib
import json
from itertools import combinations
from pathlib import Path

from atom_typing import (
    dumps_typing_report,
    type_ir,
    validate_typing_report,
)
from chem_perception import (
    dumps_report as dumps_perception_report,
    perceive_ir,
    validate_report as validate_perception_report,
)
from pcff_frc import (
    build_phase1_pcff_atom_index,
    resolve_bonded_atom_types_from_frc,
    resolve_bonded_interaction_from_frc,
)
from typing_ir import dumps_ir, parse_file, validate_ir

from .errors import AssignmentReportError, ParameterAssignmentError
from .rules import DEFAULT_RULES_PATH, INTERACTION_KINDS, load_rules, validate_rules
from .signatures import (
    canonicalize_angle,
    canonicalize_bond,
    canonicalize_dihedral,
    canonicalize_improper,
)


SCHEMA_NAME = "pcff_parameter_assignment_report"
SCHEMA_VERSION = 1

PHASE1_REPOSITORY_TUPLE_BACKFILL_RULE_IDS = {
    ("angle", ("h", "c", "h")): "angle_alkane_hc_h",
    ("dihedral", ("h", "c", "c", "h")): "dihedral_alkane_hcch",
    ("improper", ("c_1", "c", "n", "o_1")): "ap5_import_improper_c_1_c_n_o_1_v1_6mBN",
}

PHASE1_PCFF_TUPLE_REMAPS = {
    ("angle", ("n_2", "c_2", "oz")): ("n_2", "c_2", "o_2"),
}


def assign_file(
    path: str | Path,
    *,
    input_format: str | None = None,
    source_id: str | None = None,
    rules_path: str | Path | None = None,
    ruleset: dict | None = None,
) -> dict:
    ir = parse_file(path, input_format=input_format, source_id=source_id)
    perception = perceive_ir(ir)
    typing_report = type_ir(ir, perception=perception)
    return assign_ir(
        ir,
        typing_report=typing_report,
        perception=perception,
        rules_path=rules_path,
        ruleset=ruleset,
    )


def assign_ir(
    ir: dict,
    *,
    typing_report: dict,
    perception: dict | None = None,
    rules_path: str | Path | None = None,
    ruleset: dict | None = None,
) -> dict:
    validate_ir(ir)
    validate_typing_report(typing_report)
    perception_report = perceive_ir(ir) if perception is None else perception
    validate_perception_report(perception_report)

    ir_sha256 = hashlib.sha256(dumps_ir(ir).encode("utf-8")).hexdigest()
    perception_sha256 = hashlib.sha256(dumps_perception_report(perception_report).encode("utf-8")).hexdigest()
    typing_sha256 = hashlib.sha256(dumps_typing_report(typing_report).encode("utf-8")).hexdigest()

    _validate_source_chain(ir, typing_report, ir_sha256=ir_sha256, perception_sha256=perception_sha256)

    active_ruleset = load_rules(rules_path) if ruleset is None else copy.deepcopy(ruleset)
    validate_rules(active_ruleset)

    ir_component = ir["components"][0]
    typing_component = typing_report["components"][0]
    perception_component = perception_report["components"][0]

    if typing_report["typing"]["status"] != "typed":
        raise ParameterAssignmentError(
            "typing_incomplete",
            f"parameter assignment requires typing.status='typed', got {typing_report['typing']['status']!r}",
        )

    typed_atoms = _build_typed_atom_index(typing_component)
    rules_by_kind = _index_rules(active_ruleset)
    rules_by_id = _index_rules_by_id(active_ruleset)
    pcff_atom_index = build_phase1_pcff_atom_index(ir_component, perception_component, typing_component)
    interactions = _build_interactions(ir_component, perception_component, typed_atoms)
    for kind in INTERACTION_KINDS:
        for record in interactions[kind]:
            record["pcff_atom_types"] = [
                pcff_atom_index[index]["pcff_type"] if index in pcff_atom_index else None
                for index in record["atom_indices"]
            ]

    diagnostics = []
    assigned_interactions: dict[str, list[dict]] = {}
    for kind in INTERACTION_KINDS:
        assigned_interactions[kind], kind_diagnostics = _assign_interactions(
            kind,
            interactions[kind],
            rules_by_kind[kind],
            ruleset_id=active_ruleset["ruleset_id"],
            pcff_atom_index=pcff_atom_index,
            rules_by_id=rules_by_id,
        )
        diagnostics.extend(kind_diagnostics)

    status = "missing_parameters" if diagnostics else "assigned"

    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "typed_ir_sha256": ir_sha256,
            "chem_perception_sha256": perception_sha256,
            "typing_report_sha256": typing_sha256,
            "rules_sha256": hashlib.sha256(
                json.dumps(active_ruleset, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "source_id": ir["source"]["source_id"],
            "input_format": ir["source"]["input_format"],
            "ruleset_id": active_ruleset["ruleset_id"],
            "rules_path": _display_rules_path(DEFAULT_RULES_PATH if rules_path is None else Path(rules_path)),
        },
        "parameter_assignment": {
            "status": status,
            "ruleset_id": active_ruleset["ruleset_id"],
            "term_model": active_ruleset["term_model"],
        },
        "components": [
            {
                "component_id": ir_component["component_id"],
                "name": ir_component["name"],
                "atom_count": ir_component["atom_count"],
                "interaction_counts": {
                    kind: len(assigned_interactions[kind])
                    for kind in INTERACTION_KINDS
                },
                "interactions": assigned_interactions,
                "diagnostics": diagnostics,
            }
        ],
    }
    validate_assignment_report(report)
    return report


def dumps_assignment_report(report: dict) -> str:
    validate_assignment_report(report)
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def loads_assignment_report(text: str) -> dict:
    report = json.loads(text)
    validate_assignment_report(report)
    return report


def write_assignment_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_assignment_report(report), encoding="utf-8")


def validate_assignment_report(report: dict) -> None:
    if report.get("schema_name") != SCHEMA_NAME:
        raise AssignmentReportError(
            "invalid_parameter_assignment_report",
            "schema_name must be 'pcff_parameter_assignment_report'",
        )
    if report.get("schema_version") != SCHEMA_VERSION:
        raise AssignmentReportError("invalid_parameter_assignment_report", "Unsupported schema_version")

    source = report.get("source")
    if not isinstance(source, dict):
        raise AssignmentReportError("invalid_parameter_assignment_report", "source must be a mapping")
    for key in {
        "typed_ir_sha256",
        "chem_perception_sha256",
        "typing_report_sha256",
        "rules_sha256",
        "source_id",
        "input_format",
        "ruleset_id",
        "rules_path",
    }:
        if key not in source:
            raise AssignmentReportError("invalid_parameter_assignment_report", f"source.{key} is required")

    components = report.get("components")
    if not isinstance(components, list) or len(components) != 1:
        raise AssignmentReportError(
            "invalid_parameter_assignment_report",
            "parameter assignment report must contain exactly one component",
        )
    component = components[0]
    interactions = component.get("interactions")
    if not isinstance(interactions, dict):
        raise AssignmentReportError("invalid_parameter_assignment_report", "component.interactions must be a mapping")
    for kind in INTERACTION_KINDS:
        if kind not in interactions or not isinstance(interactions[kind], list):
            raise AssignmentReportError(
                "invalid_parameter_assignment_report",
                f"component.interactions.{kind} must be a list",
            )
        for expected_index, record in enumerate(interactions[kind], start=1):
            if record.get("assignment_id") != f"{kind}_{expected_index}":
                raise AssignmentReportError(
                    "invalid_parameter_assignment_report",
                    f"{kind} assignment ids must be contiguous",
                )


def _display_rules_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _validate_source_chain(
    ir: dict,
    typing_report: dict,
    *,
    ir_sha256: str,
    perception_sha256: str,
) -> None:
    if typing_report["source"]["typed_ir_sha256"] != ir_sha256:
        raise ParameterAssignmentError("source_chain_mismatch", "typing report does not match the supplied typed IR")
    if typing_report["source"]["chem_perception_sha256"] != perception_sha256:
        raise ParameterAssignmentError(
            "source_chain_mismatch",
            "typing report does not match the supplied chemical perception report",
        )
    if typing_report["source"]["source_id"] != ir["source"]["source_id"]:
        raise ParameterAssignmentError("source_chain_mismatch", "typing report source_id does not match typed IR")


def _build_typed_atom_index(typing_component: dict) -> dict[int, dict]:
    atoms_by_index = {}
    for atom in typing_component["atoms"]:
        if atom["status"] != "assigned" or atom["assigned_family"] is None:
            raise ParameterAssignmentError(
                "typing_incomplete",
                f"atom {atom['canonical_index']} is not fully assigned in typing report",
            )
        atoms_by_index[atom["canonical_index"]] = atom
    return atoms_by_index


def _index_rules(ruleset: dict) -> dict[str, dict[str, dict]]:
    indexed = {kind: {} for kind in INTERACTION_KINDS}
    for kind in INTERACTION_KINDS:
        for rule in ruleset["interaction_rules"][kind]:
            indexed[kind][rule["canonical_signature"]] = rule
    return indexed


def _index_rules_by_id(ruleset: dict) -> dict[str, dict]:
    indexed = {}
    for kind in INTERACTION_KINDS:
        for rule in ruleset["interaction_rules"][kind]:
            indexed[rule["rule_id"]] = rule
    return indexed


def _build_interactions(ir_component: dict, perception_component: dict, typed_atoms: dict[int, dict]) -> dict[str, list[dict]]:
    ir_atoms = {atom["canonical_index"]: atom for atom in ir_component["atoms"]}
    adjacency = {
        atom["canonical_index"]: list(atom["neighbor_indices"])
        for atom in perception_component["atoms"]
    }

    interactions = {
        "bond": _build_bond_interactions(ir_component, ir_atoms, typed_atoms),
        "angle": _build_angle_interactions(ir_atoms, adjacency, typed_atoms),
        "dihedral": _build_dihedral_interactions(ir_atoms, adjacency, typed_atoms),
        "improper": _build_improper_interactions(ir_atoms, perception_component, typed_atoms),
    }
    for kind in INTERACTION_KINDS:
        interactions[kind].sort(key=lambda record: (record["atom_indices"], record["canonical_signature"]))
    return interactions


def _build_bond_interactions(ir_component: dict, ir_atoms: dict[int, dict], typed_atoms: dict[int, dict]) -> list[dict]:
    interactions = []
    for bond in ir_component["bonds"]:
        atom_indices = list(bond["atom_indices"])
        atom_families = [typed_atoms[index]["assigned_family"] for index in atom_indices]
        ordered_indices, ordered_families, signature = canonicalize_bond(atom_indices, atom_families)
        interactions.append(
            _interaction_record(
                kind="bond",
                atom_indices=ordered_indices,
                atom_families=ordered_families,
                ir_atoms=ir_atoms,
                canonical_signature=signature,
            )
        )
    return interactions


def _build_angle_interactions(
    ir_atoms: dict[int, dict],
    adjacency: dict[int, list[int]],
    typed_atoms: dict[int, dict],
) -> list[dict]:
    interactions = []
    for center_index in sorted(adjacency):
        for left_index, right_index in combinations(sorted(adjacency[center_index]), 2):
            atom_indices = [left_index, center_index, right_index]
            atom_families = [typed_atoms[index]["assigned_family"] for index in atom_indices]
            ordered_indices, ordered_families, signature = canonicalize_angle(atom_indices, atom_families)
            interactions.append(
                _interaction_record(
                    kind="angle",
                    atom_indices=ordered_indices,
                    atom_families=ordered_families,
                    ir_atoms=ir_atoms,
                    canonical_signature=signature,
                )
            )
    return interactions


def _build_dihedral_interactions(
    ir_atoms: dict[int, dict],
    adjacency: dict[int, list[int]],
    typed_atoms: dict[int, dict],
) -> list[dict]:
    interactions = []
    seen = set()
    for left_center in sorted(adjacency):
        for right_center in sorted(index for index in adjacency[left_center] if left_center < index):
            left_neighbors = [index for index in adjacency[left_center] if index != right_center]
            right_neighbors = [index for index in adjacency[right_center] if index != left_center]
            for left_index in left_neighbors:
                for right_index in right_neighbors:
                    if left_index == right_index:
                        continue
                    atom_indices = [left_index, left_center, right_center, right_index]
                    atom_families = [typed_atoms[index]["assigned_family"] for index in atom_indices]
                    ordered_indices, ordered_families, signature = canonicalize_dihedral(atom_indices, atom_families)
                    key = tuple(ordered_indices)
                    if key in seen:
                        continue
                    seen.add(key)
                    interactions.append(
                        _interaction_record(
                            kind="dihedral",
                            atom_indices=ordered_indices,
                            atom_families=ordered_families,
                            ir_atoms=ir_atoms,
                            canonical_signature=signature,
                        )
                    )
    return interactions


def _build_improper_interactions(
    ir_atoms: dict[int, dict],
    perception_component: dict,
    typed_atoms: dict[int, dict],
) -> list[dict]:
    interactions = []
    for atom in perception_component["atoms"]:
        candidate = atom["improper_center_candidate"]
        if "planar_trigonal" not in candidate["kinds"]:
            continue
        neighbor_indices = list(candidate["ordered_neighbor_indices"])
        if len(neighbor_indices) != 3:
            continue
        center_index = atom["canonical_index"]
        neighbor_families = [typed_atoms[index]["assigned_family"] for index in neighbor_indices]
        ordered_indices, ordered_families, signature = canonicalize_improper(
            center_index,
            neighbor_indices,
            typed_atoms[center_index]["assigned_family"],
            neighbor_families,
        )
        interactions.append(
            _interaction_record(
                kind="improper",
                atom_indices=ordered_indices,
                atom_families=ordered_families,
                ir_atoms=ir_atoms,
                canonical_signature=signature,
            )
        )
    return interactions


def _interaction_record(
    *,
    kind: str,
    atom_indices: list[int],
    atom_families: list[str],
    ir_atoms: dict[int, dict],
    canonical_signature: str,
) -> dict:
    return {
        "interaction_kind": kind,
        "atom_indices": atom_indices,
        "source_atom_indices": [ir_atoms[index]["source_index"] for index in atom_indices],
        "source_atom_ids": [ir_atoms[index]["source_atom_id"] for index in atom_indices],
        "atom_families": atom_families,
        "canonical_signature": canonical_signature,
    }


def _assign_interactions(
    kind: str,
    interactions: list[dict],
    rules_by_signature: dict[str, dict],
    *,
    ruleset_id: str,
    pcff_atom_index: dict[int, dict],
    rules_by_id: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    assignments = []
    diagnostics = []
    for record in interactions:
        assignment = dict(record)
        assignment["assignment_id"] = f"{kind}_{len(assignments) + 1}"
        rule = rules_by_signature.get(record["canonical_signature"])
        if rule is None:
            frc_parameters, frc_provenance = resolve_bonded_interaction_from_frc(
                kind,
                record["atom_indices"],
                pcff_atom_index,
            )
            if frc_parameters is not None and frc_provenance is not None:
                assignment["status"] = "assigned"
                assignment["parameter_rule_id"] = None
                assignment["parameters"] = frc_parameters
                assignment["provenance"] = {
                    "ruleset_id": ruleset_id,
                    "rule_id": None,
                    "canonical_signature": record["canonical_signature"],
                    "rule_provenance": frc_provenance,
                }
            else:
                remapped_parameters, remapped_provenance = _resolve_phase1_pcff_tuple_remap(
                    kind,
                    record.get("pcff_atom_types"),
                )
                if remapped_parameters is not None and remapped_provenance is not None:
                    assignment["status"] = "assigned"
                    assignment["parameter_rule_id"] = None
                    assignment["parameters"] = remapped_parameters
                    assignment["provenance"] = {
                        "ruleset_id": ruleset_id,
                        "rule_id": None,
                        "canonical_signature": record["canonical_signature"],
                        "rule_provenance": remapped_provenance,
                    }
                else:
                    backfill_rule = _resolve_phase1_repository_tuple_backfill(
                        kind,
                        record.get("pcff_atom_types"),
                        rules_by_id,
                    )
                    if backfill_rule is not None:
                        assignment["status"] = "assigned"
                        assignment["parameter_rule_id"] = backfill_rule["rule_id"]
                        assignment["parameters"] = copy.deepcopy(backfill_rule["parameters"])
                        assignment["provenance"] = {
                            "ruleset_id": ruleset_id,
                            "rule_id": backfill_rule["rule_id"],
                            "canonical_signature": record["canonical_signature"],
                            "rule_provenance": _phase1_repository_tuple_backfill_provenance(
                                record["pcff_atom_types"],
                                backfill_rule,
                            ),
                        }
                    else:
                        assignment["status"] = "missing_parameter"
                        assignment["parameter_rule_id"] = None
                        assignment["parameters"] = None
                        assignment["provenance"] = None
                        diagnostics.append(
                            {
                                "scope": kind,
                                "code": "missing_parameter",
                                "atom_indices": record["atom_indices"],
                                "source_atom_indices": record["source_atom_indices"],
                                "canonical_signature": record["canonical_signature"],
                                "pcff_atom_types": record.get("pcff_atom_types"),
                                "message": f"No {kind} parameter rule matched {record['canonical_signature']}",
                            }
                        )
        else:
            assignment["status"] = "assigned"
            assignment["parameter_rule_id"] = rule["rule_id"]
            assignment["parameters"] = copy.deepcopy(rule["parameters"])
            assignment["provenance"] = {
                "ruleset_id": ruleset_id,
                "rule_id": rule["rule_id"],
                "canonical_signature": rule["canonical_signature"],
                "rule_provenance": copy.deepcopy(rule["provenance"]),
            }
        assignments.append(assignment)
    return assignments, diagnostics


def _resolve_phase1_pcff_tuple_remap(
    kind: str,
    pcff_atom_types: list[str] | None,
) -> tuple[dict | None, dict | None]:
    if pcff_atom_types is None:
        return None, None
    matched_tuple = tuple(pcff_atom_types)
    remapped_tuple = PHASE1_PCFF_TUPLE_REMAPS.get((kind, matched_tuple))
    if remapped_tuple is None:
        return None, None
    parameters, base_provenance = resolve_bonded_atom_types_from_frc(kind, list(remapped_tuple))
    if parameters is None or base_provenance is None:
        return None, None
    return (
        parameters,
        {
            "source_kind": "phase1_pcff_tuple_remap",
            "source_file": "frc_file/pcff.frc",
            "source_resolution": "phase1_tuple_remap",
            "matched_pcff_types": list(matched_tuple),
            "remapped_pcff_types": list(remapped_tuple),
            "base_provenance": copy.deepcopy(base_provenance),
        },
    )


def _resolve_phase1_repository_tuple_backfill(
    kind: str,
    pcff_atom_types: list[str] | None,
    rules_by_id: dict[str, dict],
) -> dict | None:
    if pcff_atom_types is None:
        return None
    rule_id = PHASE1_REPOSITORY_TUPLE_BACKFILL_RULE_IDS.get((kind, tuple(pcff_atom_types)))
    if rule_id is None:
        return None
    return rules_by_id.get(rule_id)


def _phase1_repository_tuple_backfill_provenance(pcff_atom_types: list[str], source_rule: dict) -> dict:
    return {
        "source_kind": "repository_tuple_backfill_rule",
        "source_file": "rules/pcff_parameters.json",
        "source_resolution": "phase1_exact_pcff_tuple_backfill",
        "matched_pcff_types": list(pcff_atom_types),
        "source_rule_id": source_rule["rule_id"],
        "source_rule_canonical_signature": source_rule["canonical_signature"],
        "base_provenance": copy.deepcopy(source_rule["provenance"]),
    }
