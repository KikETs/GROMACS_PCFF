from __future__ import annotations

import copy
import hashlib
import json
from itertools import combinations_with_replacement
from pathlib import Path

from atom_typing import dumps_typing_report, type_ir, validate_typing_report
from chem_perception import perceive_ir
from parameter_assignment import (
    assign_ir as assign_bonded_ir,
    dumps_assignment_report as dumps_bonded_report,
    validate_assignment_report as validate_bonded_report,
)
from pcff_frc import (
    build_phase1_pcff_atom_index,
    resolve_nonbonded_atom_from_frc,
    resolve_phase1_bond_increment_charges,
)
from typing_ir import dumps_ir, parse_file, validate_ir

from .errors import AssignmentReportError, NonbondedAssignmentError
from .pairs import (
    canonical_family_pair,
    class2_normal_coefficients,
    class2_pair14_coefficients,
    sixthpower_mix,
)
from .rules import DEFAULT_RULES_PATH, load_rules, validate_rules


SCHEMA_NAME = "pcff_nonbonded_assignment_report"
SCHEMA_VERSION = 1


def assign_file(
    path: str | Path,
    *,
    input_format: str | None = None,
    source_id: str | None = None,
    rules_path: str | Path | None = None,
    ruleset: dict | None = None,
) -> dict:
    ir = parse_file(path, input_format=input_format, source_id=source_id)
    typing_report = type_ir(ir)
    bonded_report = assign_bonded_ir(ir, typing_report=typing_report)
    return assign_ir(
        ir,
        typing_report=typing_report,
        bonded_report=bonded_report,
        rules_path=rules_path,
        ruleset=ruleset,
    )


def assign_ir(
    ir: dict,
    *,
    typing_report: dict,
    bonded_report: dict,
    rules_path: str | Path | None = None,
    ruleset: dict | None = None,
) -> dict:
    validate_ir(ir)
    validate_typing_report(typing_report)
    validate_bonded_report(bonded_report)

    ir_sha256 = hashlib.sha256(dumps_ir(ir).encode("utf-8")).hexdigest()
    typing_sha256 = hashlib.sha256(dumps_typing_report(typing_report).encode("utf-8")).hexdigest()
    bonded_sha256 = hashlib.sha256(dumps_bonded_report(bonded_report).encode("utf-8")).hexdigest()
    _validate_source_chain(
        ir,
        typing_report,
        bonded_report,
        ir_sha256=ir_sha256,
        typing_sha256=typing_sha256,
    )

    active_ruleset = load_rules(rules_path) if ruleset is None else copy.deepcopy(ruleset)
    validate_rules(active_ruleset)

    if typing_report["typing"]["status"] != "typed":
        raise NonbondedAssignmentError(
            "typing_incomplete",
            f"nonbonded assignment requires typing.status='typed', got {typing_report['typing']['status']!r}",
        )
    if bonded_report["parameter_assignment"]["status"] != "assigned":
        raise NonbondedAssignmentError(
            "bonded_assignment_incomplete",
            "nonbonded assignment requires a complete PT4 bonded parameter report",
        )

    ir_component = ir["components"][0]
    typing_component = typing_report["components"][0]
    bonded_component = bonded_report["components"][0]
    perception_report = perceive_ir(ir)

    typed_atoms = {atom["canonical_index"]: atom for atom in typing_component["atoms"]}
    ir_atoms = {atom["canonical_index"]: atom for atom in ir_component["atoms"]}
    pcff_atom_index = build_phase1_pcff_atom_index(
        ir_component,
        perception_report["components"][0],
        typing_component,
    )
    charge_assignments, charge_diagnostics = resolve_phase1_bond_increment_charges(ir_component, pcff_atom_index)
    atom_rule_index, pair_override_index = _index_rules(active_ruleset)

    atom_records, atom_diagnostics = _assign_atoms(
        ir_atoms,
        typed_atoms,
        atom_rule_index,
        active_ruleset["ruleset_id"],
        pcff_atom_index=pcff_atom_index,
        charge_assignments=charge_assignments,
    )
    atom_records_by_index = {record["canonical_index"]: record for record in atom_records}

    pair_style_kind = _pair_style_kind(atom_records, active_ruleset)
    pair_classes, pair_diagnostics = _build_pair_classes(
        atom_records,
        pair_override_index=pair_override_index,
        ruleset=active_ruleset,
    )
    pair_classes_by_signature = {
        record["canonical_family_pair"]: record
        for record in pair_classes
        if record["status"] == "assigned"
    }

    exclusions = _build_exclusions(
        bonded_component,
        atom_records_by_index=atom_records_by_index,
        ruleset=active_ruleset,
    )
    pair14_records, pair14_diagnostics = _build_pair14_pairs(
        bonded_component,
        atom_records_by_index=atom_records_by_index,
        pair_classes_by_signature=pair_classes_by_signature,
        pair_override_index=pair_override_index,
        ruleset=active_ruleset,
    )

    diagnostics = [*charge_diagnostics, *atom_diagnostics, *pair_diagnostics, *pair14_diagnostics]
    status = "missing_parameters" if diagnostics else "assigned"

    export_metadata = {
        "gromacs": {
            "combination_rule": active_ruleset["pair_model"]["mixing_rule"],
            "repulsion_power": active_ruleset["pair_model"]["repulsion_power"],
            "dispersion_power": active_ruleset["pair_model"]["dispersion_power"],
            "explicit_pair14_required": True,
            "pair_generation_mode": "use_explicit_pairs_section",
            "exclusion_generation_mode": "use_explicit_exclusions_section",
        },
        "lammps": {
            "pair_style_kind": pair_style_kind,
            "pair_modify": "mix sixthpower",
            "special_bonds": active_ruleset["pair_model"]["special_bonds_profile"]["value"],
            "requires_kspace": pair_style_kind.endswith("/coul/long"),
            "pair_coeff_policy": "explicit_self_only_with_optional_cross_overrides",
        },
    }

    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "typed_ir_sha256": ir_sha256,
            "typing_report_sha256": typing_sha256,
            "bonded_assignment_sha256": bonded_sha256,
            "rules_sha256": hashlib.sha256(
                json.dumps(active_ruleset, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "source_id": ir["source"]["source_id"],
            "input_format": ir["source"]["input_format"],
            "ruleset_id": active_ruleset["ruleset_id"],
            "rules_path": _display_rules_path(DEFAULT_RULES_PATH if rules_path is None else Path(rules_path)),
        },
        "nonbonded_assignment": {
            "status": status,
            "ruleset_id": active_ruleset["ruleset_id"],
            "mixing_rule": active_ruleset["pair_model"]["mixing_rule"],
            "special_bonds_profile_id": active_ruleset["pair_model"]["special_bonds_profile"]["profile_id"],
            "pair_style_kind": pair_style_kind,
        },
        "components": [
            {
                "component_id": ir_component["component_id"],
                "name": ir_component["name"],
                "atom_count": ir_component["atom_count"],
                "atoms": atom_records,
                "pair_classes": pair_classes,
                "exclusions": exclusions,
                "pair14": pair14_records,
                "export_metadata": export_metadata,
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
            "invalid_nonbonded_assignment_report",
            "schema_name must be 'pcff_nonbonded_assignment_report'",
        )
    if report.get("schema_version") != SCHEMA_VERSION:
        raise AssignmentReportError("invalid_nonbonded_assignment_report", "Unsupported schema_version")
    source = report.get("source")
    if not isinstance(source, dict):
        raise AssignmentReportError("invalid_nonbonded_assignment_report", "source must be a mapping")
    for key in {
        "typed_ir_sha256",
        "typing_report_sha256",
        "bonded_assignment_sha256",
        "rules_sha256",
        "source_id",
        "input_format",
        "ruleset_id",
        "rules_path",
    }:
        if key not in source:
            raise AssignmentReportError("invalid_nonbonded_assignment_report", f"source.{key} is required")
    components = report.get("components")
    if not isinstance(components, list) or len(components) != 1:
        raise AssignmentReportError(
            "invalid_nonbonded_assignment_report",
            "nonbonded assignment report must contain exactly one component",
        )
    component = components[0]
    for collection_name in ("atoms", "pair_classes", "exclusions", "pair14"):
        if not isinstance(component.get(collection_name), list):
            raise AssignmentReportError(
                "invalid_nonbonded_assignment_report",
                f"component.{collection_name} must be a list",
            )


def _display_rules_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _validate_source_chain(
    ir: dict,
    typing_report: dict,
    bonded_report: dict,
    *,
    ir_sha256: str,
    typing_sha256: str,
) -> None:
    if typing_report["source"]["typed_ir_sha256"] != ir_sha256:
        raise NonbondedAssignmentError("source_chain_mismatch", "typing report does not match the supplied typed IR")
    if bonded_report["source"]["typed_ir_sha256"] != ir_sha256:
        raise NonbondedAssignmentError("source_chain_mismatch", "bonded report does not match the supplied typed IR")
    if bonded_report["source"]["typing_report_sha256"] != typing_sha256:
        raise NonbondedAssignmentError(
            "source_chain_mismatch",
            "bonded report does not match the supplied typing report",
        )
    if ir["source"]["source_id"] != typing_report["source"]["source_id"]:
        raise NonbondedAssignmentError("source_chain_mismatch", "source_id mismatch between IR and typing report")


def _index_rules(ruleset: dict) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    atom_rule_index = {rule["family"]: rule for rule in ruleset["atom_type_rules"]}
    pair_override_index: dict[tuple[str, str], dict] = {}
    for rule in ruleset["pair_overrides"]:
        pair_override_index[(rule["canonical_family_pair"], rule["scope"])] = rule
    return atom_rule_index, pair_override_index


def _assign_atoms(
    ir_atoms: dict[int, dict],
    typed_atoms: dict[int, dict],
    atom_rule_index: dict[str, dict],
    ruleset_id: str,
    *,
    pcff_atom_index: dict[int, dict],
    charge_assignments: dict[int, dict] | None,
) -> tuple[list[dict], list[dict]]:
    records = []
    diagnostics = []
    for canonical_index in sorted(ir_atoms):
        ir_atom = ir_atoms[canonical_index]
        typed_atom = typed_atoms[canonical_index]
        family = typed_atom["assigned_family"]
        rule = atom_rule_index.get(family)
        charge_source, charge_value, charge_provenance = _resolve_charge(
            ir_atom,
            canonical_index=canonical_index,
            charge_assignments=charge_assignments,
        )

        record = {
            "canonical_index": canonical_index,
            "source_index": ir_atom["source_index"],
            "source_atom_id": ir_atom["source_atom_id"],
            "element": ir_atom["element"],
            "assigned_family": family,
            "charge_assignment": {
                "source": charge_source,
                "value": charge_value,
                "provenance": charge_provenance,
            },
        }

        if rule is None:
            pcff_payload = pcff_atom_index.get(canonical_index)
            if pcff_payload is None:
                record["status"] = "missing_parameter"
                record["nonbonded_type"] = None
                record["self_parameters"] = None
                record["provenance"] = None
                diagnostics.append(
                    {
                        "scope": "atom",
                        "code": "missing_nonbonded_atom_type",
                        "atom_index": canonical_index,
                        "source_index": ir_atom["source_index"],
                        "family": family,
                        "message": f"No nonbonded rule matched atom family {family!r}",
                    }
                )
            else:
                frc_params, frc_provenance = resolve_nonbonded_atom_from_frc(pcff_payload["pcff_type"])
                if frc_params is None or frc_provenance is None:
                    record["status"] = "missing_parameter"
                    record["nonbonded_type"] = None
                    record["self_parameters"] = None
                    record["provenance"] = None
                    diagnostics.append(
                        {
                            "scope": "atom",
                            "code": "missing_nonbonded_atom_type",
                            "atom_index": canonical_index,
                            "source_index": ir_atom["source_index"],
                            "family": family,
                            "pcff_type": pcff_payload["pcff_type"],
                            "message": f"No PCFF nonbonded term matched atom family {family!r}",
                        }
                    )
                else:
                    sigma = float(frc_params["sigma_angstrom"])
                    epsilon = float(frc_params["epsilon_kcal_mol"])
                    resolved_type = frc_provenance["resolved_key"][0]
                    record["status"] = "assigned"
                    record["nonbonded_type"] = resolved_type
                    record["self_parameters"] = {
                        "sigma_angstrom": sigma,
                        "epsilon_kcal_mol": epsilon,
                        "normal_coefficients": class2_normal_coefficients(epsilon, sigma),
                    }
                    record["provenance"] = {
                        "ruleset_id": ruleset_id,
                        "rule_id": None,
                        "rule_provenance": copy.deepcopy(frc_provenance),
                        "pcff_atom_type": pcff_payload["pcff_type"],
                    }
        else:
            sigma = float(rule["self_parameters"]["sigma_angstrom"])
            epsilon = float(rule["self_parameters"]["epsilon_kcal_mol"])
            record["status"] = "assigned"
            record["nonbonded_type"] = rule["nonbonded_type"]
            record["self_parameters"] = {
                "sigma_angstrom": sigma,
                "epsilon_kcal_mol": epsilon,
                "normal_coefficients": class2_normal_coefficients(epsilon, sigma),
            }
            record["provenance"] = {
                "ruleset_id": ruleset_id,
                "rule_id": rule["rule_id"],
                "rule_provenance": copy.deepcopy(rule["provenance"]),
            }
        records.append(record)
    return records, diagnostics


def _resolve_charge(
    ir_atom: dict,
    *,
    canonical_index: int,
    charge_assignments: dict[int, dict] | None,
) -> tuple[str, float, dict | None]:
    if charge_assignments is not None and canonical_index in charge_assignments:
        charge_record = charge_assignments[canonical_index]
        return charge_record["source"], float(charge_record["value"]), copy.deepcopy(charge_record["provenance"])
    if ir_atom["partial_charge"] is not None:
        return "partial_charge", float(ir_atom["partial_charge"]), None
    if ir_atom["formal_charge"] is not None:
        return "formal_charge", float(ir_atom["formal_charge"]), None
    raise NonbondedAssignmentError(
        "missing_charge",
        f"Atom {ir_atom['canonical_index']} does not declare partial_charge or formal_charge",
    )


def _pair_style_kind(atom_records: list[dict], ruleset: dict) -> str:
    has_charge = any(abs(atom["charge_assignment"]["value"]) > 0.0 for atom in atom_records)
    return (
        ruleset["pair_model"]["charged_pair_style"]
        if has_charge
        else ruleset["pair_model"]["neutral_pair_style"]
    )


def _build_pair_classes(
    atom_records: list[dict],
    *,
    pair_override_index: dict[tuple[str, str], dict],
    ruleset: dict,
) -> tuple[list[dict], list[dict]]:
    diagnostics = []
    records = []
    families = sorted({atom["assigned_family"] for atom in atom_records})
    self_parameters = {
        atom["assigned_family"]: atom["self_parameters"]
        for atom in atom_records
        if atom["status"] == "assigned"
    }

    for family_a, family_b in combinations_with_replacement(families, 2):
        ordered_families, pair_signature = canonical_family_pair(family_a, family_b)
        record = {
            "pair_class_id": f"pair_class_{len(records) + 1}",
            "canonical_family_pair": pair_signature,
            "atom_families": ordered_families,
        }

        parameters, source = _resolve_pair_parameters(
            family_a,
            family_b,
            self_parameters=self_parameters,
            pair_override_index=pair_override_index,
            scope="normal",
            ruleset=ruleset,
        )
        if parameters is None:
            record["status"] = "missing_parameter"
            record["parameter_source"] = None
            record["parameters"] = None
            record["provenance"] = None
            diagnostics.append(
                {
                    "scope": "pair_class",
                    "code": "missing_nonbonded_pair_parameter",
                    "canonical_family_pair": pair_signature,
                    "message": f"Could not resolve normal nonbonded parameters for {pair_signature}",
                }
            )
        else:
            record["status"] = "assigned"
            record["parameter_source"] = source["parameter_source"]
            record["parameters"] = parameters
            record["provenance"] = source["provenance"]
        records.append(record)
    return records, diagnostics


def _resolve_pair_parameters(
    family_a: str,
    family_b: str,
    *,
    self_parameters: dict[str, dict],
    pair_override_index: dict[tuple[str, str], dict],
    scope: str,
    ruleset: dict,
) -> tuple[dict | None, dict | None]:
    ordered_families, pair_signature = canonical_family_pair(family_a, family_b)
    override = pair_override_index.get((pair_signature, scope)) or pair_override_index.get((pair_signature, "both"))
    if override is not None:
        sigma = float(override["parameters"]["sigma_angstrom"])
        epsilon = float(override["parameters"]["epsilon_kcal_mol"])
        return (
            {
                "sigma_angstrom": sigma,
                "epsilon_kcal_mol": epsilon,
                "normal_coefficients": class2_normal_coefficients(epsilon, sigma),
                "pair14_coefficients": class2_pair14_coefficients(epsilon, sigma),
            },
            {
                "parameter_source": "override",
                "provenance": {
                    "ruleset_id": ruleset["ruleset_id"],
                    "rule_id": override["rule_id"],
                    "rule_provenance": copy.deepcopy(override["provenance"]),
                },
            },
        )
    if family_a not in self_parameters or family_b not in self_parameters:
        return None, None
    mixed = sixthpower_mix(
        self_parameters[family_a]["sigma_angstrom"],
        self_parameters[family_a]["epsilon_kcal_mol"],
        self_parameters[family_b]["sigma_angstrom"],
        self_parameters[family_b]["epsilon_kcal_mol"],
    )
    sigma = mixed["sigma_angstrom"]
    epsilon = mixed["epsilon_kcal_mol"]
    return (
        {
            "sigma_angstrom": sigma,
            "epsilon_kcal_mol": epsilon,
            "normal_coefficients": class2_normal_coefficients(epsilon, sigma),
            "pair14_coefficients": class2_pair14_coefficients(epsilon, sigma),
        },
        {
            "parameter_source": "mixed",
            "provenance": {
                "ruleset_id": ruleset["ruleset_id"],
                "rule_id": None,
                "mixing_rule": ruleset["pair_model"]["mixing_rule"],
            },
        },
    )


def _build_exclusions(
    bonded_component: dict,
    *,
    atom_records_by_index: dict[int, dict],
    ruleset: dict,
) -> list[dict]:
    special_bonds = ruleset["pair_model"]["special_bonds_profile"]
    pair_map: dict[tuple[int, int], dict] = {}

    for record in bonded_component["interactions"]["bond"]:
        _register_pair_relation(
            pair_map,
            record,
            relation="1-2",
            source_field="bond_assignment_ids",
        )
    for record in bonded_component["interactions"]["angle"]:
        _register_pair_relation(
            pair_map,
            {
                "assignment_id": record["assignment_id"],
                "atom_indices": [record["atom_indices"][0], record["atom_indices"][2]],
            },
            relation="1-3",
            source_field="angle_assignment_ids",
        )

    exclusions = []
    for atom_pair, relation_record in sorted(pair_map.items()):
        if relation_record["relation"] not in {"1-2", "1-3"}:
            continue
        exclusions.append(
            {
                "exclusion_id": f"exclusion_{len(exclusions) + 1}",
                "atom_indices": list(atom_pair),
                "source_atom_indices": [atom_records_by_index[index]["source_index"] for index in atom_pair],
                "source_atom_ids": [atom_records_by_index[index]["source_atom_id"] for index in atom_pair],
                "atom_families": [atom_records_by_index[index]["assigned_family"] for index in atom_pair],
                "topological_relation": relation_record["relation"],
                "lj_scale": float(special_bonds["lj_weights"][0 if relation_record["relation"] == "1-2" else 1]),
                "coul_scale": float(special_bonds["coul_weights"][0 if relation_record["relation"] == "1-2" else 1]),
                "source_assignment_ids": relation_record["source_assignment_ids"],
            }
        )
    return exclusions


def _register_pair_relation(pair_map: dict[tuple[int, int], dict], record: dict, *, relation: str, source_field: str) -> None:
    pair = tuple(sorted(record["atom_indices"]))
    precedence = {"1-2": 1, "1-3": 2, "1-4": 3}
    current = pair_map.get(pair)
    if current is None or precedence[relation] < precedence[current["relation"]]:
        pair_map[pair] = {
            "relation": relation,
            "source_assignment_ids": [record["assignment_id"]],
        }
        return
    if precedence[relation] == precedence[current["relation"]]:
        current["source_assignment_ids"].append(record["assignment_id"])
        current["source_assignment_ids"].sort()


def _build_pair14_pairs(
    bonded_component: dict,
    *,
    atom_records_by_index: dict[int, dict],
    pair_classes_by_signature: dict[str, dict],
    pair_override_index: dict[tuple[str, str], dict],
    ruleset: dict,
) -> tuple[list[dict], list[dict]]:
    diagnostics = []
    special_bonds = ruleset["pair_model"]["special_bonds_profile"]

    excluded_pairs = {
        tuple(sorted(record["atom_indices"]))
        for record in bonded_component["interactions"]["bond"]
    }
    excluded_pairs.update(
        tuple(sorted((record["atom_indices"][0], record["atom_indices"][2])))
        for record in bonded_component["interactions"]["angle"]
    )

    pair14_sources: dict[tuple[int, int], list[str]] = {}
    for record in bonded_component["interactions"]["dihedral"]:
        pair = tuple(sorted((record["atom_indices"][0], record["atom_indices"][3])))
        if pair in excluded_pairs:
            continue
        pair14_sources.setdefault(pair, [])
        pair14_sources[pair].append(record["assignment_id"])

    records = []
    self_parameters = {
        atom["assigned_family"]: atom["self_parameters"]
        for atom in atom_records_by_index.values()
        if atom["status"] == "assigned"
    }
    for atom_pair, source_assignment_ids in sorted(pair14_sources.items()):
        family_a = atom_records_by_index[atom_pair[0]]["assigned_family"]
        family_b = atom_records_by_index[atom_pair[1]]["assigned_family"]
        ordered_families, pair_signature = canonical_family_pair(family_a, family_b)
        pair_parameters, source = _resolve_pair_parameters(
            family_a,
            family_b,
            self_parameters=self_parameters,
            pair_override_index=pair_override_index,
            scope="pair14",
            ruleset=ruleset,
        )
        record = {
            "pair14_id": f"pair14_{len(records) + 1}",
            "atom_indices": list(atom_pair),
            "source_atom_indices": [atom_records_by_index[index]["source_index"] for index in atom_pair],
            "source_atom_ids": [atom_records_by_index[index]["source_atom_id"] for index in atom_pair],
            "atom_families": ordered_families,
            "canonical_family_pair": pair_signature,
            "topological_relation": "1-4",
            "lj_scale": float(special_bonds["lj_weights"][2]),
            "coul_scale": float(special_bonds["coul_weights"][2]),
            "source_dihedral_assignment_ids": sorted(source_assignment_ids),
        }
        if pair_parameters is None:
            record["status"] = "missing_parameter"
            record["parameter_source"] = None
            record["parameters"] = None
            record["provenance"] = None
            diagnostics.append(
                {
                    "scope": "pair14",
                    "code": "missing_pair14_parameter",
                    "atom_indices": list(atom_pair),
                    "canonical_family_pair": pair_signature,
                    "message": f"Could not resolve 1-4 parameters for {pair_signature}",
                }
            )
        else:
            record["status"] = "assigned"
            record["parameter_source"] = source["parameter_source"]
            record["parameters"] = {
                "sigma_angstrom": pair_parameters["sigma_angstrom"],
                "epsilon_kcal_mol": pair_parameters["epsilon_kcal_mol"],
                "pair14_coefficients": pair_parameters["pair14_coefficients"],
            }
            record["provenance"] = source["provenance"]
        records.append(record)
    return records, diagnostics
