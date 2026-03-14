from __future__ import annotations

import json
from pathlib import Path

from .errors import RuleSchemaError


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = REPO_ROOT / "rules" / "pcff_nonbonded.json"
SCHEMA_NAME = "pcff_nonbonded_rules"
SCHEMA_VERSION = 1


def load_rules(path: str | Path | None = None) -> dict:
    rules_path = DEFAULT_RULES_PATH if path is None else Path(path)
    with rules_path.open("r", encoding="utf-8") as handle:
        ruleset = json.load(handle)
    validate_rules(ruleset)
    return ruleset


def dump_rules_json(ruleset: dict) -> str:
    validate_rules(ruleset)
    return json.dumps(ruleset, indent=2, sort_keys=True) + "\n"


def validate_rules(ruleset: dict) -> None:
    if ruleset.get("schema_name") != SCHEMA_NAME:
        raise RuleSchemaError("invalid_rule_schema", "schema_name must be 'pcff_nonbonded_rules'")
    if ruleset.get("schema_version") != SCHEMA_VERSION:
        raise RuleSchemaError("invalid_rule_schema", "Unsupported rule schema_version")
    for key in {"ruleset_id", "pair_model", "atom_type_rules", "pair_overrides"}:
        if key not in ruleset:
            raise RuleSchemaError("invalid_rule_schema", f"Missing ruleset field {key!r}")

    pair_model = ruleset["pair_model"]
    if not isinstance(pair_model, dict):
        raise RuleSchemaError("invalid_rule_schema", "pair_model must be a mapping")
    for key in {
        "mixing_rule",
        "neutral_pair_style",
        "charged_pair_style",
        "repulsion_power",
        "dispersion_power",
        "special_bonds_profile",
        "charge_source_policy",
    }:
        if key not in pair_model:
            raise RuleSchemaError("invalid_rule_schema", f"pair_model.{key} is required")
    special_bonds = pair_model["special_bonds_profile"]
    if not isinstance(special_bonds, dict):
        raise RuleSchemaError("invalid_rule_schema", "pair_model.special_bonds_profile must be a mapping")
    for key in {"profile_id", "value", "lj_weights", "coul_weights", "angle", "dihedral"}:
        if key not in special_bonds:
            raise RuleSchemaError("invalid_rule_schema", f"special_bonds_profile.{key} is required")

    atom_rules = ruleset["atom_type_rules"]
    if not isinstance(atom_rules, list) or not atom_rules:
        raise RuleSchemaError("invalid_rule_schema", "atom_type_rules must be a non-empty list")
    families = set()
    nonbonded_types = set()
    rule_ids = set()
    for rule in atom_rules:
        required = {"rule_id", "family", "nonbonded_type", "self_parameters", "provenance"}
        if not required.issubset(rule):
            missing = sorted(required - set(rule))
            raise RuleSchemaError(
                "invalid_rule_schema",
                f"atom_type_rule {rule.get('rule_id', '<unknown>')} is missing required fields {missing}",
            )
        if rule["rule_id"] in rule_ids:
            raise RuleSchemaError("invalid_rule_schema", f"Duplicate rule_id {rule['rule_id']!r}")
        rule_ids.add(rule["rule_id"])
        if rule["family"] in families:
            raise RuleSchemaError("invalid_rule_schema", f"Duplicate nonbonded family rule {rule['family']!r}")
        families.add(rule["family"])
        if rule["nonbonded_type"] in nonbonded_types:
            raise RuleSchemaError("invalid_rule_schema", f"Duplicate nonbonded_type {rule['nonbonded_type']!r}")
        nonbonded_types.add(rule["nonbonded_type"])
        self_parameters = rule["self_parameters"]
        if not isinstance(self_parameters, dict):
            raise RuleSchemaError("invalid_rule_schema", f"self_parameters for {rule['family']!r} must be a mapping")
        for key in {"epsilon_kcal_mol", "sigma_angstrom"}:
            if key not in self_parameters:
                raise RuleSchemaError(
                    "invalid_rule_schema",
                    f"self_parameters for {rule['family']!r} must include {key}",
                )

    overrides = ruleset["pair_overrides"]
    if not isinstance(overrides, list):
        raise RuleSchemaError("invalid_rule_schema", "pair_overrides must be a list")
    override_keys = set()
    for rule in overrides:
        required = {"rule_id", "canonical_family_pair", "scope", "parameters", "provenance"}
        if not required.issubset(rule):
            missing = sorted(required - set(rule))
            raise RuleSchemaError(
                "invalid_rule_schema",
                f"pair_override {rule.get('rule_id', '<unknown>')} is missing required fields {missing}",
            )
        if rule["rule_id"] in rule_ids:
            raise RuleSchemaError("invalid_rule_schema", f"Duplicate rule_id {rule['rule_id']!r}")
        rule_ids.add(rule["rule_id"])
        if rule["scope"] not in {"normal", "pair14", "both"}:
            raise RuleSchemaError(
                "invalid_rule_schema",
                f"pair_override {rule['rule_id']} must declare scope normal, pair14, or both",
            )
        key = (rule["canonical_family_pair"], rule["scope"])
        if key in override_keys:
            raise RuleSchemaError(
                "invalid_rule_schema",
                f"Duplicate pair override for {rule['canonical_family_pair']!r} scope {rule['scope']!r}",
            )
        override_keys.add(key)
        if not isinstance(rule["parameters"], dict):
            raise RuleSchemaError("invalid_rule_schema", f"pair_override {rule['rule_id']} parameters must be a mapping")
        for parameter in {"epsilon_kcal_mol", "sigma_angstrom"}:
            if parameter not in rule["parameters"]:
                raise RuleSchemaError(
                    "invalid_rule_schema",
                    f"pair_override {rule['rule_id']} must include {parameter}",
                )
