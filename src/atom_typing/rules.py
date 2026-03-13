from __future__ import annotations

import json
from pathlib import Path

from .errors import RuleSchemaError


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = REPO_ROOT / "rules" / "pcff_atom_types.json"
SCHEMA_NAME = "pcff_atom_type_rules"
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
        raise RuleSchemaError("invalid_rule_schema", "schema_name must be 'pcff_atom_type_rules'")
    if ruleset.get("schema_version") != SCHEMA_VERSION:
        raise RuleSchemaError("invalid_rule_schema", "Unsupported rule schema_version")
    for key in {"ruleset_id", "supported_elements", "component_rules", "atom_type_rules"}:
        if key not in ruleset:
            raise RuleSchemaError("invalid_rule_schema", f"Missing ruleset field {key!r}")

    if not isinstance(ruleset["supported_elements"], list) or not ruleset["supported_elements"]:
        raise RuleSchemaError("invalid_rule_schema", "supported_elements must be a non-empty list")

    rule_ids = set()
    for collection_name, required_fields in (
        ("component_rules", {"rule_id", "kind", "precedence", "predicate"}),
        ("atom_type_rules", {"rule_id", "family", "component_family", "precedence", "match"}),
    ):
        rules = ruleset.get(collection_name)
        if not isinstance(rules, list) or not rules:
            raise RuleSchemaError("invalid_rule_schema", f"{collection_name} must be a non-empty list")
        for rule in rules:
            if not required_fields.issubset(rule):
                missing = sorted(required_fields - set(rule))
                raise RuleSchemaError(
                    "invalid_rule_schema",
                    f"Rule {rule.get('rule_id', '<unknown>')} is missing required fields {missing}",
                )
            rule_id = rule["rule_id"]
            if rule_id in rule_ids:
                raise RuleSchemaError("invalid_rule_schema", f"Duplicate rule_id {rule_id!r}")
            rule_ids.add(rule_id)

    for rule in ruleset["component_rules"]:
        if rule["kind"] not in {"support", "reject"}:
            raise RuleSchemaError("invalid_rule_schema", f"Unsupported component rule kind {rule['kind']!r}")
        predicate = rule["predicate"]
        if not isinstance(predicate, dict) or "builtin" not in predicate:
            raise RuleSchemaError("invalid_rule_schema", f"Component rule {rule['rule_id']} must declare predicate.builtin")
        if rule["kind"] == "support" and "family" not in rule:
            raise RuleSchemaError("invalid_rule_schema", f"Support rule {rule['rule_id']} must declare family")
        if rule["kind"] == "reject" and "failure_code" not in rule:
            raise RuleSchemaError("invalid_rule_schema", f"Reject rule {rule['rule_id']} must declare failure_code")

    for rule in ruleset["atom_type_rules"]:
        match = rule["match"]
        conditions = match.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            raise RuleSchemaError("invalid_rule_schema", f"Atom rule {rule['rule_id']} must declare non-empty match.conditions")
        for condition in conditions:
            if "path" not in condition:
                raise RuleSchemaError("invalid_rule_schema", f"Atom rule {rule['rule_id']} has a condition without path")
            operators = [operator for operator in ("equals", "in", "contains", "contains_all") if operator in condition]
            if len(operators) != 1:
                raise RuleSchemaError(
                    "invalid_rule_schema",
                    f"Atom rule {rule['rule_id']} condition {condition['path']!r} must declare exactly one operator",
                )
