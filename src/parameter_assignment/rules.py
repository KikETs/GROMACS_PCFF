from __future__ import annotations

import json
from pathlib import Path

from .errors import RuleSchemaError


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = REPO_ROOT / "rules" / "pcff_parameters.json"
SCHEMA_NAME = "pcff_bonded_parameter_rules"
SCHEMA_VERSION = 1
INTERACTION_KINDS = ("bond", "angle", "dihedral", "improper")


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
        raise RuleSchemaError("invalid_rule_schema", "schema_name must be 'pcff_bonded_parameter_rules'")
    if ruleset.get("schema_version") != SCHEMA_VERSION:
        raise RuleSchemaError("invalid_rule_schema", "Unsupported rule schema_version")
    for key in {"ruleset_id", "term_model", "interaction_rules"}:
        if key not in ruleset:
            raise RuleSchemaError("invalid_rule_schema", f"Missing ruleset field {key!r}")

    interaction_rules = ruleset["interaction_rules"]
    if not isinstance(interaction_rules, dict):
        raise RuleSchemaError("invalid_rule_schema", "interaction_rules must be a mapping")

    rule_ids = set()
    for kind in INTERACTION_KINDS:
        rules = interaction_rules.get(kind)
        if not isinstance(rules, list):
            raise RuleSchemaError("invalid_rule_schema", f"interaction_rules.{kind} must be a list")
        signatures = set()
        for rule in rules:
            required = {"rule_id", "canonical_signature", "parameters", "provenance"}
            if not required.issubset(rule):
                missing = sorted(required - set(rule))
                raise RuleSchemaError(
                    "invalid_rule_schema",
                    f"{kind} rule {rule.get('rule_id', '<unknown>')} is missing required fields {missing}",
                )
            rule_id = rule["rule_id"]
            if rule_id in rule_ids:
                raise RuleSchemaError("invalid_rule_schema", f"Duplicate rule_id {rule_id!r}")
            rule_ids.add(rule_id)
            signature = rule["canonical_signature"]
            if not isinstance(signature, str) or not signature.startswith(f"{kind}(") or not signature.endswith(")"):
                raise RuleSchemaError(
                    "invalid_rule_schema",
                    f"{kind} rule {rule_id} must declare a canonical_signature of the form {kind}(...)",
                )
            if signature in signatures:
                raise RuleSchemaError(
                    "invalid_rule_schema",
                    f"{kind} rules must have unique canonical_signature values; duplicate {signature!r}",
                )
            signatures.add(signature)
            if not isinstance(rule["parameters"], dict) or not rule["parameters"]:
                raise RuleSchemaError("invalid_rule_schema", f"{kind} rule {rule_id} must declare non-empty parameters")
            provenance = rule["provenance"]
            if not isinstance(provenance, dict) or "source_kind" not in provenance or "source_file" not in provenance:
                raise RuleSchemaError(
                    "invalid_rule_schema",
                    f"{kind} rule {rule_id} must declare provenance.source_kind and provenance.source_file",
                )
