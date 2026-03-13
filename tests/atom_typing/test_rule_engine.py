from __future__ import annotations

import copy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from atom_typing import (  # noqa: E402
    dumps_typing_report,
    load_rules,
    loads_typing_report,
    type_file,
    validate_rules,
)


ETHANE_PATH = REPO_ROOT / "testdata" / "typing_golden" / "cases" / "ethane_neutral" / "inputs" / "structure.mol"


def test_default_ruleset_is_valid_and_report_round_trips() -> None:
    ruleset = load_rules()
    validate_rules(ruleset)
    report = type_file(ETHANE_PATH, input_format="mol_v2000", source_id="ethane")
    assert loads_typing_report(dumps_typing_report(report)) == report


def test_duplicate_top_precedence_rules_surface_ambiguity_explicitly() -> None:
    ruleset = load_rules()
    duplicate = copy.deepcopy(ruleset["atom_type_rules"][0])
    duplicate["rule_id"] = "atom_alkane_carbon_sp3_duplicate"
    ruleset["atom_type_rules"].append(duplicate)

    report = type_file(
        ETHANE_PATH,
        input_format="mol_v2000",
        source_id="ethane_ambiguous",
        ruleset=ruleset,
    )

    component = report["components"][0]
    assert report["typing"]["status"] == "ambiguous"
    ambiguous = [diagnostic for diagnostic in component["diagnostics"] if diagnostic["code"] == "ambiguous_atom_type_match"]
    assert len(ambiguous) == 2
    assert {
        "atom_alkane_carbon_sp3",
        "atom_alkane_carbon_sp3_duplicate",
    } == set(ambiguous[0]["candidate_rule_ids"])


def test_missing_hydrogen_rule_surfaces_unresolved_atoms_explicitly() -> None:
    ruleset = load_rules()
    ruleset["atom_type_rules"] = [
        rule for rule in ruleset["atom_type_rules"] if rule["rule_id"] != "atom_hydrogen_on_alkane_sp3"
    ]

    report = type_file(
        ETHANE_PATH,
        input_format="mol_v2000",
        source_id="ethane_unresolved",
        ruleset=ruleset,
    )

    component = report["components"][0]
    assert report["typing"]["status"] == "unresolved"
    unresolved = [diagnostic for diagnostic in component["diagnostics"] if diagnostic["code"] == "unresolved_atom_type"]
    assert len(unresolved) == 6
    assert {diagnostic["source_index"] for diagnostic in unresolved} == {3, 4, 5, 6, 7, 8}
