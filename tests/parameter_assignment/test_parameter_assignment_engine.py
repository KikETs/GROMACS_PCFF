from __future__ import annotations

import copy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from parameter_assignment import (  # noqa: E402
    assign_file,
    canonicalize_angle,
    canonicalize_bond,
    canonicalize_dihedral,
    canonicalize_improper,
    dumps_assignment_report,
    load_rules,
    loads_assignment_report,
    validate_rules,
)


ETHANE_PATH = REPO_ROOT / "testdata" / "typing_golden" / "cases" / "ethane_neutral" / "inputs" / "structure.mol"


def test_default_ruleset_is_valid_and_report_round_trips() -> None:
    ruleset = load_rules()
    validate_rules(ruleset)
    report = assign_file(ETHANE_PATH, input_format="mol_v2000", source_id="ethane")
    assert loads_assignment_report(dumps_assignment_report(report)) == report


def test_canonical_signatures_are_deterministic_across_reversal_and_permutation() -> None:
    assert canonicalize_bond([7, 2], ["z_family", "a_family"]) == (
        [2, 7],
        ["a_family", "z_family"],
        "bond(a_family|z_family)",
    )
    assert canonicalize_angle([9, 4, 3], ["z_family", "center_family", "a_family"]) == (
        [3, 4, 9],
        ["a_family", "center_family", "z_family"],
        "angle(a_family|center_family|z_family)",
    )
    assert canonicalize_dihedral(
        [6, 4, 3, 1],
        ["z_family", "middle_left", "middle_right", "a_family"],
    ) == (
        [1, 3, 4, 6],
        ["a_family", "middle_right", "middle_left", "z_family"],
        "dihedral(a_family|middle_right|middle_left|z_family)",
    )
    assert canonicalize_improper(
        5,
        [8, 3, 4],
        "center_family",
        ["z_family", "a_family", "a_family"],
    ) == (
        [5, 3, 4, 8],
        ["center_family", "a_family", "a_family", "z_family"],
        "improper(center_family|a_family|a_family|z_family)",
    )


def test_missing_parameter_surfaces_explicit_diagnostics() -> None:
    ruleset = load_rules()
    ruleset["interaction_rules"]["bond"] = [
        rule
        for rule in ruleset["interaction_rules"]["bond"]
        if rule["rule_id"] != "bond_alkane_ch"
    ]

    report = assign_file(
        ETHANE_PATH,
        input_format="mol_v2000",
        source_id="ethane_missing_bond_parameter",
        ruleset=ruleset,
    )

    component = report["components"][0]
    assert report["parameter_assignment"]["status"] == "missing_parameters"
    diagnostics = component["diagnostics"]
    assert len(diagnostics) == 6
    assert {
        diagnostic["canonical_signature"]
        for diagnostic in diagnostics
    } == {"bond(alkane_carbon_sp3|hydrogen_on_alkane_sp3)"}
    missing_records = [
        record for record in component["interactions"]["bond"] if record["status"] == "missing_parameter"
    ]
    assert len(missing_records) == 6
    assigned_records = [
        record for record in component["interactions"]["bond"] if record["status"] == "assigned"
    ]
    assert len(assigned_records) == 1


def test_rule_validation_rejects_duplicate_signature_within_kind() -> None:
    ruleset = load_rules()
    duplicate = copy.deepcopy(ruleset["interaction_rules"]["bond"][0])
    duplicate["rule_id"] = "bond_alkane_cc_duplicate"
    ruleset["interaction_rules"]["bond"].append(duplicate)

    try:
        validate_rules(ruleset)
    except Exception as error:
        assert "duplicate" in str(error).lower()
    else:
        raise AssertionError("validate_rules should reject duplicate canonical signatures")
