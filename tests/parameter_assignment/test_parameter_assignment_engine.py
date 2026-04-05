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
from parameter_assignment.engine import (  # noqa: E402
    _index_rules_by_id,
    _phase1_repository_tuple_backfill_provenance,
    _resolve_phase1_pcff_tuple_remap,
    _resolve_phase1_repository_tuple_backfill,
)
from pcff_frc import _lookup_term, load_pcff_frc  # noqa: E402


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


def test_pcff_frc_improper_lookup_uses_center_atom_consistently() -> None:
    load_pcff_frc.cache_clear()
    reference = load_pcff_frc()
    record, provenance = _lookup_term(reference, "improper_main", ["cz", "oo", "oz", "oz"])
    assert record is not None
    assert provenance is not None
    assert provenance["resolved_key"] == ["cz", "oo", "oz", "oz"]
    assert provenance["line_number"] == 3210


def test_pcff_frc_wildcard_torsion_lookup_is_traceable() -> None:
    load_pcff_frc.cache_clear()
    reference = load_pcff_frc()
    record, provenance = _lookup_term(reference, "dihedral_main", ["c", "c_1", "o_2", "cp"])
    assert record is not None
    assert provenance is not None
    assert provenance["resolved_key"] == ["*", "c_1", "o_2", "*"]
    assert provenance["used_wildcard"] is True


def test_phase1_pcff_tuple_remap_surfaces_traceable_angle_bridge_fix() -> None:
    parameters, provenance = _resolve_phase1_pcff_tuple_remap("angle", ["n_2", "c_2", "oz"])
    assert parameters is not None
    assert provenance is not None
    assert parameters["main"]["theta0_deg"] == 108.44
    assert provenance["matched_pcff_types"] == ["n_2", "c_2", "oz"]
    assert provenance["remapped_pcff_types"] == ["n_2", "c_2", "o_2"]
    assert provenance["source_resolution"] == "phase1_tuple_remap"


def test_phase1_repository_tuple_backfill_is_exact_pcff_tuple_scoped() -> None:
    ruleset = load_rules()
    rules_by_id = _index_rules_by_id(ruleset)

    angle_rule = _resolve_phase1_repository_tuple_backfill("angle", ["h", "c", "h"], rules_by_id)
    assert angle_rule is not None
    assert angle_rule["rule_id"] == "angle_alkane_hc_h"

    dihedral_rule = _resolve_phase1_repository_tuple_backfill("dihedral", ["h", "c", "c", "h"], rules_by_id)
    assert dihedral_rule is not None
    assert dihedral_rule["rule_id"] == "dihedral_alkane_hcch"

    improper_rule = _resolve_phase1_repository_tuple_backfill("improper", ["c_1", "c", "n", "o_1"], rules_by_id)
    assert improper_rule is not None
    assert improper_rule["rule_id"] == "ap5_import_improper_c_1_c_n_o_1_v1_6mBN"

    assert _resolve_phase1_repository_tuple_backfill("dihedral", ["n", "c", "c", "h"], rules_by_id) is None

    provenance = _phase1_repository_tuple_backfill_provenance(["h", "c", "c", "h"], dihedral_rule)
    assert provenance["source_resolution"] == "phase1_exact_pcff_tuple_backfill"
    assert provenance["source_rule_id"] == "dihedral_alkane_hcch"

    improper_provenance = _phase1_repository_tuple_backfill_provenance(["c_1", "c", "n", "o_1"], improper_rule)
    assert improper_provenance["source_resolution"] == "phase1_exact_pcff_tuple_backfill"
    assert improper_provenance["source_rule_id"] == "ap5_import_improper_c_1_c_n_o_1_v1_6mBN"
