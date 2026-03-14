from __future__ import annotations

import copy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nonbonded_assignment import (  # noqa: E402
    assign_file,
    canonical_family_pair,
    class2_normal_coefficients,
    class2_pair14_coefficients,
    dumps_assignment_report,
    load_rules,
    loads_assignment_report,
    sixthpower_mix,
    validate_rules,
)


ETHANE_PATH = REPO_ROOT / "testdata" / "typing_golden" / "cases" / "ethane_neutral" / "inputs" / "structure.mol"


def test_default_ruleset_is_valid_and_report_round_trips() -> None:
    ruleset = load_rules()
    validate_rules(ruleset)
    report = assign_file(ETHANE_PATH, input_format="mol_v2000", source_id="ethane")
    assert loads_assignment_report(dumps_assignment_report(report)) == report


def test_sixthpower_mixing_and_class2_coefficients_match_frozen_formulae() -> None:
    mixed = sixthpower_mix(3.5, 0.1, 2.5, 0.02)
    assert mixed == {
        "sigma_angstrom": 3.18362992,
        "epsilon_kcal_mol": 0.02877423,
    }
    assert class2_normal_coefficients(0.02, 2.5) == {
        "c6_kcal_mol_angstrom6": 87.890625,
        "c9_kcal_mol_angstrom9": 1373.29101562,
        "dispersion_power": 6,
        "repulsion_power": 9,
    }
    assert class2_pair14_coefficients(0.02, 2.5) == {
        "c6_kcal_mol_angstrom6": 14.6484375,
        "c9_kcal_mol_angstrom9": 152.58789062,
        "dispersion_power": 6,
        "repulsion_power": 9,
    }
    assert canonical_family_pair("z_family", "a_family") == (
        ["a_family", "z_family"],
        "pair(a_family|z_family)",
    )


def test_ethane_exclusions_and_pair14_records_are_auditable() -> None:
    report = assign_file(ETHANE_PATH, input_format="mol_v2000", source_id="ethane")
    component = report["components"][0]

    assert len(component["exclusions"]) == 19
    assert sum(record["topological_relation"] == "1-2" for record in component["exclusions"]) == 7
    assert sum(record["topological_relation"] == "1-3" for record in component["exclusions"]) == 12
    assert len(component["pair14"]) == 9
    assert {record["canonical_family_pair"] for record in component["pair14"]} == {
        "pair(hydrogen_on_alkane_sp3|hydrogen_on_alkane_sp3)"
    }
    first_pair14 = component["pair14"][0]
    assert first_pair14["source_dihedral_assignment_ids"] == ["dihedral_1"]
    assert first_pair14["lj_scale"] == 1.0
    assert first_pair14["coul_scale"] == 1.0


def test_pair14_override_applies_without_changing_normal_pair_class() -> None:
    ruleset = load_rules()
    ruleset["pair_overrides"].append(
        {
            "rule_id": "override_ethane_hh_pair14",
            "canonical_family_pair": "pair(hydrogen_on_alkane_sp3|hydrogen_on_alkane_sp3)",
            "scope": "pair14",
            "parameters": {
                "epsilon_kcal_mol": 0.123,
                "sigma_angstrom": 2.222,
            },
            "provenance": {
                "source_kind": "test_override",
                "source_file": "tests/nonbonded_assignment/test_nonbonded_assignment_engine.py",
            },
        }
    )

    report = assign_file(
        ETHANE_PATH,
        input_format="mol_v2000",
        source_id="ethane_pair14_override",
        ruleset=ruleset,
    )
    component = report["components"][0]

    assert {record["parameter_source"] for record in component["pair14"]} == {"override"}
    assert component["pair14"][0]["parameters"]["sigma_angstrom"] == 2.222
    assert component["pair14"][0]["parameters"]["epsilon_kcal_mol"] == 0.123
    hh_pair_class = next(
        record for record in component["pair_classes"] if record["canonical_family_pair"] == "pair(hydrogen_on_alkane_sp3|hydrogen_on_alkane_sp3)"
    )
    assert hh_pair_class["parameter_source"] == "mixed"


def test_missing_nonbonded_family_rule_surfaces_explicit_diagnostics() -> None:
    ruleset = load_rules()
    ruleset["atom_type_rules"] = [
        rule for rule in ruleset["atom_type_rules"] if rule["family"] != "hydrogen_on_alkane_sp3"
    ]

    report = assign_file(
        ETHANE_PATH,
        input_format="mol_v2000",
        source_id="ethane_missing_nonbonded_family",
        ruleset=ruleset,
    )
    component = report["components"][0]

    assert report["nonbonded_assignment"]["status"] == "missing_parameters"
    assert sum(diagnostic["code"] == "missing_nonbonded_atom_type" for diagnostic in component["diagnostics"]) == 6
    assert sum(diagnostic["code"] == "missing_nonbonded_pair_parameter" for diagnostic in component["diagnostics"]) == 2
    assert sum(diagnostic["code"] == "missing_pair14_parameter" for diagnostic in component["diagnostics"]) == 9
    assert sum(record["status"] == "missing_parameter" for record in component["atoms"]) == 6
    assert sum(record["status"] == "missing_parameter" for record in component["pair14"]) == 9
