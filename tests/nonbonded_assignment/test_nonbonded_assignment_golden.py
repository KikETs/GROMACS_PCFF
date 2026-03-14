from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nonbonded_assignment import assign_file  # noqa: E402


SUPPORTED_CASE_EXPECTATIONS = {
    "ethane_neutral": {
        "pair_style_kind": "lj/class2",
        "pair_classes": 3,
        "exclusions": 19,
        "pair14": 9,
        "pair14_signatures": {"pair(hydrogen_on_alkane_sp3|hydrogen_on_alkane_sp3)"},
    },
    "dimethyl_ether_neutral": {
        "pair_style_kind": "lj/class2",
        "pair_classes": 6,
        "exclusions": 21,
        "pair14": 6,
        "pair14_signatures": {"pair(ether_alpha_carbon_sp3|hydrogen_on_ether_alpha_carbon)"},
    },
    "lithium_cation": {
        "pair_style_kind": "lj/class2/coul/long",
        "pair_classes": 1,
        "exclusions": 0,
        "pair14": 0,
        "pair14_signatures": set(),
    },
    "tfsi_anion_explicit": {
        "pair_style_kind": "lj/class2/coul/long",
        "pair_classes": 15,
        "exclusions": 39,
        "pair14": 24,
        "pair14_signatures": {
            "pair(fluorine_on_trifluoromethyl|sulfonimide_n_anion)",
            "pair(fluorine_on_trifluoromethyl|sulfonyl_oxygen)",
            "pair(sulfonyl_oxygen|sulfonyl_sulfur)",
            "pair(sulfonyl_sulfur|trifluoromethyl_carbon)",
        },
    },
}


def test_supported_golden_cases_get_complete_nonbonded_assignment() -> None:
    for case_id, expected in SUPPORTED_CASE_EXPECTATIONS.items():
        case_root = REPO_ROOT / "testdata" / "typing_golden" / "cases" / case_id
        structure_path = case_root / "inputs" / "structure.mol"
        outcome = json.loads((case_root / "expected" / "outcome.json").read_text(encoding="utf-8"))
        assert outcome["status"] == "supported"

        report = assign_file(structure_path, input_format="mol_v2000", source_id=case_id)
        component = report["components"][0]

        assert report["nonbonded_assignment"]["status"] == "assigned"
        assert report["nonbonded_assignment"]["pair_style_kind"] == expected["pair_style_kind"]
        assert component["diagnostics"] == []
        assert len(component["atoms"]) == component["atom_count"]
        assert len(component["pair_classes"]) == expected["pair_classes"]
        assert len(component["exclusions"]) == expected["exclusions"]
        assert len(component["pair14"]) == expected["pair14"]
        assert {record["canonical_family_pair"] for record in component["pair14"]} == expected["pair14_signatures"]
        for atom in component["atoms"]:
            assert atom["charge_assignment"]["source"] in {"formal_charge", "partial_charge"}
            assert atom["self_parameters"] is not None
            assert atom["provenance"]["rule_id"] is not None
        for record in component["pair_classes"]:
            assert record["status"] == "assigned"
            assert record["parameters"] is not None
            assert record["provenance"] is not None
        for record in component["exclusions"]:
            assert record["lj_scale"] == 0.0
            assert record["coul_scale"] == 0.0
            assert record["topological_relation"] in {"1-2", "1-3"}
            assert record["source_assignment_ids"]
        for record in component["pair14"]:
            assert record["status"] == "assigned"
            assert record["lj_scale"] == 1.0
            assert record["coul_scale"] == 1.0
            assert record["source_dihedral_assignment_ids"]
            assert record["parameters"]["pair14_coefficients"]["repulsion_power"] == 9
