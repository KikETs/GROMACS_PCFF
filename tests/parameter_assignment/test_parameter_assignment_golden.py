from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from parameter_assignment import assign_file  # noqa: E402


SUPPORTED_CASE_EXPECTATIONS = {
    "ethane_neutral": {
        "counts": {"bond": 7, "angle": 12, "dihedral": 9, "improper": 0},
        "signatures": {
            "bond": {
                "bond(alkane_carbon_sp3|alkane_carbon_sp3)",
                "bond(alkane_carbon_sp3|hydrogen_on_alkane_sp3)",
            },
            "angle": {
                "angle(alkane_carbon_sp3|alkane_carbon_sp3|hydrogen_on_alkane_sp3)",
                "angle(hydrogen_on_alkane_sp3|alkane_carbon_sp3|hydrogen_on_alkane_sp3)",
            },
            "dihedral": {
                "dihedral(hydrogen_on_alkane_sp3|alkane_carbon_sp3|alkane_carbon_sp3|hydrogen_on_alkane_sp3)",
            },
            "improper": set(),
        },
    },
    "dimethyl_ether_neutral": {
        "counts": {"bond": 8, "angle": 13, "dihedral": 6, "improper": 0},
        "signatures": {
            "bond": {
                "bond(ether_alpha_carbon_sp3|ether_oxygen_sp3)",
                "bond(ether_alpha_carbon_sp3|hydrogen_on_ether_alpha_carbon)",
            },
            "angle": {
                "angle(ether_alpha_carbon_sp3|ether_oxygen_sp3|ether_alpha_carbon_sp3)",
                "angle(ether_oxygen_sp3|ether_alpha_carbon_sp3|hydrogen_on_ether_alpha_carbon)",
                "angle(hydrogen_on_ether_alpha_carbon|ether_alpha_carbon_sp3|hydrogen_on_ether_alpha_carbon)",
            },
            "dihedral": {
                "dihedral(ether_alpha_carbon_sp3|ether_oxygen_sp3|ether_alpha_carbon_sp3|hydrogen_on_ether_alpha_carbon)",
            },
            "improper": set(),
        },
    },
    "lithium_cation": {
        "counts": {"bond": 0, "angle": 0, "dihedral": 0, "improper": 0},
        "signatures": {"bond": set(), "angle": set(), "dihedral": set(), "improper": set()},
    },
    "tfsi_anion_explicit": {
        "counts": {"bond": 14, "angle": 25, "dihedral": 24, "improper": 0},
        "signatures": {
            "bond": {
                "bond(fluorine_on_trifluoromethyl|trifluoromethyl_carbon)",
                "bond(sulfonimide_n_anion|sulfonyl_sulfur)",
                "bond(sulfonyl_oxygen|sulfonyl_sulfur)",
                "bond(sulfonyl_sulfur|trifluoromethyl_carbon)",
            },
            "angle": {
                "angle(fluorine_on_trifluoromethyl|trifluoromethyl_carbon|fluorine_on_trifluoromethyl)",
                "angle(fluorine_on_trifluoromethyl|trifluoromethyl_carbon|sulfonyl_sulfur)",
                "angle(sulfonimide_n_anion|sulfonyl_sulfur|sulfonyl_oxygen)",
                "angle(sulfonimide_n_anion|sulfonyl_sulfur|trifluoromethyl_carbon)",
                "angle(sulfonyl_oxygen|sulfonyl_sulfur|sulfonyl_oxygen)",
                "angle(sulfonyl_oxygen|sulfonyl_sulfur|trifluoromethyl_carbon)",
                "angle(sulfonyl_sulfur|sulfonimide_n_anion|sulfonyl_sulfur)",
            },
            "dihedral": {
                "dihedral(fluorine_on_trifluoromethyl|trifluoromethyl_carbon|sulfonyl_sulfur|sulfonimide_n_anion)",
                "dihedral(fluorine_on_trifluoromethyl|trifluoromethyl_carbon|sulfonyl_sulfur|sulfonyl_oxygen)",
                "dihedral(sulfonyl_oxygen|sulfonyl_sulfur|sulfonimide_n_anion|sulfonyl_sulfur)",
                "dihedral(sulfonyl_sulfur|sulfonimide_n_anion|sulfonyl_sulfur|trifluoromethyl_carbon)",
            },
            "improper": set(),
        },
    },
}


def test_supported_golden_cases_get_complete_bonded_parameter_assignment() -> None:
    for case_id, expected in SUPPORTED_CASE_EXPECTATIONS.items():
        case_root = REPO_ROOT / "testdata" / "typing_golden" / "cases" / case_id
        structure_path = case_root / "inputs" / "structure.mol"
        outcome = json.loads((case_root / "expected" / "outcome.json").read_text(encoding="utf-8"))
        assert outcome["status"] == "supported"

        report = assign_file(structure_path, input_format="mol_v2000", source_id=case_id)
        component = report["components"][0]

        assert report["parameter_assignment"]["status"] == "assigned"
        assert component["diagnostics"] == []
        assert component["interaction_counts"] == expected["counts"]
        for kind, records in component["interactions"].items():
            assert len(records) == expected["counts"][kind]
            assert {record["canonical_signature"] for record in records} == expected["signatures"][kind]
            for record in records:
                assert record["status"] == "assigned"
                assert record["parameter_rule_id"] is not None
                assert record["parameters"] is not None
                assert record["provenance"]["rule_id"] == record["parameter_rule_id"]
                assert record["provenance"]["canonical_signature"] == record["canonical_signature"]
