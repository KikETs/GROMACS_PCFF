from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from atom_typing import type_file  # noqa: E402


CORPUS_ROOT = REPO_ROOT / "testdata" / "typing_golden" / "cases"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_supported_golden_cases_reproduce_expected_atom_type_families() -> None:
    for case_id in [
        "ethane_neutral",
        "dimethyl_ether_neutral",
        "lithium_cation",
        "tfsi_anion_explicit",
    ]:
        case_root = CORPUS_ROOT / case_id
        outcome = load_json(case_root / "expected" / "outcome.json")
        report = type_file(case_root / "inputs" / "structure.mol", input_format="mol_v2000", source_id=case_id)

        component = report["components"][0]
        assert report["typing"]["status"] == "typed"
        assert component["classification"]["status"] == "supported"

        actual_by_source_index = {
            atom["source_index"]: atom["assigned_family"]
            for atom in component["atoms"]
        }
        for expected in outcome["atom_type_family_expectations"]:
            assert actual_by_source_index[expected["atom_index"]] == expected["family"]

        explanations_by_id = {
            explanation["explanation_id"]: explanation
            for explanation in component["atom_type_explanations"]
        }
        assert len(explanations_by_id) == component["atom_count"]
        for atom in component["atoms"]:
            assert atom["status"] == "assigned"
            assert atom["matched_rule_id"] is not None
            explanation = explanations_by_id[atom["explanation_id"]]
            assert explanation["rule_id"] == atom["matched_rule_id"]
            assert explanation["assigned_family"] == atom["assigned_family"]
            assert explanation["source_atom_indices"]


def test_unsupported_golden_cases_emit_expected_failure_codes_and_messages() -> None:
    for case_id in [
        "benzene_aromatic_unsupported",
        "acetaldehyde_carbonyl_unsupported",
    ]:
        case_root = CORPUS_ROOT / case_id
        outcome = load_json(case_root / "expected" / "outcome.json")
        report = type_file(case_root / "inputs" / "structure.mol", input_format="mol_v2000", source_id=case_id)

        component = report["components"][0]
        assert report["typing"]["status"] == "unsupported"
        assert component["classification"]["status"] == "unsupported"
        assert component["classification"]["failure_code"] == outcome["failure_code"]
        diagnostic_text = "\n".join(diagnostic["message"] for diagnostic in component["diagnostics"])
        for substring in outcome["diagnostic_substrings"]:
            assert substring in diagnostic_text
