from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from polymer_workflow import run_file  # noqa: E402


CASE_ROOT = REPO_ROOT / "testdata" / "polymer_workflow_golden" / "cases"


def _polymer_component(report: dict) -> dict:
    for component in report["components"]:
        if component["role"] == "polymer_fragment":
            return component
    raise AssertionError("polymer_fragment component not found")


def test_capped_oligomer_chain_end_metadata_is_deterministic() -> None:
    expected_repeat_counts = {
        "monoglyme_litfsi_1to1": 1,
        "diglyme_litfsi_1to1": 2,
        "triglyme_litfsi_2to2": 3,
    }

    for case_id, repeat_count in expected_repeat_counts.items():
        report = run_file(CASE_ROOT / case_id / "spec.json", dry_run=True)
        polymer = _polymer_component(report)
        metadata = polymer["polymer_fragment_metadata"]

        assert polymer["classification_family"] == "acyclic_polyether_oligomer"
        assert metadata["status"] == "valid"
        assert metadata["fragment_model"] == "linear_methoxy_capped_polyether"
        assert metadata["end_group_model"] == "methyl_ether_caps"
        assert metadata["repeat_unit_count"] == repeat_count
        assert len(metadata["terminal_cap_atom_indices"]) == 2
        assert metadata["backbone_methylene_count"] == repeat_count * 2
        assert metadata["oxygen_count"] == repeat_count + 1


def test_repeat_unit_template_metadata_is_retained_in_report() -> None:
    report = run_file(CASE_ROOT / "monoglyme_litfsi_1to1" / "spec.json", dry_run=True)
    template = next(component for component in report["components"] if component["role"] == "polymer_template")

    assert template["exportable"] is False
    assert template["template_metadata"]["status"] == "valid"
    assert template["template_metadata"]["placeholder_count"] == 2
    assert template["template_metadata"]["connection_point_count"] == 2
