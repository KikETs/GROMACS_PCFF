from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from polymer_workflow import PolymerWorkflowError, load_spec, run_spec  # noqa: E402


CASE_ROOT = REPO_ROOT / "testdata" / "polymer_workflow_golden" / "cases"


def test_charge_neutrality_and_salt_balance_pass_for_golden_cases() -> None:
    for case_id in [
        "monoglyme_litfsi_1to1",
        "diglyme_litfsi_1to1",
        "triglyme_litfsi_2to2",
    ]:
        report = run_spec(load_spec(CASE_ROOT / case_id / "spec.json"), spec_path=CASE_ROOT / case_id / "spec.json", dry_run=True)
        assert report["assembly_checks"]["charge_neutrality"]["status"] == "pass"
        assert report["assembly_checks"]["salt_balance"]["status"] == "pass"
        assert report["assembly_checks"]["fragment_consistency"]["status"] == "pass"


def test_charge_imbalance_fails_explicitly() -> None:
    spec_path = CASE_ROOT / "monoglyme_litfsi_1to1" / "spec.json"
    spec = load_spec(spec_path)
    broken = copy.deepcopy(spec)
    for component in broken["components"]:
        if component["component_id"] == "LI":
            component["count"] = 2

    with pytest.raises(PolymerWorkflowError) as excinfo:
        run_spec(broken, spec_path=spec_path, dry_run=True)

    assert excinfo.value.code == "charge_imbalance"


def test_invalid_repeat_unit_template_fails_explicitly() -> None:
    spec_path = CASE_ROOT / "diglyme_litfsi_1to1" / "spec.json"
    spec = load_spec(spec_path)
    broken = copy.deepcopy(spec)
    for component in broken["components"]:
        if component["role"] == "polymer_template":
            component["path"] = "../../components/diglyme_capped/structure.mol"

    with pytest.raises(PolymerWorkflowError) as excinfo:
        run_spec(broken, spec_path=spec_path, dry_run=True)

    assert excinfo.value.code == "invalid_repeat_unit_template"
