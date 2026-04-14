from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path):
    return json.loads(path.read_text())


def test_m10_2_gate_is_explicitly_non_exact_diagnostic() -> None:
    root = REPO_ROOT / "tests" / "reference_results" / "m10_2_ensemble_gate"
    decision = load_json(root / "m10_2_gate_decision.json")
    summary = load_json(root / "m10_2_summary.json")[0]
    report = load_json(root / "small_oligomer_medium" / "report.json")

    assert decision["evidence_class"] == "non_exact_diagnostic"
    assert decision["integrator_family"] == "plain_md"
    assert "not exact-r-RESPA evidence" in decision["note"]

    assert summary["evidence_class"] == "non_exact_diagnostic"
    assert summary["integrator_family"] == "plain_md"
    assert "not exact-r-RESPA evidence" in summary["non_claimable_statement"]

    assert report["evidence_class"] == "non_exact_diagnostic"
    assert report["integrator_family"] == "plain_md"
    assert "not exact-r-RESPA evidence" in report["non_claimable_statement"]


def test_m10_2_1_gate_is_explicitly_non_exact_diagnostic() -> None:
    root = REPO_ROOT / "tests" / "reference_results" / "m10_2_1_convergence_gate"
    decision = load_json(root / "m10_2_1_gate_decision.json")
    summary = load_json(root / "m10_2_1_summary.json")[0]
    report = load_json(root / "small_oligomer_medium_100ps" / "report.json")

    assert decision["evidence_class"] == "non_exact_diagnostic"
    assert decision["integrator_family"] == "plain_md"
    assert "not exact-r-RESPA evidence" in decision["note"]

    assert summary["evidence_class"] == "non_exact_diagnostic"
    assert summary["integrator_family"] == "plain_md"
    assert "not exact-r-RESPA evidence" in summary["non_claimable_statement"]

    assert report["evidence_class"] == "non_exact_diagnostic"
    assert report["integrator_family"] == "plain_md"
    assert "not exact-r-RESPA evidence" in report["note"]
