from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_I_ROOT = REPO_ROOT / "tests" / "reference_results" / "gate_i_charged_long_npt_conditioning"


def load_json(path: Path):
    return json.loads(path.read_text())


def test_gate_i_contract_and_manifest_exist() -> None:
    assert (GATE_I_ROOT / "gate_i_contract.json").exists()
    assert (GATE_I_ROOT / "gate_i_manifest.json").exists()
    assert (GATE_I_ROOT / "gate_i_manifest.md").exists()


def test_gate_i_contract_is_explicitly_cpu_only_and_narrow() -> None:
    contract = load_json(GATE_I_ROOT / "gate_i_contract.json")
    assert contract["schema_name"] == "gate_i_charged_long_npt_conditioning"
    assert contract["gate_id"] == "gate_i_cpu_exact_charged_long_npt_conditioning"
    assert contract["execution_policy"]["single_rank"] is True
    assert contract["execution_policy"]["cpu_only"] is True
    assert contract["execution_policy"]["exact_respa"] is True
    assert contract["system"]["system_id"] == "gate_h_dense_salt_polymer_2x2x2"
    assert contract["acceptance_criteria"]["replicas_min"] == 3
    assert "does not imply conductivity-production readiness" in " ".join(contract["non_claims"])


def test_gate_i_manifest_is_honest_while_unexecuted() -> None:
    manifest = load_json(GATE_I_ROOT / "gate_i_manifest.json")
    assert manifest["schema_name"] == "gate_i_charged_long_npt_conditioning"
    assert manifest["gate_id"] == "gate_i_cpu_exact_charged_long_npt_conditioning"
    assert manifest["status"] == "DECLARED_PENDING_EXECUTION"
    assert manifest["prepared_only"] is True
    assert manifest["executed"] is False
    assert manifest["first_failing_metric"] == "execution_state.not_run"
    assert "pending" in manifest["status"].lower()
    assert any("no completed CPU-only exact long-NPT campaign" in reason for reason in manifest["failure_reasons"])
