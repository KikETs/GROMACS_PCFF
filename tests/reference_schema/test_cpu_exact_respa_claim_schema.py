from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIM_ROOT = REPO_ROOT / "tests" / "reference_results" / "cpu_exact_respa_claim"
BOUNDARY_DOC = REPO_ROOT / "docs" / "exact_respa_cpu_claim_boundary.md"


def load_json(path: Path):
    return json.loads(path.read_text())


def test_cpu_exact_claim_outputs_exist() -> None:
    expected = {
        "boundary_and_blockers.json",
        "cpu_exact_claim_summary.json",
        "mechanical_evidence_index.json",
        "openmp_validation_summary.json",
        "support_matrix.json",
    }
    found = {path.name for path in CLAIM_ROOT.iterdir() if path.is_file()}
    assert expected <= found


def test_cpu_exact_claim_summary_is_narrow_and_explicit() -> None:
    summary = load_json(CLAIM_ROOT / "cpu_exact_claim_summary.json")
    assert summary["schema_name"] == "cpu_exact_respa_claim_summary"
    assert "narrow CPU exact-r-RESPA claim" in summary["public_claim"]
    assert "audited ntomp>1 buckets `ntompSmall` and `ntompCeiling`" in summary["public_claim"]
    assert "host-local throughput benchmarks do not broaden support" in summary["public_claim"]
    assert "does not imply conductivity-production readiness" in summary["public_claim"]
    assert "small-fixture gates only" in summary["public_claim"]
    assert "long-NPT density conditioning is still missing" in summary["public_claim"]
    assert "openmp_supported_envelope_statement" in summary
    assert "bounded desktop/workstation topology classes" not in summary["public_claim"]
    assert "not exact-r-RESPA evidence" in summary["ensemble_boundary_statement"]
    assert "sole_immediate_blocker" in summary
    assert "Gate I" in summary["next_gate"]
    assert any(
        source.endswith("gate_i_charged_long_npt_conditioning/gate_i_contract.json")
        for source in summary["primary_machine_readable_sources"]
    )


def test_cpu_exact_support_matrix_separates_exact_from_non_exact_and_transport() -> None:
    matrix = load_json(CLAIM_ROOT / "support_matrix.json")
    assert matrix["schema_name"] == "cpu_exact_respa_support_matrix"

    items = {item["id"]: item for item in matrix["items"]}
    assert items["mechanics.gate_a_cpu_oracle_event_order_restart"]["status"] == "exact"
    assert items["mechanics.desktop_cpu_openmp_inventory"]["status"] == "exact"
    assert items["mechanics.host_local_openmp_throughput_observations"]["status"] == "approximate"
    assert items["mechanics.desktop_cpu_openmp_outside_audited_buckets"]["status"] == "unsupported"
    assert items["ensemble.gate_g_small_oligomer_exact_nvt"]["status"] == "exact"
    assert items["ensemble.gate_g_small_salt_polymer_box_exact_npt"]["status"] == "exact"
    assert items["ensemble.gate_h_large_medium_exact_nvt_scaffold"]["status"] == "approximate"
    assert items["ensemble.gate_i_declared_long_npt_conditioning_contract"]["status"] == "approximate"
    assert items["ensemble.exact_medium_scale_charged_long_npt_density_conditioning"]["status"] == "unsupported"
    assert items["diagnostics.m10_medium_scale_plain_md_gates"]["status"] == "non_exact_diagnostic"
    assert items["transport.cpu_exactness_not_transport_readiness"]["status"] == "unsupported"
    assert "discrete bucket claim" in items["mechanics.desktop_cpu_openmp_inventory"]["non_claimable_statement"]
    assert "host-local throughput observations" in items["mechanics.host_local_openmp_throughput_observations"]["non_claimable_statement"]
    assert "Do not interpolate from ntomp=1 oracle correctness" in items["mechanics.desktop_cpu_openmp_outside_audited_buckets"]["non_claimable_statement"]
    assert "declared Gate I contract is not a pass" in items["ensemble.gate_i_declared_long_npt_conditioning_contract"]["non_claimable_statement"]
    assert "Do not cite M10 as exact-r-RESPA evidence" in items["diagnostics.m10_medium_scale_plain_md_gates"]["non_claimable_statement"]
    assert "Do not use CPU exactness language as a shortcut to conductivity" in items["transport.cpu_exactness_not_transport_readiness"]["non_claimable_statement"]


def test_cpu_exact_boundary_freezes_single_blocker_and_next_gate() -> None:
    boundary = load_json(CLAIM_ROOT / "boundary_and_blockers.json")
    assert boundary["schema_name"] == "cpu_exact_respa_boundary_and_blockers"
    blocker = boundary["sole_immediate_blocker"]
    assert blocker["id"] == "exact_medium_scale_charged_long_npt_density_conditioning"
    assert "sole immediate blocker" in blocker["statement"]
    next_gate = boundary["next_gate"]
    assert next_gate["id"] == "gate_i_cpu_exact_charged_long_npt_conditioning"
    assert any("density and volume block-drift" in item for item in next_gate["must_prove"])
    assert any("TP0-scale production length is still required afterward" in item for item in next_gate["still_not_implied_even_if_passes"])
    assert next_gate["frozen_contract"] == "tests/reference_results/gate_i_charged_long_npt_conditioning/gate_i_contract.json"


def test_cpu_exact_openmp_summary_is_strict_and_desktop_only() -> None:
    summary = load_json(CLAIM_ROOT / "openmp_validation_summary.json")
    assert summary["schema_version"] == 3
    assert summary["pass"] is True
    assert summary["support_claim_allowed"] is True
    assert summary["mechanics_claim_allowed"] is True
    assert summary["production_rule_allowed"] is False
    assert summary["thread_scaling_rule_allowed"] is False
    assert summary["scientific_md_production_handoff_implied"] is False
    assert "bounded CPU OpenMP mechanics claim only" in summary["scope_note"]
    assert summary["supported_envelope"]["validated_probe_labels"] == ["ntompSmall", "ntompCeiling"]
    assert summary["correctness_only_envelope"]["status"] == "none"
    assert any("ntomp=1 is the oracle baseline" in item for item in summary["unsupported_or_weak_shapes"])
    assert "host-local throughput observations" in summary["host_local_throughput_observations"]["scope_note"]
    assert "discrete mechanics claim only" in summary["final_allowed_claim"]
    assert "does not imply MD production handoff" in summary["final_allowed_claim"]
    assert "does not cover server CPUs" in summary["final_allowed_claim"]
    assert "does not imply MPI or GPU coexistence support" in summary["final_allowed_claim"]


def test_cpu_exact_boundary_doc_tracks_frozen_claim() -> None:
    summary = load_json(CLAIM_ROOT / "cpu_exact_claim_summary.json")
    document = BOUNDARY_DOC.read_text()
    assert summary["public_claim"] in document
    assert "## CPU OpenMP Envelope" in document
    assert "Correctness-only:" in document
    assert "Weak or unsupported:" in document
