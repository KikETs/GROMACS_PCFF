from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "exact_respa_openmp_validation"
    / "host_local_explicit_counts"
)
EXPECTED_ARTIFACT = ARTIFACT_ROOT / "amd_ryzen_9_9900x_numa_or_chiplet_ntomp_2_4_6_8_10_12.json"
EXPECTED_COUNTS = [2, 4, 6, 8, 10, 12]


def load_json(path: Path):
    return json.loads(path.read_text())


def test_checked_in_host_local_explicit_counts_artifact_exists() -> None:
    assert EXPECTED_ARTIFACT.exists()


def test_host_local_explicit_counts_artifact_is_explicitly_non_claimable() -> None:
    report = load_json(EXPECTED_ARTIFACT)
    assert report["schema_name"] == "exact_respa_openmp_host_local_explicit_counts"
    assert report["schema_version"] == 1
    assert "host-local explicit ntomp evidence on one audited host" in report["claim_scope_statement"]
    assert "does not broaden the bounded desktop/workstation claim" in report["claim_scope_statement"]
    assert "Do not use this file to claim a continuous ntomp envelope" in report["non_claimable_statement"]


def test_host_identity_matches_checked_in_9900x_profile() -> None:
    report = load_json(EXPECTED_ARTIFACT)
    host = report["host_identity"]
    assert host["profile_id"] == "amd_ryzen_9_9900x_numa_or_chiplet"
    assert host["host_label"] == "9900x-kiket-local"
    assert host["topology_class"] == "numa-or-chiplet"
    assert host["cpu_model_slug"] == "amd_ryzen_9_9900x"
    assert host["logical_cpus"] == 24
    assert host["affinity_visible_cpus"] == 24


def test_host_local_explicit_counts_cover_even_ntomp_2_to_12_without_broadening_claim() -> None:
    report = load_json(EXPECTED_ARTIFACT)
    counts = report["counts"]
    assert [item["ntomp"] for item in counts] == EXPECTED_COUNTS
    assert report["collection"]["requested_ntomp_counts"] == EXPECTED_COUNTS
    assert report["collection"]["overall_ok"] is True
    assert report["passing_explicit_ntomp_counts_on_this_host"] == EXPECTED_COUNTS

    for item in counts:
        assert item["category"] == "host_local_explicit_count"
        assert item["ok"] is True
        assert item["support_claimable"] is False
        assert item["host_local_mechanics_ok"] is True
        assert item["restart_affinity_ok"] is True
        assert item["openmp_affinity_ok"] is True
        assert item["openmp_oracle_parity_ok"] is True
        assert item["validated_pin_modes"] == ["auto", "on", "inherit"]
        assert item["blockers"] == []
