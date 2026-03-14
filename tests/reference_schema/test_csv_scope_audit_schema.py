from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = REPO_ROOT / "data_manifests"
REFERENCE_ROOT = REPO_ROOT / "tests" / "reference_results" / "csv_scope_audit"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_csv_scope_manifests_exist_and_match_expected_snapshot_identity() -> None:
    snapshot = load_json(MANIFEST_ROOT / "simulation_trajectory_aggregate_snapshot.json")
    unique_manifest = load_json(MANIFEST_ROOT / "simulation_trajectory_aggregate_unique_smiles.json")
    row_map = load_json(MANIFEST_ROOT / "simulation_trajectory_aggregate_row_map.json")

    assert snapshot["schema_name"] == "csv_scope_snapshot_manifest"
    assert unique_manifest["schema_name"] == "csv_scope_unique_smiles_manifest"
    assert row_map["schema_name"] == "csv_scope_row_map"

    assert snapshot["source_csv"]["sha256"] == "a67a8f86f1842cd9d35ffe6cce2de8a3cf3577635aed19564b910242dd226fcf"
    assert snapshot["row_count"] == 6270
    assert snapshot["unique_smiles_count"] == 6042
    assert snapshot["duplicate_row_count"] == 228

    assert unique_manifest["unique_smiles_count"] == 6042
    assert len(unique_manifest["entries"]) == 6042
    assert row_map["row_count"] == 6270
    assert len(row_map["rows"]) == 6270


def test_csv_scope_row_map_is_total_and_references_known_unique_ids() -> None:
    unique_manifest = load_json(MANIFEST_ROOT / "simulation_trajectory_aggregate_unique_smiles.json")
    row_map = load_json(MANIFEST_ROOT / "simulation_trajectory_aggregate_row_map.json")

    known_ids = {entry["unique_smiles_id"] for entry in unique_manifest["entries"]}
    assert row_map["rows"][0]["row_number"] == 1
    assert row_map["rows"][-1]["row_number"] == 6270
    assert all(row["unique_smiles_id"] in known_ids for row in row_map["rows"])


def test_csv_scope_audit_results_are_explicit_and_currently_parse_blocked() -> None:
    results = load_json(REFERENCE_ROOT / "coverage_audit_results.json")
    summary = load_json(REFERENCE_ROOT / "coverage_audit_summary.json")

    assert results["schema_name"] == "csv_scope_coverage_audit_results"
    assert results["entry_count"] == 6042
    assert results["pipeline_contract"]["typing_ir_supported_input_formats"] == [
        "mol2",
        "mol_v2000",
        "pdb",
        "sdf",
    ]
    assert results["pipeline_contract"]["csv_smiles_adapter_status"] == "unsupported"
    assert all(entry["status"] == "failure" for entry in results["entries"])
    assert all(entry["failure_class"] == "parse_failure" for entry in results["entries"])
    assert all(entry["failure_code"] == "unsupported_csv_smiles_input" for entry in results["entries"])

    assert summary["schema_name"] == "csv_scope_coverage_summary"
    assert summary["totals"] == {
        "row_count": 6270,
        "row_coverage_fraction": 0.0,
        "supported_row_count": 0,
        "supported_unique_smiles_count": 0,
        "unique_coverage_fraction": 0.0,
        "unique_smiles_count": 6042,
    }
    assert summary["failure_class_counts"]["unique_smiles"] == {
        "atom_typing_failure": 0,
        "chemical_perception_failure": 0,
        "emitter_export_failure": 0,
        "nonbonded_assignment_failure": 0,
        "parameter_assignment_failure": 0,
        "parse_failure": 6042,
    }
    assert summary["failure_class_counts"]["rows"] == {
        "atom_typing_failure": 0,
        "chemical_perception_failure": 0,
        "emitter_export_failure": 0,
        "nonbonded_assignment_failure": 0,
        "parameter_assignment_failure": 0,
        "parse_failure": 6270,
    }
    assert summary["release_readiness"]["status"] == "not_ready"
