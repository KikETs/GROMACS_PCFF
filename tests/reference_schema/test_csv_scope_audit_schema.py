from __future__ import annotations

import json
from collections import Counter
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
    assert snapshot["audit_adapter_contract"] == {
        "adapter_kind": "pysoftk_proto_polymer",
        "output_format": "mol2",
        "placeholder": "Br",
        "structure_generation_policy": "rdkit_embed_then_pysoftk_proto_polymer_then_placeholder_to_hydrogen",
    }

    assert unique_manifest["unique_smiles_count"] == 6042
    assert len(unique_manifest["entries"]) == 6042
    assert row_map["row_count"] == 6270
    assert len(row_map["rows"]) == 6270

    first_entry = unique_manifest["entries"][0]
    assert first_entry["unique_smiles_id"] == "csv_scope_smiles_000001"
    assert first_entry["adapter_input"]["adapter_kind"] == "pysoftk_proto_polymer"
    assert first_entry["adapter_input"]["output_format"] == "mol2"
    assert first_entry["adapter_input"]["placeholder"] == "Br"
    assert first_entry["adapter_input"]["degree_of_polymerization"] == first_entry["representative_row"]["degree_of_polymerization"]


def test_csv_scope_row_map_is_total_and_references_known_unique_ids() -> None:
    unique_manifest = load_json(MANIFEST_ROOT / "simulation_trajectory_aggregate_unique_smiles.json")
    row_map = load_json(MANIFEST_ROOT / "simulation_trajectory_aggregate_row_map.json")

    known_ids = {entry["unique_smiles_id"] for entry in unique_manifest["entries"]}
    assert row_map["rows"][0]["row_number"] == 1
    assert row_map["rows"][-1]["row_number"] == 6270
    assert all(row["unique_smiles_id"] in known_ids for row in row_map["rows"])


def test_csv_scope_audit_results_follow_schema_and_match_summary() -> None:
    results = load_json(REFERENCE_ROOT / "coverage_audit_results.json")
    summary = load_json(REFERENCE_ROOT / "coverage_audit_summary.json")

    assert results["schema_name"] == "csv_scope_coverage_audit_results"
    assert results["schema_version"] == 1
    assert results["entry_count"] == 6042
    assert len(results["entries"]) == 6042
    assert results["pipeline_contract"]["typing_ir_supported_input_formats"] == [
        "mol2",
        "mol_v2000",
        "pdb",
        "sdf",
    ]
    assert results["pipeline_contract"]["csv_smiles_adapter_status"] == "pysoftk_proto_polymer_active"
    assert results["pipeline_contract"]["csv_smiles_adapter_output_format"] == "mol2"
    assert results["pipeline_contract"]["csv_smiles_adapter_placeholder"] == "Br"

    failure_classes = Counter()
    row_failure_classes = Counter()
    observed_failure_codes: dict[str, Counter] = {}
    supported_unique = 0
    supported_rows = 0
    for entry in results["entries"]:
        assert entry["adapter_input"]["adapter_kind"] == "pysoftk_proto_polymer"
        assert entry["adapter_input"]["output_format"] == "mol2"
        assert entry["adapter_input"]["placeholder"] == "Br"
        assert entry["row_count"] >= 1
        if entry["status"] == "pass":
            supported_unique += 1
            supported_rows += entry["row_count"]
            continue
        failure_class = entry["failure_class"]
        failure_classes[failure_class] += 1
        row_failure_classes[failure_class] += entry["row_count"]
        observed_failure_codes.setdefault(failure_class, Counter())[entry["failure_code"]] += 1

    assert summary["schema_name"] == "csv_scope_coverage_summary"
    assert summary["schema_version"] == 1
    assert summary["totals"] == {
        "row_count": 6270,
        "row_coverage_fraction": supported_rows / 6270,
        "supported_row_count": supported_rows,
        "supported_unique_smiles_count": supported_unique,
        "unique_coverage_fraction": supported_unique / 6042,
        "unique_smiles_count": 6042,
    }
    assert summary["failure_class_counts"]["unique_smiles"] == {
        "atom_typing_failure": failure_classes.get("atom_typing_failure", 0),
        "chemical_perception_failure": failure_classes.get("chemical_perception_failure", 0),
        "emitter_export_failure": failure_classes.get("emitter_export_failure", 0),
        "nonbonded_assignment_failure": failure_classes.get("nonbonded_assignment_failure", 0),
        "parameter_assignment_failure": failure_classes.get("parameter_assignment_failure", 0),
        "parse_failure": failure_classes.get("parse_failure", 0),
    }
    assert summary["failure_class_counts"]["rows"] == {
        "atom_typing_failure": row_failure_classes.get("atom_typing_failure", 0),
        "chemical_perception_failure": row_failure_classes.get("chemical_perception_failure", 0),
        "emitter_export_failure": row_failure_classes.get("emitter_export_failure", 0),
        "nonbonded_assignment_failure": row_failure_classes.get("nonbonded_assignment_failure", 0),
        "parameter_assignment_failure": row_failure_classes.get("parameter_assignment_failure", 0),
        "parse_failure": row_failure_classes.get("parse_failure", 0),
    }
    assert summary["observed_failure_codes"] == {
        failure_class: dict(sorted(code_counts.items()))
        for failure_class, code_counts in {
            "atom_typing_failure": observed_failure_codes.get("atom_typing_failure", Counter()),
            "chemical_perception_failure": observed_failure_codes.get("chemical_perception_failure", Counter()),
            "emitter_export_failure": observed_failure_codes.get("emitter_export_failure", Counter()),
            "nonbonded_assignment_failure": observed_failure_codes.get("nonbonded_assignment_failure", Counter()),
            "parameter_assignment_failure": observed_failure_codes.get("parameter_assignment_failure", Counter()),
            "parse_failure": observed_failure_codes.get("parse_failure", Counter()),
        }.items()
    }
    assert summary["release_readiness"]["status"] == ("ready" if supported_unique == 6042 else "not_ready")
