from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = REPO_ROOT / "data_manifests"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_subset_manifests(tmp_path: Path, *, count: int = 2) -> Path:
    snapshot = load_json(MANIFEST_ROOT / "simulation_trajectory_aggregate_snapshot.json")
    unique_manifest = load_json(MANIFEST_ROOT / "simulation_trajectory_aggregate_unique_smiles.json")
    row_map = load_json(MANIFEST_ROOT / "simulation_trajectory_aggregate_row_map.json")

    subset_entries = unique_manifest["entries"][:count]
    subset_ids = {entry["unique_smiles_id"] for entry in subset_entries}
    subset_rows = [row for row in row_map["rows"] if row["unique_smiles_id"] in subset_ids]

    subset_snapshot = dict(snapshot)
    subset_snapshot["row_count"] = len(subset_rows)
    subset_snapshot["unique_smiles_count"] = len(subset_entries)
    subset_snapshot["duplicate_row_count"] = len(subset_rows) - len(subset_entries)

    subset_unique_manifest = dict(unique_manifest)
    subset_unique_manifest["unique_smiles_count"] = len(subset_entries)
    subset_unique_manifest["entries"] = subset_entries

    subset_row_map = dict(row_map)
    subset_row_map["row_count"] = len(subset_rows)
    subset_row_map["rows"] = [
        {
            "row_number": index,
            "trajectory_id": row["trajectory_id"],
            "unique_smiles_id": row["unique_smiles_id"],
        }
        for index, row in enumerate(subset_rows, start=1)
    ]

    out_root = tmp_path / "subset_manifests"
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "simulation_trajectory_aggregate_snapshot.json", subset_snapshot)
    write_json(out_root / "simulation_trajectory_aggregate_unique_smiles.json", subset_unique_manifest)
    write_json(out_root / "simulation_trajectory_aggregate_row_map.json", subset_row_map)
    return out_root


def run_generate(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = dict(os.environ)
    merged_env["PYTHONPATH"] = "src"
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, "tools/run_csv_scope_audit/generate.py", *args],
        cwd=REPO_ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_snapshot_manifest_generation_is_deterministic_for_duplicate_smiles(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation-trajectory-aggregate.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Trajectory ID,SMILES,Molality,Monomer Molecular Weight,Degree of Polymerization,Density,CONDUCTIVITY,TFSI Diffusivity,Li Diffusivity,Poly Diffusivity,Transference Number",
                "1,*CO*,1,10,2,1,1,1,1,1,1",
                "2,*CC*,1,10,2,1,1,1,1,1,1",
                "3,*CO*,1,10,2,1,1,1,1,1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out_dir in (out_a, out_b):
        run_generate("snapshot", "--csv", str(csv_path), "--out-manifests", str(out_dir))

    expected_files = {
        "simulation_trajectory_aggregate_row_map.json",
        "simulation_trajectory_aggregate_snapshot.json",
        "simulation_trajectory_aggregate_unique_smiles.json",
    }
    assert {path.name for path in out_a.iterdir() if path.is_file()} == expected_files
    for filename in expected_files:
        assert (out_a / filename).read_text(encoding="utf-8") == (out_b / filename).read_text(encoding="utf-8")

    unique_manifest = load_json(out_a / "simulation_trajectory_aggregate_unique_smiles.json")
    assert [entry["smiles"] for entry in unique_manifest["entries"]] == ["*CC*", "*CO*"]
    assert [entry["unique_smiles_id"] for entry in unique_manifest["entries"]] == [
        "csv_scope_smiles_000001",
        "csv_scope_smiles_000002",
    ]
    assert unique_manifest["entries"][0]["adapter_input"]["monomer_smiles"] == "BrCCBr"
    assert unique_manifest["entries"][1]["adapter_input"]["monomer_smiles"] == "BrCOBr"


def test_audit_runner_is_deterministic_on_checked_in_subset(tmp_path: Path) -> None:
    manifest_root = build_subset_manifests(tmp_path)
    out_a = tmp_path / "audit_a"
    out_b = tmp_path / "audit_b"

    for out_dir in (out_a, out_b):
        run_generate("audit", "--manifest-root", str(manifest_root), "--out", str(out_dir))

    expected_files = {
        "coverage_audit_results.json",
        "coverage_audit_summary.json",
    }
    assert {path.name for path in out_a.iterdir() if path.is_file()} == expected_files
    for filename in expected_files:
        assert (out_a / filename).read_text(encoding="utf-8") == (out_b / filename).read_text(encoding="utf-8")

    results = load_json(out_a / "coverage_audit_results.json")
    summary = load_json(out_a / "coverage_audit_summary.json")
    assert results["entry_count"] == 2
    assert results["pipeline_contract"]["csv_smiles_adapter_status"] == "pysoftk_proto_polymer_active"
    assert summary["totals"]["unique_smiles_count"] == 2
    assert summary["totals"]["row_count"] >= 2
    assert sum(summary["failure_class_counts"]["unique_smiles"].values()) + summary["totals"]["supported_unique_smiles_count"] == 2


def test_audit_validate_command_matches_reference_for_checked_in_subset(tmp_path: Path) -> None:
    manifest_root = build_subset_manifests(tmp_path)
    reference_root = tmp_path / "reference"
    run_generate("audit", "--manifest-root", str(manifest_root), "--out", str(reference_root))

    completed = run_generate("validate", "--manifest-root", str(manifest_root), "--reference", str(reference_root))
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["mismatches"] == []
