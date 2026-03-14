from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "tests" / "reference_results" / "csv_scope_audit"
MANIFEST_ROOT = REPO_ROOT / "data_manifests"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_snapshot_manifest_generation_is_deterministic_for_duplicate_smiles(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation-trajectory-aggregate.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Trajectory ID,SMILES,Molality,Monomer Molecular Weight,Degree of Polymerization,Density,CONDUCTIVITY,TFSI Diffusivity,Li Diffusivity,Poly Diffusivity,Transference Number",
                "1,C*,1,10,2,1,1,1,1,1,1",
                "2,A*,1,10,2,1,1,1,1,1,1",
                "3,C*,1,10,2,1,1,1,1,1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out_dir in (out_a, out_b):
        subprocess.run(
            [
                sys.executable,
                "tools/run_csv_scope_audit/generate.py",
                "snapshot",
                "--csv",
                str(csv_path),
                "--out-manifests",
                str(out_dir),
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )

    expected_files = {
        "simulation_trajectory_aggregate_row_map.json",
        "simulation_trajectory_aggregate_snapshot.json",
        "simulation_trajectory_aggregate_unique_smiles.json",
    }
    assert {path.name for path in out_a.iterdir() if path.is_file()} == expected_files
    for filename in expected_files:
        assert (out_a / filename).read_text(encoding="utf-8") == (out_b / filename).read_text(encoding="utf-8")

    unique_manifest = load_json(out_a / "simulation_trajectory_aggregate_unique_smiles.json")
    assert [entry["smiles"] for entry in unique_manifest["entries"]] == ["A*", "C*"]
    assert [entry["unique_smiles_id"] for entry in unique_manifest["entries"]] == [
        "csv_scope_smiles_000001",
        "csv_scope_smiles_000002",
    ]


def test_audit_runner_regenerates_checked_in_reference(tmp_path: Path) -> None:
    out_dir = tmp_path / "csv_scope_audit"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    subprocess.run(
        [
            sys.executable,
            "tools/run_csv_scope_audit/generate.py",
            "audit",
            "--manifest-root",
            str(MANIFEST_ROOT),
            "--out",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    expected_files = {
        "coverage_audit_results.json",
        "coverage_audit_summary.json",
    }
    assert {path.name for path in out_dir.iterdir() if path.is_file()} == expected_files
    for filename in expected_files:
        assert (out_dir / filename).read_text(encoding="utf-8") == (REFERENCE_ROOT / filename).read_text(encoding="utf-8")


def test_audit_validate_command_matches_reference() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_csv_scope_audit/generate.py",
            "validate",
            "--manifest-root",
            str(MANIFEST_ROOT),
            "--reference",
            str(REFERENCE_ROOT),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["mismatches"] == []
