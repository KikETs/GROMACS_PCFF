from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
MANIFEST_ROOT = REPO_ROOT / "data_manifests"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "tests" / "reference_results" / "csv_scope_audit"
WORKER_SCRIPT = REPO_ROOT / "tools" / "run_csv_scope_audit" / "worker.py"
DEFAULT_ADAPTER_PYTHON = Path("/home/kiket/anaconda3/envs/MD/bin/python")
DEFAULT_PYSOFTK_ROOT = WORKSPACE_ROOT / "torch" / "pysoftk"
SNAPSHOT_FILENAME = "simulation-trajectory-aggregate.csv"

SNAPSHOT_MANIFEST_NAME = "simulation_trajectory_aggregate_snapshot.json"
UNIQUE_SMILES_MANIFEST_NAME = "simulation_trajectory_aggregate_unique_smiles.json"
ROW_MAP_NAME = "simulation_trajectory_aggregate_row_map.json"
MANIFEST_FILENAMES = (
    SNAPSHOT_MANIFEST_NAME,
    UNIQUE_SMILES_MANIFEST_NAME,
    ROW_MAP_NAME,
)

AUDIT_RESULTS_NAME = "coverage_audit_results.json"
AUDIT_SUMMARY_NAME = "coverage_audit_summary.json"

CSV_SCOPE_ID = "simulation_trajectory_aggregate_csv_snapshot_release_target"
REQUIRED_COLUMNS = (
    "Trajectory ID",
    "SMILES",
    "Molality",
    "Monomer Molecular Weight",
    "Degree of Polymerization",
    "Density",
    "CONDUCTIVITY",
    "TFSI Diffusivity",
    "Li Diffusivity",
    "Poly Diffusivity",
    "Transference Number",
)
FAILURE_CLASSES = (
    "parse_failure",
    "chemical_perception_failure",
    "atom_typing_failure",
    "parameter_assignment_failure",
    "nonbonded_assignment_failure",
    "emitter_export_failure",
)


def dump_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_outputs(out_dir: str | Path, outputs: dict[str, dict]) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for filename, payload in outputs.items():
        (root / filename).write_text(dump_json(payload), encoding="utf-8")


def build_manifests(csv_path: str | Path) -> dict[str, dict]:
    source_path = Path(csv_path).resolve()
    rows, fieldnames = _read_csv_rows(source_path)
    snapshot_sha256 = _sha256_bytes(source_path.read_bytes())
    matching_paths = _discover_matching_paths(snapshot_sha256)

    rows_by_smiles: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_smiles[row["SMILES"]].append(row)

    unique_entries = []
    unique_id_by_smiles: dict[str, str] = {}
    sorted_smiles = sorted(rows_by_smiles)
    for index, smiles in enumerate(sorted_smiles, start=1):
        unique_smiles_id = f"csv_scope_smiles_{index:06d}"
        unique_id_by_smiles[smiles] = unique_smiles_id
        representative_rows = rows_by_smiles[smiles]
        first_row = representative_rows[0]
        dp_values = sorted({int(round(float(row["Degree of Polymerization"]))) for row in representative_rows})
        if len(dp_values) != 1:
            raise ValueError(f"Unique SMILES {smiles!r} maps to multiple polymerization degrees: {dp_values}")
        molality_values = sorted({row["Molality"] for row in representative_rows})
        unique_entries.append(
            {
                "unique_smiles_id": unique_smiles_id,
                "smiles": smiles,
                "smiles_sha256": _sha256_text(smiles),
                "row_count": len(representative_rows),
                "first_row_number": _row_number_of(rows, first_row["Trajectory ID"], smiles),
                "representative_row": {
                    "trajectory_id": first_row["Trajectory ID"],
                    "degree_of_polymerization": dp_values[0],
                    "degree_of_polymerization_raw": first_row["Degree of Polymerization"],
                    "molality_raw": first_row["Molality"],
                    "distinct_molality_values": molality_values,
                },
                "adapter_input": {
                    "adapter_kind": "pysoftk_proto_polymer",
                    "placeholder": "Br",
                    "degree_of_polymerization": dp_values[0],
                    "monomer_smiles": _psmiles_to_monomer_smiles(smiles, placeholder="Br"),
                    "output_format": "mol2",
                },
            }
        )

    snapshot_manifest = {
        "schema_name": "csv_scope_snapshot_manifest",
        "schema_version": 1,
        "scope_id": CSV_SCOPE_ID,
        "snapshot_filename": SNAPSHOT_FILENAME,
        "source_csv": {
            "workspace_root": str(WORKSPACE_ROOT),
            "provided_path": _workspace_display(source_path),
            "canonical_path": _workspace_display(source_path),
            "matching_paths": [_workspace_display(path) for path in matching_paths],
            "sha256": snapshot_sha256,
        },
        "row_identifier_column": "Trajectory ID",
        "smiles_column": "SMILES",
        "column_names": fieldnames,
        "row_count": len(rows),
        "unique_smiles_count": len(unique_entries),
        "duplicate_row_count": len(rows) - len(unique_entries),
        "deterministic_id_policy": {
            "sort_order": "lexicographic exact SMILES string",
            "id_format": "csv_scope_smiles_%06d",
        },
        "audit_adapter_contract": {
            "adapter_kind": "pysoftk_proto_polymer",
            "placeholder": "Br",
            "output_format": "mol2",
            "structure_generation_policy": "rdkit_embed_then_pysoftk_proto_polymer_then_placeholder_to_hydrogen",
        },
        "manifests": {
            "unique_smiles_path": f"data_manifests/{UNIQUE_SMILES_MANIFEST_NAME}",
            "row_map_path": f"data_manifests/{ROW_MAP_NAME}",
        },
    }

    unique_manifest = {
        "schema_name": "csv_scope_unique_smiles_manifest",
        "schema_version": 1,
        "scope_id": CSV_SCOPE_ID,
        "snapshot_sha256": snapshot_sha256,
        "unique_smiles_count": len(unique_entries),
        "entries": unique_entries,
    }

    row_map = {
        "schema_name": "csv_scope_row_map",
        "schema_version": 1,
        "scope_id": CSV_SCOPE_ID,
        "snapshot_sha256": snapshot_sha256,
        "row_count": len(rows),
        "rows": [
            {
                "row_number": row_number,
                "trajectory_id": row["Trajectory ID"],
                "unique_smiles_id": unique_id_by_smiles[row["SMILES"]],
            }
            for row_number, row in enumerate(rows, start=1)
        ],
    }

    validate_manifests(snapshot_manifest, unique_manifest, row_map)
    return {
        SNAPSHOT_MANIFEST_NAME: snapshot_manifest,
        UNIQUE_SMILES_MANIFEST_NAME: unique_manifest,
        ROW_MAP_NAME: row_map,
    }


def load_manifests(manifest_root: str | Path = MANIFEST_ROOT) -> dict[str, dict]:
    root = Path(manifest_root)
    manifests = {}
    for filename in MANIFEST_FILENAMES:
        path = root / filename
        manifests[filename] = json.loads(path.read_text(encoding="utf-8"))
    validate_manifests(
        manifests[SNAPSHOT_MANIFEST_NAME],
        manifests[UNIQUE_SMILES_MANIFEST_NAME],
        manifests[ROW_MAP_NAME],
    )
    return manifests


def validate_manifests(snapshot_manifest: dict, unique_manifest: dict, row_map: dict) -> None:
    if snapshot_manifest.get("schema_name") != "csv_scope_snapshot_manifest":
        raise ValueError("snapshot manifest schema_name mismatch")
    if unique_manifest.get("schema_name") != "csv_scope_unique_smiles_manifest":
        raise ValueError("unique smiles manifest schema_name mismatch")
    if row_map.get("schema_name") != "csv_scope_row_map":
        raise ValueError("row map schema_name mismatch")
    if snapshot_manifest.get("schema_version") != 1:
        raise ValueError("snapshot manifest schema_version mismatch")
    if unique_manifest.get("schema_version") != 1:
        raise ValueError("unique manifest schema_version mismatch")
    if row_map.get("schema_version") != 1:
        raise ValueError("row map schema_version mismatch")

    if snapshot_manifest["scope_id"] != CSV_SCOPE_ID:
        raise ValueError("snapshot manifest scope_id mismatch")
    if unique_manifest["scope_id"] != CSV_SCOPE_ID:
        raise ValueError("unique manifest scope_id mismatch")
    if row_map["scope_id"] != CSV_SCOPE_ID:
        raise ValueError("row map scope_id mismatch")

    if snapshot_manifest["smiles_column"] != "SMILES":
        raise ValueError("snapshot smiles column mismatch")
    if snapshot_manifest["row_identifier_column"] != "Trajectory ID":
        raise ValueError("snapshot row identifier column mismatch")
    if snapshot_manifest["snapshot_filename"] != SNAPSHOT_FILENAME:
        raise ValueError("snapshot filename mismatch")
    if snapshot_manifest["column_names"] != list(REQUIRED_COLUMNS):
        raise ValueError("snapshot column set mismatch")
    if snapshot_manifest["audit_adapter_contract"]["adapter_kind"] != "pysoftk_proto_polymer":
        raise ValueError("snapshot adapter kind mismatch")

    entries = unique_manifest["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("unique manifest entries must be non-empty")
    sorted_smiles = sorted(entry["smiles"] for entry in entries)
    if [entry["smiles"] for entry in entries] != sorted_smiles:
        raise ValueError("unique manifest entries must be lexicographically sorted by smiles")

    expected_ids = [f"csv_scope_smiles_{index:06d}" for index in range(1, len(entries) + 1)]
    if [entry["unique_smiles_id"] for entry in entries] != expected_ids:
        raise ValueError("unique manifest ids must be contiguous and deterministic")

    unique_index = {entry["unique_smiles_id"]: entry for entry in entries}
    if len(unique_index) != len(entries):
        raise ValueError("duplicate unique_smiles_id in unique manifest")

    for entry in entries:
        if entry["adapter_input"]["adapter_kind"] != "pysoftk_proto_polymer":
            raise ValueError("unexpected adapter kind in unique entry")
        if entry["adapter_input"]["placeholder"] != "Br":
            raise ValueError("unexpected placeholder in unique entry")
        if entry["adapter_input"]["output_format"] != "mol2":
            raise ValueError("unexpected adapter output format in unique entry")
        if entry["adapter_input"]["degree_of_polymerization"] != entry["representative_row"]["degree_of_polymerization"]:
            raise ValueError("adapter DP mismatch in unique entry")

    if snapshot_manifest["unique_smiles_count"] != len(entries):
        raise ValueError("snapshot unique_smiles_count mismatch")
    if unique_manifest["unique_smiles_count"] != len(entries):
        raise ValueError("unique manifest unique_smiles_count mismatch")
    if row_map["row_count"] != len(row_map["rows"]):
        raise ValueError("row map row_count mismatch")
    if snapshot_manifest["row_count"] != row_map["row_count"]:
        raise ValueError("snapshot row_count mismatch")
    if snapshot_manifest["duplicate_row_count"] != snapshot_manifest["row_count"] - len(entries):
        raise ValueError("snapshot duplicate row count mismatch")
    if snapshot_manifest["source_csv"]["sha256"] != unique_manifest["snapshot_sha256"]:
        raise ValueError("snapshot/unique sha256 mismatch")
    if snapshot_manifest["source_csv"]["sha256"] != row_map["snapshot_sha256"]:
        raise ValueError("snapshot/row map sha256 mismatch")

    reconstructed_counts = Counter()
    for expected_row_number, row in enumerate(row_map["rows"], start=1):
        if row["row_number"] != expected_row_number:
            raise ValueError("row map row numbers must be contiguous")
        unique_smiles_id = row["unique_smiles_id"]
        if unique_smiles_id not in unique_index:
            raise ValueError(f"row map references unknown unique_smiles_id {unique_smiles_id!r}")
        reconstructed_counts[unique_smiles_id] += 1

    for entry in entries:
        if reconstructed_counts[entry["unique_smiles_id"]] != entry["row_count"]:
            raise ValueError(f"row count mismatch for {entry['unique_smiles_id']}")


def build_audit_outputs(manifest_root: str | Path = MANIFEST_ROOT) -> dict[str, dict]:
    load_manifests(manifest_root)
    return _run_worker(manifest_root)


def compare_outputs_to_reference(reference_root: str | Path, outputs: dict[str, dict]) -> list[str]:
    root = Path(reference_root)
    mismatches = []
    for filename, payload in outputs.items():
        path = root / filename
        if not path.is_file():
            mismatches.append(f"missing:{filename}")
            continue
        if path.read_text(encoding="utf-8") != dump_json(payload):
            mismatches.append(f"mismatch:{filename}")
    for path in root.iterdir():
        if path.is_file() and path.name not in outputs:
            mismatches.append(f"unexpected:{path.name}")
    return mismatches


def _run_worker(manifest_root: str | Path) -> dict[str, dict]:
    adapter_python = _resolve_adapter_python()
    pysoftk_root = _resolve_pysoftk_root()
    env = dict(os.environ)
    env["PCFF_CSV_SCOPE_PYSOFTK_ROOT"] = str(pysoftk_root)
    env["PCFF_CSV_SCOPE_REPO_ROOT"] = str(REPO_ROOT)
    completed = subprocess.run(
        [
            str(adapter_python),
            str(WORKER_SCRIPT),
            "--manifest-root",
            str(Path(manifest_root).resolve()),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    expected = {AUDIT_RESULTS_NAME, AUDIT_SUMMARY_NAME}
    if set(payload) != expected:
        raise ValueError(f"worker returned unexpected payload keys: {sorted(payload)}")
    return payload


def _resolve_adapter_python() -> Path:
    path = Path(os.environ.get("PCFF_CSV_SCOPE_ADAPTER_PYTHON", DEFAULT_ADAPTER_PYTHON))
    if not path.is_file():
        raise FileNotFoundError(f"CSV scope adapter python not found: {path}")
    return path


def _resolve_pysoftk_root() -> Path:
    path = Path(os.environ.get("PCFF_CSV_SCOPE_PYSOFTK_ROOT", DEFAULT_PYSOFTK_ROOT))
    if not path.is_dir():
        raise FileNotFoundError(f"CSV scope pysoftk root not found: {path}")
    return path


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames != list(REQUIRED_COLUMNS):
            raise ValueError(f"Unexpected CSV columns in {path}")
        rows = []
        for row in reader:
            if row is None:
                raise ValueError(f"CSV reader returned empty row in {path}")
            rows.append({key: value for key, value in row.items()})
    return rows, list(REQUIRED_COLUMNS)


def _discover_matching_paths(snapshot_sha256: str) -> list[Path]:
    matches = []
    for path in sorted(WORKSPACE_ROOT.rglob(SNAPSHOT_FILENAME)):
        if _sha256_bytes(path.read_bytes()) == snapshot_sha256:
            matches.append(path.resolve())
    return matches


def _workspace_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path.resolve())


def _row_number_of(rows: list[dict[str, str]], trajectory_id: str, smiles: str) -> int:
    for index, row in enumerate(rows, start=1):
        if row["Trajectory ID"] == trajectory_id and row["SMILES"] == smiles:
            return index
    raise ValueError(f"row not found for trajectory_id={trajectory_id!r}")


def _psmiles_to_monomer_smiles(psmiles: str, *, placeholder: str) -> str:
    if not isinstance(psmiles, str) or not psmiles.strip():
        raise ValueError("psmiles must be a non-empty string")

    rendered = psmiles.strip()
    if rendered.count("[*]") >= 2:
        for _ in range(2):
            rendered = rendered.replace("[*]", placeholder, 1)
        return rendered.replace("[*]", "")

    if rendered.count("*") < 2:
        raise ValueError(f"Could not find at least two wildcard endpoints in pSMILES: {psmiles}")

    for _ in range(2):
        rendered = rendered.replace("*", placeholder, 1)
    return rendered.replace("*", "")


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()
