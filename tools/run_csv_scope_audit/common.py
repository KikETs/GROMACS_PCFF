from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from atom_typing import type_ir
from atom_typing.errors import AtomTypingError
from chem_perception import perceive_ir
from chem_perception.errors import ChemPerceptionError
from emitters.gromacs import emit_ir
from emitters.gromacs.errors import GromacsEmitterError
from nonbonded_assignment import assign_ir as assign_nonbonded_ir
from nonbonded_assignment.errors import NonbondedAssignmentError
from parameter_assignment import assign_ir as assign_bonded_ir
from parameter_assignment.errors import ParameterAssignmentError
from typing_ir.errors import ParseError
from typing_ir.formats import FORMAT_ALIASES


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
MANIFEST_ROOT = REPO_ROOT / "data_manifests"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "tests" / "reference_results" / "csv_scope_audit"
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
PIPELINE_SUPPORTED_INPUT_FORMATS = sorted(set(FORMAT_ALIASES.values()))


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

    unique_entries = []
    unique_id_by_smiles: dict[str, str] = {}
    row_count_by_smiles = Counter(row["SMILES"] for row in rows)
    first_row_by_smiles: dict[str, int] = {}
    for row_number, row in enumerate(rows, start=1):
        smiles = row["SMILES"]
        if smiles not in first_row_by_smiles:
            first_row_by_smiles[smiles] = row_number

    sorted_smiles = sorted(row_count_by_smiles)
    for index, smiles in enumerate(sorted_smiles, start=1):
        unique_smiles_id = f"csv_scope_smiles_{index:06d}"
        unique_id_by_smiles[smiles] = unique_smiles_id
        unique_entries.append(
            {
                "unique_smiles_id": unique_smiles_id,
                "smiles": smiles,
                "smiles_sha256": _sha256_text(smiles),
                "row_count": row_count_by_smiles[smiles],
                "first_row_number": first_row_by_smiles[smiles],
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
    manifests = load_manifests(manifest_root)
    snapshot_manifest = manifests[SNAPSHOT_MANIFEST_NAME]
    unique_manifest = manifests[UNIQUE_SMILES_MANIFEST_NAME]
    row_map = manifests[ROW_MAP_NAME]

    entries = [_audit_unique_smiles_entry(entry) for entry in unique_manifest["entries"]]
    results = {
        "schema_name": "csv_scope_coverage_audit_results",
        "schema_version": 1,
        "scope_id": CSV_SCOPE_ID,
        "snapshot_sha256": snapshot_manifest["source_csv"]["sha256"],
        "pipeline_contract": {
            "typing_ir_supported_input_formats": PIPELINE_SUPPORTED_INPUT_FORMATS,
            "polymer_workflow_supported_input_formats": ["mol_v2000"],
            "csv_smiles_adapter_status": "unsupported",
            "smiles_scope_boundary": SNAPSHOT_FILENAME,
        },
        "entry_count": len(entries),
        "entries": entries,
    }
    summary = _build_audit_summary(
        snapshot_manifest=snapshot_manifest,
        unique_manifest=unique_manifest,
        row_map=row_map,
        entries=entries,
    )
    return {
        AUDIT_RESULTS_NAME: results,
        AUDIT_SUMMARY_NAME: summary,
    }


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


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _audit_unique_smiles_entry(entry: dict) -> dict:
    try:
        ir = _parse_unique_smiles_entry(entry)
        perception = perceive_ir(ir)
        typing_report = type_ir(ir, perception=perception)
        bonded_report = assign_bonded_ir(ir, typing_report=typing_report, perception=perception)
        nonbonded_report = assign_nonbonded_ir(ir, typing_report=typing_report, bonded_report=bonded_report)
        emit_ir(
            ir,
            typing_report=typing_report,
            bonded_report=bonded_report,
            nonbonded_report=nonbonded_report,
            dry_run=True,
        )
    except ParseError as error:
        return _failure_entry(entry, "parse_failure", "parse", error.code, error.message)
    except ChemPerceptionError as error:
        return _failure_entry(entry, "chemical_perception_failure", "chemical_perception", error.code, error.message)
    except AtomTypingError as error:
        return _failure_entry(entry, "atom_typing_failure", "atom_typing", error.code, error.message)
    except ParameterAssignmentError as error:
        return _failure_entry(entry, "parameter_assignment_failure", "parameter_assignment", error.code, error.message)
    except NonbondedAssignmentError as error:
        return _failure_entry(entry, "nonbonded_assignment_failure", "nonbonded_assignment", error.code, error.message)
    except GromacsEmitterError as error:
        return _failure_entry(entry, "emitter_export_failure", "emitter_export", error.code, error.message)

    return {
        "unique_smiles_id": entry["unique_smiles_id"],
        "smiles": entry["smiles"],
        "smiles_sha256": entry["smiles_sha256"],
        "row_count": entry["row_count"],
        "first_row_number": entry["first_row_number"],
        "status": "pass",
        "completed_stages": [
            "parse",
            "chemical_perception",
            "atom_typing",
            "parameter_assignment",
            "nonbonded_assignment",
            "emitter_export",
        ],
    }


def _parse_unique_smiles_entry(entry: dict) -> dict:
    raise ParseError(
        "unsupported_csv_smiles_input",
        "Current typing/export pipeline accepts file-backed mol_v2000/sdf/mol2/pdb inputs only; "
        "the CSV scope provides SMILES strings and no deterministic SMILES-to-structure adapter exists.",
    )


def _failure_entry(entry: dict, failure_class: str, stage: str, code: str, message: str) -> dict:
    return {
        "unique_smiles_id": entry["unique_smiles_id"],
        "smiles": entry["smiles"],
        "smiles_sha256": entry["smiles_sha256"],
        "row_count": entry["row_count"],
        "first_row_number": entry["first_row_number"],
        "status": "failure",
        "stopping_stage": stage,
        "failure_class": failure_class,
        "failure_code": code,
        "failure_message": message,
    }


def _build_audit_summary(
    *,
    snapshot_manifest: dict,
    unique_manifest: dict,
    row_map: dict,
    entries: list[dict],
) -> dict:
    unique_failure_counts = {key: 0 for key in FAILURE_CLASSES}
    row_failure_counts = {key: 0 for key in FAILURE_CLASSES}
    observed_failure_codes = {key: {} for key in FAILURE_CLASSES}

    passed_unique = 0
    passed_rows = 0
    for entry in entries:
        if entry["status"] == "pass":
            passed_unique += 1
            passed_rows += entry["row_count"]
            continue
        failure_class = entry["failure_class"]
        unique_failure_counts[failure_class] += 1
        row_failure_counts[failure_class] += entry["row_count"]
        codes = observed_failure_codes[failure_class]
        codes[entry["failure_code"]] = codes.get(entry["failure_code"], 0) + 1

    unique_total = unique_manifest["unique_smiles_count"]
    row_total = row_map["row_count"]
    return {
        "schema_name": "csv_scope_coverage_summary",
        "schema_version": 1,
        "scope_id": CSV_SCOPE_ID,
        "snapshot_sha256": snapshot_manifest["source_csv"]["sha256"],
        "scope_boundary": {
            "snapshot_manifest_path": f"data_manifests/{SNAPSHOT_MANIFEST_NAME}",
            "unique_manifest_path": f"data_manifests/{UNIQUE_SMILES_MANIFEST_NAME}",
            "row_map_path": f"data_manifests/{ROW_MAP_NAME}",
        },
        "coverage_target": "100_percent_typing_export_coverage_required_for_csv_snapshot",
        "current_pipeline_contract": {
            "typing_ir_supported_input_formats": PIPELINE_SUPPORTED_INPUT_FORMATS,
            "polymer_workflow_supported_input_formats": ["mol_v2000"],
            "csv_smiles_adapter_status": "unsupported",
        },
        "totals": {
            "row_count": row_total,
            "unique_smiles_count": unique_total,
            "supported_row_count": passed_rows,
            "supported_unique_smiles_count": passed_unique,
            "row_coverage_fraction": passed_rows / row_total if row_total else 0.0,
            "unique_coverage_fraction": passed_unique / unique_total if unique_total else 0.0,
        },
        "failure_class_counts": {
            "unique_smiles": unique_failure_counts,
            "rows": row_failure_counts,
        },
        "observed_failure_codes": observed_failure_codes,
        "release_readiness": {
            "status": "not_ready",
            "reason": "The current pipeline has no deterministic SMILES input path for the CSV snapshot scope.",
        },
    }
