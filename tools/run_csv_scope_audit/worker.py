#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from common import (
    AUDIT_RESULTS_NAME,
    AUDIT_SUMMARY_NAME,
    CSV_SCOPE_ID,
    FAILURE_CLASSES,
    SNAPSHOT_MANIFEST_NAME,
    UNIQUE_SMILES_MANIFEST_NAME,
    load_manifests,
)


REPO_ROOT = Path(os.environ["PCFF_CSV_SCOPE_REPO_ROOT"])
PYSOFTK_ROOT = Path(os.environ["PCFF_CSV_SCOPE_PYSOFTK_ROOT"])

import sys

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(PYSOFTK_ROOT))

from rdkit import Chem
from rdkit.Chem import AllChem

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
from pysoftk.format_printers.format_mol import Fmt
from pysoftk.linear_polymer.linear_polymer import Lp
from pysoftk.tools.utils_rdkit import remove_plcholder
from typing_ir import parse_file
from typing_ir.errors import ParseError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CSV scope audit inside the MD conda environment.")
    parser.add_argument("--manifest-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = load_manifests(args.manifest_root)
    snapshot_manifest = manifests[SNAPSHOT_MANIFEST_NAME]
    unique_manifest = manifests[UNIQUE_SMILES_MANIFEST_NAME]
    entries = [_audit_unique_smiles_entry(entry) for entry in unique_manifest["entries"]]
    payload = {
        AUDIT_RESULTS_NAME: {
            "schema_name": "csv_scope_coverage_audit_results",
            "schema_version": 1,
            "scope_id": CSV_SCOPE_ID,
            "snapshot_sha256": snapshot_manifest["source_csv"]["sha256"],
            "pipeline_contract": {
                "typing_ir_supported_input_formats": ["mol2", "mol_v2000", "pdb", "sdf"],
                "polymer_workflow_supported_input_formats": ["mol_v2000"],
                "csv_smiles_adapter_status": "pysoftk_proto_polymer_active",
                "csv_smiles_adapter_python": sys.executable,
                "csv_smiles_adapter_placeholder": "Br",
                "csv_smiles_adapter_output_format": "mol2",
                "csv_smiles_adapter_pysoftk_root": str(PYSOFTK_ROOT),
            },
            "entry_count": len(entries),
            "entries": entries,
        },
        AUDIT_SUMMARY_NAME: _build_summary(snapshot_manifest=snapshot_manifest, entries=entries),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _audit_unique_smiles_entry(entry: dict) -> dict:
    try:
        with tempfile.TemporaryDirectory(prefix=f"{entry['unique_smiles_id']}_") as tmp_dir:
            mol2_path = Path(tmp_dir) / f"{entry['unique_smiles_id']}.mol2"
            _write_pysoftk_mol2(entry, mol2_path)
            ir = parse_file(mol2_path, input_format="mol2", source_id=entry["unique_smiles_id"])
            perception = perceive_ir(ir)
            typing_report = type_ir(ir, perception=perception)
            if typing_report["typing"]["status"] != "typed":
                return _typing_status_failure(entry, typing_report)

            bonded_report = assign_bonded_ir(ir, typing_report=typing_report, perception=perception)
            if bonded_report["parameter_assignment"]["status"] != "assigned":
                return _report_status_failure(
                    entry,
                    failure_class="parameter_assignment_failure",
                    stopping_stage="parameter_assignment",
                    report=bonded_report,
                    status_key="parameter_assignment",
                )

            nonbonded_report = assign_nonbonded_ir(ir, typing_report=typing_report, bonded_report=bonded_report)
            if nonbonded_report["nonbonded_assignment"]["status"] != "assigned":
                return _report_status_failure(
                    entry,
                    failure_class="nonbonded_assignment_failure",
                    stopping_stage="nonbonded_assignment",
                    report=nonbonded_report,
                    status_key="nonbonded_assignment",
                )

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
    except Exception as error:  # pragma: no cover - unexpected worker failures must still surface explicitly
        return _failure_entry(entry, "parse_failure", "parse", "csv_smiles_adapter_failure", repr(error))

    return {
        "unique_smiles_id": entry["unique_smiles_id"],
        "smiles": entry["smiles"],
        "smiles_sha256": entry["smiles_sha256"],
        "row_count": entry["row_count"],
        "first_row_number": entry["first_row_number"],
        "representative_row": entry["representative_row"],
        "adapter_input": entry["adapter_input"],
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


def _write_pysoftk_mol2(entry: dict, output_path: Path) -> None:
    adapter_input = entry["adapter_input"]
    placeholder = adapter_input["placeholder"]
    monomer_smiles = adapter_input["monomer_smiles"]
    dp = int(adapter_input["degree_of_polymerization"])

    monomer = Chem.MolFromSmiles(monomer_smiles)
    if monomer is None:
        raise ParseError("csv_smiles_adapter_failure", "RDKit failed to parse monomer SMILES")

    monomer = Chem.AddHs(monomer)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    conformer_id = AllChem.EmbedMolecule(monomer, params)
    if conformer_id < 0:
        raise ParseError("csv_smiles_adapter_failure", "RDKit 3D embedding failed for monomer SMILES")
    monomer = Chem.RemoveHs(monomer)

    chain = Lp(monomer, placeholder, dp, shift=1.0).proto_polymer()
    chain = remove_plcholder(chain, placeholder)
    Fmt(chain).format_print(str(output_path))


def _typing_status_failure(entry: dict, typing_report: dict) -> dict:
    first = _first_diagnostic(typing_report, component_key="components")
    return _failure_entry(
        entry,
        "atom_typing_failure",
        "atom_typing",
        first["code"],
        first["message"],
    )


def _report_status_failure(entry: dict, *, failure_class: str, stopping_stage: str, report: dict, status_key: str) -> dict:
    first = _first_diagnostic(report, component_key="components")
    return _failure_entry(
        entry,
        failure_class,
        stopping_stage,
        first["code"],
        first["message"],
    )


def _first_diagnostic(report: dict, *, component_key: str) -> dict:
    components = report.get(component_key, [])
    if not components:
        return {
            "code": "missing_report_diagnostic",
            "message": "Report status was non-success but no component diagnostics were present.",
        }
    diagnostics = components[0].get("diagnostics", [])
    if not diagnostics:
        return {
            "code": "missing_report_diagnostic",
            "message": "Report status was non-success but no diagnostics were present.",
        }
    first = diagnostics[0]
    return {
        "code": first.get("code", "missing_report_diagnostic_code"),
        "message": first.get("message", "Report diagnostic did not contain a message."),
    }


def _failure_entry(entry: dict, failure_class: str, stopping_stage: str, code: str, message: str) -> dict:
    return {
        "unique_smiles_id": entry["unique_smiles_id"],
        "smiles": entry["smiles"],
        "smiles_sha256": entry["smiles_sha256"],
        "row_count": entry["row_count"],
        "first_row_number": entry["first_row_number"],
        "representative_row": entry["representative_row"],
        "adapter_input": entry["adapter_input"],
        "status": "failure",
        "stopping_stage": stopping_stage,
        "failure_class": failure_class,
        "failure_code": code,
        "failure_message": message,
    }


def _build_summary(*, snapshot_manifest: dict, entries: list[dict]) -> dict:
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
        observed_failure_codes[failure_class][entry["failure_code"]] = (
            observed_failure_codes[failure_class].get(entry["failure_code"], 0) + 1
        )

    unique_total = len(entries)
    row_total = snapshot_manifest["row_count"]
    return {
        "schema_name": "csv_scope_coverage_summary",
        "schema_version": 1,
        "scope_id": CSV_SCOPE_ID,
        "snapshot_sha256": snapshot_manifest["source_csv"]["sha256"],
        "scope_boundary": {
            "snapshot_manifest_path": "data_manifests/simulation_trajectory_aggregate_snapshot.json",
            "unique_manifest_path": "data_manifests/simulation_trajectory_aggregate_unique_smiles.json",
            "row_map_path": "data_manifests/simulation_trajectory_aggregate_row_map.json",
        },
        "coverage_target": "100_percent_typing_export_coverage_required_for_csv_snapshot",
        "current_pipeline_contract": {
            "typing_ir_supported_input_formats": ["mol2", "mol_v2000", "pdb", "sdf"],
            "polymer_workflow_supported_input_formats": ["mol_v2000"],
            "csv_smiles_adapter_status": "pysoftk_proto_polymer_active",
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
            "status": "ready" if passed_unique == unique_total else "not_ready",
            "reason": "CSV snapshot is release-ready only when every unique SMILES reaches emitter_export successfully.",
        },
    }


if __name__ == "__main__":
    main()
