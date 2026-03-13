from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .canonical import refine_atom_order
from .errors import SchemaError


SCHEMA_NAME = "typed_system"
SCHEMA_VERSION = 1
IR_STAGE = "parsed_only"
CANONICALIZATION_ALGORITHM = "graph_refine_distance_v1"


def build_ir(parsed: dict, *, input_format: str, source_id: str, source_bytes: bytes) -> dict:
    ordered_atoms = refine_atom_order(parsed["atoms"], parsed["bonds"])
    source_to_canonical = {
        atom["source_index"]: canonical_index
        for canonical_index, atom in enumerate(ordered_atoms, start=1)
    }

    atoms = []
    for canonical_index, atom in enumerate(ordered_atoms, start=1):
        atoms.append(
            {
                "canonical_index": canonical_index,
                "source_index": atom["source_index"],
                "source_atom_id": atom["source_atom_id"],
                "element": atom["element"],
                "atom_name": atom["atom_name"],
                "formal_charge": atom["formal_charge"],
                "partial_charge": atom["partial_charge"],
                "coordinates": atom["coordinates"],
                "annotations": atom["annotations"],
                "provenance": atom["provenance"],
            }
        )

    ordered_bonds = sorted(
        parsed["bonds"],
        key=lambda bond: (
            min(source_to_canonical[index] for index in bond["atom_source_indices"]),
            max(source_to_canonical[index] for index in bond["atom_source_indices"]),
            bond["bond_code"],
            "" if bond["order"] is None else str(bond["order"]),
            bond["source_index"],
        ),
    )
    bonds = []
    for canonical_index, bond in enumerate(ordered_bonds, start=1):
        atom_indices = sorted(source_to_canonical[index] for index in bond["atom_source_indices"])
        bonds.append(
            {
                "canonical_index": canonical_index,
                "source_index": bond["source_index"],
                "source_bond_id": bond["source_bond_id"],
                "source_atom_indices": bond["atom_source_indices"],
                "atom_indices": atom_indices,
                "order": bond["order"],
                "bond_code": bond["bond_code"],
                "annotations": bond["annotations"],
                "provenance": bond["provenance"],
            }
        )

    element_counts: dict[str, int] = {}
    for atom in atoms:
        element_counts.setdefault(atom["element"], 0)
        element_counts[atom["element"]] += 1

    bond_order_histogram: dict[str, int] = {}
    bond_code_histogram: dict[str, int] = {}
    for bond in bonds:
        if bond["order"] is not None:
            key = str(bond["order"])
            bond_order_histogram.setdefault(key, 0)
            bond_order_histogram[key] += 1
        bond_code_histogram.setdefault(bond["bond_code"], 0)
        bond_code_histogram[bond["bond_code"]] += 1

    net_formal_charge = None
    formal_charges = [atom["formal_charge"] for atom in atoms]
    if all(charge is not None for charge in formal_charges):
        net_formal_charge = sum(formal_charges)

    partial_charge_sum = None
    partial_charges = [atom["partial_charge"] for atom in atoms]
    if all(charge is not None for charge in partial_charges):
        partial_charge_sum = round(sum(partial_charges), 8)

    ir = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "ir_stage": IR_STAGE,
        "canonicalization": {
            "algorithm": CANONICALIZATION_ALGORITHM,
            "atom_index_base": 1,
        },
        "source": {
            "source_id": source_id,
            "input_format": input_format,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "typing": {
            "status": "not_run",
            "ruleset_id": None,
        },
        "components": [
            {
                "component_id": "component_1",
                "name": parsed["title"],
                "atom_count": len(atoms),
                "bond_count": len(bonds),
                "element_counts": dict(sorted(element_counts.items())),
                "bond_order_histogram": dict(sorted(bond_order_histogram.items())),
                "bond_code_histogram": dict(sorted(bond_code_histogram.items())),
                "net_formal_charge": net_formal_charge,
                "partial_charge_sum": partial_charge_sum,
                "atoms": atoms,
                "bonds": bonds,
            }
        ],
    }
    validate_ir(ir)
    return ir


def dumps_ir(ir: dict) -> str:
    validate_ir(ir)
    return json.dumps(ir, indent=2, sort_keys=True) + "\n"


def write_ir(path: Path, ir: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_ir(ir), encoding="utf-8")


def loads_ir(text: str) -> dict:
    ir = json.loads(text)
    validate_ir(ir)
    return ir


def load_ir(path: Path) -> dict:
    return loads_ir(path.read_text(encoding="utf-8"))


def validate_ir(ir: dict) -> None:
    if ir.get("schema_name") != SCHEMA_NAME:
        raise SchemaError("invalid_ir_schema", "schema_name must be 'typed_system'")
    if ir.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError("invalid_ir_schema", "Unsupported schema_version")
    if ir.get("ir_stage") != IR_STAGE:
        raise SchemaError("invalid_ir_schema", "IR stage must be 'parsed_only'")
    if ir.get("typing", {}).get("status") != "not_run":
        raise SchemaError("invalid_ir_schema", "PT1 IR must keep typing.status='not_run'")

    source = ir.get("source")
    if not isinstance(source, dict):
        raise SchemaError("invalid_ir_schema", "source must be a mapping")
    for key in {"source_id", "input_format", "sha256"}:
        if key not in source:
            raise SchemaError("invalid_ir_schema", f"source.{key} is required")

    components = ir.get("components")
    if not isinstance(components, list) or len(components) != 1:
        raise SchemaError("invalid_ir_schema", "PT1 IR must contain exactly one component")

    component = components[0]
    atoms = component.get("atoms")
    bonds = component.get("bonds")
    if not isinstance(atoms, list) or not isinstance(bonds, list):
        raise SchemaError("invalid_ir_schema", "component atoms and bonds must be lists")
    if component.get("atom_count") != len(atoms):
        raise SchemaError("invalid_ir_schema", "component atom_count mismatch")
    if component.get("bond_count") != len(bonds):
        raise SchemaError("invalid_ir_schema", "component bond_count mismatch")

    for expected_index, atom in enumerate(atoms, start=1):
        if atom.get("canonical_index") != expected_index:
            raise SchemaError("invalid_ir_schema", "Atom canonical indices must be contiguous")
        for key in {"source_index", "source_atom_id", "element", "coordinates", "provenance", "annotations"}:
            if key not in atom:
                raise SchemaError("invalid_ir_schema", f"Atom field {key!r} is required")

    for expected_index, bond in enumerate(bonds, start=1):
        if bond.get("canonical_index") != expected_index:
            raise SchemaError("invalid_ir_schema", "Bond canonical indices must be contiguous")
        for key in {
            "source_index",
            "source_bond_id",
            "source_atom_indices",
            "atom_indices",
            "bond_code",
            "provenance",
            "annotations",
        }:
            if key not in bond:
                raise SchemaError("invalid_ir_schema", f"Bond field {key!r} is required")
