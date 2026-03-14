from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from atom_typing import dumps_typing_report, type_ir, validate_typing_report
from nonbonded_assignment import (
    assign_ir as assign_nonbonded_ir,
    dumps_assignment_report as dumps_nonbonded_report,
    validate_assignment_report as validate_nonbonded_report,
)
from parameter_assignment import (
    assign_ir as assign_bonded_ir,
    dumps_assignment_report as dumps_bonded_report,
    validate_assignment_report as validate_bonded_report,
)
from typing_ir import dumps_ir, parse_file, validate_ir

from .errors import GromacsEmitterError, ManifestError
from .formatting import (
    ELEMENT_MASSES,
    angstrom_to_nm,
    atom_name,
    bond_angle_k_to_gromacs,
    bond_bond_k_to_gromacs,
    bond_k2_to_gromacs,
    bond_k3_to_gromacs,
    bond_k4_to_gromacs,
    dihedral_bond_torsion_k_to_gromacs,
    format_float,
    kcal_to_kj,
    molecule_name,
    residue_name,
)


SCHEMA_NAME = "pcff_gromacs_emitter_manifest"
SCHEMA_VERSION = 1
BUNDLE_FILENAMES = ("forcefield_pcff.itp", "molecule.itp", "topol.top")


def emit_file(
    path: str | Path,
    *,
    input_format: str | None = None,
    source_id: str | None = None,
    out_dir: str | Path | None = None,
    dry_run: bool = False,
    validate_existing: bool = False,
) -> dict:
    ir = parse_file(path, input_format=input_format, source_id=source_id)
    typing_report = type_ir(ir)
    bonded_report = assign_bonded_ir(ir, typing_report=typing_report)
    nonbonded_report = assign_nonbonded_ir(ir, typing_report=typing_report, bonded_report=bonded_report)
    return emit_ir(
        ir,
        typing_report=typing_report,
        bonded_report=bonded_report,
        nonbonded_report=nonbonded_report,
        out_dir=out_dir,
        dry_run=dry_run,
        validate_existing=validate_existing,
    )


def emit_ir(
    ir: dict,
    *,
    typing_report: dict,
    bonded_report: dict,
    nonbonded_report: dict,
    out_dir: str | Path | None = None,
    dry_run: bool = False,
    validate_existing: bool = False,
) -> dict:
    validate_ir(ir)
    validate_typing_report(typing_report)
    validate_bonded_report(bonded_report)
    validate_nonbonded_report(nonbonded_report)

    ir_sha256 = hashlib.sha256(dumps_ir(ir).encode("utf-8")).hexdigest()
    typing_sha256 = hashlib.sha256(dumps_typing_report(typing_report).encode("utf-8")).hexdigest()
    bonded_sha256 = hashlib.sha256(dumps_bonded_report(bonded_report).encode("utf-8")).hexdigest()
    nonbonded_sha256 = hashlib.sha256(dumps_nonbonded_report(nonbonded_report).encode("utf-8")).hexdigest()

    _validate_source_chain(
        ir,
        typing_report,
        bonded_report,
        nonbonded_report,
        ir_sha256=ir_sha256,
        typing_sha256=typing_sha256,
        bonded_sha256=bonded_sha256,
    )

    if typing_report["typing"]["status"] != "typed":
        raise GromacsEmitterError(
            "typing_incomplete",
            f"gromacs emitter requires typing.status='typed', got {typing_report['typing']['status']!r}",
        )
    if bonded_report["parameter_assignment"]["status"] != "assigned":
        raise GromacsEmitterError("bonded_assignment_incomplete", "gromacs emitter requires complete bonded parameters")
    if nonbonded_report["nonbonded_assignment"]["status"] != "assigned":
        raise GromacsEmitterError(
            "nonbonded_assignment_incomplete",
            "gromacs emitter requires complete nonbonded parameters",
        )

    _validate_gromacs_representability(nonbonded_report["components"][0])

    bundle = render_bundle(
        ir,
        typing_report=typing_report,
        bonded_report=bonded_report,
        nonbonded_report=nonbonded_report,
    )
    validate_bundle(bundle)

    output_root = None if out_dir is None else Path(out_dir)
    existing_matches = None
    if validate_existing:
        if output_root is None:
            raise GromacsEmitterError("missing_output_dir", "validate_existing requires out_dir")
        existing_matches = _validate_existing_output(output_root, bundle)

    if not dry_run:
        if output_root is None:
            raise GromacsEmitterError("missing_output_dir", "emit_ir requires out_dir unless dry_run=True")
        _write_bundle(output_root, bundle)

    typing_component = typing_report["components"][0]
    bonded_component = bonded_report["components"][0]
    nonbonded_component = nonbonded_report["components"][0]
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "typed_ir_sha256": ir_sha256,
            "typing_report_sha256": typing_sha256,
            "bonded_assignment_sha256": bonded_sha256,
            "nonbonded_assignment_sha256": nonbonded_sha256,
            "source_id": ir["source"]["source_id"],
            "input_format": ir["source"]["input_format"],
        },
        "emitter": {
            "kind": "gromacs",
            "status": "dry_run" if dry_run else "written",
            "dry_run": dry_run,
            "validate_existing": validate_existing,
            "existing_output_matches_rendered": existing_matches,
        },
        "component": {
            "component_id": ir["components"][0]["component_id"],
            "system_name": ir["components"][0]["name"],
            "molecule_name": molecule_name(typing_component["classification"]["family"]),
            "residue_name": residue_name(typing_component["classification"]["family"]),
            "atom_count": ir["components"][0]["atom_count"],
            "bond_count": len(bonded_component["interactions"]["bond"]),
            "angle_count": len(bonded_component["interactions"]["angle"]),
            "dihedral_count": len(bonded_component["interactions"]["dihedral"]),
            "improper_count": len(bonded_component["interactions"]["improper"]),
            "pair14_count": len(nonbonded_component["pair14"]),
        },
        "outputs": {
            filename: {
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "bytes": len(text.encode("utf-8")),
                "line_count": len(text.splitlines()),
            }
            for filename, text in bundle.items()
        },
    }
    validate_manifest(manifest)
    return manifest


def render_bundle(
    ir: dict,
    *,
    typing_report: dict,
    bonded_report: dict,
    nonbonded_report: dict,
) -> dict[str, str]:
    validate_ir(ir)
    validate_typing_report(typing_report)
    validate_bonded_report(bonded_report)
    validate_nonbonded_report(nonbonded_report)

    typing_component = typing_report["components"][0]
    bonded_component = bonded_report["components"][0]
    nonbonded_component = nonbonded_report["components"][0]
    _validate_gromacs_representability(nonbonded_component)
    component_family = typing_component["classification"]["family"]
    mol_name = molecule_name(component_family)
    res_name = residue_name(component_family)

    forcefield_text = _render_forcefield_itp(ir, typing_report, nonbonded_report)
    molecule_text = _render_molecule_itp(
        ir,
        typing_report=typing_report,
        bonded_report=bonded_report,
        nonbonded_report=nonbonded_report,
        molecule_name_value=mol_name,
        residue_name_value=res_name,
    )
    topol_text = _render_topol_top(
        system_name=ir["components"][0]["name"],
        molecule_name_value=mol_name,
    )

    return {
        "forcefield_pcff.itp": forcefield_text,
        "molecule.itp": molecule_text,
        "topol.top": topol_text,
    }


def dumps_manifest(manifest: dict) -> str:
    validate_manifest(manifest)
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def loads_manifest(text: str) -> dict:
    manifest = json.loads(text)
    validate_manifest(manifest)
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_manifest(manifest), encoding="utf-8")


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_name") != SCHEMA_NAME:
        raise ManifestError("invalid_emitter_manifest", "schema_name must be 'pcff_gromacs_emitter_manifest'")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("invalid_emitter_manifest", "Unsupported schema_version")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ManifestError("invalid_emitter_manifest", "source must be a mapping")
    for key in {
        "typed_ir_sha256",
        "typing_report_sha256",
        "bonded_assignment_sha256",
        "nonbonded_assignment_sha256",
        "source_id",
        "input_format",
    }:
        if key not in source:
            raise ManifestError("invalid_emitter_manifest", f"source.{key} is required")
    emitter = manifest.get("emitter")
    if not isinstance(emitter, dict):
        raise ManifestError("invalid_emitter_manifest", "emitter must be a mapping")
    if emitter.get("kind") != "gromacs":
        raise ManifestError("invalid_emitter_manifest", "emitter.kind must be 'gromacs'")
    if emitter.get("status") not in {"dry_run", "written"}:
        raise ManifestError("invalid_emitter_manifest", "emitter.status must be 'dry_run' or 'written'")
    for key in {"dry_run", "validate_existing"}:
        if not isinstance(emitter.get(key), bool):
            raise ManifestError("invalid_emitter_manifest", f"emitter.{key} must be a boolean")
    if emitter.get("existing_output_matches_rendered") not in {None, True}:
        raise ManifestError(
            "invalid_emitter_manifest",
            "emitter.existing_output_matches_rendered must be null or true",
        )
    component = manifest.get("component")
    if not isinstance(component, dict):
        raise ManifestError("invalid_emitter_manifest", "component must be a mapping")
    for key in {
        "component_id",
        "system_name",
        "molecule_name",
        "residue_name",
        "atom_count",
        "bond_count",
        "angle_count",
        "dihedral_count",
        "improper_count",
        "pair14_count",
    }:
        if key not in component:
            raise ManifestError("invalid_emitter_manifest", f"component.{key} is required")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ManifestError("invalid_emitter_manifest", "outputs must be a mapping")
    for filename in BUNDLE_FILENAMES:
        payload = outputs.get(filename)
        if not isinstance(payload, dict):
            raise ManifestError("invalid_emitter_manifest", f"outputs.{filename} must be a mapping")
        for key in {"sha256", "bytes", "line_count"}:
            if key not in payload:
                raise ManifestError("invalid_emitter_manifest", f"outputs.{filename}.{key} is required")


def validate_bundle(bundle: dict[str, str]) -> None:
    if list(bundle) != list(BUNDLE_FILENAMES):
        raise GromacsEmitterError("invalid_bundle", f"bundle keys must be {BUNDLE_FILENAMES!r}")
    for filename, text in bundle.items():
        if not text.endswith("\n"):
            raise GromacsEmitterError("invalid_bundle", f"{filename} must end with a newline")
        if "\r" in text:
            raise GromacsEmitterError("invalid_bundle", f"{filename} must use LF line endings only")


def _validate_source_chain(
    ir: dict,
    typing_report: dict,
    bonded_report: dict,
    nonbonded_report: dict,
    *,
    ir_sha256: str,
    typing_sha256: str,
    bonded_sha256: str,
) -> None:
    if typing_report["source"]["typed_ir_sha256"] != ir_sha256:
        raise GromacsEmitterError("source_chain_mismatch", "typing report does not match typed IR")
    if bonded_report["source"]["typed_ir_sha256"] != ir_sha256:
        raise GromacsEmitterError("source_chain_mismatch", "bonded report does not match typed IR")
    if nonbonded_report["source"]["typed_ir_sha256"] != ir_sha256:
        raise GromacsEmitterError("source_chain_mismatch", "nonbonded report does not match typed IR")
    if nonbonded_report["source"]["typing_report_sha256"] != typing_sha256:
        raise GromacsEmitterError("source_chain_mismatch", "nonbonded report does not match typing report")
    if nonbonded_report["source"]["bonded_assignment_sha256"] != bonded_sha256:
        raise GromacsEmitterError("source_chain_mismatch", "nonbonded report does not match bonded report")
    if ir["source"]["source_id"] != typing_report["source"]["source_id"]:
        raise GromacsEmitterError("source_chain_mismatch", "source_id mismatch between IR and typing report")


def _render_forcefield_itp(ir: dict, typing_report: dict, nonbonded_report: dict) -> str:
    nonbonded_component = nonbonded_report["components"][0]
    typing_component = typing_report["components"][0]
    atom_records = nonbonded_component["atoms"]
    atomtypes = _unique_atomtype_records(atom_records)
    pair_overrides = [
        record
        for record in nonbonded_component["pair_classes"]
        if record["parameter_source"] == "override"
    ]
    pair14_overrides = _unique_pair14_overrides(nonbonded_component["pair14"])

    lines = [
        "; deterministic PT6 GROMACS emitter output",
        f"; component_family={typing_component['classification']['family']}",
        f"; typed_ir_sha256={nonbonded_report['source']['typed_ir_sha256']}",
        f"; typing_report_sha256={nonbonded_report['source']['typing_report_sha256']}",
        f"; bonded_assignment_sha256={nonbonded_report['source']['bonded_assignment_sha256']}",
        "",
        "[ defaults ]",
        "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ rep-pow",
        "1 4 yes 1.0 1.0 9.0",
        "",
        "[ atomtypes ]",
        "; name mass charge ptype sigma epsilon",
    ]

    for record in atomtypes:
        mass = _mass_for_element(record["element"])
        sigma_nm = angstrom_to_nm(record["self_parameters"]["sigma_angstrom"])
        epsilon_kj = kcal_to_kj(record["self_parameters"]["epsilon_kcal_mol"])
        lines.append(
            f"{record['nonbonded_type']:<22s} {mass:8.3f} 0.0 A {format_float(sigma_nm)} {format_float(epsilon_kj)}"
        )

    if pair_overrides:
        lines.extend(["", "[ nonbond_params ]", "; i j func sigma epsilon"])
        for record in sorted(pair_overrides, key=lambda item: tuple(item["atom_families"])):
            type_i, type_j = _types_for_pair(record["atom_families"], atom_records)
            sigma_nm = angstrom_to_nm(record["parameters"]["sigma_angstrom"])
            epsilon_kj = kcal_to_kj(record["parameters"]["epsilon_kcal_mol"])
            lines.append(
                f"{type_i:<22s} {type_j:<22s} 1 {format_float(sigma_nm)} {format_float(epsilon_kj)}"
            )

    if pair14_overrides:
        lines.extend(["", "[ pairtypes ]", "; i j func sigma epsilon"])
        for record in pair14_overrides:
            type_i, type_j = _types_for_pair(record["atom_families"], atom_records)
            sigma_nm = angstrom_to_nm(record["parameters"]["sigma_angstrom"])
            epsilon_kj = kcal_to_kj(record["parameters"]["epsilon_kcal_mol"])
            lines.append(
                f"{type_i:<22s} {type_j:<22s} 1 {format_float(sigma_nm)} {format_float(epsilon_kj)}"
            )

    return "\n".join(lines) + "\n"


def _render_molecule_itp(
    ir: dict,
    *,
    typing_report: dict,
    bonded_report: dict,
    nonbonded_report: dict,
    molecule_name_value: str,
    residue_name_value: str,
) -> str:
    ir_component = ir["components"][0]
    typing_component = typing_report["components"][0]
    bonded_component = bonded_report["components"][0]
    nonbonded_component = nonbonded_report["components"][0]

    atom_index_map = {record["canonical_index"]: record for record in nonbonded_component["atoms"]}
    atom_name_counts: dict[str, int] = {}
    nrexcl = 3 if len(ir_component["atoms"]) > 1 else 1

    lines = [
        "; deterministic PT6 GROMACS emitter output",
        f"; molecule_name={molecule_name_value}",
        f"; residue_name={residue_name_value}",
        "",
        "[ moleculetype ]",
        "; Name nrexcl",
        f"{molecule_name_value} {nrexcl}",
        "",
        "[ atoms ]",
        "; nr type resnr residue atom cgnr charge mass",
    ]

    for atom in ir_component["atoms"]:
        element = atom["element"]
        atom_name_counts.setdefault(element, 0)
        atom_name_counts[element] += 1
        assigned = atom_index_map[atom["canonical_index"]]
        charge = assigned["charge_assignment"]["value"]
        mass = _mass_for_element(element)
        lines.append(
            f"{atom['canonical_index']:>3d} {assigned['nonbonded_type']:<22s} 1 {residue_name_value:<6s} "
            f"{atom_name(element, atom_name_counts[element]):<5s} {atom['canonical_index']:>3d} "
            f"{charge: .8f} {mass: .6f}"
        )

    if bonded_component["interactions"]["bond"]:
        lines.extend(["", "[ bonds ]", "; ai aj funct c0 c1 c2 c3"])
        for record in bonded_component["interactions"]["bond"]:
            params = _gromacs_bond_params(record["parameters"]["main"])
            lines.append(
                f"{record['atom_indices'][0]:>3d} {record['atom_indices'][1]:>3d} 11 "
                + " ".join(format_float(value) for value in params)
            )

    if nonbonded_component["pair14"]:
        lines.extend(["", "[ pairs ]", "; ai aj funct"])
        for record in nonbonded_component["pair14"]:
            lines.append(f"{record['atom_indices'][0]:>3d} {record['atom_indices'][1]:>3d} 1")

    if bonded_component["interactions"]["angle"]:
        lines.extend(["", "[ angles ]", "; ai aj ak funct c0..c10"])
        for record in bonded_component["interactions"]["angle"]:
            params = _gromacs_angle_params(record["parameters"])
            lines.append(
                f"{record['atom_indices'][0]:>3d} {record['atom_indices'][1]:>3d} {record['atom_indices'][2]:>3d} 11 "
                + " ".join(format_float(value) for value in params)
            )

    dihedral_records = []
    for record in bonded_component["interactions"]["dihedral"]:
        dihedral_records.append(
            {
                "atom_indices": record["atom_indices"],
                "funct": 13,
                "params": _gromacs_dihedral_params(record["parameters"]),
            }
        )
    for record in bonded_component["interactions"]["improper"]:
        dihedral_records.append(
            {
                "atom_indices": record["atom_indices"],
                "funct": 12,
                "params": _gromacs_improper_params(record["parameters"]),
            }
        )
    if dihedral_records:
        lines.extend(["", "[ dihedrals ]", "; ai aj ak al funct c0..c31"])
        for record in dihedral_records:
            atom_fields = " ".join(f"{atom_index:>3d}" for atom_index in record["atom_indices"])
            lines.append(f"{atom_fields} {record['funct']:>2d} " + " ".join(format_float(value) for value in record["params"]))

    return "\n".join(lines) + "\n"


def _render_topol_top(*, system_name: str, molecule_name_value: str) -> str:
    lines = [
        "; deterministic PT6 GROMACS emitter output",
        '#include "forcefield_pcff.itp"',
        '#include "molecule.itp"',
        "",
        "[ system ]",
        system_name,
        "",
        "[ molecules ]",
        "; Name number",
        f"{molecule_name_value} 1",
    ]
    return "\n".join(lines) + "\n"


def _gromacs_bond_params(coeff: dict) -> list[float]:
    return [
        angstrom_to_nm(coeff["r0_angstrom"]),
        bond_k2_to_gromacs(coeff["k2_kcal_mol_per_a2"]),
        bond_k3_to_gromacs(coeff["k3_kcal_mol_per_a3"]),
        bond_k4_to_gromacs(coeff["k4_kcal_mol_per_a4"]),
    ]


def _gromacs_angle_params(coeff: dict) -> list[float]:
    return [
        coeff["main"]["theta0_deg"],
        kcal_to_kj(coeff["main"]["k2_kcal_mol"]),
        kcal_to_kj(coeff["main"]["k3_kcal_mol"]),
        kcal_to_kj(coeff["main"]["k4_kcal_mol"]),
        bond_bond_k_to_gromacs(coeff["bb"]["k_kcal_mol_per_a2"]),
        angstrom_to_nm(coeff["bb"]["r1_angstrom"]),
        angstrom_to_nm(coeff["bb"]["r2_angstrom"]),
        bond_angle_k_to_gromacs(coeff["ba"]["k1_kcal_mol_per_a"]),
        bond_angle_k_to_gromacs(coeff["ba"]["k2_kcal_mol_per_a"]),
        angstrom_to_nm(coeff["ba"]["r1_angstrom"]),
        angstrom_to_nm(coeff["ba"]["r2_angstrom"]),
    ]


def _gromacs_dihedral_params(coeff: dict) -> list[float]:
    return [
        kcal_to_kj(coeff["main"]["k1_kcal_mol"]),
        coeff["main"]["phi1_deg"],
        kcal_to_kj(coeff["main"]["k2_kcal_mol"]),
        coeff["main"]["phi2_deg"],
        kcal_to_kj(coeff["main"]["k3_kcal_mol"]),
        coeff["main"]["phi3_deg"],
        dihedral_bond_torsion_k_to_gromacs(coeff["mbt"]["f1_kcal_mol_per_a"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["mbt"]["f2_kcal_mol_per_a"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["mbt"]["f3_kcal_mol_per_a"]),
        angstrom_to_nm(coeff["mbt"]["r0_angstrom"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["ebt"]["f1_1_kcal_mol_per_a"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["ebt"]["f2_1_kcal_mol_per_a"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["ebt"]["f3_1_kcal_mol_per_a"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["ebt"]["f1_2_kcal_mol_per_a"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["ebt"]["f2_2_kcal_mol_per_a"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["ebt"]["f3_2_kcal_mol_per_a"]),
        angstrom_to_nm(coeff["ebt"]["r0_1_angstrom"]),
        angstrom_to_nm(coeff["ebt"]["r0_2_angstrom"]),
        kcal_to_kj(coeff["at"]["f1_1_kcal_mol"]),
        kcal_to_kj(coeff["at"]["f2_1_kcal_mol"]),
        kcal_to_kj(coeff["at"]["f3_1_kcal_mol"]),
        kcal_to_kj(coeff["at"]["f1_2_kcal_mol"]),
        kcal_to_kj(coeff["at"]["f2_2_kcal_mol"]),
        kcal_to_kj(coeff["at"]["f3_2_kcal_mol"]),
        coeff["at"]["theta0_1_deg"],
        coeff["at"]["theta0_2_deg"],
        kcal_to_kj(coeff["aat"]["k_kcal_mol"]),
        coeff["aat"]["theta0_1_deg"],
        coeff["aat"]["theta0_2_deg"],
        bond_bond_k_to_gromacs(coeff["bb13"]["k_kcal_mol_per_a2"]),
        angstrom_to_nm(coeff["bb13"]["r1_angstrom"]),
        angstrom_to_nm(coeff["bb13"]["r3_angstrom"]),
    ]


def _gromacs_improper_params(coeff: dict) -> list[float]:
    return [
        kcal_to_kj(coeff["main"]["k0_kcal_mol"]),
        coeff["main"]["chi0_deg"],
        kcal_to_kj(coeff["aa"]["k1_kcal_mol"]),
        kcal_to_kj(coeff["aa"]["k2_kcal_mol"]),
        kcal_to_kj(coeff["aa"]["k3_kcal_mol"]),
        coeff["aa"]["theta0_1_deg"],
        coeff["aa"]["theta0_2_deg"],
        coeff["aa"]["theta0_3_deg"],
    ]


def _mass_for_element(element: str) -> float:
    if element not in ELEMENT_MASSES:
        raise GromacsEmitterError("unsupported_element_mass", f"No emitter mass is defined for element {element!r}")
    return ELEMENT_MASSES[element]


def _unique_atomtype_records(atom_records: list[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for record in atom_records:
        nonbonded_type = record["nonbonded_type"]
        if nonbonded_type is None:
            raise GromacsEmitterError("missing_nonbonded_type", "cannot emit an atom without a nonbonded type")
        current = unique.get(nonbonded_type)
        if current is None:
            unique[nonbonded_type] = copy.deepcopy(record)
            continue
        if current["element"] != record["element"] or current["self_parameters"] != record["self_parameters"]:
            raise GromacsEmitterError(
                "conflicting_atomtype_definition",
                f"nonbonded type {nonbonded_type!r} is assigned conflicting element or self parameters",
            )
    return [unique[key] for key in sorted(unique)]


def _types_for_pair(atom_families: list[str], atom_records: list[dict]) -> tuple[str, str]:
    family_to_type = {record["assigned_family"]: record["nonbonded_type"] for record in atom_records}
    return family_to_type[atom_families[0]], family_to_type[atom_families[1]]


def _unique_pair14_overrides(pair14_records: list[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for record in pair14_records:
        if record["parameter_source"] != "override":
            continue
        signature = record["canonical_family_pair"]
        if signature not in unique:
            unique[signature] = {
                "canonical_family_pair": signature,
                "atom_families": copy.deepcopy(record["atom_families"]),
                "parameters": copy.deepcopy(record["parameters"]),
            }
    return [unique[key] for key in sorted(unique)]


def _write_bundle(out_dir: Path, bundle: dict[str, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in bundle.items():
        (out_dir / filename).write_text(text, encoding="utf-8")


def _validate_existing_output(out_dir: Path, bundle: dict[str, str]) -> bool:
    for filename, rendered_text in bundle.items():
        path = out_dir / filename
        if not path.is_file():
            raise GromacsEmitterError("missing_rendered_output", f"Expected existing file {path}")
        actual_text = path.read_text(encoding="utf-8")
        if actual_text != rendered_text:
            raise GromacsEmitterError("rendered_output_mismatch", f"Existing file {path} does not match rendered output")
    return True


def _validate_gromacs_representability(nonbonded_component: dict) -> None:
    for record in nonbonded_component["exclusions"]:
        relation = record["topological_relation"]
        if relation not in {"1-2", "1-3"}:
            raise GromacsEmitterError(
                "unsupported_exclusion_relation",
                f"GROMACS emitter only supports exclusions collapsed into nrexcl=3, got {relation!r}",
            )
        if float(record["lj_scale"]) != 0.0 or float(record["coul_scale"]) != 0.0:
            raise GromacsEmitterError(
                "unsupported_exclusion_scaling",
                "GROMACS emitter only supports zero-scaled 1-2 and 1-3 exclusions",
            )
    for record in nonbonded_component["pair14"]:
        if float(record["lj_scale"]) != 1.0 or float(record["coul_scale"]) != 1.0:
            raise GromacsEmitterError(
                "unsupported_pair14_scaling",
                "GROMACS emitter requires full-strength 1-4 scaling because [defaults] is fixed to 1.0/1.0",
            )
