from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from atom_typing import dumps_typing_report, type_ir, validate_typing_report
from chem_perception import dumps_report as dumps_perception_report
from chem_perception import perceive_ir, validate_report as validate_perception_report
from emitters.gromacs import engine as gmx_engine
from emitters.gromacs.formatting import ELEMENT_MASSES, angstrom_to_nm, kcal_to_kj, format_float
from nonbonded_assignment import dumps_assignment_report as dumps_nonbonded_report
from nonbonded_assignment import assign_ir as assign_nonbonded_ir
from nonbonded_assignment import validate_assignment_report as validate_nonbonded_report
from parameter_assignment import dumps_assignment_report as dumps_bonded_report
from parameter_assignment import assign_ir as assign_bonded_ir
from parameter_assignment import validate_assignment_report as validate_bonded_report
from typing_ir import dumps_ir, parse_file, validate_ir

from .errors import PolymerWorkflowError, PolymerWorkflowReportError, PolymerWorkflowSpecError
from .rules import (
    POLYETHER_BACKBONE_CARBON,
    POLYETHER_COMPONENT_FAMILY,
    build_nonbonded_ruleset,
    build_parameter_ruleset,
    build_typing_ruleset,
)


SPEC_SCHEMA_NAME = "pcff_polymer_workflow_spec"
SPEC_SCHEMA_VERSION = 1
REPORT_SCHEMA_NAME = "pcff_polymer_workflow_report"
REPORT_SCHEMA_VERSION = 1

EXPORTABLE_WORKFLOW_KINDS = {"capped_oligomer", "neutral_additive", "salt_species"}
ROLE_BY_WORKFLOW_KIND = {
    "capped_oligomer": {"polymer_fragment"},
    "neutral_additive": {"neutral_additive"},
    "repeat_unit_template": {"polymer_template"},
    "salt_species": {"salt_cation", "salt_anion"},
}
EXPECTED_ROLE_FAMILY = {
    "neutral_additive": "acyclic_alkane",
    "polymer_fragment": POLYETHER_COMPONENT_FAMILY,
    "salt_cation": "lithium_cation",
    "salt_anion": "tfsi_like_sulfonimide",
}


def load_spec(path: str | Path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_spec(spec)
    return spec


def dumps_report(report: dict) -> str:
    validate_report(report)
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def loads_report(text: str) -> dict:
    report = json.loads(text)
    validate_report(report)
    return report


def write_report(path: str | Path, report: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps_report(report), encoding="utf-8")


def run_file(
    path: str | Path,
    *,
    out_dir: str | Path | None = None,
    dry_run: bool = False,
    validate_existing: bool = False,
) -> dict:
    spec_path = Path(path)
    spec = load_spec(spec_path)
    return run_spec(
        spec,
        spec_path=spec_path,
        out_dir=out_dir,
        dry_run=dry_run,
        validate_existing=validate_existing,
    )


def run_spec(
    spec: dict,
    *,
    spec_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    dry_run: bool = False,
    validate_existing: bool = False,
) -> dict:
    validate_spec(spec)
    spec_file = None if spec_path is None else Path(spec_path)
    base_dir = Path.cwd() if spec_file is None else spec_file.parent
    output_root = None if out_dir is None else Path(out_dir)

    typing_ruleset = build_typing_ruleset()
    parameter_ruleset = build_parameter_ruleset()
    nonbonded_ruleset = build_nonbonded_ruleset()

    processed = []
    for component_spec in spec["components"]:
        processed.append(
            _process_component(
                component_spec,
                base_dir=base_dir,
                typing_ruleset=typing_ruleset,
                parameter_ruleset=parameter_ruleset,
                nonbonded_ruleset=nonbonded_ruleset,
            )
        )

    _validate_assembly_checks(processed)

    bundle = render_gromacs_bundle(spec, processed)
    validate_bundle(bundle)

    existing_matches = None
    if validate_existing:
        if output_root is None:
            raise PolymerWorkflowError("missing_output_dir", "validate_existing requires out_dir")
        existing_matches = _validate_existing_output(output_root, bundle)

    if not dry_run:
        if output_root is None:
            raise PolymerWorkflowError("missing_output_dir", "run_spec requires out_dir unless dry_run=True")
        _write_bundle(output_root, bundle)

    report = {
        "schema_name": REPORT_SCHEMA_NAME,
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": {
            "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest(),
            "spec_path": None if spec_file is None else _display_path(spec_file),
            "system_id": spec["system_id"],
        },
        "workflow": {
            "status": "dry_run" if dry_run else "written",
            "dry_run": dry_run,
            "validate_existing": validate_existing,
            "existing_output_matches_rendered": existing_matches,
            "export_kind": "gromacs",
        },
        "components": [_component_report_payload(item) for item in processed],
        "assembly_checks": _assembly_checks_payload(processed),
        "outputs": {
            filename: {
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "bytes": len(text.encode("utf-8")),
                "line_count": len(text.splitlines()),
            }
            for filename, text in bundle.items()
        },
    }
    validate_report(report)
    if not dry_run and output_root is not None:
        write_report(output_root / "polymer_workflow_report.json", report)
    return report


def render_gromacs_bundle(spec: dict, processed_components: list[dict]) -> dict[str, str]:
    validate_spec(spec)
    exportable = [item for item in processed_components if item["exportable"]]
    if not exportable:
        raise PolymerWorkflowError("no_exportable_components", "workflow spec does not contain any exportable components")

    bundle: dict[str, str] = {}
    bundle["forcefield_pcff.itp"] = _render_shared_forcefield_itp(spec["system_id"], exportable)
    for item in exportable:
        bundle[item["output_filename"]] = gmx_engine._render_molecule_itp(
            item["ir"],
            typing_report=item["typing_report"],
            bonded_report=item["bonded_report"],
            nonbonded_report=item["nonbonded_report"],
            molecule_name_value=item["molecule_name"],
            residue_name_value=item["residue_name"],
        )
    bundle["topol.top"] = _render_topol_top(spec["system_id"], exportable)
    return bundle


def validate_spec(spec: dict) -> None:
    if spec.get("schema_name") != SPEC_SCHEMA_NAME:
        raise PolymerWorkflowSpecError("invalid_polymer_workflow_spec", f"schema_name must be {SPEC_SCHEMA_NAME!r}")
    if spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise PolymerWorkflowSpecError("invalid_polymer_workflow_spec", "Unsupported schema_version")
    for key in {"system_id", "description", "components"}:
        if key not in spec:
            raise PolymerWorkflowSpecError("invalid_polymer_workflow_spec", f"{key} is required")
    components = spec["components"]
    if not isinstance(components, list) or not components:
        raise PolymerWorkflowSpecError("invalid_polymer_workflow_spec", "components must be a non-empty list")
    component_ids = set()
    molecule_names = set()
    for component in components:
        for key in {"component_id", "role", "workflow_kind", "path", "input_format", "source_id", "count"}:
            if key not in component:
                raise PolymerWorkflowSpecError("invalid_polymer_workflow_spec", f"component.{key} is required")
        component_id = component["component_id"]
        if component_id in component_ids:
            raise PolymerWorkflowSpecError("invalid_polymer_workflow_spec", f"Duplicate component_id {component_id!r}")
        component_ids.add(component_id)

        workflow_kind = component["workflow_kind"]
        role = component["role"]
        allowed_roles = ROLE_BY_WORKFLOW_KIND.get(workflow_kind)
        if allowed_roles is None:
            raise PolymerWorkflowSpecError(
                "invalid_polymer_workflow_spec",
                f"Unsupported workflow_kind {workflow_kind!r}",
            )
        if role not in allowed_roles:
            raise PolymerWorkflowSpecError(
                "invalid_polymer_workflow_spec",
                f"workflow_kind {workflow_kind!r} does not allow role {role!r}",
            )
        if component["input_format"] != "mol_v2000":
            raise PolymerWorkflowSpecError(
                "invalid_polymer_workflow_spec",
                "PT7 polymer workflow only supports mol_v2000 inputs",
            )
        if not isinstance(component["count"], int) or component["count"] < 1:
            raise PolymerWorkflowSpecError("invalid_polymer_workflow_spec", "component.count must be a positive integer")

        molecule_name = _component_molecule_name(component)
        if workflow_kind in EXPORTABLE_WORKFLOW_KINDS and molecule_name in molecule_names:
            raise PolymerWorkflowSpecError(
                "invalid_polymer_workflow_spec",
                f"Exportable molecule_name {molecule_name!r} must be unique within one assembly",
            )
        molecule_names.add(molecule_name)


def validate_report(report: dict) -> None:
    if report.get("schema_name") != REPORT_SCHEMA_NAME:
        raise PolymerWorkflowReportError("invalid_polymer_workflow_report", f"schema_name must be {REPORT_SCHEMA_NAME!r}")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise PolymerWorkflowReportError("invalid_polymer_workflow_report", "Unsupported schema_version")
    if not isinstance(report.get("components"), list) or not report["components"]:
        raise PolymerWorkflowReportError("invalid_polymer_workflow_report", "components must be a non-empty list")
    outputs = report.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise PolymerWorkflowReportError("invalid_polymer_workflow_report", "outputs must be a non-empty mapping")


def validate_bundle(bundle: dict[str, str]) -> None:
    if "forcefield_pcff.itp" not in bundle or "topol.top" not in bundle:
        raise PolymerWorkflowError("invalid_bundle", "bundle must include forcefield_pcff.itp and topol.top")
    for filename, text in bundle.items():
        if not text.endswith("\n"):
            raise PolymerWorkflowError("invalid_bundle", f"{filename} must end with LF newline")
        if "\r" in text:
            raise PolymerWorkflowError("invalid_bundle", f"{filename} must use LF endings only")


def _process_component(
    component_spec: dict,
    *,
    base_dir: Path,
    typing_ruleset: dict,
    parameter_ruleset: dict,
    nonbonded_ruleset: dict,
) -> dict:
    path = (base_dir / component_spec["path"]).resolve()
    if not path.is_file():
        raise PolymerWorkflowError("missing_component_input", f"Component input {path} does not exist")

    if component_spec["workflow_kind"] == "repeat_unit_template":
        ir = parse_file(path, input_format=component_spec["input_format"], source_id=component_spec["source_id"])
        perception = perceive_ir(ir)
        validate_ir(ir)
        validate_perception_report(perception)
        template_metadata = _analyze_repeat_unit_template(ir, perception)
        return {
            "component_spec": copy.deepcopy(component_spec),
            "path": path,
            "exportable": False,
            "ir": ir,
            "perception": perception,
            "template_metadata": template_metadata,
        }

    ir = parse_file(path, input_format=component_spec["input_format"], source_id=component_spec["source_id"])
    perception = perceive_ir(ir)
    typing_report = type_ir(ir, perception=perception, ruleset=typing_ruleset)
    bonded_report = assign_bonded_ir(ir, typing_report=typing_report, perception=perception, ruleset=parameter_ruleset)
    nonbonded_report = assign_nonbonded_ir(
        ir,
        typing_report=typing_report,
        bonded_report=bonded_report,
        ruleset=nonbonded_ruleset,
    )

    validate_ir(ir)
    validate_perception_report(perception)
    validate_typing_report(typing_report)
    validate_bonded_report(bonded_report)
    validate_nonbonded_report(nonbonded_report)
    gmx_engine._validate_gromacs_representability(nonbonded_report["components"][0])

    family = typing_report["components"][0]["classification"]["family"]
    expected_family = EXPECTED_ROLE_FAMILY[component_spec["role"]]
    if family != expected_family:
        raise PolymerWorkflowError(
            "unexpected_component_family",
            f"Component {component_spec['component_id']!r} role {component_spec['role']!r} requires family "
            f"{expected_family!r}, got {family!r}",
        )

    component_payload = {
        "component_spec": copy.deepcopy(component_spec),
        "path": path,
        "exportable": True,
        "ir": ir,
        "perception": perception,
        "typing_report": typing_report,
        "bonded_report": bonded_report,
        "nonbonded_report": nonbonded_report,
        "molecule_name": _component_molecule_name(component_spec),
        "residue_name": _component_residue_name(component_spec),
        "output_filename": _component_output_filename(component_spec),
        "net_charge_per_molecule": _component_charge(nonbonded_report),
    }
    if component_spec["role"] == "polymer_fragment":
        component_payload["polymer_fragment_metadata"] = _analyze_polyether_fragment(ir, typing_report)
    return component_payload


def _analyze_repeat_unit_template(ir: dict, perception: dict) -> dict:
    component = perception["components"][0]
    placeholders = [
        atom["canonical_index"]
        for atom in component["atoms"]
        if atom["polymer_connection"]["is_placeholder"]
    ]
    points = component["polymer_connection_points"]
    if len(placeholders) != 2 or len(points) != 2:
        raise PolymerWorkflowError(
            "invalid_repeat_unit_template",
            "repeat_unit_template requires exactly two placeholders and two propagated connection points",
        )
    return {
        "status": "valid",
        "placeholder_count": len(placeholders),
        "placeholder_indices": placeholders,
        "connection_point_count": len(points),
        "connection_points": copy.deepcopy(points),
        "typed_ir_sha256": hashlib.sha256(dumps_ir(ir).encode("utf-8")).hexdigest(),
        "chem_perception_sha256": hashlib.sha256(dumps_perception_report(perception).encode("utf-8")).hexdigest(),
    }


def _analyze_polyether_fragment(ir: dict, typing_report: dict) -> dict:
    ir_component = ir["components"][0]
    typing_component = typing_report["components"][0]
    typed_atoms = typing_component["atoms"]
    by_index = {atom["canonical_index"]: atom for atom in typed_atoms}
    ir_atoms = {atom["canonical_index"]: atom for atom in ir_component["atoms"]}
    adjacency = {atom["canonical_index"]: [] for atom in ir_component["atoms"]}
    for bond in ir_component["bonds"]:
        left, right = bond["atom_indices"]
        adjacency[left].append(right)
        adjacency[right].append(left)

    end_caps = sorted(
        atom["canonical_index"]
        for atom in typed_atoms
        if atom["assigned_family"] == "ether_alpha_carbon_sp3"
    )
    backbone = sorted(
        atom["canonical_index"]
        for atom in typed_atoms
        if atom["assigned_family"] == POLYETHER_BACKBONE_CARBON
    )
    oxygens = sorted(
        atom["canonical_index"]
        for atom in typed_atoms
        if atom["assigned_family"] == "ether_oxygen_sp3"
    )
    if len(end_caps) != 2:
        raise PolymerWorkflowError("invalid_polyether_fragment", "polymer fragment must contain exactly two methyl ether caps")
    if not backbone or len(backbone) % 2 != 0:
        raise PolymerWorkflowError(
            "invalid_polyether_fragment",
            "polymer fragment backbone methylene count must be positive and even",
        )
    if len(oxygens) != (len(backbone) // 2) + 1:
        raise PolymerWorkflowError(
            "invalid_polyether_fragment",
            "polymer fragment oxygen count is inconsistent with the repeat-unit backbone count",
        )
    for atom_index in end_caps:
        heavy_neighbors = [index for index in adjacency[atom_index] if ir_atoms[index]["element"] != "H"]
        if len(heavy_neighbors) != 1 or by_index[heavy_neighbors[0]]["assigned_family"] != "ether_oxygen_sp3":
            raise PolymerWorkflowError(
                "invalid_polyether_fragment",
                f"end cap atom {atom_index} does not terminate in a single ether oxygen neighbor",
            )
    for atom_index in backbone:
        heavy_neighbors = sorted(index for index in adjacency[atom_index] if ir_atoms[index]["element"] != "H")
        heavy_families = sorted(by_index[index]["assigned_family"] for index in heavy_neighbors)
        if heavy_families != ["ether_oxygen_sp3", POLYETHER_BACKBONE_CARBON]:
            raise PolymerWorkflowError(
                "invalid_polyether_fragment",
                f"backbone atom {atom_index} has unsupported heavy-neighbor families {heavy_families!r}",
            )

    return {
        "status": "valid",
        "fragment_model": "linear_methoxy_capped_polyether",
        "repeat_unit_count": len(backbone) // 2,
        "backbone_methylene_count": len(backbone),
        "oxygen_count": len(oxygens),
        "end_group_model": "methyl_ether_caps",
        "terminal_cap_atom_indices": end_caps,
        "terminal_cap_source_indices": [ir_atoms[index]["source_index"] for index in end_caps],
        "typed_ir_sha256": hashlib.sha256(dumps_ir(ir).encode("utf-8")).hexdigest(),
        "typing_report_sha256": hashlib.sha256(dumps_typing_report(typing_report).encode("utf-8")).hexdigest(),
    }


def _component_report_payload(item: dict) -> dict:
    component_spec = item["component_spec"]
    payload = {
        "component_id": component_spec["component_id"],
        "role": component_spec["role"],
        "workflow_kind": component_spec["workflow_kind"],
        "count": component_spec["count"],
        "source_id": component_spec["source_id"],
        "path": _display_path(item["path"]),
        "exportable": item["exportable"],
    }
    if not item["exportable"]:
        payload["template_metadata"] = copy.deepcopy(item["template_metadata"])
        return payload

    payload.update(
        {
            "molecule_name": item["molecule_name"],
            "residue_name": item["residue_name"],
            "output_filename": item["output_filename"],
            "classification_family": item["typing_report"]["components"][0]["classification"]["family"],
            "net_charge_per_molecule": item["net_charge_per_molecule"],
            "total_charge": item["net_charge_per_molecule"] * component_spec["count"],
            "source_chain": {
                "typed_ir_sha256": hashlib.sha256(dumps_ir(item["ir"]).encode("utf-8")).hexdigest(),
                "chem_perception_sha256": hashlib.sha256(dumps_perception_report(item["perception"]).encode("utf-8")).hexdigest(),
                "typing_report_sha256": hashlib.sha256(dumps_typing_report(item["typing_report"]).encode("utf-8")).hexdigest(),
                "bonded_assignment_sha256": hashlib.sha256(
                    dumps_bonded_report(item["bonded_report"]).encode("utf-8")
                ).hexdigest(),
                "nonbonded_assignment_sha256": hashlib.sha256(
                    dumps_nonbonded_report(item["nonbonded_report"]).encode("utf-8")
                ).hexdigest(),
            },
        }
    )
    if "polymer_fragment_metadata" in item:
        payload["polymer_fragment_metadata"] = copy.deepcopy(item["polymer_fragment_metadata"])
    return payload


def _assembly_checks_payload(processed_components: list[dict]) -> dict:
    exportable = [item for item in processed_components if item["exportable"]]
    total_charge = sum(item["net_charge_per_molecule"] * item["component_spec"]["count"] for item in exportable)
    cation_count = sum(
        item["component_spec"]["count"]
        for item in exportable
        if item["component_spec"]["role"] == "salt_cation"
    )
    anion_count = sum(
        item["component_spec"]["count"]
        for item in exportable
        if item["component_spec"]["role"] == "salt_anion"
    )
    polymer_components = [
        item["component_spec"]["component_id"]
        for item in exportable
        if item["component_spec"]["role"] == "polymer_fragment"
    ]
    fragment_consistency = {
        "status": "pass",
        "polymer_component_ids": polymer_components,
    }
    neutral_additive_components = [
        item["component_spec"]["component_id"]
        for item in exportable
        if item["component_spec"]["role"] == "neutral_additive"
    ]
    if neutral_additive_components:
        fragment_consistency["neutral_additive_component_ids"] = neutral_additive_components
    return {
        "charge_neutrality": {
            "status": "pass" if abs(total_charge) < 1.0e-8 else "fail",
            "total_charge": total_charge,
        },
        "salt_balance": {
            "status": "pass" if cation_count == anion_count else "fail",
            "cation_count": cation_count,
            "anion_count": anion_count,
        },
        "fragment_consistency": fragment_consistency,
    }


def _validate_assembly_checks(processed_components: list[dict]) -> None:
    exportable = [item for item in processed_components if item["exportable"]]
    total_charge = sum(item["net_charge_per_molecule"] * item["component_spec"]["count"] for item in exportable)
    if abs(total_charge) >= 1.0e-8:
        raise PolymerWorkflowError("charge_imbalance", f"assembly total charge must be zero, got {total_charge:.8f}")

    cation_count = sum(
        item["component_spec"]["count"]
        for item in exportable
        if item["component_spec"]["role"] == "salt_cation"
    )
    anion_count = sum(
        item["component_spec"]["count"]
        for item in exportable
        if item["component_spec"]["role"] == "salt_anion"
    )
    if cation_count != anion_count:
        raise PolymerWorkflowError(
            "salt_stoichiometry_mismatch",
            f"current PT7 workflow requires equal Li/TFSI counts, got {cation_count} cations and {anion_count} anions",
        )


def _component_charge(nonbonded_report: dict) -> float:
    component = nonbonded_report["components"][0]
    return sum(atom["charge_assignment"]["value"] for atom in component["atoms"])


def _render_shared_forcefield_itp(system_id: str, exportable_components: list[dict]) -> str:
    atomtype_records: dict[str, dict] = {}
    family_to_type: dict[str, str] = {}
    pair_overrides: dict[str, dict] = {}
    pair14_overrides: dict[str, dict] = {}
    source_hashes = []

    for item in exportable_components:
        nonbonded_component = item["nonbonded_report"]["components"][0]
        source_hashes.append(item["nonbonded_report"]["source"]["typed_ir_sha256"])
        for atom in nonbonded_component["atoms"]:
            nonbonded_type = atom["nonbonded_type"]
            current = atomtype_records.get(nonbonded_type)
            if current is None:
                atomtype_records[nonbonded_type] = copy.deepcopy(atom)
            elif current["element"] != atom["element"] or current["self_parameters"] != atom["self_parameters"]:
                raise PolymerWorkflowError(
                    "conflicting_atomtype_definition",
                    f"nonbonded type {nonbonded_type!r} has conflicting definitions across assembly components",
                )
            family_to_type[atom["assigned_family"]] = nonbonded_type
        for record in nonbonded_component["pair_classes"]:
            if record["parameter_source"] != "override":
                continue
            signature = record["canonical_family_pair"]
            pair_overrides.setdefault(signature, copy.deepcopy(record))
        for record in nonbonded_component["pair14"]:
            if record["parameter_source"] != "override":
                continue
            signature = record["canonical_family_pair"]
            pair14_overrides.setdefault(signature, copy.deepcopy(record))

    atomtypes = [atomtype_records[key] for key in sorted(atomtype_records)]
    lines = [
        "; deterministic PT7 polymer workflow shared GROMACS forcefield",
        f"; system_id={system_id}",
        f"; component_count={len(exportable_components)}",
        f"; typed_ir_sha256_chain={','.join(sorted(source_hashes))}",
        "",
        "[ defaults ]",
        "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ rep-pow",
        "1 4 yes 1.0 1.0 9.0",
        "",
        "[ atomtypes ]",
        "; name mass charge ptype sigma epsilon",
    ]
    for atom in atomtypes:
        mass = _mass_for_element(atom["element"])
        sigma_nm = angstrom_to_nm(atom["self_parameters"]["sigma_angstrom"])
        epsilon_kj = kcal_to_kj(atom["self_parameters"]["epsilon_kcal_mol"])
        lines.append(
            f"{atom['nonbonded_type']:<22s} {mass:8.3f} 0.0 A {format_float(sigma_nm)} {format_float(epsilon_kj)}"
        )

    if pair_overrides:
        lines.extend(["", "[ nonbond_params ]", "; i j func sigma epsilon"])
        for signature in sorted(pair_overrides):
            record = pair_overrides[signature]
            type_i = family_to_type[record["atom_families"][0]]
            type_j = family_to_type[record["atom_families"][1]]
            sigma_nm = angstrom_to_nm(record["parameters"]["sigma_angstrom"])
            epsilon_kj = kcal_to_kj(record["parameters"]["epsilon_kcal_mol"])
            lines.append(f"{type_i:<22s} {type_j:<22s} 1 {format_float(sigma_nm)} {format_float(epsilon_kj)}")

    if pair14_overrides:
        lines.extend(["", "[ pairtypes ]", "; i j func sigma epsilon"])
        for signature in sorted(pair14_overrides):
            record = pair14_overrides[signature]
            type_i = family_to_type[record["atom_families"][0]]
            type_j = family_to_type[record["atom_families"][1]]
            sigma_nm = angstrom_to_nm(record["parameters"]["sigma_angstrom"])
            epsilon_kj = kcal_to_kj(record["parameters"]["epsilon_kcal_mol"])
            lines.append(f"{type_i:<22s} {type_j:<22s} 1 {format_float(sigma_nm)} {format_float(epsilon_kj)}")
    return "\n".join(lines) + "\n"


def _render_topol_top(system_id: str, exportable_components: list[dict]) -> str:
    lines = [
        "; deterministic PT7 polymer workflow topol.top",
        '#include "forcefield_pcff.itp"',
    ]
    for item in exportable_components:
        lines.append(f'#include "{item["output_filename"]}"')
    lines.extend(
        [
            "",
            "[ system ]",
            system_id,
            "",
            "[ molecules ]",
            "; Name number",
        ]
    )
    for item in exportable_components:
        lines.append(f"{item['molecule_name']} {item['component_spec']['count']}")
    return "\n".join(lines) + "\n"


def _mass_for_element(element: str) -> float:
    if element not in ELEMENT_MASSES:
        raise PolymerWorkflowError("unsupported_element_mass", f"No mass is defined for element {element!r}")
    return ELEMENT_MASSES[element]


def _component_molecule_name(component_spec: dict) -> str:
    raw = component_spec.get("molecule_name", component_spec["component_id"])
    sanitized = "".join(ch for ch in raw.upper() if ch.isalnum())
    if not sanitized:
        raise PolymerWorkflowSpecError("invalid_polymer_workflow_spec", "component molecule_name must contain alphanumeric characters")
    return sanitized[:12]


def _component_residue_name(component_spec: dict) -> str:
    raw = component_spec.get("residue_name", _component_molecule_name(component_spec))
    sanitized = "".join(ch for ch in raw.upper() if ch.isalnum())
    if not sanitized:
        raise PolymerWorkflowSpecError("invalid_polymer_workflow_spec", "component residue_name must contain alphanumeric characters")
    return sanitized[:8]


def _component_output_filename(component_spec: dict) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in component_spec["component_id"].lower())
    return f"molecule_{sanitized}.itp"


def _write_bundle(out_dir: Path, bundle: dict[str, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in bundle.items():
        (out_dir / filename).write_text(text, encoding="utf-8")


def _validate_existing_output(out_dir: Path, bundle: dict[str, str]) -> bool:
    for filename, rendered_text in bundle.items():
        path = out_dir / filename
        if not path.is_file():
            raise PolymerWorkflowError("missing_rendered_output", f"Expected existing file {path}")
        if path.read_text(encoding="utf-8") != rendered_text:
            raise PolymerWorkflowError("rendered_output_mismatch", f"Existing file {path} does not match rendered output")
    return True


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
