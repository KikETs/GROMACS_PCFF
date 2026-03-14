from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from atom_typing import type_ir
from atom_typing.errors import AtomTypingError
from atom_typing.rules import load_rules as load_base_typing_rules
from chem_perception import perceive_ir, validate_report as validate_perception_report
from chem_perception.errors import ChemPerceptionError
from emitters.gromacs import emit_ir
from emitters.gromacs.errors import GromacsEmitterError
from nonbonded_assignment import assign_ir as assign_nonbonded_ir
from nonbonded_assignment.errors import NonbondedAssignmentError
from parameter_assignment import assign_ir as assign_bonded_ir
from parameter_assignment.errors import ParameterAssignmentError
from polymer_workflow import load_spec, run_file
from polymer_workflow.rules import build_nonbonded_ruleset, build_parameter_ruleset, build_typing_ruleset
from typing_ir import parse_file, validate_ir
from typing_ir.errors import TypingIRError


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = REPO_ROOT / "testdata" / "polymer_workflow_golden" / "cases"
LAMMPS_ROOT = REPO_ROOT / "testdata" / "lammps_golden"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "tests" / "reference_results" / "pt8_typing_validation"

PER_CASE_RESULTS_NAME = "per_case_results.json"
FAILURE_PROBE_SUMMARY_NAME = "failure_probe_summary.json"
LAMMPS_SMOKE_SUMMARY_NAME = "lammps_smoke_parity_summary.json"
VALIDATION_SUMMARY_NAME = "validation_summary.json"


def dump_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_outputs(out_dir: str | Path, outputs: dict[str, dict]) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for filename, payload in outputs.items():
        (root / filename).write_text(dump_json(payload), encoding="utf-8")


def build_outputs() -> dict[str, dict]:
    per_case = _build_per_case_results()
    failure_probes = _build_failure_probe_summary()
    lammps_smoke = _build_lammps_smoke_parity_summary(per_case["cases"])
    validation = _build_validation_summary(
        per_case_results=per_case,
        failure_probes=failure_probes,
        lammps_smoke=lammps_smoke,
    )
    return {
        PER_CASE_RESULTS_NAME: per_case,
        FAILURE_PROBE_SUMMARY_NAME: failure_probes,
        LAMMPS_SMOKE_SUMMARY_NAME: lammps_smoke,
        VALIDATION_SUMMARY_NAME: validation,
    }


def compare_outputs_to_reference(reference_root: str | Path, outputs: dict[str, dict]) -> list[str]:
    root = Path(reference_root)
    mismatches = []
    for filename, payload in outputs.items():
        path = root / filename
        if not path.is_file():
            mismatches.append(f"missing:{filename}")
            continue
        rendered = dump_json(payload)
        existing = path.read_text(encoding="utf-8")
        if existing != rendered:
            mismatches.append(f"mismatch:{filename}")
    for path in root.iterdir():
        if path.is_file() and path.name not in outputs:
            mismatches.append(f"unexpected:{path.name}")
    return mismatches


def _build_per_case_results() -> dict:
    case_results = []
    for case_dir in sorted(CASE_ROOT.iterdir()):
        if not case_dir.is_dir():
            continue
        case_meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        spec_path = case_dir / case_meta["spec"]["path"]
        case_results.append(_run_supported_case(case_meta=case_meta, spec_path=spec_path))
    return {
        "schema_name": "pcff_typing_validation_per_case_results",
        "schema_version": 1,
        "milestone": "PT8",
        "scope": "supported_spe_systems_only",
        "case_count": len(case_results),
        "cases": case_results,
    }


def _run_supported_case(*, case_meta: dict, spec_path: Path) -> dict:
    dry_report = run_file(spec_path, dry_run=True)
    with tempfile.TemporaryDirectory() as tmp_root:
        out_dir = Path(tmp_root) / case_meta["id"]
        written_report = run_file(spec_path, out_dir=out_dir)
        validate_report = run_file(spec_path, out_dir=out_dir, dry_run=True, validate_existing=True)
        written_hashes = _hash_written_bundle(out_dir)

    components = []
    for component in dry_report["components"]:
        payload = {
            "component_id": component["component_id"],
            "role": component["role"],
            "workflow_kind": component["workflow_kind"],
            "exportable": component["exportable"],
            "count": component["count"],
        }
        if component["exportable"]:
            payload["classification_family"] = component["classification_family"]
            payload["net_charge_per_molecule"] = component["net_charge_per_molecule"]
            payload["total_charge"] = component["total_charge"]
            payload["output_filename"] = component["output_filename"]
            payload["source_chain"] = copy.deepcopy(component["source_chain"])
            if component["role"] == "polymer_fragment":
                payload["polymer_fragment_metadata"] = copy.deepcopy(component["polymer_fragment_metadata"])
        else:
            payload["template_metadata"] = copy.deepcopy(component["template_metadata"])
        components.append(payload)

    molecule_counts = {
        component["molecule_name"]: component["count"]
        for component in dry_report["components"]
        if component["exportable"]
    }
    polymer_component = next(component for component in components if component["role"] == "polymer_fragment")
    salt_components = [component for component in components if component["role"] in {"salt_cation", "salt_anion"}]

    return {
        "case_id": case_meta["id"],
        "display_name": case_meta["display_name"],
        "description": case_meta["description"],
        "status": "pass",
        "source": {
            "case_path": _repo_rel(spec_path.parent / "case.json"),
            "spec_path": _repo_rel(spec_path),
            "spec_sha256": dry_report["source"]["spec_sha256"],
            "system_id": dry_report["source"]["system_id"],
        },
        "workflow_validation": {
            "dry_run_status": dry_report["workflow"]["status"],
            "written_status": written_report["workflow"]["status"],
            "validate_existing_status": validate_report["workflow"]["status"],
            "existing_output_matches_rendered": validate_report["workflow"]["existing_output_matches_rendered"],
            "written_outputs_match_report_hashes": written_hashes == {
                filename: payload["sha256"] for filename, payload in written_report["outputs"].items()
            },
        },
        "assembly_checks": copy.deepcopy(dry_report["assembly_checks"]),
        "polymer_fragment": {
            "component_id": polymer_component["component_id"],
            "classification_family": polymer_component["classification_family"],
            "repeat_unit_count": polymer_component["polymer_fragment_metadata"]["repeat_unit_count"],
            "backbone_methylene_count": polymer_component["polymer_fragment_metadata"]["backbone_methylene_count"],
            "oxygen_count": polymer_component["polymer_fragment_metadata"]["oxygen_count"],
            "end_group_model": polymer_component["polymer_fragment_metadata"]["end_group_model"],
            "terminal_cap_atom_indices": polymer_component["polymer_fragment_metadata"]["terminal_cap_atom_indices"],
        },
        "salt_species": [
            {
                "component_id": component["component_id"],
                "classification_family": component["classification_family"],
                "count": component["count"],
                "total_charge": component["total_charge"],
            }
            for component in salt_components
        ],
        "molecule_counts": molecule_counts,
        "outputs": copy.deepcopy(dry_report["outputs"]),
        "components": components,
    }


def _hash_written_bundle(out_dir: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(out_dir.iterdir()):
        if not path.is_file() or path.name == "polymer_workflow_report.json":
            continue
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _build_failure_probe_summary() -> dict:
    probe_functions = [
        _typing_failure_probe,
        _bonded_parameter_failure_probe,
        _nonbonded_parameter_failure_probe,
        _emitter_failure_probe,
    ]
    probes = [probe() for probe in probe_functions]
    return {
        "schema_name": "pcff_typing_validation_failure_probe_summary",
        "schema_version": 1,
        "milestone": "PT8",
        "probe_count": len(probes),
        "classes_observed": sorted({probe["failure_class"] for probe in probes}),
        "probes": probes,
    }


def _build_lammps_smoke_parity_summary(case_results: list[dict]) -> dict:
    reference_path = LAMMPS_ROOT / "systems" / "small_salt_polymer_box" / "system.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    checks = []
    for case in case_results:
        spec_path = REPO_ROOT / case["source"]["spec_path"]
        component_contracts = _collect_exportable_component_contracts(spec_path)
        special_bonds_values = sorted({item["special_bonds"] for item in component_contracts})
        pair_modify_values = sorted({item["pair_modify"] for item in component_contracts})
        pair_coeff_policies = sorted({item["pair_coeff_policy"] for item in component_contracts})
        charged_pair_styles = sorted(
            {
                item["pair_style_kind"]
                for item in component_contracts
                if abs(item["net_charge_per_molecule"]) > 1.0e-8
            }
        )
        requires_kspace = any(item["requires_kspace"] for item in component_contracts)
        check_payload = {
            "case_id": case["case_id"],
            "status": "pass",
            "parity_level": "contract_only",
            "reference_system_id": reference["id"],
            "reference_system_path": _repo_rel(reference_path),
            "checks": {
                "charge_neutrality_passes": case["assembly_checks"]["charge_neutrality"]["status"] == "pass",
                "salt_balance_passes": case["assembly_checks"]["salt_balance"]["status"] == "pass",
                "special_bonds_matches_reference": special_bonds_values == [reference["styles"]["special_bonds"]],
                "pair_modify_uses_sixthpower": pair_modify_values == ["mix sixthpower"],
                "charged_pair_style_matches_reference": charged_pair_styles == [reference["styles"]["pair_style"]],
                "requires_kspace_for_salt_system": requires_kspace,
                "pair_coeff_policy_matches_self_only_contract": pair_coeff_policies
                == ["explicit_self_only_with_optional_cross_overrides"],
            },
            "observed_contract": {
                "special_bonds_values": special_bonds_values,
                "pair_modify_values": pair_modify_values,
                "charged_pair_styles": charged_pair_styles,
                "requires_kspace": requires_kspace,
                "pair_coeff_policies": pair_coeff_policies,
            },
        }
        if not all(check_payload["checks"].values()):
            check_payload["status"] = "fail"
        checks.append(check_payload)

    return {
        "schema_name": "pcff_typing_validation_lammps_smoke_parity_summary",
        "schema_version": 1,
        "milestone": "PT8",
        "parity_level": "contract_only",
        "reference_system": {
            "system_id": reference["id"],
            "path": _repo_rel(reference_path),
            "pair_style": reference["styles"]["pair_style"],
            "special_bonds": reference["styles"]["special_bonds"],
            "pair_coeff_source": reference["styles"]["pair_coeff_source"],
        },
        "case_count": len(checks),
        "checks": checks,
    }


def _build_validation_summary(*, per_case_results: dict, failure_probes: dict, lammps_smoke: dict) -> dict:
    cases = per_case_results["cases"]
    case_count = len(cases)
    passed_case_ids = [case["case_id"] for case in cases if case["status"] == "pass"]
    failure_counts = Counter(probe["failure_class"] for probe in failure_probes["probes"])
    lammps_passed = [check["case_id"] for check in lammps_smoke["checks"] if check["status"] == "pass"]
    return {
        "schema_name": "pcff_typing_validation_summary",
        "schema_version": 1,
        "milestone": "PT8",
        "overall_status": (
            "supported_spe_scope_validated_for_m10_dependency_handoff"
            if len(passed_case_ids) == case_count and len(lammps_passed) == case_count
            else "validation_incomplete"
        ),
        "supported_scope": {
            "case_count": case_count,
            "passed_case_ids": passed_case_ids,
            "failed_case_ids": sorted(set(case["case_id"] for case in cases) - set(passed_case_ids)),
            "chemistry": [
                "linear methoxy-capped polyether oligomer",
                "explicit Li+",
                "explicit TFSI-like sulfonimide",
            ],
        },
        "golden_reference_contract": {
            "reference_root": _repo_rel(DEFAULT_REFERENCE_ROOT),
            "reference_files": [
                PER_CASE_RESULTS_NAME,
                FAILURE_PROBE_SUMMARY_NAME,
                LAMMPS_SMOKE_SUMMARY_NAME,
                VALIDATION_SUMMARY_NAME,
            ],
            "validation_mode": "generate.py validate compares regenerated JSON byte-for-byte against checked-in reference summaries",
        },
        "failure_classification": {
            "required_classes": ["typing_failure", "parameter_failure", "emitter_failure"],
            "observed_counts": dict(sorted(failure_counts.items())),
            "all_required_classes_observed": all(
                failure_counts.get(name, 0) > 0 for name in ("typing_failure", "parameter_failure", "emitter_failure")
            ),
        },
        "lammps_smoke_parity": {
            "parity_level": lammps_smoke["parity_level"],
            "reference_system_id": lammps_smoke["reference_system"]["system_id"],
            "passed_case_ids": lammps_passed,
            "failed_case_ids": sorted(
                set(check["case_id"] for check in lammps_smoke["checks"]) - set(lammps_passed)
            ),
        },
        "m10_handoff": {
            "document_path": "docs/m10_handoff_typing.md",
            "downstream_target": "GROMACS PCFF + r-RESPA M10 workflow",
            "machine_readable_failure_channels": [
                "typing_failure",
                "parameter_failure",
                "emitter_failure",
            ],
        },
        "limits": [
            "Validation scope is limited to the PT7 supported SPE set: monoglyme/diglyme/triglyme with explicit Li/TFSI.",
            "LAMMPS comparison is contract-only because the repository does not contain trajectory-matched LAMMPS references for these PT7 SPE systems.",
        ],
    }


def _collect_exportable_component_contracts(spec_path: Path) -> list[dict]:
    spec = load_spec(spec_path)
    base_dir = spec_path.parent
    typing_ruleset = build_typing_ruleset()
    parameter_ruleset = build_parameter_ruleset()
    nonbonded_ruleset = build_nonbonded_ruleset()

    records = []
    for component in spec["components"]:
        if component["workflow_kind"] == "repeat_unit_template":
            continue
        path = (base_dir / component["path"]).resolve()
        ir = parse_file(path, input_format=component["input_format"], source_id=component["source_id"])
        perception = perceive_ir(ir)
        validate_ir(ir)
        validate_perception_report(perception)
        typing_report = type_ir(ir, perception=perception, ruleset=typing_ruleset)
        bonded_report = assign_bonded_ir(ir, typing_report=typing_report, perception=perception, ruleset=parameter_ruleset)
        nonbonded_report = assign_nonbonded_ir(
            ir,
            typing_report=typing_report,
            bonded_report=bonded_report,
            ruleset=nonbonded_ruleset,
        )
        component_report = nonbonded_report["components"][0]
        records.append(
            {
                "component_id": component["component_id"],
                "pair_style_kind": nonbonded_report["nonbonded_assignment"]["pair_style_kind"],
                "special_bonds": component_report["export_metadata"]["lammps"]["special_bonds"],
                "pair_modify": component_report["export_metadata"]["lammps"]["pair_modify"],
                "pair_coeff_policy": component_report["export_metadata"]["lammps"]["pair_coeff_policy"],
                "requires_kspace": component_report["export_metadata"]["lammps"]["requires_kspace"],
                "net_charge_per_molecule": sum(
                    atom["charge_assignment"]["value"] for atom in component_report["atoms"]
                ),
            }
        )
    return records


def _typing_failure_probe() -> dict:
    component_spec, base_dir = _monoglyme_component_spec()
    path = (base_dir / component_spec["path"]).resolve()
    ir = parse_file(path, input_format=component_spec["input_format"], source_id=component_spec["source_id"])
    perception = perceive_ir(ir)
    report = type_ir(ir, perception=perception, ruleset=load_base_typing_rules())
    component = report["components"][0]
    diagnostic = component["diagnostics"][0]
    return {
        "probe_id": "typing_missing_polyether_atom_rules",
        "status": "observed_expected_failure",
        "failure_class": "typing_failure",
        "stage": "typing",
        "failure_code": diagnostic["code"],
        "report_status": report["typing"]["status"],
        "component_id": component_spec["component_id"],
        "message": diagnostic["message"],
    }


def _bonded_parameter_failure_probe() -> dict:
    component_spec, base_dir = _monoglyme_component_spec()
    ir, perception, typing_report = _prepare_monoglyme_typed_component(component_spec, base_dir)
    ruleset = build_parameter_ruleset()
    ruleset["interaction_rules"]["bond"] = [
        rule for rule in ruleset["interaction_rules"]["bond"] if rule["rule_id"] != "param_polyether_backbone_c_o"
    ]
    report = assign_bonded_ir(ir, typing_report=typing_report, perception=perception, ruleset=ruleset)
    component = report["components"][0]
    diagnostic = component["diagnostics"][0]
    return {
        "probe_id": "bonded_missing_polyether_backbone_c_o_rule",
        "status": "observed_expected_failure",
        "failure_class": "parameter_failure",
        "stage": "bonded_parameter_assignment",
        "failure_code": diagnostic["code"],
        "report_status": report["parameter_assignment"]["status"],
        "component_id": component_spec["component_id"],
        "message": diagnostic["message"],
    }


def _nonbonded_parameter_failure_probe() -> dict:
    component_spec, base_dir = _monoglyme_component_spec()
    ir, perception, typing_report = _prepare_monoglyme_typed_component(component_spec, base_dir)
    bonded_report = assign_bonded_ir(
        ir,
        typing_report=typing_report,
        perception=perception,
        ruleset=build_parameter_ruleset(),
    )
    ruleset = build_nonbonded_ruleset()
    ruleset["atom_type_rules"] = [
        rule for rule in ruleset["atom_type_rules"] if rule["rule_id"] != "nb_polyether_backbone_methylene_sp3"
    ]
    report = assign_nonbonded_ir(ir, typing_report=typing_report, bonded_report=bonded_report, ruleset=ruleset)
    component = report["components"][0]
    diagnostic = component["diagnostics"][0]
    return {
        "probe_id": "nonbonded_missing_polyether_backbone_rule",
        "status": "observed_expected_failure",
        "failure_class": "parameter_failure",
        "stage": "nonbonded_parameter_assignment",
        "failure_code": diagnostic["code"],
        "report_status": report["nonbonded_assignment"]["status"],
        "component_id": component_spec["component_id"],
        "message": diagnostic["message"],
    }


def _emitter_failure_probe() -> dict:
    component_spec, base_dir = _monoglyme_component_spec()
    ir, perception, typing_report = _prepare_monoglyme_typed_component(component_spec, base_dir)
    bonded_report = assign_bonded_ir(
        ir,
        typing_report=typing_report,
        perception=perception,
        ruleset=build_parameter_ruleset(),
    )
    nonbonded_report = assign_nonbonded_ir(
        ir,
        typing_report=typing_report,
        bonded_report=bonded_report,
        ruleset=build_nonbonded_ruleset(),
    )
    broken = copy.deepcopy(nonbonded_report)
    broken["components"][0]["pair14"][0]["lj_scale"] = 0.5
    try:
        emit_ir(
            ir,
            typing_report=typing_report,
            bonded_report=bonded_report,
            nonbonded_report=broken,
            dry_run=True,
        )
    except GromacsEmitterError as exc:
        return {
            "probe_id": "emitter_pair14_scaling_rejection",
            "status": "observed_expected_failure",
            "failure_class": "emitter_failure",
            "stage": "gromacs_emitter",
            "failure_code": exc.code,
            "report_status": "exception",
            "component_id": component_spec["component_id"],
            "message": exc.message,
        }
    raise RuntimeError("emitter failure probe did not fail as expected")


def _prepare_monoglyme_typed_component(component_spec: dict, base_dir: Path) -> tuple[dict, dict, dict]:
    path = (base_dir / component_spec["path"]).resolve()
    ir = parse_file(path, input_format=component_spec["input_format"], source_id=component_spec["source_id"])
    perception = perceive_ir(ir)
    validate_ir(ir)
    validate_perception_report(perception)
    typing_report = type_ir(ir, perception=perception, ruleset=build_typing_ruleset())
    return ir, perception, typing_report


def _monoglyme_component_spec() -> tuple[dict, Path]:
    spec_path = CASE_ROOT / "monoglyme_litfsi_1to1" / "spec.json"
    spec = load_spec(spec_path)
    component = next(item for item in spec["components"] if item["component_id"] == "MONOGLY")
    return component, spec_path.parent


def _repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def classify_failure(exc: Exception) -> str:
    if isinstance(exc, (TypingIRError, ChemPerceptionError, AtomTypingError)):
        return "typing_failure"
    if isinstance(exc, (ParameterAssignmentError, NonbondedAssignmentError)):
        return "parameter_failure"
    if isinstance(exc, GromacsEmitterError):
        return "emitter_failure"
    return "unexpected_failure"
