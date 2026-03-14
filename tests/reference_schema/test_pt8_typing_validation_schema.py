from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PT8_ROOT = REPO_ROOT / "tests" / "reference_results" / "pt8_typing_validation"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pt8_validation_outputs_exist() -> None:
    expected = {
        "failure_probe_summary.json",
        "lammps_smoke_parity_summary.json",
        "per_case_results.json",
        "validation_summary.json",
    }
    found = {path.name for path in PT8_ROOT.iterdir() if path.is_file()}
    assert expected <= found


def test_pt8_validation_summary_is_m10_handoff_ready_for_supported_scope() -> None:
    summary = load_json(PT8_ROOT / "validation_summary.json")
    assert summary["milestone"] == "PT8"
    assert summary["overall_status"] == "supported_spe_scope_validated_for_m10_dependency_handoff"
    assert summary["supported_scope"]["case_count"] == 3
    assert summary["supported_scope"]["passed_case_ids"] == [
        "diglyme_litfsi_1to1",
        "monoglyme_litfsi_1to1",
        "triglyme_litfsi_2to2",
    ]
    assert summary["supported_scope"]["failed_case_ids"] == []
    assert summary["failure_classification"]["all_required_classes_observed"] is True
    assert summary["failure_classification"]["observed_counts"] == {
        "emitter_failure": 1,
        "parameter_failure": 2,
        "typing_failure": 1,
    }
    assert summary["lammps_smoke_parity"]["parity_level"] == "contract_only"
    assert summary["lammps_smoke_parity"]["reference_system_id"] == "small_salt_polymer_box"
    assert summary["m10_handoff"]["document_path"] == "docs/m10_handoff_typing.md"


def test_pt8_per_case_results_cover_realistic_spe_examples() -> None:
    per_case = load_json(PT8_ROOT / "per_case_results.json")
    assert per_case["case_count"] == 3
    case_map = {case["case_id"]: case for case in per_case["cases"]}

    monoglyme = case_map["monoglyme_litfsi_1to1"]
    assert monoglyme["workflow_validation"]["written_outputs_match_report_hashes"] is True
    assert monoglyme["workflow_validation"]["existing_output_matches_rendered"] is True
    assert monoglyme["assembly_checks"]["charge_neutrality"]["status"] == "pass"
    assert monoglyme["assembly_checks"]["salt_balance"]["status"] == "pass"
    assert monoglyme["polymer_fragment"]["repeat_unit_count"] == 1
    assert monoglyme["molecule_counts"] == {"LI": 1, "MONOGLY": 1, "TFSI": 1}

    diglyme = case_map["diglyme_litfsi_1to1"]
    assert diglyme["polymer_fragment"]["repeat_unit_count"] == 2
    assert diglyme["outputs"]["molecule_diglyme.itp"]["sha256"] == "ca08d1577f04358403787c9422d7fa1f1ebefcc88ce8c380db875307eff3efe9"

    triglyme = case_map["triglyme_litfsi_2to2"]
    assert triglyme["polymer_fragment"]["repeat_unit_count"] == 3
    assert triglyme["molecule_counts"] == {"LI": 2, "TFSI": 2, "TRIGLYME": 1}


def test_pt8_failure_probes_and_lammps_smoke_parity_are_explicit() -> None:
    failure_probes = load_json(PT8_ROOT / "failure_probe_summary.json")
    assert failure_probes["classes_observed"] == [
        "emitter_failure",
        "parameter_failure",
        "typing_failure",
    ]
    probe_map = {probe["probe_id"]: probe for probe in failure_probes["probes"]}
    assert probe_map["typing_missing_polyether_atom_rules"]["failure_code"] == "unresolved_atom_type"
    assert probe_map["bonded_missing_polyether_backbone_c_o_rule"]["failure_code"] == "missing_parameter"
    assert probe_map["nonbonded_missing_polyether_backbone_rule"]["failure_code"] == "missing_nonbonded_atom_type"
    assert probe_map["emitter_pair14_scaling_rejection"]["failure_code"] == "unsupported_pair14_scaling"

    lammps_smoke = load_json(PT8_ROOT / "lammps_smoke_parity_summary.json")
    assert lammps_smoke["parity_level"] == "contract_only"
    assert lammps_smoke["reference_system"]["system_id"] == "small_salt_polymer_box"
    assert lammps_smoke["case_count"] == 3
    assert all(check["status"] == "pass" for check in lammps_smoke["checks"])
    assert all(check["checks"]["special_bonds_matches_reference"] is True for check in lammps_smoke["checks"])
    assert all(check["checks"]["pair_modify_uses_sixthpower"] is True for check in lammps_smoke["checks"])
