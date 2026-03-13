from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
M10_ROOT = REPO_ROOT / "tests" / "reference_results" / "m10"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_m10_outputs_exist() -> None:
    expected = {
        "chain_size_artifact_status.csv",
        "comparison_summary.json",
        "pcff_paired_provenance_gate.csv",
        "strict_parity_summary.json",
        "screening_usefulness_summary.json",
        "method_readiness_summary.json",
        "paired_artifact_registry_audit.csv",
        "paired_density_provenance.csv",
        "paired_topology_recovery.csv",
        "strict_parity_metrics.csv",
        "screening_metric_rows.csv",
        "density_local_subset.csv",
        "transport_decomposition.csv",
    }
    found = {path.name for path in M10_ROOT.iterdir() if path.is_file()}
    assert expected <= found


def test_m10_summary_has_required_sections() -> None:
    summary = load_json(M10_ROOT / "comparison_summary.json")
    assert summary["milestone"] == "M10"
    assert set(summary) >= {
        "milestone",
        "strict_parity",
        "screening_usefulness",
        "provenance_diagnostics",
        "transport_mismatch_diagnostics",
        "method_readiness",
    }

    strict = summary["strict_parity"]
    assert strict["candidate_paired_system_ids"] == ["14748", "27670"]
    assert strict["paired_system_ids"] == []
    assert strict["n_paired_systems"] == 0
    assert strict["status"] == "blocked_by_pcff_provenance"
    assert strict["rejected_candidate_systems"] == [
        {
            "trajectory_id": "14748",
            "gromacs_preparation": "acpype_gaff2_topology",
            "lammps_reference": "pcff_class2",
            "reason": "GROMACS paired topology was generated with ACPYPE/GAFF2, not PCFF",
        },
        {
            "trajectory_id": "27670",
            "gromacs_preparation": "acpype_gaff2_atomtyping_failed",
            "lammps_reference": "pcff_class2",
            "reason": "GROMACS paired topology is missing and the preserved typing attempt is ACPYPE/GAFF2",
        },
    ]

    screening = summary["screening_usefulness"]
    assert screening["completion"]["n_total"] == 120
    assert screening["completion"]["n_completed"] == 107
    assert screening["completion"]["n_failed"] == 13
    assert screening["status"] == "not_pcff_qualified"
    assert screening["provenance_status"]["pcff_qualified_for_current_m10_claim"] is False

    readiness = summary["method_readiness"]
    assert readiness["overall_status"] == "pcff_provenance_blocked"
    assert "no paired system currently passes the GROMACS PCFF provenance gate" in readiness["blocking_gaps"]
    assert "screening cohort is prepared with ACPYPE/GAFF2 rather than PCFF" in readiness["blocking_gaps"]
    assert "paired density provenance is unresolved" in readiness["blocking_gaps"]
    assert "paired raw production artifacts are missing while run_results.csv still reports completed analysis" in readiness["blocking_gaps"]
    assert readiness["artifact_registry_status_counts"] == {"derived_metrics_without_raw_artifacts": 2}
    assert readiness["pcff_provenance_gate_status_counts"] == {
        "acpype_gaff2_atomtyping_failed": 1,
        "acpype_gaff2_topology": 1,
    }

    provenance = summary["provenance_diagnostics"]
    assert provenance["pcff_paired_provenance_gate"]["status_counts"] == {
        "acpype_gaff2_atomtyping_failed": 1,
        "acpype_gaff2_topology": 1,
    }
    assert provenance["pcff_paired_provenance_gate"]["eligible_for_pcff_strict_parity"] == []
    assert provenance["paired_density_provenance"]["status_counts"] == {"inconsistent": 1, "unavailable": 1}
    assert provenance["paired_topology_recovery"]["recoverable_with_donor_topology"] == ["27670"]
    assert provenance["paired_chain_size_artifacts"]["status_counts"] == {"unavailable": 2}
    assert provenance["paired_artifact_registry_audit"]["status_counts"] == {"derived_metrics_without_raw_artifacts": 2}

    transport = summary["transport_mismatch_diagnostics"]
    assert transport["n_paired_systems"] == 2
    assert transport["conductivity_error_modes"]["sigma_NE_mean_abs_log10_error"] < transport["conductivity_error_modes"]["sigma_cNE_mean_abs_log10_error"]
    assert transport["ion_lj_deltas"]["mean_abs_delta_li_epsilon_kj"] > 1.0
    assert transport["heuristic_driver_order"][0] == "electrostatics_relative_delta_score"


def test_m10_metric_tables_are_nonempty_and_consistent() -> None:
    with (M10_ROOT / "strict_parity_metrics.csv").open(newline="") as handle:
        strict_rows = list(csv.DictReader(handle))
    with (M10_ROOT / "screening_metric_rows.csv").open(newline="") as handle:
        screening_rows = list(csv.DictReader(handle))
    with (M10_ROOT / "density_local_subset.csv").open(newline="") as handle:
        density_rows = list(csv.DictReader(handle))

    assert len(strict_rows) == 0
    assert len(screening_rows) >= 400
    assert len(density_rows) == 8


def test_m10_conductivity_provenance_is_explicit() -> None:
    screening = load_json(M10_ROOT / "screening_usefulness_summary.json")
    conductivity = screening["metrics"]["conductivity"]
    assert conductivity["prediction_source_counts"] == {"NE_fallback": 58, "cNE": 49}
    assert conductivity["n_compared"] == 107
    assert conductivity["top10_overlap"]["overlap_count"] == 0
    assert conductivity["bottom10_overlap"]["overlap_count"] == 2


def test_m10_provenance_and_transport_diagnostics_are_machine_readable() -> None:
    with (M10_ROOT / "pcff_paired_provenance_gate.csv").open(newline="") as handle:
        pcff_gate_rows = list(csv.DictReader(handle))
    with (M10_ROOT / "paired_density_provenance.csv").open(newline="") as handle:
        density_rows = list(csv.DictReader(handle))
    with (M10_ROOT / "paired_topology_recovery.csv").open(newline="") as handle:
        topology_rows = list(csv.DictReader(handle))
    with (M10_ROOT / "paired_artifact_registry_audit.csv").open(newline="") as handle:
        registry_rows = list(csv.DictReader(handle))
    with (M10_ROOT / "chain_size_artifact_status.csv").open(newline="") as handle:
        chain_rows = list(csv.DictReader(handle))
    with (M10_ROOT / "transport_decomposition.csv").open(newline="") as handle:
        transport_rows = list(csv.DictReader(handle))

    assert len(pcff_gate_rows) == 2
    gate_by_id = {row["trajectory_id"]: row for row in pcff_gate_rows}
    assert gate_by_id["14748"]["status"] == "acpype_gaff2_topology"
    assert gate_by_id["14748"]["gromacs_preparation"] == "acpype_gaff2_topology"
    assert gate_by_id["14748"]["lammps_reference"] == "pcff_class2"
    assert gate_by_id["14748"]["keep_for_pcff_strict_parity"] == "False"
    assert gate_by_id["27670"]["status"] == "acpype_gaff2_atomtyping_failed"
    assert gate_by_id["27670"]["gromacs_preparation"] == "acpype_gaff2_atomtyping_failed"
    assert gate_by_id["27670"]["lammps_reference"] == "pcff_class2"
    assert gate_by_id["27670"]["keep_for_pcff_strict_parity"] == "False"
    assert all(row["global_gromacs_pipeline_uses_acpype"] == "True" for row in pcff_gate_rows)
    assert all(row["global_gromacs_pipeline_li_fallback"] == "True" for row in pcff_gate_rows)

    assert len(density_rows) == 2
    assert {row["trajectory_id"] for row in density_rows} == {"14748", "27670"}
    assert {row["status"] for row in density_rows} == {"inconsistent", "unavailable"}
    assert all(row["packmol_counts"] for row in density_rows)

    assert len(topology_rows) == 2
    topology_by_id = {row["trajectory_id"]: row for row in topology_rows}
    assert topology_by_id["14748"]["topol_exists"] == "True"
    assert topology_by_id["27670"]["topol_exists"] == "False"
    assert topology_by_id["27670"]["donor_trajectory_id"] == "14768"
    assert topology_by_id["27670"]["donor_atom_signature_match"] == "True"
    assert topology_by_id["27670"]["donor_topology_dry_run_status"] == "ok"

    assert len(registry_rows) == 2
    registry_by_id = {row["trajectory_id"]: row for row in registry_rows}
    assert all(row["status"] == "derived_metrics_without_raw_artifacts" for row in registry_rows)
    assert registry_by_id["14748"]["run_results_analysis_status"] == "ok"
    assert registry_by_id["27670"]["run_results_analysis_status"] == "ok"
    assert registry_by_id["14748"]["run_results_analysis_csv_exists"] == "False"
    assert registry_by_id["27670"]["run_results_analysis_csv_exists"] == "False"
    assert registry_by_id["14748"]["gromacs_production_stage_trace_present"] == "True"
    assert registry_by_id["27670"]["gromacs_production_stage_trace_present"] == "True"
    assert registry_by_id["14748"]["gromacs_sampled_rdf_present"] == "True"
    assert registry_by_id["27670"]["gromacs_sampled_rdf_present"] == "True"

    assert len(chain_rows) == 2
    assert all(row["status"] == "unavailable" for row in chain_rows)
    assert all(row["lammps_rg_mean_nm"] for row in chain_rows)
    assert all(row["gromacs_rg_generation_ready"] == "False" for row in chain_rows)
    assert all(row["run_results_analysis_csv_exists"] == "False" for row in chain_rows)

    assert len(transport_rows) == 2
    assert {row["trajectory_id"] for row in transport_rows} == {"14748", "27670"}
    for row in transport_rows:
        assert row["sigma_cNE_abs_log10_error"]
        assert row["sigma_NE_abs_log10_error"]
        assert row["li_o_qprod_gromacs"]
        assert row["li_n_qprod_gromacs"]
        assert row["lj_li_epsilon_gromacs_kj"]
        assert row["lj_o_sigma_gromacs_nm"]
