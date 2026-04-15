#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "tests" / "reference_results" / "cpu_exact_respa_claim"

sys.path.append(str(REPO_ROOT / "tools" / "exact_respa_openmp_validation"))
from aggregate_reports import summarize_reports_from_paths  # noqa: E402


def load_json(relative_path: str):
    path = REPO_ROOT / relative_path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def existing_claim_date() -> str | None:
    path = OUT_ROOT / "cpu_exact_claim_summary.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = payload.get("claim_status_as_of")
    return value if isinstance(value, str) and value else None


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def find_gate_a_system(oracle_manifest: dict, system_id: str) -> dict:
    for system in oracle_manifest["systems"]:
        if system["system_id"] == system_id:
            return system
    raise KeyError(f"missing Gate A system: {system_id}")


def find_manifest_system(manifest: dict, system_id: str) -> dict:
    for system in manifest["systems"]:
        if system["system_id"] == system_id:
            return system
    raise KeyError(f"missing manifest system: {system_id}")


def main() -> None:
    claim_date = existing_claim_date() or date.today().isoformat()

    gate_a_path = "tests/reference_results/gate_a_cpu_oracle/oracle_manifest.json"
    gate_g_path = "tests/reference_results/gate_g_long_ensemble_validation/gate_g_manifest.json"
    gate_h_small_path = "tests/reference_results/gate_h_transport_validation/gate_h_manifest.json"
    gate_h_large_path = "tests/reference_results/gate_h_transport_validation_large_medium/gate_h_manifest.json"
    gate_i_contract_path = "tests/reference_results/gate_i_charged_long_npt_conditioning/gate_i_contract.json"
    gate_i_manifest_path = "tests/reference_results/gate_i_charged_long_npt_conditioning/gate_i_manifest.json"
    m10_1_gate_path = "tests/reference_results/m10_1_trajectory_gate/m10_1_gate_decision.json"
    m10_2_gate_path = "tests/reference_results/m10_2_ensemble_gate/m10_2_gate_decision.json"
    m10_2_1_gate_path = "tests/reference_results/m10_2_1_convergence_gate/m10_2_1_gate_decision.json"
    m10_2_1_summary_path = "tests/reference_results/m10_2_1_convergence_gate/m10_2_1_summary.json"

    gate_a = load_json(gate_a_path)
    gate_g = load_json(gate_g_path)
    gate_h_small = load_json(gate_h_small_path)
    gate_h_large = load_json(gate_h_large_path)
    gate_i_contract = load_json(gate_i_contract_path)
    gate_i_manifest = load_json(gate_i_manifest_path)
    m10_1_gate = load_json(m10_1_gate_path)
    m10_2_gate = load_json(m10_2_gate_path)
    m10_2_1_gate = load_json(m10_2_1_gate_path)
    m10_2_1_summary = load_json(m10_2_1_summary_path)[0]

    gate_a_small_oligomer = find_gate_a_system(gate_a, "small_oligomer")
    gate_a_small_salt = find_gate_a_system(gate_a, "small_salt_polymer_box")
    gate_g_small_oligomer = find_manifest_system(gate_g, "small_oligomer")
    gate_g_small_salt = find_manifest_system(gate_g, "small_salt_polymer_box")
    gate_h_small_salt = find_manifest_system(gate_h_small, "small_salt_polymer_box")
    gate_h_large_salt = find_manifest_system(gate_h_large, "gate_h_dense_salt_polymer_2x2x2")
    gate_h_large_oligomer = find_manifest_system(gate_h_large, "gate_h_dense_oligomer_2x2x2")

    host_report_dir = REPO_ROOT / "tests" / "reference_results" / "exact_respa_openmp_validation" / "host_reports"
    host_reports = sorted(host_report_dir.glob("*.json"))
    openmp_summary = summarize_reports_from_paths(host_reports, allow_missing_tsan=False)

    openmp_summary_path = OUT_ROOT / "openmp_validation_summary.json"
    write_json(openmp_summary_path, openmp_summary)

    public_claim = (
        "Current evidence supports a narrow CPU exact-r-RESPA claim: for single-rank, CPU-only, standalone exact "
        "r-RESPA, exact event order, restart continuity, and small-fixture mechanical behavior are frozen on the "
        "Gate A oracle, and a bounded exact CPU OpenMP mechanics claim is allowed across the tested low-core, "
        "hybrid-desktop, and chiplet workstation classes only for the audited ntomp>1 buckets `ntompSmall` and "
        "`ntompCeiling` under `-pin auto`, `-pin on`, and `-pin inherit`. That OpenMP claim is discrete, not a "
        "continuous ntomp envelope: ntomp=1 remains the oracle baseline, host-local throughput benchmarks do not "
        "broaden support, and intermediate or larger ntomp counts remain unsupported. Exact long-run ensemble "
        "evidence is still narrow: Gate G passes a 40 ps NVT `small_oligomer` check and a 40 ps NPT "
        "`small_salt_polymer_box` check, but those are small-fixture gates only. Gate H reuses the exact NVT path on "
        "larger transport scaffolds, yet transport production remains NO-GO, and charged medium-scale long-NPT "
        "density conditioning is still missing. This claim does not imply conductivity-production readiness, "
        "LAMMPS-vs-GROMACS transport parity, generic medium-scale NPT convergence, server-CPU coverage, MPI support, "
        "or GPU coexistence support."
    )

    mechanical_statement = (
        "Exact CPU mechanics are frozen by Gate A on `small_oligomer` and `small_salt_polymer_box`, then extended "
        "to a bounded OpenMP claim on the tested desktop/workstation topology classes by the checked-in exact OpenMP "
        "host-report inventory. That OpenMP claim is limited to the discrete audited ntomp>1 buckets "
        "`ntompSmall` and `ntompCeiling` under `-pin auto`, `-pin on`, and `-pin inherit`; there is no "
        "correctness-only envelope beyond those buckets. This is a mechanical correctness claim, not a transport or "
        "generic ensemble claim."
    )

    openmp_scope_statement = openmp_summary["supported_envelope"]["statement"]
    openmp_weak_shapes_statement = " ".join(openmp_summary["unsupported_or_weak_shapes"])

    ensemble_statement = (
        "Exact ensemble evidence is split deliberately. Gate G gives small-fixture exact-r-RESPA NVT/NPT evidence "
        "(`small_oligomer` NVT PASS, `small_salt_polymer_box` NPT PASS), while Gate H only reuses the exact NVT path "
        "for transport scaffolds and remains a production NO-GO. The legacy M10 medium-scale diagnostics still support "
        "the blocker narrative, but they are not exact-r-RESPA evidence because the M10 runners generate plain "
        "`integrator = md` inputs."
    )

    transport_statement = (
        "Mechanical parity and short exact-NVT scaffold reuse do not establish transport readiness. Current public "
        "evidence still blocks conductivity-production language until charged medium-scale exact-r-RESPA long-NPT "
        "density/volume convergence is frozen on the intended scaffold and only then advanced to TP0-scale production."
    )

    blocker_statement = (
        "The sole immediate blocker for promoting the current exact CPU claim toward transport-valid charged use is the "
        "absence of a checked-in exact-r-RESPA charged medium-scale long-NPT density/volume convergence artifact. "
        "Gate H already says the charged scaffold still requires long NPT density conditioning; M10.2.1 independently "
        "shows density/volume convergence is the bottleneck, but only as a non-exact diagnostic."
    )

    next_gate_statement = (
        "Next gate: Gate I is now frozen as a predeclared single-rank CPU exact-r-RESPA long-NPT conditioning "
        "campaign on the charged large/medium scaffold. It must still be executed and passed, with density/volume "
        "block-drift and cross-replica criteria frozen in the checked-in Gate I contract, before any TP0-scale "
        "conductivity production campaign or readiness wording is reopened."
    )

    claim_summary = {
        "schema_name": "cpu_exact_respa_claim_summary",
        "schema_version": 1,
        "claim_status_as_of": claim_date,
        "public_claim": public_claim,
        "mechanical_scope_statement": mechanical_statement,
        "openmp_supported_envelope_statement": openmp_scope_statement,
        "openmp_weak_shapes_statement": openmp_weak_shapes_statement,
        "ensemble_boundary_statement": ensemble_statement,
        "transport_boundary_statement": transport_statement,
        "sole_immediate_blocker": blocker_statement,
        "next_gate": next_gate_statement,
        "primary_machine_readable_sources": [
            repo_rel(OUT_ROOT / "mechanical_evidence_index.json"),
            repo_rel(openmp_summary_path),
            repo_rel(OUT_ROOT / "support_matrix.json"),
            repo_rel(OUT_ROOT / "boundary_and_blockers.json"),
            gate_i_contract_path,
            gate_i_manifest_path,
        ],
    }

    mechanical_index = {
        "schema_name": "cpu_exact_respa_mechanical_evidence_index",
        "schema_version": 1,
        "claim_status_as_of": claim_date,
        "gate_a": {
            "status": gate_a["status"],
            "objective": gate_a["objective"],
            "single_rank": gate_a["ntmpi"] == 1,
            "cpu_only": gate_a["reproducibility_flags"][-5:] == ["-nb cpu", "-pme cpu", "-bonded cpu", "-update cpu", "GMX_DISABLE_MODULAR_SIMULATOR=1"],
            "systems": [
                {
                    "system_id": gate_a_small_oligomer["system_id"],
                    "event_trace": repo_rel(Path(gate_a_small_oligomer["event_trace"])),
                    "per_level_force_totals": repo_rel(Path(gate_a_small_oligomer["per_level_force_totals"])),
                    "restart_summary": repo_rel(Path(gate_a_small_oligomer["restart_summary"])),
                    "total_force_summary": repo_rel(Path(gate_a_small_oligomer["total_force_summary"])),
                },
                {
                    "system_id": gate_a_small_salt["system_id"],
                    "event_trace": repo_rel(Path(gate_a_small_salt["event_trace"])),
                    "per_level_force_totals": repo_rel(Path(gate_a_small_salt["per_level_force_totals"])),
                    "restart_summary": repo_rel(Path(gate_a_small_salt["restart_summary"])),
                    "total_force_summary": repo_rel(Path(gate_a_small_salt["total_force_summary"])),
                },
            ],
        },
        "openmp_inventory": {
            "summary": repo_rel(openmp_summary_path),
            "status": openmp_summary["pass"],
            "final_allowed_claim": openmp_summary["final_allowed_claim"],
            "supported_envelope": openmp_summary["supported_envelope"],
            "correctness_only_envelope": openmp_summary["correctness_only_envelope"],
            "unsupported_or_weak_shapes": openmp_summary["unsupported_or_weak_shapes"],
            "host_local_throughput_observations": openmp_summary["host_local_throughput_observations"],
            "host_reports": [repo_rel(path) for path in host_reports],
        },
    }

    status_definitions = {
        "exact": "Checked-in artifacts directly support the statement inside the frozen exact-r-RESPA CPU scope.",
        "approximate": "Evidence exists, but only as a scaffold or caveated short-horizon diagnostic; it does not justify readiness language.",
        "non_exact_diagnostic": "Artifact is useful for blocker diagnosis, but it is not exact-r-RESPA evidence and must not broaden the exact claim.",
        "unsupported": "The repository explicitly says the statement must not be claimed as supported now.",
    }

    support_matrix = {
        "schema_name": "cpu_exact_respa_support_matrix",
        "schema_version": 1,
        "claim_status_as_of": claim_date,
        "status_definitions": status_definitions,
        "top_level_claim": public_claim,
        "items": [
            {
                "id": "mechanics.gate_a_cpu_oracle_event_order_restart",
                "domain": "mechanics",
                "status": "exact",
                "claimable_statement": "Gate A freezes single-rank CPU-only exact-r-RESPA event order, per-level force totals, total-force ledgers, and restart continuity on the two frozen fixtures.",
                "non_claimable_statement": "This is a frozen small-fixture correctness oracle, not medium-scale ensemble or transport readiness evidence.",
                "evidence": [
                    {"path": gate_a_path, "key": "status", "value": gate_a["status"]},
                    {"path": gate_a_path, "key": "objective", "value": gate_a["objective"]},
                    {"path": repo_rel(Path(gate_a_small_oligomer["event_trace"])), "key": "system_id", "value": gate_a_small_oligomer["system_id"]},
                    {"path": repo_rel(Path(gate_a_small_salt["event_trace"])), "key": "system_id", "value": gate_a_small_salt["system_id"]},
                ],
            },
            {
                "id": "mechanics.desktop_cpu_openmp_inventory",
                "domain": "mechanics",
                "status": "exact",
                "claimable_statement": "Across the tested desktop/workstation topology classes, a single-rank CPU-only exact OpenMP mechanics claim is allowed only for the discrete audited ntomp>1 buckets `ntompSmall` and `ntompCeiling` under `-pin auto`, `-pin on`, and `-pin inherit`.",
                "non_claimable_statement": "Do not extend this discrete bucket claim to intermediate or larger ntomp counts, benchmark-only host-local throughput scans, server CPU support, MPI support, or GPU coexistence support.",
                "evidence": [
                    {"path": repo_rel(openmp_summary_path), "key": "pass", "value": openmp_summary["pass"]},
                    {"path": repo_rel(openmp_summary_path), "key": "final_allowed_claim", "value": openmp_summary["final_allowed_claim"]},
                    {"path": repo_rel(openmp_summary_path), "key": "supported_envelope", "value": openmp_summary["supported_envelope"]},
                    {"path": repo_rel(openmp_summary_path), "key": "reports", "value": openmp_summary["reports"]},
                ],
            },
            {
                "id": "mechanics.host_local_openmp_throughput_observations",
                "domain": "mechanics",
                "status": "approximate",
                "claimable_statement": "Checked-in `-pin inherit` benchmark scans provide host-local throughput observations on the tested hosts, including locality-knee notes where available.",
                "non_claimable_statement": "Do not treat these host-local throughput observations as a supported or correctness-only ntomp envelope.",
                "evidence": [
                    {"path": repo_rel(openmp_summary_path), "key": "host_local_throughput_observations", "value": openmp_summary["host_local_throughput_observations"]},
                    {"path": repo_rel(openmp_summary_path), "key": "scope_note", "value": openmp_summary["scope_note"]},
                ],
            },
            {
                "id": "mechanics.desktop_cpu_openmp_outside_audited_buckets",
                "domain": "mechanics",
                "status": "unsupported",
                "claimable_statement": "No current claim extends exact CPU OpenMP support beyond the audited discrete ntomp buckets.",
                "non_claimable_statement": "Do not interpolate from ntomp=1 oracle correctness, no-crash runs, or benchmark throughput probes to intermediate or larger ntomp counts.",
                "evidence": [
                    {"path": repo_rel(openmp_summary_path), "key": "correctness_only_envelope", "value": openmp_summary["correctness_only_envelope"]},
                    {"path": repo_rel(openmp_summary_path), "key": "unsupported_or_weak_shapes", "value": openmp_summary["unsupported_or_weak_shapes"]},
                ],
            },
            {
                "id": "ensemble.gate_g_small_oligomer_exact_nvt",
                "domain": "ensemble",
                "status": "exact",
                "claimable_statement": "Gate G provides narrow exact-r-RESPA long-run NVT evidence on the `small_oligomer` fixture.",
                "non_claimable_statement": "This is not medium-scale transport readiness, and it does not certify charged NPT density conditioning.",
                "evidence": [
                    {"path": gate_g_path, "key": "status", "value": gate_g["status"]},
                    {"path": gate_g_path, "key": "systems[small_oligomer].status", "value": gate_g_small_oligomer["status"]},
                    {"path": gate_g_path, "key": "systems[small_oligomer].ensemble", "value": gate_g_small_oligomer["ensemble"]},
                    {"path": gate_g_path, "key": "systems[small_oligomer].required_observables", "value": gate_g_small_oligomer["required_observables"]},
                ],
            },
            {
                "id": "ensemble.gate_g_small_salt_polymer_box_exact_npt",
                "domain": "ensemble",
                "status": "exact",
                "claimable_statement": "Gate G provides narrow exact-r-RESPA NPT evidence on the `small_salt_polymer_box` fixture, including temperature, pressure, box, volume, and density observables.",
                "non_claimable_statement": "This is still a small-fixture gate only; do not relabel it as medium-scale density convergence or conductivity readiness.",
                "evidence": [
                    {"path": gate_g_path, "key": "systems[small_salt_polymer_box].status", "value": gate_g_small_salt["status"]},
                    {"path": gate_g_path, "key": "systems[small_salt_polymer_box].ensemble", "value": gate_g_small_salt["ensemble"]},
                    {"path": gate_g_path, "key": "systems[small_salt_polymer_box].required_observables", "value": gate_g_small_salt["required_observables"]},
                    {
                        "path": gate_g_path,
                        "key": "systems[small_salt_polymer_box].layout_aggregates.cpu.observables.Density.mean_abs_block_drift",
                        "value": gate_g_small_salt["layout_aggregates"]["cpu"]["observables"]["Density"]["mean_abs_block_drift"],
                    },
                ],
            },
            {
                "id": "ensemble.gate_h_large_medium_exact_nvt_scaffold",
                "domain": "ensemble",
                "status": "approximate",
                "claimable_statement": "Gate H shows that the exact NVT path can be reused on larger transport scaffolds and can generate transport-facing CPU/GPU observables, but only as a NO-GO scaffold exercise.",
                "non_claimable_statement": "Do not turn Gate H PASS-like observable comparisons into transport-production readiness or LAMMPS-vs-GROMACS transport parity.",
                "evidence": [
                    {"path": gate_h_large_path, "key": "status", "value": gate_h_large["status"]},
                    {"path": gate_h_large_path, "key": "production_recommendation", "value": gate_h_large["production_recommendation"]},
                    {"path": gate_h_large_path, "key": "protocol_caveat", "value": gate_h_large["protocol_caveat"]},
                    {"path": gate_h_large_path, "key": "systems[gate_h_dense_oligomer_2x2x2].status", "value": gate_h_large_oligomer["status"]},
                    {"path": gate_h_large_path, "key": "systems[gate_h_dense_salt_polymer_2x2x2].status", "value": gate_h_large_salt["status"]},
                ],
            },
            {
                "id": "ensemble.gate_i_declared_long_npt_conditioning_contract",
                "domain": "ensemble",
                "status": "approximate",
                "claimable_statement": "The repository now freezes Gate I as the exact CPU-only charged long-NPT conditioning contract for the remaining blocker.",
                "non_claimable_statement": "A declared Gate I contract is not a pass and does not create new ensemble or transport evidence by itself.",
                "evidence": [
                    {"path": gate_i_contract_path, "key": "gate_id", "value": gate_i_contract["gate_id"]},
                    {"path": gate_i_contract_path, "key": "status", "value": gate_i_contract["status"]},
                    {"path": gate_i_manifest_path, "key": "status", "value": gate_i_manifest["status"]},
                    {"path": gate_i_contract_path, "key": "acceptance_criteria", "value": gate_i_contract["acceptance_criteria"]},
                ],
            },
            {
                "id": "ensemble.exact_medium_scale_charged_long_npt_density_conditioning",
                "domain": "ensemble",
                "status": "unsupported",
                "claimable_statement": "No checked-in exact-r-RESPA medium-scale charged long-NPT density/volume convergence claim survives today.",
                "non_claimable_statement": "Do not hide the missing charged long-NPT density conditioning behind temperature agreement, short NPT behavior, or NVT-only scaffold runs.",
                "evidence": [
                    {"path": gate_h_large_path, "key": "protocol_caveat", "value": gate_h_large["protocol_caveat"]},
                    {"path": gate_i_manifest_path, "key": "status", "value": gate_i_manifest["status"]},
                    {"path": gate_h_small_path, "key": "systems[small_salt_polymer_box].failure_reasons[-1]", "value": gate_h_small_salt["failure_reasons"][-1]},
                ],
            },
            {
                "id": "diagnostics.m10_medium_scale_plain_md_gates",
                "domain": "diagnostics",
                "status": "non_exact_diagnostic",
                "claimable_statement": "The M10 medium-scale diagnostics still support the blocker narrative only: fixed-volume NVT parity passes, short NPT remains partial, and longer NPT density/volume convergence remains blocked.",
                "non_claimable_statement": "Do not cite M10 as exact-r-RESPA evidence; its generated mdp inputs use plain `integrator = md` and omit the exact-r-RESPA ownership contract.",
                "evidence": [
                    {"path": m10_1_gate_path, "key": "status", "value": m10_1_gate["status"]},
                    {"path": m10_2_gate_path, "key": "overall_status", "value": m10_2_gate["overall_status"]},
                    {"path": m10_2_gate_path, "key": "reports", "value": m10_2_gate["reports"]},
                    {"path": m10_2_1_gate_path, "key": "overall_status", "value": m10_2_1_gate["overall_status"]},
                    {
                        "path": m10_2_1_summary_path,
                        "key": "[0].density_diff_rel",
                        "value": m10_2_1_summary["density_diff_rel"],
                    },
                    {
                        "path": "tools/run_m10_2_ensemble_gate/run_m10_2.py",
                        "key": "mdp_basis",
                        "value": "get_mdp_nvt/get_mdp_npt write plain integrator = md inputs; this is diagnostic-only, not exact-r-RESPA.",
                    },
                    {
                        "path": "tools/run_m10_2_1_convergence_gate/run_m10_2_1.py",
                        "key": "mdp_basis",
                        "value": "get_mdp_npt writes plain integrator = md inputs; this is diagnostic-only, not exact-r-RESPA.",
                    },
                ],
            },
            {
                "id": "transport.cpu_exactness_not_transport_readiness",
                "domain": "transport",
                "status": "unsupported",
                "claimable_statement": "CPU exactness, small-fixture exact NPT, and NVT-only scaffold reuse do not establish conductivity-production readiness.",
                "non_claimable_statement": "Do not use CPU exactness language as a shortcut to conductivity, cNE, transference, or production-readiness wording.",
                "evidence": [
                    {"path": gate_h_large_path, "key": "production_recommendation", "value": gate_h_large["production_recommendation"]},
                    {"path": gate_h_large_path, "key": "status", "value": gate_h_large["status"]},
                    {"path": repo_rel(OUT_ROOT / "boundary_and_blockers.json"), "key": "sole_immediate_blocker.id", "value": "exact_medium_scale_charged_long_npt_density_conditioning"},
                ],
            },
        ],
    }

    boundary_and_blockers = {
        "schema_name": "cpu_exact_respa_boundary_and_blockers",
        "schema_version": 1,
        "claim_status_as_of": claim_date,
        "exact_scope": {
            "mechanics": {
                "status": "closed",
                "statement": mechanical_statement,
                "basis": [gate_a_path, repo_rel(openmp_summary_path)],
            },
            "small_fixture_ensemble": {
                "status": "closed_but_narrow",
                "statement": "Gate G closes a narrow exact-r-RESPA long-run ensemble boundary on small fixtures only.",
                "basis": [gate_g_path],
            },
            "medium_scale_exact_nvt": {
                "status": "scaffold_only",
                "statement": "Gate H large/medium confirms exact-NVT scaffold reuse only; it remains a NO-GO for production transport.",
                "basis": [gate_h_large_path],
            },
            "medium_scale_exact_npt": {
                "status": "missing",
                "statement": "No checked-in exact-r-RESPA charged medium-scale long-NPT density/volume convergence artifact is available.",
                "basis": [gate_h_large_path, gate_h_small_path, gate_i_manifest_path],
            },
            "legacy_non_exact_medium_scale_diagnostics": {
                "status": "diagnostic_only",
                "statement": "M10 medium-scale NVT/NPT files remain useful diagnostics, but they are not exact-r-RESPA evidence.",
                "basis": [m10_2_gate_path, m10_2_1_gate_path],
            },
        },
        "sole_immediate_blocker": {
            "id": "exact_medium_scale_charged_long_npt_density_conditioning",
            "statement": blocker_statement,
            "basis": [
                {"path": gate_h_large_path, "key": "protocol_caveat", "value": gate_h_large["protocol_caveat"]},
                {"path": gate_i_manifest_path, "key": "status", "value": gate_i_manifest["status"]},
                {"path": gate_h_small_path, "key": "systems[small_salt_polymer_box].failure_reasons[-1]", "value": gate_h_small_salt["failure_reasons"][-1]},
                {"path": m10_2_1_gate_path, "key": "overall_status", "value": m10_2_1_gate["overall_status"]},
            ],
        },
        "next_gate": {
            "id": gate_i_contract["gate_id"],
            "statement": next_gate_statement,
            "must_prove": [
                "single-rank CPU exact-r-RESPA charged large/medium long-NPT run remains stable on the intended scaffold",
                "density and volume block-drift fall within predeclared convergence thresholds across replicas",
                "the resulting conditioned state is frozen before any TP0-scale transport production campaign",
            ],
            "still_not_implied_even_if_passes": [
                "TP0-scale production length is still required afterward",
                "LAMMPS-vs-GROMACS transport parity is still a separate question",
                "conductivity-production readiness still requires transport-side uncertainty and linearity gates",
            ],
            "frozen_contract": gate_i_contract_path,
            "current_manifest": gate_i_manifest_path,
        },
    }

    write_json(OUT_ROOT / "cpu_exact_claim_summary.json", claim_summary)
    write_json(OUT_ROOT / "mechanical_evidence_index.json", mechanical_index)
    write_json(OUT_ROOT / "support_matrix.json", support_matrix)
    write_json(OUT_ROOT / "boundary_and_blockers.json", boundary_and_blockers)


if __name__ == "__main__":
    main()
