from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_CSV = WORKSPACE_ROOT / "MY_PAPER_RELATED" / "MODELS" / "data" / "simulation-trajectory-aggregate_aligned.csv"
DEFAULT_LAMMPS_BATCH_ROOT = WORKSPACE_ROOT / "MY_PAPER_RELATED" / "LAMMPS_BATCH" / "batch_runs"
DEFAULT_GROMACS_BATCH_ROOT = WORKSPACE_ROOT / "MY_PAPER_RELATED" / "GROMACS_PCFF_BATCH" / "batch_runs"
DEFAULT_OUT_ROOT = REPO_ROOT / "tests" / "reference_results" / "lunar_gromacs_pcff_converter" / "database_scope"

REQUIRED_CORE_COLUMNS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether an aligned polymer database snapshot can support a converter coverage claim."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--lammps-batch-root", type=Path, default=DEFAULT_LAMMPS_BATCH_ROOT)
    parser.add_argument("--gromacs-batch-root", type=Path, default=DEFAULT_GROMACS_BATCH_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    resolved = path.resolve()
    for root in (REPO_ROOT, WORKSPACE_ROOT):
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(resolved)


def read_csv_preserving_blank_headers(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    if header[: len(REQUIRED_CORE_COLUMNS)] != REQUIRED_CORE_COLUMNS:
        raise ValueError(f"Unexpected core CSV columns in {path}")
    row_widths = {len(row) for row in rows}
    if row_widths != {len(header)}:
        raise ValueError(f"CSV row widths do not match header width in {path}: {sorted(row_widths)}")
    return header, rows


def build_snapshot(csv_path: Path) -> dict:
    header, rows = read_csv_preserving_blank_headers(csv_path)
    trajectory_ids = [row[0].strip() for row in rows]
    smiles_values = [row[1].strip() for row in rows]
    rows_by_smiles: dict[str, list[list[str]]] = defaultdict(list)
    for row in rows:
        rows_by_smiles[row[1].strip()].append(row)

    dp_conflicts = []
    for smiles, smiles_rows in sorted(rows_by_smiles.items()):
        values = sorted({row[4].strip() for row in smiles_rows})
        if len(values) > 1:
            dp_conflicts.append(
                {
                    "smiles_sha256": hashlib.sha256(smiles.encode("utf-8")).hexdigest(),
                    "degree_of_polymerization_values": values,
                    "row_count": len(smiles_rows),
                }
            )

    optional_columns = []
    for index, name in enumerate(header[len(REQUIRED_CORE_COLUMNS) :], start=len(REQUIRED_CORE_COLUMNS)):
        nonempty_values = [row[index].strip() for row in rows if row[index].strip()]
        optional_columns.append(
            {
                "index": index,
                "name": name,
                "nonempty_count": len(nonempty_values),
                "distinct_count": len(set(nonempty_values)),
                "distinct_sample": sorted(set(nonempty_values))[:10],
            }
        )

    return {
        "schema_name": "lunar_gromacs_pcff_database_scope_snapshot",
        "schema_version": 1,
        "scope_id": "simulation_trajectory_aggregate_aligned_csv_snapshot",
        "source_csv": {
            "path": rel(csv_path),
            "absolute_path": str(csv_path.resolve()),
            "sha256": sha256_file(csv_path),
            "byte_count": csv_path.stat().st_size,
        },
        "columns": {
            "count": len(header),
            "names": header,
            "required_core_columns": REQUIRED_CORE_COLUMNS,
            "blank_header_indexes": [index for index, name in enumerate(header) if not name.strip()],
            "optional_columns": optional_columns,
        },
        "counts": {
            "row_count": len(rows),
            "unique_trajectory_id_count": len(set(trajectory_ids)),
            "duplicate_trajectory_id_count": len(trajectory_ids) - len(set(trajectory_ids)),
            "unique_smiles_count": len(set(smiles_values)),
            "duplicate_smiles_row_count": len(smiles_values) - len(set(smiles_values)),
            "degree_of_polymerization_conflict_unique_smiles_count": len(dp_conflicts),
        },
        "label_counts": dict(sorted(Counter(row[11].strip() for row in rows if row[11].strip()).items())),
        "degree_of_polymerization_conflicts": dp_conflicts[:20],
        "important_interpretation": [
            "This CSV is a database snapshot and not itself a LUNAR/LAMMPS topology conversion artifact.",
            "A database-wide success claim requires per-case parser, mapping, emission, and grompp artifacts.",
            "The Li/TFSI/molality columns indicate electrolyte-context records; they do not prove charged GROMACS conversion support.",
        ],
    }


def load_json_if_exists(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_checked_in_scope_comparison(snapshot: dict) -> dict:
    unique_manifest_path = REPO_ROOT / "data_manifests" / "simulation_trajectory_aggregate_unique_smiles.json"
    snapshot_manifest_path = REPO_ROOT / "data_manifests" / "simulation_trajectory_aggregate_snapshot.json"
    unique_manifest = load_json_if_exists(unique_manifest_path)
    snapshot_manifest = load_json_if_exists(snapshot_manifest_path)

    comparison = {
        "checked_in_snapshot_manifest": rel(snapshot_manifest_path),
        "checked_in_unique_manifest": rel(unique_manifest_path),
        "available": unique_manifest is not None and snapshot_manifest is not None,
    }
    if not unique_manifest or not snapshot_manifest:
        return comparison

    csv_header, csv_rows = read_csv_preserving_blank_headers(Path(snapshot["source_csv"]["absolute_path"]))
    aligned_smiles = {row[1].strip() for row in csv_rows}
    checked_in_smiles = {entry["smiles"] for entry in unique_manifest["entries"]}

    comparison.update(
        {
            "checked_in_snapshot_sha256": snapshot_manifest["source_csv"]["sha256"],
            "same_snapshot_hash": snapshot_manifest["source_csv"]["sha256"] == snapshot["source_csv"]["sha256"],
            "same_row_count": snapshot_manifest["row_count"] == snapshot["counts"]["row_count"],
            "same_unique_smiles_count": snapshot_manifest["unique_smiles_count"]
            == snapshot["counts"]["unique_smiles_count"],
            "same_unique_smiles_set": aligned_smiles == checked_in_smiles,
            "aligned_minus_checked_in_unique_smiles_count": len(aligned_smiles - checked_in_smiles),
            "checked_in_minus_aligned_unique_smiles_count": len(checked_in_smiles - aligned_smiles),
        }
    )
    return comparison


def trajectory_dirs(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    dirs = {}
    for path in sorted(root.glob("Traj_*")):
        if path.is_dir():
            dirs[path.name.removeprefix("Traj_")] = path
    return dirs


def find_case_artifacts(case_dir: Path) -> dict:
    prepare_report_path = case_dir / "build" / "gromacs_pcff" / "gromacs_pcff_prepare_report.json"
    prepare_report = load_json_if_exists(prepare_report_path)
    topologies = sorted(case_dir.rglob("topol.top"))
    tprs = sorted(case_dir.rglob("*.tpr"))
    grompp_reports = sorted(case_dir.rglob("grompp_smoke_report.json"))

    return {
        "case_dir": rel(case_dir),
        "prepare_report": {
            "path": rel(prepare_report_path),
            "exists": prepare_report is not None,
            "overall_status": prepare_report.get("workflow", {}).get("overall_status") if prepare_report else None,
            "failure_reason": prepare_report.get("workflow", {}).get("failure_reason") if prepare_report else None,
            "polymer_typing_status": prepare_report.get("polymer_typing", {}).get("status") if prepare_report else None,
            "polymer_topology_status": prepare_report.get("polymer_topology", {}).get("status") if prepare_report else None,
        },
        "topol_top_count": len(topologies),
        "topol_top_paths": [rel(path) for path in topologies[:10]],
        "tpr_count": len(tprs),
        "tpr_paths": [rel(path) for path in tprs[:10]],
        "grompp_smoke_report_count": len(grompp_reports),
        "grompp_smoke_report_paths": [rel(path) for path in grompp_reports[:10]],
    }


def find_repo_lunar_smoke_summary() -> tuple[Path, dict | None]:
    base = REPO_ROOT / "tests" / "reference_results" / "lunar_gromacs_pcff_converter"
    candidates = [
        base / "database_lunar_smoke_pcff_fixed_all" / "database_lunar_smoke_summary.json",
        base / "database_lunar_smoke_pcff_build" / "database_lunar_smoke_summary.json",
        base / "database_lunar_smoke" / "database_lunar_smoke_summary.json",
    ]
    for path in candidates:
        payload = load_json_if_exists(path)
        if payload is not None:
            return path, payload
    return candidates[0], None


def build_artifact_audit(snapshot: dict, lammps_batch_root: Path, gromacs_batch_root: Path) -> dict:
    csv_path = Path(snapshot["source_csv"]["absolute_path"])
    _, rows = read_csv_preserving_blank_headers(csv_path)
    csv_trajectory_ids = {row[0].strip() for row in rows}
    lunar_smoke_summary_path, lunar_smoke_summary = find_repo_lunar_smoke_summary()

    lammps_dirs = trajectory_dirs(lammps_batch_root)
    gromacs_dirs = trajectory_dirs(gromacs_batch_root)
    gromacs_case_artifacts = {
        trajectory_id: find_case_artifacts(case_dir) for trajectory_id, case_dir in sorted(gromacs_dirs.items())
    }
    prepare_report_count = sum(
        1 for artifacts in gromacs_case_artifacts.values() if artifacts["prepare_report"]["exists"]
    )
    grompp_report_count = sum(
        artifacts["grompp_smoke_report_count"] for artifacts in gromacs_case_artifacts.values()
    )
    topol_top_count = sum(artifacts["topol_top_count"] for artifacts in gromacs_case_artifacts.values())
    tpr_count = sum(artifacts["tpr_count"] for artifacts in gromacs_case_artifacts.values())

    reasons = []
    if len(lammps_dirs) != snapshot["counts"]["row_count"]:
        reasons.append(
            f"LAMMPS batch directories cover {len(lammps_dirs)} trajectory IDs, not {snapshot['counts']['row_count']} CSV rows."
        )
    if len(gromacs_dirs) != snapshot["counts"]["row_count"]:
        reasons.append(
            f"GROMACS batch directories cover {len(gromacs_dirs)} trajectory IDs, not {snapshot['counts']['row_count']} CSV rows."
        )
    if grompp_report_count == 0:
        reasons.append("No grompp smoke reports were found under the audited GROMACS batch root.")
    if topol_top_count == 0:
        reasons.append("No emitted topol.top files were found under the audited GROMACS batch root.")
    if lunar_smoke_summary is None:
        reasons.append("No repository database_lunar_smoke summary was found.")
    else:
        smoke_totals = lunar_smoke_summary.get("totals", {})
        if smoke_totals.get("pass_count", 0) != snapshot["counts"]["row_count"]:
            reasons.append(
                "Repository database_lunar_smoke evidence covers "
                f"{smoke_totals.get('pass_count', 0)} passing rows, not {snapshot['counts']['row_count']} CSV rows."
            )

    blocked_reports = [
        artifacts
        for artifacts in gromacs_case_artifacts.values()
        if artifacts["prepare_report"]["exists"] and artifacts["prepare_report"]["overall_status"] != "prepared"
    ]
    if blocked_reports:
        reasons.append(
            "At least one available GROMACS_PCFF prepare report is blocked or unsupported, not a completed conversion."
        )

    smoke_totals = lunar_smoke_summary.get("totals", {}) if lunar_smoke_summary else {}
    smoke_pass_count = smoke_totals.get("pass_count")
    smoke_failure_count = smoke_totals.get("failure_count")
    smoke_missing_count = smoke_totals.get("missing_lunar_pcff_data_count")
    db_wide_smoke_claimable = (
        lunar_smoke_summary is not None
        and smoke_totals.get("selected_row_count") == snapshot["counts"]["row_count"]
        and smoke_pass_count == snapshot["counts"]["row_count"]
        and smoke_failure_count == 0
        and smoke_missing_count == 0
    )
    if db_wide_smoke_claimable:
        reasons = []
        strongest_supported_database_statement = (
            "The aligned CSV snapshot identity is evidenced and all selected LUNAR PCFF single-chain rows have "
            "parser->mapping->emission->grompp PASS artifacts in the repository smoke summary."
        )
    elif lunar_smoke_summary is not None and smoke_pass_count:
        strongest_supported_database_statement = (
            "The aligned CSV snapshot identity is evidenced; repository database_lunar_smoke currently has "
            f"{smoke_pass_count} passing parser->mapping->emission->grompp rows and {smoke_missing_count} rows "
            "missing LUNAR PCFF data, so database-wide success remains not_claimable."
        )
    else:
        strongest_supported_database_statement = (
            "The aligned CSV snapshot identity and current lack of database-wide public converter artifacts are evidenced."
        )

    return {
        "schema_name": "lunar_gromacs_pcff_database_claim_audit",
        "schema_version": 1,
        "scope_id": snapshot["scope_id"],
        "source_csv_sha256": snapshot["source_csv"]["sha256"],
        "batch_roots": {
            "lammps": rel(lammps_batch_root),
            "gromacs_pcff": rel(gromacs_batch_root),
        },
        "coverage": {
            "csv_row_count": snapshot["counts"]["row_count"],
            "csv_unique_smiles_count": snapshot["counts"]["unique_smiles_count"],
            "lammps_batch_trajectory_dir_count": len(lammps_dirs),
            "gromacs_batch_trajectory_dir_count": len(gromacs_dirs),
            "lammps_batch_ids_in_csv_count": len(set(lammps_dirs) & csv_trajectory_ids),
            "gromacs_batch_ids_in_csv_count": len(set(gromacs_dirs) & csv_trajectory_ids),
            "gromacs_pcff_prepare_report_count": prepare_report_count,
            "gromacs_topol_top_count": topol_top_count,
            "gromacs_tpr_count": tpr_count,
            "grompp_smoke_report_count": grompp_report_count,
            "repo_database_lunar_smoke_summary_exists": lunar_smoke_summary is not None,
            "repo_database_lunar_smoke_selected_row_count": (
                lunar_smoke_summary.get("totals", {}).get("selected_row_count") if lunar_smoke_summary else None
            ),
            "repo_database_lunar_smoke_pass_count": (
                lunar_smoke_summary.get("totals", {}).get("pass_count") if lunar_smoke_summary else None
            ),
            "repo_database_lunar_smoke_missing_lunar_pcff_data_count": (
                lunar_smoke_summary.get("totals", {}).get("missing_lunar_pcff_data_count")
                if lunar_smoke_summary
                else None
            ),
        },
        "repo_database_lunar_smoke_summary": {
            "path": rel(lunar_smoke_summary_path),
            "exists": lunar_smoke_summary is not None,
            "database_wide_converter_success_status": (
                lunar_smoke_summary.get("claim_evaluation", {}).get("database_wide_converter_success_status")
                if lunar_smoke_summary
                else None
            ),
        },
        "available_gromacs_cases": gromacs_case_artifacts,
        "claim_evaluation": {
            "database_wide_converter_success_status": "claimable" if db_wide_smoke_claimable else "not_claimable",
            "strongest_supported_database_statement": strongest_supported_database_statement,
            "rejected_overclaim": (
                None
                if db_wide_smoke_claimable
                else "All rows or unique SMILES in the aligned database successfully convert through parser, mapping, "
                "emission, and grompp."
            ),
            "reasons": reasons,
        },
        "charged_ion_boundary": {
            "status": "closed_to_claim_until_separate_evidence_exists",
            "reason": (
                "The database contains electrolyte-context columns and batch templates mention Li/TFSI, "
                "but no charged parser->mapping->emission->grompp evidence chain was found for the database."
            ),
        },
    }


def build_support_matrix_extension(snapshot: dict, audit: dict, comparison: dict) -> dict:
    repo_smoke_pass_count = audit["coverage"].get("repo_database_lunar_smoke_pass_count")
    repo_smoke_selected_count = audit["coverage"].get("repo_database_lunar_smoke_selected_row_count")
    database_success = audit["claim_evaluation"]["database_wide_converter_success_status"] == "claimable"
    return {
        "schema_name": "lunar_gromacs_pcff_database_scope_support_matrix_extension",
        "schema_version": 1,
        "claim_status_as_of": date.today().isoformat(),
        "items": [
            {
                "id": "database_scope.aligned_csv_snapshot_identity",
                "status": "exact",
                "claimable_statement": "The aligned CSV database snapshot identity is fixed by path, hash, schema, and counts.",
                "evidence": ["tests/reference_results/lunar_gromacs_pcff_converter/database_scope/database_scope_snapshot.json"],
            },
            {
                "id": "database_scope.unique_smiles_equivalence_to_existing_scope",
                "status": "exact" if comparison.get("same_unique_smiles_set") else "unsupported",
                "claimable_statement": (
                    "The aligned CSV has the same unique SMILES set as the existing checked-in CSV scope snapshot."
                ),
                "evidence": [
                    "tests/reference_results/lunar_gromacs_pcff_converter/database_scope/database_scope_snapshot.json",
                    "data_manifests/simulation_trajectory_aggregate_unique_smiles.json",
                ],
            },
            {
                "id": "database_scope.database_wide_converter_success",
                "status": "exact" if database_success else "unsupported",
                "claimable_statement": (
                    "The current public artifacts support a polymer-only database-wide grompp smoke claim for the aligned CSV."
                    if database_success
                    else "The current public artifacts do not support a database-wide converter success claim for the aligned CSV."
                ),
                "evidence": ["tests/reference_results/lunar_gromacs_pcff_converter/database_scope/database_claim_audit.json"],
            },
            {
                "id": "database_scope.repository_lunar_smoke_coverage",
                "status": (
                    "partial"
                    if repo_smoke_pass_count and repo_smoke_selected_count != repo_smoke_pass_count
                    else "exact"
                    if repo_smoke_pass_count and repo_smoke_selected_count == repo_smoke_pass_count
                    else "unsupported"
                ),
                "claimable_statement": (
                    "Repository database_lunar_smoke artifacts support all selected LUNAR PCFF single-chain rows."
                    if database_success
                    else "Repository database_lunar_smoke artifacts support only the passing available LUNAR PCFF "
                    "single-chain rows, not database-wide conversion."
                ),
                "evidence": [
                    audit["repo_database_lunar_smoke_summary"]["path"],
                    "tests/reference_results/lunar_gromacs_pcff_converter/database_scope/database_claim_audit.json",
                ],
            },
            {
                "id": "database_scope.charged_ion_extension",
                "status": "gated",
                "claimable_statement": (
                    "Electrolyte/charged support for this database requires a separate charged parser->mapping->emission->grompp chain."
                ),
                "evidence": ["tests/reference_results/lunar_gromacs_pcff_converter/database_scope/database_claim_audit.json"],
            },
        ],
        "claim_evaluation": audit["claim_evaluation"],
    }


def write_readme(out_root: Path, snapshot: dict, audit: dict) -> None:
    (out_root / "README.md").write_text(
        "\n".join(
            [
                "# Aligned Database Scope Audit",
                "",
                "This directory records whether the provided aligned polymer database can support a database-wide converter claim.",
                "",
                "Snapshot:",
                "",
                f"- path: `{snapshot['source_csv']['path']}`",
                f"- sha256: `{snapshot['source_csv']['sha256']}`",
                f"- rows: `{snapshot['counts']['row_count']}`",
                f"- unique SMILES: `{snapshot['counts']['unique_smiles_count']}`",
                "",
                "Verdict:",
                "",
                f"- database-wide converter success: `{audit['claim_evaluation']['database_wide_converter_success_status']}`",
                f"- LAMMPS batch trajectory directories found: `{audit['coverage']['lammps_batch_trajectory_dir_count']}`",
                f"- GROMACS batch trajectory directories found: `{audit['coverage']['gromacs_batch_trajectory_dir_count']}`",
                f"- grompp smoke reports found: `{audit['coverage']['grompp_smoke_report_count']}`",
                f"- repository database_lunar_smoke pass count: `{audit['coverage']['repo_database_lunar_smoke_pass_count']}`",
                (
                    "- repository database_lunar_smoke missing LUNAR PCFF data count: "
                    f"`{audit['coverage']['repo_database_lunar_smoke_missing_lunar_pcff_data_count']}`"
                ),
                "",
                "This audit does not run conversion. It prevents overclaiming by checking whether the public artifact chain exists.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    csv_path = args.csv.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    snapshot = build_snapshot(csv_path)
    comparison = build_checked_in_scope_comparison(snapshot)
    snapshot["checked_in_csv_scope_comparison"] = comparison
    audit = build_artifact_audit(snapshot, args.lammps_batch_root.resolve(), args.gromacs_batch_root.resolve())
    support_extension = build_support_matrix_extension(snapshot, audit, comparison)

    dump_json(out_root / "database_scope_snapshot.json", snapshot)
    dump_json(out_root / "database_claim_audit.json", audit)
    dump_json(out_root / "support_matrix_extension.json", support_extension)
    write_readme(out_root, snapshot, audit)

    print(
        json.dumps(
            {
                "status": audit["claim_evaluation"]["database_wide_converter_success_status"],
                "out_root": rel(out_root),
                "csv_sha256": snapshot["source_csv"]["sha256"],
                "row_count": snapshot["counts"]["row_count"],
                "unique_smiles_count": snapshot["counts"]["unique_smiles_count"],
                "grompp_smoke_report_count": audit["coverage"]["grompp_smoke_report_count"],
                "repo_database_lunar_smoke_pass_count": audit["coverage"]["repo_database_lunar_smoke_pass_count"],
                "repo_database_lunar_smoke_missing_lunar_pcff_data_count": audit["coverage"][
                    "repo_database_lunar_smoke_missing_lunar_pcff_data_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
