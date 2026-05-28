from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

from audit_database_scope import DEFAULT_CSV, DEFAULT_LAMMPS_BATCH_ROOT, REPO_ROOT, WORKSPACE_ROOT
from lunar_data_converter import (
    dump_json,
    convert_lunar_data_text,
    sha256_file,
    sha256_text,
    smoke_validation_cutoff_nm,
)


DEFAULT_OUT_ROOT = REPO_ROOT / "tests" / "reference_results" / "lunar_gromacs_pcff_converter" / "database_lunar_smoke"


SMOKE_TEMPLATE = """integrator = md
nsteps = 1
dt = 0.001
cutoff-scheme = Verlet
vdw-type = Cut-off
rvdw = {cutoff:.3f}
coulombtype = PME
rcoulomb = {cutoff:.3f}
pbc = xyz
nstlist = 10
rlist = {cutoff:.3f}
verlet-buffer-tolerance = 0.005
constraints = none
gen-vel = yes
gen-temp = 300
gen-seed = 12345
ld-seed = 12345
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parser -> mapping -> emission -> grompp smoke on LUNAR PCFF data artifacts for a CSV database."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--lammps-batch-root", type=Path, default=DEFAULT_LAMMPS_BATCH_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--gmx", type=Path, default=None)
    parser.add_argument("--scope", choices=["available", "csv"], default="available")
    parser.add_argument("--trajectory-id", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--allow-failures", action="store_true", help="Record failures in the summary but exit 0.")
    return parser.parse_args()


def rel(path: Path) -> str:
    resolved = path.resolve()
    for root in (REPO_ROOT, WORKSPACE_ROOT):
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(resolved)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_gmx(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    build_gmx = REPO_ROOT / "build" / "bin" / "gmx"
    if build_gmx.is_file():
        return build_gmx
    path_gmx = shutil.which("gmx")
    if path_gmx:
        return Path(path_gmx).resolve()
    raise FileNotFoundError("No GROMACS executable found. Pass --gmx explicitly.")


def source_data_for_trajectory(batch_root: Path, trajectory_id: str) -> Path | None:
    case_root = batch_root / f"Traj_{trajectory_id}"
    preferred = case_root / "build" / "lunar_pcff" / "chain_fixed_typed_nodup_IFF_nodup.data"
    if preferred.is_file():
        return preferred
    candidates = sorted((case_root / "build" / "lunar_pcff").glob("*_nodup.data"))
    return candidates[0] if candidates else None


def select_rows(rows: list[dict[str, str]], batch_root: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    if args.trajectory_id:
        requested = set(args.trajectory_id)
        selected = [row for row in rows if row["Trajectory ID"] in requested]
    elif args.scope == "available":
        selected = [row for row in rows if source_data_for_trajectory(batch_root, row["Trajectory ID"]) is not None]
    else:
        selected = list(rows)
    if args.max_cases is not None:
        selected = selected[: max(0, int(args.max_cases))]
    return selected


def topology_sections(topology_text: str) -> list[str]:
    sections = []
    for line in topology_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            sections.append(stripped.strip("[] "))
    return sections


def count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


def build_parser_summary(row: dict[str, str], data_path: Path, parsed_data: dict) -> dict:
    coeffs = parsed_data["coeffs"]
    return {
        "schema_name": "lunar_gromacs_pcff_database_lunar_parser_summary",
        "schema_version": 1,
        "status": "PASS",
        "trajectory_id": row["Trajectory ID"],
        "smiles": row["SMILES"],
        "source_lunar_data": {"path": rel(data_path), "sha256": sha256_file(data_path)},
        "parsed_counts": {
            "atoms": parsed_data["header_counts"]["atoms"],
            "bonds": parsed_data["header_counts"]["bonds"],
            "angles": parsed_data["header_counts"]["angles"],
            "dihedrals": parsed_data["header_counts"]["dihedrals"],
            "impropers": parsed_data["header_counts"]["impropers"],
            "atom_types": parsed_data["type_counts"]["atom types"],
            "bond_types": parsed_data["type_counts"]["bond types"],
            "angle_types": parsed_data["type_counts"]["angle types"],
            "dihedral_types": parsed_data["type_counts"]["dihedral types"],
            "improper_types": parsed_data["type_counts"]["improper types"],
        },
        "coefficient_counts": {
            "pair_coeffs": len(coeffs["pair_coeffs"]),
            "bond_coeffs": len(coeffs["bond_coeffs"]),
            "angle_coeffs": len(coeffs["angle_coeffs"]),
            "dihedral_coeffs": len(coeffs["dihedral_coeffs"]),
            "improper_coeffs": len(coeffs["improper_coeffs"]),
        },
        "source_line_samples": {
            "header": parsed_data["header_line"],
            "pair_coeffs": parsed_data["section_sources"].get("Pair Coeffs"),
            "atoms": parsed_data["atoms"][0]["source"] if parsed_data["atoms"] else None,
        },
    }


def build_mapping_summary(row: dict[str, str], typed_ir: dict) -> dict:
    template_count = len(typed_ir["molecule_templates"])
    instance_count = len(typed_ir["molecule_instances"])
    generated_pairs = sum(len(template["generated_pairs"]) for template in typed_ir["molecule_templates"])
    atom_count = sum(len(template["atoms"]) for template in typed_ir["molecule_templates"])
    net_charge = sum(
        atom["charge_e"] * sum(1 for instance in typed_ir["molecule_instances"] if instance["template_name"] == template["name"])
        for template in typed_ir["molecule_templates"]
        for atom in template["atoms"]
    )
    return {
        "schema_name": "lunar_gromacs_pcff_database_lunar_mapping_summary",
        "schema_version": 1,
        "status": "PASS",
        "trajectory_id": row["Trajectory ID"],
        "smiles": row["SMILES"],
        "mapping_contract": "LUNAR embedded Class2 coefficient sections -> GROMACS PCFF class2 topology terms",
        "molecule_scope": {
            "template_count": template_count,
            "instance_count": instance_count,
            "template_atom_count_total": atom_count,
            "generated_pair14_count": generated_pairs,
            "net_charge_e": round(net_charge, 8),
            "polymer_only_single_chain_data": True,
            "separate_ion_or_salt_molecules": False,
        },
        "styles": typed_ir["styles"],
        "parameter_fallbacks": typed_ir.get("diagnostics", {}).get("parameter_fallbacks", []),
        "unsupported_by_this_mapping_claim": [
            "charged Li/TFSI or salt molecule conversion",
            "dense polymer-electrolyte assembly",
            "physical parameter-completion claims for smoke-only fallback atom types",
            "database-wide success unless every selected row has a PASS grompp smoke report",
        ],
    }


def build_emission_summary(row: dict[str, str], topol_path: Path, topology_text: str) -> dict:
    return {
        "schema_name": "lunar_gromacs_pcff_database_lunar_emission_summary",
        "schema_version": 1,
        "status": "PASS",
        "trajectory_id": row["Trajectory ID"],
        "smiles": row["SMILES"],
        "emitted_topology": rel(topol_path),
        "topology_sha256": sha256_text(topology_text),
        "topology_line_count": len(topology_text.splitlines()),
        "sections": topology_sections(topology_text),
        "grompp_required_for_viability_claim": True,
    }


def run_grompp(case_root: Path, gmx: Path, cutoff: float, row: dict[str, str]) -> dict:
    grompp_root = case_root / "grompp"
    grompp_root.mkdir(parents=True, exist_ok=True)
    smoke_mdp = grompp_root / "smoke.mdp"
    stdout_path = grompp_root / "grompp.stdout"
    stderr_path = grompp_root / "grompp.stderr"
    tpr_path = grompp_root / "smoke.tpr"
    mdout_path = grompp_root / "mdout.mdp"

    smoke_mdp.write_text(SMOKE_TEMPLATE.format(cutoff=cutoff), encoding="utf-8")
    for path in (stdout_path, stderr_path, tpr_path, mdout_path):
        if path.exists():
            path.unlink()

    command = [
        str(gmx),
        "grompp",
        "-f",
        str(smoke_mdp),
        "-c",
        str(case_root / "system.gro"),
        "-p",
        str(case_root / "topol.top"),
        "-o",
        str(tpr_path),
        "-po",
        str(mdout_path),
    ]
    env = dict(os.environ)
    env["GMX_NO_QUOTES"] = "1"
    result = subprocess.run(command, cwd=case_root, text=True, capture_output=True, env=env, check=False)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")

    warning_count = count_pattern(result.stderr, r"\bWARNING\s+\d+\b")
    note_count = count_pattern(result.stderr, r"\bNOTE\s+\d+\b")
    status = "PASS" if result.returncode == 0 and warning_count == 0 and tpr_path.exists() else "FAIL"
    report = {
        "schema_name": "lunar_gromacs_pcff_database_lunar_grompp_smoke",
        "schema_version": 1,
        "status": status,
        "trajectory_id": row["Trajectory ID"],
        "smiles": row["SMILES"],
        "scope": "polymer-only LUNAR PCFF single-chain data smoke; not charged electrolyte conversion",
        "command": command,
        "working_directory": rel(case_root),
        "cutoff_nm": round(cutoff, 6),
        "returncode": result.returncode,
        "warning_count": warning_count,
        "note_count": note_count,
        "stdout_path": rel(stdout_path),
        "stderr_path": rel(stderr_path),
        "generated_outputs": {
            "smoke_tpr": {
                "path": rel(tpr_path),
                "exists": tpr_path.exists(),
                "sha256": sha256_file(tpr_path) if tpr_path.exists() else None,
            },
            "mdout_mdp": {
                "path": rel(mdout_path),
                "exists": mdout_path.exists(),
                "sha256": sha256_file(mdout_path) if mdout_path.exists() else None,
            },
        },
        "success_criteria": {
            "returncode_zero": result.returncode == 0,
            "warning_count_zero": warning_count == 0,
            "tpr_generated": tpr_path.exists(),
        },
    }
    dump_json(grompp_root / "grompp_smoke_report.json", report)
    return report


def run_case(row: dict[str, str], data_path: Path, out_root: Path, gmx: Path) -> dict:
    case_root = out_root / "cases" / f"Traj_{row['Trajectory ID']}"
    if case_root.exists():
        shutil.rmtree(case_root)
    case_root.mkdir(parents=True)

    source_copy_path = case_root / "source_lunar_pcff.data"
    shutil.copy2(data_path, source_copy_path)
    parsed_data, typed_ir, topol_text, gro_text = convert_lunar_data_text(
        source_copy_path, system_id=f"Traj_{row['Trajectory ID']}"
    )
    topol_path = case_root / "topol.top"
    gro_path = case_root / "system.gro"
    topol_path.write_text(topol_text, encoding="utf-8")
    gro_path.write_text(gro_text, encoding="utf-8")
    dump_json(case_root / "typed_system.json", typed_ir)
    parser_summary = build_parser_summary(row, source_copy_path, parsed_data)
    parser_summary["original_source_lunar_data"] = {"path": rel(data_path), "sha256": sha256_file(data_path)}
    dump_json(case_root / "parser_summary.json", parser_summary)
    dump_json(case_root / "mapping_summary.json", build_mapping_summary(row, typed_ir))
    dump_json(case_root / "emission_summary.json", build_emission_summary(row, topol_path, topol_text))

    grompp_report = run_grompp(case_root, gmx, smoke_validation_cutoff_nm(parsed_data), row)
    return {
        "trajectory_id": row["Trajectory ID"],
        "smiles": row["SMILES"],
        "status": "pass" if grompp_report["status"] == "PASS" else "failure",
        "source_lunar_data": rel(source_copy_path),
        "original_source_lunar_data": rel(data_path),
        "case_artifact_root": rel(case_root),
        "grompp_status": grompp_report["status"],
        "grompp_report": rel(case_root / "grompp" / "grompp_smoke_report.json"),
        "warning_count": grompp_report["warning_count"],
        "returncode": grompp_report["returncode"],
    }


def write_readme(out_root: Path, summary: dict) -> None:
    (out_root / "README.md").write_text(
        "\n".join(
            [
                "# Database LUNAR PCFF Smoke Evidence",
                "",
                "This directory contains parser -> mapping -> emission -> grompp smoke artifacts for LUNAR PCFF data files.",
                "",
                "Boundary:",
                "",
                "- Input artifacts are existing LUNAR `all2lmp` PCFF single-chain `.data` files.",
                "- Each passing case includes the inspected input copy as `source_lunar_pcff.data`.",
                "- This is polymer-only single-chain topology smoke evidence.",
                "- This is not charged Li/TFSI or dense polymer-electrolyte conversion support.",
                "- LUNAR generation warnings, when present in `lunar_generation_status.jsonl`, are not closed by `grompp` smoke.",
                "",
                "Summary:",
                "",
                f"- selected scope: `{summary['selected_scope']}`",
                f"- selected rows: `{summary['totals']['selected_row_count']}`",
                f"- pass: `{summary['totals']['pass_count']}`",
                f"- failure: `{summary['totals']['failure_count']}`",
                f"- missing LUNAR PCFF data: `{summary['totals']['missing_lunar_pcff_data_count']}`",
                f"- database-wide claim: `{summary['claim_evaluation']['database_wide_converter_success_status']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    csv_path = args.csv.resolve()
    batch_root = args.lammps_batch_root.resolve()
    out_root = args.out_root.resolve()
    gmx = resolve_gmx(args.gmx)
    rows = read_csv_rows(csv_path)
    selected_rows = select_rows(rows, batch_root, args)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "cases").mkdir(parents=True, exist_ok=True)

    entries = []
    for index, row in enumerate(selected_rows, start=1):
        trajectory_id = row["Trajectory ID"]
        data_path = source_data_for_trajectory(batch_root, trajectory_id)
        if data_path is None:
            stale_case_root = out_root / "cases" / f"Traj_{trajectory_id}"
            if stale_case_root.exists():
                shutil.rmtree(stale_case_root)
            entries.append(
                {
                    "trajectory_id": trajectory_id,
                    "smiles": row["SMILES"],
                    "status": "missing_lunar_pcff_data",
                    "expected_case_root": rel(batch_root / f"Traj_{trajectory_id}"),
                }
            )
            continue
        try:
            print(f"[{index}/{len(selected_rows)}] Traj_{trajectory_id}: converting {data_path}")
            entries.append(run_case(row, data_path, out_root, gmx))
        except Exception as exc:
            entries.append(
                {
                    "trajectory_id": trajectory_id,
                    "smiles": row["SMILES"],
                    "status": "failure",
                    "source_lunar_data": rel(data_path),
                    "error": repr(exc),
                }
            )

    counts = {
        "selected_row_count": len(selected_rows),
        "pass_count": sum(1 for entry in entries if entry["status"] == "pass"),
        "failure_count": sum(1 for entry in entries if entry["status"] == "failure"),
        "missing_lunar_pcff_data_count": sum(1 for entry in entries if entry["status"] == "missing_lunar_pcff_data"),
    }
    db_wide_claimable = args.scope == "csv" and counts["selected_row_count"] == len(rows) and counts["pass_count"] == len(rows)
    if counts["pass_count"] == counts["selected_row_count"] and counts["selected_row_count"] > 0:
        strongest_supported_statement = (
            "All selected existing LUNAR PCFF single-chain data artifacts pass parser -> mapping -> emission -> "
            "grompp smoke."
        )
    elif counts["pass_count"] > 0 and counts["failure_count"] == 0:
        strongest_supported_statement = (
            f"{counts['pass_count']} selected rows with existing LUNAR PCFF single-chain data pass parser -> mapping -> "
            f"emission -> grompp smoke, but {counts['missing_lunar_pcff_data_count']} selected rows are missing "
            "LUNAR PCFF data artifacts."
        )
    elif counts["pass_count"] > 0:
        strongest_supported_statement = (
            f"{counts['pass_count']} of {counts['selected_row_count']} selected LUNAR PCFF single-chain data artifacts "
            "have parser -> mapping -> emission -> grompp PASS artifacts. Full selected-scope or database-wide "
            f"converter success is not claimable because {counts['failure_count']} failures remain."
        )
    else:
        strongest_supported_statement = "The selected scope does not have a passing parser -> mapping -> emission -> grompp path."
    summary = {
        "schema_name": "lunar_gromacs_pcff_database_lunar_smoke_summary",
        "schema_version": 1,
        "claim_status_as_of": date.today().isoformat(),
        "selected_scope": args.scope,
        "source_csv": {"path": rel(csv_path), "sha256": sha256_file(csv_path), "row_count": len(rows)},
        "batch_root": rel(batch_root),
        "gmx": str(gmx),
        "totals": counts,
        "entries": entries,
        "claim_evaluation": {
            "available_lunar_data_success_status": "pass" if counts["pass_count"] and counts["failure_count"] == 0 else "not_pass",
            "database_wide_converter_success_status": "claimable" if db_wide_claimable else "not_claimable",
            "strongest_supported_statement": strongest_supported_statement,
            "charged_ion_boundary": (
                "closed_to_claim_until_separate_Li_TFSI_parser_mapping_emission_grompp_artifacts_exist"
            ),
        },
    }
    dump_json(out_root / "database_lunar_smoke_summary.json", summary)
    write_readme(out_root, summary)
    print(
        json.dumps(
            {
                "status": summary["claim_evaluation"]["database_wide_converter_success_status"],
                "selected": counts["selected_row_count"],
                "pass": counts["pass_count"],
                "failure": counts["failure_count"],
                "missing": counts["missing_lunar_pcff_data_count"],
                "out_root": rel(out_root),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if args.allow_failures or counts["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
