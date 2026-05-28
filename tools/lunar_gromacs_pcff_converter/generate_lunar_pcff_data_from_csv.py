from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from audit_database_scope import DEFAULT_CSV, DEFAULT_LAMMPS_BATCH_ROOT, REPO_ROOT, WORKSPACE_ROOT


DEFAULT_LUNAR_DIR = WORKSPACE_ROOT / "MY_PAPER_RELATED" / "LAMMPS_BATCH" / "extern" / "LUNAR"
DEFAULT_BATCH_UTILS_ROOT = WORKSPACE_ROOT / "MY_PAPER_RELATED" / "LAMMPS_BATCH"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LUNAR PCFF single-chain data artifacts from the aligned CSV.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_LAMMPS_BATCH_ROOT)
    parser.add_argument("--lunar-dir", type=Path, default=DEFAULT_LUNAR_DIR)
    parser.add_argument("--batch-utils-root", type=Path, default=DEFAULT_BATCH_UTILS_ROOT)
    parser.add_argument("--trajectory-id", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--pysoftk-max-attempts", type=int, default=1)
    parser.add_argument("--min-chain-heavy-atoms", type=int, default=None)
    parser.add_argument("--minimum-dp-for-heavy-atoms", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status-jsonl", type=Path, default=None)
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


def source_data_for_project(proj: Path) -> Path | None:
    preferred = proj / "build" / "lunar_pcff" / "chain_fixed_typed_nodup_IFF_nodup.data"
    if preferred.is_file():
        return preferred
    candidates = sorted((proj / "build" / "lunar_pcff").glob("*_nodup.data"))
    return candidates[0] if candidates else None


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.trajectory_id:
        requested = set(args.trajectory_id)
        selected = [row for row in rows if row["Trajectory ID"] in requested]
    else:
        selected = list(rows)
    if args.max_cases is not None:
        selected = selected[: max(0, int(args.max_cases))]
    return selected


def require_md_dependencies() -> None:
    missing = []
    for module_name in ("rdkit", "pysoftk", "openbabel"):
        try:
            __import__(module_name)
        except Exception as exc:
            missing.append(f"{module_name}: {exc!r}")
    if missing:
        raise SystemExit(
            "Missing generation dependencies. Run with the MD conda environment, for example: "
            "PATH=/home/kiket/anaconda3/envs/MD/bin:$PATH "
            "/home/kiket/anaconda3/envs/MD/bin/python "
            "tools/lunar_gromacs_pcff_converter/generate_lunar_pcff_data_from_csv.py\n"
            + "\n".join(missing)
        )


def load_batch_utils(batch_utils_root: Path) -> None:
    root = batch_utils_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # LUNAR helper subprocesses call `python3`; force that to resolve to the active environment first.
    active_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = str(active_bin) + os.pathsep + os.environ.get("PATH", "")


def parse_warning_count(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Number of WARNING\(s\).*?\n\s*(\d+) WARNING\(s\)", text, flags=re.S)
    if match:
        return int(match.group(1))
    return len(re.findall(r"\bWARNING\b", text))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def repeat_unit_heavy_atoms(monomer_smiles: str, *, placeholder: str = "Br") -> int:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(monomer_smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse monomer SMILES: {monomer_smiles}")

    placeholder_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == placeholder)
    repeat_heavy = mol.GetNumHeavyAtoms() - placeholder_count
    if repeat_heavy < 1:
        raise ValueError(f"Repeat unit has no heavy atoms after removing placeholders: {monomer_smiles}")
    return int(repeat_heavy)


def resolve_degree_of_polymerization(
    row: dict[str, str],
    monomer_smiles: str,
    args: argparse.Namespace,
    *,
    placeholder: str = "Br",
) -> dict:
    csv_dp = max(1, int(round(float(row["Degree of Polymerization"]))))
    repeat_heavy = repeat_unit_heavy_atoms(monomer_smiles, placeholder=placeholder)

    if args.minimum_dp_for_heavy_atoms:
        if args.min_chain_heavy_atoms is None or int(args.min_chain_heavy_atoms) < 1:
            raise ValueError("--minimum-dp-for-heavy-atoms requires --min-chain-heavy-atoms >= 1")
        min_heavy = int(args.min_chain_heavy_atoms)
        dp = max(1, int(math.ceil(min_heavy / repeat_heavy)))
        policy = "minimum_dp_for_min_chain_heavy_atoms"
    else:
        min_heavy = args.min_chain_heavy_atoms
        dp = csv_dp
        policy = "csv_degree_of_polymerization"

    return {
        "degree_of_polymerization": int(dp),
        "csv_degree_of_polymerization": int(csv_dp),
        "dp_selection_policy": policy,
        "min_chain_heavy_atoms": int(min_heavy) if min_heavy is not None else None,
        "repeat_unit_heavy_atoms": int(repeat_heavy),
        "estimated_chain_heavy_atoms": int(repeat_heavy * dp),
    }


def generate_case(row: dict[str, str], args: argparse.Namespace) -> dict:
    from batch_utils.batch_run_utils import psmiles_to_monomer_smiles
    from batch_utils.pysoftk_utils import build_polymer_inputs
    from batch_utils.lunar_utils import run_lunar_pipeline

    trajectory_id = row["Trajectory ID"]
    proj = args.batch_root.resolve() / f"Traj_{trajectory_id}"
    existing_data = source_data_for_project(proj)
    if existing_data is not None and not args.force:
        return {
            "trajectory_id": trajectory_id,
            "status": "skipped_existing",
            "project_dir": rel(proj),
            "lunar_pcff_data": rel(existing_data),
        }

    start = time.monotonic()
    report_path = proj / "build" / "lunar_pcff_generation_report.json"
    monomer_smiles = None
    dp_info = None
    dp = None
    molality = None
    try:
        proj.mkdir(parents=True, exist_ok=True)
        monomer_smiles = psmiles_to_monomer_smiles(row["SMILES"], placeholder="Br")
        dp_info = resolve_degree_of_polymerization(row, monomer_smiles, args, placeholder="Br")
        dp = int(dp_info["degree_of_polymerization"])
        molality = float(row["Molality"])
        ctx = build_polymer_inputs(
            proj,
            monomer_smiles=monomer_smiles,
            placeholder="Br",
            dp=dp,
            n_salt=100,
            m_min=max(0.01, molality - 0.05),
            m_max=molality + 0.05,
            m_target=molality,
            pysoftk_max_attempts=args.pysoftk_max_attempts,
        )
        lunar = run_lunar_pipeline(
            proj,
            n_chains=int(ctx["n_chains"]),
            lunar_dir=args.lunar_dir.resolve(),
            force_field="PCFF",
        )
        poly_chain_data = Path(lunar["poly_chain_data"])
        payload = {
            "schema_name": "lunar_gromacs_pcff_csv_lunar_generation_report",
            "schema_version": 1,
            "trajectory_id": trajectory_id,
            "smiles": row["SMILES"],
            "status": "generated",
            "project_dir": rel(proj),
            "monomer_smiles": monomer_smiles,
            "degree_of_polymerization": dp,
            "csv_degree_of_polymerization": dp_info["csv_degree_of_polymerization"],
            "dp_selection_policy": dp_info["dp_selection_policy"],
            "min_chain_heavy_atoms": dp_info["min_chain_heavy_atoms"],
            "repeat_unit_heavy_atoms": dp_info["repeat_unit_heavy_atoms"],
            "estimated_chain_heavy_atoms": dp_info["estimated_chain_heavy_atoms"],
            "molality": molality,
            "n_chains": int(ctx["n_chains"]),
            "lunar_pcff_data": rel(poly_chain_data),
            "lunar_atom_typing_warning_count": parse_warning_count(proj / "logs" / "lunar_atom_typing.log"),
            "lunar_all2lmp_warning_count": parse_warning_count(proj / "logs" / "lunar_all2lmp.log"),
            "elapsed_seconds": round(time.monotonic() - start, 3),
        }
    except Exception as exc:
        payload = {
            "schema_name": "lunar_gromacs_pcff_csv_lunar_generation_report",
            "schema_version": 1,
            "trajectory_id": trajectory_id,
            "smiles": row["SMILES"],
            "status": "failure",
            "project_dir": rel(proj),
            "error": repr(exc),
            "elapsed_seconds": round(time.monotonic() - start, 3),
        }
        if monomer_smiles is not None:
            payload["monomer_smiles"] = monomer_smiles
        if dp is not None:
            payload["degree_of_polymerization"] = int(dp)
        if molality is not None:
            payload["molality"] = float(molality)
        if dp_info is not None:
            payload.update(
                {
                    "csv_degree_of_polymerization": dp_info["csv_degree_of_polymerization"],
                    "dp_selection_policy": dp_info["dp_selection_policy"],
                    "min_chain_heavy_atoms": dp_info["min_chain_heavy_atoms"],
                    "repeat_unit_heavy_atoms": dp_info["repeat_unit_heavy_atoms"],
                    "estimated_chain_heavy_atoms": dp_info["estimated_chain_heavy_atoms"],
                }
            )
    write_json(report_path, payload)
    return payload


def main() -> int:
    args = parse_args()
    load_batch_utils(args.batch_utils_root)
    require_md_dependencies()

    rows = select_rows(read_csv_rows(args.csv.resolve()), args)
    statuses = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] Traj_{row['Trajectory ID']} generate LUNAR PCFF data")
        status = generate_case(row, args)
        statuses.append(status)
        append_jsonl(args.status_jsonl, status)
        print(json.dumps({key: status.get(key) for key in ("trajectory_id", "status", "elapsed_seconds")}, sort_keys=True))

    summary = {
        "selected": len(rows),
        "generated": sum(1 for item in statuses if item["status"] == "generated"),
        "skipped_existing": sum(1 for item in statuses if item["status"] == "skipped_existing"),
        "failure": sum(1 for item in statuses if item["status"] == "failure"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if args.allow_failures or summary["failure"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
