#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from polymer_workflow import run_file  # noqa: E402


DEFAULT_SPEC = REPO_ROOT / "testdata/polymer_workflow_m5/cases/monoglyme_ethane_litfsi_1to1/spec.json"
DEFAULT_OUT = (
    REPO_ROOT
    / "tests/reference_results/pcff_charged_expansion/m5_monoglyme_ethane_litfsi_1to1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M5 chemistry-scope expansion validation.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gmx", type=Path, default=REPO_ROOT / "build/bin/gmx")
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(
    cmd: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    stdin_text: str | None = None,
) -> int:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        input=stdin_text,
        text=True,
        capture_output=True,
        errors="replace",
        env={**os.environ, "GMX_MAXBACKUP": "-1"},
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return result.returncode


def parse_itp_atoms(path: Path) -> list[dict[str, object]]:
    atoms: list[dict[str, object]] = []
    section: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip()
            continue
        if section != "atoms":
            continue
        parts = line.split()
        if len(parts) >= 8 and parts[0].isdigit():
            atoms.append(
                {
                    "nr": int(parts[0]),
                    "type": parts[1],
                    "residue": parts[3],
                    "atom": parts[4],
                    "charge": float(parts[6]),
                    "mass": float(parts[7]),
                }
            )
    if not atoms:
        raise ValueError(f"No [ atoms ] records parsed from {path}")
    return atoms


def component_output_filename(component_id: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in component_id.lower())
    return f"molecule_{sanitized}.itp"


def write_smoke_gro(spec: dict, generated_dir: Path, gro_path: Path) -> dict:
    lines = ["M5 monoglyme ethane LiTFSI smoke coordinates"]
    atom_records: list[str] = []
    serial = 1
    residue_index = 1
    molecule_index = 0
    component_summaries = []
    for component in spec["components"]:
        if component["workflow_kind"] not in {"capped_oligomer", "neutral_additive", "salt_species"}:
            continue
        itp_path = generated_dir / component_output_filename(component["component_id"])
        atoms = parse_itp_atoms(itp_path)
        count = int(component["count"])
        component_summaries.append(
            {
                "component_id": component["component_id"],
                "role": component["role"],
                "workflow_kind": component["workflow_kind"],
                "molecule_name": component.get("molecule_name", component["component_id"]),
                "count": count,
                "atom_count_per_molecule": len(atoms),
            }
        )
        for _ in range(count):
            molecule_index += 1
            base_x = 0.35 + 0.70 * ((molecule_index - 1) % 5)
            base_y = 0.35 + 0.70 * (((molecule_index - 1) // 5) % 5)
            base_z = 0.35 + 0.70 * (((molecule_index - 1) // 25) % 5)
            for local_index, atom in enumerate(atoms):
                x = base_x + 0.075 * (local_index % 4)
                y = base_y + 0.075 * ((local_index // 4) % 4)
                z = base_z + 0.075 * ((local_index // 16) % 4)
                residue = str(atom["residue"])[:5]
                atom_name = str(atom["atom"])[:5]
                atom_records.append(
                    f"{residue_index % 100000:5d}{residue:<5s}{atom_name:>5s}{serial % 100000:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
                )
                serial += 1
            residue_index += 1
    lines.append(f"{len(atom_records):5d}")
    lines.extend(atom_records)
    lines.append(f"{5.00000:10.5f}{5.00000:10.5f}{5.00000:10.5f}")
    write_text(gro_path, "\n".join(lines) + "\n")
    return {
        "atom_count": len(atom_records),
        "box_nm": [5.0, 5.0, 5.0],
        "components": component_summaries,
    }


def smoke_mdp() -> str:
    return "\n".join(
        [
            "integrator = md",
            "nsteps = 0",
            "cutoff-scheme = Verlet",
            "nstlist = 10",
            "verlet-buffer-tolerance = -1",
            "rlist = 0.9",
            "coulombtype = PME",
            "coulomb-modifier = none",
            "rcoulomb = 0.9",
            "pme-order = 4",
            "fourierspacing = 0.12",
            "ewald-rtol = 1e-4",
            "vdw-type = Cut-off",
            "vdw-modifier = none",
            "rvdw = 0.9",
            "DispCorr = no",
            "pbc = xyz",
            "",
        ]
    )


def count_gromacs_warnings(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if "WARNING" in line)


def write_sha256_manifest(root: Path, manifest_path: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {repo_rel(path)}")
    write_text(manifest_path, "\n".join(rows) + "\n")


def main() -> int:
    args = parse_args()
    out_root = args.out.resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    generated_dir = out_root / "generated_gromacs"
    smoke_dir = out_root / "gromacs_smoke"
    generated_dir.mkdir(parents=True)
    smoke_dir.mkdir(parents=True)

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    dry_report = run_file(args.spec, dry_run=True)
    written_report = run_file(args.spec, out_dir=generated_dir)
    validate_report = run_file(args.spec, out_dir=generated_dir, dry_run=True, validate_existing=True)
    coord_summary = write_smoke_gro(spec, generated_dir, smoke_dir / "system.gro")
    shutil.copy(generated_dir / "topol.top", smoke_dir / "topol.top")
    for itp in generated_dir.glob("*.itp"):
        shutil.copy(itp, smoke_dir / itp.name)
    write_text(smoke_dir / "m5_smoke.mdp", smoke_mdp())

    grompp_rc = run_command(
        [
            str(args.gmx.resolve()),
            "grompp",
            "-f",
            "m5_smoke.mdp",
            "-c",
            "system.gro",
            "-p",
            "topol.top",
            "-o",
            "m5_smoke.tpr",
            "-po",
            "m5_smoke_mdout.mdp",
        ],
        smoke_dir,
        smoke_dir / "grompp.stdout",
        smoke_dir / "grompp.stderr",
    )
    mdrun_rc = -1
    grompp_warning_count = count_gromacs_warnings(smoke_dir / "grompp.stderr")
    if grompp_rc == 0:
        mdrun_rc = run_command(
            [
                str(args.gmx.resolve()),
                "mdrun",
                "-s",
                "m5_smoke.tpr",
                "-rerun",
                "system.gro",
                "-deffnm",
                "m5_smoke",
                "-nt",
                "1",
                "-pin",
                "off",
                "-reprod",
            ],
            smoke_dir,
            smoke_dir / "mdrun.stdout",
            smoke_dir / "mdrun.stderr",
        )

    exportable = [component for component in dry_report["components"] if component["exportable"]]
    family_counts = {}
    for component in exportable:
        family = component["classification_family"]
        family_counts[family] = family_counts.get(family, 0) + int(component["count"])
    neutral_additives = [
        component
        for component in exportable
        if component["role"] == "neutral_additive"
    ]
    status = (
        dry_report["assembly_checks"]["charge_neutrality"]["status"] == "pass"
        and dry_report["assembly_checks"]["salt_balance"]["status"] == "pass"
        and validate_report["workflow"]["existing_output_matches_rendered"] is True
        and bool(neutral_additives)
        and all(component["classification_family"] == "acyclic_alkane" for component in neutral_additives)
        and grompp_rc == 0
        and grompp_warning_count == 0
        and mdrun_rc == 0
    )
    report = {
        "schema_name": "pcff_charged_m5_chemistry_expansion_report",
        "schema_version": 1,
        "system_id": spec["system_id"],
        "status": "PASS" if status else "FAIL",
        "old_boundary": "PT8 glyme + Li/TFSI SPE cases, small charged fixtures, and one dense_salt_polymer-derived strict subset.",
        "new_boundary": "Adds one charged assembly containing an acyclic alkane neutral additive: monoglyme + ethane + Li/TFSI.",
        "chemistry_delta": {
            "new_component_family": "acyclic_alkane",
            "new_role": "neutral_additive",
            "new_component_id": "ETHANE",
            "charged_context": "Li/TFSI salt balance retained with monoglyme polyether fragment.",
            "not_claimed": [
                "broad alkane chemistry",
                "arbitrary neutral co-solvents",
                "dense ensemble parity",
                "charged transport readiness",
            ],
        },
        "workflow_status": {
            "dry_run": dry_report["workflow"]["status"],
            "written": written_report["workflow"]["status"],
            "validate_existing": validate_report["workflow"]["status"],
            "existing_output_matches_rendered": validate_report["workflow"]["existing_output_matches_rendered"],
        },
        "assembly_checks": dry_report["assembly_checks"],
        "family_counts": family_counts,
        "neutral_additives": neutral_additives,
        "coordinate_smoke": coord_summary,
        "gromacs_smoke": {
            "gmx_binary": str(args.gmx.resolve()),
            "grompp_returncode": grompp_rc,
            "grompp_warning_count": grompp_warning_count,
            "grompp_warning_policy": "no_maxwarn; grompp warnings fail this M5 smoke gate",
            "mdrun_returncode": mdrun_rc,
            "status": "PASS" if grompp_rc == 0 and grompp_warning_count == 0 and mdrun_rc == 0 else "FAIL",
        },
        "artifacts": {
            "spec": repo_rel(args.spec),
            "generated_gromacs": repo_rel(generated_dir),
            "gromacs_smoke": repo_rel(smoke_dir),
            "grompp_stderr": repo_rel(smoke_dir / "grompp.stderr"),
            "mdrun_stderr": repo_rel(smoke_dir / "mdrun.stderr"),
            "sha256_manifest": repo_rel(out_root / "sha256_manifest.txt"),
        },
        "claimable_statement": "M5 adds one validated acyclic-alkane neutral-additive charged assembly: monoglyme + ethane + Li/TFSI.",
        "non_claimable_statement": "M5 does not establish broad PCFF chemistry, dense charged ensemble parity, or charged transport readiness.",
    }
    write_json(out_root / "m5_chemistry_expansion_report.json", report)
    write_json(
        out_root / "m5_chemistry_scope_manifest.json",
        {
            "schema_name": "pcff_charged_m5_chemistry_scope_manifest",
            "schema_version": 1,
            "status": report["status"],
            "system_id": spec["system_id"],
            "added_scope": report["chemistry_delta"],
            "evidence_report": repo_rel(out_root / "m5_chemistry_expansion_report.json"),
        },
    )
    write_sha256_manifest(out_root, out_root / "sha256_manifest.txt")
    return 0 if status else 1


if __name__ == "__main__":
    raise SystemExit(main())
