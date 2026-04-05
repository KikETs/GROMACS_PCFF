#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.pcff_fixture_bridge.common import ANGSTROM_TO_NM, build_typed_ir, parse_lammps_data, render_gromacs_topology  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold a larger Gate H transport fixture from an in-repo LAMMPS golden system."
    )
    parser.add_argument("--seed-system", required=True, help="Seed system id under testdata/lammps_golden/systems.")
    parser.add_argument("--system-id", required=True, help="Derived scaffold system id.")
    parser.add_argument("--replicate", nargs=3, type=int, metavar=("NX", "NY", "NZ"), required=True)
    parser.add_argument("--out", required=True, help="Output root for scaffold artifacts.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def create_gro_from_lammps(lammps_data_path: Path, gro_path: Path, typed_ir: dict[str, object]) -> dict:
    data = parse_lammps_data(lammps_data_path)
    box_x = (data["box"]["x"]["hi"] - data["box"]["x"]["lo"]) * ANGSTROM_TO_NM
    box_y = (data["box"]["y"]["hi"] - data["box"]["y"]["lo"]) * ANGSTROM_TO_NM
    box_z = (data["box"]["z"]["hi"] - data["box"]["z"]["lo"]) * ANGSTROM_TO_NM
    template_by_name = {
        str(template["name"]): template for template in typed_ir["molecule_templates"]  # type: ignore[index]
    }
    template_for_molecule_id = {
        int(instance["molecule_id"]): template_by_name[str(instance["template_name"])]  # type: ignore[index]
        for instance in typed_ir["molecule_instances"]  # type: ignore[index]
    }
    local_atom_index_by_global_id: dict[int, int] = {}
    atoms_by_molecule: dict[int, list[dict[str, object]]] = {}
    for atom in data["atoms"]:
        atoms_by_molecule.setdefault(int(atom["molecule_id"]), []).append(atom)
    for molecule_id, atoms in atoms_by_molecule.items():
        for local_index, atom in enumerate(atoms, start=1):
            local_atom_index_by_global_id[int(atom["id"])] = local_index

    with gro_path.open("w", encoding="utf-8") as handle:
        handle.write("Generated from LAMMPS data\n")
        handle.write(f"{len(data['atoms']):>5d}\n")
        for atom in data["atoms"]:
            molecule_id = int(atom["molecule_id"])
            template = template_for_molecule_id[molecule_id]
            residue_name = str(template["residue_name"])
            local_index = local_atom_index_by_global_id[int(atom["id"])]
            atom_name = str(template["atoms"][local_index - 1]["atom_name"])  # type: ignore[index]
            x = atom["x_angstrom"] * ANGSTROM_TO_NM
            y = atom["y_angstrom"] * ANGSTROM_TO_NM
            z = atom["z_angstrom"] * ANGSTROM_TO_NM
            handle.write(
                f"{molecule_id:>5d}{residue_name:<5.5s}{atom_name:>5.5s}{int(atom['id']):>5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"
            )
        handle.write(f"{box_x:10.5f}{box_y:10.5f}{box_z:10.5f}\n")

    return {
        "natoms": len(data["atoms"]),
        "box_nm": [box_x, box_y, box_z],
    }


def replicate_lammps(seed_data: Path, out_data: Path, nx: int, ny: int, nz: int) -> None:
    work_dir = out_data.parent
    shutil.copy(seed_data, work_dir / "orig.data")
    replicate_input = work_dir / "replicate.in"
    replicate_input.write_text(
        "\n".join(
            [
                "units real",
                "atom_style full",
                "read_data orig.data",
                f"replicate {nx} {ny} {nz}",
                "write_data system.data nocoeff",
                "",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(["lmp", "-in", "replicate.in"], cwd=work_dir, check=True, capture_output=True, text=True)


def main() -> int:
    args = parse_args()
    seed_system = args.seed_system
    derived_system = args.system_id
    nx, ny, nz = args.replicate

    source_root = REPO_ROOT / "testdata" / "lammps_golden" / "systems" / seed_system
    out_root = Path(args.out).resolve() / derived_system
    mini_corpus_root = out_root / "mini_corpus"
    system_root = mini_corpus_root / "systems" / derived_system
    lammps_root = system_root / "lammps"
    generated_root = out_root / "generated"

    if out_root.exists():
        shutil.rmtree(out_root)
    lammps_root.mkdir(parents=True, exist_ok=True)
    generated_root.mkdir(parents=True, exist_ok=True)

    seed_meta = load_json(source_root / "system.json")
    derived_meta = dict(seed_meta)
    derived_meta["id"] = derived_system
    derived_meta["display_name"] = f"{seed_meta['display_name']} transport scaffold"
    derived_meta["description"] = (
        f"Derived from {seed_system} via LAMMPS replicate {nx}x{ny}x{nz} for Gate H transport-scope scaffolding."
    )
    derived_meta["derived_from"] = {
        "seed_system": seed_system,
        "replicate": [nx, ny, nz],
        "source_data": str((source_root / 'lammps' / 'system.data').resolve()),
        "source_input": str((source_root / 'lammps' / 'system.in').resolve()),
    }
    dump_json(system_root / "system.json", derived_meta)

    shutil.copy(source_root / "lammps" / "system.in", lammps_root / "system.in")
    replicate_lammps(source_root / "lammps" / "system.data", lammps_root / "system.data", nx, ny, nz)

    typed_ir = build_typed_ir({"id": derived_system, "path": f"systems/{derived_system}"}, mini_corpus_root)
    dump_json(generated_root / "typed_system.json", typed_ir)
    (generated_root / "topol.top").write_text(render_gromacs_topology(typed_ir), encoding="utf-8")
    gro_summary = create_gro_from_lammps(lammps_root / "system.data", generated_root / "system.gro", typed_ir)

    meets_tp0_size = gro_summary["natoms"] >= 1000
    meets_tp0_box = min(gro_summary["box_nm"]) > 3.0
    manifest = {
        "schema_version": 1,
        "seed_system": seed_system,
        "derived_system": derived_system,
        "replicate": [nx, ny, nz],
        "natoms": gro_summary["natoms"],
        "box_nm": gro_summary["box_nm"],
        "tp0_size_fit": meets_tp0_size,
        "tp0_box_fit": meets_tp0_box,
        "artifacts": {
            "mini_corpus_root": str(mini_corpus_root),
            "system_json": str(system_root / "system.json"),
            "system_data": str(lammps_root / "system.data"),
            "system_in": str(lammps_root / "system.in"),
            "typed_system": str(generated_root / "typed_system.json"),
            "topology": str(generated_root / "topol.top"),
            "gro": str(generated_root / "system.gro"),
        },
    }
    dump_json(out_root / "fixture_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
