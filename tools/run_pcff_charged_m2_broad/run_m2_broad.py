#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from polymer_workflow.engine import (  # noqa: E402
    _process_component,
    _write_bundle,
    load_spec,
    render_gromacs_bundle,
    run_file,
)
from polymer_workflow.rules import (  # noqa: E402
    build_nonbonded_ruleset,
    build_parameter_ruleset,
    build_typing_ruleset,
)


DEFAULT_OUT = REPO_ROOT / "tests/reference_results/pcff_charged_expansion/m2_broad"
DEFAULT_M5_SPEC = REPO_ROOT / "testdata/polymer_workflow_m5/cases/monoglyme_ethane_litfsi_1to1/spec.json"
DEFAULT_GATE_H_FIXTURE = (
    REPO_ROOT
    / "tests/reference_results/gate_h_fixture_scaffold/gate_h_dense_salt_polymer_2x2x2/fixture_manifest.json"
)

KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1


@dataclass(frozen=True)
class AtomType:
    name: str
    mass: float
    sigma_nm: float
    epsilon_kj_mol: float


@dataclass(frozen=True)
class AtomRecord:
    nr: int
    atom_type: str
    residue: str
    atom_name: str
    charge: float
    mass: float


@dataclass(frozen=True)
class MoleculeRecord:
    name: str
    residue: str
    count: int
    itp_path: Path
    ir: dict
    atoms: list[AtomRecord]
    bonds: list[tuple[int, int, tuple[float, ...]]]
    angles: list[tuple[int, int, int, tuple[float, ...]]]
    dihedrals: list[tuple[int, int, int, int, tuple[float, ...]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the broader M2 dense charged parity campaign.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gmx", type=Path, default=REPO_ROOT / "build/bin/gmx")
    parser.add_argument("--m5-spec", type=Path, default=DEFAULT_M5_SPEC)
    parser.add_argument("--gate-h-fixture", type=Path, default=DEFAULT_GATE_H_FIXTURE)
    parser.add_argument("--npt-ps", type=float, default=100.0)
    parser.add_argument("--analysis-window-ps", type=float, default=50.0)
    parser.add_argument("--density-threshold", type=float, default=0.05)
    parser.add_argument("--volume-threshold", type=float, default=0.05)
    parser.add_argument("--warmup-ps", type=float, default=5.0)
    parser.add_argument("--warmup-scope", choices=["gmx-only", "paired"], default="gmx-only")
    parser.add_argument("--m5-formula-count", type=int, default=18)
    parser.add_argument("--m5-box-nm", type=float, default=2.4)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--gmx-integrator", choices=["md", "md-vv"], default="md-vv")
    parser.add_argument("--gmx-tcoupl", choices=["v-rescale", "nose-hoover"], default="nose-hoover")
    parser.add_argument("--gmx-pcoupl", choices=["berendsen", "c-rescale", "parrinello-rahman", "mttk"], default="mttk")
    parser.add_argument("--thermal-start", choices=["generated", "fixture"], default="generated")
    parser.add_argument("--tau-t-ps", type=float, default=0.1)
    parser.add_argument("--tau-p-ps", type=float, default=1.0)
    parser.add_argument("--ref-p-bar", type=float, default=1.0)
    parser.add_argument("--compressibility-bar-inv", type=float, default=4.5e-5)
    parser.add_argument("--lmp-target-barostat", choices=["npt", "berendsen"], default="npt")
    parser.add_argument("--lmp-neighbor-skin-angstrom", type=float)
    parser.add_argument("--lmp-neighbor-every", type=int)
    parser.add_argument("--gmx-threads", type=int, default=8)
    parser.add_argument("--lmp-ranks", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="Run the formal NPT gates after freezing protocol.")
    parser.add_argument("--systems", nargs="*", help="Optional system subset for execution.")
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(cmd: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        errors="replace",
        env={**os.environ, "GMX_MAXBACKUP": "-1"},
    )
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)
    return result.returncode


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_itp(path: Path) -> dict[str, list]:
    sections: dict[str, list[list[str]]] = {}
    section: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip()
            sections.setdefault(section, [])
            continue
        if section is not None:
            sections.setdefault(section, []).append(line.split())
    return sections


def parse_atomtypes(path: Path) -> dict[str, AtomType]:
    sections = parse_itp(path)
    atomtypes = {}
    for parts in sections.get("atomtypes", []):
        require(len(parts) >= 6, f"Malformed atomtypes row in {path}: {' '.join(parts)}")
        atomtypes[parts[0]] = AtomType(parts[0], float(parts[1]), float(parts[4]), float(parts[5]))
    require(atomtypes, f"No atomtypes parsed from {path}")
    return atomtypes


def parse_molecule_itp(path: Path) -> tuple[str, list[AtomRecord], list, list, list]:
    sections = parse_itp(path)
    moleculetype = sections.get("moleculetype", [])
    require(moleculetype and len(moleculetype[0]) >= 1, f"No moleculetype parsed from {path}")
    name = moleculetype[0][0]
    atoms = [
        AtomRecord(
            nr=int(parts[0]),
            atom_type=parts[1],
            residue=parts[3],
            atom_name=parts[4],
            charge=float(parts[6]),
            mass=float(parts[7]),
        )
        for parts in sections.get("atoms", [])
    ]
    bonds = [
        (int(parts[0]), int(parts[1]), tuple(float(value) for value in parts[3:7]))
        for parts in sections.get("bonds", [])
    ]
    angles = [
        (int(parts[0]), int(parts[1]), int(parts[2]), tuple(float(value) for value in parts[4:15]))
        for parts in sections.get("angles", [])
    ]
    dihedrals = []
    for parts in sections.get("dihedrals", []):
        funct = int(parts[4])
        require(funct == 13, f"M2 broad fixture generator only supports proper class2 dihedrals, got funct={funct} in {path}")
        dihedrals.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), tuple(float(value) for value in parts[5:37])))
    require(atoms, f"No atoms parsed from {path}")
    return name, atoms, bonds, angles, dihedrals


def reverse_bond_params(params: tuple[float, ...]) -> tuple[float, ...]:
    return (params[0] * 10.0, params[1] / 418.4, params[2] / 4184.0, params[3] / 41840.0)


def reverse_angle_params(params: tuple[float, ...]) -> tuple[float, ...]:
    return (
        params[0],
        params[1] / KCAL_TO_KJ,
        params[2] / KCAL_TO_KJ,
        params[3] / KCAL_TO_KJ,
        params[4] / 418.4,
        params[5] * 10.0,
        params[6] * 10.0,
        params[7] / 41.84,
        params[8] / 41.84,
        params[9] * 10.0,
        params[10] * 10.0,
    )


def reverse_dihedral_params(params: tuple[float, ...]) -> tuple[float, ...]:
    return (
        params[0] / KCAL_TO_KJ,
        params[1],
        params[2] / KCAL_TO_KJ,
        params[3],
        params[4] / KCAL_TO_KJ,
        params[5],
        params[6] / 41.84,
        params[7] / 41.84,
        params[8] / 41.84,
        params[9] * 10.0,
        params[10] / 41.84,
        params[11] / 41.84,
        params[12] / 41.84,
        params[13] / 41.84,
        params[14] / 41.84,
        params[15] / 41.84,
        params[16] * 10.0,
        params[17] * 10.0,
        params[18] / KCAL_TO_KJ,
        params[19] / KCAL_TO_KJ,
        params[20] / KCAL_TO_KJ,
        params[21] / KCAL_TO_KJ,
        params[22] / KCAL_TO_KJ,
        params[23] / KCAL_TO_KJ,
        params[24],
        params[25],
        params[26] / KCAL_TO_KJ,
        params[27],
        params[28],
        params[29] / 418.4,
        params[30] * 10.0,
        params[31] * 10.0,
    )


def key_params(params: Iterable[float]) -> tuple[str, ...]:
    return tuple(f"{value:.10g}" for value in params)


def format_coeff(value: float) -> str:
    return f"{value:.10g}"


def process_m5_components(spec_path: Path, generated_dir: Path, formula_count: int) -> tuple[dict, list[MoleculeRecord]]:
    spec = load_spec(spec_path)
    typing_ruleset = build_typing_ruleset()
    parameter_ruleset = build_parameter_ruleset()
    nonbonded_ruleset = build_nonbonded_ruleset()
    processed = []
    for component_spec in spec["components"]:
        item = _process_component(
            component_spec,
            base_dir=spec_path.parent,
            typing_ruleset=typing_ruleset,
            parameter_ruleset=parameter_ruleset,
            nonbonded_ruleset=nonbonded_ruleset,
        )
        processed.append(item)
    bundle = render_gromacs_bundle(spec, processed)
    _write_bundle(generated_dir, bundle)
    workflow_report = run_file(spec_path, out_dir=generated_dir, dry_run=True, validate_existing=True)
    write_json(generated_dir / "polymer_workflow_validate_report.json", workflow_report)

    molecules = []
    for item in processed:
        if not item["exportable"]:
            continue
        itp_path = generated_dir / item["output_filename"]
        name, atoms, bonds, angles, dihedrals = parse_molecule_itp(itp_path)
        molecules.append(
            MoleculeRecord(
                name=name,
                residue=item["residue_name"],
                count=formula_count,
                itp_path=itp_path,
                ir=item["ir"],
                atoms=atoms,
                bonds=bonds,
                angles=angles,
                dihedrals=dihedrals,
            )
        )
    return spec, molecules


def write_single_molecule_gro(molecule: MoleculeRecord, path: Path) -> None:
    atoms_by_index = {atom["canonical_index"]: atom for atom in molecule.ir["components"][0]["atoms"]}
    coords = [atoms_by_index[idx]["coordinates"] for idx in sorted(atoms_by_index)]
    center = [sum(coord[axis] for coord in coords) / len(coords) for axis in range(3)]
    lines = [f"{molecule.name} M2 insertion molecule", f"{len(molecule.atoms):5d}"]
    for atom in molecule.atoms:
        coord_a = atoms_by_index[atom.nr]["coordinates"]
        coord_nm = [(coord_a[axis] - center[axis]) * ANGSTROM_TO_NM for axis in range(3)]
        lines.append(
            f"{1:5d}{molecule.residue[:5]:<5s}{atom.atom_name[:5]:>5s}{atom.nr % 100000:5d}"
            f"{coord_nm[0]:8.3f}{coord_nm[1]:8.3f}{coord_nm[2]:8.3f}"
        )
    lines.append(f"{2.00000:10.5f}{2.00000:10.5f}{2.00000:10.5f}")
    write_text(path, "\n".join(lines) + "\n")


def build_m5_coordinates(args: argparse.Namespace, work_dir: Path, molecules: list[MoleculeRecord]) -> Path:
    insert_dir = work_dir / "insert_molecules"
    if insert_dir.exists():
        shutil.rmtree(insert_dir)
    insert_dir.mkdir(parents=True)
    previous: Path | None = None
    for index, molecule in enumerate(molecules, start=1):
        ci = insert_dir / f"{molecule.name.lower()}.gro"
        write_single_molecule_gro(molecule, ci)
        output = insert_dir / f"packed_{index:02d}_{molecule.name.lower()}.gro"
        cmd = [
            str(args.gmx.resolve()),
            "insert-molecules",
            "-ci",
            ci.name,
            "-nmol",
            str(molecule.count),
            "-o",
            output.name,
            "-try",
            "10000",
            "-seed",
            str(args.seed + index),
            "-radius",
            "0.04",
            "-scale",
            "0.35",
        ]
        if previous is None:
            cmd.extend(["-box", str(args.m5_box_nm), str(args.m5_box_nm), str(args.m5_box_nm)])
        else:
            cmd.extend(["-f", previous.name])
        rc = run_command(cmd, insert_dir, insert_dir / f"insert_{index:02d}.stdout", insert_dir / f"insert_{index:02d}.stderr")
        require(rc == 0 and output.is_file(), f"gmx insert-molecules failed for {molecule.name}; see {insert_dir}")
        previous = output
    require(previous is not None, "No M5 molecules were packed")
    return previous


def parse_gro(path: Path) -> tuple[list[tuple[str, str, int, float, float, float]], tuple[float, float, float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    natoms = int(lines[1].strip())
    atoms = []
    for raw in lines[2 : 2 + natoms]:
        atoms.append(
            (
                raw[5:10].strip(),
                raw[10:15].strip(),
                int(raw[15:20]),
                float(raw[20:28]),
                float(raw[28:36]),
                float(raw[36:44]),
            )
        )
    box = tuple(float(value) for value in lines[2 + natoms].split()[:3])
    return atoms, box  # type: ignore[return-value]


def write_topol_with_counts(generated_dir: Path, spec: dict, molecules: list[MoleculeRecord]) -> None:
    lines = [
        "; M2 broader dense-pair self-contained topol.top generated from M5 PCFF workflow components",
        "; include expansion is intentional because run_m1_m3.py fixture mode copies only system.top.",
        "",
        generated_dir.joinpath("forcefield_pcff.itp").read_text(encoding="utf-8").rstrip(),
    ]
    for molecule in molecules:
        lines.extend(["", molecule.itp_path.read_text(encoding="utf-8").rstrip()])
    lines.extend(["", "[ system ]", f"{spec['system_id']}_dense{molecules[0].count}", "", "[ molecules ]", "; Name number"])
    for molecule in molecules:
        lines.append(f"{molecule.name} {molecule.count}")
    write_text(generated_dir / "topol.top", "\n".join(lines) + "\n")


def write_lammps_system(generated_dir: Path, packed_gro: Path, molecules: list[MoleculeRecord], atomtypes: dict[str, AtomType]) -> None:
    gro_atoms, box_nm = parse_gro(packed_gro)
    expected = sum(molecule.count * len(molecule.atoms) for molecule in molecules)
    require(len(gro_atoms) == expected, f"Packed GRO atom count {len(gro_atoms)} != expected {expected}")

    type_ids = {name: idx + 1 for idx, name in enumerate(sorted({atom.atom_type for mol in molecules for atom in mol.atoms}))}
    bond_type_ids: dict[tuple[str, ...], int] = {}
    angle_type_ids: dict[tuple[str, ...], int] = {}
    dihedral_type_ids: dict[tuple[str, ...], int] = {}
    atoms_rows: list[str] = []
    bonds_rows: list[str] = []
    angles_rows: list[str] = []
    dihedrals_rows: list[str] = []
    bond_coeffs: dict[int, tuple[float, ...]] = {}
    angle_coeffs: dict[int, tuple[float, ...]] = {}
    dihedral_coeffs: dict[int, tuple[float, ...]] = {}

    atom_id = 1
    molecule_id = 1
    gro_index = 0
    bond_id = 1
    angle_id = 1
    dihedral_id = 1
    for molecule in molecules:
        local_count = len(molecule.atoms)
        for _ in range(molecule.count):
            offset = atom_id - 1
            for local_atom in molecule.atoms:
                _, _, _, x_nm, y_nm, z_nm = gro_atoms[gro_index]
                atoms_rows.append(
                    f"{atom_id} {molecule_id} {type_ids[local_atom.atom_type]} {local_atom.charge:.8f} "
                    f"{x_nm * 10.0:.8f} {y_nm * 10.0:.8f} {z_nm * 10.0:.8f} 0 0 0"
                )
                atom_id += 1
                gro_index += 1
            for a, b, params in molecule.bonds:
                lmp_params = reverse_bond_params(params)
                key = key_params(lmp_params)
                if key not in bond_type_ids:
                    bond_type_ids[key] = len(bond_type_ids) + 1
                    bond_coeffs[bond_type_ids[key]] = lmp_params
                bonds_rows.append(f"{bond_id} {bond_type_ids[key]} {offset + a} {offset + b}")
                bond_id += 1
            for a, b, c, params in molecule.angles:
                lmp_params = reverse_angle_params(params)
                key = key_params(lmp_params)
                if key not in angle_type_ids:
                    angle_type_ids[key] = len(angle_type_ids) + 1
                    angle_coeffs[angle_type_ids[key]] = lmp_params
                angles_rows.append(f"{angle_id} {angle_type_ids[key]} {offset + a} {offset + b} {offset + c}")
                angle_id += 1
            for a, b, c, d, params in molecule.dihedrals:
                lmp_params = reverse_dihedral_params(params)
                key = key_params(lmp_params)
                if key not in dihedral_type_ids:
                    dihedral_type_ids[key] = len(dihedral_type_ids) + 1
                    dihedral_coeffs[dihedral_type_ids[key]] = lmp_params
                dihedrals_rows.append(f"{dihedral_id} {dihedral_type_ids[key]} {offset + a} {offset + b} {offset + c} {offset + d}")
                dihedral_id += 1
            molecule_id += 1
            require((atom_id - 1 - offset) == local_count, "Internal atom offset mismatch")

    box_a = [value * 10.0 for value in box_nm]
    data_lines = [
        "LAMMPS data file for M2 broader M5 dense pair",
        "",
        f"{len(atoms_rows)} atoms",
        f"{len(bonds_rows)} bonds",
        f"{len(angles_rows)} angles",
        f"{len(dihedrals_rows)} dihedrals",
        "",
        f"{len(type_ids)} atom types",
        f"{len(bond_coeffs)} bond types",
        f"{len(angle_coeffs)} angle types",
        f"{len(dihedral_coeffs)} dihedral types",
        "",
        f"0.0 {box_a[0]:.8f} xlo xhi",
        f"0.0 {box_a[1]:.8f} ylo yhi",
        f"0.0 {box_a[2]:.8f} zlo zhi",
        "",
        "Masses",
        "",
    ]
    for name, type_id in sorted(type_ids.items(), key=lambda item: item[1]):
        data_lines.append(f"{type_id} {atomtypes[name].mass:.8f}")
    data_lines.extend(["", "Atoms # full", "", *atoms_rows, "", "Bonds", "", *bonds_rows, "", "Angles", "", *angles_rows, "", "Dihedrals", "", *dihedrals_rows, ""])
    write_text(generated_dir / "system.data", "\n".join(data_lines))

    in_lines = [
        "units real",
        "atom_style full",
        "boundary p p p",
        "",
        "pair_style lj/class2/coul/long 9.0 9.0",
        "pair_modify mix sixthpower",
        "bond_style class2",
        "angle_style class2",
        "dihedral_style class2",
        "improper_style none",
        "kspace_style pppm 1.0e-4",
        "special_bonds lj/coul 0.0 0.0 1.0 angle no dihedral no",
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check no",
        "",
        "read_data system.data",
        "",
    ]
    for name, type_id in sorted(type_ids.items(), key=lambda item: item[1]):
        atomtype = atomtypes[name]
        in_lines.append(f"pair_coeff {type_id} {type_id} {atomtype.epsilon_kj_mol / KCAL_TO_KJ:.10g} {atomtype.sigma_nm * 10.0:.10g}")
    for type_id, params in sorted(bond_coeffs.items()):
        in_lines.append(f"bond_coeff {type_id} " + " ".join(format_coeff(value) for value in params))
    for type_id, params in sorted(angle_coeffs.items()):
        main = params[:4]
        bb = params[4:7]
        ba = params[7:11]
        in_lines.append(f"angle_coeff {type_id} " + " ".join(format_coeff(value) for value in main))
        in_lines.append(f"angle_coeff {type_id} bb " + " ".join(format_coeff(value) for value in bb))
        in_lines.append(f"angle_coeff {type_id} ba " + " ".join(format_coeff(value) for value in ba))
    for type_id, params in sorted(dihedral_coeffs.items()):
        in_lines.append(f"dihedral_coeff {type_id} " + " ".join(format_coeff(value) for value in params[:6]))
        in_lines.append(f"dihedral_coeff {type_id} mbt " + " ".join(format_coeff(value) for value in params[6:10]))
        in_lines.append(f"dihedral_coeff {type_id} ebt " + " ".join(format_coeff(value) for value in params[10:18]))
        in_lines.append(f"dihedral_coeff {type_id} at " + " ".join(format_coeff(value) for value in params[18:26]))
        in_lines.append(f"dihedral_coeff {type_id} aat " + " ".join(format_coeff(value) for value in params[26:29]))
        in_lines.append(f"dihedral_coeff {type_id} bb13 " + " ".join(format_coeff(value) for value in params[29:32]))
    write_text(generated_dir / "system.in", "\n".join(in_lines) + "\n")


def prepare_m5_dense_fixture(args: argparse.Namespace, out_root: Path) -> Path:
    system_id = f"monoglyme_ethane_litfsi_1to1_dense{args.m5_formula_count}"
    fixture_root = out_root / "fixtures" / system_id
    generated_dir = fixture_root / "generated"
    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    generated_dir.mkdir(parents=True)
    spec, molecules = process_m5_components(args.m5_spec.resolve(), generated_dir, args.m5_formula_count)
    write_topol_with_counts(generated_dir, spec, molecules)
    packed_gro = build_m5_coordinates(args, fixture_root, molecules)
    shutil.copy(packed_gro, generated_dir / "system.gro")
    atomtypes = parse_atomtypes(generated_dir / "forcefield_pcff.itp")
    write_lammps_system(generated_dir, generated_dir / "system.gro", molecules, atomtypes)
    typed_system = {
        "schema_name": "m2_broad_generated_dense_pair",
        "schema_version": 1,
        "system_id": system_id,
        "source_spec": repo_rel(args.m5_spec.resolve()),
        "formula_count": args.m5_formula_count,
        "box_nm": [args.m5_box_nm, args.m5_box_nm, args.m5_box_nm],
        "components": [
            {"molecule_name": molecule.name, "count": molecule.count, "atom_count_per_molecule": len(molecule.atoms)}
            for molecule in molecules
        ],
    }
    system_json = {
        "schema_version": 1,
        "id": system_id,
        "display_name": "monoglyme ethane LiTFSI dense charged M2 scaffold",
        "description": "M2 broader dense charged scaffold generated from the strict PCFF M5 workflow components.",
        "category": "polymer_box",
        "reference_terms": ["bond/class2", "angle/class2", "dihedral/class2", "lj/class2/coul/long", "special_bonds", "kspace/pppm"],
        "derived_from": {
            "source_system": "monoglyme_ethane_litfsi_1to1",
            "source_spec": str(args.m5_spec.resolve()),
            "formula_count": args.m5_formula_count,
            "packing": "gmx insert-molecules deterministic seeded pack; LAMMPS data converted from the same coordinates",
        },
        "styles": {
            "units": "real",
            "atom_style": "full",
            "pair_style": "lj/class2/coul/long",
            "pair_modify": "mix sixthpower",
            "bond_style": "class2",
            "angle_style": "class2",
            "dihedral_style": "class2",
            "improper_style": "none",
            "kspace_style": "pppm 1.0e-4",
            "special_bonds": "lj/coul 0.0 0.0 1.0 angle no dihedral no",
        },
    }
    write_json(generated_dir / "typed_system.json", typed_system)
    write_json(generated_dir / "system.json", system_json)
    manifest = {
        "schema_name": "m2_broad_strict_pair_fixture_manifest",
        "schema_version": 1,
        "derived_system": system_id,
        "seed_system": "monoglyme_ethane_litfsi_1to1",
        "formula_count": args.m5_formula_count,
        "box_nm": [args.m5_box_nm, args.m5_box_nm, args.m5_box_nm],
        "natoms": sum(molecule.count * len(molecule.atoms) for molecule in molecules),
        "pair_status": "strict_pcff_qualified",
        "acpype_gaff2_dependency": False,
        "same_pcff_source": True,
        "artifacts": {
            "typed_system": str((generated_dir / "typed_system.json").resolve()),
            "topology": str((generated_dir / "topol.top").resolve()),
            "gro": str((generated_dir / "system.gro").resolve()),
            "system_data": str((generated_dir / "system.data").resolve()),
            "system_in": str((generated_dir / "system.in").resolve()),
            "system_json": str((generated_dir / "system.json").resolve()),
        },
        "strict_basis": [
            "GROMACS topology is generated by the repository PCFF polymer workflow.",
            "LAMMPS topology is generated from the same PCFF GROMACS workflow artifacts and same packed coordinates.",
            "ACPYPE/GAFF2 and non-PCFF surrogate topology paths are not used.",
        ],
    }
    write_json(fixture_root / "fixture_manifest.json", manifest)
    return fixture_root / "fixture_manifest.json"


def freeze_protocol(args: argparse.Namespace, out_root: Path, m5_manifest: Path) -> Path:
    m5_system_id = load_json(m5_manifest)["derived_system"]
    protocol = {
        "schema_name": "pcff_charged_m2_broad_protocol",
        "schema_version": 1,
        "milestone": "M2.1-M2.5",
        "predeclared_before_result_interpretation": True,
        "old_boundary": {
            "status": "narrow explicit-subset M2 PASS only",
            "system": "gate_h_dense_salt_polymer_2x2x2",
            "target_horizon_ps": 10.0,
            "analysis_window_ps": 5.0,
            "weaknesses": [
                "single system",
                "short horizon",
                "near 5 percent thresholds",
                "GROMACS-only warmup",
                "M5 smoke is not M2 dense parity evidence",
            ],
        },
        "required_systems": [
            {
                "system_id": "gate_h_dense_salt_polymer_2x2x2",
                "strict_fixture_manifest": str(args.gate_h_fixture.resolve()),
                "role": "existing dense scaffold strengthening",
            },
            {
                "system_id": m5_system_id,
                "strict_fixture_manifest": str(m5_manifest.resolve()),
                "role": "second strict paired dense chemistry promoted from M5 workflow",
            },
        ],
        "strict_qualification_rule": {
            "same_pcff_source_required": True,
            "acpype_gaff2_allowed": False,
            "gromacs_side_must_be_pcff_derived": True,
            "lammps_side_must_be_pcff_class2": True,
        },
        "protocol": {
            "ensemble": "NPT",
            "target_horizon_ps": args.npt_ps,
            "final_analysis_window_ps": args.analysis_window_ps,
            "density_relative_difference_max": args.density_threshold,
            "volume_relative_difference_max": args.volume_threshold,
            "fail_rule": "campaign PASS requires every predeclared required system to pass density and volume thresholds with final_analysis_window_ps >= 50; one failure makes campaign FAIL",
            "warmup_policy": {
                "warmup_ps": args.warmup_ps,
                "warmup_scope": args.warmup_scope,
                "claim_caveat": "GROMACS-only warmup is explicitly predeclared; if used, the result is broader dense parity evidence but not a fully symmetric-paired equilibration claim.",
            },
            "barostat_thermostat": {
                "gmx_integrator": args.gmx_integrator,
                "gmx_tcoupl": args.gmx_tcoupl,
                "gmx_pcoupl": args.gmx_pcoupl,
                "thermal_start": args.thermal_start,
                "tau_t_ps": args.tau_t_ps,
                "tau_p_ps": args.tau_p_ps,
                "ref_p_bar": args.ref_p_bar,
                "compressibility_bar_inv": args.compressibility_bar_inv,
                "lmp_target_barostat": args.lmp_target_barostat,
                "lmp_neighbor_skin_angstrom": args.lmp_neighbor_skin_angstrom,
                "lmp_neighbor_every": args.lmp_neighbor_every,
            },
            "execution_resources": {
                "gmx_threads": args.gmx_threads,
                "lmp_ranks": args.lmp_ranks,
            },
            "raw_artifact_inventory": [
                "strict paired manifest",
                "GROMACS topology and coordinates",
                "LAMMPS data and input",
                "GROMACS mdp/tpr/log/edr/gro/cpt/xvg/stdout/stderr",
                "LAMMPS input/log/final data/stdout/stderr",
                "dense_npt_parity_report.json",
                "campaign summary",
            ],
        },
        "anti_cherry_pick_rule": "No best-case reporting: all required_systems must pass, otherwise broader M2 PASS is forbidden.",
    }
    path = out_root / "m2_broad_protocol.json"
    write_json(path, protocol)
    return path


def run_formal_system(args: argparse.Namespace, out_root: Path, system_id: str, fixture_manifest: Path) -> int:
    target = out_root / "systems" / system_id
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/run_pcff_charged_capability_subset/run_m1_m3.py"),
        "--system",
        system_id,
        "--out",
        str(target),
        "--fixture-manifest",
        str(fixture_manifest),
        "--milestone-subset",
        "M2.1-M2.5 broader dense parity",
        "--warmup-ps",
        str(args.warmup_ps),
        "--warmup-scope",
        args.warmup_scope,
        "--npt-ps",
        str(args.npt_ps),
        "--analysis-window-ps",
        str(args.analysis_window_ps),
        "--skip-nvt",
        "--density-threshold",
        str(args.density_threshold),
        "--volume-threshold",
        str(args.volume_threshold),
        "--seed",
        str(args.seed),
        "--gmx-integrator",
        args.gmx_integrator,
        "--gmx-tcoupl",
        args.gmx_tcoupl,
        "--gmx-pcoupl",
        args.gmx_pcoupl,
        "--thermal-start",
        args.thermal_start,
        "--tau-t-ps",
        str(args.tau_t_ps),
        "--tau-p-ps",
        str(args.tau_p_ps),
        "--ref-p-bar",
        str(args.ref_p_bar),
        "--compressibility-bar-inv",
        str(args.compressibility_bar_inv),
        "--lmp-target-barostat",
        args.lmp_target_barostat,
        "--gmx-threads",
        str(args.gmx_threads),
        "--lmp-ranks",
        str(args.lmp_ranks),
    ]
    if args.lmp_neighbor_skin_angstrom is not None:
        cmd.extend(["--lmp-neighbor-skin-angstrom", str(args.lmp_neighbor_skin_angstrom)])
    if args.lmp_neighbor_every is not None:
        cmd.extend(["--lmp-neighbor-every", str(args.lmp_neighbor_every)])
    return run_command(cmd, out_root, out_root / f"{system_id}.stdout", out_root / f"{system_id}.stderr")


def summarize_campaign(args: argparse.Namespace, out_root: Path, protocol_path: Path, system_manifests: dict[str, Path]) -> Path:
    systems = []
    for system_id, manifest in system_manifests.items():
        system_root = out_root / "systems" / system_id
        pair_manifest = system_root / "qualified_pair_manifest.json"
        parity_report = system_root / "paired_npt/dense_npt_parity_report.json"
        entry = {
            "system_id": system_id,
            "strict_fixture_manifest": str(manifest.resolve()),
            "qualified_pair_manifest": str(pair_manifest.resolve()) if pair_manifest.exists() else None,
            "dense_npt_parity_report": str(parity_report.resolve()) if parity_report.exists() else None,
            "status": "NOT_RUN",
            "density_rel_diff": None,
            "volume_rel_diff": None,
            "analysis_window_ps": None,
            "duration_ps": None,
        }
        if parity_report.exists():
            report = load_json(parity_report)
            entry["status"] = report.get("status")
            entry["density_rel_diff"] = report.get("parity_metrics", {}).get("density_rel_diff")
            entry["volume_rel_diff"] = report.get("parity_metrics", {}).get("volume_rel_diff")
            entry["analysis_window_ps"] = report.get("protocol", {}).get("analysis_window_ps")
            entry["duration_ps"] = report.get("protocol", {}).get("duration_ps")
        systems.append(entry)

    all_run = all(system["status"] in {"PASS", "FAIL"} for system in systems)
    all_pass = all(
        system["status"] == "PASS"
        and float(system["analysis_window_ps"] or 0.0) >= 50.0
        and float(system["duration_ps"] or 0.0) >= args.npt_ps
        for system in systems
    )
    summary = {
        "schema_name": "pcff_charged_m2_broad_campaign_summary",
        "schema_version": 1,
        "status": "PASS" if all_pass else "FAIL" if all_run else "PENDING",
        "protocol": str(protocol_path.resolve()),
        "old_narrow_boundary": "single gate_h_dense_salt_polymer_2x2x2 10 ps target / 5 ps analysis result",
        "new_candidate_boundary": "broader M2 only if both predeclared strict-PCFF dense charged pairs pass 100 ps target / 50 ps final-window density and volume thresholds",
        "systems": systems,
        "anti_cherry_pick_rule_enforced": True,
        "claim_honesty": {
            "tp1_thermal_recovery_counted_as_m2": False,
            "m5_smoke_counted_as_m2": False,
            "single_system_short_horizon_counted_as_broad_m2": False,
        },
    }
    path = out_root / "m2_broad_campaign_summary.json"
    write_json(path, summary)
    return path


def write_sha_manifest(root: Path) -> Path:
    manifest_path = root / "sha256_manifest.txt"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest_path:
            rows.append(f"{sha256(path)}  {repo_rel(path)}")
    write_text(manifest_path, "\n".join(rows) + "\n")
    return manifest_path


def main() -> int:
    args = parse_args()
    require(args.npt_ps >= 100.0, "M2 broad protocol requires npt_ps >= 100")
    require(args.analysis_window_ps >= 50.0, "M2 broad protocol requires analysis_window_ps >= 50")
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    m5_manifest = prepare_m5_dense_fixture(args, out_root)
    m5_system_id = load_json(m5_manifest)["derived_system"]
    protocol_path = freeze_protocol(args, out_root, m5_manifest)
    manifests = {
        "gate_h_dense_salt_polymer_2x2x2": args.gate_h_fixture.resolve(),
        m5_system_id: m5_manifest.resolve(),
    }

    requested = set(args.systems or manifests.keys())
    unknown = requested.difference(manifests)
    require(not unknown, f"Unknown --systems entries: {sorted(unknown)}")
    if args.execute:
        for system_id, manifest in manifests.items():
            if system_id not in requested:
                continue
            rc = run_formal_system(args, out_root, system_id, manifest)
            if rc != 0:
                break

    summary_path = summarize_campaign(args, out_root, protocol_path, manifests)
    sha_path = write_sha_manifest(out_root)
    print(json.dumps({"protocol": str(protocol_path), "summary": str(summary_path), "sha256_manifest": str(sha_path)}, indent=2))
    summary = load_json(summary_path)
    return 0 if summary["status"] == "PASS" else 2 if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
