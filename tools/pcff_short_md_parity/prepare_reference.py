from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / "testdata" / "lammps_golden" / "systems"
M4_ROOT = REPO_ROOT / "tests" / "reference_results" / "m4"
M5_ROOT = REPO_ROOT / "tests" / "reference_results" / "m5"

ANGSTROM_TO_NM = 0.1
KCAL_TO_KJ = 4.184
BAR_TO_ATM = 0.9869232667160128
ANGSTROM_PER_FS_TO_NM_PER_PS = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen M5 PCFF short-MD parity references.")
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="System id to regenerate. Default: all M5 systems.",
    )
    parser.add_argument(
        "--out",
        default=str(M5_ROOT),
        help="Output directory for generated M5 reference artifacts.",
    )
    parser.add_argument(
        "--workdir",
        default="/tmp/pcff_short_md_parity",
        help="Temporary directory used for auxiliary LAMMPS runs.",
    )
    parser.add_argument(
        "--lammps-cmd",
        default="/home/user/.local/bin/lmp",
        help="LAMMPS executable used to stage exact NVE initial velocities.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def parse_lammps_data(path: Path) -> dict:
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    box = {}
    counts = {}

    lines = path.read_text(encoding="utf-8").splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith("xlo xhi"):
            xlo, xhi, *_ = line.split()
            box["x"] = (float(xlo), float(xhi))
            continue
        if line.endswith("ylo yhi"):
            ylo, yhi, *_ = line.split()
            box["y"] = (float(ylo), float(yhi))
            continue
        if line.endswith("zlo zhi"):
            zlo, zhi, *_ = line.split()
            box["z"] = (float(zlo), float(zhi))
            continue
        if line.endswith("atoms"):
            counts["atoms"] = int(line.split()[0])
            continue
        if line.endswith("bonds"):
            counts["bonds"] = int(line.split()[0])
            continue
        if line.endswith("angles"):
            counts["angles"] = int(line.split()[0])
            continue
        if line.endswith("dihedrals"):
            counts["dihedrals"] = int(line.split()[0])
            continue
        if line in {"Masses", "Atoms # full", "Bonds", "Angles", "Dihedrals"}:
            current_section = line
            sections[current_section] = []
            continue
        if current_section is not None:
            sections[current_section].append(line)

    masses = {}
    for line in sections.get("Masses", []):
        atom_type, mass = line.split()
        masses[int(atom_type)] = float(mass)

    atoms = []
    for line in sections.get("Atoms # full", []):
        atom_id, molecule, atom_type, charge, x, y, z = line.split()
        atoms.append(
            {
                "id": int(atom_id),
                "molecule": int(molecule),
                "type": int(atom_type),
                "charge": float(charge),
                "mass": masses[int(atom_type)],
                "x": float(x),
                "y": float(y),
                "z": float(z),
            }
        )
    atoms.sort(key=lambda atom: atom["id"])

    def parse_bonded(section: str, atom_width: int) -> list[dict]:
        result = []
        for line in sections.get(section, []):
            fields = line.split()
            result.append(
                {
                    "id": int(fields[0]),
                    "type": int(fields[1]),
                    "atoms": tuple(int(value) for value in fields[2 : 2 + atom_width]),
                }
            )
        return result

    return {
        "counts": counts,
        "box": box,
        "atoms": atoms,
        "bonds": parse_bonded("Bonds", 2),
        "angles": parse_bonded("Angles", 3),
        "dihedrals": parse_bonded("Dihedrals", 4),
    }


def parse_lammps_coefficients(path: Path) -> dict:
    coeffs = {
        "pair": {},
        "bond": None,
        "angle": {"main": None, "bb": None, "ba": None},
        "dihedral": {"main": None, "mbt": None, "ebt": None, "at": None, "aat": None, "bb13": None},
    }
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == "pair_coeff":
            coeffs["pair"][int(fields[1])] = {
                "epsilon_kcal": float(fields[3]),
                "sigma_angstrom": float(fields[4]),
            }
        elif fields[0] == "bond_coeff":
            coeffs["bond"] = {
                "r0_angstrom": float(fields[2]),
                "k2_kcal": float(fields[3]),
                "k3_kcal": float(fields[4]),
                "k4_kcal": float(fields[5]),
            }
        elif fields[0] == "angle_coeff" and fields[2] == "bb":
            coeffs["angle"]["bb"] = {
                "k_kcal_per_a2": float(fields[3]),
                "r1_angstrom": float(fields[4]),
                "r2_angstrom": float(fields[5]),
            }
        elif fields[0] == "angle_coeff" and fields[2] == "ba":
            coeffs["angle"]["ba"] = {
                "k1_kcal_per_a": float(fields[3]),
                "k2_kcal_per_a": float(fields[4]),
                "r1_angstrom": float(fields[5]),
                "r2_angstrom": float(fields[6]),
            }
        elif fields[0] == "angle_coeff" and len(fields) == 6:
            coeffs["angle"]["main"] = {
                "theta0_deg": float(fields[2]),
                "k2_kcal": float(fields[3]),
                "k3_kcal": float(fields[4]),
                "k4_kcal": float(fields[5]),
            }
        elif fields[0] == "dihedral_coeff" and len(fields) == 8:
            coeffs["dihedral"]["main"] = {
                "k1_kcal": float(fields[2]),
                "phi1_deg": float(fields[3]),
                "k2_kcal": float(fields[4]),
                "phi2_deg": float(fields[5]),
                "k3_kcal": float(fields[6]),
                "phi3_deg": float(fields[7]),
            }
        elif fields[0] == "dihedral_coeff" and fields[2] == "mbt":
            coeffs["dihedral"]["mbt"] = {
                "f1_kcal_per_a": float(fields[3]),
                "f2_kcal_per_a": float(fields[4]),
                "f3_kcal_per_a": float(fields[5]),
                "r0_angstrom": float(fields[6]),
            }
        elif fields[0] == "dihedral_coeff" and fields[2] == "ebt":
            coeffs["dihedral"]["ebt"] = {
                "f1_1_kcal_per_a": float(fields[3]),
                "f2_1_kcal_per_a": float(fields[4]),
                "f3_1_kcal_per_a": float(fields[5]),
                "f1_2_kcal_per_a": float(fields[6]),
                "f2_2_kcal_per_a": float(fields[7]),
                "f3_2_kcal_per_a": float(fields[8]),
                "r0_1_angstrom": float(fields[9]),
                "r0_2_angstrom": float(fields[10]),
            }
        elif fields[0] == "dihedral_coeff" and fields[2] == "at":
            coeffs["dihedral"]["at"] = {
                "f1_1_kcal": float(fields[3]),
                "f2_1_kcal": float(fields[4]),
                "f3_1_kcal": float(fields[5]),
                "f1_2_kcal": float(fields[6]),
                "f2_2_kcal": float(fields[7]),
                "f3_2_kcal": float(fields[8]),
                "theta0_1_deg": float(fields[9]),
                "theta0_2_deg": float(fields[10]),
            }
        elif fields[0] == "dihedral_coeff" and fields[2] == "aat":
            coeffs["dihedral"]["aat"] = {
                "k_kcal": float(fields[3]),
                "theta0_1_deg": float(fields[4]),
                "theta0_2_deg": float(fields[5]),
            }
        elif fields[0] == "dihedral_coeff" and fields[2] == "bb13":
            coeffs["dihedral"]["bb13"] = {
                "k_kcal_per_a2": float(fields[3]),
                "r10_angstrom": float(fields[4]),
                "r30_angstrom": float(fields[5]),
            }
    return coeffs


def kcal_to_kj(value: float) -> float:
    return value * KCAL_TO_KJ


def angstrom_to_nm(value: float) -> float:
    return value * ANGSTROM_TO_NM


def bond_k2_to_gromacs(value: float) -> float:
    return kcal_to_kj(value) / (ANGSTROM_TO_NM**2)


def bond_k3_to_gromacs(value: float) -> float:
    return kcal_to_kj(value) / (ANGSTROM_TO_NM**3)


def bond_k4_to_gromacs(value: float) -> float:
    return kcal_to_kj(value) / (ANGSTROM_TO_NM**4)


def bond_bond_k_to_gromacs(value: float) -> float:
    return kcal_to_kj(value) / (ANGSTROM_TO_NM**2)


def bond_angle_k_to_gromacs(value: float) -> float:
    return kcal_to_kj(value) / ANGSTROM_TO_NM


def dihedral_bond_torsion_k_to_gromacs(value: float) -> float:
    return kcal_to_kj(value) / ANGSTROM_TO_NM


def box_lengths_nm(parsed_data: dict) -> tuple[float, float, float]:
    return tuple((parsed_data["box"][axis][1] - parsed_data["box"][axis][0]) * ANGSTROM_TO_NM for axis in ("x", "y", "z"))


def box_centers_angstrom(parsed_data: dict) -> tuple[float, float, float]:
    return tuple(0.5 * (parsed_data["box"][axis][1] + parsed_data["box"][axis][0]) for axis in ("x", "y", "z"))


def shifted_coordinates_nm(atom: dict, parsed_data: dict) -> tuple[float, float, float]:
    x_center, y_center, z_center = box_centers_angstrom(parsed_data)
    x_len, y_len, z_len = box_lengths_nm(parsed_data)
    return (
        (atom["x"] - x_center) * ANGSTROM_TO_NM + 0.5 * x_len,
        (atom["y"] - y_center) * ANGSTROM_TO_NM + 0.5 * y_len,
        (atom["z"] - z_center) * ANGSTROM_TO_NM + 0.5 * z_len,
    )


def minimum_image(delta: float, box_length: float) -> float:
    if box_length <= 0:
        return delta
    return delta - box_length * round(delta / box_length)


def vector_with_minimum_image(a: tuple[float, float, float], b: tuple[float, float, float], box_nm: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(minimum_image(a[d] - b[d], box_nm[d]) for d in range(3))


def norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def structural_metrics(system_id: str, atoms: list[dict], parsed_data: dict) -> dict[str, float]:
    box_nm = box_lengths_nm(parsed_data)
    atom_by_id = {atom["id"]: atom for atom in atoms}
    polymer_atoms = [atom for atom in atoms if atom["id"] <= (6 if system_id == "small_oligomer" else 8)]
    polymer_coords = [
        (
            atom["x"] * ANGSTROM_TO_NM,
            atom["y"] * ANGSTROM_TO_NM,
            atom["z"] * ANGSTROM_TO_NM,
        )
        for atom in polymer_atoms
    ]

    end_a = polymer_coords[0]
    end_b = polymer_coords[-1]
    end_to_end_vector = vector_with_minimum_image(end_b, end_a, box_nm)
    metrics = {
        "polymer_end_to_end_nm": norm(end_to_end_vector),
    }

    anchor = polymer_coords[0]
    unwrapped = [anchor]
    for coordinate in polymer_coords[1:]:
        delta = vector_with_minimum_image(coordinate, unwrapped[-1], box_nm)
        unwrapped.append(tuple(unwrapped[-1][d] + delta[d] for d in range(3)))
    center_of_mass = tuple(sum(coord[d] for coord in unwrapped) / len(unwrapped) for d in range(3))
    metrics["polymer_rg_nm"] = math.sqrt(
        sum(sum((coord[d] - center_of_mass[d]) ** 2 for d in range(3)) for coord in unwrapped) / len(unwrapped)
    )

    if system_id == "small_salt_polymer_box":
        ion_a = atom_by_id[9]
        ion_b = atom_by_id[10]
        ion_a_coord = (ion_a["x"] * ANGSTROM_TO_NM, ion_a["y"] * ANGSTROM_TO_NM, ion_a["z"] * ANGSTROM_TO_NM)
        ion_b_coord = (ion_b["x"] * ANGSTROM_TO_NM, ion_b["y"] * ANGSTROM_TO_NM, ion_b["z"] * ANGSTROM_TO_NM)
        metrics["ion_distance_nm"] = norm(vector_with_minimum_image(ion_b_coord, ion_a_coord, box_nm))

    return metrics


def generate_topology(system_id: str, parsed_data: dict, coeffs: dict) -> str:
    atom_type_names = {atom_type: f"T{atom_type}" for atom_type in sorted(coeffs["pair"])}
    molecule_ids = sorted({atom["molecule"] for atom in parsed_data["atoms"]})
    atoms_by_molecule = {
        molecule_id: [atom for atom in parsed_data["atoms"] if atom["molecule"] == molecule_id] for molecule_id in molecule_ids
    }

    bonds = parsed_data["bonds"]
    angles = parsed_data["angles"]
    dihedrals = parsed_data["dihedrals"]

    molecule_names = []
    if system_id == "small_oligomer":
        molecule_names = ["OLI"]
    else:
        molecule_names = ["POL", "CAT", "ANI"]

    lines = [
        "[ defaults ]",
        "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ rep-pow",
        "1 4 yes 1.0 1.0 9.0",
        "",
        "[ atomtypes ]",
        "; name mass charge ptype sigma epsilon",
    ]
    for atom_type in sorted(coeffs["pair"]):
        pair = coeffs["pair"][atom_type]
        mass = next(atom["mass"] for atom in parsed_data["atoms"] if atom["type"] == atom_type)
        lines.append(
            f"{atom_type_names[atom_type]:<6} {mass:8.3f} 0.0 A {angstrom_to_nm(pair['sigma_angstrom']):.8f} {kcal_to_kj(pair['epsilon_kcal']):.8f}"
        )

    for molecule_index, molecule_id in enumerate(molecule_ids):
        molecule_atoms = atoms_by_molecule[molecule_id]
        local_index = {atom["id"]: idx + 1 for idx, atom in enumerate(molecule_atoms)}
        mol_name = molecule_names[molecule_index]
        nrexcl = 3 if len(molecule_atoms) > 3 else 1
        lines.extend(
            [
                "",
                "[ moleculetype ]",
                "; Name nrexcl",
                f"{mol_name} {nrexcl}",
                "",
                "[ atoms ]",
                "; nr type resnr residue atom cgnr charge mass",
            ]
        )
        for idx, atom in enumerate(molecule_atoms, start=1):
            lines.append(
                f"{idx:>3} {atom_type_names[atom['type']]:<6} 1 {mol_name:<6} A{idx:<2} {idx:>3} {atom['charge']: .8f} {atom['mass']: .6f}"
            )

        molecule_bonds = [bond for bond in bonds if all(atom_id in local_index for atom_id in bond["atoms"])]
        if molecule_bonds:
            bond_coeff = coeffs["bond"]
            lines.extend(["", "[ bonds ]", "; ai aj funct c0 c1 c2 c3"])
            for bond in molecule_bonds:
                ai, aj = (local_index[atom_id] for atom_id in bond["atoms"])
                lines.append(
                    f"{ai:>3} {aj:>3} 11 {angstrom_to_nm(bond_coeff['r0_angstrom']):.8f} {bond_k2_to_gromacs(bond_coeff['k2_kcal']):.8f} "
                    f"{bond_k3_to_gromacs(bond_coeff['k3_kcal']):.8f} {bond_k4_to_gromacs(bond_coeff['k4_kcal']):.8f}"
                )
            one_four_pairs = generate_one_four_pairs(molecule_atoms, molecule_bonds)
            if one_four_pairs:
                lines.extend(["", "[ pairs ]", "; ai aj funct"])
                for ai, aj in one_four_pairs:
                    lines.append(f"{ai:>3} {aj:>3} 1")

        molecule_angles = [angle for angle in angles if all(atom_id in local_index for atom_id in angle["atoms"])]
        if molecule_angles:
            angle_main = coeffs["angle"]["main"]
            angle_bb = coeffs["angle"]["bb"]
            angle_ba = coeffs["angle"]["ba"]
            lines.extend(["", "[ angles ]", "; ai aj ak funct c0..c10"])
            for angle in molecule_angles:
                ai, aj, ak = (local_index[atom_id] for atom_id in angle["atoms"])
                lines.append(
                    f"{ai:>3} {aj:>3} {ak:>3} 11 {angle_main['theta0_deg']:.8f} {kcal_to_kj(angle_main['k2_kcal']):.8f} "
                    f"{kcal_to_kj(angle_main['k3_kcal']):.8f} {kcal_to_kj(angle_main['k4_kcal']):.8f} "
                    f"{bond_bond_k_to_gromacs(angle_bb['k_kcal_per_a2']):.8f} {angstrom_to_nm(angle_bb['r1_angstrom']):.8f} "
                    f"{angstrom_to_nm(angle_bb['r2_angstrom']):.8f} {bond_angle_k_to_gromacs(angle_ba['k1_kcal_per_a']):.8f} "
                    f"{bond_angle_k_to_gromacs(angle_ba['k2_kcal_per_a']):.8f} {angstrom_to_nm(angle_ba['r1_angstrom']):.8f} "
                    f"{angstrom_to_nm(angle_ba['r2_angstrom']):.8f}"
                )

        molecule_dihedrals = [dihedral for dihedral in dihedrals if all(atom_id in local_index for atom_id in dihedral["atoms"])]
        if molecule_dihedrals:
            dih_main = coeffs["dihedral"]["main"]
            dih_mbt = coeffs["dihedral"]["mbt"]
            dih_ebt = coeffs["dihedral"]["ebt"]
            dih_at = coeffs["dihedral"]["at"]
            dih_aat = coeffs["dihedral"]["aat"]
            dih_bb13 = coeffs["dihedral"]["bb13"]
            lines.extend(["", "[ dihedrals ]", "; ai aj ak al funct c0..c31"])
            for dihedral in molecule_dihedrals:
                ai, aj, ak, al = (local_index[atom_id] for atom_id in dihedral["atoms"])
                lines.append(
                    f"{ai:>3} {aj:>3} {ak:>3} {al:>3} 13 "
                    f"{kcal_to_kj(dih_main['k1_kcal']):.8f} {dih_main['phi1_deg']:.8f} "
                    f"{kcal_to_kj(dih_main['k2_kcal']):.8f} {dih_main['phi2_deg']:.8f} "
                    f"{kcal_to_kj(dih_main['k3_kcal']):.8f} {dih_main['phi3_deg']:.8f} "
                    f"{dihedral_bond_torsion_k_to_gromacs(dih_mbt['f1_kcal_per_a']):.8f} {dihedral_bond_torsion_k_to_gromacs(dih_mbt['f2_kcal_per_a']):.8f} "
                    f"{dihedral_bond_torsion_k_to_gromacs(dih_mbt['f3_kcal_per_a']):.8f} {angstrom_to_nm(dih_mbt['r0_angstrom']):.8f} "
                    f"{dihedral_bond_torsion_k_to_gromacs(dih_ebt['f1_1_kcal_per_a']):.8f} {dihedral_bond_torsion_k_to_gromacs(dih_ebt['f2_1_kcal_per_a']):.8f} "
                    f"{dihedral_bond_torsion_k_to_gromacs(dih_ebt['f3_1_kcal_per_a']):.8f} {dihedral_bond_torsion_k_to_gromacs(dih_ebt['f1_2_kcal_per_a']):.8f} "
                    f"{dihedral_bond_torsion_k_to_gromacs(dih_ebt['f2_2_kcal_per_a']):.8f} {dihedral_bond_torsion_k_to_gromacs(dih_ebt['f3_2_kcal_per_a']):.8f} "
                    f"{angstrom_to_nm(dih_ebt['r0_1_angstrom']):.8f} {angstrom_to_nm(dih_ebt['r0_2_angstrom']):.8f} "
                    f"{kcal_to_kj(dih_at['f1_1_kcal']):.8f} {kcal_to_kj(dih_at['f2_1_kcal']):.8f} {kcal_to_kj(dih_at['f3_1_kcal']):.8f} "
                    f"{kcal_to_kj(dih_at['f1_2_kcal']):.8f} {kcal_to_kj(dih_at['f2_2_kcal']):.8f} {kcal_to_kj(dih_at['f3_2_kcal']):.8f} "
                    f"{dih_at['theta0_1_deg']:.8f} {dih_at['theta0_2_deg']:.8f} {kcal_to_kj(dih_aat['k_kcal']):.8f} "
                    f"{dih_aat['theta0_1_deg']:.8f} {dih_aat['theta0_2_deg']:.8f} {bond_bond_k_to_gromacs(dih_bb13['k_kcal_per_a2']):.8f} "
                    f"{angstrom_to_nm(dih_bb13['r10_angstrom']):.8f} {angstrom_to_nm(dih_bb13['r30_angstrom']):.8f}"
                )

    lines.extend(["", "[ system ]", system_id, "", "[ molecules ]", "; Name number"])
    for molecule_index, molecule_id in enumerate(molecule_ids):
        lines.append(f"{molecule_names[molecule_index]} 1")
    lines.append("")
    return "\n".join(lines)


def generate_one_four_pairs(molecule_atoms: list[dict], molecule_bonds: list[dict]) -> list[tuple[int, int]]:
    local_index = {atom["id"]: idx + 1 for idx, atom in enumerate(molecule_atoms)}
    adjacency = {local_atom: set() for local_atom in local_index.values()}
    for bond in molecule_bonds:
        ai, aj = (local_index[atom_id] for atom_id in bond["atoms"])
        adjacency[ai].add(aj)
        adjacency[aj].add(ai)

    pairs = set()
    for atom_i in sorted(adjacency):
        visited = {atom_i}
        frontier = {atom_i}
        for depth in range(1, 4):
            next_frontier = set()
            for atom_j in frontier:
                next_frontier.update(adjacency[atom_j])
            next_frontier -= visited
            if depth == 3:
                for atom_j in next_frontier:
                    if atom_i < atom_j:
                        pairs.add((atom_i, atom_j))
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

    return sorted(pairs)


def local_atom_names(parsed_data: dict) -> dict[int, str]:
    local_names = {}
    atoms_by_molecule = {}
    for atom in parsed_data["atoms"]:
        atoms_by_molecule.setdefault(atom["molecule"], []).append(atom)
    for molecule_atoms in atoms_by_molecule.values():
        for local_idx, atom in enumerate(sorted(molecule_atoms, key=lambda atom: atom["id"]), start=1):
            local_names[atom["id"]] = f"A{local_idx}"
    return local_names


def write_gro(path: Path, title: str, atoms: list[dict], parsed_data: dict, velocities: dict[int, tuple[float, float, float]] | None) -> None:
    box_nm = box_lengths_nm(parsed_data)
    atom_names = local_atom_names(parsed_data)
    lines = [title, f"{len(atoms)}"]
    for atom in atoms:
        x_nm, y_nm, z_nm = shifted_coordinates_nm(atom, parsed_data)
        velocity_tuple = None if velocities is None else velocities.get(atom["id"])
        atom_name = atom_names[atom["id"]]
        if velocity_tuple is None:
            lines.append(
                f"{atom['molecule']:5d}{'SYS':<5}{atom_name:>5}{atom['id'] % 100000:5d}{x_nm:12.8f}{y_nm:12.8f}{z_nm:12.8f}"
            )
        else:
            vx_nm_ps, vy_nm_ps, vz_nm_ps = velocity_tuple
            lines.append(
                f"{atom['molecule']:5d}{'SYS':<5}{atom_name:>5}{atom['id'] % 100000:5d}"
                f"{x_nm:12.8f}{y_nm:12.8f}{z_nm:12.8f}{vx_nm_ps:12.8f}{vy_nm_ps:12.8f}{vz_nm_ps:12.8f}"
            )
    lines.append(f"{box_nm[0]:12.8f}{box_nm[1]:12.8f}{box_nm[2]:12.8f}")
    write_text(path, "\n".join(lines) + "\n")


def parse_dump_custom(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    frames = []
    index = 0
    while index < len(lines):
        if lines[index] != "ITEM: TIMESTEP":
            index += 1
            continue
        timestep = int(lines[index + 1].strip())
        natoms = int(lines[index + 3].strip())
        atom_fields = lines[index + 8].split()[2:]
        atoms = []
        atom_start = index + 9
        atom_end = atom_start + natoms
        for atom_line in lines[atom_start:atom_end]:
            raw_values = atom_line.split()
            atom = {}
            for field, raw_value in zip(atom_fields, raw_values):
                if field in {"id", "type"}:
                    atom[field] = int(raw_value)
                else:
                    atom[field] = float(raw_value)
            atoms.append(atom)
        atoms.sort(key=lambda atom: atom["id"])
        frames.append({"timestep": timestep, "atoms": atoms})
        index = atom_end
    return frames


def nve_initial_velocities(system_id: str, system_meta: dict, workdir: Path, lammps_cmd: str) -> dict[int, tuple[float, float, float]]:
    stage_dir = workdir / system_id
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    source_dir = CORPUS_ROOT / system_id / "lammps"
    shutil.copy2(source_dir / "system.data", stage_dir / "system.data")
    shutil.copy2(source_dir / "system.in", stage_dir / "system.in")

    config = system_meta["expected_observables"]["nve_drift"]
    input_contents = "\n".join(
        [
            "log nve_initial.log",
            "include system.in",
            "reset_timestep 0",
            f"timestep {config['timestep_fs']}",
            f"velocity all create {config['initial_temperature_K']} {config['velocity_seed']} mom yes rot yes dist gaussian",
            "write_dump all custom nve_initial.dump id type q x y z vx vy vz modify sort id",
        ]
    )
    write_text(stage_dir / "nve_initial.in", input_contents + "\n")
    subprocess.run(
        ["/bin/bash", "-lc", f"OMPI_MCA_plm=isolated {lammps_cmd} -in nve_initial.in"],
        cwd=stage_dir,
        check=True,
    )

    frames = parse_dump_custom(stage_dir / "nve_initial.dump")
    velocities = {}
    for atom in frames[-1]["atoms"]:
        velocities[atom["id"]] = (
            atom["vx"] * ANGSTROM_PER_FS_TO_NM_PER_PS,
            atom["vy"] * ANGSTROM_PER_FS_TO_NM_PER_PS,
            atom["vz"] * ANGSTROM_PER_FS_TO_NM_PER_PS,
        )
    return velocities


def tolerances_for(system_id: str) -> dict:
    common_nve = {
        "step0_potential_kcal_mol": 0.02,
        "initial_total_kcal_mol": 0.03,
        "final_total_kcal_mol": 0.06,
        "total_energy_drift_abs_kcal_mol": 0.06,
        "total_energy_span_kcal_mol": 0.08,
        "polymer_end_to_end_nm": 0.01,
        "polymer_rg_nm": 0.01,
    }
    common_nvt = {
        "step0_potential_kcal_mol": 0.02,
        "final_potential_kcal_mol": 0.25,
        "final_total_kcal_mol": 0.25,
        "final_temperature_K": 25.0,
        "final_pressure_atm": 120.0,
        "polymer_end_to_end_nm": 0.015,
        "polymer_rg_nm": 0.015,
    }
    if system_id == "small_salt_polymer_box":
        common_nve["ion_distance_nm"] = 0.02
        common_nvt["ion_distance_nm"] = 0.02
    return {"nve": common_nve, "nvt": common_nvt}


def build_reference_summary(system_id: str, parsed_data: dict, system_meta: dict) -> dict:
    single_point = load_json(M4_ROOT / system_id / "single_point.json")
    nve_json = load_json(M4_ROOT / system_id / "nve_drift.json")
    nvt_json = load_json(M4_ROOT / system_id / "nvt_snapshot.json")

    nve_final_metrics = structural_metrics(system_id, nve_json["final_frame"]["atoms"], parsed_data)
    nvt_final_metrics = structural_metrics(system_id, nvt_json["final_frame"]["atoms"], parsed_data)

    nve_trace = nve_json["trace"]
    nvt_trace = nvt_json["trace"]
    reference = {
        "schema_version": 1,
        "system_id": system_id,
        "sources": {
            "topology": f"testdata/lammps_golden/systems/{system_id}/lammps/system.data",
            "parameters": f"testdata/lammps_golden/systems/{system_id}/lammps/system.in",
            "lammps_observables": {
                "single_point": f"tests/reference_results/m4/{system_id}/single_point.json",
                "nve_drift": f"tests/reference_results/m4/{system_id}/nve_drift.json",
                "nvt_snapshot": f"tests/reference_results/m4/{system_id}/nvt_snapshot.json",
            },
        },
        "reference": {
            "single_point": {
                "potential_kcal_mol": single_point["fields"]["pe"],
            },
            "nve": {
                "step0_potential_kcal_mol": nve_trace[0]["pe"],
                "initial_total_kcal_mol": nve_trace[0]["etotal"],
                "final_total_kcal_mol": nve_trace[-1]["etotal"],
                "total_energy_drift_abs_kcal_mol": abs(nve_trace[-1]["etotal"] - nve_trace[0]["etotal"]),
                "total_energy_span_kcal_mol": max(frame["etotal"] for frame in nve_trace) - min(frame["etotal"] for frame in nve_trace),
                **nve_final_metrics,
            },
            "nvt": {
                "step0_potential_kcal_mol": nvt_trace[0]["pe"],
                "final_potential_kcal_mol": nvt_trace[-1]["pe"],
                "final_total_kcal_mol": nvt_trace[-1]["etotal"],
                "final_temperature_K": nvt_trace[-1]["temp"],
                "final_pressure_atm": nvt_trace[-1]["press"],
                **nvt_final_metrics,
            },
        },
        "tolerances": tolerances_for(system_id),
        "unresolved_items": [
            "NVT parity remains thermostat-algorithm-sensitive across engines; tolerances are defined on final observables rather than exact trajectory identity."
        ],
    }
    return reference


def reference_tsv(reference_summary: dict) -> str:
    lines = [
        "# schema_version 1",
        f"system {reference_summary['system_id']}",
    ]
    for section_name in ("single_point", "nve", "nvt"):
        for key, value in sorted(reference_summary["reference"][section_name].items()):
            lines.append(f"reference {section_name} {key} {value:.12f}")
    for section_name in ("nve", "nvt"):
        for key, value in sorted(reference_summary["tolerances"][section_name].items()):
            lines.append(f"tolerance {section_name} {key} {value:.12f}")
    return "\n".join(lines) + "\n"


def generate_system(system_id: str, out_root: Path, workdir: Path, lammps_cmd: str) -> None:
    system_root = CORPUS_ROOT / system_id
    system_meta = load_json(system_root / "system.json")
    parsed_data = parse_lammps_data(system_root / "lammps" / "system.data")
    coeffs = parse_lammps_coefficients(system_root / "lammps" / "system.in")

    nve_velocities = nve_initial_velocities(system_id, system_meta, workdir, lammps_cmd)
    output_dir = out_root / system_id
    output_dir.mkdir(parents=True, exist_ok=True)

    write_text(output_dir / "topol.top", generate_topology(system_id, parsed_data, coeffs))
    write_gro(output_dir / "initial_nve.gro", f"{system_id} nve initial state", parsed_data["atoms"], parsed_data, nve_velocities)
    zero_velocities = {atom["id"]: (0.0, 0.0, 0.0) for atom in parsed_data["atoms"]}
    write_gro(output_dir / "initial_nvt.gro", f"{system_id} nvt initial state", parsed_data["atoms"], parsed_data, zero_velocities)

    reference_summary = build_reference_summary(system_id, parsed_data, system_meta)
    dump_json(output_dir / "reference_summary.json", reference_summary)
    write_text(output_dir / "reference_summary.tsv", reference_tsv(reference_summary))


def main() -> None:
    args = parse_args()
    out_root = Path(args.out).resolve()
    workdir = Path(args.workdir).resolve()
    systems = args.systems or ["small_oligomer", "small_salt_polymer_box"]

    out_root.mkdir(parents=True, exist_ok=True)
    for system_id in systems:
        generate_system(system_id, out_root, workdir, args.lammps_cmd)


if __name__ == "__main__":
    main()
