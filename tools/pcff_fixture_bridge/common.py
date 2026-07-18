from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / "testdata" / "lammps_golden"

KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1
DEG_TO_RAD = math.pi / 180.0


class BridgeError(ValueError):
    pass


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def corpus_manifest(root: Path | None = None) -> dict:
    base = CORPUS_ROOT if root is None else root
    return load_json(base / "corpus_manifest.json")


def iter_system_records(system_ids: Iterable[str] | None = None, root: Path | None = None) -> list[dict]:
    manifest = corpus_manifest(root)
    requested = None if system_ids is None else set(system_ids)
    records = []
    for record in manifest["systems"]:
        if requested is None or record["id"] in requested:
            records.append(record)
    return records


def system_root(system_record: dict, root: Path | None = None) -> Path:
    base = CORPUS_ROOT if root is None else root
    return base / system_record["path"]


def system_metadata(system_record: dict, root: Path | None = None) -> dict:
    return load_json(system_root(system_record, root) / "system.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BridgeError(message)


def source_ref(path: Path, line_number: int, raw_line: str) -> dict:
    return {
        "file": path.name,
        "line": line_number,
        "text": raw_line.rstrip("\n"),
    }


def is_float_token(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def kcal_to_kj(value: float) -> float:
    return value * KCAL_TO_KJ


def angstrom_to_nm(value: float) -> float:
    return value * ANGSTROM_TO_NM


def angstrom_per_fs_to_nm_per_ps(value: float) -> float:
    return value * 100.0


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


def type_name(type_id: int) -> str:
    return f"T{type_id}"


def molecule_label(system_meta: dict, molecule: dict, template_index: int) -> tuple[str, str]:
    category = system_meta["category"]
    system_id = system_meta["id"]
    total_charge = round(sum(atom["charge_e"] for atom in molecule["atoms"]), 8)
    if category == "oligomer":
        return ("OLI", "OLI")
    if category == "polymer_box":
        if len(molecule["atoms"]) > 1:
            return ("POL", "POL")
        if total_charge > 0:
            return ("CAT", "CAT")
        if total_charge < 0:
            return ("ANI", "ANI")
        raise BridgeError(
            f"Cannot derive polymer-box molecule label for neutral single-atom molecule {molecule['id']}"
        )
    if category == "toy":
        labels = {
            "bond_toy": "BOND",
            "angle_toy": "ANGL",
            "dihedral_toy": "DIHD",
            "improper_toy": "IMPR",
        }
        label = labels.get(system_id, f"TOY{template_index}")
        return (label, label)
    label = f"MOL{template_index}"
    return (label, label)


def generate_topological_one_four_pairs(local_bonds: list[dict], local_dihedrals: list[dict]) -> list[dict]:
    """Return the shortest-path 1-4 pairs used by LAMMPS special_bonds."""

    adjacency: dict[int, set[int]] = {}
    bond_by_pair: dict[tuple[int, int], dict] = {}
    for bond in local_bonds:
        ai, aj = bond["atoms"]
        adjacency.setdefault(ai, set()).add(aj)
        adjacency.setdefault(aj, set()).add(ai)
        bond_by_pair[tuple(sorted((ai, aj)))] = bond

    distance_three_paths: dict[tuple[int, int], list[int]] = {}
    for source in sorted(adjacency):
        distance = {source: 0}
        path = {source: [source]}
        queue = [source]
        for current in queue:
            if distance[current] == 3:
                continue
            for neighbor in sorted(adjacency[current]):
                if neighbor in distance:
                    continue
                distance[neighbor] = distance[current] + 1
                path[neighbor] = [*path[current], neighbor]
                queue.append(neighbor)
        for target, separation in distance.items():
            if separation == 3 and source < target:
                distance_three_paths[(source, target)] = path[target]

    generated_pairs = []
    seen_pairs = set()
    for dihedral in local_dihedrals:
        ai, aj = dihedral["atoms"][0], dihedral["atoms"][3]
        pair = tuple(sorted((ai, aj)))
        if pair not in distance_three_paths or pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        generated_pairs.append(
            {
                "ai": ai,
                "aj": aj,
                "funct": 1,
                "topological_distance_bonds": 3,
                "derived_from_dihedral_id": dihedral["id"],
                "source": dihedral["source"],
            }
        )

    for pair in sorted(distance_three_paths.keys() - seen_pairs):
        atom_path = distance_three_paths[pair]
        bond_path = [
            bond_by_pair[tuple(sorted((left, right)))]
            for left, right in zip(atom_path, atom_path[1:])
        ]
        generated_pairs.append(
            {
                "ai": pair[0],
                "aj": pair[1],
                "funct": 1,
                "topological_distance_bonds": 3,
                "derived_from_dihedral_id": None,
                "derived_from_bond_ids": [bond["id"] for bond in bond_path],
                "source": bond_path[0]["source"],
            }
        )

    return generated_pairs


def parse_lammps_data(path: Path) -> dict:
    count_keywords = {"atoms", "bonds", "angles", "dihedrals", "impropers"}
    type_keywords = {"atom types", "bond types", "angle types", "dihedral types", "improper types"}
    topology_section_names = {"Masses", "Atoms", "Bonds", "Angles", "Dihedrals", "Impropers", "Velocities"}
    coeff_section_names = {
        "Pair Coeffs",
        "Bond Coeffs",
        "Angle Coeffs",
        "Dihedral Coeffs",
        "Improper Coeffs",
        "BondBond Coeffs",
        "BondAngle Coeffs",
        "AngleAngleTorsion Coeffs",
        "EndBondTorsion Coeffs",
        "MiddleBondTorsion Coeffs",
        "BondBond13 Coeffs",
        "AngleTorsion Coeffs",
        "AngleAngle Coeffs",
    }
    section_names = topology_section_names | coeff_section_names
    header_counts = {key: 0 for key in count_keywords}
    type_counts = {key: 0 for key in type_keywords}
    box = {}
    sections = {
        "Masses": [],
        "Atoms": [],
        "Bonds": [],
        "Angles": [],
        "Dihedrals": [],
        "Impropers": [],
        "Velocities": [],
    }
    inline_coeffs = {
        "pair_coeffs": {},
        "bond_coeffs": {},
        "angle_coeffs": {},
        "dihedral_coeffs": {},
        "improper_coeffs": {},
    }

    current_section = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if line_number == 1 or stripped.startswith("LAMMPS data file"):
                continue

            tokens = stripped.split()
            if len(tokens) == 2 and tokens[1] in count_keywords:
                header_counts[tokens[1]] = int(tokens[0])
                current_section = None
                continue
            if len(tokens) == 3 and " ".join(tokens[1:]) in type_keywords:
                type_counts[" ".join(tokens[1:])] = int(tokens[0])
                current_section = None
                continue
            if len(tokens) == 4 and tokens[2] in {"xlo", "ylo", "zlo"} and tokens[3] in {"xhi", "yhi", "zhi"}:
                axis = tokens[2][0]
                box[axis] = {
                    "lo": float(tokens[0]),
                    "hi": float(tokens[1]),
                    "source": source_ref(path, line_number, raw_line),
                }
                current_section = None
                continue
            if len(tokens) == 6 and tokens[3:] == ["xy", "xz", "yz"]:
                raise BridgeError(
                    f"Restricted-triclinic LAMMPS boxes are not supported by the GRO bridge at "
                    f"{path}:{line_number}; refusing to discard xy/xz/yz tilt factors"
                )

            section_name = stripped.split(" #", 1)[0].strip()
            if section_name in section_names:
                current_section = section_name
                continue

            require(current_section is not None, f"Unexpected data-file line outside a section: {stripped}")
            data_tokens = raw_line.split("#", 1)[0].split()
            src = source_ref(path, line_number, raw_line)
            if current_section == "Pair Coeffs":
                require(len(data_tokens) >= 3, f"Malformed Pair Coeffs line at {path}:{line_number}")
                type_id = int(data_tokens[0])
                inline_coeffs["pair_coeffs"][type_id] = {
                    "type_id": type_id,
                    "epsilon_kcal_mol": float(data_tokens[1]),
                    "sigma_angstrom": float(data_tokens[2]),
                    "source": src,
                }
            elif current_section == "Bond Coeffs":
                require(len(data_tokens) >= 5, f"Malformed Bond Coeffs line at {path}:{line_number}")
                type_id = int(data_tokens[0])
                inline_coeffs["bond_coeffs"][type_id] = {
                    "type_id": type_id,
                    "r0_angstrom": float(data_tokens[1]),
                    "k2_kcal_mol_per_a2": float(data_tokens[2]),
                    "k3_kcal_mol_per_a3": float(data_tokens[3]),
                    "k4_kcal_mol_per_a4": float(data_tokens[4]),
                    "source": src,
                }
            elif current_section == "Angle Coeffs":
                require(len(data_tokens) >= 5, f"Malformed Angle Coeffs line at {path}:{line_number}")
                type_id = int(data_tokens[0])
                coeff = inline_coeffs["angle_coeffs"].setdefault(type_id, {"type_id": type_id})
                coeff["main"] = {
                    "theta0_deg": float(data_tokens[1]),
                    "k2_kcal_mol": float(data_tokens[2]),
                    "k3_kcal_mol": float(data_tokens[3]),
                    "k4_kcal_mol": float(data_tokens[4]),
                    "source": src,
                }
            elif current_section == "BondBond Coeffs":
                require(len(data_tokens) >= 4, f"Malformed BondBond Coeffs line at {path}:{line_number}")
                type_id = int(data_tokens[0])
                coeff = inline_coeffs["angle_coeffs"].setdefault(type_id, {"type_id": type_id})
                coeff["bb"] = {
                    "k_kcal_mol_per_a2": float(data_tokens[1]),
                    "r1_angstrom": float(data_tokens[2]),
                    "r2_angstrom": float(data_tokens[3]),
                    "source": src,
                }
            elif current_section == "BondAngle Coeffs":
                require(len(data_tokens) >= 5, f"Malformed BondAngle Coeffs line at {path}:{line_number}")
                type_id = int(data_tokens[0])
                coeff = inline_coeffs["angle_coeffs"].setdefault(type_id, {"type_id": type_id})
                coeff["ba"] = {
                    "k1_kcal_mol_per_a": float(data_tokens[1]),
                    "k2_kcal_mol_per_a": float(data_tokens[2]),
                    "r1_angstrom": float(data_tokens[3]),
                    "r2_angstrom": float(data_tokens[4]),
                    "source": src,
                }
            elif current_section == "Dihedral Coeffs":
                require(len(data_tokens) >= 7, f"Malformed Dihedral Coeffs line at {path}:{line_number}")
                type_id = int(data_tokens[0])
                coeff = inline_coeffs["dihedral_coeffs"].setdefault(type_id, {"type_id": type_id})
                coeff["main"] = {
                    "k1_kcal_mol": float(data_tokens[1]),
                    "phi1_deg": float(data_tokens[2]),
                    "k2_kcal_mol": float(data_tokens[3]),
                    "phi2_deg": float(data_tokens[4]),
                    "k3_kcal_mol": float(data_tokens[5]),
                    "phi3_deg": float(data_tokens[6]),
                    "source": src,
                }
            elif current_section == "MiddleBondTorsion Coeffs":
                require(len(data_tokens) >= 5, f"Malformed MiddleBondTorsion Coeffs line at {path}:{line_number}")
                type_id = int(data_tokens[0])
                coeff = inline_coeffs["dihedral_coeffs"].setdefault(type_id, {"type_id": type_id})
                coeff["mbt"] = {
                    "f1_kcal_mol_per_a": float(data_tokens[1]),
                    "f2_kcal_mol_per_a": float(data_tokens[2]),
                    "f3_kcal_mol_per_a": float(data_tokens[3]),
                    "r0_angstrom": float(data_tokens[4]),
                    "source": src,
                }
            elif current_section == "EndBondTorsion Coeffs":
                require(len(data_tokens) >= 9, f"Malformed EndBondTorsion Coeffs line at {path}:{line_number}")
                type_id = int(data_tokens[0])
                coeff = inline_coeffs["dihedral_coeffs"].setdefault(type_id, {"type_id": type_id})
                coeff["ebt"] = {
                    "f1_1_kcal_mol_per_a": float(data_tokens[1]),
                    "f2_1_kcal_mol_per_a": float(data_tokens[2]),
                    "f3_1_kcal_mol_per_a": float(data_tokens[3]),
                    "f1_2_kcal_mol_per_a": float(data_tokens[4]),
                    "f2_2_kcal_mol_per_a": float(data_tokens[5]),
                    "f3_2_kcal_mol_per_a": float(data_tokens[6]),
                    "r0_1_angstrom": float(data_tokens[7]),
                    "r0_2_angstrom": float(data_tokens[8]),
                    "source": src,
                }
            elif current_section == "AngleTorsion Coeffs":
                require(len(data_tokens) >= 9, f"Malformed AngleTorsion Coeffs line at {path}:{line_number}")
                type_id = int(data_tokens[0])
                coeff = inline_coeffs["dihedral_coeffs"].setdefault(type_id, {"type_id": type_id})
                coeff["at"] = {
                    "f1_1_kcal_mol": float(data_tokens[1]),
                    "f2_1_kcal_mol": float(data_tokens[2]),
                    "f3_1_kcal_mol": float(data_tokens[3]),
                    "f1_2_kcal_mol": float(data_tokens[4]),
                    "f2_2_kcal_mol": float(data_tokens[5]),
                    "f3_2_kcal_mol": float(data_tokens[6]),
                    "theta0_1_deg": float(data_tokens[7]),
                    "theta0_2_deg": float(data_tokens[8]),
                    "source": src,
                }
            elif current_section == "AngleAngleTorsion Coeffs":
                require(len(data_tokens) >= 4, f"Malformed AngleAngleTorsion Coeffs line at {path}:{line_number}")
                type_id = int(data_tokens[0])
                coeff = inline_coeffs["dihedral_coeffs"].setdefault(type_id, {"type_id": type_id})
                coeff["aat"] = {
                    "k_kcal_mol": float(data_tokens[1]),
                    "theta0_1_deg": float(data_tokens[2]),
                    "theta0_2_deg": float(data_tokens[3]),
                    "source": src,
                }
            elif current_section == "BondBond13 Coeffs":
                require(len(data_tokens) >= 4, f"Malformed BondBond13 Coeffs line at {path}:{line_number}")
                type_id = int(data_tokens[0])
                coeff = inline_coeffs["dihedral_coeffs"].setdefault(type_id, {"type_id": type_id})
                coeff["bb13"] = {
                    "k_kcal_mol_per_a2": float(data_tokens[1]),
                    "r1_angstrom": float(data_tokens[2]),
                    "r3_angstrom": float(data_tokens[3]),
                    "source": src,
                }
            elif current_section == "Improper Coeffs":
                require(len(data_tokens) >= 3, f"Malformed Improper Coeffs line at {path}:{line_number}")
                type_id = int(data_tokens[0])
                coeff = inline_coeffs["improper_coeffs"].setdefault(type_id, {"type_id": type_id})
                coeff["main"] = {
                    "k0_kcal_mol": float(data_tokens[1]),
                    "chi0_deg": float(data_tokens[2]),
                    "source": src,
                }
            elif current_section == "AngleAngle Coeffs":
                require(len(data_tokens) >= 7, f"Malformed AngleAngle Coeffs line at {path}:{line_number}")
                type_id = int(data_tokens[0])
                coeff = inline_coeffs["improper_coeffs"].setdefault(type_id, {"type_id": type_id})
                coeff["aa"] = {
                    "k1_kcal_mol": float(data_tokens[1]),
                    "k2_kcal_mol": float(data_tokens[2]),
                    "k3_kcal_mol": float(data_tokens[3]),
                    "theta0_1_deg": float(data_tokens[4]),
                    "theta0_2_deg": float(data_tokens[5]),
                    "theta0_3_deg": float(data_tokens[6]),
                    "source": src,
                }
            elif current_section == "Masses":
                require(len(data_tokens) >= 2, f"Malformed Masses line at {path}:{line_number}")
                sections[current_section].append(
                    {
                        "type_id": int(data_tokens[0]),
                        "mass_amu": float(data_tokens[1]),
                        "source": src,
                    }
                )
            elif current_section == "Atoms":
                require(len(data_tokens) >= 7, f"Malformed Atoms line at {path}:{line_number}")
                require(
                    len(data_tokens) == 7 or len(data_tokens) >= 10,
                    f"Incomplete LAMMPS image flags on Atoms line at {path}:{line_number}",
                )
                image_flags = (
                    {
                        "ix": int(data_tokens[7]),
                        "iy": int(data_tokens[8]),
                        "iz": int(data_tokens[9]),
                    }
                    if len(data_tokens) >= 10
                    else {"ix": 0, "iy": 0, "iz": 0}
                )
                sections[current_section].append(
                    {
                        "id": int(data_tokens[0]),
                        "molecule_id": int(data_tokens[1]),
                        "type_id": int(data_tokens[2]),
                        "charge_e": float(data_tokens[3]),
                        "x_angstrom": float(data_tokens[4]),
                        "y_angstrom": float(data_tokens[5]),
                        "z_angstrom": float(data_tokens[6]),
                        **image_flags,
                        "source": src,
                    }
                )
            elif current_section == "Bonds":
                require(len(data_tokens) >= 4, f"Malformed Bonds line at {path}:{line_number}")
                sections[current_section].append(
                    {
                        "id": int(data_tokens[0]),
                        "type_id": int(data_tokens[1]),
                        "atoms": [int(data_tokens[2]), int(data_tokens[3])],
                        "source": src,
                    }
                )
            elif current_section == "Angles":
                require(len(data_tokens) >= 5, f"Malformed Angles line at {path}:{line_number}")
                sections[current_section].append(
                    {
                        "id": int(data_tokens[0]),
                        "type_id": int(data_tokens[1]),
                        "atoms": [int(data_tokens[2]), int(data_tokens[3]), int(data_tokens[4])],
                        "source": src,
                    }
                )
            elif current_section in {"Dihedrals", "Impropers"}:
                require(len(data_tokens) >= 6, f"Malformed {current_section} line at {path}:{line_number}")
                sections[current_section].append(
                    {
                        "id": int(data_tokens[0]),
                        "type_id": int(data_tokens[1]),
                        "atoms": [int(data_tokens[2]), int(data_tokens[3]), int(data_tokens[4]), int(data_tokens[5])],
                        "source": src,
                    }
                )
            elif current_section == "Velocities":
                require(len(data_tokens) >= 4, f"Malformed Velocities line at {path}:{line_number}")
                sections[current_section].append(
                    {
                        "atom_id": int(data_tokens[0]),
                        "vx_angstrom_per_fs": float(data_tokens[1]),
                        "vy_angstrom_per_fs": float(data_tokens[2]),
                        "vz_angstrom_per_fs": float(data_tokens[3]),
                        "source": src,
                    }
                )

    require(set(box) == {"x", "y", "z"}, f"Missing box bounds in {path}")
    require(len(sections["Masses"]) == type_counts["atom types"], f"Mass count mismatch in {path}")
    require(len(sections["Atoms"]) == header_counts["atoms"], f"Atom count mismatch in {path}")
    require(len(sections["Bonds"]) == header_counts["bonds"], f"Bond count mismatch in {path}")
    require(len(sections["Angles"]) == header_counts["angles"], f"Angle count mismatch in {path}")
    require(len(sections["Dihedrals"]) == header_counts["dihedrals"], f"Dihedral count mismatch in {path}")
    require(len(sections["Impropers"]) == header_counts["impropers"], f"Improper count mismatch in {path}")

    return {
        "box": box,
        "header_counts": header_counts,
        "type_counts": type_counts,
        "masses": sorted(sections["Masses"], key=lambda item: item["type_id"]),
        "atoms": sorted(sections["Atoms"], key=lambda item: item["id"]),
        "bonds": sorted(sections["Bonds"], key=lambda item: item["id"]),
        "angles": sorted(sections["Angles"], key=lambda item: item["id"]),
        "dihedrals": sorted(sections["Dihedrals"], key=lambda item: item["id"]),
        "impropers": sorted(sections["Impropers"], key=lambda item: item["id"]),
        "velocities": sorted(sections["Velocities"], key=lambda item: item["atom_id"]),
        "inline_coeffs": inline_coeffs,
    }


def parse_lammps_input(path: Path) -> dict:
    parsed = {
        "styles": {},
        "style_sources": {},
        "pair_coeffs": {},
        "bond_coeffs": {},
        "angle_coeffs": {},
        "dihedral_coeffs": {},
        "improper_coeffs": {},
        "read_data": None,
    }

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.split("#", 1)[0].strip()
            if not stripped:
                continue
            tokens = stripped.split()
            command = tokens[0]
            args = tokens[1:]
            src = source_ref(path, line_number, raw_line)

            if command in {"units", "atom_style", "bond_style", "angle_style", "dihedral_style", "improper_style"}:
                require(len(args) == 1, f"{command} expects exactly one argument at {path}:{line_number}")
                parsed["styles"][command] = args[0]
                parsed["style_sources"][command] = src
            elif command == "pair_style":
                require(args, f"pair_style is missing its style name at {path}:{line_number}")
                parsed["styles"]["pair_style"] = {"kind": args[0], "args": args[1:]}
                parsed["style_sources"]["pair_style"] = src
            elif command == "pair_modify":
                require(args == ["mix", "sixthpower"], f"Unsupported pair_modify at {path}:{line_number}: {stripped}")
                parsed["styles"]["pair_modify"] = "mix sixthpower"
                parsed["style_sources"]["pair_modify"] = src
            elif command == "kspace_style":
                parsed["styles"]["kspace_style"] = " ".join(args)
                parsed["style_sources"]["kspace_style"] = src
            elif command == "special_bonds":
                parsed["styles"]["special_bonds"] = " ".join(args)
                parsed["style_sources"]["special_bonds"] = src
            elif command == "read_data":
                require(len(args) == 1, f"read_data expects exactly one path at {path}:{line_number}")
                parsed["read_data"] = {"path": args[0], "source": src}
            elif command == "pair_coeff":
                require(len(args) == 4, f"Unsupported pair_coeff at {path}:{line_number}: {stripped}")
                i_type = int(args[0])
                j_type = int(args[1])
                require(
                    i_type == j_type,
                    f"Unsupported cross pair_coeff at {path}:{line_number}; explicit self-only coefficients are required",
                )
                parsed["pair_coeffs"][i_type] = {
                    "type_id": i_type,
                    "epsilon_kcal_mol": float(args[2]),
                    "sigma_angstrom": float(args[3]),
                    "source": src,
                }
            elif command == "bond_coeff":
                require(len(args) == 5, f"Unsupported bond_coeff at {path}:{line_number}: {stripped}")
                type_id = int(args[0])
                parsed["bond_coeffs"][type_id] = {
                    "type_id": type_id,
                    "r0_angstrom": float(args[1]),
                    "k2_kcal_mol_per_a2": float(args[2]),
                    "k3_kcal_mol_per_a3": float(args[3]),
                    "k4_kcal_mol_per_a4": float(args[4]),
                    "source": src,
                }
            elif command == "angle_coeff":
                require(len(args) >= 2, f"Unsupported angle_coeff at {path}:{line_number}: {stripped}")
                type_id = int(args[0])
                coeff = parsed["angle_coeffs"].setdefault(type_id, {"type_id": type_id})
                if args[1] == "bb":
                    require(len(args) == 5, f"Unsupported angle_coeff bb at {path}:{line_number}: {stripped}")
                    coeff["bb"] = {
                        "k_kcal_mol_per_a2": float(args[2]),
                        "r1_angstrom": float(args[3]),
                        "r2_angstrom": float(args[4]),
                        "source": src,
                    }
                elif args[1] == "ba":
                    require(len(args) == 6, f"Unsupported angle_coeff ba at {path}:{line_number}: {stripped}")
                    coeff["ba"] = {
                        "k1_kcal_mol_per_a": float(args[2]),
                        "k2_kcal_mol_per_a": float(args[3]),
                        "r1_angstrom": float(args[4]),
                        "r2_angstrom": float(args[5]),
                        "source": src,
                    }
                else:
                    require(len(args) == 5, f"Unsupported angle_coeff main at {path}:{line_number}: {stripped}")
                    coeff["main"] = {
                        "theta0_deg": float(args[1]),
                        "k2_kcal_mol": float(args[2]),
                        "k3_kcal_mol": float(args[3]),
                        "k4_kcal_mol": float(args[4]),
                        "source": src,
                    }
            elif command == "dihedral_coeff":
                require(len(args) >= 2, f"Unsupported dihedral_coeff at {path}:{line_number}: {stripped}")
                type_id = int(args[0])
                coeff = parsed["dihedral_coeffs"].setdefault(type_id, {"type_id": type_id})
                subtype = args[1]
                if subtype == "mbt":
                    require(len(args) == 6, f"Unsupported dihedral_coeff mbt at {path}:{line_number}: {stripped}")
                    coeff["mbt"] = {
                        "f1_kcal_mol_per_a": float(args[2]),
                        "f2_kcal_mol_per_a": float(args[3]),
                        "f3_kcal_mol_per_a": float(args[4]),
                        "r0_angstrom": float(args[5]),
                        "source": src,
                    }
                elif subtype == "ebt":
                    require(len(args) == 10, f"Unsupported dihedral_coeff ebt at {path}:{line_number}: {stripped}")
                    coeff["ebt"] = {
                        "f1_1_kcal_mol_per_a": float(args[2]),
                        "f2_1_kcal_mol_per_a": float(args[3]),
                        "f3_1_kcal_mol_per_a": float(args[4]),
                        "f1_2_kcal_mol_per_a": float(args[5]),
                        "f2_2_kcal_mol_per_a": float(args[6]),
                        "f3_2_kcal_mol_per_a": float(args[7]),
                        "r0_1_angstrom": float(args[8]),
                        "r0_2_angstrom": float(args[9]),
                        "source": src,
                    }
                elif subtype == "at":
                    require(len(args) == 10, f"Unsupported dihedral_coeff at at {path}:{line_number}: {stripped}")
                    coeff["at"] = {
                        "f1_1_kcal_mol": float(args[2]),
                        "f2_1_kcal_mol": float(args[3]),
                        "f3_1_kcal_mol": float(args[4]),
                        "f1_2_kcal_mol": float(args[5]),
                        "f2_2_kcal_mol": float(args[6]),
                        "f3_2_kcal_mol": float(args[7]),
                        "theta0_1_deg": float(args[8]),
                        "theta0_2_deg": float(args[9]),
                        "source": src,
                    }
                elif subtype == "aat":
                    require(len(args) == 5, f"Unsupported dihedral_coeff aat at {path}:{line_number}: {stripped}")
                    coeff["aat"] = {
                        "k_kcal_mol": float(args[2]),
                        "theta0_1_deg": float(args[3]),
                        "theta0_2_deg": float(args[4]),
                        "source": src,
                    }
                elif subtype == "bb13":
                    require(len(args) == 5, f"Unsupported dihedral_coeff bb13 at {path}:{line_number}: {stripped}")
                    coeff["bb13"] = {
                        "k_kcal_mol_per_a2": float(args[2]),
                        "r1_angstrom": float(args[3]),
                        "r3_angstrom": float(args[4]),
                        "source": src,
                    }
                else:
                    require(len(args) == 7, f"Unsupported dihedral_coeff main at {path}:{line_number}: {stripped}")
                    coeff["main"] = {
                        "k1_kcal_mol": float(args[1]),
                        "phi1_deg": float(args[2]),
                        "k2_kcal_mol": float(args[3]),
                        "phi2_deg": float(args[4]),
                        "k3_kcal_mol": float(args[5]),
                        "phi3_deg": float(args[6]),
                        "source": src,
                    }
            elif command == "improper_coeff":
                require(len(args) >= 2, f"Unsupported improper_coeff at {path}:{line_number}: {stripped}")
                type_id = int(args[0])
                coeff = parsed["improper_coeffs"].setdefault(type_id, {"type_id": type_id})
                if args[1] == "aa":
                    require(len(args) == 8, f"Unsupported improper_coeff aa at {path}:{line_number}: {stripped}")
                    coeff["aa"] = {
                        "k1_kcal_mol": float(args[2]),
                        "k2_kcal_mol": float(args[3]),
                        "k3_kcal_mol": float(args[4]),
                        "theta0_1_deg": float(args[5]),
                        "theta0_2_deg": float(args[6]),
                        "theta0_3_deg": float(args[7]),
                        "source": src,
                    }
                else:
                    require(len(args) == 3, f"Unsupported improper_coeff main at {path}:{line_number}: {stripped}")
                    coeff["main"] = {
                        "k0_kcal_mol": float(args[1]),
                        "chi0_deg": float(args[2]),
                        "source": src,
                    }
            elif command in {"boundary", "neighbor", "neigh_modify"}:
                continue
            else:
                raise BridgeError(f"Unsupported LAMMPS command at {path}:{line_number}: {stripped}")

    require(parsed["read_data"] is not None, f"Missing read_data command in {path}")
    return parsed


def validate_fixture_support(system_meta: dict, parsed_data: dict, parsed_input: dict) -> None:
    styles = parsed_input["styles"]
    require(styles.get("units") == "real", f"Only `units real` is supported for {system_meta['id']}")
    require(styles.get("atom_style") == "full", f"Only `atom_style full` is supported for {system_meta['id']}")
    require(styles.get("bond_style") in {"class2", "none"}, f"Unsupported bond_style for {system_meta['id']}")
    require(styles.get("angle_style") in {"class2", "none"}, f"Unsupported angle_style for {system_meta['id']}")
    require(styles.get("dihedral_style") in {"class2", "none"}, f"Unsupported dihedral_style for {system_meta['id']}")
    require(styles.get("improper_style") in {"class2", "none"}, f"Unsupported improper_style for {system_meta['id']}")
    require(
        styles.get("special_bonds") == "lj/coul 0.0 0.0 1.0 angle no dihedral no",
        f"Unsupported special_bonds for {system_meta['id']}: {styles.get('special_bonds')}",
    )
    require(
        styles.get("pair_modify") == "mix sixthpower",
        f"Only `pair_modify mix sixthpower` is supported for {system_meta['id']}",
    )
    require(parsed_input["read_data"]["path"] == "system.data", f"read_data must target system.data for {system_meta['id']}")

    pair_style = styles.get("pair_style")
    require(pair_style is not None, f"Missing pair_style in {system_meta['id']}")
    require(
        pair_style["kind"] in {"lj/class2", "lj/class2/coul/long"},
        f"Unsupported pair_style for {system_meta['id']}: {pair_style['kind']}",
    )
    require(
        pair_style["kind"] == system_meta["styles"]["pair_style"],
        f"pair_style metadata mismatch for {system_meta['id']}: {pair_style['kind']} vs {system_meta['styles']['pair_style']}",
    )
    require(styles.get("bond_style") == system_meta["styles"]["bond_style"], f"bond_style metadata mismatch for {system_meta['id']}")
    require(styles.get("angle_style") == system_meta["styles"]["angle_style"], f"angle_style metadata mismatch for {system_meta['id']}")
    require(
        styles.get("dihedral_style") == system_meta["styles"]["dihedral_style"],
        f"dihedral_style metadata mismatch for {system_meta['id']}",
    )
    require(
        styles.get("improper_style") == system_meta["styles"]["improper_style"],
        f"improper_style metadata mismatch for {system_meta['id']}",
    )
    require(styles.get("special_bonds") == system_meta["styles"]["special_bonds"], f"special_bonds metadata mismatch for {system_meta['id']}")
    if system_meta["styles"]["kspace_style"] is None:
        require("kspace_style" not in styles, f"kspace_style should be absent for {system_meta['id']}")
    else:
        require(
            styles.get("kspace_style") == system_meta["styles"]["kspace_style"],
            f"kspace_style metadata mismatch for {system_meta['id']}",
        )

    used_atom_types = {atom["type_id"] for atom in parsed_data["atoms"]}
    require(
        used_atom_types.issubset(parsed_input["pair_coeffs"].keys()),
        f"Missing pair_coeff for atom types {sorted(used_atom_types - set(parsed_input['pair_coeffs']))} in {system_meta['id']}",
    )

    atom_to_molecule = {atom["id"]: atom["molecule_id"] for atom in parsed_data["atoms"]}
    for section_name in ("bonds", "angles", "dihedrals", "impropers"):
        for interaction in parsed_data[section_name]:
            molecule_ids = {atom_to_molecule[atom_id] for atom_id in interaction["atoms"]}
            require(
                len(molecule_ids) == 1,
                f"{section_name[:-1].capitalize()} {interaction['id']} crosses molecule boundaries in {system_meta['id']}",
            )

    interaction_to_style = {
        "bonds": ("bond_style", "bond_coeffs"),
        "angles": ("angle_style", "angle_coeffs"),
        "dihedrals": ("dihedral_style", "dihedral_coeffs"),
        "impropers": ("improper_style", "improper_coeffs"),
    }
    for section_name, (style_key, coeff_key) in interaction_to_style.items():
        used_types = {item["type_id"] for item in parsed_data[section_name]}
        if used_types:
            require(styles.get(style_key) == "class2", f"{style_key} must be class2 when {section_name} exist in {system_meta['id']}")
        else:
            continue
        missing = sorted(used_types - set(parsed_input[coeff_key]))
        require(not missing, f"Missing {coeff_key[:-1]} definitions for types {missing} in {system_meta['id']}")

    for type_id in {item["type_id"] for item in parsed_data["angles"]}:
        coeff = parsed_input["angle_coeffs"][type_id]
        for key in ("main", "bb", "ba"):
            require(key in coeff, f"Missing angle_coeff {key} for angle type {type_id} in {system_meta['id']}")
    for type_id in {item["type_id"] for item in parsed_data["dihedrals"]}:
        coeff = parsed_input["dihedral_coeffs"][type_id]
        for key in ("main", "mbt", "ebt", "at", "aat", "bb13"):
            require(key in coeff, f"Missing dihedral_coeff {key} for dihedral type {type_id} in {system_meta['id']}")
    for type_id in {item["type_id"] for item in parsed_data["impropers"]}:
        coeff = parsed_input["improper_coeffs"][type_id]
        for key in ("main", "aa"):
            require(key in coeff, f"Missing improper_coeff {key} for improper type {type_id} in {system_meta['id']}")


def build_typed_ir(system_record: dict, root: Path | None = None) -> dict:
    meta = system_metadata(system_record, root)
    base = system_root(system_record, root)
    parsed_data = parse_lammps_data(base / "lammps" / "system.data")
    parsed_input = parse_lammps_input(base / "lammps" / "system.in")
    validate_fixture_support(meta, parsed_data, parsed_input)

    masses_by_type = {item["type_id"]: item for item in parsed_data["masses"]}
    atom_types = []
    for type_id in sorted({atom["type_id"] for atom in parsed_data["atoms"]}):
        mass = masses_by_type[type_id]
        pair_coeff = parsed_input["pair_coeffs"][type_id]
        atom_types.append(
            {
                "id": type_id,
                "label": type_name(type_id),
                "mass_amu": mass["mass_amu"],
                "mass_source": mass["source"],
                "pair_coeff": pair_coeff,
            }
        )

    bond_types = []
    for type_id in sorted({item["type_id"] for item in parsed_data["bonds"]}):
        bond_types.append(parsed_input["bond_coeffs"][type_id])

    angle_types = []
    for type_id in sorted({item["type_id"] for item in parsed_data["angles"]}):
        angle_types.append(parsed_input["angle_coeffs"][type_id])

    dihedral_types = []
    for type_id in sorted({item["type_id"] for item in parsed_data["dihedrals"]}):
        dihedral_types.append(parsed_input["dihedral_coeffs"][type_id])

    improper_types = []
    for type_id in sorted({item["type_id"] for item in parsed_data["impropers"]}):
        improper_types.append(parsed_input["improper_coeffs"][type_id])

    atom_to_molecule = {atom["id"]: atom["molecule_id"] for atom in parsed_data["atoms"]}
    molecules = OrderedDict()
    for atom in parsed_data["atoms"]:
        molecules.setdefault(
            atom["molecule_id"],
            {"id": atom["molecule_id"], "atoms": [], "bonds": [], "angles": [], "dihedrals": [], "impropers": []},
        )
        molecules[atom["molecule_id"]]["atoms"].append(atom)
    for section_name in ("bonds", "angles", "dihedrals", "impropers"):
        for interaction in parsed_data[section_name]:
            molecule_id = atom_to_molecule[interaction["atoms"][0]]
            molecules[molecule_id][section_name].append(interaction)

    template_name_to_index = {}
    template_signature_to_name = {}
    molecule_templates = []
    molecule_instances = []

    for template_index, molecule in enumerate(molecules.values(), start=1):
        global_to_local = {atom["id"]: index for index, atom in enumerate(molecule["atoms"], start=1)}
        local_atoms = []
        for local_index, atom in enumerate(molecule["atoms"], start=1):
            local_atoms.append(
                {
                    "id": local_index,
                    "global_id": atom["id"],
                    "type_id": atom["type_id"],
                    "type_label": type_name(atom["type_id"]),
                    "charge_e": atom["charge_e"],
                    "mass_amu": masses_by_type[atom["type_id"]]["mass_amu"],
                    "coordinates_angstrom": {
                        "x": atom["x_angstrom"],
                        "y": atom["y_angstrom"],
                        "z": atom["z_angstrom"],
                    },
                    "atom_name": f"A{local_index}",
                    "source": atom["source"],
                }
            )

        def localize(section_name: str) -> list[dict]:
            localized = []
            for item in molecule[section_name]:
                localized.append(
                    {
                        "id": item["id"],
                        "type_id": item["type_id"],
                        "atoms": [global_to_local[atom_id] for atom_id in item["atoms"]],
                        "source": item["source"],
                    }
                )
            return localized

        local_bonds = localize("bonds")
        local_angles = localize("angles")
        local_dihedrals = localize("dihedrals")
        local_impropers = localize("impropers")
        generated_pairs = generate_topological_one_four_pairs(local_bonds, local_dihedrals)

        base_name, residue_name = molecule_label(meta, molecule, template_index)
        signature_payload = {
            "atoms": [(atom["type_id"], atom["charge_e"]) for atom in local_atoms],
            "bonds": [(item["type_id"], tuple(item["atoms"])) for item in local_bonds],
            "angles": [(item["type_id"], tuple(item["atoms"])) for item in local_angles],
            "dihedrals": [(item["type_id"], tuple(item["atoms"])) for item in local_dihedrals],
            "impropers": [(item["type_id"], tuple(item["atoms"])) for item in local_impropers],
        }
        signature = json.dumps(signature_payload, sort_keys=True)
        template_name = template_signature_to_name.get(signature)
        if template_name is None:
            template_name = base_name
            if template_name in template_name_to_index:
                template_name = f"{base_name}{template_index}"
            template_name_to_index[template_name] = len(molecule_templates)
            template_signature_to_name[signature] = template_name
            molecule_templates.append(
                {
                    "name": template_name,
                    "residue_name": residue_name,
                    "nrexcl": 3 if local_bonds else 1,
                    "atoms": local_atoms,
                    "bonds": local_bonds,
                    "angles": local_angles,
                    "dihedrals": local_dihedrals,
                    "impropers": local_impropers,
                    "generated_pairs": generated_pairs,
                }
            )
        molecule_instances.append(
            {
                "molecule_id": molecule["id"],
                "template_name": template_name,
                "num_atoms": len(local_atoms),
                "source": molecule["atoms"][0]["source"],
            }
        )

    pair_style = parsed_input["styles"]["pair_style"]
    typed_ir = {
        "schema_version": 1,
        "system_id": meta["id"],
        "display_name": meta["display_name"],
        "category": meta["category"],
        "description": meta["description"],
        "reference_terms": meta["reference_terms"],
        "source_files": {
            "system_json": "system.json",
            "system_data": "system.data",
            "system_in": "system.in",
        },
        "units": {
            "distance": "angstrom",
            "energy": "kcal/mol",
            "charge": "e",
            "mass": "amu",
        },
        "styles": {
            "units": parsed_input["styles"]["units"],
            "atom_style": parsed_input["styles"]["atom_style"],
            "pair_style": {
                "kind": pair_style["kind"],
                "args": pair_style["args"],
                "source": parsed_input["style_sources"]["pair_style"],
            },
            "pair_modify": {
                "value": parsed_input["styles"]["pair_modify"],
                "source": parsed_input["style_sources"]["pair_modify"],
            },
            "bond_style": parsed_input["styles"]["bond_style"],
            "angle_style": parsed_input["styles"]["angle_style"],
            "dihedral_style": parsed_input["styles"]["dihedral_style"],
            "improper_style": parsed_input["styles"]["improper_style"],
            "kspace_style": (
                {
                    "value": parsed_input["styles"]["kspace_style"],
                    "source": parsed_input["style_sources"]["kspace_style"],
                }
                if "kspace_style" in parsed_input["styles"]
                else None
            ),
            "special_bonds": {
                "value": parsed_input["styles"]["special_bonds"],
                "source": parsed_input["style_sources"]["special_bonds"],
            },
        },
        "box_angstrom": {
            axis: {"lo": parsed_data["box"][axis]["lo"], "hi": parsed_data["box"][axis]["hi"], "source": parsed_data["box"][axis]["source"]}
            for axis in ("x", "y", "z")
        },
        "atom_types": atom_types,
        "bond_types": bond_types,
        "angle_types": angle_types,
        "dihedral_types": dihedral_types,
        "improper_types": improper_types,
        "molecule_templates": molecule_templates,
        "molecule_instances": molecule_instances,
        "diagnostics": {
            "supported_gromacs_export": True,
            "generated_pair_rule": "Each unique shortest-path 1-4 pair is derived from the bond graph because LAMMPS special_bonds is 0 0 1 and GROMACS needs explicit [ pairs ]; explicit dihedrals provide provenance when present.",
            "notes": [
                "This IR is frozen in LAMMPS real units and retains source line provenance for every typed record.",
                "The current exporter supports only the repository fixture style subset and fails on any unsupported command or missing coefficient family.",
            ],
        },
    }
    return typed_ir


def normalize_special_bonds(value: str) -> str:
    tokens = value.split()
    if tokens == ["lj/coul", "0", "0", "1"] or tokens == ["lj/coul", "0.0", "0.0", "1.0"]:
        return "lj/coul 0.0 0.0 1.0 angle no dihedral no"
    return value


def validate_inline_data_support(system_id: str, parsed_data: dict, inline_coeffs: dict) -> None:
    used_atom_types = {atom["type_id"] for atom in parsed_data["atoms"]}
    require(
        used_atom_types.issubset(inline_coeffs["pair_coeffs"].keys()),
        f"Missing Pair Coeffs for atom types {sorted(used_atom_types - set(inline_coeffs['pair_coeffs']))} in {system_id}",
    )

    atom_to_molecule = {atom["id"]: atom["molecule_id"] for atom in parsed_data["atoms"]}
    for section_name in ("bonds", "angles", "dihedrals", "impropers"):
        for interaction in parsed_data[section_name]:
            molecule_ids = {atom_to_molecule[atom_id] for atom_id in interaction["atoms"]}
            require(
                len(molecule_ids) == 1,
                f"{section_name[:-1].capitalize()} {interaction['id']} crosses molecule boundaries in {system_id}",
            )

    coeff_keys = {
        "bonds": "bond_coeffs",
        "angles": "angle_coeffs",
        "dihedrals": "dihedral_coeffs",
        "impropers": "improper_coeffs",
    }
    for section_name, coeff_key in coeff_keys.items():
        used_types = {item["type_id"] for item in parsed_data[section_name]}
        missing = sorted(used_types - set(inline_coeffs[coeff_key]))
        require(not missing, f"Missing {coeff_key[:-1]} definitions for types {missing} in {system_id}")

    for type_id in {item["type_id"] for item in parsed_data["angles"]}:
        coeff = inline_coeffs["angle_coeffs"][type_id]
        for key in ("main", "bb", "ba"):
            require(key in coeff, f"Missing inline Angle Coeffs family {key} for angle type {type_id} in {system_id}")
    for type_id in {item["type_id"] for item in parsed_data["dihedrals"]}:
        coeff = inline_coeffs["dihedral_coeffs"][type_id]
        for key in ("main", "mbt", "ebt", "at", "aat", "bb13"):
            require(key in coeff, f"Missing inline Dihedral Coeffs family {key} for dihedral type {type_id} in {system_id}")
    for type_id in {item["type_id"] for item in parsed_data["impropers"]}:
        coeff = inline_coeffs["improper_coeffs"][type_id]
        for key in ("main", "aa"):
            require(key in coeff, f"Missing inline Improper Coeffs family {key} for improper type {type_id} in {system_id}")


def build_typed_ir_from_lammps_data(
    data_path: Path,
    *,
    system_id: str | None = None,
    display_name: str | None = None,
    category: str = "polymer_box",
    description: str | None = None,
    pair_style: str = "lj/class2/coul/long",
    pair_style_args: list[str] | None = None,
    pair_modify: str = "mix sixthpower",
    special_bonds: str = "lj/coul 0.0 0.0 1.0 angle no dihedral no",
    kspace_style: str | None = "pppm 1.0e-6",
) -> dict:
    parsed_data = parse_lammps_data(data_path)
    inline_coeffs = parsed_data["inline_coeffs"]
    resolved_system_id = system_id or data_path.stem
    validate_inline_data_support(resolved_system_id, parsed_data, inline_coeffs)

    masses_by_type = {item["type_id"]: item for item in parsed_data["masses"]}
    atom_types = []
    for type_id in sorted({atom["type_id"] for atom in parsed_data["atoms"]}):
        mass = masses_by_type[type_id]
        pair_coeff = inline_coeffs["pair_coeffs"][type_id]
        atom_types.append(
            {
                "id": type_id,
                "label": type_name(type_id),
                "mass_amu": mass["mass_amu"],
                "mass_source": mass["source"],
                "pair_coeff": pair_coeff,
            }
        )

    bond_types = [inline_coeffs["bond_coeffs"][type_id] for type_id in sorted({item["type_id"] for item in parsed_data["bonds"]})]
    angle_types = [inline_coeffs["angle_coeffs"][type_id] for type_id in sorted({item["type_id"] for item in parsed_data["angles"]})]
    dihedral_types = [
        inline_coeffs["dihedral_coeffs"][type_id] for type_id in sorted({item["type_id"] for item in parsed_data["dihedrals"]})
    ]
    improper_types = [
        inline_coeffs["improper_coeffs"][type_id] for type_id in sorted({item["type_id"] for item in parsed_data["impropers"]})
    ]

    atom_to_molecule = {atom["id"]: atom["molecule_id"] for atom in parsed_data["atoms"]}
    molecules = OrderedDict()
    for atom in parsed_data["atoms"]:
        molecules.setdefault(
            atom["molecule_id"],
            {"id": atom["molecule_id"], "atoms": [], "bonds": [], "angles": [], "dihedrals": [], "impropers": []},
        )
        molecules[atom["molecule_id"]]["atoms"].append(atom)
    for section_name in ("bonds", "angles", "dihedrals", "impropers"):
        for interaction in parsed_data[section_name]:
            molecule_id = atom_to_molecule[interaction["atoms"][0]]
            molecules[molecule_id][section_name].append(interaction)

    meta = {
        "id": resolved_system_id,
        "display_name": display_name or resolved_system_id,
        "category": category,
        "description": description or f"PCFF/Class2 system imported from {data_path.name}",
        "reference_terms": [
            "inline LAMMPS data-file Masses, Coeffs, Atoms, Bonds, Angles, Dihedrals, and Impropers sections",
        ],
    }
    template_name_to_index = {}
    template_signature_to_name = {}
    molecule_templates = []
    molecule_instances = []

    for template_index, molecule in enumerate(molecules.values(), start=1):
        global_to_local = {atom["id"]: index for index, atom in enumerate(molecule["atoms"], start=1)}
        local_atoms = []
        for local_index, atom in enumerate(molecule["atoms"], start=1):
            local_atoms.append(
                {
                    "id": local_index,
                    "global_id": atom["id"],
                    "type_id": atom["type_id"],
                    "type_label": type_name(atom["type_id"]),
                    "charge_e": atom["charge_e"],
                    "mass_amu": masses_by_type[atom["type_id"]]["mass_amu"],
                    "coordinates_angstrom": {
                        "x": atom["x_angstrom"],
                        "y": atom["y_angstrom"],
                        "z": atom["z_angstrom"],
                    },
                    "atom_name": f"A{local_index}",
                    "source": atom["source"],
                }
            )

        def localize(section_name: str) -> list[dict]:
            localized = []
            for item in molecule[section_name]:
                localized.append(
                    {
                        "id": item["id"],
                        "type_id": item["type_id"],
                        "atoms": [global_to_local[atom_id] for atom_id in item["atoms"]],
                        "source": item["source"],
                    }
                )
            return localized

        local_bonds = localize("bonds")
        local_angles = localize("angles")
        local_dihedrals = localize("dihedrals")
        local_impropers = localize("impropers")
        generated_pairs = generate_topological_one_four_pairs(local_bonds, local_dihedrals)

        base_name, residue = molecule_label(meta, molecule, template_index)
        signature_payload = {
            "atoms": [(atom["type_id"], atom["charge_e"]) for atom in local_atoms],
            "bonds": [(item["type_id"], tuple(item["atoms"])) for item in local_bonds],
            "angles": [(item["type_id"], tuple(item["atoms"])) for item in local_angles],
            "dihedrals": [(item["type_id"], tuple(item["atoms"])) for item in local_dihedrals],
            "impropers": [(item["type_id"], tuple(item["atoms"])) for item in local_impropers],
        }
        signature = json.dumps(signature_payload, sort_keys=True)
        template_name = template_signature_to_name.get(signature)
        if template_name is None:
            template_name = base_name
            if template_name in template_name_to_index:
                template_name = f"{base_name}{template_index}"
            template_name_to_index[template_name] = len(molecule_templates)
            template_signature_to_name[signature] = template_name
            molecule_templates.append(
                {
                    "name": template_name,
                    "residue_name": residue,
                    "nrexcl": 3 if local_bonds else 1,
                    "atoms": local_atoms,
                    "bonds": local_bonds,
                    "angles": local_angles,
                    "dihedrals": local_dihedrals,
                    "impropers": local_impropers,
                    "generated_pairs": generated_pairs,
                }
            )
        molecule_instances.append(
            {
                "molecule_id": molecule["id"],
                "template_name": template_name,
                "num_atoms": len(local_atoms),
                "source": molecule["atoms"][0]["source"],
            }
        )

    assumption_source = {"file": data_path.name, "line": None, "text": "Assumed by lammps_data_bridge CLI; LAMMPS data files do not store this command."}
    normalized_special_bonds = normalize_special_bonds(special_bonds)
    resolved_pair_style_args = [] if pair_style_args is None else pair_style_args
    return {
        "schema_version": 1,
        "system_id": resolved_system_id,
        "display_name": display_name or resolved_system_id,
        "category": category,
        "description": meta["description"],
        "reference_terms": meta["reference_terms"],
        "source_files": {
            "system_json": None,
            "system_data": str(data_path),
            "system_in": None,
        },
        "units": {
            "distance": "angstrom",
            "energy": "kcal/mol",
            "charge": "e",
            "mass": "amu",
        },
        "styles": {
            "units": "real",
            "atom_style": "full",
            "pair_style": {
                "kind": pair_style,
                "args": resolved_pair_style_args,
                "source": assumption_source,
            },
            "pair_modify": {
                "value": pair_modify,
                "source": assumption_source,
            },
            "bond_style": "class2" if parsed_data["bonds"] else "none",
            "angle_style": "class2" if parsed_data["angles"] else "none",
            "dihedral_style": "class2" if parsed_data["dihedrals"] else "none",
            "improper_style": "class2" if parsed_data["impropers"] else "none",
            "kspace_style": (
                {
                    "value": kspace_style,
                    "source": assumption_source,
                }
                if kspace_style is not None
                else None
            ),
            "special_bonds": {
                "value": normalized_special_bonds,
                "source": assumption_source,
            },
        },
        "box_angstrom": {
            axis: {"lo": parsed_data["box"][axis]["lo"], "hi": parsed_data["box"][axis]["hi"], "source": parsed_data["box"][axis]["source"]}
            for axis in ("x", "y", "z")
        },
        "atom_types": atom_types,
        "bond_types": bond_types,
        "angle_types": angle_types,
        "dihedral_types": dihedral_types,
        "improper_types": improper_types,
        "molecule_templates": molecule_templates,
        "molecule_instances": molecule_instances,
        "diagnostics": {
            "supported_gromacs_export": True,
            "import_mode": "single_lammps_data_file_with_inline_coeffs",
            "style_assumption_warning": (
                "LAMMPS data files do not encode units, pair_modify, special_bonds, or kspace_style commands; "
                "the CLI records those assumptions explicitly."
            ),
            "generated_pair_rule": "Each unique shortest-path 1-4 pair is derived from the bond graph because LAMMPS special_bonds is 0 0 1 and GROMACS needs explicit [ pairs ]; explicit dihedrals provide provenance when present.",
            "notes": [
                "This IR is frozen in LAMMPS real units and retains source line provenance for every typed record.",
                "Missing Class2 cross-term families abort export instead of silently degrading the topology.",
            ],
        },
    }


def format_float(value: float) -> str:
    return f"{value:.8f}"


def lammps_data_local_atom_names(parsed_data: dict) -> dict[int, str]:
    names = {}
    atoms_by_molecule = OrderedDict()
    for atom in parsed_data["atoms"]:
        atoms_by_molecule.setdefault(atom["molecule_id"], []).append(atom)
    for molecule_atoms in atoms_by_molecule.values():
        for local_index, atom in enumerate(sorted(molecule_atoms, key=lambda item: item["id"]), start=1):
            names[atom["id"]] = f"A{local_index}"
    return names


def render_gromacs_gro_from_lammps_data(
    parsed_data: dict,
    *,
    title: str = "Generated from LAMMPS data",
    residue_name: str = "MOL",
    shift_to_origin: bool = True,
) -> str:
    box_x = (parsed_data["box"]["x"]["hi"] - parsed_data["box"]["x"]["lo"]) * ANGSTROM_TO_NM
    box_y = (parsed_data["box"]["y"]["hi"] - parsed_data["box"]["y"]["lo"]) * ANGSTROM_TO_NM
    box_z = (parsed_data["box"]["z"]["hi"] - parsed_data["box"]["z"]["lo"]) * ANGSTROM_TO_NM
    box_lengths_angstrom = {
        axis: parsed_data["box"][axis]["hi"] - parsed_data["box"][axis]["lo"]
        for axis in ("x", "y", "z")
    }
    require(
        all(length > 0.0 and math.isfinite(length) for length in box_lengths_angstrom.values()),
        "LAMMPS box lengths must be finite and positive for GRO export",
    )
    origin = {
        axis: parsed_data["box"][axis]["lo"] if shift_to_origin else 0.0
        for axis in ("x", "y", "z")
    }
    atom_names = lammps_data_local_atom_names(parsed_data)

    lines = [title[:80], f"{len(parsed_data['atoms']):>5d}"]
    for atom in parsed_data["atoms"]:
        # LAMMPS `velocity ... zero angular` uses the unwrapped coordinates
        # reconstructed from atom image flags.  Preserve that same periodic
        # image in the initial GROMACS state so later gen-vel stages (Eq04)
        # inherit the representation rather than recomputing with wrapped x.
        x_unwrapped = atom["x_angstrom"] + atom.get("ix", 0) * box_lengths_angstrom["x"]
        y_unwrapped = atom["y_angstrom"] + atom.get("iy", 0) * box_lengths_angstrom["y"]
        z_unwrapped = atom["z_angstrom"] + atom.get("iz", 0) * box_lengths_angstrom["z"]
        x = (x_unwrapped - origin["x"]) * ANGSTROM_TO_NM
        y = (y_unwrapped - origin["y"]) * ANGSTROM_TO_NM
        z = (z_unwrapped - origin["z"]) * ANGSTROM_TO_NM
        residue_id = atom["molecule_id"] % 100000
        atom_name_value = atom_names[atom["id"]]
        lines.append(
            f"{residue_id:>5d}{residue_name[:5]:<5s}{atom_name_value[:5]:>5s}{atom['id'] % 100000:>5d}"
            f"{x:18.12f}{y:18.12f}{z:18.12f}"
        )
    lines.append(f"{box_x:18.12f}{box_y:18.12f}{box_z:18.12f}")
    return "\n".join(lines) + "\n"


def gromacs_atomtypes_lines(typed_ir: dict) -> list[str]:
    lines = [
        "[ atomtypes ]",
        "; name mass charge ptype sigma epsilon",
    ]
    for atom_type in typed_ir["atom_types"]:
        sigma_nm = angstrom_to_nm(atom_type["pair_coeff"]["sigma_angstrom"])
        epsilon_kj_mol = kcal_to_kj(atom_type["pair_coeff"]["epsilon_kcal_mol"])
        lines.append(
            f"{atom_type['label']:<8s} {atom_type['mass_amu']:.3f} 0.0 A {format_float(sigma_nm)} {format_float(epsilon_kj_mol)}"
        )
    return lines


def gromacs_bond_params(coeff: dict) -> list[float]:
    return [
        angstrom_to_nm(coeff["r0_angstrom"]),
        bond_k2_to_gromacs(coeff["k2_kcal_mol_per_a2"]),
        bond_k3_to_gromacs(coeff["k3_kcal_mol_per_a3"]),
        bond_k4_to_gromacs(coeff["k4_kcal_mol_per_a4"]),
    ]


def gromacs_angle_params(coeff: dict) -> list[float]:
    return [
        coeff["main"]["theta0_deg"],
        kcal_to_kj(coeff["main"]["k2_kcal_mol"]),
        kcal_to_kj(coeff["main"]["k3_kcal_mol"]),
        kcal_to_kj(coeff["main"]["k4_kcal_mol"]),
        bond_bond_k_to_gromacs(coeff["bb"]["k_kcal_mol_per_a2"]),
        angstrom_to_nm(coeff["bb"]["r1_angstrom"]),
        angstrom_to_nm(coeff["bb"]["r2_angstrom"]),
        bond_angle_k_to_gromacs(coeff["ba"]["k1_kcal_mol_per_a"]),
        bond_angle_k_to_gromacs(coeff["ba"]["k2_kcal_mol_per_a"]),
        angstrom_to_nm(coeff["ba"]["r1_angstrom"]),
        angstrom_to_nm(coeff["ba"]["r2_angstrom"]),
    ]


def gromacs_dihedral_params(coeff: dict) -> list[float]:
    return [
        kcal_to_kj(coeff["main"]["k1_kcal_mol"]),
        coeff["main"]["phi1_deg"],
        kcal_to_kj(coeff["main"]["k2_kcal_mol"]),
        coeff["main"]["phi2_deg"],
        kcal_to_kj(coeff["main"]["k3_kcal_mol"]),
        coeff["main"]["phi3_deg"],
        dihedral_bond_torsion_k_to_gromacs(coeff["mbt"]["f1_kcal_mol_per_a"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["mbt"]["f2_kcal_mol_per_a"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["mbt"]["f3_kcal_mol_per_a"]),
        angstrom_to_nm(coeff["mbt"]["r0_angstrom"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["ebt"]["f1_1_kcal_mol_per_a"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["ebt"]["f2_1_kcal_mol_per_a"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["ebt"]["f3_1_kcal_mol_per_a"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["ebt"]["f1_2_kcal_mol_per_a"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["ebt"]["f2_2_kcal_mol_per_a"]),
        dihedral_bond_torsion_k_to_gromacs(coeff["ebt"]["f3_2_kcal_mol_per_a"]),
        angstrom_to_nm(coeff["ebt"]["r0_1_angstrom"]),
        angstrom_to_nm(coeff["ebt"]["r0_2_angstrom"]),
        kcal_to_kj(coeff["at"]["f1_1_kcal_mol"]),
        kcal_to_kj(coeff["at"]["f2_1_kcal_mol"]),
        kcal_to_kj(coeff["at"]["f3_1_kcal_mol"]),
        kcal_to_kj(coeff["at"]["f1_2_kcal_mol"]),
        kcal_to_kj(coeff["at"]["f2_2_kcal_mol"]),
        kcal_to_kj(coeff["at"]["f3_2_kcal_mol"]),
        coeff["at"]["theta0_1_deg"],
        coeff["at"]["theta0_2_deg"],
        kcal_to_kj(coeff["aat"]["k_kcal_mol"]),
        coeff["aat"]["theta0_1_deg"],
        coeff["aat"]["theta0_2_deg"],
        bond_bond_k_to_gromacs(coeff["bb13"]["k_kcal_mol_per_a2"]),
        angstrom_to_nm(coeff["bb13"]["r1_angstrom"]),
        angstrom_to_nm(coeff["bb13"]["r3_angstrom"]),
    ]


def gromacs_improper_params(coeff: dict) -> list[float]:
    return [
        kcal_to_kj(coeff["main"]["k0_kcal_mol"]),
        coeff["main"]["chi0_deg"],
        kcal_to_kj(coeff["aa"]["k1_kcal_mol"]),  # aa_k1 = K1
        kcal_to_kj(coeff["aa"]["k2_kcal_mol"]),  # aa_k2 = K2
        kcal_to_kj(coeff["aa"]["k3_kcal_mol"]),  # aa_k3 = K3
        coeff["aa"]["theta0_1_deg"],             # aa_theta0_1 = theta1
        coeff["aa"]["theta0_2_deg"],             # aa_theta0_2 = theta2
        coeff["aa"]["theta0_3_deg"],             # aa_theta0_3 = theta3
    ]


def render_gromacs_topology(typed_ir: dict) -> str:
    require(
        typed_ir["styles"]["special_bonds"]["value"] == "lj/coul 0.0 0.0 1.0 angle no dihedral no",
        f"Unsupported special_bonds during export for {typed_ir['system_id']}",
    )
    require(
        typed_ir["styles"]["pair_modify"]["value"] == "mix sixthpower",
        f"Unsupported pair mixing during export for {typed_ir['system_id']}",
    )
    require(
        typed_ir["styles"]["pair_style"]["kind"] in {"lj/class2", "lj/class2/coul/long"},
        f"Unsupported pair_style during export for {typed_ir['system_id']}",
    )

    bond_types = {entry["type_id"]: entry for entry in typed_ir["bond_types"]}
    angle_types = {entry["type_id"]: entry for entry in typed_ir["angle_types"]}
    dihedral_types = {entry["type_id"]: entry for entry in typed_ir["dihedral_types"]}
    improper_types = {entry["type_id"]: entry for entry in typed_ir["improper_types"]}

    lines = [
        "[ defaults ]",
        "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ rep-pow",
        "1 4 yes 1.0 1.0 9.0",
        "",
    ]
    lines.extend(gromacs_atomtypes_lines(typed_ir))
    lines.append("")

    for template in typed_ir["molecule_templates"]:
        lines.extend(
            [
                "[ moleculetype ]",
                "; Name nrexcl",
                f"{template['name']} {template['nrexcl']}",
                "",
                "[ atoms ]",
                "; nr type resnr residue atom cgnr charge mass",
            ]
        )
        for atom in template["atoms"]:
            lines.append(
                f"{atom['id']:3d} {atom['type_label']:<6s} {1:5d} {template['residue_name']:<6s} {atom['atom_name']:<5s} "
                f"{atom['id']:4d} {atom['charge_e']: .8f} {atom['mass_amu']: .6f}"
            )

        if template["bonds"]:
            lines.extend(["", "[ bonds ]", "; ai aj funct c0 c1 c2 c3"])
            for bond in template["bonds"]:
                params = gromacs_bond_params(bond_types[bond["type_id"]])
                lines.append(
                    f"{bond['atoms'][0]:3d} {bond['atoms'][1]:3d} 11 " + " ".join(format_float(value) for value in params)
                )

        if template["generated_pairs"]:
            lines.extend(["", "[ pairs ]", "; ai aj funct"])
            for pair in template["generated_pairs"]:
                lines.append(f"{pair['ai']:3d} {pair['aj']:3d} {pair['funct']}")

        if template["angles"]:
            lines.extend(["", "[ angles ]", "; ai aj ak funct c0..c10"])
            for angle in template["angles"]:
                params = gromacs_angle_params(angle_types[angle["type_id"]])
                lines.append(
                    f"{angle['atoms'][0]:3d} {angle['atoms'][1]:3d} {angle['atoms'][2]:3d} 11 "
                    + " ".join(format_float(value) for value in params)
                )

        dihedral_records = []
        for dihedral in template["dihedrals"]:
            dihedral_records.append(
                {
                    "atoms": dihedral["atoms"],
                    "funct": 13,
                    "params": gromacs_dihedral_params(dihedral_types[dihedral["type_id"]]),
                }
            )
        for improper in template["impropers"]:
            # LAMMPS toy fixtures have (a, central, b, c)
            # GROMACS improper_class2 expects (a, central, b, c) where B is central
            # based on delr[0]=AB, delr[1]=BC, delr[2]=BD
            orig_atoms = improper["atoms"]
            rotated_atoms = [orig_atoms[0], orig_atoms[1], orig_atoms[2], orig_atoms[3]]
            dihedral_records.append(
                {
                    "atoms": rotated_atoms,
                    "funct": 12,
                    "params": gromacs_improper_params(improper_types[improper["type_id"]]),
                }
            )
        if dihedral_records:
            comment = "; ai aj ak al funct c0..c31" if template["dihedrals"] else "; ai aj ak al funct c0..c7"
            lines.extend(["", "[ dihedrals ]", comment])
            for record in dihedral_records:
                atom_fields = " ".join(f"{atom_id:3d}" for atom_id in record["atoms"])
                lines.append(f"{atom_fields} {record['funct']:2d} " + " ".join(format_float(value) for value in record["params"]))

        lines.append("")

    molecule_runs: list[tuple[str, int]] = []
    for instance in typed_ir["molecule_instances"]:
        template_name = instance["template_name"]
        if molecule_runs and molecule_runs[-1][0] == template_name:
            molecule_runs[-1] = (template_name, molecule_runs[-1][1] + 1)
        else:
            molecule_runs.append((template_name, 1))

    lines.extend(["[ system ]", typed_ir["system_id"], "", "[ molecules ]", "; Name number"])
    for template_name, count in molecule_runs:
        lines.append(f"{template_name} {count}")
    return "\n".join(lines) + "\n"
