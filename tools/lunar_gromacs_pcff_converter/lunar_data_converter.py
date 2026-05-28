from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from collections import deque
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_ROOT = REPO_ROOT / "tools" / "pcff_fixture_bridge"

import sys

if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from common import (  # noqa: E402
    ANGSTROM_TO_NM,
    BridgeError,
    render_gromacs_topology,
    source_ref,
    type_name,
)


STRUCTURAL_SECTIONS = {"Masses", "Atoms", "Bonds", "Angles", "Dihedrals", "Impropers", "Velocities"}
COEFF_SECTIONS = {
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
SECTION_NAMES = STRUCTURAL_SECTIONS | COEFF_SECTIONS
SMOKE_COORDINATE_MARGIN_NM = 0.2
SMOKE_EXCLUSION_CUTOFF_FRACTION = 0.85


ZERO_PARAMETER_FALLBACKS = {
    "c4o": {
        "mass_amu": 12.01115,
        "pair_coeff": {"epsilon_kcal_mol": 0.0748, "sigma_angstrom": 3.8700},
        "basis": "pcff_interface_v1_6mBN.frc mass line 273 and nonbond line 4166",
        "claim_scope": "known PCFF interface type recovered from LUNAR source comment",
    },
    "s_m": {
        "mass_amu": 32.06400,
        "pair_coeff": {"epsilon_kcal_mol": 0.2500, "sigma_angstrom": 4.3000},
        "basis": "pcff_interface_v1_6mBN.frc mass line 263 and nonbond line 4150",
        "claim_scope": "known PCFF interface type recovered from LUNAR source comment",
    },
    "S-type-yourself": {
        "mass_amu": 32.06400,
        "pair_coeff": {"epsilon_kcal_mol": 0.2500, "sigma_angstrom": 4.3000},
        "basis": "smoke-only sulfur fallback using s_m mass/nonbond values",
        "claim_scope": "LUNAR atom-typing failure fallback; not physical parameter-completion evidence",
    },
}


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lunar_comment_label(source: dict | None) -> str | None:
    if not source:
        return None
    text = source.get("text", "")
    if "#" not in text:
        return None
    comment = text.split("#", 1)[1].strip()
    if not comment:
        return None
    return comment.split()[0]


def max_topological_exclusion_distance_nm(parsed_data: dict, *, nrexcl: int = 3) -> float:
    atoms_by_id = {atom["id"]: atom for atom in parsed_data["atoms"]}
    graph: dict[int, set[int]] = {}
    for bond in parsed_data["bonds"]:
        ai, aj = bond["atoms"]
        graph.setdefault(ai, set()).add(aj)
        graph.setdefault(aj, set()).add(ai)

    max_distance_nm = 0.0
    for start in atoms_by_id:
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            atom_id, depth = queue.popleft()
            if 0 < depth <= nrexcl:
                ai = atoms_by_id[start]
                aj = atoms_by_id[atom_id]
                dx = ai["x_angstrom"] - aj["x_angstrom"]
                dy = ai["y_angstrom"] - aj["y_angstrom"]
                dz = ai["z_angstrom"] - aj["z_angstrom"]
                max_distance_nm = max(max_distance_nm, math.sqrt(dx * dx + dy * dy + dz * dz) * ANGSTROM_TO_NM)
            if depth >= nrexcl:
                continue
            for neighbor in graph.get(atom_id, ()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))
    return max_distance_nm


def smoke_validation_cutoff_nm(parsed_data: dict) -> float:
    max_exclusion_nm = max_topological_exclusion_distance_nm(parsed_data)
    exclusion_safe_cutoff = max_exclusion_nm / SMOKE_EXCLUSION_CUTOFF_FRACTION if max_exclusion_nm else 0.0
    return max(0.9, exclusion_safe_cutoff)


def smoke_validation_box_nm(parsed_data: dict, cutoff_nm: float | None = None) -> tuple[dict[str, float], dict[str, float]]:
    cutoff = smoke_validation_cutoff_nm(parsed_data) if cutoff_nm is None else cutoff_nm
    min_length_for_cutoff = 2.0 * cutoff + SMOKE_COORDINATE_MARGIN_NM
    lengths = {}
    shifts = {}
    for axis, key in (("x", "x_angstrom"), ("y", "y_angstrom"), ("z", "z_angstrom")):
        coords = [atom[key] * ANGSTROM_TO_NM for atom in parsed_data["atoms"]]
        coord_min = min(coords)
        coord_max = max(coords)
        original_length = (parsed_data["box"][axis]["hi"] - parsed_data["box"][axis]["lo"]) * ANGSTROM_TO_NM
        coordinate_length = coord_max - coord_min + 2.0 * SMOKE_COORDINATE_MARGIN_NM
        lengths[axis] = max(original_length, coordinate_length, min_length_for_cutoff)
        shifts[axis] = SMOKE_COORDINATE_MARGIN_NM - coord_min
    return lengths, shifts


def parse_lunar_pcff_data(path: Path) -> dict:
    count_keywords = {"atoms", "bonds", "angles", "dihedrals", "impropers"}
    type_keywords = {"atom types", "bond types", "angle types", "dihedral types", "improper types"}
    header_counts = {key: 0 for key in count_keywords}
    type_counts = {key: 0 for key in type_keywords}
    box = {}
    sections = {name: [] for name in STRUCTURAL_SECTIONS}
    coeffs = {
        "pair_coeffs": {},
        "bond_coeffs": {},
        "angle_coeffs": {},
        "dihedral_coeffs": {},
        "improper_coeffs": {},
    }
    section_sources = {}

    current_section = None
    header_line = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped.startswith("HEADER,"):
                header_line = source_ref(path, line_number, raw_line)
                current_section = None
                continue
            if stripped.startswith("LAMMPS data file"):
                current_section = None
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

            section_name = stripped.split("#", 1)[0].strip()
            if section_name in SECTION_NAMES:
                current_section = section_name
                section_sources[current_section] = source_ref(path, line_number, raw_line)
                continue

            if current_section is None:
                raise BridgeError(f"Unexpected LUNAR data-file line outside a section at {path}:{line_number}: {stripped}")

            data_tokens = raw_line.split("#", 1)[0].split()
            if not data_tokens:
                continue
            src = source_ref(path, line_number, raw_line)
            if current_section == "Masses":
                sections[current_section].append(
                    {"type_id": int(data_tokens[0]), "mass_amu": float(data_tokens[1]), "source": src}
                )
            elif current_section == "Atoms":
                sections[current_section].append(
                    {
                        "id": int(data_tokens[0]),
                        "molecule_id": int(data_tokens[1]),
                        "type_id": int(data_tokens[2]),
                        "charge_e": float(data_tokens[3]),
                        "x_angstrom": float(data_tokens[4]),
                        "y_angstrom": float(data_tokens[5]),
                        "z_angstrom": float(data_tokens[6]),
                        "source": src,
                    }
                )
            elif current_section == "Bonds":
                sections[current_section].append(
                    {
                        "id": int(data_tokens[0]),
                        "type_id": int(data_tokens[1]),
                        "atoms": [int(data_tokens[2]), int(data_tokens[3])],
                        "source": src,
                    }
                )
            elif current_section == "Angles":
                sections[current_section].append(
                    {
                        "id": int(data_tokens[0]),
                        "type_id": int(data_tokens[1]),
                        "atoms": [int(data_tokens[2]), int(data_tokens[3]), int(data_tokens[4])],
                        "source": src,
                    }
                )
            elif current_section in {"Dihedrals", "Impropers"}:
                sections[current_section].append(
                    {
                        "id": int(data_tokens[0]),
                        "type_id": int(data_tokens[1]),
                        "atoms": [int(data_tokens[2]), int(data_tokens[3]), int(data_tokens[4]), int(data_tokens[5])],
                        "source": src,
                    }
                )
            elif current_section == "Velocities":
                sections[current_section].append(
                    {
                        "atom_id": int(data_tokens[0]),
                        "vx_angstrom_per_fs": float(data_tokens[1]),
                        "vy_angstrom_per_fs": float(data_tokens[2]),
                        "vz_angstrom_per_fs": float(data_tokens[3]),
                        "source": src,
                    }
                )
            else:
                _parse_coeff_line(current_section, data_tokens, src, coeffs)

    _validate_parsed_lunar_data(path, header_counts, type_counts, box, sections, coeffs)
    return {
        "source_path": str(path),
        "header_line": header_line,
        "section_sources": section_sources,
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
        "coeffs": coeffs,
    }


def _parse_coeff_line(section: str, tokens: list[str], src: dict, coeffs: dict) -> None:
    type_id = int(tokens[0])
    values = [float(token) for token in tokens[1:]]
    if section == "Pair Coeffs":
        if len(values) != 2:
            raise BridgeError(f"Pair Coeffs type {type_id} expected 2 values")
        coeffs["pair_coeffs"][type_id] = {
            "type_id": type_id,
            "epsilon_kcal_mol": values[0],
            "sigma_angstrom": values[1],
            "source": src,
        }
    elif section == "Bond Coeffs":
        if len(values) != 4:
            raise BridgeError(f"Bond Coeffs type {type_id} expected 4 values")
        coeffs["bond_coeffs"][type_id] = {
            "type_id": type_id,
            "r0_angstrom": values[0],
            "k2_kcal_mol_per_a2": values[1],
            "k3_kcal_mol_per_a3": values[2],
            "k4_kcal_mol_per_a4": values[3],
            "source": src,
        }
    elif section == "Angle Coeffs":
        if len(values) != 4:
            raise BridgeError(f"Angle Coeffs type {type_id} expected 4 values")
        coeff = coeffs["angle_coeffs"].setdefault(type_id, {"type_id": type_id})
        coeff["main"] = {
            "theta0_deg": values[0],
            "k2_kcal_mol": values[1],
            "k3_kcal_mol": values[2],
            "k4_kcal_mol": values[3],
            "source": src,
        }
    elif section == "BondBond Coeffs":
        if len(values) != 3:
            raise BridgeError(f"BondBond Coeffs type {type_id} expected 3 values")
        coeff = coeffs["angle_coeffs"].setdefault(type_id, {"type_id": type_id})
        coeff["bb"] = {
            "k_kcal_mol_per_a2": values[0],
            "r1_angstrom": values[1],
            "r2_angstrom": values[2],
            "source": src,
        }
    elif section == "BondAngle Coeffs":
        if len(values) != 4:
            raise BridgeError(f"BondAngle Coeffs type {type_id} expected 4 values")
        coeff = coeffs["angle_coeffs"].setdefault(type_id, {"type_id": type_id})
        coeff["ba"] = {
            "k1_kcal_mol_per_a": values[0],
            "k2_kcal_mol_per_a": values[1],
            "r1_angstrom": values[2],
            "r2_angstrom": values[3],
            "source": src,
        }
    elif section == "Dihedral Coeffs":
        if len(values) != 6:
            raise BridgeError(f"Dihedral Coeffs type {type_id} expected 6 values")
        coeff = coeffs["dihedral_coeffs"].setdefault(type_id, {"type_id": type_id})
        coeff["main"] = {
            "k1_kcal_mol": values[0],
            "phi1_deg": values[1],
            "k2_kcal_mol": values[2],
            "phi2_deg": values[3],
            "k3_kcal_mol": values[4],
            "phi3_deg": values[5],
            "source": src,
        }
    elif section == "MiddleBondTorsion Coeffs":
        if len(values) != 4:
            raise BridgeError(f"MiddleBondTorsion Coeffs type {type_id} expected 4 values")
        coeff = coeffs["dihedral_coeffs"].setdefault(type_id, {"type_id": type_id})
        coeff["mbt"] = {
            "f1_kcal_mol_per_a": values[0],
            "f2_kcal_mol_per_a": values[1],
            "f3_kcal_mol_per_a": values[2],
            "r0_angstrom": values[3],
            "source": src,
        }
    elif section == "EndBondTorsion Coeffs":
        if len(values) != 8:
            raise BridgeError(f"EndBondTorsion Coeffs type {type_id} expected 8 values")
        coeff = coeffs["dihedral_coeffs"].setdefault(type_id, {"type_id": type_id})
        coeff["ebt"] = {
            "f1_1_kcal_mol_per_a": values[0],
            "f2_1_kcal_mol_per_a": values[1],
            "f3_1_kcal_mol_per_a": values[2],
            "f1_2_kcal_mol_per_a": values[3],
            "f2_2_kcal_mol_per_a": values[4],
            "f3_2_kcal_mol_per_a": values[5],
            "r0_1_angstrom": values[6],
            "r0_2_angstrom": values[7],
            "source": src,
        }
    elif section == "AngleTorsion Coeffs":
        if len(values) != 8:
            raise BridgeError(f"AngleTorsion Coeffs type {type_id} expected 8 values")
        coeff = coeffs["dihedral_coeffs"].setdefault(type_id, {"type_id": type_id})
        coeff["at"] = {
            "f1_1_kcal_mol": values[0],
            "f2_1_kcal_mol": values[1],
            "f3_1_kcal_mol": values[2],
            "f1_2_kcal_mol": values[3],
            "f2_2_kcal_mol": values[4],
            "f3_2_kcal_mol": values[5],
            "theta0_1_deg": values[6],
            "theta0_2_deg": values[7],
            "source": src,
        }
    elif section == "AngleAngleTorsion Coeffs":
        if len(values) != 3:
            raise BridgeError(f"AngleAngleTorsion Coeffs type {type_id} expected 3 values")
        coeff = coeffs["dihedral_coeffs"].setdefault(type_id, {"type_id": type_id})
        coeff["aat"] = {
            "k_kcal_mol": values[0],
            "theta0_1_deg": values[1],
            "theta0_2_deg": values[2],
            "source": src,
        }
    elif section == "BondBond13 Coeffs":
        if len(values) != 3:
            raise BridgeError(f"BondBond13 Coeffs type {type_id} expected 3 values")
        coeff = coeffs["dihedral_coeffs"].setdefault(type_id, {"type_id": type_id})
        coeff["bb13"] = {
            "k_kcal_mol_per_a2": values[0],
            "r1_angstrom": values[1],
            "r3_angstrom": values[2],
            "source": src,
        }
    elif section == "Improper Coeffs":
        if len(values) != 2:
            raise BridgeError(f"Improper Coeffs type {type_id} expected 2 values")
        coeff = coeffs["improper_coeffs"].setdefault(type_id, {"type_id": type_id})
        coeff["main"] = {
            "k0_kcal_mol": values[0],
            "chi0_deg": values[1],
            "source": src,
        }
    elif section == "AngleAngle Coeffs":
        if len(values) != 6:
            raise BridgeError(f"AngleAngle Coeffs type {type_id} expected 6 values")
        coeff = coeffs["improper_coeffs"].setdefault(type_id, {"type_id": type_id})
        coeff["aa"] = {
            "k1_kcal_mol": values[0],
            "k2_kcal_mol": values[1],
            "k3_kcal_mol": values[2],
            "theta0_1_deg": values[3],
            "theta0_2_deg": values[4],
            "theta0_3_deg": values[5],
            "source": src,
        }
    else:
        raise BridgeError(f"Unsupported coefficient section: {section}")


def _validate_parsed_lunar_data(
    path: Path,
    header_counts: dict,
    type_counts: dict,
    box: dict,
    sections: dict,
    coeffs: dict,
) -> None:
    if set(box) != {"x", "y", "z"}:
        raise BridgeError(f"Missing box bounds in {path}")
    expected_counts = {
        "Masses": type_counts["atom types"],
        "Atoms": header_counts["atoms"],
        "Bonds": header_counts["bonds"],
        "Angles": header_counts["angles"],
        "Dihedrals": header_counts["dihedrals"],
        "Impropers": header_counts["impropers"],
    }
    for section, expected in expected_counts.items():
        if len(sections[section]) != expected:
            raise BridgeError(f"{section} count mismatch in {path}: {len(sections[section])} != {expected}")

    _require_coeffs(path, "pair_coeffs", {atom["type_id"] for atom in sections["Atoms"]}, coeffs["pair_coeffs"])
    _require_coeffs(path, "bond_coeffs", {item["type_id"] for item in sections["Bonds"]}, coeffs["bond_coeffs"])
    _require_coeffs(path, "angle_coeffs", {item["type_id"] for item in sections["Angles"]}, coeffs["angle_coeffs"])
    _require_coeffs(path, "dihedral_coeffs", {item["type_id"] for item in sections["Dihedrals"]}, coeffs["dihedral_coeffs"])
    _require_coeffs(path, "improper_coeffs", {item["type_id"] for item in sections["Impropers"]}, coeffs["improper_coeffs"])

    for type_id in {item["type_id"] for item in sections["Angles"]}:
        for key in ("main", "bb", "ba"):
            if key not in coeffs["angle_coeffs"][type_id]:
                raise BridgeError(f"Missing angle {key} coefficient for type {type_id} in {path}")
    for type_id in {item["type_id"] for item in sections["Dihedrals"]}:
        for key in ("main", "mbt", "ebt", "at", "aat", "bb13"):
            if key not in coeffs["dihedral_coeffs"][type_id]:
                raise BridgeError(f"Missing dihedral {key} coefficient for type {type_id} in {path}")
    for type_id in {item["type_id"] for item in sections["Impropers"]}:
        for key in ("main", "aa"):
            if key not in coeffs["improper_coeffs"][type_id]:
                raise BridgeError(f"Missing improper {key} coefficient for type {type_id} in {path}")


def _require_coeffs(path: Path, name: str, used_types: set[int], coeffs: dict[int, dict]) -> None:
    missing = sorted(used_types - set(coeffs))
    if missing:
        raise BridgeError(f"Missing {name} for used types {missing} in {path}")


def build_typed_ir_from_lunar_data(
    parsed_data: dict,
    *,
    system_id: str,
    display_name: str | None = None,
    pair_style_kind: str = "lj/class2/coul/long",
) -> dict:
    coeffs = parsed_data["coeffs"]
    masses_by_type = {item["type_id"]: item for item in parsed_data["masses"]}
    effective_masses_by_type = {type_id: dict(mass) for type_id, mass in masses_by_type.items()}
    effective_pair_coeffs = {type_id: dict(coeff) for type_id, coeff in coeffs["pair_coeffs"].items()}
    parameter_fallbacks = []
    atom_types = []
    for type_id in sorted({atom["type_id"] for atom in parsed_data["atoms"]}):
        mass = masses_by_type[type_id]
        effective_mass = effective_masses_by_type[type_id]
        source_label = lunar_comment_label(mass["source"])
        pair_coeff = effective_pair_coeffs[type_id]
        fallback = ZERO_PARAMETER_FALLBACKS.get(source_label or "")
        if fallback and mass["mass_amu"] <= 0.0:
            effective_mass["mass_amu"] = fallback["mass_amu"]
            effective_mass["fallback_source"] = {
                "label": source_label,
                "original_mass_amu": mass["mass_amu"],
                "basis": fallback["basis"],
                "claim_scope": fallback["claim_scope"],
            }
            parameter_fallbacks.append(
                {
                    "type_id": type_id,
                    "source_label": source_label,
                    "field": "mass_amu",
                    "original_value": mass["mass_amu"],
                    "fallback_value": fallback["mass_amu"],
                    "basis": fallback["basis"],
                    "claim_scope": fallback["claim_scope"],
                }
            )
        if fallback and pair_coeff["epsilon_kcal_mol"] == 0.0 and pair_coeff["sigma_angstrom"] == 0.0:
            original_pair = {
                "epsilon_kcal_mol": pair_coeff["epsilon_kcal_mol"],
                "sigma_angstrom": pair_coeff["sigma_angstrom"],
            }
            pair_coeff.update(fallback["pair_coeff"])
            pair_coeff["fallback_source"] = {
                "label": source_label,
                "original_pair_coeff": original_pair,
                "basis": fallback["basis"],
                "claim_scope": fallback["claim_scope"],
            }
            parameter_fallbacks.append(
                {
                    "type_id": type_id,
                    "source_label": source_label,
                    "field": "pair_coeff",
                    "original_value": original_pair,
                    "fallback_value": fallback["pair_coeff"],
                    "basis": fallback["basis"],
                    "claim_scope": fallback["claim_scope"],
                }
            )
        atom_types.append(
            {
                "id": type_id,
                "label": type_name(type_id),
                "source_label": source_label,
                "mass_amu": effective_mass["mass_amu"],
                "original_mass_amu": mass["mass_amu"],
                "mass_source": mass["source"],
                "pair_coeff": pair_coeff,
            }
        )

    typed_ir = {
        "schema_version": 1,
        "system_id": system_id,
        "display_name": display_name or system_id,
        "category": "lunar_pcff_data_polymer",
        "description": "Generated from a LUNAR all2lmp PCFF LAMMPS data file with embedded coefficients.",
        "reference_terms": ["LUNAR all2lmp class2 data-file coefficient sections"],
        "source_files": {"lunar_pcff_data": parsed_data["source_path"]},
        "units": {"distance": "angstrom", "energy": "kcal/mol", "charge": "e", "mass": "amu"},
        "styles": {
            "units": "real",
            "atom_style": "full",
            "pair_style": {"kind": pair_style_kind, "args": [], "source": parsed_data["section_sources"].get("Pair Coeffs")},
            "pair_modify": {"value": "mix sixthpower", "source": parsed_data["section_sources"].get("Pair Coeffs")},
            "bond_style": "class2" if parsed_data["bonds"] else "none",
            "angle_style": "class2" if parsed_data["angles"] else "none",
            "dihedral_style": "class2" if parsed_data["dihedrals"] else "none",
            "improper_style": "class2" if parsed_data["impropers"] else "none",
            "kspace_style": {"value": "pppm 1.0e-4", "source": parsed_data["section_sources"].get("Pair Coeffs")},
            "special_bonds": {
                "value": "lj/coul 0.0 0.0 1.0 angle no dihedral no",
                "source": parsed_data["header_line"] or parsed_data["section_sources"].get("Pair Coeffs"),
            },
        },
        "box_angstrom": {
            axis: {
                "lo": parsed_data["box"][axis]["lo"],
                "hi": parsed_data["box"][axis]["hi"],
                "source": parsed_data["box"][axis]["source"],
            }
            for axis in ("x", "y", "z")
        },
        "atom_types": atom_types,
        "bond_types": [coeffs["bond_coeffs"][type_id] for type_id in sorted({item["type_id"] for item in parsed_data["bonds"]})],
        "angle_types": [coeffs["angle_coeffs"][type_id] for type_id in sorted({item["type_id"] for item in parsed_data["angles"]})],
        "dihedral_types": [
            coeffs["dihedral_coeffs"][type_id] for type_id in sorted({item["type_id"] for item in parsed_data["dihedrals"]})
        ],
        "improper_types": [
            coeffs["improper_coeffs"][type_id] for type_id in sorted({item["type_id"] for item in parsed_data["impropers"]})
        ],
        "molecule_templates": [],
        "molecule_instances": [],
        "diagnostics": {
            "supported_gromacs_export": True,
            "generated_pair_rule": "Each unique 1-4 pair is derived from the first and fourth atom of each Class2 dihedral because special_bonds is 0 0 1 and GROMACS needs explicit [ pairs ].",
            "notes": [
                "This path handles LUNAR data files with embedded Class2 coefficient sections.",
                "It is polymer-only until charged ion/salt components have separate public evidence.",
            ],
            "parameter_fallbacks": parameter_fallbacks,
        },
    }
    _populate_molecule_templates(typed_ir, parsed_data, effective_masses_by_type)
    return typed_ir


def _populate_molecule_templates(typed_ir: dict, parsed_data: dict, masses_by_type: dict[int, dict]) -> None:
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
            molecule_ids = {atom_to_molecule[atom_id] for atom_id in interaction["atoms"]}
            if len(molecule_ids) != 1:
                raise BridgeError(f"{section_name[:-1]} {interaction['id']} crosses molecule boundaries")
            molecule_id = atom_to_molecule[interaction["atoms"][0]]
            molecules[molecule_id][section_name].append(interaction)

    template_name_to_index = {}
    template_signature_to_name = {}
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
            return [
                {
                    "id": item["id"],
                    "type_id": item["type_id"],
                    "atoms": [global_to_local[atom_id] for atom_id in item["atoms"]],
                    "source": item["source"],
                }
                for item in molecule[section_name]
            ]

        local_bonds = localize("bonds")
        local_angles = localize("angles")
        local_dihedrals = localize("dihedrals")
        local_impropers = localize("impropers")
        generated_pairs = []
        seen_pairs = set()
        for dihedral in local_dihedrals:
            pair = tuple(sorted((dihedral["atoms"][0], dihedral["atoms"][3])))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            generated_pairs.append(
                {
                    "ai": pair[0],
                    "aj": pair[1],
                    "funct": 1,
                    "derived_from_dihedral_id": dihedral["id"],
                    "source": dihedral["source"],
                }
            )

        base_name = "POL" if len(local_atoms) > 1 else "MOL"
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
            template_name_to_index[template_name] = len(typed_ir["molecule_templates"])
            template_signature_to_name[signature] = template_name
            typed_ir["molecule_templates"].append(
                {
                    "name": template_name,
                    "residue_name": base_name,
                    "nrexcl": 3 if local_bonds else 1,
                    "atoms": local_atoms,
                    "bonds": local_bonds,
                    "angles": local_angles,
                    "dihedrals": local_dihedrals,
                    "impropers": local_impropers,
                    "generated_pairs": generated_pairs,
                }
            )
        typed_ir["molecule_instances"].append(
            {
                "molecule_id": molecule["id"],
                "template_name": template_name,
                "num_atoms": len(local_atoms),
                "source": molecule["atoms"][0]["source"],
            }
        )


def render_gro_from_lunar_data(parsed_data: dict) -> str:
    cutoff_nm = smoke_validation_cutoff_nm(parsed_data)
    box_lengths, coordinate_shifts = smoke_validation_box_nm(parsed_data, cutoff_nm)

    local_names = {}
    atoms_by_molecule = {}
    for atom in parsed_data["atoms"]:
        atoms_by_molecule.setdefault(atom["molecule_id"], []).append(atom)
    for atoms in atoms_by_molecule.values():
        for index, atom in enumerate(sorted(atoms, key=lambda item: item["id"]), start=1):
            local_names[atom["id"]] = f"A{index}"

    lines = ["Generated from LUNAR PCFF data for database converter smoke", f"{len(parsed_data['atoms']):>5d}"]
    for atom in parsed_data["atoms"]:
        residue_id = atom["molecule_id"] % 100000
        atom_id = atom["id"] % 100000
        lines.append(
            f"{residue_id:>5d}{'POL':<5s}{local_names[atom['id']]:>5s}{atom_id:>5d}"
            f"{atom['x_angstrom'] * ANGSTROM_TO_NM + coordinate_shifts['x']:15.7f}"
            f"{atom['y_angstrom'] * ANGSTROM_TO_NM + coordinate_shifts['y']:15.7f}"
            f"{atom['z_angstrom'] * ANGSTROM_TO_NM + coordinate_shifts['z']:15.7f}"
        )
    lines.append(f"{box_lengths['x']:15.7f}{box_lengths['y']:15.7f}{box_lengths['z']:15.7f}")
    return "\n".join(lines) + "\n"


def convert_lunar_data_text(data_path: Path, *, system_id: str) -> tuple[dict, dict, str, str]:
    parsed_data = parse_lunar_pcff_data(data_path)
    typed_ir = build_typed_ir_from_lunar_data(parsed_data, system_id=system_id)
    topol_text = render_gromacs_topology(typed_ir)
    gro_text = render_gro_from_lunar_data(parsed_data)
    return parsed_data, typed_ir, topol_text, gro_text
