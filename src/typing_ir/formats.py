from __future__ import annotations

from pathlib import Path
import re

from .errors import ParseError


FORMAT_ALIASES = {
    "mol": "mol_v2000",
    "mol_v2000": "mol_v2000",
    "sdf": "sdf",
    "mol2": "mol2",
    "pdb": "pdb",
}

MOLFILE_CHARGE_CODE_MAP = {
    0: 0,
    1: 3,
    2: 2,
    3: 1,
    5: -1,
    6: -2,
    7: -3,
}

ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mol":
        return "mol_v2000"
    if suffix == ".sdf":
        return "sdf"
    if suffix == ".mol2":
        return "mol2"
    if suffix in {".pdb", ".ent"}:
        return "pdb"
    raise ParseError(
        "unsupported_input_format",
        f"Unsupported input suffix {suffix!r}",
        path=str(path),
    )


def parse_path(path: Path, input_format: str | None = None) -> dict:
    normalized_format = _normalize_format(input_format) if input_format else detect_format(path)
    text = path.read_text(encoding="utf-8")
    if normalized_format == "mol_v2000":
        parsed = _parse_molfile_v2000(text, path)
    elif normalized_format == "sdf":
        parsed = _parse_sdf(text, path)
    elif normalized_format == "mol2":
        parsed = _parse_mol2(text, path)
    else:
        parsed = _parse_pdb(text, path)

    _ensure_single_component(parsed["atoms"], parsed["bonds"], path)
    parsed["input_format"] = normalized_format
    return parsed


def _normalize_format(input_format: str) -> str:
    try:
        return FORMAT_ALIASES[input_format.lower()]
    except KeyError as error:
        raise ParseError(
            "unsupported_input_format",
            f"Unsupported input format {input_format!r}",
        ) from error


def _parse_molfile_v2000(text: str, path: Path) -> dict:
    return _parse_mol_block(
        text.splitlines(),
        path=path,
        error_code="malformed_mol_v2000",
        allow_trailing_data=False,
    )


def _parse_sdf(text: str, path: Path) -> dict:
    records = []
    current = []
    for raw_line in text.splitlines():
        if raw_line.strip() == "$$$$":
            if current or not records:
                records.append(current)
                current = []
            continue
        current.append(raw_line)
    if current:
        records.append(current)

    nonempty_records = [record for record in records if any(line.strip() for line in record)]
    if len(nonempty_records) != 1:
        raise ParseError(
            "unsupported_multirecord_sdf",
            "SDF parser supports exactly one structure record",
            path=str(path),
        )
    return _parse_mol_block(
        nonempty_records[0],
        path=path,
        error_code="malformed_sdf",
        allow_trailing_data=True,
    )


def _parse_mol_block(
    lines: list[str],
    *,
    path: Path,
    error_code: str,
    allow_trailing_data: bool,
) -> dict:
    if len(lines) < 4:
        raise ParseError(error_code, "Molfile is too short", path=str(path))

    counts_tokens = lines[3].split()
    if len(counts_tokens) < 2 or not lines[3].rstrip().endswith("V2000"):
        raise ParseError(error_code, "Counts line must declare V2000", path=str(path), line=4)

    try:
        atom_count = int(counts_tokens[0])
        bond_count = int(counts_tokens[1])
    except ValueError as error:
        raise ParseError(error_code, "Counts line is malformed", path=str(path), line=4) from error

    if len(lines) < 4 + atom_count + bond_count:
        raise ParseError(error_code, "Molfile body is truncated", path=str(path))

    atoms = []
    for offset, raw_line in enumerate(lines[4 : 4 + atom_count], start=5):
        tokens = raw_line.split()
        if len(tokens) < 4:
            raise ParseError(error_code, "Atom line is malformed", path=str(path), line=offset)
        try:
            x_coord = float(tokens[0])
            y_coord = float(tokens[1])
            z_coord = float(tokens[2])
        except ValueError as error:
            raise ParseError(error_code, "Atom coordinates are malformed", path=str(path), line=offset) from error
        charge_code = 0
        if len(tokens) >= 6:
            try:
                charge_code = int(tokens[5])
            except ValueError as error:
                raise ParseError(error_code, "Atom charge code is malformed", path=str(path), line=offset) from error
        if charge_code not in MOLFILE_CHARGE_CODE_MAP:
            raise ParseError(error_code, "Unsupported Molfile atom charge code", path=str(path), line=offset)
        atoms.append(
            {
                "source_index": len(atoms) + 1,
                "source_atom_id": str(len(atoms) + 1),
                "element": tokens[3],
                "atom_name": None,
                "formal_charge": MOLFILE_CHARGE_CODE_MAP[charge_code],
                "partial_charge": None,
                "coordinates": [x_coord, y_coord, z_coord],
                "annotations": {},
                "provenance": {"lines": [offset]},
            }
        )

    bonds = []
    seen_pairs = set()
    for offset, raw_line in enumerate(
        lines[4 + atom_count : 4 + atom_count + bond_count],
        start=5 + atom_count,
    ):
        tokens = raw_line.split()
        if len(tokens) < 3:
            raise ParseError(error_code, "Bond line is malformed", path=str(path), line=offset)
        try:
            begin = int(tokens[0])
            end = int(tokens[1])
            order = int(tokens[2])
        except ValueError as error:
            raise ParseError(error_code, "Bond line contains non-integer fields", path=str(path), line=offset) from error
        if begin == end or not (1 <= begin <= atom_count) or not (1 <= end <= atom_count):
            raise ParseError(error_code, "Bond atom index is out of range", path=str(path), line=offset)
        pair = tuple(sorted((begin, end)))
        if pair in seen_pairs:
            raise ParseError(error_code, "Duplicate bond pair detected", path=str(path), line=offset)
        seen_pairs.add(pair)
        bonds.append(
            {
                "source_index": len(bonds) + 1,
                "source_bond_id": str(len(bonds) + 1),
                "atom_source_indices": [begin, end],
                "order": order,
                "bond_code": str(order),
                "annotations": {},
                "provenance": {"lines": [offset]},
            }
        )

    saw_end = False
    property_start = 4 + atom_count + bond_count
    for offset, raw_line in enumerate(lines[property_start:], start=property_start + 1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped == "M  END":
            saw_end = True
            trailing = lines[offset:]
            if not allow_trailing_data and any(line.strip() for line in trailing):
                raise ParseError(error_code, "Unexpected content after M  END", path=str(path), line=offset + 1)
            break
        if stripped.startswith("M  CHG"):
            _apply_molfile_charges(atoms, stripped, path, error_code, offset)
            continue
        raise ParseError(
            error_code,
            f"Unsupported Molfile property line {stripped!r}",
            path=str(path),
            line=offset,
        )

    if not saw_end:
        raise ParseError(error_code, "Molfile is missing M  END", path=str(path))

    return {
        "title": lines[0].strip() or path.stem,
        "source_metadata": {
            "title": lines[0].strip(),
            "program": lines[1].strip() if len(lines) > 1 else "",
            "comment": lines[2].strip() if len(lines) > 2 else "",
        },
        "atoms": atoms,
        "bonds": bonds,
    }


def _apply_molfile_charges(
    atoms: list[dict],
    stripped_line: str,
    path: Path,
    error_code: str,
    line_number: int,
) -> None:
    tokens = stripped_line.split()
    if len(tokens) < 4:
        raise ParseError(error_code, "M  CHG line is malformed", path=str(path), line=line_number)
    try:
        pair_count = int(tokens[2])
    except ValueError as error:
        raise ParseError(error_code, "M  CHG pair count is malformed", path=str(path), line=line_number) from error
    if len(tokens) != 3 + (2 * pair_count):
        raise ParseError(error_code, "M  CHG pair count does not match payload", path=str(path), line=line_number)
    for index in range(pair_count):
        try:
            atom_index = int(tokens[3 + (2 * index)])
            charge = int(tokens[4 + (2 * index)])
        except ValueError as error:
            raise ParseError(error_code, "M  CHG payload is malformed", path=str(path), line=line_number) from error
        if not (1 <= atom_index <= len(atoms)):
            raise ParseError(error_code, "M  CHG atom index is out of range", path=str(path), line=line_number)
        atoms[atom_index - 1]["formal_charge"] = charge


def _parse_mol2(text: str, path: Path) -> dict:
    sections: dict[str, list[str]] = {}
    current_name = None
    current_lines: list[str] = []
    for raw_line in text.splitlines():
        if raw_line.startswith("@<TRIPOS>"):
            if current_name is not None:
                sections[current_name] = current_lines
            current_name = raw_line[len("@<TRIPOS>") :].strip().upper()
            current_lines = []
            continue
        if current_name is not None:
            current_lines.append(raw_line)
    if current_name is not None:
        sections[current_name] = current_lines

    if "MOLECULE" not in sections or "ATOM" not in sections:
        raise ParseError("malformed_mol2", "MOL2 file must contain MOLECULE and ATOM sections", path=str(path))

    molecule_lines = [line for line in sections["MOLECULE"] if line.strip()]
    if len(molecule_lines) < 2:
        raise ParseError("malformed_mol2", "MOLECULE section is incomplete", path=str(path))

    try:
        atom_count, bond_count = (int(token) for token in molecule_lines[1].split()[:2])
    except (TypeError, ValueError) as error:
        raise ParseError("malformed_mol2", "MOLECULE counts line is malformed", path=str(path)) from error

    atoms = []
    atom_lines = [line for line in sections["ATOM"] if line.strip() and not line.lstrip().startswith("#")]
    if len(atom_lines) != atom_count:
        raise ParseError("malformed_mol2", "ATOM section count does not match MOLECULE header", path=str(path))

    atom_ids = set()
    atom_id_to_source = {}
    for offset, raw_line in enumerate(atom_lines, start=1):
        tokens = raw_line.split()
        if len(tokens) < 6:
            raise ParseError("malformed_mol2", "ATOM line is malformed", path=str(path), line=offset)
        atom_id = tokens[0]
        if atom_id in atom_ids:
            raise ParseError("malformed_mol2", "Duplicate atom id in ATOM section", path=str(path), line=offset)
        atom_ids.add(atom_id)
        element = _extract_mol2_element(tokens[5], tokens[1], path, offset)
        try:
            coordinates = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
        except ValueError as error:
            raise ParseError("malformed_mol2", "ATOM coordinates are malformed", path=str(path), line=offset) from error
        formal_charge, partial_charge = _parse_optional_mol2_charge(tokens[8] if len(tokens) >= 9 else None, path, offset)
        source_index = len(atoms) + 1
        atom_id_to_source[atom_id] = source_index
        atoms.append(
            {
                "source_index": source_index,
                "source_atom_id": atom_id,
                "element": element,
                "atom_name": tokens[1],
                "formal_charge": formal_charge,
                "partial_charge": partial_charge,
                "coordinates": coordinates,
                "annotations": {
                    "raw_atom_type": tokens[5],
                    "subst_id": tokens[6] if len(tokens) >= 7 else None,
                    "subst_name": tokens[7] if len(tokens) >= 8 else None,
                },
                "provenance": {"lines": [offset]},
            }
        )

    bonds = []
    bond_lines = [line for line in sections.get("BOND", []) if line.strip() and not line.lstrip().startswith("#")]
    if atom_count > 1 and len(bond_lines) != bond_count:
        raise ParseError("malformed_mol2", "BOND section count does not match MOLECULE header", path=str(path))

    seen_pairs = set()
    for offset, raw_line in enumerate(bond_lines, start=1):
        tokens = raw_line.split()
        if len(tokens) < 4:
            raise ParseError("malformed_mol2", "BOND line is malformed", path=str(path), line=offset)
        atom_a = atom_id_to_source.get(tokens[1])
        atom_b = atom_id_to_source.get(tokens[2])
        if atom_a is None or atom_b is None or atom_a == atom_b:
            raise ParseError("malformed_mol2", "BOND references invalid atom ids", path=str(path), line=offset)
        pair = tuple(sorted((atom_a, atom_b)))
        if pair in seen_pairs:
            raise ParseError("malformed_mol2", "Duplicate bond pair detected", path=str(path), line=offset)
        seen_pairs.add(pair)
        bonds.append(
            {
                "source_index": len(bonds) + 1,
                "source_bond_id": tokens[0],
                "atom_source_indices": [atom_a, atom_b],
                "order": _normalize_mol2_bond_order(tokens[3]),
                "bond_code": tokens[3],
                "annotations": {},
                "provenance": {"lines": [offset]},
            }
        )

    return {
        "title": molecule_lines[0].strip() or path.stem,
        "source_metadata": {
            "title": molecule_lines[0].strip(),
            "molecule_type": molecule_lines[2].strip() if len(molecule_lines) >= 3 else "",
            "charge_type": molecule_lines[3].strip() if len(molecule_lines) >= 4 else "",
        },
        "atoms": atoms,
        "bonds": bonds,
    }


def _extract_mol2_element(atom_type: str, atom_name: str, path: Path, line_number: int) -> str:
    match = ELEMENT_RE.match(atom_type)
    if match:
        return match.group(1)
    match = ELEMENT_RE.match(atom_name)
    if match:
        return match.group(1)
    raise ParseError("malformed_mol2", "Unable to extract element symbol from atom record", path=str(path), line=line_number)


def _parse_optional_mol2_charge(
    raw_charge: str | None,
    path: Path,
    line_number: int,
) -> tuple[int | None, float | None]:
    if raw_charge is None:
        return None, None
    try:
        charge = float(raw_charge)
    except ValueError as error:
        raise ParseError("malformed_mol2", "ATOM charge is malformed", path=str(path), line=line_number) from error
    rounded = round(charge)
    if abs(charge - rounded) < 1e-8:
        return int(rounded), charge
    return None, charge


def _normalize_mol2_bond_order(bond_code: str) -> int | None:
    if bond_code.isdigit():
        return int(bond_code)
    return None


def _parse_pdb(text: str, path: Path) -> dict:
    atoms = []
    serial_to_source = {}
    conect_lines = []
    title_lines = []
    model_count = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        record = raw_line[:6].strip()
        if record == "MODEL":
            model_count += 1
            if model_count > 1:
                raise ParseError(
                    "unsupported_pdb_multiple_models",
                    "PDB parser supports at most one model",
                    path=str(path),
                    line=line_number,
                )
            continue
        if record in {"TITLE", "COMPND"}:
            title_lines.append(raw_line[10:].strip())
            continue
        if record in {"ATOM", "HETATM"}:
            atom = _parse_pdb_atom_line(raw_line, path, line_number)
            if atom["source_atom_id"] in serial_to_source:
                raise ParseError(
                    "malformed_pdb",
                    "Duplicate atom serial number",
                    path=str(path),
                    line=line_number,
                )
            serial_to_source[atom["source_atom_id"]] = len(atoms) + 1
            atom["source_index"] = len(atoms) + 1
            atoms.append(atom)
            continue
        if record == "CONECT":
            conect_lines.append((line_number, raw_line))
            continue
        if record in {"END", "ENDMDL", "TER", "MASTER"} or not record:
            continue

    bonds = _parse_pdb_bonds(conect_lines, serial_to_source, path)
    if len(atoms) > 1 and not bonds:
        raise ParseError(
            "unsupported_pdb_missing_connectivity",
            "PDB parser requires explicit CONECT records for multi-atom inputs",
            path=str(path),
        )
    return {
        "title": title_lines[0] if title_lines else path.stem,
        "source_metadata": {"title": " ".join(title_lines).strip()},
        "atoms": atoms,
        "bonds": bonds,
    }


def _parse_pdb_atom_line(raw_line: str, path: Path, line_number: int) -> dict:
    if len(raw_line) < 54:
        raise ParseError("malformed_pdb", "ATOM/HETATM line is too short", path=str(path), line=line_number)

    altloc = raw_line[16].strip()
    if altloc:
        raise ParseError(
            "unsupported_pdb_altloc",
            "Alternate locations are not supported",
            path=str(path),
            line=line_number,
        )

    serial = raw_line[6:11].strip()
    if not serial:
        raise ParseError("malformed_pdb", "Missing atom serial number", path=str(path), line=line_number)
    try:
        int(serial)
    except ValueError as error:
        raise ParseError("malformed_pdb", "Atom serial number is malformed", path=str(path), line=line_number) from error

    element = raw_line[76:78].strip()
    if not element:
        raise ParseError(
            "unsupported_pdb_missing_element",
            "PDB parser requires explicit element columns",
            path=str(path),
            line=line_number,
        )

    try:
        x_coord = float(raw_line[30:38].strip())
        y_coord = float(raw_line[38:46].strip())
        z_coord = float(raw_line[46:54].strip())
    except ValueError as error:
        raise ParseError("malformed_pdb", "Atom coordinates are malformed", path=str(path), line=line_number) from error

    return {
        "source_atom_id": serial,
        "element": element,
        "atom_name": raw_line[12:16].strip() or None,
        "formal_charge": _parse_pdb_charge(raw_line[78:80].strip(), path, line_number),
        "partial_charge": None,
        "coordinates": [x_coord, y_coord, z_coord],
        "annotations": {
            "record_type": raw_line[:6].strip(),
            "residue_name": raw_line[17:20].strip() or None,
            "chain_id": raw_line[21].strip() or None,
            "residue_sequence": raw_line[22:26].strip() or None,
            "insertion_code": raw_line[26].strip() or None,
        },
        "provenance": {"lines": [line_number]},
    }


def _parse_pdb_charge(raw_charge: str, path: Path, line_number: int) -> int | None:
    if not raw_charge:
        return None
    if len(raw_charge) != 2 or raw_charge[0] not in "123456789" or raw_charge[1] not in "+-":
        raise ParseError("malformed_pdb", "PDB charge field is malformed", path=str(path), line=line_number)
    magnitude = int(raw_charge[0])
    return magnitude if raw_charge[1] == "+" else -magnitude


def _parse_pdb_bonds(
    conect_lines: list[tuple[int, str]],
    serial_to_source: dict[str, int],
    path: Path,
) -> list[dict]:
    pair_to_lines: dict[tuple[int, int], list[int]] = {}
    for line_number, raw_line in conect_lines:
        tokens = raw_line.split()
        if len(tokens) < 3:
            raise ParseError("malformed_pdb", "CONECT line is malformed", path=str(path), line=line_number)
        source_atom = serial_to_source.get(tokens[1])
        if source_atom is None:
            raise ParseError("malformed_pdb", "CONECT references unknown source atom", path=str(path), line=line_number)
        for token in tokens[2:]:
            target_atom = serial_to_source.get(token)
            if target_atom is None:
                raise ParseError("malformed_pdb", "CONECT references unknown target atom", path=str(path), line=line_number)
            if target_atom == source_atom:
                raise ParseError("malformed_pdb", "CONECT self-bonds are invalid", path=str(path), line=line_number)
            pair = tuple(sorted((source_atom, target_atom)))
            pair_to_lines.setdefault(pair, [])
            pair_to_lines[pair].append(line_number)

    bonds = []
    for pair in sorted(pair_to_lines):
        bonds.append(
            {
                "source_index": len(bonds) + 1,
                "source_bond_id": str(len(bonds) + 1),
                "atom_source_indices": [pair[0], pair[1]],
                "order": None,
                "bond_code": "conect",
                "annotations": {},
                "provenance": {"lines": sorted(set(pair_to_lines[pair]))},
            }
        )
    return bonds


def _ensure_single_component(atoms: list[dict], bonds: list[dict], path: Path) -> None:
    if not atoms:
        raise ParseError("invalid_input", "Input file does not contain any atoms", path=str(path))
    if len(atoms) == 1:
        return

    adjacency = {atom["source_index"]: set() for atom in atoms}
    for bond in bonds:
        left, right = bond["atom_source_indices"]
        adjacency[left].add(right)
        adjacency[right].add(left)

    first_atom = atoms[0]["source_index"]
    stack = [first_atom]
    visited = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(sorted(adjacency[current] - visited))

    if len(visited) != len(atoms):
        raise ParseError(
            "unsupported_multicomponent_input",
            "Parser layer supports exactly one connected component per input",
            path=str(path),
        )
