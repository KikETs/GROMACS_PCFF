from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / "testdata" / "lammps_golden"
OBSERVABLE_ORDER = (
    "single_point",
    "forces",
    "finite_difference",
    "nve_drift",
    "nvt_snapshot",
)


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


def enabled_observables(system_meta: dict) -> list[str]:
    enabled = []
    for observable in OBSERVABLE_ORDER:
        if system_meta["expected_observables"][observable]["enabled"]:
            enabled.append(observable)
    return enabled


def is_float_token(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def parse_thermo_table(path: Path, fields: list[str]) -> list[dict]:
    rows: list[dict] = []
    active = False
    expected = list(fields)

    aliases = {
        "step": "step",
        "ke": "kineng",
        "pe": "poteng",
        "ebond": "ebond",
        "eangle": "eangle",
        "edihed": "edihed",
        "eimp": "eimpro",
        "evdwl": "evdwl",
        "ecoul": "ecoul",
        "elong": "elong",
        "epair": "epair",
        "emol": "emol",
        "temp": "temp",
        "press": "press",
        "etotal": "toteng",
    }

    def canonicalize_header_token(token: str) -> str:
        normalized = "".join(ch.lower() for ch in token if ch.isalnum())
        return aliases.get(normalized, normalized)

    expected_canonical = [canonicalize_header_token(field) for field in expected]

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            tokens = stripped.split()
            if [canonicalize_header_token(token) for token in tokens] == expected_canonical:
                active = True
                continue

            if active:
                if len(tokens) == len(expected) and all(is_float_token(token) for token in tokens):
                    rows.append({field: float(value) for field, value in zip(expected, tokens)})
                    continue

                if stripped.startswith("Loop time of") or stripped.startswith("ERROR:"):
                    active = False
                    continue

    if not rows:
        raise ValueError(f"Could not find thermo table with fields {fields!r} in {path}")
    return rows


def parse_dump_custom(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]

    frames = []
    index = 0
    while index < len(lines):
        if lines[index] != "ITEM: TIMESTEP":
            index += 1
            continue

        timestep = int(lines[index + 1].strip())
        if lines[index + 2] != "ITEM: NUMBER OF ATOMS":
            raise ValueError(f"Malformed dump file near line {index + 3}: {path}")
        natoms = int(lines[index + 3].strip())
        if not lines[index + 4].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"Missing BOX BOUNDS section in {path}")
        if not lines[index + 8].startswith("ITEM: ATOMS "):
            raise ValueError(f"Missing ATOMS section in {path}")

        atom_fields = lines[index + 8].split()[2:]
        atom_start = index + 9
        atom_end = atom_start + natoms
        atoms = []
        for atom_line in lines[atom_start:atom_end]:
            values = atom_line.split()
            if len(values) != len(atom_fields):
                raise ValueError(f"Unexpected atom record width in {path}: {atom_line}")
            atom = {}
            for field, raw in zip(atom_fields, values):
                if field in {"id", "type"}:
                    atom[field] = int(raw)
                else:
                    atom[field] = float(raw)
            atoms.append(atom)
        atoms.sort(key=lambda atom: atom["id"])
        frames.append({"timestep": timestep, "fields": atom_fields, "atoms": atoms})
        index = atom_end

    if not frames:
        raise ValueError(f"No frames found in dump file {path}")
    return frames
