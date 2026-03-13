from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / "testdata" / "typing_golden"


class TypingGoldenError(ValueError):
    pass


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TypingGoldenError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_molfile_v2000(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    require(len(lines) >= 4, f"Molfile is too short: {path}")
    counts_tokens = lines[3].split()
    require(len(counts_tokens) >= 2, f"Malformed counts line in {path}")
    require(lines[3].rstrip().endswith("V2000"), f"Only Molfile V2000 is supported in {path}")

    atom_count = int(counts_tokens[0])
    bond_count = int(counts_tokens[1])
    require(len(lines) >= 4 + atom_count + bond_count, f"Molfile body is truncated: {path}")

    atoms = []
    for raw_line in lines[4 : 4 + atom_count]:
        tokens = raw_line.split()
        require(len(tokens) >= 4, f"Malformed atom line in {path}: {raw_line}")
        atoms.append(
            {
                "x": float(tokens[0]),
                "y": float(tokens[1]),
                "z": float(tokens[2]),
                "element": tokens[3],
                "formal_charge": 0,
            }
        )

    bonds = []
    bond_start = 4 + atom_count
    for raw_line in lines[bond_start : bond_start + bond_count]:
        tokens = raw_line.split()
        require(len(tokens) >= 3, f"Malformed bond line in {path}: {raw_line}")
        bonds.append(
            {
                "begin": int(tokens[0]),
                "end": int(tokens[1]),
                "order": int(tokens[2]),
            }
        )

    saw_end = False
    for raw_line in lines[bond_start + bond_count :]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped == "M  END":
            saw_end = True
            break
        if stripped.startswith("M  CHG"):
            tokens = stripped.split()
            require(len(tokens) >= 4, f"Malformed M  CHG line in {path}: {raw_line}")
            num_pairs = int(tokens[2])
            require(len(tokens) == 3 + (2 * num_pairs), f"Malformed M  CHG pair count in {path}: {raw_line}")
            for index in range(num_pairs):
                atom_index = int(tokens[3 + 2 * index])
                charge = int(tokens[4 + 2 * index])
                require(1 <= atom_index <= atom_count, f"M  CHG atom index out of range in {path}: {raw_line}")
                atoms[atom_index - 1]["formal_charge"] = charge
            continue
        require(
            False,
            f"Unsupported Molfile property line in {path}: {raw_line}",
        )

    require(saw_end, f"Molfile is missing M  END: {path}")

    element_counts: dict[str, int] = {}
    for atom in atoms:
        element_counts.setdefault(atom["element"], 0)
        element_counts[atom["element"]] += 1

    bond_order_histogram: dict[str, int] = {}
    for bond in bonds:
        order_key = str(bond["order"])
        bond_order_histogram.setdefault(order_key, 0)
        bond_order_histogram[order_key] += 1

    total_formal_charge = sum(atom["formal_charge"] for atom in atoms)
    return {
        "atom_count": atom_count,
        "bond_count": bond_count,
        "atoms": atoms,
        "bonds": bonds,
        "element_counts": dict(sorted(element_counts.items())),
        "bond_order_histogram": dict(sorted(bond_order_histogram.items())),
        "formal_charge_total": total_formal_charge,
    }


def case_dirs(root: Path | None = None) -> list[Path]:
    base = CORPUS_ROOT if root is None else root
    cases_root = base / "cases"
    require(cases_root.is_dir(), f"Missing cases directory under {base}")
    return sorted(path for path in cases_root.iterdir() if path.is_dir())


def validate_case_dir(case_dir: Path) -> dict:
    case_path = case_dir / "case.json"
    input_path = case_dir / "inputs" / "structure.mol"
    expected_path = case_dir / "expected" / "outcome.json"
    require(case_path.is_file(), f"Missing case.json in {case_dir}")
    require(input_path.is_file(), f"Missing inputs/structure.mol in {case_dir}")
    require(expected_path.is_file(), f"Missing expected/outcome.json in {case_dir}")

    case_meta = load_json(case_path)
    expected = load_json(expected_path)
    parsed_mol = parse_molfile_v2000(input_path)

    required_case_keys = {
        "schema_version",
        "id",
        "display_name",
        "status",
        "description",
        "input",
        "chemistry",
        "expected",
        "notes",
    }
    require(required_case_keys.issubset(case_meta), f"Missing required keys in {case_path}")
    require(case_meta["schema_version"] == 1, f"Unsupported case schema_version in {case_path}")
    require(case_meta["id"] == case_dir.name, f"Case id/path mismatch in {case_path}")
    require(case_meta["status"] in {"supported", "unsupported"}, f"Invalid status in {case_path}")
    require(case_meta["input"]["format"] == "mol_v2000", f"Unsupported input format in {case_path}")
    require(case_meta["input"]["path"] == "inputs/structure.mol", f"Unexpected input path in {case_path}")
    require(case_meta["expected"]["path"] == "expected/outcome.json", f"Unexpected expected path in {case_path}")

    chemistry = case_meta["chemistry"]
    require(chemistry["atom_count"] == parsed_mol["atom_count"], f"Atom count mismatch in {case_path}")
    require(chemistry["bond_count"] == parsed_mol["bond_count"], f"Bond count mismatch in {case_path}")
    require(
        chemistry["formal_charge_total"] == parsed_mol["formal_charge_total"],
        f"Formal charge mismatch in {case_path}",
    )
    require(
        chemistry["element_counts"] == parsed_mol["element_counts"],
        f"Element counts mismatch in {case_path}",
    )
    require(
        chemistry["bond_order_histogram"] == parsed_mol["bond_order_histogram"],
        f"Bond histogram mismatch in {case_path}",
    )

    require(expected["schema_version"] == 1, f"Unsupported outcome schema_version in {expected_path}")
    require(expected["case_id"] == case_meta["id"], f"Outcome case_id mismatch in {expected_path}")
    require(expected["status"] == case_meta["status"], f"Outcome status mismatch in {expected_path}")

    if case_meta["status"] == "supported":
        require("atom_type_family_expectations" in expected, f"Missing atom expectations in {expected_path}")
        require("expected_diagnostics" in expected, f"Missing expected_diagnostics in {expected_path}")
        require(
            len(expected["atom_type_family_expectations"]) == parsed_mol["atom_count"],
            f"Atom family count mismatch in {expected_path}",
        )
        for atom_expectation, parsed_atom in zip(expected["atom_type_family_expectations"], parsed_mol["atoms"]):
            require(
                atom_expectation["element"] == parsed_atom["element"],
                f"Element mismatch between outcome and input in {expected_path}",
            )
    else:
        require("failure_code" in expected, f"Missing failure_code in {expected_path}")
        require("diagnostic_substrings" in expected, f"Missing diagnostic_substrings in {expected_path}")

    return {
        "id": case_meta["id"],
        "display_name": case_meta["display_name"],
        "status": case_meta["status"],
        "description": case_meta["description"],
        "input_format": case_meta["input"]["format"],
        "path": f"cases/{case_dir.name}",
        "chemistry_tags": case_meta["chemistry"]["tags"],
        "atom_count": parsed_mol["atom_count"],
        "bond_count": parsed_mol["bond_count"],
        "formal_charge_total": parsed_mol["formal_charge_total"],
        "element_counts": parsed_mol["element_counts"],
        "bond_order_histogram": parsed_mol["bond_order_histogram"],
        "input_sha256": sha256_file(input_path),
        "expected_sha256": sha256_file(expected_path),
        "case_sha256": sha256_file(case_path),
        "expected_failure_code": expected.get("failure_code"),
    }


def build_manifest(root: Path | None = None) -> dict:
    base = CORPUS_ROOT if root is None else root
    cases = [validate_case_dir(case_dir) for case_dir in case_dirs(base)]
    unsupported_failure_codes = sorted(
        {case["expected_failure_code"] for case in cases if case["expected_failure_code"] is not None}
    )
    return {
        "schema_version": 1,
        "milestone": "PT0",
        "corpus_version": "pt0-v1",
        "supported_input_formats": ["mol_v2000"],
        "planned_output_formats": ["typed_system_json_v1", "typing_diagnostics_json_v1"],
        "structure": {
            "case_metadata": "cases/<id>/case.json",
            "input_structure": "cases/<id>/inputs/structure.mol",
            "expected_outcome": "cases/<id>/expected/outcome.json",
        },
        "unsupported_failure_codes": unsupported_failure_codes,
        "cases": cases,
    }


def stage_corpus(out_root: Path, root: Path | None = None, case_ids: list[str] | None = None) -> dict:
    base = CORPUS_ROOT if root is None else root
    manifest = build_manifest(base)
    selected = None if case_ids is None else set(case_ids)

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base / "README.md", out_root / "README.md")
    (out_root / "cases").mkdir(parents=True, exist_ok=True)

    staged_cases = []
    for case in manifest["cases"]:
        if selected is not None and case["id"] not in selected:
            continue
        shutil.copytree(base / case["path"], out_root / case["path"])
        staged_cases.append(case)

    staged_manifest = dict(manifest)
    staged_manifest["cases"] = staged_cases
    dump_json(out_root / "corpus_manifest.json", staged_manifest)
    return staged_manifest


def validate_manifest(root: Path | None = None) -> None:
    base = CORPUS_ROOT if root is None else root
    expected_manifest = load_json(base / "corpus_manifest.json")
    actual_manifest = build_manifest(base)
    if actual_manifest != expected_manifest:
        raise TypingGoldenError("typing_golden manifest is stale; regenerate it with tools/build_typing_golden/generate.py manifest")
