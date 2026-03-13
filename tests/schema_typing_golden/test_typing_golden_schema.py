from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / "testdata" / "typing_golden"
TOOL = REPO_ROOT / "tools" / "build_typing_golden" / "generate.py"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def relative_file_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def test_typing_corpus_manifest_references_existing_cases() -> None:
    manifest = load_json(CORPUS_ROOT / "corpus_manifest.json")
    assert manifest["schema_version"] == 1
    assert manifest["milestone"] == "PT0"
    assert manifest["supported_input_formats"] == ["mol_v2000"]
    assert len(manifest["cases"]) >= 4

    for case in manifest["cases"]:
        case_root = CORPUS_ROOT / case["path"]
        assert case_root.is_dir()
        assert (case_root / "case.json").is_file()
        assert (case_root / "inputs" / "structure.mol").is_file()
        assert (case_root / "expected" / "outcome.json").is_file()


def test_typing_case_metadata_and_expected_outcomes_are_complete() -> None:
    manifest = load_json(CORPUS_ROOT / "corpus_manifest.json")
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

    for record in manifest["cases"]:
        case_root = CORPUS_ROOT / record["path"]
        case_meta = load_json(case_root / "case.json")
        outcome = load_json(case_root / "expected" / "outcome.json")

        assert required_case_keys.issubset(case_meta)
        assert case_meta["status"] in {"supported", "unsupported"}
        assert case_meta["input"]["format"] == "mol_v2000"
        assert case_meta["expected"]["path"] == "expected/outcome.json"
        assert outcome["case_id"] == case_meta["id"]
        assert outcome["status"] == case_meta["status"]
        if case_meta["status"] == "supported":
            assert len(outcome["atom_type_family_expectations"]) == case_meta["chemistry"]["atom_count"]
        else:
            assert "failure_code" in outcome
            assert outcome["diagnostic_substrings"]


def test_typing_manifest_validation_passes() -> None:
    subprocess.run(
        [sys.executable, str(TOOL), "validate"],
        check=True,
        cwd=REPO_ROOT,
    )


def test_typing_stage_generation_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "stage_a"
    out_b = tmp_path / "stage_b"

    subprocess.run([sys.executable, str(TOOL), "stage", "--out", str(out_a)], check=True, cwd=REPO_ROOT)
    subprocess.run([sys.executable, str(TOOL), "stage", "--out", str(out_b)], check=True, cwd=REPO_ROOT)

    assert relative_file_hashes(out_a) == relative_file_hashes(out_b)
