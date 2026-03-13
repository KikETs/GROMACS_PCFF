from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from typing_ir import dumps_ir, loads_ir, parse_file  # noqa: E402


CORPUS_ROOT = REPO_ROOT / "testdata" / "typing_golden"
DATA_ROOT = Path(__file__).resolve().parent / "data"


def relative_file_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def test_ir_json_round_trip_preserves_exact_payload() -> None:
    fixture_paths = [
        CORPUS_ROOT / "cases" / "ethane_neutral" / "inputs" / "structure.mol",
        DATA_ROOT / "supported" / "ethane.mol2",
        DATA_ROOT / "supported" / "ethane.sdf",
        DATA_ROOT / "supported" / "ethane.pdb",
    ]
    for path in fixture_paths:
        ir = parse_file(path, source_id=path.name)
        assert loads_ir(dumps_ir(ir)) == ir


def test_cli_export_typing_golden_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "typing_ir",
            "export-typing-golden",
            "--out-root",
            str(out_a),
        ],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "typing_ir",
            "export-typing-golden",
            "--out-root",
            str(out_b),
        ],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )

    assert relative_file_hashes(out_a) == relative_file_hashes(out_b)
