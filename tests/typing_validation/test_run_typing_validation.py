from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "tests" / "reference_results" / "pt8_typing_validation"


def test_validation_runner_regenerates_checked_in_reference(tmp_path: Path) -> None:
    out_dir = tmp_path / "pt8"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    subprocess.run(
        [
            sys.executable,
            "tools/run_typing_validation/generate.py",
            "generate",
            "--out",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    expected_files = {
        "failure_probe_summary.json",
        "lammps_smoke_parity_summary.json",
        "per_case_results.json",
        "validation_summary.json",
    }
    assert {path.name for path in out_dir.iterdir() if path.is_file()} == expected_files
    for filename in expected_files:
        assert (out_dir / filename).read_text(encoding="utf-8") == (REFERENCE_ROOT / filename).read_text(encoding="utf-8")


def test_validation_runner_validate_command_matches_reference() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_typing_validation/generate.py",
            "validate",
            "--reference",
            str(REFERENCE_ROOT),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["mismatches"] == []
