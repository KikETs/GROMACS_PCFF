from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from atom_typing import type_file  # noqa: E402
from emitters.gromacs import render_bundle  # noqa: E402
from nonbonded_assignment import assign_file as assign_nonbonded_file  # noqa: E402
from parameter_assignment import assign_file as assign_bonded_file  # noqa: E402
from typing_ir import parse_file  # noqa: E402


CORPUS_ROOT = REPO_ROOT / "testdata" / "typing_golden" / "cases"
SNAPSHOT_ROOT = Path(__file__).resolve().parent / "snapshots" / "gromacs"
SUPPORTED_CASES = [
    "ethane_neutral",
    "dimethyl_ether_neutral",
    "lithium_cation",
    "tfsi_anion_explicit",
]


def _render_case(case_id: str) -> dict[str, str]:
    structure_path = CORPUS_ROOT / case_id / "inputs" / "structure.mol"
    ir = parse_file(structure_path, input_format="mol_v2000", source_id=case_id)
    typing_report = type_file(structure_path, input_format="mol_v2000", source_id=case_id)
    bonded_report = assign_bonded_file(structure_path, input_format="mol_v2000", source_id=case_id)
    nonbonded_report = assign_nonbonded_file(structure_path, input_format="mol_v2000", source_id=case_id)
    return render_bundle(
        ir,
        typing_report=typing_report,
        bonded_report=bonded_report,
        nonbonded_report=nonbonded_report,
    )


def test_supported_cases_match_committed_gromacs_snapshots() -> None:
    for case_id in SUPPORTED_CASES:
        bundle = _render_case(case_id)
        case_root = SNAPSHOT_ROOT / case_id
        for filename, text in bundle.items():
            expected_path = case_root / filename
            assert expected_path.read_text(encoding="utf-8") == text
