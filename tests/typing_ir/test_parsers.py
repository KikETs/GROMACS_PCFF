from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from typing_ir import ParseError, parse_file  # noqa: E402


CORPUS_ROOT = REPO_ROOT / "testdata" / "typing_golden"
DATA_ROOT = Path(__file__).resolve().parent / "data"


def structural_projection(ir: dict) -> dict:
    component = ir["components"][0]
    return {
        "atoms": [
            {
                "element": atom["element"],
                "formal_charge": atom["formal_charge"],
                "coordinates": atom["coordinates"],
            }
            for atom in component["atoms"]
        ],
        "bonds": [
            {
                "atom_indices": bond["atom_indices"],
                "order": bond["order"],
                "bond_code": bond["bond_code"],
            }
            for bond in component["bonds"]
        ],
    }


@pytest.mark.parametrize(
    ("fixture_name", "expected_format", "expected_bond_code", "expected_net_charge"),
    [
        ("ethane.mol2", "mol2", "1", 0),
        ("ethane.sdf", "sdf", "1", 0),
        ("ethane.pdb", "pdb", "conect", None),
    ],
)
def test_supported_format_fixtures_parse_into_ir(
    fixture_name: str,
    expected_format: str,
    expected_bond_code: str,
    expected_net_charge: int | None,
) -> None:
    path = DATA_ROOT / "supported" / fixture_name
    ir = parse_file(path, source_id=f"fixtures/{fixture_name}")
    component = ir["components"][0]

    assert ir["schema_name"] == "typed_system"
    assert ir["ir_stage"] == "parsed_only"
    assert ir["source"]["input_format"] == expected_format
    assert ir["typing"]["status"] == "not_run"
    assert component["atom_count"] == 8
    assert component["bond_count"] == 7
    assert component["element_counts"] == {"C": 2, "H": 6}
    assert component["net_formal_charge"] == expected_net_charge
    assert component["bond_code_histogram"] == {expected_bond_code: 7}

    if expected_format == "pdb":
        assert component["bond_order_histogram"] == {}
        assert all(atom["formal_charge"] is None for atom in component["atoms"])
    else:
        assert component["bond_order_histogram"] == {"1": 7}


def test_canonical_indexing_is_stable_for_reordered_molfile(tmp_path: Path) -> None:
    original_path = CORPUS_ROOT / "cases" / "ethane_neutral" / "inputs" / "structure.mol"
    reordered_path = tmp_path / "ethane_reordered.mol"
    reordered_path.write_text(
        "\n".join(
            [
                "ethane_reordered",
                "Codex PT1",
                "",
                "  8  7  0  0  0  0            999 V2000",
                "    2.0800   -0.9350    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0",
                "    2.0800    0.4670   -0.8090 H   0  0  0  0  0  0  0  0  0  0  0  0",
                "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
                "   -0.5400    0.9350    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0",
                "    1.5400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
                "   -0.5400   -0.4670    0.8090 H   0  0  0  0  0  0  0  0  0  0  0  0",
                "    2.0800    0.4670    0.8090 H   0  0  0  0  0  0  0  0  0  0  0  0",
                "   -0.5400   -0.4670   -0.8090 H   0  0  0  0  0  0  0  0  0  0  0  0",
                "  3  5  1  0  0  0  0",
                "  3  4  1  0  0  0  0",
                "  3  6  1  0  0  0  0",
                "  3  8  1  0  0  0  0",
                "  5  7  1  0  0  0  0",
                "  5  2  1  0  0  0  0",
                "  5  1  1  0  0  0  0",
                "M  END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    original_ir = parse_file(
        original_path,
        input_format="mol_v2000",
        source_id="original/ethane.mol",
    )
    reordered_ir = parse_file(
        reordered_path,
        input_format="mol_v2000",
        source_id="reordered/ethane.mol",
    )

    assert structural_projection(original_ir) == structural_projection(reordered_ir)


def test_pt0_golden_examples_match_parser_output() -> None:
    for case_dir in sorted((CORPUS_ROOT / "cases").iterdir()):
        if not case_dir.is_dir():
            continue
        input_path = case_dir / "inputs" / "structure.mol"
        expected_path = case_dir / "examples" / "typed_system.json"
        source_id = input_path.relative_to(REPO_ROOT).as_posix()
        actual = parse_file(input_path, input_format="mol_v2000", source_id=source_id)
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        assert actual == expected


@pytest.mark.parametrize(
    ("fixture_name", "expected_code", "expected_message"),
    [
        ("multi_record.sdf", "unsupported_multirecord_sdf", "exactly one structure record"),
        ("missing_bond_section.mol2", "malformed_mol2", "BOND section count"),
        ("no_conect.pdb", "unsupported_pdb_missing_connectivity", "requires explicit CONECT"),
    ],
)
def test_malformed_inputs_fail_explicitly(
    fixture_name: str,
    expected_code: str,
    expected_message: str,
) -> None:
    path = DATA_ROOT / "malformed" / fixture_name
    with pytest.raises(ParseError) as error_info:
        parse_file(path, source_id=f"malformed/{fixture_name}")

    assert error_info.value.code == expected_code
    assert expected_message in str(error_info.value)
