from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from chem_perception import dumps_report, loads_report, perceive_file  # noqa: E402


DATA_ROOT = Path(__file__).resolve().parent / "data"
CORPUS_ROOT = REPO_ROOT / "testdata" / "typing_golden" / "cases"


def atom(report: dict, atom_index: int) -> dict:
    return report["components"][0]["atoms"][atom_index - 1]


def find_atom(report: dict, *, element: str, predicate=None) -> dict:
    for candidate in report["components"][0]["atoms"]:
        if candidate["element"] != element:
            continue
        if predicate is None or predicate(candidate):
            return candidate
    raise AssertionError(f"Atom not found for element={element!r}")


def test_benzene_with_explicit_aromatic_mol2_bonds_is_aromatic() -> None:
    path = DATA_ROOT / "edge" / "benzene_aromatic.mol2"
    report = perceive_file(path, input_format="mol2", source_id="benzene_ar")

    ring = report["components"][0]["rings"][0]
    assert ring["aromaticity"]["status"] == "aromatic"
    assert ring["aromaticity"]["reason"] == "explicit_aromatic_bond_code_cycle"
    assert find_atom(report, element="C")["aromaticity"]["status"] == "aromatic"


def test_benzene_from_pdb_keeps_aromaticity_indeterminate_without_bond_orders() -> None:
    path = DATA_ROOT / "edge" / "benzene_ring.pdb"
    report = perceive_file(path, input_format="pdb", source_id="benzene_pdb")

    ring = report["components"][0]["rings"][0]
    assert ring["aromaticity"]["status"] == "indeterminate"
    assert ring["aromaticity"]["reason"] == "missing_bond_orders"
    for atom_index in range(1, 7):
        assert atom(report, atom_index)["aromaticity"]["status"] == "indeterminate"


def test_carboxyl_like_planar_center_candidates_are_detected_from_multiple_bonds() -> None:
    path = CORPUS_ROOT / "acetaldehyde_carbonyl_unsupported" / "inputs" / "structure.mol"
    report = perceive_file(path, input_format="mol_v2000", source_id="acetaldehyde")

    carbonyl_carbon = find_atom(
        report,
        element="C",
        predicate=lambda candidate: candidate["improper_center_candidate"]["is_candidate"],
    )
    methyl_carbon = find_atom(
        report,
        element="C",
        predicate=lambda candidate: candidate["improper_center_candidate"]["is_candidate"] is False
        and candidate["coordination"]["geometry_hint"] == "tetrahedral_candidate",
    )
    assert carbonyl_carbon["improper_center_candidate"]["is_candidate"] is True
    assert carbonyl_carbon["improper_center_candidate"]["kinds"] == ["planar_trigonal"]
    assert methyl_carbon["improper_center_candidate"]["is_candidate"] is False


def test_lithium_monatomic_environment_stays_well_defined() -> None:
    path = CORPUS_ROOT / "lithium_cation" / "inputs" / "structure.mol"
    report = perceive_file(path, input_format="mol_v2000", source_id="lithium")

    lithium = atom(report, 1)
    assert lithium["coordination"]["coordination_number"] == 0
    assert lithium["coordination"]["geometry_hint"] == "monatomic"
    assert lithium["valence"]["inferred_valence"] == 0


def test_perception_report_round_trip_is_exact() -> None:
    path = CORPUS_ROOT / "benzene_aromatic_unsupported" / "inputs" / "structure.mol"
    report = perceive_file(path, input_format="mol_v2000", source_id="benzene")
    assert loads_report(dumps_report(report)) == report
