from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from chem_perception import perceive_file, query_neighbor_shell  # noqa: E402


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


def test_ethane_perception_is_deterministic_and_queryable() -> None:
    path = CORPUS_ROOT / "ethane_neutral" / "inputs" / "structure.mol"
    report_a = perceive_file(path, input_format="mol_v2000", source_id="ethane_a")
    report_b = perceive_file(path, input_format="mol_v2000", source_id="ethane_b")

    assert report_a["components"][0]["atoms"] == report_b["components"][0]["atoms"]
    carbon = atom(report_a, 1)
    assert carbon["valence"]["status"] == "exact"
    assert carbon["valence"]["inferred_valence"] == 4
    assert carbon["coordination"]["geometry_hint"] == "tetrahedral_candidate"

    shell_1 = query_neighbor_shell(report_a, 1, depth=1)
    shell_2 = query_neighbor_shell(report_a, 1, depth=2)
    assert shell_1["element_counts"] == {"C": 1, "H": 3}
    assert shell_2["element_counts"] == {"H": 3}


def test_dimethyl_ether_neighbor_shells_capture_local_environment() -> None:
    path = CORPUS_ROOT / "dimethyl_ether_neutral" / "inputs" / "structure.mol"
    report = perceive_file(path, input_format="mol_v2000", source_id="dme")

    oxygen = find_atom(report, element="O")
    assert oxygen["coordination"]["coordination_number"] == 2
    assert oxygen["neighbor_element_counts"] == {"C": 2}
    assert query_neighbor_shell(report, oxygen["canonical_index"], depth=1)["element_counts"] == {"C": 2}
    assert query_neighbor_shell(report, oxygen["canonical_index"], depth=2)["element_counts"] == {"H": 6}


def test_tfsi_detects_hypervalent_sulfur_and_planar_improper_candidates() -> None:
    path = CORPUS_ROOT / "tfsi_anion_explicit" / "inputs" / "structure.mol"
    report = perceive_file(path, input_format="mol_v2000", source_id="tfsi")

    sulfur = find_atom(report, element="S", predicate=lambda candidate: candidate["valence"]["inferred_valence"] == 6)
    nitrogen = find_atom(report, element="N")
    assert sulfur["valence"]["inferred_valence"] == 6
    assert sulfur["coordination"]["geometry_hint"] == "tetrahedral_hypervalent_candidate"
    assert nitrogen["valence"]["inferred_valence"] == 2
    assert nitrogen["improper_center_candidate"]["is_candidate"] is False


def test_cyclohexane_is_ring_but_not_aromatic() -> None:
    path = DATA_ROOT / "toy" / "cyclohexane.mol"
    report = perceive_file(path, input_format="mol_v2000", source_id="cyclohexane")

    ring = report["components"][0]["rings"][0]
    assert ring["size"] == 6
    assert ring["aromaticity"]["status"] == "non_aromatic"
    for atom_index in range(1, 7):
        carbon = atom(report, atom_index)
        assert carbon["ring"]["smallest_ring_size"] == 6
        assert carbon["aromaticity"]["status"] == "non_aromatic"


def test_pyridine_kekule_ring_is_aromatic() -> None:
    path = DATA_ROOT / "toy" / "pyridine_aromatic.mol"
    report = perceive_file(path, input_format="mol_v2000", source_id="pyridine")

    ring = report["components"][0]["rings"][0]
    assert ring["aromaticity"]["status"] == "aromatic"
    assert ring["aromaticity"]["electron_count"] == 6
    assert find_atom(report, element="N")["aromaticity"]["status"] == "aromatic"


def test_polymer_placeholders_generate_connection_tags() -> None:
    path = DATA_ROOT / "toy" / "polymer_repeat_du.mol"
    report = perceive_file(path, input_format="mol_v2000", source_id="poly_du")

    placeholders = [
        candidate
        for candidate in report["components"][0]["atoms"]
        if candidate["polymer_connection"]["is_placeholder"]
    ]
    targets = [
        candidate
        for candidate in report["components"][0]["atoms"]
        if candidate["element"] == "C" and candidate["polymer_connection"]["tags"] == ["Du"]
    ]
    assert len(placeholders) == 2
    assert len(targets) == 2
    assert len(report["components"][0]["polymer_connection_points"]) == 2
