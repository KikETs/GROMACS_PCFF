from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.lunar_gromacs_pcff_converter.lunar_data_converter import _populate_molecule_templates

from tools.pcff_fixture_bridge.common import (
    BridgeError,
    generate_topological_one_four_pairs,
    parse_lammps_data,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE = REPO_ROOT / "tools" / "pcff_fixture_bridge" / "generate.py"
DATA_BRIDGE = REPO_ROOT / "tools" / "pcff_fixture_bridge" / "lammps_data_bridge.py"
CORPUS_ROOT = REPO_ROOT / "testdata" / "lammps_golden"
M4_ROOT = REPO_ROOT / "tests" / "reference_results" / "m4"


def relative_file_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def canonical_lines(path: Path) -> list[list[str]]:
    lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        lines.append(stripped.split())
    return lines


def tokens_match(actual: str, expected: str) -> bool:
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except ValueError:
        return actual == expected
    return abs(actual_value - expected_value) < 1e-12


def assert_topology_equivalent(actual: Path, expected: Path) -> None:
    actual_lines = canonical_lines(actual)
    expected_lines = canonical_lines(expected)
    assert len(actual_lines) == len(expected_lines)
    for actual_tokens, expected_tokens in zip(actual_lines, expected_lines):
        assert len(actual_tokens) == len(expected_tokens)
        for actual_token, expected_token in zip(actual_tokens, expected_tokens):
            assert tokens_match(actual_token, expected_token)


def run_bridge(out_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BRIDGE), "--out", str(out_root), *extra],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def build_minimal_corpus(tmp_path: Path, system_id: str, system_in_text: str) -> Path:
    corpus_root = tmp_path / "corpus"
    system_root = corpus_root / "systems" / system_id / "lammps"
    system_root.mkdir(parents=True, exist_ok=True)

    original_root = CORPUS_ROOT / "systems" / system_id
    (corpus_root / "systems" / system_id / "system.json").write_text(
        (original_root / "system.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (system_root / "system.data").write_text(
        (original_root / "lammps" / "system.data").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (system_root / "system.in").write_text(system_in_text, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "reference_engine": {"name": "LAMMPS", "scope": "test", "unit_style": "real", "primary_sources": []},
        "generator_contract": {"stage_layout_version": 1, "normalized_outputs": []},
        "systems": [{"id": system_id, "path": f"systems/{system_id}", "category": "oligomer", "enabled_observables": []}],
    }
    (corpus_root / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return corpus_root


def test_pcff_bridge_export_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    result_a = run_bridge(out_a, "--system", "small_oligomer", "--system", "small_salt_polymer_box", "export-gromacs")
    assert result_a.returncode == 0, result_a.stderr
    result_b = run_bridge(out_b, "--system", "small_oligomer", "--system", "small_salt_polymer_box", "export-gromacs")
    assert result_b.returncode == 0, result_b.stderr

    assert relative_file_hashes(out_a) == relative_file_hashes(out_b)


def test_pcff_bridge_emits_traceable_ir(tmp_path: Path) -> None:
    out_root = tmp_path / "bridge"
    result = run_bridge(out_root, "--system", "small_oligomer", "export-gromacs")
    assert result.returncode == 0, result.stderr

    typed_ir = json.loads((out_root / "small_oligomer" / "typed_system.json").read_text(encoding="utf-8"))
    first_atom_type = typed_ir["atom_types"][0]
    first_bond_type = typed_ir["bond_types"][0]
    first_angle_type = typed_ir["angle_types"][0]
    first_dihedral_type = typed_ir["dihedral_types"][0]
    first_pair = typed_ir["molecule_templates"][0]["generated_pairs"][0]

    assert first_atom_type["pair_coeff"]["source"]["file"] == "system.in"
    assert first_atom_type["mass_source"]["file"] == "system.data"
    assert first_bond_type["source"]["file"] == "system.in"
    assert first_angle_type["main"]["source"]["file"] == "system.in"
    assert first_dihedral_type["bb13"]["source"]["file"] == "system.in"
    assert first_pair["derived_from_dihedral_id"] == 1
    assert first_pair["source"]["file"] == "system.data"


def test_one_four_pairs_follow_shortest_bond_topology_not_dihedral_endpoints() -> None:
    source = {"file": "ring.data", "line": 1, "text": "test"}
    bonds = [
        {"id": 1, "atoms": [1, 2], "source": source},
        {"id": 2, "atoms": [2, 3], "source": source},
        {"id": 3, "atoms": [3, 4], "source": source},
        {"id": 4, "atoms": [4, 5], "source": source},
        {"id": 5, "atoms": [1, 4], "source": source},
    ]
    dihedrals = [{"id": 1, "atoms": [1, 2, 3, 4], "source": source}]

    pairs = generate_topological_one_four_pairs(bonds, dihedrals)

    assert [(pair["ai"], pair["aj"]) for pair in pairs] == [(2, 5)]
    assert pairs[0]["derived_from_dihedral_id"] is None
    assert pairs[0]["derived_from_bond_ids"] == [1, 5, 4]


def test_pcff_bridge_matches_representative_reference_topologies(tmp_path: Path) -> None:
    out_root = tmp_path / "bridge"
    result = run_bridge(out_root, "--system", "small_oligomer", "--system", "small_salt_polymer_box", "export-gromacs")
    assert result.returncode == 0, result.stderr

    assert_topology_equivalent(
        out_root / "small_oligomer" / "topol.top",
        M4_ROOT / "small_oligomer" / "topol.top",
    )
    assert_topology_equivalent(
        out_root / "small_salt_polymer_box" / "topol.top",
        M4_ROOT / "small_salt_polymer_box" / "topol.top",
    )


def test_pcff_bridge_fails_on_missing_class2_cross_term(tmp_path: Path) -> None:
    original_text = (CORPUS_ROOT / "systems" / "small_oligomer" / "lammps" / "system.in").read_text(encoding="utf-8")
    broken_text = "\n".join(
        line for line in original_text.splitlines() if not line.startswith("dihedral_coeff 1 bb13 ")
    ) + "\n"
    corpus_root = build_minimal_corpus(tmp_path, "small_oligomer", broken_text)

    out_root = tmp_path / "broken_out"
    result = subprocess.run(
        [
            sys.executable,
            str(BRIDGE),
            "--corpus-root",
            str(corpus_root),
            "--out",
            str(out_root),
            "--system",
            "small_oligomer",
            "export-gromacs",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Missing dihedral_coeff bb13 for dihedral type 1 in small_oligomer" in result.stderr


def test_lammps_data_bridge_imports_inline_lunar_style_coefficients(tmp_path: Path) -> None:
    data_path = tmp_path / "inline_pcff.data"
    data_path.write_text(
        """HEADER, inline LUNAR-like PCFF Class2 data

4 atoms
3 bonds
2 angles
1 dihedrals
0 impropers

2 atom types
1 bond types
1 angle types
1 dihedral types
0 improper types

-1.0 1.0 xlo xhi
-1.0 1.0 ylo yhi
-1.0 1.0 zlo zhi

Masses

1 12.011 # c2
2 14.007 # n2

Pair Coeffs  # lj/class2/coul/long

1 0.12 3.40 # c2
2 0.08 3.00 # n2

Bond Coeffs  # class2

1 1.48 230.0 -32.0 7.0 # c2 n2

Angle Coeffs  # class2

1 111.0 32.0 -3.6 1.0 # c2 n2 c2

BondBond Coeffs  # class2

1 5.0 1.48 1.48 # c2 n2 c2

BondAngle Coeffs  # class2

1 1.7 1.4 1.48 1.48 # c2 n2 c2

Dihedral Coeffs  # class2

1 0.9 0.0 0.5 180.0 0.3 0.0 # c2 n2 c2 n2

MiddleBondTorsion Coeffs  # class2

1 0.14 -0.09 0.05 1.48 # c2 n2 c2 n2

EndBondTorsion Coeffs  # class2

1 0.12 -0.06 0.03 0.10 -0.04 0.02 1.48 1.48 # c2 n2 c2 n2

AngleTorsion Coeffs  # class2

1 0.05 -0.03 0.02 0.04 -0.02 0.01 111.0 111.0 # c2 n2 c2 n2

AngleAngleTorsion Coeffs  # class2

1 0.22 111.0 111.0 # c2 n2 c2 n2

BondBond13 Coeffs  # class2

1 0.16 1.48 1.48 # c2 n2 c2 n2

Atoms # full

1 1 1 0.25 -0.5 0.0 0.0 1 -1 1
2 1 2 -0.25 -0.1 0.1 0.0
3 1 1 0.25 0.2 -0.1 0.0
4 1 2 -0.25 0.6 0.0 0.1

Bonds

1 1 1 2
2 1 2 3
3 1 3 4

Angles

1 1 1 2 3
2 1 2 3 4

Dihedrals

1 1 1 2 3 4
""",
        encoding="utf-8",
    )
    out_root = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(DATA_BRIDGE),
            "--data",
            str(data_path),
            "--out",
            str(out_root),
            "--system-id",
            "inline_pcff",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    typed_ir = json.loads((out_root / "typed_system.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_root / "bridge_manifest.json").read_text(encoding="utf-8"))
    topology = (out_root / "topol.top").read_text(encoding="utf-8")
    gro_lines = (out_root / "system.gro").read_text(encoding="utf-8").splitlines()

    assert typed_ir["atom_types"][0]["pair_coeff"]["source"]["file"] == "inline_pcff.data"
    assert typed_ir["angle_types"][0]["bb"]["source"]["file"] == "inline_pcff.data"
    assert typed_ir["dihedral_types"][0]["bb13"]["source"]["file"] == "inline_pcff.data"
    assert manifest["counts"]["atoms"] == 4
    assert "[ pairs ]" in topology
    assert " 13 " in topology
    assert gro_lines[1].strip() == "4"
    assert [float(value) for value in gro_lines[2].split()[-3:]] == pytest.approx(
        [0.25, -0.1, 0.3], abs=1.0e-12
    )
    assert [float(value) for value in gro_lines[-1].split()] == [0.2, 0.2, 0.2]


def test_lammps_data_bridge_rejects_triclinic_tilts_instead_of_dropping_them(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "triclinic.data"
    data_path.write_text(
        """LAMMPS data file

1 atoms
0 bonds
0 angles
0 dihedrals
0 impropers

1 atom types
0 bond types
0 angle types
0 dihedral types
0 improper types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi
1.0 0.5 -0.25 xy xz yz

Masses

1 12.0

Atoms # full

1 1 1 0.0 1.0 2.0 3.0 0 0 0
""",
        encoding="utf-8",
    )

    with pytest.raises(BridgeError, match="refusing to discard xy/xz/yz tilt factors"):
        parse_lammps_data(data_path)


def test_lunar_converter_uses_shortest_bond_topology_for_one_four_pairs() -> None:
    source = {"file": "ring.data", "line": 1, "text": "test"}
    atoms = [
        {
            "id": atom_id,
            "molecule_id": 1,
            "type_id": 1,
            "charge_e": 0.0,
            "x_angstrom": float(atom_id),
            "y_angstrom": 0.0,
            "z_angstrom": 0.0,
            "source": source,
        }
        for atom_id in range(1, 6)
    ]
    bonds = [
        {"id": 1, "type_id": 1, "atoms": [1, 2], "source": source},
        {"id": 2, "type_id": 1, "atoms": [2, 3], "source": source},
        {"id": 3, "type_id": 1, "atoms": [3, 4], "source": source},
        {"id": 4, "type_id": 1, "atoms": [4, 5], "source": source},
        {"id": 5, "type_id": 1, "atoms": [1, 4], "source": source},
    ]
    parsed_data = {
        "atoms": atoms,
        "bonds": bonds,
        "angles": [],
        "dihedrals": [{"id": 1, "type_id": 1, "atoms": [1, 2, 3, 4], "source": source}],
        "impropers": [],
    }
    typed_ir = {"molecule_templates": [], "molecule_instances": []}

    _populate_molecule_templates(typed_ir, parsed_data, {1: {"mass_amu": 12.0}})

    pairs = typed_ir["molecule_templates"][0]["generated_pairs"]
    assert [(pair["ai"], pair["aj"]) for pair in pairs] == [(2, 5)]
    assert pairs[0]["derived_from_dihedral_id"] is None
