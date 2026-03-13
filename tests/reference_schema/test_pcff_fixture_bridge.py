from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE = REPO_ROOT / "tools" / "pcff_fixture_bridge" / "generate.py"
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
