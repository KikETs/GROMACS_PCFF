from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / "testdata" / "lammps_golden"
TOOLS_ROOT = REPO_ROOT / "tools" / "generate_lammps_golden"
REFERENCE_RESULTS_ROOT = REPO_ROOT / "tests" / "reference_results"
OBSERVABLES = ("single_point", "forces", "finite_difference", "nve_drift", "nvt_snapshot")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_data_file_counts(path: Path) -> tuple[dict, dict]:
    count_keywords = {"atoms", "bonds", "angles", "dihedrals", "impropers"}
    type_keywords = {"atom types", "bond types", "angle types", "dihedral types", "improper types"}
    section_names = {"Masses", "Atoms", "Bonds", "Angles", "Dihedrals", "Impropers"}
    header_counts = {key: 0 for key in count_keywords}
    type_counts = {key: 0 for key in type_keywords}
    section_counts = {key: 0 for key in section_names}

    current_section = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("LAMMPS data file"):
                continue

            tokens = stripped.split()
            if len(tokens) == 2 and tokens[1] in count_keywords:
                header_counts[tokens[1]] = int(tokens[0])
                current_section = None
                continue
            if len(tokens) == 3 and " ".join(tokens[1:]) in type_keywords:
                type_counts[" ".join(tokens[1:])] = int(tokens[0])
                current_section = None
                continue

            section_name = stripped.split(" #", 1)[0]
            if section_name in section_names:
                current_section = section_name
                continue

            if current_section in section_counts:
                section_counts[current_section] += 1

    return header_counts, section_counts


def relative_file_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def test_corpus_manifest_references_existing_systems() -> None:
    manifest = load_json(CORPUS_ROOT / "corpus_manifest.json")
    assert manifest["schema_version"] == 1
    assert len(manifest["systems"]) == 6

    for record in manifest["systems"]:
        system_dir = CORPUS_ROOT / record["path"]
        assert system_dir.is_dir()
        assert (system_dir / "system.json").is_file()
        assert (system_dir / "lammps" / "system.in").is_file()
        assert (system_dir / "lammps" / "system.data").is_file()


def test_system_metadata_is_complete() -> None:
    manifest = load_json(CORPUS_ROOT / "corpus_manifest.json")
    required_root_keys = {
        "schema_version",
        "id",
        "display_name",
        "description",
        "category",
        "reference_terms",
        "topology_source",
        "parameter_source",
        "styles",
        "execution",
        "expected_observables",
        "unresolved_items",
        "notes",
    }
    required_style_keys = {
        "units",
        "atom_style",
        "pair_style",
        "bond_style",
        "angle_style",
        "dihedral_style",
        "improper_style",
        "kspace_style",
        "special_bonds",
        "pair_coeff_source",
        "relies_on_mixing",
    }

    for record in manifest["systems"]:
        metadata = load_json(CORPUS_ROOT / record["path"] / "system.json")
        assert required_root_keys.issubset(metadata.keys())
        assert required_style_keys.issubset(metadata["styles"].keys())
        assert tuple(metadata["expected_observables"].keys()) == OBSERVABLES
        for observable in OBSERVABLES:
            assert "enabled" in metadata["expected_observables"][observable]
            assert "normalized_output" in metadata["expected_observables"][observable]


def test_lammps_data_files_match_declared_counts() -> None:
    manifest = load_json(CORPUS_ROOT / "corpus_manifest.json")
    expected_sections = {
        "atoms": "Atoms",
        "bonds": "Bonds",
        "angles": "Angles",
        "dihedrals": "Dihedrals",
        "impropers": "Impropers",
    }

    for record in manifest["systems"]:
        header_counts, section_counts = parse_data_file_counts(CORPUS_ROOT / record["path"] / "lammps" / "system.data")
        assert section_counts["Masses"] >= 1
        for count_key, section_name in expected_sections.items():
            assert section_counts[section_name] == header_counts[count_key]


def test_stage_generation_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "stage_a"
    out_b = tmp_path / "stage_b"
    generator = TOOLS_ROOT / "generate.py"

    subprocess.run([sys.executable, str(generator), "stage", "--out", str(out_a)], check=True, cwd=REPO_ROOT)
    subprocess.run([sys.executable, str(generator), "stage", "--out", str(out_b)], check=True, cwd=REPO_ROOT)

    assert relative_file_hashes(out_a) == relative_file_hashes(out_b)


def test_compare_harness_accepts_identical_normalized_payloads(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    candidate_dir = tmp_path / "candidate"
    generator = TOOLS_ROOT / "generate.py"
    comparator = TOOLS_ROOT / "compare.py"

    subprocess.run([sys.executable, str(generator), "stage", "--out", str(stage_dir)], check=True, cwd=REPO_ROOT)
    subprocess.run([sys.executable, str(generator), "stage", "--out", str(candidate_dir)], check=True, cwd=REPO_ROOT)

    normalized_single_point = {
        "schema_version": 1,
        "system_id": "bond_toy",
        "observable": "single_point",
        "units": {"energy": "kcal/mol"},
        "fields": {
            "step": 0.0,
            "pe": 1.0,
            "ebond": 1.0,
            "eangle": 0.0,
            "edihed": 0.0,
            "eimp": 0.0,
            "evdwl": 0.0,
            "ecoul": 0.0,
            "elong": 0.0,
            "epair": 0.0,
            "emol": 1.0
        }
    }
    normalized_forces = {
        "schema_version": 1,
        "system_id": "bond_toy",
        "observable": "forces",
        "units": {
            "distance": "angstrom",
            "force": "kcal/mol/angstrom",
            "charge": "e"
        },
        "frame": {
            "timestep": 0,
            "fields": ["id", "type", "q", "x", "y", "z", "fx", "fy", "fz"],
            "atoms": [
                {"id": 1, "type": 1, "q": 0.0, "x": -0.8, "y": 0.0, "z": 0.0, "fx": 0.5, "fy": 0.0, "fz": 0.0},
                {"id": 2, "type": 1, "q": 0.0, "x": 0.8, "y": 0.0, "z": 0.0, "fx": -0.5, "fy": 0.0, "fz": 0.0}
            ]
        }
    }
    normalized_finite_difference = {
        "schema_version": 1,
        "system_id": "bond_toy",
        "observable": "finite_difference",
        "units": {
            "energy": "kcal/mol",
            "distance": "angstrom",
            "force": "kcal/mol/angstrom"
        },
        "checks": [
            {"atom_id": 1, "component": "x", "delta": 1e-06, "analytic_force": 0.5, "finite_difference_force": 0.5, "residual": 0.0},
            {"atom_id": 2, "component": "x", "delta": 1e-06, "analytic_force": -0.5, "finite_difference_force": -0.5, "residual": 0.0}
        ]
    }

    for root in (stage_dir, candidate_dir):
        normalized_root = root / "bond_toy" / "normalized"
        normalized_root.mkdir(parents=True, exist_ok=True)
        for name, payload in (
            ("single_point.json", normalized_single_point),
            ("forces.json", normalized_forces),
            ("finite_difference.json", normalized_finite_difference),
        ):
            with (normalized_root / name).open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")

    subprocess.run(
        [
            sys.executable,
            str(comparator),
            "--golden",
            str(stage_dir),
            "--candidate",
            str(candidate_dir),
            "--system",
            "bond_toy",
        ],
        check=True,
        cwd=REPO_ROOT,
    )


def test_m4_reference_results_are_complete_and_match_metadata() -> None:
    expected_systems = {
        "small_oligomer": {
            "single_point_pe": 25.320359,
            "single_point_evdwl": -0.15323047,
            "single_point_ecoul": 9.4886598,
            "single_point_elong": -20.012769,
            "num_atoms": 6,
            "num_fd_checks": 6,
        },
        "small_salt_polymer_box": {
            "single_point_pe": 92.11412,
            "single_point_evdwl": -0.33788696,
            "single_point_ecoul": 6.3682874,
            "single_point_elong": -64.379354,
            "num_atoms": 10,
            "num_fd_checks": 6,
        },
    }

    for system_id, expected in expected_systems.items():
        metadata = load_json(CORPUS_ROOT / "systems" / system_id / "system.json")
        results_dir = REFERENCE_RESULTS_ROOT / "m4" / system_id
        assert results_dir.is_dir()
        assert (results_dir / "topol.top").is_file()
        assert (results_dir / "initial_nve.gro").is_file()

        observable_paths = {}
        for observable, observable_meta in metadata["expected_observables"].items():
            if not observable_meta["enabled"]:
                continue
            result_path = results_dir / observable_meta["normalized_output"]
            assert result_path.is_file()
            observable_paths[observable] = result_path

        single_point = load_json(observable_paths["single_point"])
        assert single_point["schema_version"] == 1
        assert single_point["system_id"] == system_id
        assert single_point["observable"] == "single_point"
        assert abs(single_point["fields"]["pe"] - expected["single_point_pe"]) < 1e-12
        assert abs(single_point["fields"]["evdwl"] - expected["single_point_evdwl"]) < 1e-12
        assert abs(single_point["fields"]["ecoul"] - expected["single_point_ecoul"]) < 1e-12
        assert abs(single_point["fields"]["elong"] - expected["single_point_elong"]) < 1e-12

        forces = load_json(observable_paths["forces"])
        assert forces["schema_version"] == 1
        assert forces["system_id"] == system_id
        assert forces["observable"] == "forces"
        assert forces["frame"]["timestep"] == 0
        assert len(forces["frame"]["atoms"]) == expected["num_atoms"]
        assert forces["frame"]["fields"] == metadata["expected_observables"]["forces"]["dump_fields"]

        finite_difference = load_json(observable_paths["finite_difference"])
        assert finite_difference["schema_version"] == 1
        assert finite_difference["system_id"] == system_id
        assert finite_difference["observable"] == "finite_difference"
        assert len(finite_difference["checks"]) == expected["num_fd_checks"]

        for observable in ("nve_drift", "nvt_snapshot"):
            trace_payload = load_json(observable_paths[observable])
            nsteps = metadata["expected_observables"][observable]["nsteps"]
            assert trace_payload["schema_version"] == 1
            assert trace_payload["system_id"] == system_id
            assert trace_payload["observable"] == observable
            assert len(trace_payload["trace"]) == nsteps + 1
            assert trace_payload["final_frame"]["timestep"] == nsteps
            assert len(trace_payload["final_frame"]["atoms"]) == expected["num_atoms"]
