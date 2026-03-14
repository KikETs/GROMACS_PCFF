from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from atom_typing import type_file  # noqa: E402
from emitters.gromacs import (  # noqa: E402
    GromacsEmitterError,
    dumps_manifest,
    emit_file,
    emit_ir,
    loads_manifest,
)
from nonbonded_assignment import assign_file as assign_nonbonded_file  # noqa: E402
from parameter_assignment import assign_file as assign_bonded_file  # noqa: E402
from typing_ir import parse_file  # noqa: E402


CORPUS_ROOT = REPO_ROOT / "testdata" / "typing_golden" / "cases"


def _build_chain(case_id: str) -> tuple[dict, dict, dict, dict]:
    structure_path = CORPUS_ROOT / case_id / "inputs" / "structure.mol"
    ir = parse_file(structure_path, input_format="mol_v2000", source_id=case_id)
    typing_report = type_file(structure_path, input_format="mol_v2000", source_id=case_id)
    bonded_report = assign_bonded_file(structure_path, input_format="mol_v2000", source_id=case_id)
    nonbonded_report = assign_nonbonded_file(structure_path, input_format="mol_v2000", source_id=case_id)
    return ir, typing_report, bonded_report, nonbonded_report


def test_emit_file_dry_run_manifest_round_trips() -> None:
    structure_path = CORPUS_ROOT / "ethane_neutral" / "inputs" / "structure.mol"

    manifest = emit_file(
        structure_path,
        input_format="mol_v2000",
        source_id="ethane_neutral",
        dry_run=True,
    )
    round_tripped = loads_manifest(dumps_manifest(manifest))

    assert round_tripped == manifest
    assert manifest["emitter"]["status"] == "dry_run"
    assert manifest["emitter"]["dry_run"] is True
    assert manifest["emitter"]["existing_output_matches_rendered"] is None
    assert set(manifest["outputs"]) == {"forcefield_pcff.itp", "molecule.itp", "topol.top"}


def test_emit_file_writes_bundle_and_validate_existing(tmp_path: Path) -> None:
    structure_path = CORPUS_ROOT / "dimethyl_ether_neutral" / "inputs" / "structure.mol"

    written = emit_file(
        structure_path,
        input_format="mol_v2000",
        source_id="dimethyl_ether_neutral",
        out_dir=tmp_path,
    )
    validated = emit_file(
        structure_path,
        input_format="mol_v2000",
        source_id="dimethyl_ether_neutral",
        out_dir=tmp_path,
        dry_run=True,
        validate_existing=True,
    )

    assert written["emitter"]["status"] == "written"
    assert validated["emitter"]["existing_output_matches_rendered"] is True
    for filename in ("forcefield_pcff.itp", "molecule.itp", "topol.top"):
        assert (tmp_path / filename).is_file()


def test_validate_existing_mismatch_fails_explicitly(tmp_path: Path) -> None:
    structure_path = CORPUS_ROOT / "lithium_cation" / "inputs" / "structure.mol"
    emit_file(
        structure_path,
        input_format="mol_v2000",
        source_id="lithium_cation",
        out_dir=tmp_path,
    )
    topol_path = tmp_path / "topol.top"
    topol_path.write_text(topol_path.read_text(encoding="utf-8") + "; drift\n", encoding="utf-8")

    with pytest.raises(GromacsEmitterError) as excinfo:
        emit_file(
            structure_path,
            input_format="mol_v2000",
            source_id="lithium_cation",
            out_dir=tmp_path,
            dry_run=True,
            validate_existing=True,
        )

    assert excinfo.value.code == "rendered_output_mismatch"


def test_emit_ir_rejects_source_chain_mismatch() -> None:
    ir, typing_report, bonded_report, nonbonded_report = _build_chain("ethane_neutral")
    typing_report = copy.deepcopy(typing_report)
    typing_report["source"]["typed_ir_sha256"] = "0" * 64

    with pytest.raises(GromacsEmitterError) as excinfo:
        emit_ir(
            ir,
            typing_report=typing_report,
            bonded_report=bonded_report,
            nonbonded_report=nonbonded_report,
            dry_run=True,
        )

    assert excinfo.value.code == "source_chain_mismatch"


def test_emit_ir_rejects_unrepresentable_exclusion_policy() -> None:
    ir, typing_report, bonded_report, nonbonded_report = _build_chain("tfsi_anion_explicit")
    nonbonded_report = copy.deepcopy(nonbonded_report)
    nonbonded_report["components"][0]["exclusions"][0]["lj_scale"] = 0.5

    with pytest.raises(GromacsEmitterError) as excinfo:
        emit_ir(
            ir,
            typing_report=typing_report,
            bonded_report=bonded_report,
            nonbonded_report=nonbonded_report,
            dry_run=True,
        )

    assert excinfo.value.code == "unsupported_exclusion_scaling"


def test_emit_ir_rejects_unrepresentable_pair14_scaling() -> None:
    ir, typing_report, bonded_report, nonbonded_report = _build_chain("ethane_neutral")
    nonbonded_report = copy.deepcopy(nonbonded_report)
    nonbonded_report["components"][0]["pair14"][0]["coul_scale"] = 0.5

    with pytest.raises(GromacsEmitterError) as excinfo:
        emit_ir(
            ir,
            typing_report=typing_report,
            bonded_report=bonded_report,
            nonbonded_report=nonbonded_report,
            dry_run=True,
        )

    assert excinfo.value.code == "unsupported_pair14_scaling"
