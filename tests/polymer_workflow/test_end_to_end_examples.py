from __future__ import annotations

import hashlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from polymer_workflow import run_file  # noqa: E402


CASE_ROOT = REPO_ROOT / "testdata" / "polymer_workflow_golden" / "cases"
M5_CASE_ROOT = REPO_ROOT / "testdata" / "polymer_workflow_m5" / "cases"


def _relative_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.name == "polymer_workflow_report.json":
            continue
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def test_spe_examples_render_reproducibly(tmp_path: Path) -> None:
    for case_id in [
        "monoglyme_litfsi_1to1",
        "diglyme_litfsi_1to1",
        "triglyme_litfsi_2to2",
    ]:
        out_a = tmp_path / f"{case_id}_a"
        out_b = tmp_path / f"{case_id}_b"
        spec_path = CASE_ROOT / case_id / "spec.json"

        report_a = run_file(spec_path, out_dir=out_a)
        report_b = run_file(spec_path, out_dir=out_b)
        report_v = run_file(spec_path, out_dir=out_a, dry_run=True, validate_existing=True)

        assert report_a["workflow"]["status"] == "written"
        assert report_b["workflow"]["status"] == "written"
        assert report_v["workflow"]["existing_output_matches_rendered"] is True
        assert _relative_hashes(out_a) == _relative_hashes(out_b)


def test_topol_contains_expected_molecule_counts(tmp_path: Path) -> None:
    spec_path = CASE_ROOT / "triglyme_litfsi_2to2" / "spec.json"
    run_file(spec_path, out_dir=tmp_path)
    topol = (tmp_path / "topol.top").read_text(encoding="utf-8")

    assert '#include "molecule_triglyme.itp"' in topol
    assert '#include "molecule_li.itp"' in topol
    assert '#include "molecule_tfsi.itp"' in topol
    assert "TRIGLYME 1" in topol
    assert "LI 2" in topol
    assert "TFSI 2" in topol


def test_m5_neutral_additive_renders_reproducibly(tmp_path: Path) -> None:
    spec_path = M5_CASE_ROOT / "monoglyme_ethane_litfsi_1to1" / "spec.json"

    report = run_file(spec_path, out_dir=tmp_path)
    validate_report = run_file(spec_path, out_dir=tmp_path, dry_run=True, validate_existing=True)
    topol = (tmp_path / "topol.top").read_text(encoding="utf-8")

    neutral = next(component for component in report["components"] if component["role"] == "neutral_additive")
    fragment_check = report["assembly_checks"]["fragment_consistency"]

    assert report["workflow"]["status"] == "written"
    assert validate_report["workflow"]["existing_output_matches_rendered"] is True
    assert neutral["classification_family"] == "acyclic_alkane"
    assert neutral["exportable"] is True
    assert fragment_check["neutral_additive_component_ids"] == ["ETHANE"]
    assert '#include "molecule_ethane.itp"' in topol
    assert "ETHANE 1" in topol
    assert (tmp_path / "molecule_ethane.itp").is_file()
