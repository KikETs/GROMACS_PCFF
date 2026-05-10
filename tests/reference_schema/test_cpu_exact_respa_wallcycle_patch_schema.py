from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "tests/reference_results/cpu_exact_respa_wallcycle_patch_revalidation_20260424"


def _load_report(case: str) -> dict:
    return json.loads((ROOT / case / "report.json").read_text(encoding="utf-8"))


def test_wallcycle_patch_revalidation_keeps_exact_mechanics() -> None:
    for case in (
        "small_deterministic_ownerfallback_update",
        "medium_nvt_dense_salt_polymer_ownerfallback_update",
    ):
        report = _load_report(case)
        comparisons = report["comparisons"]

        assert comparisons["total_force"]["matches"] is True
        assert comparisons["total_force"]["max_abs_component_delta"] == 0.0
        assert comparisons["per_level_force"]["matches"] is True
        assert comparisons["per_level_force"]["max_abs_component_delta"] == 0.0
        assert comparisons["energy"]["matches"] is True
        assert comparisons["energy"]["max_abs_delta"] == 0.0
        assert comparisons["gro"]["matches"] is True
        assert comparisons["gro"]["max_abs_coord_delta_nm"] == 0.0

        continuation = comparisons["same_coordinate_probe"]["comparisons"]
        assert continuation["total_force"]["matches"] is True
        assert continuation["total_force"]["max_abs_component_delta"] == 0.0
        assert continuation["per_level_force"]["matches"] is True
        assert continuation["per_level_force"]["max_abs_component_delta"] == 0.0
        assert continuation["energy"]["matches"] is True
        assert continuation["energy"]["max_abs_delta"] == 0.0
        assert continuation["gro"]["matches"] is True


def test_wallcycle_patch_revalidation_keeps_restart_bitwise_boundary() -> None:
    for case in (
        "small_deterministic_ownerfallback_update",
        "medium_nvt_dense_salt_polymer_ownerfallback_update",
    ):
        report = _load_report(case)
        hashes = report["comparisons"]["hashes"]

        assert hashes["gro_sha256_equal"] is True
        assert hashes["edr_sha256_equal"] is True
        assert hashes["cpt_sha256_equal"] is False


def test_wallcycle_patch_report_tables_are_compact_and_passed() -> None:
    for report_tsv in ROOT.glob("*/report.tsv"):
        with report_tsv.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))

        assert len(rows) == 1
        row = rows[0]
        assert row["total_force_max_abs_component_delta"] == "0.0"
        assert row["per_level_force_max_abs_component_delta"] == "0.0"
        assert row["energy_max_abs_delta"] == "0.0"
        assert row["gro_max_abs_coord_delta_nm"] == "0.0"
