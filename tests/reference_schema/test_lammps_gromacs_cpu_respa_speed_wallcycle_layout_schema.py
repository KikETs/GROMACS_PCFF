from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = (
    REPO_ROOT
    / "tests/reference_results/lammps_gromacs_cpu_respa_speed_protocol_20260424_wallcycle_layout_patch"
)
NSTLIST80_ROOT = (
    REPO_ROOT
    / "tests/reference_results/lammps_gromacs_cpu_respa_speed_protocol_20260424_wallcycle_layout_patch_nstlist80"
)


def test_speed_protocol_keeps_matched_hierarchy_and_runtime_flags() -> None:
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    contract = protocol["runtime_contract"]

    assert protocol["schema_name"] == "lammps_gromacs_cpu_respa_speed_protocol"
    assert contract["duration_ps"] >= 50.0
    assert contract["gromacs_exact_respa"] is True
    assert contract["lammps_run_style"] == "respa 2 4"
    assert contract["hierarchy_match"] is True
    assert contract["gromacs_ntmpi"] == 1
    assert contract["gromacs_ntomp"] == 12
    assert contract["gromacs_pin"] == "on"
    assert contract["gromacs_reprod"] is False
    assert contract["gromacs_native_multi_owner_mode"] == "default"
    assert contract["gromacs_nstlist"] == 20


def test_speed_verdict_does_not_overclaim_lammps_omp12() -> None:
    verdict = json.loads((ROOT / "verdict.json").read_text(encoding="utf-8"))

    assert verdict["schema_name"] == "lammps_gromacs_cpu_respa_speed_verdict"
    assert verdict["gromacs_runtime"]["ns_per_day"] > verdict["lammps_runtime"]["omp1_ns_per_day"]
    assert verdict["gromacs_runtime"]["ns_per_day"] < verdict["lammps_runtime"]["omp12_ns_per_day"]
    assert verdict["verdict"]["gromacs_vs_lammps_omp12"].startswith("GROMACS slower")

    forbidden = " ".join(verdict["verdict"]["not_claimable"]).lower()
    assert "generally faster" in forbidden
    assert "transport-readiness" in forbidden
    assert "gpu/hybrid" in forbidden


def test_speed_table_has_expected_passed_cases() -> None:
    with (ROOT / "cpu_speed_results.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    by_case = {row["case"]: row for row in rows}
    assert set(by_case) == {"gromacs_cpu_ntomp12", "lammps_omp1", "lammps_omp12"}
    for row in rows:
        assert row["status"] == "pass"
        assert float(row["outer_dt_fs"]) == 2.0
        assert float(row["ns_per_day"]) > 0.0


def test_nstlist80_speed_probe_is_bounded_and_still_slower_than_lammps_omp12() -> None:
    protocol = json.loads((NSTLIST80_ROOT / "protocol.json").read_text(encoding="utf-8"))
    verdict = json.loads((NSTLIST80_ROOT / "verdict.json").read_text(encoding="utf-8"))

    assert protocol["runtime_contract"]["gromacs_nstlist"] == 80
    assert protocol["runtime_contract"]["hierarchy_match"] is True
    assert verdict["gromacs_runtime"]["ns_per_day"] > verdict["lammps_runtime"]["omp1_ns_per_day"]
    assert verdict["gromacs_runtime"]["ns_per_day"] < verdict["lammps_runtime"]["omp12_ns_per_day"]
    assert "generally faster" in " ".join(verdict["verdict"]["not_claimable"]).lower()
