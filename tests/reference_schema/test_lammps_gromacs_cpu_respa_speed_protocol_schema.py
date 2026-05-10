from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "tests/reference_results/lammps_gromacs_cpu_respa_speed_protocol_20260424"
NOSPLIT_FACTOR4_ROOT = (
    REPO_ROOT / "tests/reference_results/lammps_gromacs_cpu_respa_speed_protocol_20260424_nosplit_factor4"
)
NOSPLIT_FACTOR10_ROOT = (
    REPO_ROOT / "tests/reference_results/lammps_gromacs_cpu_respa_speed_protocol_20260424_nosplit_factor10"
)
NOSPLIT_FACTOR10_NPME1_ROOT = (
    REPO_ROOT / "tests/reference_results/lammps_gromacs_cpu_respa_speed_protocol_20260424_nosplit_factor10_npme1"
)


def read_speed_rows(root: Path) -> dict[str, dict[str, str]]:
    with (root / "cpu_speed_results.tsv").open(encoding="utf-8") as handle:
        return {row["case"]: row for row in csv.DictReader(handle, delimiter="\t")}


def test_protocol_freezes_matched_respa_hierarchy() -> None:
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))

    assert protocol["schema_name"] == "lammps_gromacs_cpu_respa_speed_protocol"
    assert protocol["runtime_contract"]["duration_ps"] >= 50.0
    assert protocol["runtime_contract"]["gromacs_exact_respa"] is True
    assert protocol["runtime_contract"]["lammps_run_style"] == (
        f"respa 2 {protocol['runtime_contract']['gromacs_level3_factor']}"
    )
    assert protocol["runtime_contract"]["hierarchy_match"] is True

    gmx_outer = protocol["runtime_contract"]["gromacs_outer_dt_fs"]
    lmp_outer = protocol["runtime_contract"]["lammps_outer_dt_fs"]
    assert abs(gmx_outer - lmp_outer) < 1.0e-9


def test_lammps_input_uses_configured_respa_factor(tmp_path: Path) -> None:
    script = REPO_ROOT / "tools/pcff_respa_parity/benchmark_lammps_gromacs_cpu_respa.py"
    spec = importlib.util.spec_from_file_location("benchmark_lammps_gromacs_cpu_respa", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    bench_input = tmp_path / "bench.in"
    nsteps = module.write_lammps_input(
        bench_input,
        header_name="system.in",
        duration_ps=12.0,
        outer_dt_fs=3.0,
        level3_factor=6,
        temperature_k=353.0,
        tau_t_ps=0.2,
        thermo_every=100,
    )

    assert nsteps == 4000
    assert "run_style respa 2 6" in bench_input.read_text(encoding="utf-8")


def test_gromacs_input_can_disable_pair_splitting(tmp_path: Path) -> None:
    script = REPO_ROOT / "tools/pcff_respa_parity/benchmark_lammps_gromacs_cpu_respa.py"
    spec = importlib.util.spec_from_file_location("benchmark_lammps_gromacs_cpu_respa", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    bench_mdp = tmp_path / "bench.mdp"
    module.write_gmx_mdp(
        bench_mdp,
        duration_ps=12.0,
        base_dt_ps=0.0005,
        temperature_k=353.0,
        tau_t_ps=0.2,
        level3_factor=6,
        pair_splitting="none",
        nstlist=60,
        nstcalcenergy=1002,
        nstenergy=1002,
        nstlog=1002,
    )

    mdp_text = bench_mdp.read_text(encoding="utf-8")
    assert "exact-respa-pair-level  = 3" in mdp_text
    assert "exact-respa-inner-level" not in mdp_text
    assert "exact-respa-middle-level" not in mdp_text
    assert "exact-respa-outer-level" not in mdp_text


def test_cpu_speed_table_has_only_bounded_passed_cases() -> None:
    rows = list(read_speed_rows(ROOT).values())
    by_case = {row["case"]: row for row in rows}
    assert set(by_case) == {"gromacs_cpu_ntomp12", "lammps_omp1", "lammps_omp12"}

    for row in rows:
        assert row["status"] == "pass"
        assert float(row["ns_per_day"]) > 0.0
        assert float(row["outer_dt_fs"]) == 2.0


def test_protocol_keeps_speed_claim_non_transport() -> None:
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    forbidden = " ".join(protocol["claim_boundary"]["forbidden"]).lower()

    assert "transport-readiness" in forbidden
    assert "gpu" in forbidden
    assert "kokkos" in forbidden
    assert "ensemble-equivalence" in forbidden

    lammps_cases = [r for r in protocol["results"] if r["engine"] == "lammps"]
    assert lammps_cases
    for case in lammps_cases:
        assert case["lammps_acceleration"] == "OPENMP package via -sf omp -pk omp"


def test_nosplit_protocol_matches_simple_lammps_pair_kspace_layout() -> None:
    for root, factor in ((NOSPLIT_FACTOR4_ROOT, 4), (NOSPLIT_FACTOR10_ROOT, 10)):
        protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        contract = protocol["runtime_contract"]

        assert contract["gromacs_pair_splitting"] == "none"
        assert contract["gromacs_level3_factor"] == factor
        assert contract["lammps_run_style"] == f"respa 2 {factor}"
        assert contract["hierarchy_match"] is True


def test_nosplit_speed_artifacts_beat_lammps_omp12() -> None:
    factor4 = read_speed_rows(NOSPLIT_FACTOR4_ROOT)
    factor10 = read_speed_rows(NOSPLIT_FACTOR10_ROOT)
    factor10_npme = read_speed_rows(NOSPLIT_FACTOR10_NPME1_ROOT)

    assert float(factor4["gromacs_cpu_ntomp12"]["ns_per_day"]) > float(
        factor4["lammps_omp12"]["ns_per_day"]
    )
    assert float(factor10["gromacs_cpu_ntomp12"]["ns_per_day"]) > float(
        factor10["lammps_omp12"]["ns_per_day"]
    )
    assert float(factor10_npme["gromacs_cpu_ntomp6"]["ns_per_day"]) > float(
        factor10["gromacs_cpu_ntomp12"]["ns_per_day"]
    )
    assert float(factor10_npme["gromacs_cpu_ntomp6"]["ns_per_day"]) > float(
        factor10["lammps_omp12"]["ns_per_day"]
    )
