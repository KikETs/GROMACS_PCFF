from __future__ import annotations

from pathlib import Path

import pytest

from tools.pcff_respa_parity.polygen_multisystem_worker import (
    GMX_PCFF_RUNTIME_ENV,
    _assert_gromacs_beta_coverage,
    _assert_sibling_lammps_lane_ok,
    _first_lammps_g_vector,
    _lammps_production_log_endpoint,
    _lammps_log_gromacs_stage_keys,
    _should_outer_skip_gromacs_equilibration,
    _should_outer_skip_gromacs_production,
    bridge_script,
    lammps_beta_stage_layouts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_g_vector(log_path: Path, value: float) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"LAMMPS setup\nG vector (1/distance) = {value:.8f}\n",
        encoding="utf-8",
    )


def _write_lammps_production_log(
    log_path: Path,
    *,
    endpoint_fs: float,
    normal_exit: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Per MPI rank memory allocation = 1 Mbytes",
        "   Step         v_time         Press           Temp",
        "         0   1e-06          1.0             353.0",
        f"  10000000   {endpoint_fs:.9g}          1.0             353.0",
        "Loop time of 10 on 12 procs for 10000000 steps",
    ]
    if normal_exit:
        lines.append("Total wall time: 10:00:00")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_restart(lmp_proj: Path) -> None:
    restart = lmp_proj / "MD/final.restart"
    restart.parent.mkdir(parents=True, exist_ok=True)
    restart.write_bytes(b"LAMMPS restart\n")


def test_worker_uses_the_bridge_versioned_in_its_own_checkout(tmp_path: Path) -> None:
    expected = REPO_ROOT / "tools/pcff_fixture_bridge/lammps_data_bridge.py"

    assert bridge_script(tmp_path) == expected
    assert expected.is_file()
    assert "GROMACS_PCFF-lunar-data-bridge" not in str(bridge_script(tmp_path))


def test_runtime_uses_no_dd_but_keeps_pair_grid_coordinates_wrapped() -> None:
    env = dict(
        item.split("=", 1)
        for item in GMX_PCFF_RUNTIME_ENV.split(";")
        if item
    )

    assert env["GMX_DD_SINGLE_RANK"] == "0"
    assert env["GMX_PCFF_EXACT_RESPA_IMAGE_HANDOFF"] == "1"
    assert "GMX_PCFF_LAMMPS_CG_EM_SKIP_PUT_ATOMS_IN_BOX" not in env


@pytest.mark.parametrize(
    ("lineage_state", "expected"),
    [
        (None, False),
        ("active_equil", False),
        ("production_pending", False),
    ],
)
def test_worker_always_reenters_equilibration_for_protocol_validation(
    lineage_state: str | None, expected: bool
) -> None:
    assert (
        _should_outer_skip_gromacs_equilibration(
            resume_existing=True,
            force_restart=False,
            completion_flag_exists=True,
            lineage_rebuild_state=lineage_state,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("lineage_state", "expected"),
    [
        (None, True),
        ("active_equil", False),
        ("production_pending", False),
    ],
)
def test_worker_never_outer_skips_production_while_lineage_rebuild_is_pending(
    lineage_state: str | None, expected: bool
) -> None:
    assert (
        _should_outer_skip_gromacs_production(
            resume_existing=True,
            force_restart=False,
            completion_flag_exists=True,
            lineage_rebuild_state=lineage_state,
        )
        is expected
    )


def test_lammps_log_names_map_to_expanded_gromacs_stage_keys() -> None:
    expected = {
        "lammps_lammps_equil_01_eq01_nvt_0p5fs_50ps_chunk0001_cpu.log": (
            "eq01_soft_langevin_10ps",
            "eq01_nvt_40ps",
        ),
        "lammps_lammps_equil_02_eq02_npt_0p5fs_100ps_chunk0002_cpu.log": (
            "eq02_npt_compress_100ps_chunk0002",
        ),
        "lammps_lammps_equil_04_eq04_npt_compress_500ps_chunk0003_cpu.log": (
            "eq04_npt_compress_500ps_chunk0003",
        ),
        "lammps_lammps_equil_12_npt_avg_cell_1200ps_cpu.log": (
            "eq12_npt_1200ps",
        ),
        "lammps_prod_stage00_nvt_cpu.log": ("prod_nvt",),
    }
    for log_name, stage_keys in expected.items():
        assert _lammps_log_gromacs_stage_keys(Path(log_name)) == stage_keys

    assert not _lammps_log_gromacs_stage_keys(
        Path("lammps_lammps_equil_09_eq09_npt_compress_300ps_chunk0001_retry1_cpu.log")
    )


def test_lammps_beta_stage_layouts_are_trajectory_local_and_preserve_explicit_values(
    tmp_path: Path,
) -> None:
    trajectory_a = tmp_path / "Traj_1474101"
    trajectory_b = tmp_path / "Traj_1474102"
    _write_g_vector(
        trajectory_a
        / "logs/lammps_lammps_equil_02_eq02_npt_0p5fs_100ps_chunk0001_cpu.log",
        0.23784641,
    )
    _write_g_vector(
        trajectory_b
        / "logs/lammps_lammps_equil_02_eq02_npt_0p5fs_100ps_chunk0001_cpu.log",
        0.23123456,
    )

    base = {
        "default": {"ntomp": 12},
        "eq02_npt_compress_100ps_chunk0001": {
            "env": {"GMX_PCFF_EWALD_BETA_INV_A": "0.19000000"}
        },
    }
    explicit = lammps_beta_stage_layouts(trajectory_a, base)
    assert (
        explicit["eq02_npt_compress_100ps_chunk0001"]["env"][
            "GMX_PCFF_EWALD_BETA_INV_A"
        ]
        == "0.19000000"
    )
    assert base["eq02_npt_compress_100ps_chunk0001"]["env"]["GMX_PCFF_EWALD_BETA_INV_A"] == "0.19000000"

    derived_a = lammps_beta_stage_layouts(trajectory_a, {"default": {"ntomp": 12}})
    derived_b = lammps_beta_stage_layouts(trajectory_b, {"default": {"ntomp": 12}})
    assert (
        derived_a["eq02_npt_compress_100ps_chunk0001"]["env"][
            "GMX_PCFF_EWALD_BETA_INV_A"
        ]
        == "0.23784641"
    )
    assert (
        derived_b["eq02_npt_compress_100ps_chunk0001"]["env"][
            "GMX_PCFF_EWALD_BETA_INV_A"
        ]
        == "0.23123456"
    )
    assert _first_lammps_g_vector(
        trajectory_a
        / "logs/lammps_lammps_equil_02_eq02_npt_0p5fs_100ps_chunk0001_cpu.log"
    ) == 0.23784641


def test_completed_trajectory_can_bridge_before_active_batch_writes_status(
    tmp_path: Path,
) -> None:
    lmp_proj = tmp_path / "lammps_cpu/Traj_1475301"
    _write_final_restart(lmp_proj)
    production_log = lmp_proj / "logs/lammps_prod_stage00_nvt_cpu.log"
    _write_lammps_production_log(
        production_log,
        endpoint_fs=20_000_000.0,
        normal_exit=True,
    )

    assert _lammps_production_log_endpoint(production_log) == (
        20_000_000.0,
        True,
    )
    _assert_sibling_lammps_lane_ok(
        lmp_proj,
        1475301,
        expected_production_ns=20.0,
    )


def test_partial_trajectory_without_batch_status_fails_closed(
    tmp_path: Path,
) -> None:
    lmp_proj = tmp_path / "lammps_cpu/Traj_1475302"
    _write_final_restart(lmp_proj)
    _write_lammps_production_log(
        lmp_proj / "logs/lammps_prod_stage00_nvt_cpu.log",
        endpoint_fs=12_480_000.0,
        normal_exit=False,
    )

    with pytest.raises(RuntimeError, match="no terminal evidence for 20 ns"):
        _assert_sibling_lammps_lane_ok(
            lmp_proj,
            1475302,
            expected_production_ns=20.0,
        )


def test_partial_stage_log_is_not_masked_by_stale_aggregate_log(
    tmp_path: Path,
) -> None:
    lmp_proj = tmp_path / "lammps_cpu/Traj_1475302"
    _write_final_restart(lmp_proj)
    _write_lammps_production_log(
        lmp_proj / "logs/lammps_prod_stage00_nvt_cpu.log",
        endpoint_fs=12_480_000.0,
        normal_exit=False,
    )
    _write_lammps_production_log(
        lmp_proj / "MD/log.lammps",
        endpoint_fs=20_000_000.0,
        normal_exit=True,
    )

    with pytest.raises(RuntimeError, match="endpoint_fs=12480000.0"):
        _assert_sibling_lammps_lane_ok(
            lmp_proj,
            1475302,
            expected_production_ns=20.0,
        )


def test_status_ok_alone_is_not_lammps_completion_evidence(tmp_path: Path) -> None:
    lane_root = tmp_path / "lammps_cpu"
    lmp_proj = lane_root / "Traj_1474101"
    _write_final_restart(lmp_proj)
    (lane_root / "batch_status.csv").write_text(
        "Trajectory ID,status,failed_phase,error\n1474101,ok,,\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="no terminal evidence for 20 ns"):
        _assert_sibling_lammps_lane_ok(
            lmp_proj,
            1474101,
            expected_production_ns=20.0,
        )


def test_completion_flag_requires_matching_persisted_duration(tmp_path: Path) -> None:
    lmp_proj = tmp_path / "lammps_cpu/Traj_1474101"
    _write_final_restart(lmp_proj)
    flag = lmp_proj / "MD/production_complete.flag"
    flag.write_text("done\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no terminal evidence for 20 ns"):
        _assert_sibling_lammps_lane_ok(
            lmp_proj,
            1474101,
            expected_production_ns=20.0,
        )

    (lmp_proj / "MD/meta.json").write_text(
        '{"production_total_ns": 20.0}\n',
        encoding="utf-8",
    )
    _assert_sibling_lammps_lane_ok(
        lmp_proj,
        1474101,
        expected_production_ns=20.0,
    )


def test_non_ok_status_rejects_even_terminal_lammps_artifacts(tmp_path: Path) -> None:
    lane_root = tmp_path / "lammps_cpu"
    lmp_proj = lane_root / "Traj_1474101"
    _write_final_restart(lmp_proj)
    _write_lammps_production_log(
        lmp_proj / "logs/lammps_prod_stage00_nvt_cpu.log",
        endpoint_fs=20_000_000.0,
        normal_exit=True,
    )
    (lane_root / "batch_status.csv").write_text(
        "Trajectory ID,status,failed_phase,error\n"
        "1474101,failed,lammps_prod,return code 1\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="sibling LAMMPS lane is not ok"):
        _assert_sibling_lammps_lane_ok(
            lmp_proj,
            1474101,
            expected_production_ns=20.0,
        )


def test_beta_coverage_rejects_any_generated_stage_without_value() -> None:
    runtime_ctx = {
        "schedule": "polygen_em_handoff",
        "stages": [
            {"name": "eq03_pre_2fs_minimize", "kind": "em"},
            {"name": "prod_nvt", "kind": "md"},
        ],
        "gromacs_stage_layouts": {
            "eq03_pre_2fs_minimize": {
                "env": {"GMX_PCFF_EWALD_BETA_INV_A": "0.23887529"}
            }
        },
    }

    with pytest.raises(RuntimeError, match="missing=prod_nvt"):
        _assert_gromacs_beta_coverage(runtime_ctx)


def test_explicit_default_beta_covers_every_generated_stage() -> None:
    runtime_ctx = {
        "schedule": "polygen_em_handoff",
        "stages": [
            {"name": "eq03_pre_2fs_minimize", "kind": "em"},
            {"name": "prod_nvt", "kind": "md"},
        ],
        "gromacs_stage_layouts": {
            "default": {
                "env": {"GMX_PCFF_EWALD_BETA_INV_A": "0.23469998"}
            }
        },
    }

    _assert_gromacs_beta_coverage(runtime_ctx)


def test_explicit_default_beta_is_materialized_into_shadowing_stage_layout(
    tmp_path: Path,
) -> None:
    layouts = lammps_beta_stage_layouts(
        tmp_path / "Traj_1474101",
        {
            "default": {
                "env": {"GMX_PCFF_EWALD_BETA_INV_A": "0.23469998"}
            },
            "prod_nvt": {"ntomp": 12, "env": {}},
        },
    )

    assert (
        layouts["prod_nvt"]["env"]["GMX_PCFF_EWALD_BETA_INV_A"]
        == "0.23469998"
    )
