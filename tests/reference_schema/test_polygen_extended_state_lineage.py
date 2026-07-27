from __future__ import annotations

from tools.pcff_respa_parity.polygen_gromacs_runtime import (
    _polygen_md_mdp,
    _setup_polygen_em_handoff_environment,
)


def test_short_suffix_keeps_a_usable_energy_window() -> None:
    mdp, nsteps, _ = _polygen_md_mdp(
        name="eq04_npt_compress_500ps_chunk0003",
        ensemble="npt",
        duration_ps=15.0,
        base_dt_ps=0.0005,
        eqfactor=1.0,
        temp_k=353.0,
        pressure_atm=4000.0,
        tau_t_ps=0.2,
        tau_p_ps=2.0,
    )

    assert nsteps == 30_000
    assert "nstcalcenergy           = 3000" in mdp
    assert "nstenergy               = 3000" in mdp
    assert "nstlog                  = 3000" in mdp


def test_actual_1474101_eq09_rescue_matches_completed_lammps_debug_rescue(
    tmp_path, monkeypatch
) -> None:
    project = tmp_path / "Traj_1474101"
    md_dir = project / "MD_GMX"
    md_dir.mkdir(parents=True)
    (md_dir / "conf.gro").write_text("fixture\n", encoding="utf-8")
    (md_dir / "topol.top").write_text("fixture\n", encoding="utf-8")
    monkeypatch.setenv("GROMACS_BATCH_EQ09_RESCUE_MODE", "actual_1474101")

    context = _setup_polygen_em_handoff_environment(
        project,
        prepare_report={
            "binary": {
                "batch_binary": "/opt/gromacs/bin/gmx",
                "gmxlib": None,
            }
        },
        temperature_k=353.0,
        eqfactor=0.83,
        production_total_ns=20.0,
        nproc=12,
        gromacs_stage_layouts={},
    )

    eq09 = [
        stage for stage in context["stages"] if stage["name"].startswith("eq09")
    ]
    stages_by_name = {stage["name"]: stage for stage in context["stages"]}
    assert stages_by_name["eq07_npt_heat_400ps_chunk0001"]["init_step"] == 0
    assert (
        stages_by_name["eq07_npt_heat_400ps_chunk0002"]["init_step"]
        == 400_000
    )
    assert stages_by_name["eq08_npt_cool_400ps_chunk0001"]["init_step"] == 0
    assert (
        stages_by_name["eq08_npt_cool_400ps_chunk0002"]["init_step"]
        == 400_000
    )
    assert not any(
        stage.get("preserve_extended_state_from_previous_base", False)
        for stage in context["stages"]
    )
    assert [stage["name"] for stage in eq09] == [
        "eq09_rescue_recondition_50ps",
        "eq09_rescue_npt_segment01_50ps",
        "eq09_rescue_npt_segment02_50ps",
        "eq09_rescue_npt_segment03_50ps",
        "eq09_rescue_npt_segment04_50ps",
        "eq09_npt_compress_chunk0002_49ps",
    ]

    recondition, *segments, chunk2 = eq09
    assert recondition["dt_ps"] == 0.0000625
    assert recondition["outer_timestep_fs"] == 0.25
    assert recondition["nsteps"] == 800_000
    assert recondition["soft_start"] == {
        "displacement_limit_nm": 0.0005,
        "langevin_temp_k": 353.0,
        "langevin_damp_ps": 0.05,
        "langevin_seed": 1474101,
        "zero_net_force": True,
    }

    assert [
        (stage["pressure_start_atm"], stage["pressure_end_atm"])
        for stage in segments
    ] == [
        (1.0, 500.0),
        (500.0, 1500.0),
        (1500.0, 2500.0),
        (2500.0, 3213.04819),
    ]
    for segment in segments:
        assert segment["dt_ps"] == 0.0000625
        assert segment["outer_timestep_fs"] == 0.25
        assert segment["nsteps"] == 800_000
        assert segment["init_step"] == 0
        assert segment["tau_t_ps"] == 0.1
        assert segment["tau_p_ps"] == 5.0
        assert segment["lammps_fix_nh_drag"] == 4.0

    assert chunk2["dt_ps"] == 0.0005
    assert chunk2["outer_timestep_fs"] == 2.0
    assert chunk2["nsteps"] == 98_000
    assert chunk2["init_step"] == 0
    assert chunk2["pressure_start_atm"] == 3213.04819
    assert chunk2["pressure_end_atm"] == 4000.0
