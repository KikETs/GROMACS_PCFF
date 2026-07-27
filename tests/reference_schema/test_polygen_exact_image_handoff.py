from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.pcff_respa_parity import polygen_gromacs_runtime as runtime
from tools.pcff_respa_parity.polygen_gromacs_runtime import (
    _exact_image_stage_input_binding,
    _materialize_exact_respa_continuous_gro,
    _read_exact_handoff_gro,
    _read_exact_respa_image_sidecar,
    _rebase_exact_respa_image_sidecar_step,
    _render_exact_respa_image_sidecar,
    _stage_requires_exact_image_handoff,
    _validate_exact_image_completion_manifest,
    _write_exact_image_completion_manifest,
    _write_initial_exact_respa_image_sidecar,
    _write_text_atomically_without_overwrite,
)


def _execution_contract() -> dict[str, object]:
    return {
        "gromacs_runtime": {
            "executable": {"path": "/opt/gromacs/bin/gmx", "sha256": "a" * 64},
            "version_output": "GROMACS version: 2024.4\n",
            "version_output_sha256": "b" * 64,
            "libgromacs": {
                "path": "/opt/gromacs/lib/libgromacs.so",
                "sha256": "c" * 64,
            },
        },
        "material_stage_env": {
            "CUDA_VISIBLE_DEVICES": "0",
            "GMXLIB": "/opt/gromacs/share/top",
            "GMX_PCFF_EXACT_RESPA": "1",
            "OMP_NUM_THREADS": "12",
        },
        "layout": {
            "ntmpi": 1,
            "ntomp": 12,
            "env": {"OMP_PROC_BIND": "close"},
            "extra_args": ["-nb", "gpu"],
            "source": "gromacs_stage_layouts.eq01",
        },
        "merged_mdrun_args": ["-nb", "gpu", "-pme", "cpu"],
        "grompp_extra_args": ["-maxwarn", "1"],
    }


def _write_gro(
    path: Path,
    coordinates: list[tuple[float, float, float]],
    *,
    width: int,
    decimals: int,
    velocities: list[tuple[float, float, float]] | None = None,
) -> None:
    lines = ["handoff fixture", f"{len(coordinates):5d}"]
    for index, coordinate in enumerate(coordinates, start=1):
        identity = f"{1:5d}{'POL':<5}{'C':>5}{index:5d}"
        line = identity + "".join(
            f"{value:{width}.{decimals}f}" for value in coordinate
        )
        if velocities is not None:
            line += "".join(
                f"{value:{width}.{decimals}f}" for value in velocities[index - 1]
            )
        lines.append(line)
    lines.append("   10.00000   11.00000   12.00000")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_initial_sidecar_accepts_bridge_high_precision_gro(tmp_path: Path) -> None:
    source = tmp_path / "conf.gro"
    _write_gro(
        source,
        [(11.25, -0.5, 30.125), (-20.25, 21.5, 9.75)],
        width=18,
        decimals=12,
    )

    sidecar_path = _write_initial_exact_respa_image_sidecar(
        source, tmp_path / "initial.sidecar", step=0
    )
    sidecar = _read_exact_respa_image_sidecar(sidecar_path)

    assert sidecar["step"] == 0
    assert sidecar["natoms"] == 2
    assert sidecar["atoms"][0]["image"] == [0, 0, 0]
    assert sidecar["atoms"][0]["continuous_position_nm"] == [11.25, -0.5, 30.125]


def test_rebase_and_continuous_gro_preserve_exact_images_and_velocities(
    tmp_path: Path,
) -> None:
    source_gro = tmp_path / "stage.gro"
    _write_gro(
        source_gro,
        [(1.25, 10.5, 6.125), (9.75, 0.5, 1.0)],
        width=8,
        decimals=3,
        velocities=[(0.1, -0.2, 0.3), (-0.4, 0.5, -0.6)],
    )
    atoms = [
        {
            "image": [1, -1, 2],
            "state_position_nm": [1.25, 10.5, 6.125],
            "continuous_position_nm": [11.25, -0.5, 30.125],
        },
        {
            "image": [-2, 2, 0],
            "state_position_nm": [9.75, 0.5, 1.0],
            "continuous_position_nm": [-10.25, 22.5, 1.0],
        },
    ]
    output_sidecar = tmp_path / "stage.image_out.sidecar"
    _write_text_atomically_without_overwrite(
        output_sidecar,
        _render_exact_respa_image_sidecar(
            step=80000, box_nm=[10.0, 11.0, 12.0], atoms=atoms
        ),
    )

    rebased_path = _rebase_exact_respa_image_sidecar_step(
        output_sidecar, tmp_path / "next.image_in.sidecar", step=0
    )
    rebased = _read_exact_respa_image_sidecar(rebased_path)
    assert rebased["step"] == 0
    assert rebased["atoms"] == atoms

    continuous_gro = _materialize_exact_respa_continuous_gro(
        source_gro, output_sidecar, tmp_path / "stage.continuous.gro"
    )
    frame = _read_exact_handoff_gro(continuous_gro)
    assert frame["atoms"][0]["coordinates_nm"] == [11.25, -0.5, 30.125]
    assert frame["atoms"][1]["coordinates_nm"] == [-10.25, 22.5, 1.0]
    assert frame["atoms"][0]["velocities_nm_ps"] == [0.1, -0.2, 0.3]


def test_handoff_file_writer_refuses_nonmatching_existing_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "protected.sidecar"
    _write_text_atomically_without_overwrite(path, "first\n")
    _write_text_atomically_without_overwrite(path, "first\n")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        _write_text_atomically_without_overwrite(path, "second\n")


def test_continuous_gro_rejects_stale_sidecar_state(tmp_path: Path) -> None:
    source_gro = tmp_path / "stage.gro"
    _write_gro(source_gro, [(1.0, 2.0, 3.0)], width=8, decimals=3)
    sidecar_path = tmp_path / "stale.sidecar"
    _write_text_atomically_without_overwrite(
        sidecar_path,
        _render_exact_respa_image_sidecar(
            step=1,
            box_nm=[10.0, 11.0, 12.0],
            atoms=[
                {
                    "image": [0, 0, 0],
                    "state_position_nm": [1.1, 2.0, 3.0],
                    "continuous_position_nm": [1.1, 2.0, 3.0],
                }
            ],
        ),
    )

    with pytest.raises(RuntimeError, match="state mismatch"):
        _materialize_exact_respa_continuous_gro(
            source_gro, sidecar_path, tmp_path / "continuous.gro"
        )


def _completion_manifest_fixture(tmp_path: Path) -> dict[str, object]:
    gro_path = tmp_path / "eq01.gro"
    _write_gro(gro_path, [(1.0, 2.0, 3.0)], width=8, decimals=3)
    sidecar_path = tmp_path / "eq01.image_out.sidecar"
    _write_text_atomically_without_overwrite(
        sidecar_path,
        _render_exact_respa_image_sidecar(
            step=10,
            box_nm=[10.0, 11.0, 12.0],
            atoms=[
                {
                    "image": [0, 0, 0],
                    "state_position_nm": [1.0, 2.0, 3.0],
                    "continuous_position_nm": [1.0, 2.0, 3.0],
                }
            ],
        ),
    )
    tpr_path = tmp_path / "eq01.tpr"
    checkpoint_path = tmp_path / "eq01.cpt"
    image_input_path = tmp_path / "eq01.image_in.sidecar"
    upstream_state_path = tmp_path / "upstream.final_state.trr"
    mdp_path = tmp_path / "eq01.mdp"
    topology_path = tmp_path / "topol.top"
    tpr_path.write_bytes(b"tpr-v1")
    checkpoint_path.write_bytes(b"cpt-v1")
    image_input_path.write_bytes(b"image-input-v1")
    upstream_state_path.write_bytes(b"upstream-state-v1")
    mdp_path.write_bytes(b"mdp-v1")
    topology_path.write_bytes(b"topology-v1")
    stage = {"name": "eq01", "kind": "md", "init_step": 0, "nsteps": 10}
    input_binding = _exact_image_stage_input_binding(
        stage=stage,
        stage_image_input=image_input_path,
        upstream_state=upstream_state_path,
        upstream_state_kind="state_trr",
        mdp_path=mdp_path,
        topology_path=topology_path,
        execution_contract=_execution_contract(),
    )

    _write_exact_image_completion_manifest(
        md_dir=tmp_path,
        stage=stage,
        deffnm="eq01",
        gro_path=gro_path,
        sidecar_path=sidecar_path,
        tpr_path=tpr_path,
        checkpoint_path=checkpoint_path,
        input_binding=input_binding,
    )
    return {
        "stage": stage,
        "gro": gro_path,
        "sidecar": sidecar_path,
        "tpr": tpr_path,
        "checkpoint": checkpoint_path,
        "image_input": image_input_path,
        "upstream_state": upstream_state_path,
        "mdp": mdp_path,
        "topology": topology_path,
        "execution_contract": _execution_contract(),
        "input_binding": input_binding,
    }


def _validate_completion_fixture(
    tmp_path: Path,
    fixture: dict[str, object],
    *,
    stage: dict[str, object] | None = None,
    input_binding: dict[str, object] | None = None,
    allow_runtime_drift: bool = False,
) -> bool:
    return _validate_exact_image_completion_manifest(
        md_dir=tmp_path,
        stage=stage or fixture["stage"],
        deffnm="eq01",
        gro_path=fixture["gro"],
        sidecar_path=fixture["sidecar"],
        tpr_path=fixture["tpr"],
        checkpoint_path=fixture["checkpoint"],
        input_binding=input_binding or fixture["input_binding"],
        allow_runtime_drift=allow_runtime_drift,
    )


def test_completion_manifest_binds_gro_sidecar_tpr_and_checkpoint(
    tmp_path: Path,
) -> None:
    fixture = _completion_manifest_fixture(tmp_path)

    assert _validate_completion_fixture(tmp_path, fixture)

    fixture["checkpoint"].write_bytes(b"cpt-modified")
    assert not _validate_completion_fixture(tmp_path, fixture)


@pytest.mark.parametrize(
    "changed_input",
    ["image_input", "upstream_state", "mdp", "topology"],
)
def test_downstream_manifest_rejects_changed_pre_stage_input(
    tmp_path: Path, changed_input: str
) -> None:
    fixture = _completion_manifest_fixture(tmp_path)
    fixture[changed_input].write_bytes(f"{changed_input}-v2".encode())
    current_binding = _exact_image_stage_input_binding(
        stage=fixture["stage"],
        stage_image_input=fixture["image_input"],
        upstream_state=fixture["upstream_state"],
        upstream_state_kind="state_trr",
        mdp_path=fixture["mdp"],
        topology_path=fixture["topology"],
        execution_contract=fixture["execution_contract"],
    )
    assert not _validate_completion_fixture(
        tmp_path, fixture, input_binding=current_binding
    )


def test_completion_manifest_rejects_current_stage_spec_change(tmp_path: Path) -> None:
    fixture = _completion_manifest_fixture(tmp_path)
    changed_stage = dict(fixture["stage"])
    changed_stage["nsteps"] = 5
    current_binding = _exact_image_stage_input_binding(
        stage=changed_stage,
        stage_image_input=fixture["image_input"],
        upstream_state=fixture["upstream_state"],
        upstream_state_kind="state_trr",
        mdp_path=fixture["mdp"],
        topology_path=fixture["topology"],
        execution_contract=fixture["execution_contract"],
    )
    assert not _validate_completion_fixture(
        tmp_path,
        fixture,
        stage=changed_stage,
        input_binding=current_binding,
    )


@pytest.mark.parametrize(
    "changed_component",
    [
        "executable_sha",
        "version_output",
        "libgromacs_sha",
        "material_env",
        "layout",
        "merged_mdrun_args",
        "grompp_extra_args",
    ],
)
def test_completion_manifest_rejects_changed_execution_contract(
    tmp_path: Path, changed_component: str
) -> None:
    fixture = _completion_manifest_fixture(tmp_path)
    contract = copy.deepcopy(fixture["execution_contract"])
    if changed_component == "executable_sha":
        contract["gromacs_runtime"]["executable"]["sha256"] = "d" * 64
    elif changed_component == "version_output":
        contract["gromacs_runtime"]["version_output"] = "GROMACS version: other\n"
    elif changed_component == "libgromacs_sha":
        contract["gromacs_runtime"]["libgromacs"]["sha256"] = "e" * 64
    elif changed_component == "material_env":
        contract["material_stage_env"]["CUDA_VISIBLE_DEVICES"] = "1"
    elif changed_component == "layout":
        contract["layout"]["ntomp"] = 6
    elif changed_component == "merged_mdrun_args":
        contract["merged_mdrun_args"] = ["-nb", "cpu"]
    else:
        contract["grompp_extra_args"] = ["-maxwarn", "2"]
    current_binding = _exact_image_stage_input_binding(
        stage=fixture["stage"],
        stage_image_input=fixture["image_input"],
        upstream_state=fixture["upstream_state"],
        upstream_state_kind="state_trr",
        mdp_path=fixture["mdp"],
        topology_path=fixture["topology"],
        execution_contract=contract,
    )
    assert not _validate_completion_fixture(
        tmp_path, fixture, input_binding=current_binding
    )


def test_completion_manifest_can_allow_only_gromacs_runtime_drift(
    tmp_path: Path,
) -> None:
    fixture = _completion_manifest_fixture(tmp_path)
    contract = copy.deepcopy(fixture["execution_contract"])
    contract["gromacs_runtime"]["executable"]["path"] = "/opt/new/bin/gmx"
    contract["gromacs_runtime"]["executable"]["sha256"] = "d" * 64
    contract["gromacs_runtime"]["libgromacs"]["sha256"] = "e" * 64
    contract["layout"]["source"] = "/new/provenance-only/layout.json"
    current_binding = _exact_image_stage_input_binding(
        stage=fixture["stage"],
        stage_image_input=fixture["image_input"],
        upstream_state=fixture["upstream_state"],
        upstream_state_kind="state_trr",
        mdp_path=fixture["mdp"],
        topology_path=fixture["topology"],
        execution_contract=contract,
    )
    assert not _validate_completion_fixture(
        tmp_path, fixture, input_binding=current_binding
    )
    assert _validate_completion_fixture(
        tmp_path,
        fixture,
        input_binding=current_binding,
        allow_runtime_drift=True,
    )

    contract["layout"]["ntomp"] = 6
    changed_layout_binding = _exact_image_stage_input_binding(
        stage=fixture["stage"],
        stage_image_input=fixture["image_input"],
        upstream_state=fixture["upstream_state"],
        upstream_state_kind="state_trr",
        mdp_path=fixture["mdp"],
        topology_path=fixture["topology"],
        execution_contract=contract,
    )
    assert not _validate_completion_fixture(
        tmp_path,
        fixture,
        input_binding=changed_layout_binding,
        allow_runtime_drift=True,
    )


def test_stage_execution_contract_captures_material_env_and_final_args() -> None:
    contract = runtime._stage_execution_contract(
        runtime_identity=_execution_contract()["gromacs_runtime"],
        stage_env={
            "PATH": "/usr/bin",
            "GMXLIB": "/gmx/top",
            "GMX_PCFF_SWITCH": "1",
            "GMX_PCFF_EXACT_RESPA_IMAGE_SIDECAR_IN": "/dynamic/in",
            "OMP_NUM_THREADS": "8",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        layout={
            "ntmpi": 1,
            "ntomp": 8,
            "env": {"OMP_PROC_BIND": "spread"},
            "extra_args": ["-nb", "gpu"],
            "source": "test-layout",
        },
        merged_mdrun_args=["-nb", "gpu", "-pme", "cpu"],
        grompp_extra_args=["-maxwarn", "1"],
    )

    assert contract["material_stage_env"] == {
        "CUDA_VISIBLE_DEVICES": "0",
        "GMXLIB": "/gmx/top",
        "GMX_PCFF_SWITCH": "1",
        "OMP_NUM_THREADS": "8",
    }
    assert contract["layout"] == {
        "ntmpi": 1,
        "ntomp": 8,
        "env": {"OMP_PROC_BIND": "spread"},
        "extra_args": ["-nb", "gpu"],
        "source": "test-layout",
    }
    assert contract["merged_mdrun_args"] == ["-nb", "gpu", "-pme", "cpu"]
    assert contract["grompp_extra_args"] == ["-maxwarn", "1"]


def test_runtime_identity_hashes_resolved_binary_version_and_loaded_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "gmx"
    library = tmp_path / "libgromacs.so.9"
    executable.write_bytes(b"gmx-binary-v1")
    library.write_bytes(b"libgromacs-v1")

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        if command[1:] == ["--version"]:
            return SimpleNamespace(stdout="GROMACS version: 2024.4  \r\nBuild: patched\r\n\r\n")
        assert command[0] == "ldd"
        return SimpleNamespace(
            stdout=f"libgromacs.so.9 => {library} (0x00000000)\n"
        )

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    identity = runtime._gromacs_runtime_identity(
        gmx=str(executable), cwd=tmp_path, env={"PATH": "/usr/bin"}
    )

    assert identity["executable"] == {
        "path": str(executable.resolve()),
        "sha256": runtime._sha256_file(executable),
    }
    assert identity["version_output"] == (
        "GROMACS version: 2024.4\nBuild: patched\n"
    )
    assert identity["libgromacs"] == {
        "path": str(library.resolve()),
        "sha256": runtime._sha256_file(library),
    }


@pytest.mark.parametrize("invalid_field", ["schema", "state", "contract"])
def test_lineage_marker_schema_state_and_contract_fail_closed(
    tmp_path: Path, invalid_field: str
) -> None:
    marker = {
        "schema_name": "gromacs_exact_image_lineage_rebuild",
        "schema_version": 1,
        "state": "production_pending",
        "origin_stage": "eq03_pre_2fs_minimize",
        "equil_protocol_contract_sha256": "a" * 64,
    }
    if invalid_field == "schema":
        marker["schema_version"] = 99
    elif invalid_field == "state":
        marker["state"] = "unknown"
    else:
        marker["equil_protocol_contract_sha256"] = "not-a-sha"
    marker_path = tmp_path / runtime.GROMACS_EXACT_IMAGE_LINEAGE_REBUILD_MARKER
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError):
        runtime.gromacs_exact_image_lineage_rebuild_state(tmp_path)


def test_explicit_protocol_change_rebuild_boundary_is_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = ["eq01", "eq07", "eq08"]

    assert runtime._requested_protocol_change_rebuild_index(stages) is None

    monkeypatch.setenv("GROMACS_BATCH_LINEAGE_REBUILD_FROM_STAGE", "eq07")
    assert runtime._requested_protocol_change_rebuild_index(stages) == 1

    monkeypatch.setenv("GROMACS_BATCH_LINEAGE_REBUILD_FROM_STAGE", "missing")
    with pytest.raises(RuntimeError, match="unknown equilibration stage"):
        runtime._requested_protocol_change_rebuild_index(stages)


def test_eq03_lineage_failure_forces_existing_eq04_clean_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    md_dir = tmp_path / "MD_GMX"
    md_dir.mkdir()
    _write_gro(md_dir / "conf.gro", [(1.0, 2.0, 3.0)], width=8, decimals=3)
    (md_dir / "topol.top").write_text("topology\n", encoding="utf-8")

    eq03 = {
        "name": "eq03_pre_2fs_minimize",
        "phase": "equilibration",
        "kind": "em",
        "mdp_path": str(md_dir / "eq03.mdp"),
        "init_step": 0,
        "nsteps": 10,
        "dt_ps": 0.0,
        "base_index": 3,
        "segment_index": 1,
    }
    eq04 = {
        "name": "eq04_npt_compress_500ps_chunk0001",
        "phase": "equilibration",
        "kind": "md",
        "mdp_path": str(md_dir / "eq04.mdp"),
        "init_step": 0,
        "nsteps": 10,
        "dt_ps": 0.002,
        "base_index": 4,
        "segment_index": 1,
    }
    production = {
        "name": "prod_nvt",
        "phase": "production",
        "kind": "md",
        "mdp_path": str(md_dir / "prod.mdp"),
        "init_step": 0,
        "nsteps": 10,
        "dt_ps": 0.002,
        "base_index": 100,
        "segment_index": 1,
    }
    Path(eq03["mdp_path"]).write_text("integrator = cg\n", encoding="utf-8")
    Path(eq04["mdp_path"]).write_text("integrator = md-vv\n", encoding="utf-8")
    Path(production["mdp_path"]).write_text("integrator = md-vv\n", encoding="utf-8")

    for stage in (eq03, eq04):
        name = str(stage["name"])
        _write_gro(md_dir / f"{name}.gro", [(1.0, 2.0, 3.0)], width=8, decimals=3)
        (md_dir / f"{name}.tpr").write_bytes(f"old-{name}-tpr".encode())
    (md_dir / f"{eq04['name']}.cpt").write_bytes(b"old-eq04-checkpoint")
    _write_gro(md_dir / "prod_nvt.gro", [(1.0, 2.0, 3.0)], width=8, decimals=3)
    (md_dir / "prod_nvt.tpr").write_bytes(b"old-production-tpr")
    (md_dir / "prod_nvt.cpt").write_bytes(b"old-production-checkpoint")
    (md_dir / "production_complete.flag").write_text("old-done\n", encoding="utf-8")
    (md_dir / "analysis").mkdir()
    (md_dir / "analysis" / "old_report.json").write_text(
        '{"lineage":"old"}\n', encoding="utf-8"
    )
    (md_dir / "analysis_done.flag").write_text("old-analysis\n", encoding="utf-8")
    _write_text_atomically_without_overwrite(
        md_dir / f"{eq03['name']}.image_out.sidecar",
        _render_exact_respa_image_sidecar(
            step=1,
            box_nm=[10.0, 11.0, 12.0],
            atoms=[
                {
                    "image": [0, 0, 0],
                    "state_position_nm": [1.0, 2.0, 3.0],
                    "continuous_position_nm": [1.0, 2.0, 3.0],
                }
            ],
        ),
    )

    commands: list[list[str]] = []

    def fake_run_cmd(command: list[str], *, cwd: Path, env: dict[str, str], **_: object) -> None:
        commands.append([str(value) for value in command])
        if command[1] == "grompp":
            Path(command[command.index("-o") + 1]).write_bytes(b"new-tpr")
            return
        assert command[1] == "mdrun"
        deffnm = command[command.index("-deffnm") + 1]
        _write_gro(md_dir / f"{deffnm}.gro", [(1.0, 2.0, 3.0)], width=8, decimals=3)
        if deffnm == eq03["name"]:
            Path(env["GMX_PCFF_EXACT_RESPA_IMAGE_SIDECAR_OUT"]).write_text(
                Path(env["GMX_PCFF_EXACT_RESPA_IMAGE_SIDECAR_IN"]).read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
        else:
            checkpoint = (
                b"new-eq04-checkpoint"
                if deffnm == eq04["name"]
                else b"new-production-checkpoint"
            )
            (md_dir / f"{deffnm}.cpt").write_bytes(checkpoint)

    def fake_final_state_trr(
        *, md_dir: Path, stage: dict[str, object], **_: object
    ) -> Path:
        destination = md_dir / f"{stage['name']}_final_state.trr"
        destination.write_bytes(b"new-final-state")
        return destination

    monkeypatch.setenv("GMX_PCFF_EXACT_RESPA_IMAGE_HANDOFF", "1")
    monkeypatch.setattr(runtime, "_run_cmd", fake_run_cmd)
    monkeypatch.setattr(runtime, "_require_completed_md_stage", lambda **_: None)
    monkeypatch.setattr(runtime, "_materialize_final_state_trr", fake_final_state_trr)
    monkeypatch.setattr(runtime, "_gromacs_checkpoint_step", lambda **_: 10)
    monkeypatch.setattr(
        runtime,
        "_gromacs_runtime_identity",
        lambda **_: copy.deepcopy(_execution_contract()["gromacs_runtime"]),
    )

    context = {
        "overall_status": "ready_for_md",
        "md_dir": str(md_dir),
        "gmx_binary": "gmx",
        "stages": [eq03, eq04, production],
        "resume_existing_effective": True,
        "nproc": 1,
    }
    runtime.run_gromacs_equilibration(context)

    eq04_mdruns = [
        command
        for command in commands
        if len(command) > 3
        and command[1] == "mdrun"
        and command[command.index("-deffnm") + 1] == eq04["name"]
    ]
    assert len(eq04_mdruns) == 1
    archived_eq04_checkpoints = list(
        (md_dir / "incomplete_stage_restarts" / str(eq04["name"])).rglob(
            f"{eq04['name']}.cpt"
        )
    )
    assert len(archived_eq04_checkpoints) == 1
    assert archived_eq04_checkpoints[0].read_bytes() == b"old-eq04-checkpoint"
    assert (md_dir / f"{eq04['name']}.cpt").read_bytes() == b"new-eq04-checkpoint"
    lineage_marker = md_dir / runtime.GROMACS_EXACT_IMAGE_LINEAGE_REBUILD_MARKER
    assert lineage_marker.exists()
    assert runtime.gromacs_exact_image_lineage_rebuild_state(md_dir) == "production_pending"

    # Simulate a process crash after equilibration but before production. The
    # completed equilibration cascade must preserve both Eq03 and rebuilt Eq04.
    commands.clear()
    runtime.run_gromacs_equilibration(context)
    assert not any(
        command[1] == "mdrun"
        and command[command.index("-deffnm") + 1] == eq03["name"]
        for command in commands
        if "-deffnm" in command
    )
    assert sum(
        command[1] == "mdrun"
        and command[command.index("-deffnm") + 1] == eq04["name"]
        for command in commands
        if "-deffnm" in command
    ) == 0
    assert lineage_marker.exists()

    # A change to an untracked downstream equilibration stage still changes
    # the whole protocol contract. Reentry must rebuild from the first stage,
    # not merely rerun Eq04 or trust the production-pending marker.
    Path(eq04["mdp_path"]).write_text(
        "integrator = md-vv\nnstlist = 20\n", encoding="utf-8"
    )
    commands.clear()
    with pytest.raises(RuntimeError, match="run equilibration validation"):
        runtime.run_gromacs_production(context)
    assert lineage_marker.exists()
    assert (md_dir / "prod_nvt.cpt").read_bytes() == b"old-production-checkpoint"
    assert (md_dir / "analysis" / "old_report.json").exists()

    commands.clear()
    runtime.run_gromacs_equilibration(context)
    for stage_name in (eq03["name"], eq04["name"]):
        assert sum(
            command[1] == "mdrun"
            and command[command.index("-deffnm") + 1] == stage_name
            for command in commands
            if "-deffnm" in command
        ) == 1
    assert runtime.gromacs_exact_image_lineage_rebuild_state(md_dir) == (
        "production_pending"
    )

    commands.clear()
    runtime.run_gromacs_production(context)
    assert not lineage_marker.exists()
    assert (md_dir / "production_complete.flag").read_text(encoding="utf-8") == "done\n"
    production_archives = md_dir / "incomplete_stage_restarts" / "prod_nvt"
    assert any(
        path.read_bytes() == b"old-production-checkpoint"
        for path in production_archives.rglob("prod_nvt.cpt")
    )
    assert any(
        path.read_text(encoding="utf-8") == "old-done\n"
        for path in production_archives.rglob("production_complete.flag")
    )
    assert any(
        path.read_text(encoding="utf-8") == '{"lineage":"old"}\n'
        for path in production_archives.rglob("old_report.json")
    )
    assert any(
        path.read_text(encoding="utf-8") == "old-analysis\n"
        for path in production_archives.rglob("analysis_done.flag")
    )
    assert not (md_dir / "analysis").exists()
    assert not (md_dir / "analysis_done.flag").exists()

    # Crash-window recovery: once the marker is gone, a missing completion
    # flag must be restored by checkpoint validation without rerunning MD.
    (md_dir / "production_complete.flag").unlink()
    commands.clear()
    runtime.run_gromacs_production(context)
    assert not any(
        command[1] == "mdrun" for command in commands if len(command) > 1
    )
    assert (md_dir / "production_complete.flag").read_text(encoding="utf-8") == (
        "done\n"
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("eq01_soft_langevin_10ps", True),
        ("eq02_npt_compress_100ps_chunk0003", True),
        ("eq03_pre_2fs_minimize", True),
        ("eq04_npt_compress_500ps_chunk0001", False),
        ("prod_nvt", False),
    ],
)
def test_only_pre_eq04_stages_require_image_handoff(
    name: str, expected: bool
) -> None:
    assert _stage_requires_exact_image_handoff({"name": name}) is expected
