#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from polygen_stage_metric_audit import (
    AMU_PER_NM3_TO_G_CM3,
    DEFAULT_NOTEBOOK,
    KJ_MOL_NM3_TO_BAR,
    SYSTEM_MASS_AMU,
    expected_signature_fragment,
    gmx_terms_to_metrics,
    parse_gmx_energy_terms,
)
from run_polygen_stage_short_probe import (
    GMX as DEFAULT_GMX_MIXED,
    LMP,
    REPO,
    add_deltas,
    gmx_stage_env,
    parse_lammps_rows,
    parse_lammps_run_steps,
    parse_lammps_timestep_fs,
    read_text,
    run,
    run_live_to_file,
    summarize_lammps,
    write_rows,
    write_text,
)


OUT_ROOT = REPO / "output/polygen_pcff_gromacs_initial_em_notebook"
LAMMPS_WORK = OUT_ROOT / "lammps_openmp"
GMX_CPU_WORK = OUT_ROOT / "gromacs_cpu_openmp"
DEFAULT_GMX_DOUBLE = REPO / "build_gateb_double_cpu/bin/gmx_d"

STAGES = [
    {
        "stage_key": "eq01_nvt_50ps",
        "stem": "01_eq01_nvt_50ps",
        "lammps_input": "lammps_equil_01_eq01_nvt_0p5fs_50ps_chunk0001.in",
        "lammps_restart": "equil_01_eq01_nvt_0p5fs_50ps_chunk0001.restart",
        "start_coord": "01_eq01_nvt_50ps.lammps_velocity.gro",
        "grompp_t": None,
        "grompp_e": None,
    },
    {
        "stage_key": "eq02_npt_compress_100ps_chunk0001",
        "stem": "02_eq02_npt_compress_100ps_chunk0001",
        "lammps_input": "lammps_equil_02_eq02_npt_0p5fs_100ps_chunk0001.in",
        "lammps_restart": "equil_02_eq02_npt_0p5fs_100ps_chunk0001.restart",
        "start_coord": "01_eq01_nvt_50ps.finalcoord.g96",
        "grompp_t": None,
        "grompp_e": None,
    },
    {
        "stage_key": "eq02_npt_compress_100ps_chunk0002",
        "stem": "02_eq02_npt_compress_100ps_chunk0002",
        "lammps_input": "lammps_equil_02_eq02_npt_0p5fs_100ps_chunk0002.in",
        "lammps_restart": "equil_02_eq02_npt_0p5fs_100ps_chunk0002.restart",
        "start_coord": "02_eq02_npt_compress_100ps_chunk0001.finalcoord.g96",
        "grompp_t": "02_eq02_npt_compress_100ps_chunk0001.cpt",
        "grompp_e": "02_eq02_npt_compress_100ps_chunk0001.edr",
    },
]


def default_root() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return REPO / "output/probes" / f"eq02_checkpoint_chain_clean_{stamp}"


def copy_required_inputs(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "topol.top", "01_eq01_nvt_50ps.lammps_velocity.gro"):
        shutil.copy2(GMX_CPU_WORK / name, work / name)
    for stage in STAGES:
        stem = stage["stem"]
        shutil.copy2(GMX_CPU_WORK / f"{stem}.mdp", work / f"{stem}.mdp")


def lammps_endpoint_input(original_text: str, final_restart: Path) -> str:
    text = re.sub(
        r"^\s*read_restart\s+.+$",
        f"read_restart    {final_restart}",
        original_text,
        count=1,
        flags=re.M,
    )
    boundary = re.search(r"^\s*thermo_style\s+", text, flags=re.M)
    if boundary is None:
        boundary = re.search(r"^\s*fix\s+1\s+all\s+", text, flags=re.M)
    if boundary is None:
        boundary = re.search(r"^\s*run\s+\S+", text, flags=re.M)
    if boundary is None:
        raise ValueError("Cannot find LAMMPS endpoint probe insertion boundary")
    prefix = text[: boundary.start()].rstrip()
    prefix = re.sub(r"^\s*velocity\s+.+$", "", prefix, flags=re.M).rstrip()
    if not re.search(r"^\s*compute\s+t_thermo\s+", prefix, flags=re.M):
        prefix += "\ncompute         t_thermo all temp"
    if not re.search(r"^\s*compute\s+p_full\s+", prefix, flags=re.M):
        prefix += "\ncompute         p_full all pressure t_thermo"
    if not re.search(r"^\s*compute\s+p_vir\s+", prefix, flags=re.M):
        prefix += "\ncompute         p_vir all pressure NULL virial"
    if not re.search(r"^\s*variable\s+sysdensity\s+equal\b", prefix, flags=re.M):
        prefix += (
            "\nvariable        sysvol      equal vol"
            "\nvariable        sysmass     equal mass(all)/6.0221367e+23"
            "\nvariable        sysdensity  equal v_sysmass/v_sysvol/1.0e-24"
        )
    return (
        prefix
        + "\n"
        + "thermo_style    custom step v_time press "
        + "c_p_full c_p_full[1] c_p_full[2] c_p_full[3] c_p_full[4] c_p_full[5] c_p_full[6] "
        + "c_p_vir c_p_vir[1] c_p_vir[2] c_p_vir[3] c_p_vir[4] c_p_vir[5] c_p_vir[6] "
        + "vol v_sysdensity temp pe ke etotal\n"
        + "thermo          1\n"
        + "thermo_modify   flush yes\n"
        + "run             0 post no\n"
    )


def run_lammps_endpoint_probe(root: Path, stage: dict[str, str], lammps_ntomp: int) -> dict[str, Any]:
    stage_key = stage["stage_key"]
    lammps_input = LAMMPS_WORK / "resume_inputs" / stage["lammps_input"]
    final_restart = LAMMPS_WORK / ".resume_state" / stage["lammps_restart"]
    stage_root = root / "lammps_endpoints"
    stage_root.mkdir(parents=True, exist_ok=True)
    probe_input = stage_root / f"{stage_key}.run0.in"
    probe_log = stage_root / f"{stage_key}.run0.stdout.log"
    write_text(probe_input, lammps_endpoint_input(read_text(lammps_input), final_restart))
    if probe_log.exists() and probe_log.stat().st_size > 0:
        try:
            row: dict[str, Any] = {
                "stage_key": stage_key,
                "lammps_endpoint_returncode": 0,
                "lammps_endpoint_status": "ok",
                "lammps_endpoint_log": str(probe_log),
                "lammps_endpoint_reused": True,
            }
            row.update(summarize_lammps(parse_lammps_rows(probe_log)))
            return row
        except Exception:
            pass
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": str(lammps_ntomp), "OMP_PROC_BIND": "true", "OMP_PLACES": "cores"})
    rc = run_live_to_file(
        [LMP, "-nonbuf", "-sf", "omp", "-pk", "omp", str(lammps_ntomp), "-in", probe_input],
        cwd=stage_root,
        env=env,
        stdout_path=probe_log,
        label=f"lammps_endpoint_{stage_key}",
    )
    row: dict[str, Any] = {
        "stage_key": stage_key,
        "lammps_endpoint_returncode": rc,
        "lammps_endpoint_status": "ok" if rc == 0 else "failed",
        "lammps_endpoint_log": str(probe_log),
    }
    if rc == 0:
        row.update(summarize_lammps(parse_lammps_rows(probe_log)))
    return row


def run_gmx_grompp(
    gmx: Path,
    work: Path,
    stem: str,
    coord: Path,
    *,
    t_file: Path | None = None,
    e_file: Path | None = None,
) -> None:
    cmd: list[str | Path] = [
        gmx,
        "grompp",
        "-f",
        f"{stem}.mdp",
        "-c",
        coord.name,
        "-p",
        "topol.top",
        "-o",
        f"{stem}.tpr",
        "-maxwarn",
        "2",
    ]
    if t_file is not None:
        cmd.extend(["-t", t_file.name])
    if e_file is not None:
        cmd.extend(["-e", e_file.name])
    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=work,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    stdout = result.stdout.decode("utf-8", errors="replace")
    write_text(work / f"{stem}.grompp.stdout.log", "$ " + " ".join(map(str, cmd)) + "\n" + stdout)
    if result.returncode != 0:
        raise RuntimeError(f"grompp failed for {stem}; see {work / f'{stem}.grompp.stdout.log'}")


def gmx_finalcoord_g96(gmx: Path, work: Path, stem: str) -> Path:
    out = work / f"{stem}.finalcoord.g96"
    cmd = [gmx, "trjconv", "-s", f"{stem}.tpr", "-f", f"{stem}.cpt", "-o", out.name]
    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=work,
        input="0\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    write_text(work / f"{stem}.finalcoord.trjconv.stdout.log", "$ " + " ".join(map(str, cmd)) + "\n" + result.stdout)
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(f"trjconv finalcoord failed for {stem}")
    return out


def collect_gmx_completed_stage_metrics(gmx: Path, root: Path, work: Path, stem: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "gmx_stem": stem,
        "gmx_returncode": 0,
        "gmx_status": "ok",
        "gmx_reused_completed_outputs": True,
        "gmx_mdrun_log": str(work / f"{stem}.mdrun.stdout.log"),
        "gmx_grompp_log": str(work / f"{stem}.grompp.stdout.log"),
    }
    if not (work / f"{stem}.finalcoord.g96").exists() and (work / f"{stem}.cpt").exists():
        gmx_finalcoord_g96(gmx, work, stem)
    if (work / f"{stem}.finalcoord.g96").exists():
        row["gmx_finalcoord_g96"] = str(work / f"{stem}.finalcoord.g96")
    terms = parse_gmx_energy_terms(gmx, work / f"{stem}.edr", root)
    row.update(gmx_terms_to_metrics(terms))
    gro_path = work / f"{stem}.gro"
    if gro_path.exists():
        vals = [float(value) for value in gro_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-1].split()[:3]]
        if len(vals) == 3:
            gro_volume = vals[0] * vals[1] * vals[2]
            row["gmx_gro_volume_nm3"] = gro_volume
            row["gmx_density_from_gro_g_cm3"] = SYSTEM_MASS_AMU * AMU_PER_NM3_TO_G_CM3 / gro_volume
            row.setdefault("volume_nm3_final", gro_volume)
            row.setdefault("density_g_cm3_final", row["gmx_density_from_gro_g_cm3"])
    volume_for_virial = row.get("volume_nm3_final") or row.get("gmx_gro_volume_nm3")
    if "virial_xx_kj_mol_final" in row and volume_for_virial:
        volume = float(volume_for_virial)
        diag = [float(row[f"virial_{axis}_kj_mol_final"]) for axis in ("xx", "yy", "zz")]
        row["gmx_virial_pressure_bar"] = sum((-2.0 * value / volume * KJ_MOL_NM3_TO_BAR) for value in diag) / 3.0
    return row


def run_gmx_stage(
    gmx: Path,
    root: Path,
    work: Path,
    stage: dict[str, str],
    config: dict[str, Any],
    ntomp: int,
) -> dict[str, Any]:
    stage_key = stage["stage_key"]
    stem = stage["stem"]
    if all((work / f"{stem}{suffix}").exists() for suffix in (".tpr", ".edr", ".gro", ".cpt")):
        row = {"stage_key": stage_key}
        row.update(collect_gmx_completed_stage_metrics(gmx, root, work, stem))
        return row
    lammps_input = LAMMPS_WORK / "resume_inputs" / stage["lammps_input"]
    original_text = read_text(lammps_input)
    original_run_steps = parse_lammps_run_steps(original_text)
    timestep_fs = parse_lammps_timestep_fs(original_text)
    coord = work / stage["start_coord"]
    t_file = work / stage["grompp_t"] if stage.get("grompp_t") else None
    e_file = work / stage["grompp_e"] if stage.get("grompp_e") else None
    run_gmx_grompp(gmx, work, stem, coord, t_file=t_file, e_file=e_file)

    env = os.environ.copy()
    env.update(gmx_stage_env(stage_key, original_text, original_run_steps, timestep_fs, config, LAMMPS_WORK / "equil_from_em.stdout.log"))
    env.update({"OMP_NUM_THREADS": str(ntomp), "OMP_PROC_BIND": "true", "OMP_PLACES": "cores", "GMX_MAXBACKUP": "-1"})
    if config.get("GMX_PCFF_MTTK_BOXV_INTEGRATOR"):
        env["GMX_PCFF_MTTK_BOXV_INTEGRATOR"] = str(config["GMX_PCFF_MTTK_BOXV_INTEGRATOR"])
    if e_file is not None:
        env["GMX_PCFF_RESTORE_NH_MTTK_STATE_FROM_EDR"] = str(e_file.resolve())
        env["GMX_PCFF_RESTORE_NH_MTTK_STATE_TIME_PS"] = f"{original_run_steps * timestep_fs / 1000.0:.9g}"

    cmd: list[str | Path] = [
        "taskset",
        "-c",
        "0-11",
        gmx,
        "mdrun",
        "-s",
        f"{stem}.tpr",
        "-deffnm",
        stem,
        "-ntmpi",
        "1",
        "-ntomp",
        str(ntomp),
        "-pin",
        "off",
        "-dlb",
        "no",
        "-notunepme",
        "-cpt",
        "10",
        "-update",
        "cpu",
        "-nb",
        "cpu",
        "-bonded",
        "cpu",
    ]
    rc = run_live_to_file(
        cmd,
        cwd=work,
        env=env,
        stdout_path=work / f"{stem}.mdrun.stdout.log",
        label=f"gmx_chain_{stem}",
    )
    row: dict[str, Any] = {
        "stage_key": stage_key,
        "gmx_stem": stem,
        "gmx_returncode": rc,
        "gmx_status": "ok" if rc == 0 else "failed",
        "gmx_mdrun_log": str(work / f"{stem}.mdrun.stdout.log"),
        "gmx_grompp_log": str(work / f"{stem}.grompp.stdout.log"),
    }
    if rc != 0:
        return row
    final_g96 = gmx_finalcoord_g96(gmx, work, stem)
    terms = parse_gmx_energy_terms(gmx, work / f"{stem}.edr", root)
    row.update(gmx_terms_to_metrics(terms))
    row["gmx_finalcoord_g96"] = str(final_g96)
    gro_volume = None
    gro_path = work / f"{stem}.gro"
    if gro_path.exists():
        vals = [float(value) for value in gro_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-1].split()[:3]]
        if len(vals) == 3:
            gro_volume = vals[0] * vals[1] * vals[2]
            row["gmx_gro_volume_nm3"] = gro_volume
            row["gmx_density_from_gro_g_cm3"] = SYSTEM_MASS_AMU * AMU_PER_NM3_TO_G_CM3 / gro_volume
            row.setdefault("volume_nm3_final", gro_volume)
            row.setdefault("density_g_cm3_final", row["gmx_density_from_gro_g_cm3"])
    volume_for_virial = row.get("volume_nm3_final") or row.get("gmx_gro_volume_nm3")
    if "virial_xx_kj_mol_final" in row and volume_for_virial:
        volume = float(volume_for_virial)
        diag = [float(row[f"virial_{axis}_kj_mol_final"]) for axis in ("xx", "yy", "zz")]
        row["gmx_virial_pressure_bar"] = sum((-2.0 * value / volume * KJ_MOL_NM3_TO_BAR) for value in diag) / 3.0
    return row


def comparison_row(stage_key: str, lmp: dict[str, Any], gmx: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"stage_key": stage_key, "lammps_status": lmp.get("lammps_endpoint_status"), "gmx_status": gmx.get("gmx_status")}
    pairs = {
        "volume_nm3": ("lammps_volume_nm3", "volume_nm3_final"),
        "density_g_cm3": ("lammps_density_g_cm3", "density_g_cm3_final"),
        "temperature_k": ("lammps_temperature_k", "temperature_k_final"),
        "pressure_bar": ("lammps_pressure_bar", "pressure_bar_final"),
        "potential_kj_mol": ("lammps_potential_kj_mol", "potential_kj_mol_final"),
        "kinetic_kj_mol": ("lammps_kinetic_kj_mol", "kinetic_energy_kj_mol_final"),
        "total_kj_mol": ("lammps_total_kj_mol", "total_energy_kj_mol_final"),
        "virial_pressure_bar": ("lammps_pressure_virial_bar", "gmx_virial_pressure_bar"),
    }
    for label, (lmp_key, gmx_key) in pairs.items():
        lmp_value = lmp.get(lmp_key)
        gmx_value = gmx.get(gmx_key)
        row[f"lammps_{label}"] = lmp_value
        row[f"gmx_{label}"] = gmx_value
        if lmp_value is not None and gmx_value is not None:
            row[f"delta_{label}"] = float(gmx_value) - float(lmp_value)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a clean eq01->eq02 GROMACS checkpoint-chain probe and compare to LAMMPS restart endpoints.")
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--gmx", type=Path, default=DEFAULT_GMX_DOUBLE if DEFAULT_GMX_DOUBLE.exists() else DEFAULT_GMX_MIXED)
    parser.add_argument("--ntomp", type=int, default=12)
    parser.add_argument("--lammps-ntomp", type=int, default=12)
    args = parser.parse_args()

    args.root = args.root.resolve()
    work = args.root / "gromacs_cpu_openmp"
    copy_required_inputs(work)
    config = __import__("polygen_stage_metric_audit").load_notebook_config(args.notebook.resolve())
    lammps_rows: dict[str, dict[str, Any]] = {}
    gmx_rows: dict[str, dict[str, Any]] = {}
    comparison: list[dict[str, Any]] = []

    for stage in STAGES:
        lmp_row = run_lammps_endpoint_probe(args.root, stage, args.lammps_ntomp)
        lammps_rows[stage["stage_key"]] = lmp_row
        gmx_row = run_gmx_stage(args.gmx.resolve(), args.root, work, stage, config, args.ntomp)
        gmx_rows[stage["stage_key"]] = gmx_row
        comparison.append(comparison_row(stage["stage_key"], lmp_row, gmx_row))
        write_rows(args.root / "lammps_endpoint_metrics.csv", list(lammps_rows.values()))
        write_rows(args.root / "gromacs_chain_metrics.csv", list(gmx_rows.values()))
        write_rows(args.root / "eq02_checkpoint_chain_comparison.csv", comparison)
        print(comparison[-1], flush=True)

    print(args.root / "eq02_checkpoint_chain_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
