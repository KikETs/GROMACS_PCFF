#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.pcff_fixture_bridge.common import (  # noqa: E402
    ANGSTROM_TO_NM,
    KCAL_TO_KJ,
    build_typed_ir,
    parse_lammps_data,
    render_gromacs_topology,
)


ANGSTROM_PER_FS_TO_NM_PER_PS = 100.0
DEFAULT_TEMPERATURE_K = 300.0
DEFAULT_VELOCITY_SEED = 20260330
AMU_TO_KG_PER_NM3 = 1.66053906660
AMU_TO_KG_PER_ANG3 = 1660.53906660

NVT_EQUIL_STEPS = 2000
NVT_PROD_STEPS = 8000
NPT_EQUIL_STEPS = 5000
NPT_PROD_STEPS = 15000
DEFAULT_DT_PS = 0.001


def parse_lammps_thermo(log_path: Path, expected_columns: int):
    rows = []
    skipped_rows = []
    with log_path.open("r", encoding="utf-8") as handle:
        in_thermo = False
        for line_number, line in enumerate(handle, start=1):
            clean_line = line.strip()
            if clean_line.startswith("Step"):
                in_thermo = True
                continue
            if clean_line.startswith("Loop time"):
                in_thermo = False
                continue
            if in_thermo:
                parts = clean_line.split()
                if parts and parts[0].isdigit():
                    if len(parts) != expected_columns:
                        skipped_rows.append({"line": line_number, "raw": clean_line, "columns": len(parts)})
                        continue
                    try:
                        rows.append([float(x) for x in parts])
                    except ValueError:
                        skipped_rows.append({"line": line_number, "raw": clean_line, "columns": len(parts)})
    return np.array(rows), skipped_rows


def parse_xvg(path: Path):
    data = []
    if not path.exists():
        return np.array(data)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("#", "@")):
                continue
            parts = line.split()
            if parts:
                data.append([float(x) for x in parts])
    return np.array(data)


def extract_gmx_energy(work_dir: Path, edr_name: str, output_name: str):
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    selection = "Potential\nTemperature\nPressure\nVolume\nDensity\n0\n"
    subprocess.run(
        [str(REPO_ROOT / "build" / "bin" / "gmx"), "energy", "-f", edr_name, "-o", output_name],
        cwd=work_dir,
        input=selection,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )


def parse_dump_custom_velocities(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    velocities = {}
    index = 0
    while index < len(lines):
        if lines[index] != "ITEM: TIMESTEP":
            index += 1
            continue
        natoms = int(lines[index + 3].strip())
        atom_fields = lines[index + 8].split()[2:]
        atom_start = index + 9
        atom_end = atom_start + natoms
        for atom_line in lines[atom_start:atom_end]:
            raw_values = atom_line.split()
            atom = {}
            for field, raw_value in zip(atom_fields, raw_values):
                atom[field] = int(raw_value) if field == "id" else float(raw_value)
            velocities[atom["id"]] = (atom["vx"], atom["vy"], atom["vz"])
        index = atom_end
    return velocities


def build_lammps_data_with_velocities(
    source_data_path: Path,
    output_data_path: Path,
    velocities_angstrom_fs: dict[int, tuple[float, float, float]],
):
    lines = source_data_path.read_text(encoding="utf-8").splitlines()
    section_headers = {"Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"}
    velocity_start = None
    velocity_end = None
    for idx, line in enumerate(lines):
        stripped = line.split(" #", 1)[0].strip()
        if stripped == "Velocities":
            velocity_start = idx
            continue
        if velocity_start is not None and stripped in section_headers:
            velocity_end = idx
            break

    prefix = lines if velocity_start is None else lines[:velocity_start]
    suffix = [] if velocity_start is None else lines[velocity_end:] if velocity_end is not None else []
    velocity_lines = ["Velocities", ""]
    for atom_id in sorted(velocities_angstrom_fs):
        vx, vy, vz = velocities_angstrom_fs[atom_id]
        velocity_lines.append(f"{atom_id} {vx:.12f} {vy:.12f} {vz:.12f}")
    output_data_path.write_text("\n".join(prefix + [""] + velocity_lines + [""] + suffix + [""]), encoding="utf-8")


def local_atom_names(data: dict) -> dict[int, str]:
    names = {}
    atoms_by_molecule = {}
    for atom in data["atoms"]:
        atoms_by_molecule.setdefault(atom["molecule_id"], []).append(atom)
    for molecule_atoms in atoms_by_molecule.values():
        for local_idx, atom in enumerate(sorted(molecule_atoms, key=lambda item: item["id"]), start=1):
            names[atom["id"]] = f"A{local_idx}"
    return names


def create_gro_from_lammps(
    lammps_data_path: Path,
    gro_path: Path,
    velocities_nm_ps: dict[int, tuple[float, float, float]] | None = None,
):
    data = parse_lammps_data(lammps_data_path)
    box_x = (data["box"]["x"]["hi"] - data["box"]["x"]["lo"]) * ANGSTROM_TO_NM
    box_y = (data["box"]["y"]["hi"] - data["box"]["y"]["lo"]) * ANGSTROM_TO_NM
    box_z = (data["box"]["z"]["hi"] - data["box"]["z"]["lo"]) * ANGSTROM_TO_NM
    atom_names = local_atom_names(data)

    with gro_path.open("w", encoding="utf-8") as handle:
        handle.write("Generated from LAMMPS data\n")
        handle.write(f"{len(data['atoms']):>5d}\n")
        for atom in data["atoms"]:
            x = atom["x_angstrom"] * ANGSTROM_TO_NM
            y = atom["y_angstrom"] * ANGSTROM_TO_NM
            z = atom["z_angstrom"] * ANGSTROM_TO_NM
            velocity = None if velocities_nm_ps is None else velocities_nm_ps.get(atom["id"])
            residue_id = atom["molecule_id"] % 100000
            atom_name = atom_names[atom["id"]]
            if velocity is None:
                handle.write(
                    f"{residue_id:>5d}{'OLI':<5s}{atom_name:>5s}{atom['id']:>5d}{x:15.7f}{y:15.7f}{z:15.7f}\n"
                )
            else:
                vx, vy, vz = velocity
                handle.write(
                    f"{residue_id:>5d}{'OLI':<5s}{atom_name:>5s}{atom['id']:>5d}"
                    f"{x:15.7f}{y:15.7f}{z:15.7f}{vx:15.7f}{vy:15.7f}{vz:15.7f}\n"
                )
        handle.write(f"{box_x:15.7f}{box_y:15.7f}{box_z:15.7f}\n")


def stage_shared_initial_velocities(system_data_path: Path, work_dir: Path, dt_ps: float, temperature_k: float, seed: int):
    stage_dir = work_dir / "_velocity_stage"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    shutil.copy(system_data_path, stage_dir / "system.data")
    input_contents = "\n".join(
        [
            "units real",
            "atom_style full",
            "read_data system.data",
            f"timestep {dt_ps * 1000}",
            f"velocity all create {temperature_k} {seed} mom yes rot yes dist gaussian",
            "write_dump all custom velocity_init.dump id vx vy vz modify sort id",
        ]
    )
    (stage_dir / "velocity_stage.in").write_text(input_contents + "\n", encoding="utf-8")
    subprocess.run(["lmp", "-in", "velocity_stage.in"], cwd=stage_dir, capture_output=True, text=True, check=True)

    velocities_angstrom_fs = parse_dump_custom_velocities(stage_dir / "velocity_init.dump")
    build_lammps_data_with_velocities(system_data_path, work_dir / "system.data", velocities_angstrom_fs)
    return {
        atom_id: tuple(component * ANGSTROM_PER_FS_TO_NM_PER_PS for component in velocity)
        for atom_id, velocity in velocities_angstrom_fs.items()
    }


def get_mdp_nvt(dt_ps: float, nsteps: int, use_input_velocities: bool):
    return f"""
integrator  = md
dt          = {dt_ps}
nsteps      = {nsteps}
cutoff-scheme = Verlet
vdw-type     = Cut-off
rvdw         = 1.2
coulombtype  = PME
rcoulomb     = 1.2
pbc          = xyz
nstxout      = 0
nstvout      = 0
nstfout      = 0
nstenergy    = 100
nstlog       = 1000
tcoupl      = v-rescale
tc-grps     = System
tau-t       = 0.1
ref-t       = 300
gen-vel     = {'no' if use_input_velocities else 'yes'}
gen-temp    = 300
pcoupl      = no
"""


def get_mdp_npt(dt_ps: float, nsteps: int, use_input_velocities: bool):
    return f"""
integrator  = md
dt          = {dt_ps}
nsteps      = {nsteps}
cutoff-scheme = Verlet
vdw-type     = Cut-off
rvdw         = 1.2
coulombtype  = PME
rcoulomb     = 1.2
pbc          = xyz
nstxout      = 0
nstvout      = 0
nstfout      = 0
nstenergy    = 100
nstlog       = 1000
tcoupl      = v-rescale
tc-grps     = System
tau-t       = 0.1
ref-t       = 300
gen-vel     = {'no' if use_input_velocities else 'yes'}
gen-temp    = 300
pcoupl      = Berendsen
pcoupltype  = isotropic
tau-p       = 1.0
compressibility = 4.5e-5
ref-p       = 1.0
"""


def run_gmx(cmd: list[str], work_dir: Path, log_name: str):
    gmx_bin = str(REPO_ROOT / "build" / "bin" / "gmx")
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    full_cmd = [gmx_bin] + cmd
    result = subprocess.run(full_cmd, cwd=work_dir, capture_output=True, text=True, env=env, errors="replace")
    (work_dir / f"{log_name}.stdout").write_text(result.stdout, encoding="utf-8")
    (work_dir / f"{log_name}.stderr").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"GROMACS command failed: {' '.join(full_cmd)}")


def write_lammps_common_header(handle):
    handle.write(
        """
units real
atom_style full
boundary p p p
pair_style lj/class2/coul/long 9.0 9.0
pair_modify mix sixthpower
bond_style class2
angle_style class2
dihedral_style class2
improper_style none
kspace_style pppm 1.0e-6
special_bonds lj/coul 0.0 0.0 1.0 angle no dihedral no
read_data system.data
pair_coeff 1 1 0.12 3.40
pair_coeff 2 2 0.08 3.00
bond_coeff 1 1.48 230.0 -32.0 7.0
angle_coeff 1 111.0 32.0 -3.6 1.0
angle_coeff 1 bb 5.0 1.48 1.48
angle_coeff 1 ba 1.7 1.4 1.48 1.48
dihedral_coeff 1 0.9 0.0 0.5 180.0 0.3 0.0
dihedral_coeff 1 mbt 0.14 -0.09 0.05 1.48
dihedral_coeff 1 ebt 0.12 -0.06 0.03 0.10 -0.04 0.02 1.48 1.48
dihedral_coeff 1 at 0.05 -0.03 0.02 0.04 -0.02 0.01 111.0 111.0
dihedral_coeff 1 aat 0.22 111.0 111.0
dihedral_coeff 1 bb13 0.16 1.48 1.48
"""
    )


def run_lammps(work_dir: Path, input_name: str):
    subprocess.run(["lmp", "-in", input_name], cwd=work_dir, capture_output=True, text=True, check=True)


def total_mass_amu(parsed_data: dict) -> float:
    masses_by_type = {item["type_id"]: item["mass_amu"] for item in parsed_data["masses"]}
    return sum(masses_by_type[atom["type_id"]] for atom in parsed_data["atoms"])


def density_from_volume_nm3(total_mass_amu_value: float, volumes_nm3: np.ndarray) -> np.ndarray:
    return (total_mass_amu_value * AMU_TO_KG_PER_NM3) / volumes_nm3


def density_from_volume_ang3(total_mass_amu_value: float, volumes_ang3: np.ndarray) -> np.ndarray:
    return (total_mass_amu_value * AMU_TO_KG_PER_ANG3) / volumes_ang3


def internal_density_consistency(raw_density: np.ndarray, recomputed_density: np.ndarray) -> float:
    return float(np.max(np.abs(raw_density - recomputed_density) / np.maximum(np.abs(recomputed_density), 1.0e-12)))


def quarter_mean(values: np.ndarray, which: str) -> float:
    size = len(values) // 4
    if size == 0:
        return float(np.mean(values))
    if which == "first":
        chunk = values[:size]
    else:
        chunk = values[-size:]
    return float(np.mean(chunk))


def relative_drift(values: np.ndarray) -> float:
    mean_value = float(np.mean(values))
    if abs(mean_value) < 1.0e-12:
        return 0.0
    first = quarter_mean(values, "first")
    last = quarter_mean(values, "last")
    return float(abs(last - first) / abs(mean_value))


def build_nvt_parity_report(
    gmx_data: np.ndarray,
    lmp_data: np.ndarray,
    total_mass_amu_value: float,
    initial_volume_nm3: float,
    initial_volume_ang3: float,
    skipped_rows: list[dict],
):
    gmx_mean = np.mean(gmx_data, axis=0)
    lmp_mean = np.mean(lmp_data, axis=0)

    gmx_recomputed_density = density_from_volume_nm3(
        total_mass_amu_value,
        np.full(len(gmx_data), initial_volume_nm3),
    )
    lmp_raw_density = lmp_data[:, 7] * 1000.0
    lmp_recomputed_density = density_from_volume_ang3(total_mass_amu_value, lmp_data[:, 6])

    temp_diff = float(abs(gmx_mean[2] - lmp_mean[1]))
    density_recomputed_diff_rel = float(
        abs(np.mean(gmx_recomputed_density) - np.mean(lmp_recomputed_density)) / abs(np.mean(gmx_recomputed_density))
    )
    lmp_density_consistency = internal_density_consistency(lmp_raw_density, lmp_recomputed_density)

    if (
        temp_diff < 5.0
        and density_recomputed_diff_rel < 0.01
        and lmp_density_consistency < 0.02
    ):
        status = "pass"
    elif temp_diff < 10.0 and density_recomputed_diff_rel < 0.05:
        status = "partial"
    else:
        status = "fail"

    return {
        "status": status,
        "temperature": {
            "gmx_mean_k": float(gmx_mean[2]),
            "lmp_mean_k": float(lmp_mean[1]),
            "diff_k": temp_diff,
        },
        "pressure": {
            "gmx_mean_bar": float(gmx_mean[3]),
            "lmp_mean_bar_assumed": float(lmp_mean[5]),
        },
        "density": {
            "gmx_recomputed_mean_kg_m3": float(np.mean(gmx_recomputed_density)),
            "lmp_raw_mean_kg_m3": float(np.mean(lmp_raw_density)),
            "lmp_recomputed_mean_kg_m3": float(np.mean(lmp_recomputed_density)),
            "gmx_fixed_volume_nm3": initial_volume_nm3,
            "lmp_initial_volume_ang3": initial_volume_ang3,
            "recomputed_diff_rel": density_recomputed_diff_rel,
            "lmp_internal_consistency_max_rel": lmp_density_consistency,
        },
        "potential_energy": {
            "gmx_mean_kj_mol": float(gmx_mean[1]),
            "lmp_mean_kj_mol": float(lmp_mean[2] * KCAL_TO_KJ),
            "note": "Absolute cross-engine PE is diagnostic only; it is not a blocking parity metric.",
        },
        "skipped_lammps_rows": skipped_rows,
    }


def classify_npt_stability(temp_drift_gmx: float, temp_drift_lmp: float, dens_drift_gmx: float, dens_drift_lmp: float):
    if max(temp_drift_gmx, temp_drift_lmp) < 0.05 and max(dens_drift_gmx, dens_drift_lmp) < 0.05:
        return "pass"
    if max(temp_drift_gmx, temp_drift_lmp) < 0.25 and max(dens_drift_gmx, dens_drift_lmp) < 1.00:
        return "partial"
    return "fail"


def build_npt_stability_report(
    gmx_data: np.ndarray,
    lmp_data: np.ndarray,
    total_mass_amu_value: float,
    skipped_rows: list[dict],
):
    gmx_raw_density = gmx_data[:, 5]
    gmx_recomputed_density = density_from_volume_nm3(total_mass_amu_value, gmx_data[:, 4])
    lmp_raw_density = lmp_data[:, 7] * 1000.0
    lmp_recomputed_density = density_from_volume_ang3(total_mass_amu_value, lmp_data[:, 6])

    gmx_temp = gmx_data[:, 2]
    lmp_temp = lmp_data[:, 1]

    gmx_temp_drift = relative_drift(gmx_temp)
    lmp_temp_drift = relative_drift(lmp_temp)
    gmx_density_drift = relative_drift(gmx_recomputed_density)
    lmp_density_drift = relative_drift(lmp_recomputed_density)

    gmx_density_consistency = internal_density_consistency(gmx_raw_density, gmx_recomputed_density)
    lmp_density_consistency = internal_density_consistency(lmp_raw_density, lmp_recomputed_density)

    status = classify_npt_stability(gmx_temp_drift, lmp_temp_drift, gmx_density_drift, lmp_density_drift)

    return {
        "status": status,
        "note": "Short NPT is treated as a stability/trend diagnostic. Full ensemble convergence remains owned by M10.2.1 or later.",
        "gmx": {
            "temperature_mean_k": float(np.mean(gmx_temp)),
            "temperature_drift_rel": gmx_temp_drift,
            "density_raw_mean_kg_m3": float(np.mean(gmx_raw_density)),
            "density_recomputed_mean_kg_m3": float(np.mean(gmx_recomputed_density)),
            "density_drift_rel": gmx_density_drift,
            "volume_mean_nm3": float(np.mean(gmx_data[:, 4])),
            "density_internal_consistency_max_rel": gmx_density_consistency,
        },
        "lmp": {
            "temperature_mean_k": float(np.mean(lmp_temp)),
            "temperature_drift_rel": lmp_temp_drift,
            "density_raw_mean_kg_m3": float(np.mean(lmp_raw_density)),
            "density_recomputed_mean_kg_m3": float(np.mean(lmp_recomputed_density)),
            "density_drift_rel": lmp_density_drift,
            "volume_mean_ang3": float(np.mean(lmp_data[:, 6])),
            "density_internal_consistency_max_rel": lmp_density_consistency,
        },
        "skipped_lammps_rows": skipped_rows,
    }


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def replicate_lammps(system_dir: Path, work_dir: Path, nx: int, ny: int, nz: int):
    data_file = system_dir / "lammps" / "system.data"
    shutil.copy(data_file, work_dir / "orig.data")
    gen_in = work_dir / "replicate.in"
    with gen_in.open("w", encoding="utf-8") as handle:
        handle.write("units real\natom_style full\nread_data orig.data\n")
        handle.write(f"replicate {nx} {ny} {nz}\n")
        handle.write("write_data system.data nocoeff\n")
    subprocess.run(["lmp", "-in", "replicate.in"], cwd=work_dir, capture_output=True, text=True, check=True)


def run_nvt_parity(work_dir: Path, total_mass_amu_value: float, initial_volume_nm3: float, initial_volume_ang3: float):
    print(f"Running medium-scale NVT parity ({NVT_EQUIL_STEPS + NVT_PROD_STEPS} steps total)...")

    (work_dir / "nvt_equil.mdp").write_text(get_mdp_nvt(DEFAULT_DT_PS, NVT_EQUIL_STEPS, use_input_velocities=True), encoding="utf-8")
    run_gmx(
        ["grompp", "-f", "nvt_equil.mdp", "-c", "system.gro", "-p", "system.top", "-o", "nvt_equil.tpr", "-maxwarn", "10"],
        work_dir,
        "grompp_nvt_equil",
    )
    run_gmx(
        ["mdrun", "-s", "nvt_equil.tpr", "-c", "nvt_equil.gro", "-e", "nvt_equil.edr", "-g", "nvt_equil.log", "-cpo", "nvt_equil.cpt", "-nt", "8"],
        work_dir,
        "mdrun_nvt_equil",
    )

    (work_dir / "nvt_run.mdp").write_text(get_mdp_nvt(DEFAULT_DT_PS, NVT_PROD_STEPS, use_input_velocities=True), encoding="utf-8")
    run_gmx(
        ["grompp", "-f", "nvt_run.mdp", "-c", "nvt_equil.gro", "-t", "nvt_equil.cpt", "-p", "system.top", "-o", "nvt_run.tpr", "-maxwarn", "10"],
        work_dir,
        "grompp_nvt",
    )
    run_gmx(
        ["mdrun", "-s", "nvt_run.tpr", "-c", "nvt_out.gro", "-e", "nvt_run.edr", "-g", "nvt_run.log", "-nt", "8"],
        work_dir,
        "mdrun_nvt",
    )
    extract_gmx_energy(work_dir, "nvt_run.edr", "nvt_energy.xvg")
    gmx_data = parse_xvg(work_dir / "nvt_energy.xvg")

    with (work_dir / "nvt_system.in").open("w", encoding="utf-8") as handle:
        write_lammps_common_header(handle)
        handle.write(f"\ntimestep {DEFAULT_DT_PS * 1000}\n")
        handle.write("fix 1 all nvt temp 300.0 300.0 100.0\n")
        handle.write("thermo 100\n")
        handle.write("thermo_style custom step temp pe ke etotal press vol density\n")
        handle.write(f"run {NVT_EQUIL_STEPS + NVT_PROD_STEPS}\n")
    run_lammps(work_dir, "nvt_system.in")
    lmp_data, skipped_rows = parse_lammps_thermo(work_dir / "log.lammps", expected_columns=8)
    if len(gmx_data) == 0 or len(lmp_data) == 0:
        return {
            "status": "fail",
            "reason": "missing_observables",
            "skipped_lammps_rows": skipped_rows,
        }
    return build_nvt_parity_report(
        gmx_data,
        lmp_data[NVT_EQUIL_STEPS // 100 :],
        total_mass_amu_value,
        initial_volume_nm3,
        initial_volume_ang3,
        skipped_rows,
    )


def run_npt_stability(work_dir: Path, total_mass_amu_value: float):
    print(f"Running longer NPT stability diagnostic ({NPT_EQUIL_STEPS + NPT_PROD_STEPS} steps total)...")

    (work_dir / "npt_equil.mdp").write_text(get_mdp_npt(DEFAULT_DT_PS, NPT_EQUIL_STEPS, use_input_velocities=True), encoding="utf-8")
    run_gmx(
        ["grompp", "-f", "npt_equil.mdp", "-c", "system.gro", "-p", "system.top", "-o", "npt_equil.tpr", "-maxwarn", "10"],
        work_dir,
        "grompp_npt_equil",
    )
    run_gmx(
        ["mdrun", "-s", "npt_equil.tpr", "-c", "npt_equil.gro", "-e", "npt_equil.edr", "-g", "npt_equil.log", "-cpo", "npt_equil.cpt", "-nt", "8"],
        work_dir,
        "mdrun_npt_equil",
    )

    (work_dir / "npt_run.mdp").write_text(get_mdp_npt(DEFAULT_DT_PS, NPT_PROD_STEPS, use_input_velocities=True), encoding="utf-8")
    run_gmx(
        ["grompp", "-f", "npt_run.mdp", "-c", "npt_equil.gro", "-t", "npt_equil.cpt", "-p", "system.top", "-o", "npt_run.tpr", "-maxwarn", "10"],
        work_dir,
        "grompp_npt",
    )
    run_gmx(
        ["mdrun", "-s", "npt_run.tpr", "-c", "npt_out.gro", "-e", "npt_run.edr", "-g", "npt_run.log", "-nt", "8"],
        work_dir,
        "mdrun_npt",
    )
    extract_gmx_energy(work_dir, "npt_run.edr", "npt_energy.xvg")
    gmx_data = parse_xvg(work_dir / "npt_energy.xvg")

    with (work_dir / "npt_system.in").open("w", encoding="utf-8") as handle:
        write_lammps_common_header(handle)
        handle.write(f"\ntimestep {DEFAULT_DT_PS * 1000}\n")
        handle.write("fix 1 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0\n")
        handle.write("thermo 100\n")
        handle.write("thermo_style custom step temp pe ke etotal press vol density\n")
        handle.write(f"run {NPT_EQUIL_STEPS + NPT_PROD_STEPS}\n")
    run_lammps(work_dir, "npt_system.in")
    npt_log_path = work_dir / "npt_log.lammps"
    shutil.copy(work_dir / "log.lammps", npt_log_path)
    lmp_data, skipped_rows = parse_lammps_thermo(npt_log_path, expected_columns=8)
    if len(gmx_data) == 0 or len(lmp_data) == 0:
        return {
            "status": "fail",
            "reason": "missing_observables",
            "skipped_lammps_rows": skipped_rows,
        }
    return build_npt_stability_report(gmx_data, lmp_data[NPT_EQUIL_STEPS // 100 :], total_mass_amu_value, skipped_rows)


def build_gate_decision(system_id: str, nvt_report: dict, npt_report: dict):
    required_passes = [f"{system_id}.nvt_parity"]
    non_blocking_diagnostics = [f"{system_id}.npt_stability"]

    if nvt_report["status"] == "pass":
        overall_status = "pass"
    elif nvt_report["status"] == "partial":
        overall_status = "partial"
    else:
        overall_status = "fail"

    failure_reason = None if overall_status == "pass" else "medium_nvt_parity_not_satisfied"

    return {
        "system_id": system_id,
        "evidence_class": "non_exact_diagnostic",
        "integrator_family": "plain_md",
        "required_passes": required_passes,
        "non_blocking_diagnostics": non_blocking_diagnostics,
        "overall_status": overall_status,
        "failure_reason": failure_reason,
        "note": (
            "M10.2 blocks on medium-scale NVT thermal parity. Short NPT remains a longer stability/trend diagnostic "
            "and does not certify transport-production readiness. This runner generates plain `integrator = md` "
            "inputs, so the result is diagnostic-only and not exact-r-RESPA evidence."
        ),
        "reports": {
            "nvt_parity": nvt_report["status"],
            "npt_stability": npt_report["status"],
        },
    }


def check_system(sid: str, corpus_root: Path, results_root: Path):
    work_dir = results_root / sid
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- M10.2 Medium-Scale Ensemble Gate for {sid} ---")

    system_dir = corpus_root / "systems" / "small_oligomer"
    replicate_lammps(system_dir, work_dir, 4, 4, 4)

    parsed_data = parse_lammps_data(work_dir / "system.data")
    total_mass_amu_value = total_mass_amu(parsed_data)
    initial_volume_ang3 = (
        (parsed_data["box"]["x"]["hi"] - parsed_data["box"]["x"]["lo"])
        * (parsed_data["box"]["y"]["hi"] - parsed_data["box"]["y"]["lo"])
        * (parsed_data["box"]["z"]["hi"] - parsed_data["box"]["z"]["lo"])
    )
    initial_volume_nm3 = initial_volume_ang3 * (ANGSTROM_TO_NM**3)

    ir = build_typed_ir({"id": "small_oligomer", "path": "systems/small_oligomer"}, corpus_root)
    top_content = render_gromacs_topology(ir).replace("OLI 1", "OLI 64")
    (work_dir / "system.top").write_text(top_content, encoding="utf-8")

    shared_velocities = stage_shared_initial_velocities(
        work_dir / "system.data", work_dir, DEFAULT_DT_PS, DEFAULT_TEMPERATURE_K, DEFAULT_VELOCITY_SEED
    )
    create_gro_from_lammps(work_dir / "system.data", work_dir / "system.gro", shared_velocities)

    nvt_report = run_nvt_parity(work_dir, total_mass_amu_value, initial_volume_nm3, initial_volume_ang3)
    npt_report = run_npt_stability(work_dir, total_mass_amu_value)

    report = {
        "system_id": sid,
        "evidence_class": "non_exact_diagnostic",
        "integrator_family": "plain_md",
        "non_claimable_statement": "This report is useful for medium-scale thermal and density diagnostics, but it is not exact-r-RESPA evidence because the generated MDPs use plain `integrator = md`.",
        "scale": {
            "atoms": parsed_data["header_counts"]["atoms"],
            "molecules": len({atom["molecule_id"] for atom in parsed_data["atoms"]}),
            "total_mass_amu": total_mass_amu_value,
        },
        "nvt_parity": nvt_report,
        "npt_stability": npt_report,
    }
    dump_json(work_dir / "report.json", report)

    gate_decision = build_gate_decision(sid, nvt_report, npt_report)
    dump_json(results_root / "m10_2_gate_decision.json", gate_decision)

    summary = {
        "system_id": sid,
        "evidence_class": "non_exact_diagnostic",
        "integrator_family": "plain_md",
        "nvt_parity_status": nvt_report["status"],
        "npt_stability_status": npt_report["status"],
        "overall_status": gate_decision["overall_status"],
        "temperature_diff_k": nvt_report.get("temperature", {}).get("diff_k"),
        "density_recomputed_diff_rel": nvt_report.get("density", {}).get("recomputed_diff_rel"),
        "npt_density_drift_rel": npt_report.get("lmp", {}).get("density_drift_rel"),
        "non_claimable_statement": "This is a plain-md medium-scale diagnostic handoff summary, not exact-r-RESPA evidence and not a CPU completion claim.",
    }

    print(f"NVT parity: {nvt_report['status']}")
    print(f"NPT stability: {npt_report['status']}")
    print(f"Overall M10.2 gate: {gate_decision['overall_status']}")
    return summary


def main():
    corpus_root = REPO_ROOT / "testdata" / "lammps_golden"
    results_root = REPO_ROOT / "tests" / "reference_results" / "m10_2_ensemble_gate"
    summary = [check_system("small_oligomer_medium", corpus_root, results_root)]
    dump_json(results_root / "m10_2_summary.json", summary)


if __name__ == "__main__":
    main()
