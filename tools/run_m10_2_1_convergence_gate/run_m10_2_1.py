#!/home/kiket/anaconda3/envs/MD/bin/python3
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
AMU_TO_KG_PER_NM3 = 1.66053906660
AMU_TO_KG_PER_ANG3 = 1660.53906660
DEFAULT_TEMPERATURE_K = 300.0
DEFAULT_VELOCITY_SEED = 20260330

DEFAULT_DT_PS = 0.002
NPT_EQUIL_STEPS = 5000
NPT_PROD_STEPS = 50000
N_BLOCKS = 5


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
            residue_id = atom["molecule_id"] % 100000
            atom_name = atom_names[atom["id"]]
            velocity = None if velocities_nm_ps is None else velocities_nm_ps.get(atom["id"])
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


def run_gmx(cmd: list[str], work_dir: Path, log_name: str, nthreads: int = 12):
    gmx_bin = str(REPO_ROOT / "build" / "bin" / "gmx")
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"

    full_cmd = [gmx_bin] + cmd
    if "mdrun" in cmd:
        full_cmd += ["-nt", str(nthreads)]

    result = subprocess.run(full_cmd, cwd=work_dir, capture_output=True, text=True, env=env, errors="replace")
    (work_dir / f"{log_name}.stdout").write_text(result.stdout, encoding="utf-8")
    (work_dir / f"{log_name}.stderr").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"GROMACS command failed: {' '.join(full_cmd)}")


def run_gmx_async(cmd: list[str], work_dir: Path, log_name: str, nthreads: int = 12):
    gmx_bin = str(REPO_ROOT / "build" / "bin" / "gmx")
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    full_cmd = [gmx_bin] + cmd
    if "mdrun" in cmd:
        full_cmd += ["-nt", str(nthreads)]
    return subprocess.Popen(
        full_cmd,
        cwd=work_dir,
        stdout=open(work_dir / f"{log_name}.stdout", "w", encoding="utf-8"),
        stderr=open(work_dir / f"{log_name}.stderr", "w", encoding="utf-8"),
        env=env,
    )


def run_lammps_async(work_dir: Path, dt_ps: float, nsteps: int, nprocs: int = 12):
    with (work_dir / "system.in").open("w", encoding="utf-8") as handle:
        handle.write(
            f"""
units real
atom_style full
boundary p p p
pair_style lj/class2/coul/long 9.0 9.0
pair_modify mix sixthpower
bond_style class2
angle_style class2
dihedral_style class2
improper_style none
kspace_style pppm 1.0e-4
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
timestep {dt_ps * 1000}
fix 1 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0
thermo 100
thermo_style custom step temp pe ke etotal press vol density
run {nsteps}
"""
        )
    cmd = ["mpirun", "-np", str(nprocs), "lmp", "-in", "system.in"]
    return subprocess.Popen(
        cmd,
        cwd=work_dir,
        stdout=open(work_dir / "lammps.stdout", "w", encoding="utf-8"),
        stderr=open(work_dir / "lammps.stderr", "w", encoding="utf-8"),
    )


def extract_gmx_energy(work_dir: Path, edr_name: str, output_name: str):
    selection = "Potential\nTemperature\nPressure\nVolume\nDensity\n0\n"
    subprocess.run(
        [str(REPO_ROOT / "build" / "bin" / "gmx"), "energy", "-f", edr_name, "-o", output_name],
        cwd=work_dir,
        input=selection,
        capture_output=True,
        text=True,
        check=True,
    )


def total_mass_amu(parsed_data: dict) -> float:
    masses_by_type = {item["type_id"]: item["mass_amu"] for item in parsed_data["masses"]}
    return sum(masses_by_type[atom["type_id"]] for atom in parsed_data["atoms"])


def density_from_volume_nm3(total_mass_amu_value: float, volumes_nm3: np.ndarray) -> np.ndarray:
    return (total_mass_amu_value * AMU_TO_KG_PER_NM3) / volumes_nm3


def density_from_volume_ang3(total_mass_amu_value: float, volumes_ang3: np.ndarray) -> np.ndarray:
    return (total_mass_amu_value * AMU_TO_KG_PER_ANG3) / volumes_ang3


def internal_density_consistency(raw_density: np.ndarray, recomputed_density: np.ndarray) -> float:
    return float(np.max(np.abs(raw_density - recomputed_density) / np.maximum(np.abs(recomputed_density), 1.0e-12)))


def block_statistics(values: np.ndarray, nblocks: int):
    block_size = len(values) // nblocks
    if block_size == 0:
        raise ValueError("Not enough samples for block statistics")
    blocks = [values[i * block_size : (i + 1) * block_size] for i in range(nblocks)]
    block_means = np.array([float(np.mean(block)) for block in blocks])
    sem = float(np.std(block_means, ddof=0) / np.sqrt(nblocks))
    mean_value = float(np.mean(values))
    if abs(mean_value) < 1.0e-12:
        drift_rel = 0.0
    else:
        drift_rel = float(abs(block_means[-1] - block_means[0]) / abs(mean_value))
    return {
        "mean": mean_value,
        "sem": sem,
        "block_means": block_means.tolist(),
        "drift_rel": drift_rel,
    }


def classify_temperature_convergence(drift_rel: float):
    if drift_rel < 0.05:
        return "pass"
    if drift_rel < 0.10:
        return "partial"
    return "fail"


def classify_density_convergence(drift_rel: float):
    if drift_rel < 0.05:
        return "pass"
    if drift_rel < 0.25:
        return "partial"
    return "fail"


def summarize_engine(gmx_or_lmp: str, data: np.ndarray, total_mass_amu_value: float):
    if gmx_or_lmp == "gmx":
        potential = data[:, 1]
        temperature = data[:, 2]
        pressure = data[:, 3]
        volume = data[:, 4]
        raw_density = data[:, 5]
        recomputed_density = density_from_volume_nm3(total_mass_amu_value, volume)
    else:
        potential = data[:, 2] * KCAL_TO_KJ
        temperature = data[:, 1]
        pressure = data[:, 5]
        volume = data[:, 6]
        raw_density = data[:, 7] * 1000.0
        recomputed_density = density_from_volume_ang3(total_mass_amu_value, volume)

    potential_stats = block_statistics(potential, N_BLOCKS)
    temperature_stats = block_statistics(temperature, N_BLOCKS)
    density_stats = block_statistics(recomputed_density, N_BLOCKS)

    return {
        "potential_energy_kj_mol": potential_stats,
        "temperature_k": {
            **temperature_stats,
            "status": classify_temperature_convergence(temperature_stats["drift_rel"]),
        },
        "pressure": block_statistics(pressure, N_BLOCKS),
        "volume": block_statistics(volume, N_BLOCKS),
        "density_kg_m3": {
            **density_stats,
            "raw_mean": float(np.mean(raw_density)),
            "internal_consistency_max_rel": internal_density_consistency(raw_density, recomputed_density),
            "status": classify_density_convergence(density_stats["drift_rel"]),
        },
    }


def overall_status(gmx_summary: dict, lmp_summary: dict):
    temp_diff = abs(gmx_summary["temperature_k"]["mean"] - lmp_summary["temperature_k"]["mean"])
    dens_diff_rel = abs(gmx_summary["density_kg_m3"]["mean"] - lmp_summary["density_kg_m3"]["mean"]) / max(
        abs(gmx_summary["density_kg_m3"]["mean"]), 1.0e-12
    )

    gmx_temp_status = gmx_summary["temperature_k"]["status"]
    lmp_temp_status = lmp_summary["temperature_k"]["status"]
    gmx_density_status = gmx_summary["density_kg_m3"]["status"]
    lmp_density_status = lmp_summary["density_kg_m3"]["status"]

    if (
        temp_diff < 5.0
        and dens_diff_rel < 0.25
        and gmx_temp_status == "pass"
        and lmp_temp_status == "pass"
        and gmx_density_status in {"pass", "partial"}
        and lmp_density_status in {"pass", "partial"}
    ):
        return "pass", temp_diff, dens_diff_rel
    if temp_diff < 10.0 and dens_diff_rel < 1.0:
        return "partial", temp_diff, dens_diff_rel
    return "fail", temp_diff, dens_diff_rel


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def check_system(sid: str, results_root: Path):
    work_dir = results_root / sid
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- M10.2.1 Convergence Gate for {sid} ---")
    system_dir = REPO_ROOT / "testdata" / "lammps_golden" / "systems" / "small_oligomer"

    shutil.copy(system_dir / "lammps" / "system.data", work_dir / "orig.data")
    with (work_dir / "replicate.in").open("w", encoding="utf-8") as handle:
        handle.write("units real\natom_style full\nread_data orig.data\nreplicate 4 4 4\nwrite_data system.data nocoeff\n")
    subprocess.run(["lmp", "-in", "replicate.in"], cwd=work_dir, capture_output=True, text=True, check=True)

    parsed_data = parse_lammps_data(work_dir / "system.data")
    total_mass_amu_value = total_mass_amu(parsed_data)

    ir = build_typed_ir({"id": "small_oligomer", "path": "systems/small_oligomer"}, REPO_ROOT / "testdata" / "lammps_golden")
    top_content = render_gromacs_topology(ir).replace("OLI 1", "OLI 64")
    (work_dir / "system.top").write_text(top_content, encoding="utf-8")

    shared_velocities = stage_shared_initial_velocities(
        work_dir / "system.data", work_dir, DEFAULT_DT_PS, DEFAULT_TEMPERATURE_K, DEFAULT_VELOCITY_SEED
    )
    create_gro_from_lammps(work_dir / "system.data", work_dir / "system.gro", shared_velocities)

    (work_dir / "equil.mdp").write_text(get_mdp_npt(DEFAULT_DT_PS, NPT_EQUIL_STEPS, use_input_velocities=True), encoding="utf-8")
    run_gmx(
        ["grompp", "-f", "equil.mdp", "-c", "system.gro", "-p", "system.top", "-o", "equil.tpr", "-maxwarn", "10"],
        work_dir,
        "grompp_equil",
    )
    run_gmx(
        ["mdrun", "-s", "equil.tpr", "-c", "equil.gro", "-e", "equil.edr", "-g", "equil.log", "-cpo", "equil.cpt"],
        work_dir,
        "mdrun_equil",
    )

    print("Starting GROMACS and LAMMPS in parallel (100 ps NPT production)...")
    (work_dir / "run.mdp").write_text(get_mdp_npt(DEFAULT_DT_PS, NPT_PROD_STEPS, use_input_velocities=True), encoding="utf-8")
    run_gmx(
        ["grompp", "-f", "run.mdp", "-c", "equil.gro", "-t", "equil.cpt", "-p", "system.top", "-o", "run.tpr", "-maxwarn", "10"],
        work_dir,
        "grompp",
    )

    p_gmx = run_gmx_async(["mdrun", "-s", "run.tpr", "-c", "out.gro", "-e", "run.edr", "-g", "run.log"], work_dir, "mdrun")
    p_lmp = run_lammps_async(work_dir, DEFAULT_DT_PS, NPT_EQUIL_STEPS + NPT_PROD_STEPS)

    gmx_rc = p_gmx.wait()
    lmp_rc = p_lmp.wait()
    if gmx_rc != 0 or lmp_rc != 0:
        raise RuntimeError(f"Production failed: gmx_rc={gmx_rc}, lmp_rc={lmp_rc}")

    extract_gmx_energy(work_dir, "run.edr", "energy.xvg")
    gmx_data = parse_xvg(work_dir / "energy.xvg")
    lmp_data, skipped_rows = parse_lammps_thermo(work_dir / "log.lammps", expected_columns=8)
    lmp_prod = lmp_data[NPT_EQUIL_STEPS // 100 :]

    gmx_summary = summarize_engine("gmx", gmx_data, total_mass_amu_value)
    lmp_summary = summarize_engine("lmp", lmp_prod, total_mass_amu_value)
    status, temp_diff, dens_diff_rel = overall_status(gmx_summary, lmp_summary)

    report = {
        "system_id": sid,
        "evidence_class": "non_exact_diagnostic",
        "integrator_family": "plain_md",
        "duration_ps": NPT_PROD_STEPS * DEFAULT_DT_PS,
        "equilibration_ps": NPT_EQUIL_STEPS * DEFAULT_DT_PS,
        "total_mass_amu": total_mass_amu_value,
        "gmx": gmx_summary,
        "lmp": lmp_summary,
        "cross_engine": {
            "temperature_diff_k": temp_diff,
            "density_diff_rel": dens_diff_rel,
            "status": status,
        },
        "skipped_lammps_rows": skipped_rows,
        "note": "M10.2.1 evaluates longer-horizon NPT convergence and does not by itself certify transport-production readiness. This runner generates plain `integrator = md` inputs, so the result is diagnostic-only and not exact-r-RESPA evidence.",
    }

    gate_decision = {
        "system_id": sid,
        "evidence_class": "non_exact_diagnostic",
        "integrator_family": "plain_md",
        "required_passes": [
            f"{sid}.temperature_convergence",
            f"{sid}.density_convergence",
        ],
        "overall_status": status,
        "failure_reason": None if status == "pass" else "npt_convergence_not_yet_sufficient",
        "note": "Longer NPT convergence is necessary before conductivity production, but even pass here is still not a transport-production claim. This is a plain-md convergence diagnostic, not exact-r-RESPA evidence.",
    }

    dump_json(work_dir / "report.json", report)
    dump_json(results_root / "m10_2_1_gate_decision.json", gate_decision)

    print(f"Convergence status: {status}")
    print(f"  GMX temp mean: {gmx_summary['temperature_k']['mean']:.2f} K ({gmx_summary['temperature_k']['status']})")
    print(f"  LMP temp mean: {lmp_summary['temperature_k']['mean']:.2f} K ({lmp_summary['temperature_k']['status']})")
    print(f"  GMX density mean: {gmx_summary['density_kg_m3']['mean']:.2f} kg/m^3 ({gmx_summary['density_kg_m3']['status']})")
    print(f"  LMP density mean: {lmp_summary['density_kg_m3']['mean']:.2f} kg/m^3 ({lmp_summary['density_kg_m3']['status']})")

    return {
        "system_id": sid,
        "evidence_class": "non_exact_diagnostic",
        "integrator_family": "plain_md",
        "duration_ps": NPT_PROD_STEPS * DEFAULT_DT_PS,
        "overall_status": status,
        "temperature_diff_k": temp_diff,
        "density_diff_rel": dens_diff_rel,
        "gmx_density_drift_rel": gmx_summary["density_kg_m3"]["drift_rel"],
        "lmp_density_drift_rel": lmp_summary["density_kg_m3"]["drift_rel"],
        "non_claimable_statement": "This is a plain-md longer-horizon convergence diagnostic, not exact-r-RESPA evidence and not a production handoff.",
    }


def main():
    results_root = REPO_ROOT / "tests" / "reference_results" / "m10_2_1_convergence_gate"
    results_root.mkdir(parents=True, exist_ok=True)
    summary = [check_system("small_oligomer_medium_100ps", results_root)]
    dump_json(results_root / "m10_2_1_summary.json", summary)


if __name__ == "__main__":
    main()
