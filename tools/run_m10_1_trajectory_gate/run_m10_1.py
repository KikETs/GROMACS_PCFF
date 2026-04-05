from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys
import os
import re
import math
import shutil

# Add repo root to sys.path to import from tools
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.pcff_fixture_bridge.common import (
    build_typed_ir,
    render_gromacs_topology,
    parse_lammps_data,
    ANGSTROM_TO_NM,
    KCAL_TO_KJ,
)

KCAL_PER_A_TO_KJ_PER_NM = KCAL_TO_KJ / ANGSTROM_TO_NM
ANGSTROM_PER_FS_TO_NM_PER_PS = 100.0
DEFAULT_VELOCITY_SEED = 20260330
DEFAULT_TEMPERATURE_K = 300.0


def parse_lammps_thermo(log_path: Path, expected_columns: int):
    rows = []
    if not log_path.exists():
        return rows
    with log_path.open("r") as f:
        in_thermo = False
        for line in f:
            clean_line = line.strip()
            if clean_line.startswith("Step"):
                in_thermo = True
                continue
            if clean_line.startswith("Loop time"):
                in_thermo = False
                continue
            if in_thermo:
                parts = clean_line.split()
                if parts and parts[0].isdigit() and len(parts) == expected_columns:
                    try:
                        rows.append([float(x) for x in parts])
                    except ValueError:
                        continue
    return rows


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


def extract_gmx_energy(work_dir: Path, selection: str):
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    subprocess.run(
        [str(REPO_ROOT / "build" / "bin" / "gmx"), "energy", "-f", "run.edr", "-o", "energy.xvg"],
        cwd=work_dir,
        input=selection,
        capture_output=True,
        text=True,
        env=env,
    )


def m10_1_status(system_id: str, protocol: str, dt: float, metrics: dict):
    # Absolute cross-engine PE contains a system-dependent offset and is not a
    # valid short-time parity metric by itself. Use delta/trend agreement for
    # the neutral path and keep the charged-box case as a strict sanity check.
    if system_id == "small_oligomer" and protocol == "nve":
        pe_delta_limit = 3.0 if math.isclose(dt, 0.0001, rel_tol=0, abs_tol=1e-12) else 2.0
        press_delta_limit = 50.0 if math.isclose(dt, 0.0001, rel_tol=0, abs_tol=1e-12) else 100.0
        return (
            "pass"
            if metrics["pe_delta_diff"] <= pe_delta_limit and metrics["press_delta_diff"] <= press_delta_limit
            else "fail"
        )
    if system_id == "small_oligomer" and protocol == "nvt":
        return (
            "pass"
            if metrics["pe_delta_diff"] <= 10.0 and metrics["press_delta_diff"] <= 200.0
            else "fail"
        )
    if system_id == "small_oligomer" and protocol == "npt":
        return (
            "pass"
            if metrics["gmx_volume_drift_rel"] <= 0.02
            and metrics["pe_delta_diff"] <= 5.0
            and metrics["press_delta_diff"] <= 120.0
            else "fail"
        )
    if system_id == "small_salt_polymer_box" and protocol == "nve":
        return (
            "pass"
            if metrics["pe_delta_diff"] <= 5.0 and metrics["press_delta_diff"] <= 75.0
            else "fail"
        )
    if system_id == "small_salt_polymer_box" and protocol == "nvt":
        return (
            "pass"
            if metrics["pe_delta_diff"] <= 50.0 and metrics["press_delta_diff"] <= 300.0
            else "fail"
        )
    return "fail"


def m10_1_overall_status(results: list[dict]) -> str:
    by_id = {result["test_id"]: result for result in results if "test_id" in result}
    required_passes = [
        "small_oligomer_nve_dt0.0001",
        "small_oligomer_nve_dt0.0005",
        "small_salt_polymer_box_nve_dt0.0001",
    ]
    if any(by_id.get(test_id, {}).get("status") != "pass" for test_id in required_passes):
        return "fail"

    # NVT/NPT in this milestone are short-time stability/trend gates, not
    # cross-engine ensemble-parity gates. They may remain fail without blocking
    # handoff, provided no deterministic gate is failing.
    return "pass"

def create_gro_from_lammps(lammps_data_path: Path, gro_path: Path, velocities_nm_ps: dict[int, tuple[float, float, float]] | None = None):
    data = parse_lammps_data(lammps_data_path)
    box_x = (data["box"]["x"]["hi"] - data["box"]["x"]["lo"]) * ANGSTROM_TO_NM
    box_y = (data["box"]["y"]["hi"] - data["box"]["y"]["lo"]) * ANGSTROM_TO_NM
    box_z = (data["box"]["z"]["hi"] - data["box"]["z"]["lo"]) * ANGSTROM_TO_NM
    
    with gro_path.open("w") as f:
        f.write("Generated from LAMMPS data\n")
        f.write(f"{len(data['atoms']):>5d}\n")
        for i, atom in enumerate(data["atoms"], start=1):
            x = atom["x_angstrom"] * ANGSTROM_TO_NM
            y = atom["y_angstrom"] * ANGSTROM_TO_NM
            z = atom["z_angstrom"] * ANGSTROM_TO_NM
            velocity = None if velocities_nm_ps is None else velocities_nm_ps.get(atom["id"])
            if velocity is None:
                f.write(f"{1:>5d}{'MOL':<5s}{f'A{i}':>5s}{atom['id']:>5d}{x:12.8f}{y:12.8f}{z:12.8f}\n")
            else:
                vx, vy, vz = velocity
                f.write(
                    f"{1:>5d}{'MOL':<5s}{f'A{i}':>5s}{atom['id']:>5d}"
                    f"{x:12.8f}{y:12.8f}{z:12.8f}{vx:12.8f}{vy:12.8f}{vz:12.8f}\n"
                )
        f.write(f"{box_x:12.8f}{box_y:12.8f}{box_z:12.8f}\n")


def build_lammps_data_with_velocities(source_data_path: Path, output_data_path: Path, velocities_angstrom_fs: dict[int, tuple[float, float, float]]):
    lines = source_data_path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        if line.split(" #", 1)[0].strip() == "Velocities":
            lines = lines[:idx]
            break
    velocity_lines = ["Velocities", ""]
    for atom_id in sorted(velocities_angstrom_fs):
        vx, vy, vz = velocities_angstrom_fs[atom_id]
        velocity_lines.append(f"{atom_id} {vx:.12f} {vy:.12f} {vz:.12f}")
    output_data_path.write_text("\n".join(lines + [""] + velocity_lines + [""]), encoding="utf-8")


def stage_shared_initial_velocities(system_dir: Path, work_dir: Path, dt: float, temperature_k: float, seed: int):
    stage_dir = work_dir / "_velocity_stage"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    shutil.copy(system_dir / "lammps" / "system.data", stage_dir / "system.data")
    shutil.copy(system_dir / "lammps" / "system.in", stage_dir / "system.in")

    input_contents = "\n".join(
        [
            "log velocity_stage.log",
            "include system.in",
            "reset_timestep 0",
            f"timestep {dt * 1000}",
            f"velocity all create {temperature_k} {seed} mom yes rot yes dist gaussian",
            "write_dump all custom velocity_init.dump id vx vy vz modify sort id",
        ]
    )
    (stage_dir / "velocity_stage.in").write_text(input_contents + "\n", encoding="utf-8")
    subprocess.run(["lmp", "-in", "velocity_stage.in"], cwd=stage_dir, capture_output=True, text=True, check=True)

    velocities_angstrom_fs = parse_dump_custom_velocities(stage_dir / "velocity_init.dump")
    velocities_nm_ps = {
        atom_id: tuple(component * ANGSTROM_PER_FS_TO_NM_PER_PS for component in velocity)
        for atom_id, velocity in velocities_angstrom_fs.items()
    }

    prepared_data_path = work_dir / "system.data"
    build_lammps_data_with_velocities(system_dir / "lammps" / "system.data", prepared_data_path, velocities_angstrom_fs)
    return velocities_nm_ps, prepared_data_path

def get_mdp(protocol: str, dt: float, nsteps: int, has_kspace: bool, use_input_velocities: bool = False):
    coulomb_type = "PME" if has_kspace else "Cut-off"
    
    mdp = f"""
integrator  = md
dt          = {dt}
nsteps      = {nsteps}
cutoff-scheme = Verlet
vdw-type     = Cut-off
rvdw         = 0.9
coulombtype  = {coulomb_type}
rcoulomb     = 0.9
pbc          = xyz
nstxout      = 10
nstvout      = 0
nstfout      = 0
nstenergy    = 10
"""
    if protocol == "nve":
        mdp += "tcoupl = no\npcoupl = no\n"
    elif protocol == "nvt":
        mdp += """
tcoupl      = v-rescale
tc-grps     = System
tau-t       = 0.1
ref-t       = 300
"""
        if use_input_velocities:
            mdp += "gen-vel     = no\n"
        else:
            mdp += "gen-vel     = yes\ngen-temp    = 300\n"
    elif protocol == "npt":
        mdp += """
tcoupl      = v-rescale
tc-grps     = System
tau-t       = 0.1
ref-t       = 300
pcoupl      = Berendsen
pcoupltype  = isotropic
tau-p       = 1.0
compressibility = 4.5e-5
ref-p       = 1.0
"""
        if use_input_velocities:
            mdp += "gen-vel     = no\n"
        else:
            mdp += "gen-vel     = yes\ngen-temp    = 300\n"
    return mdp

def run_gmx(cmd: list[str], work_dir: Path, log_name: str):
    gmx_bin = str(REPO_ROOT / "build" / "bin" / "gmx")
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    
    full_cmd = [gmx_bin] + cmd
    result = subprocess.run(
        full_cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
        env=env,
        errors="replace"
    )
    
    (work_dir / f"{log_name}.stdout").write_text(result.stdout)
    (work_dir / f"{log_name}.stderr").write_text(result.stderr)
    return result.returncode == 0

def run_lammps(system_dir: Path, work_dir: Path, protocol: str, dt: float, nsteps: int, prepared_data_path: Path | None = None):
    orig_in = system_dir / "lammps" / "system.in"
    data_file = prepared_data_path if prepared_data_path is not None else system_dir / "lammps" / "system.data"

    if data_file.resolve() != (work_dir / "system.data").resolve():
        shutil.copy(data_file, work_dir / "system.data")
    
    with orig_in.open("r") as f:
        lines = f.readlines()
    
    new_in = work_dir / "system.in"
    with new_in.open("w") as f:
        for line in lines:
            if line.startswith("read_data"):
                f.write("read_data system.data\n")
            elif line.startswith("timestep"):
                continue
            elif "run" in line and not line.strip().startswith("#"):
                continue
            elif "fix" in line and ("nvt" in line or "nve" in line or "npt" in line):
                continue
            else:
                f.write(line)
        
        f.write(f"\ntimestep {dt * 1000}\n") # GMX dt is ps, LAMMPS real units dt is fs
        
        if protocol == "nve":
            f.write("fix 1 all nve\n")
        elif protocol == "nvt":
            f.write("fix 1 all nvt temp 300.0 300.0 100.0\n")
        elif protocol == "npt":
            f.write("fix 1 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0\n")
            
        f.write("thermo 10\n")
        f.write("thermo_style custom step temp pe ke etotal press vol\n")
        f.write(f"run {nsteps}\n")
        
    subprocess.run(["lmp", "-in", "system.in"], cwd=work_dir, capture_output=True, text=True, check=True)

def parse_xvg(path: Path):
    data = []
    if not path.exists(): return data
    with path.open("r") as f:
        for line in f:
            if line.startswith(("#", "@")): continue
            parts = line.split()
            if parts:
                data.append([float(x) for x in parts])
    return data

def check_system(system_id: str, protocol: str, dt: float, nsteps: int, corpus_root: Path, results_root: Path):
    test_id = f"{system_id}_{protocol}_dt{dt}"
    work_dir = results_root / test_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- M10.1: {test_id} ---")
    
    system_dir = corpus_root / "systems" / system_id
    typed_ir = build_typed_ir({"id": system_id, "path": f"systems/{system_id}"}, corpus_root)
    has_kspace = typed_ir["styles"]["kspace_style"] is not None
    use_shared_initial_velocities = protocol in {"nvt", "npt"}
    shared_velocities = None
    prepared_data_path = None
    if use_shared_initial_velocities:
        shared_velocities, prepared_data_path = stage_shared_initial_velocities(
            system_dir, work_dir, dt, DEFAULT_TEMPERATURE_K, DEFAULT_VELOCITY_SEED
        )
    
    # GMX
    (work_dir / "system.top").write_text(render_gromacs_topology(typed_ir))
    create_gro_from_lammps(system_dir / "lammps" / "system.data", work_dir / "system.gro", shared_velocities)
    (work_dir / "run.mdp").write_text(get_mdp(protocol, dt, nsteps, has_kspace, use_shared_initial_velocities))
    
    success = run_gmx(["grompp", "-f", "run.mdp", "-c", "system.gro", "-p", "system.top", "-o", "run.tpr", "-maxwarn", "10"], work_dir, "grompp")
    if not success: return {"test_id": test_id, "status": "fail", "reason": "grompp"}
    
    success = run_gmx(["mdrun", "-s", "run.tpr", "-c", "out.gro", "-e", "run.edr", "-g", "run.log", "-o", "run.trr"], work_dir, "mdrun")
    if not success: return {"test_id": test_id, "status": "fail", "reason": "mdrun"}
    
    # Extract GMX energies
    energy_input = "Potential\nKinetic-En.\nTemperature\nPressure\nVolume\n0\n"
    extract_gmx_energy(work_dir, energy_input)
    gmx_data = parse_xvg(work_dir / "energy.xvg")
    
    # LAMMPS
    run_lammps(system_dir, work_dir, protocol, dt, nsteps, prepared_data_path)
    # Parse log.lammps
    lmp_data = parse_lammps_thermo(work_dir / "log.lammps", expected_columns=7)
    if not gmx_data or not lmp_data:
        return {"test_id": test_id, "status": "fail", "reason": "missing_observables"}
                
    # Summary
    # Step Temp PotEng KinEng TotEng Press Volume (LAMMPS)
    # 0    1    2      3      4      5     6
    # Time Pot  Kin    Temp   Press  Vol (GMX)
    # 0    1    2      3      4      5
    
    # Compare start/end PE
    lmp_pe_start = lmp_data[0][2] * KCAL_TO_KJ
    lmp_pe_end = lmp_data[-1][2] * KCAL_TO_KJ
    gmx_pe_start = gmx_data[0][1]
    gmx_pe_end = gmx_data[-1][1]
    
    gmx_vol_start = gmx_data[0][5] if len(gmx_data[0]) > 5 else None
    gmx_vol_end = gmx_data[-1][5] if len(gmx_data[-1]) > 5 else None
    metrics = {
        "pe_start_diff": abs(lmp_pe_start - gmx_pe_start),
        "pe_end_diff": abs(lmp_pe_end - gmx_pe_end),
        "pe_offset_start": lmp_pe_start - gmx_pe_start,
        "pe_offset_end": lmp_pe_end - gmx_pe_end,
        "lmp_pe_delta": lmp_pe_end - lmp_pe_start,
        "gmx_pe_delta": gmx_pe_end - gmx_pe_start,
        "pe_delta_diff": abs((lmp_pe_end - lmp_pe_start) - (gmx_pe_end - gmx_pe_start)),
        "lmp_press_delta": lmp_data[-1][5] - lmp_data[0][5],
        "gmx_press_delta": gmx_data[-1][4] - gmx_data[0][4],
        "press_delta_diff": abs((lmp_data[-1][5] - lmp_data[0][5]) - (gmx_data[-1][4] - gmx_data[0][4])),
        "gmx_volume_drift_rel": (
            abs(gmx_vol_end - gmx_vol_start) / abs(gmx_vol_start)
            if gmx_vol_start not in (None, 0.0) and gmx_vol_end is not None
            else None
        ),
    }

    report = {
        "test_id": test_id,
        "status": m10_1_status(system_id, protocol, dt, metrics),
        "dt": dt,
        "pe_start": {"lmp": lmp_pe_start, "gmx": gmx_pe_start, "diff": metrics["pe_start_diff"]},
        "pe_end": {"lmp": lmp_pe_end, "gmx": gmx_pe_end, "diff": metrics["pe_end_diff"]},
        "pe_offset_start": metrics["pe_offset_start"],
        "pe_offset_end": metrics["pe_offset_end"],
        "lmp_pe_delta": metrics["lmp_pe_delta"],
        "gmx_pe_delta": metrics["gmx_pe_delta"],
        "pe_delta_diff": metrics["pe_delta_diff"],
        "lmp_press_delta": metrics["lmp_press_delta"],
        "gmx_press_delta": metrics["gmx_press_delta"],
        "press_delta_diff": metrics["press_delta_diff"],
        "gmx_volume_drift_rel": metrics["gmx_volume_drift_rel"],
    }
    dump_json(work_dir / "report.json", report)
    return report

def dump_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

def main():
    corpus_root = REPO_ROOT / "testdata" / "lammps_golden"
    results_root = REPO_ROOT / "tests" / "reference_results" / "m10_1_trajectory_gate"
    
    # Protocols for neutral system (small_oligomer)
    results = []
    # NVE for timestep convergence
    results.append(check_system("small_oligomer", "nve", 0.0001, 100, corpus_root, results_root)) # 0.1 fs
    results.append(check_system("small_oligomer", "nve", 0.0005, 100, corpus_root, results_root)) # 0.5 fs
    # NVT for trajectory trend
    results.append(check_system("small_oligomer", "nvt", 0.001, 100, corpus_root, results_root)) # 1.0 fs
    # NPT for barostat sanity
    results.append(check_system("small_oligomer", "npt", 0.001, 100, corpus_root, results_root))
    
    # Charged-system deterministic gate to separate core dynamics from thermostat mismatch
    results.append(check_system("small_salt_polymer_box", "nve", 0.0001, 100, corpus_root, results_root))
    # Charged-system NVT sanity
    results.append(check_system("small_salt_polymer_box", "nvt", 0.001, 100, corpus_root, results_root))
    
    dump_json(results_root / "m10_1_summary.json", results)
    dump_json(
        results_root / "m10_1_gate_decision.json",
        {
            "status": m10_1_overall_status(results),
            "required_passes": [
                "small_oligomer_nve_dt0.0001",
                "small_oligomer_nve_dt0.0005",
                "small_salt_polymer_box_nve_dt0.0001",
            ],
            "non_blocking_stability_gates": [
                "small_oligomer_nvt_dt0.001",
                "small_oligomer_npt_dt0.001",
                "small_salt_polymer_box_nvt_dt0.001",
            ],
            "note": "M10.1 blocks only on deterministic NVE gates. NVT/NPT remain short-time stability/trend diagnostics and are handed off to M10.2 for ensemble parity.",
        },
    )

if __name__ == "__main__":
    main()
