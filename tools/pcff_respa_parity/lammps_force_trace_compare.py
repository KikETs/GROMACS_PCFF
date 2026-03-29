from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_DIR = REPO_ROOT / "build-worktree"
if not DEFAULT_BUILD_DIR.exists():
    DEFAULT_BUILD_DIR = REPO_ROOT / "build"
DEFAULT_GMX = DEFAULT_BUILD_DIR / "bin" / "gmx"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "tests" / "reference_results" / "m6_respa"
DEFAULT_OUT_ROOT = DEFAULT_REFERENCE_ROOT / "lammps_force_trace_compare_last"
DEFAULT_SYSTEMS = ("small_oligomer", "small_salt_polymer_box")
ENERGY_INTERVAL = 4
FOURIER_SPACING_NM = 0.08
DEFAULT_LEVEL_STEP_FACTORS = (1, 2, 4)
DEFAULT_MAX_ABS_TOL = 1.5
DEFAULT_MAX_ATOM_NORM_TOL = 2.0
DEFAULT_RMS_TOL = 0.35

NATOMS_RE = re.compile(r"natoms=\s*(\d+)\s+step=\s*(-?\d+)\s+time=([0-9eE+.\-]+)")
FORCE_RE = re.compile(r"^\s*f\[\s*(\d+)\]=\{\s*([0-9eE+.\-]+),\s*([0-9eE+.\-]+),\s*([0-9eE+.\-]+)\}")
LAMMPS_ATOM_LINE_RE = re.compile(r"^\d+\s+\d+\s+\d+\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare standalone exact-r-RESPA outer-step total force traces directly against LAMMPS run 0."
    )
    parser.add_argument(
        "--gmx",
        default=str(DEFAULT_GMX),
        help="Path to the GROMACS CLI binary.",
    )
    parser.add_argument(
        "--lammps-cmd",
        default=shutil.which("lmp") or "",
        help="LAMMPS executable used for direct run-0 force probes.",
    )
    parser.add_argument(
        "--reference-root",
        default=str(DEFAULT_REFERENCE_ROOT),
        help="Directory containing the frozen M6 exact-r-RESPA fixtures.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_ROOT),
        help="Directory for temporary runs and JSON summaries.",
    )
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="System id to compare. Repeat to select multiple systems.",
    )
    parser.add_argument(
        "--outer-steps",
        type=int,
        default=5,
        help="Number of outer r-RESPA steps for the exact diagnostic run.",
    )
    parser.add_argument(
        "--pair14-level",
        type=int,
        default=1,
        help="Value for exact-respa-pair14-level in the exact diagnostic run.",
    )
    parser.add_argument(
        "--build-target",
        default="gmx",
        help="CMake target to build before running diagnostics.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the build step.",
    )
    parser.add_argument(
        "--max-abs-tol",
        type=float,
        default=DEFAULT_MAX_ABS_TOL,
        help="Maximum allowed per-component force delta in kJ/mol/nm.",
    )
    parser.add_argument(
        "--max-atom-norm-tol",
        type=float,
        default=DEFAULT_MAX_ATOM_NORM_TOL,
        help="Maximum allowed per-atom force-vector norm delta in kJ/mol/nm.",
    )
    parser.add_argument(
        "--rms-tol",
        type=float,
        default=DEFAULT_RMS_TOL,
        help="Maximum allowed RMS per-component force delta in kJ/mol/nm.",
    )
    return parser.parse_args()


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> None:
    subprocess.run(
        command,
        cwd=cwd,
        env=env,
        input=stdin,
        text=(stdin is not None),
        check=True,
    )


def run_command_capture(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)
    return completed.stdout


def write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def exact_respa_mdp(outer_steps: int, pair14_level: int) -> str:
    nsteps = outer_steps * ENERGY_INTERVAL
    return (
        "title                   = pcff exact respa lammps force trace compare\n"
        "integrator              = md-vv\n"
        "dt                      = 0.0005\n"
        f"nsteps                  = {nsteps}\n"
        "constraints             = none\n"
        "cutoff-scheme           = Verlet\n"
        f"nstlist                 = {ENERGY_INTERVAL}\n"
        "rlist                   = 0.99\n"
        "rvdw                    = 0.9\n"
        "rcoulomb                = 0.9\n"
        "vdwtype                 = Cut-off\n"
        "vdw-modifier            = none\n"
        "coulombtype             = PME\n"
        "coulomb-modifier        = none\n"
        "ewald-rtol              = 1e-6\n"
        "pme-order               = 4\n"
        f"fourierspacing          = {FOURIER_SPACING_NM}\n"
        "epsilon-r               = 1\n"
        "pbc                     = xyz\n"
        "tcoupl                  = no\n"
        "pcoupl                  = no\n"
        "comm-mode               = none\n"
        "verlet-buffer-tolerance = -1\n"
        "gen-vel                 = no\n"
        "exact-respa             = yes\n"
        "exact-respa-levels      = 3\n"
        "exact-respa-level2-factor = 2\n"
        "exact-respa-level3-factor = 4\n"
        "exact-respa-bond-level  = 1\n"
        "exact-respa-angle-level = 1\n"
        "exact-respa-dihedral-level = 1\n"
        "exact-respa-improper-level = 1\n"
        f"exact-respa-pair14-level = {pair14_level}\n"
        "exact-respa-pair-level  = 3\n"
        "exact-respa-kspace-level = 3\n"
        "exact-respa-inner-level = 1\n"
        "exact-respa-middle-level = 2\n"
        "exact-respa-outer-level = 3\n"
        "exact-respa-inner-off   = 0.30\n"
        "exact-respa-inner-on    = 0.45\n"
        "exact-respa-outer-on    = 0.60\n"
        "exact-respa-outer-off   = 0.80\n"
        f"nstcalcenergy           = {ENERGY_INTERVAL}\n"
        f"nstenergy               = {ENERGY_INTERVAL}\n"
        f"nstlog                  = {ENERGY_INTERVAL}\n"
        f"nstxout                 = {ENERGY_INTERVAL}\n"
        "nstvout                 = 0\n"
        f"nstfout                 = {ENERGY_INTERVAL}\n"
        "nstxout-compressed      = 0\n"
    )


def system_paths(reference_root: Path, system_id: str) -> tuple[Path, Path]:
    system_root = reference_root / system_id
    return system_root / "initial_nve.gro", system_root / "topol.top"


def lammps_system_paths(system_id: str) -> tuple[Path, Path]:
    system_root = REPO_ROOT / "testdata" / "lammps_golden" / "systems" / system_id / "lammps"
    return system_root / "system.in", system_root / "system.data"


def parse_gro(path: Path) -> tuple[list[tuple[int, float, float, float]], list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    num_atoms = int(lines[1].strip())
    atoms: list[tuple[int, float, float, float]] = []
    for line in lines[2 : 2 + num_atoms]:
        atomnr = int(line[15:20])
        x, y, z = map(float, line[20:].split()[:3])
        atoms.append((atomnr, x, y, z))
    box = [float(value) for value in lines[2 + num_atoms].split()[:3]]
    return atoms, box


def parse_g96(path: Path) -> tuple[list[tuple[int, float, float, float]], list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    atoms: list[tuple[int, float, float, float]] = []
    box: list[float] = []
    in_position = False
    in_box = False

    for line in lines:
        stripped = line.strip()
        if stripped == "POSITION":
            in_position = True
            in_box = False
            continue
        if stripped == "BOX":
            in_position = False
            in_box = True
            continue
        if stripped == "END":
            in_position = False
            in_box = False
            continue

        if in_position and stripped:
            columns = line.split()
            atomnr = int(columns[3])
            atoms.append((atomnr, float(columns[4]), float(columns[5]), float(columns[6])))
        elif in_box and stripped:
            box = [float(value) for value in stripped.split()[:3]]

    if not atoms or len(box) != 3:
        raise ValueError(f"Could not parse POSITION/BOX blocks from {path}")
    return atoms, box


def parse_coordinate_file(path: Path) -> tuple[list[tuple[int, float, float, float]], list[float]]:
    if path.suffix == ".g96":
        return parse_g96(path)
    return parse_gro(path)


def highest_active_exact_respa_level(step: int, level_step_factors: tuple[int, ...]) -> int:
    highest_level = 0
    for level, step_factor in enumerate(level_step_factors[1:], start=1):
        if step_factor <= 0:
            raise ValueError(f"Invalid exact-respa step factor at level {level}: {step_factor}")
        if step % step_factor == 0:
            highest_level = level
        else:
            break
    return highest_level


def parse_trr_force_frames(
    gmx_binary: Path, workdir: Path, trr_path: Path, level_step_factors: tuple[int, ...]
) -> dict[int, dict[str, object]]:
    dump_output = run_command_capture([str(gmx_binary), "dump", "-f", str(trr_path)], cwd=workdir)
    frames: dict[int, dict[str, object]] = {}
    current_step: int | None = None
    current_time = 0.0
    in_force_block = False

    for line in dump_output.splitlines():
        natoms_match = NATOMS_RE.search(line)
        if natoms_match:
            current_step = int(natoms_match.group(2))
            current_time = float(natoms_match.group(3))
            frames[current_step] = {
                "time_ps": current_time,
                "highest_active_level": highest_active_exact_respa_level(current_step, level_step_factors),
                "forces": {},
            }
            in_force_block = False
            continue

        if current_step is None:
            continue

        if "   f (" in line:
            in_force_block = True
            continue

        if in_force_block:
            force_match = FORCE_RE.match(line)
            if force_match:
                atom = int(force_match.group(1))
                frames[current_step]["forces"][atom] = [
                    float(force_match.group(2)),
                    float(force_match.group(3)),
                    float(force_match.group(4)),
                ]
                continue
            if line.startswith("   ") and not line.lstrip().startswith("f["):
                in_force_block = False

    return frames


def parse_force_dump(path: Path) -> dict[int, dict[str, object]]:
    frames: dict[int, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            columns = stripped.split("\t")
            if len(columns) != 7:
                raise ValueError(f"Unexpected force dump line in {path}: {stripped}")
            step = int(columns[0])
            time = float(columns[1])
            highest_active_level = int(columns[2])
            atom = int(columns[3])
            force = [float(columns[4]), float(columns[5]), float(columns[6])]
            frame = frames.setdefault(
                step,
                {"time_ps": time, "highest_active_level": highest_active_level, "forces": {}},
            )
            frame["forces"][atom] = force
    return frames


def rewrite_lammps_data_with_coordinates(system_data: Path, coordinate_file: Path, out_data: Path) -> None:
    coordinates, box_nm = parse_coordinate_file(coordinate_file)
    original_lines = system_data.read_text(encoding="utf-8").splitlines()

    out_lines: list[str] = []
    in_atoms_section = False
    atom_index = 0
    num_atoms = len(coordinates)

    for line in original_lines:
        stripped = line.strip()
        if stripped == "Atoms # full":
            in_atoms_section = True
            out_lines.append(line)
            continue
        if in_atoms_section and atom_index < num_atoms and stripped == "":
            out_lines.append(line)
            continue
        if in_atoms_section and atom_index < num_atoms and stripped and LAMMPS_ATOM_LINE_RE.match(stripped):
            atomnr, molecule, atom_type, charge = stripped.split()[:4]
            coordinate = coordinates[atom_index]
            if int(atomnr) != coordinate[0]:
                raise ValueError(
                    f"LAMMPS atom id {atomnr} does not match GROMACS atom id {coordinate[0]} in {coordinate_file}"
                )
            x_angstrom = (coordinate[1] - box_nm[0] / 2.0) * 10.0
            y_angstrom = (coordinate[2] - box_nm[1] / 2.0) * 10.0
            z_angstrom = (coordinate[3] - box_nm[2] / 2.0) * 10.0
            out_lines.append(
                f"{atomnr} {molecule} {atom_type} {charge} {x_angstrom:.9f} {y_angstrom:.9f} {z_angstrom:.9f}"
            )
            atom_index += 1
            continue
        out_lines.append(line)

    write_text(out_data, "\n".join(out_lines) + "\n")


def parse_lammps_force_dump(path: Path) -> dict[int, list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = max(index for index, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))

    # LAMMPS real units dump forces in kcal/mol/Angstrom.
    kcal_per_angstrom_to_kj_per_nm = 41.84
    forces: dict[int, list[float]] = {}
    for line in lines[start + 1 :]:
        if line.startswith("ITEM:"):
            break
        atom, fx, fy, fz = line.split()
        forces[int(atom) - 1] = [
            float(fx) * kcal_per_angstrom_to_kj_per_nm,
            float(fy) * kcal_per_angstrom_to_kj_per_nm,
            float(fz) * kcal_per_angstrom_to_kj_per_nm,
        ]
    return forces


def run_exact_case(
    gmx_binary: Path,
    workdir: Path,
    initial_gro: Path,
    topol_top: Path,
    outer_steps: int,
    pair14_level: int,
) -> tuple[Path, Path, dict[int, dict[str, object]]]:
    exact_mdp = workdir / "exact.mdp"
    exact_tpr = workdir / "exact.tpr"
    exact_trr = workdir / "exact.trr"
    write_text(exact_mdp, exact_respa_mdp(outer_steps, pair14_level))

    run_command(
        [str(gmx_binary), "grompp", "-f", str(exact_mdp), "-c", str(initial_gro), "-p", str(topol_top), "-o", str(exact_tpr), "-maxwarn", "1"],
        cwd=workdir,
    )
    env = os.environ.copy()
    env["GMX_DISABLE_MODULAR_SIMULATOR"] = "ON"
    run_command(
        [str(gmx_binary), "mdrun", "-s", str(exact_tpr), "-deffnm", str(workdir / "exact"), "-ntmpi", "1", "-ntomp", "1"],
        cwd=workdir,
        env=env,
    )
    return exact_tpr, exact_trr, parse_trr_force_frames(gmx_binary, workdir, exact_trr, DEFAULT_LEVEL_STEP_FACTORS)


def extract_frame(gmx_binary: Path, workdir: Path, exact_tpr: Path, exact_trr: Path, time_ps: float, out_coordinate_file: Path) -> None:
    run_command(
        [
            str(gmx_binary),
            "trjconv",
            "-f",
            str(exact_trr),
            "-s",
            str(exact_tpr),
            "-o",
            str(out_coordinate_file),
            "-dump",
            f"{time_ps:.15g}",
            "-ndec",
            "9",
        ],
        cwd=workdir,
        stdin="0\n",
    )


def run_lammps_probe(lammps_cmd: str, workdir: Path, system_in: Path, system_data: Path, coordinate_file: Path) -> dict[int, list[float]]:
    shutil.copy2(system_data, workdir / "system.data.orig")
    data_path = workdir / "frame.data"
    rewrite_lammps_data_with_coordinates(system_data, coordinate_file, data_path)

    input_path = workdir / "probe.in"
    input_contents = system_in.read_text(encoding="utf-8").replace("read_data system.data", f"read_data {data_path.name}")
    input_contents += (
        "\nthermo 1\n"
        "thermo_style custom step pe ke etotal\n"
        "dump f all custom 1 force.dump id fx fy fz\n"
        "dump_modify f sort id\n"
        "run 0\n"
    )
    write_text(input_path, input_contents)

    run_command(
        ["/bin/bash", "-lc", f"OMPI_MCA_plm=isolated {lammps_cmd} -in {input_path.name}"],
        cwd=workdir,
    )
    return parse_lammps_force_dump(workdir / "force.dump")


def compare_force_frames(gmx_forces: dict[int, list[float]], lammps_forces: dict[int, list[float]]) -> dict[str, object]:
    gmx_atoms = sorted(gmx_forces)
    lammps_atoms = sorted(lammps_forces)
    if gmx_atoms != lammps_atoms:
        raise ValueError(f"Atom index mismatch: gmx={gmx_atoms} lammps={lammps_atoms}")

    max_abs_component_delta = 0.0
    sum_squared_component_delta = 0.0
    max_atom_norm_delta = 0.0
    atom_with_max_norm_delta = -1
    component_count = 0

    for atom in gmx_atoms:
        squared_norm_delta = 0.0
        for gmx_component, lammps_component in zip(gmx_forces[atom], lammps_forces[atom]):
            delta = gmx_component - lammps_component
            abs_delta = abs(delta)
            max_abs_component_delta = max(max_abs_component_delta, abs_delta)
            sum_squared_component_delta += delta * delta
            squared_norm_delta += delta * delta
            component_count += 1
        norm_delta = math.sqrt(squared_norm_delta)
        if norm_delta > max_atom_norm_delta:
            max_atom_norm_delta = norm_delta
            atom_with_max_norm_delta = atom

    rms_component_delta = math.sqrt(sum_squared_component_delta / component_count)
    return {
        "num_atoms": len(gmx_atoms),
        "max_abs_component_delta_kj_mol_nm": max_abs_component_delta,
        "rms_component_delta_kj_mol_nm": rms_component_delta,
        "max_atom_norm_delta_kj_mol_nm": max_atom_norm_delta,
        "atom_with_max_norm_delta": atom_with_max_norm_delta,
    }


def summarize_system(
    gmx_binary: Path,
    lammps_cmd: str,
    reference_root: Path,
    out_root: Path,
    system_id: str,
    outer_steps: int,
    pair14_level: int,
    max_abs_tol: float,
    max_atom_norm_tol: float,
    rms_tol: float,
) -> dict[str, object]:
    initial_gro, topol_top = system_paths(reference_root, system_id)
    system_in, system_data = lammps_system_paths(system_id)

    workdir = out_root / system_id
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    exact_tpr, exact_trr, exact_frames = run_exact_case(
        gmx_binary, workdir, initial_gro, topol_top, outer_steps, pair14_level
    )

    frame_summaries: list[dict[str, object]] = []
    for step in sorted(exact_frames):
        exact_frame = exact_frames[step]
        frame_time = float(exact_frame["time_ps"])
        coordinate_file = initial_gro if step == 0 else workdir / f"frame_step_{step:04d}.g96"
        if step != 0:
            extract_frame(gmx_binary, workdir, exact_tpr, exact_trr, frame_time, coordinate_file)

        probe_workdir = workdir / f"probe_step_{step:04d}"
        probe_workdir.mkdir(parents=True, exist_ok=True)
        probe_coordinate_file = probe_workdir / f"coords{coordinate_file.suffix}"
        shutil.copy2(coordinate_file, probe_coordinate_file)

        lammps_forces = run_lammps_probe(lammps_cmd, probe_workdir, system_in, system_data, probe_coordinate_file)

        comparison = compare_force_frames(
            exact_frame["forces"],  # type: ignore[arg-type]
            lammps_forces,
        )
        frame_summaries.append(
            {
                "step": step,
                "time_ps": frame_time,
                "highest_active_level": int(exact_frame["highest_active_level"]),
                **comparison,
            }
        )

    summary = {
        "system": system_id,
        "outer_steps": outer_steps,
        "pair14_level": pair14_level,
        "tolerances": {
            "max_abs_component_delta_kj_mol_nm": max_abs_tol,
            "max_atom_norm_delta_kj_mol_nm": max_atom_norm_tol,
            "rms_component_delta_kj_mol_nm": rms_tol,
        },
        "frames": frame_summaries,
        "overall": {
            "max_abs_component_delta_kj_mol_nm": max(
                frame["max_abs_component_delta_kj_mol_nm"] for frame in frame_summaries
            ),
            "max_atom_norm_delta_kj_mol_nm": max(
                frame["max_atom_norm_delta_kj_mol_nm"] for frame in frame_summaries
            ),
            "max_rms_component_delta_kj_mol_nm": max(
                frame["rms_component_delta_kj_mol_nm"] for frame in frame_summaries
            ),
        },
    }
    summary["pass"] = (
        summary["overall"]["max_abs_component_delta_kj_mol_nm"] <= max_abs_tol
        and summary["overall"]["max_atom_norm_delta_kj_mol_nm"] <= max_atom_norm_tol
        and summary["overall"]["max_rms_component_delta_kj_mol_nm"] <= rms_tol
    )
    write_text(workdir / "lammps_force_trace_compare_summary.json", json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    args = parse_args()
    if not args.lammps_cmd:
        raise SystemExit("Could not find a LAMMPS executable. Pass --lammps-cmd explicitly.")

    gmx_binary = Path(args.gmx).resolve()
    reference_root = Path(args.reference_root).resolve()
    out_root = Path(args.out).resolve()
    systems = args.systems or list(DEFAULT_SYSTEMS)

    if not args.skip_build:
        run_command(
            ["cmake", "--build", str(DEFAULT_BUILD_DIR), "--target", args.build_target],
            cwd=REPO_ROOT,
        )

    out_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        summarize_system(
            gmx_binary,
            args.lammps_cmd,
            reference_root,
            out_root,
            system_id,
            args.outer_steps,
            args.pair14_level,
            args.max_abs_tol,
            args.max_atom_norm_tol,
            args.rms_tol,
        )
        for system_id in systems
    ]

    aggregate = {
        "systems": summaries,
        "overall": {
            "max_abs_component_delta_kj_mol_nm": max(
                system["overall"]["max_abs_component_delta_kj_mol_nm"] for system in summaries
            ),
            "max_atom_norm_delta_kj_mol_nm": max(
                system["overall"]["max_atom_norm_delta_kj_mol_nm"] for system in summaries
            ),
            "max_rms_component_delta_kj_mol_nm": max(
                system["overall"]["max_rms_component_delta_kj_mol_nm"] for system in summaries
            ),
        },
    }
    aggregate["pass"] = all(system["pass"] for system in summaries)
    write_text(out_root / "aggregate_lammps_force_trace_compare.json", json.dumps(aggregate, indent=2, sort_keys=True))
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
