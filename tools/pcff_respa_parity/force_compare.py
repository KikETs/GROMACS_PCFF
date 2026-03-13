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
BUILD_DIR = REPO_ROOT / "build"
DEFAULT_GMX = BUILD_DIR / "bin" / "gmx"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "tests" / "reference_results" / "m6_respa"
DEFAULT_OUT_ROOT = DEFAULT_REFERENCE_ROOT / "force_compare_last"
DEFAULT_SYSTEMS = ("small_oligomer", "small_salt_polymer_box")
ENERGY_INTERVAL = 4
FOURIER_SPACING_NM = 0.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exact r-RESPA total forces against unsplit single-point forces at identical coordinates."
    )
    parser.add_argument(
        "--gmx",
        default=str(DEFAULT_GMX),
        help="Path to the GROMACS CLI binary.",
    )
    parser.add_argument(
        "--reference-root",
        default=str(DEFAULT_REFERENCE_ROOT),
        help="Directory containing the frozen M6 r-RESPA fixtures.",
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
        help="Value for mts-respa-pair14-level in the exact diagnostic run.",
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
        text=True if stdin is not None else None,
        check=True,
    )


def run_command_capture(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def exact_respa_mdp(outer_steps: int, pair14_level: int) -> str:
    nsteps = outer_steps * ENERGY_INTERVAL
    return (
        "title                   = pcff exact respa force compare\n"
        "integrator              = md-vv\n"
        "dt                      = 0.0005\n"
        f"nsteps                  = {nsteps}\n"
        "constraints             = none\n"
        "cutoff-scheme           = Verlet\n"
        "nstlist                 = 4\n"
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
        "mts                     = yes\n"
        "mts-mode                = lammps-respa\n"
        "mts-levels              = 3\n"
        "mts-level2-factor       = 2\n"
        "mts-level3-factor       = 4\n"
        "mts-respa-bond-level    = 1\n"
        "mts-respa-angle-level   = 1\n"
        "mts-respa-dihedral-level = 1\n"
        "mts-respa-improper-level = 1\n"
        f"mts-respa-pair14-level  = {pair14_level}\n"
        "mts-respa-kspace-level  = 3\n"
        "mts-respa-inner-level   = 1\n"
        "mts-respa-middle-level  = 2\n"
        "mts-respa-outer-level   = 3\n"
        "mts-respa-inner-off     = 0.30\n"
        "mts-respa-inner-on      = 0.45\n"
        "mts-respa-outer-on      = 0.60\n"
        "mts-respa-outer-off     = 0.80\n"
        f"nstcalcenergy           = {ENERGY_INTERVAL}\n"
        f"nstenergy               = {ENERGY_INTERVAL}\n"
        f"nstlog                  = {ENERGY_INTERVAL}\n"
        f"nstxout                 = {ENERGY_INTERVAL}\n"
        "nstvout                 = 0\n"
        "nstfout                 = 0\n"
        "nstxout-compressed      = 0\n"
    )


def unsplit_single_point_mdp() -> str:
    return (
        "title                   = pcff unsplit force probe\n"
        "integrator              = md-vv\n"
        "dt                      = 0.0005\n"
        "nsteps                  = 1\n"
        "continuation            = yes\n"
        "constraints             = none\n"
        "cutoff-scheme           = Verlet\n"
        "nstlist                 = 1\n"
        "rlist                   = 0.9\n"
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
        "nstcalcenergy           = 1\n"
        "nstenergy               = 2\n"
        "nstlog                  = 2\n"
        "nstxout                 = 1\n"
        "nstvout                 = 0\n"
        "nstfout                 = 1\n"
        "nstxout-compressed      = 0\n"
    )


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


def parse_trr_force_frames(gmx_binary: Path, workdir: Path, trr_path: Path) -> dict[int, dict[str, object]]:
    dump_output = run_command_capture([str(gmx_binary), "dump", "-f", str(trr_path)], cwd=workdir)

    natoms_re = re.compile(r"natoms=\s*(\d+)\s+step=\s*(-?\d+)\s+time=([0-9eE+.\-]+)")
    force_re = re.compile(
        r"^\s*f\[\s*(\d+)\]=\{\s*([0-9eE+.\-]+),\s*([0-9eE+.\-]+),\s*([0-9eE+.\-]+)\}"
    )

    frames: dict[int, dict[str, object]] = {}
    current_step: int | None = None
    current_time = 0.0
    in_force_block = False

    for line in dump_output.splitlines():
        natoms_match = natoms_re.search(line)
        if natoms_match:
            current_step = int(natoms_match.group(2))
            current_time = float(natoms_match.group(3))
            frames[current_step] = {
                "time_ps": current_time,
                "highest_active_level": 0,
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
            force_match = force_re.match(line)
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


def compare_force_frames(
    exact_forces: dict[int, list[float]], unsplit_forces: dict[int, list[float]]
) -> dict[str, object]:
    exact_atoms = sorted(exact_forces)
    unsplit_atoms = sorted(unsplit_forces)
    if exact_atoms != unsplit_atoms:
        raise ValueError(f"Atom index mismatch: exact={exact_atoms} unsplit={unsplit_atoms}")

    max_abs_component_delta = 0.0
    sum_squared_component_delta = 0.0
    max_atom_norm_delta = 0.0
    atom_with_max_norm_delta = -1
    component_count = 0

    for atom in exact_atoms:
        squared_norm_delta = 0.0
        for exact_component, unsplit_component in zip(exact_forces[atom], unsplit_forces[atom]):
            delta = exact_component - unsplit_component
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
        "num_atoms": len(exact_atoms),
        "max_abs_component_delta_kj_mol_nm": max_abs_component_delta,
        "rms_component_delta_kj_mol_nm": rms_component_delta,
        "max_atom_norm_delta_kj_mol_nm": max_atom_norm_delta,
        "atom_with_max_norm_delta": atom_with_max_norm_delta,
    }


def system_paths(reference_root: Path, system_id: str) -> tuple[Path, Path]:
    system_root = reference_root / system_id
    return system_root / "initial_nve.gro", system_root / "topol.top"


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
    exact_force_dump = workdir / "exact_force.tsv"
    write_text(exact_mdp, exact_respa_mdp(outer_steps, pair14_level))
    if exact_force_dump.exists():
        exact_force_dump.unlink()

    run_command(
        [str(gmx_binary), "grompp", "-f", str(exact_mdp), "-c", str(initial_gro), "-p", str(topol_top), "-o", str(exact_tpr), "-maxwarn", "1"],
        cwd=workdir,
    )
    env = os.environ.copy()
    env["GMX_TOTAL_FORCE_DUMP_FILE"] = str(exact_force_dump)
    env["GMX_DISABLE_MODULAR_SIMULATOR"] = "ON"
    run_command(
        [str(gmx_binary), "mdrun", "-s", str(exact_tpr), "-deffnm", str(workdir / "exact"), "-ntmpi", "1", "-ntomp", "1"],
        cwd=workdir,
        env=env,
    )
    return exact_tpr, exact_trr, parse_force_dump(exact_force_dump)


def extract_frame(gmx_binary: Path, workdir: Path, exact_tpr: Path, exact_trr: Path, time_ps: float, out_gro: Path) -> None:
    run_command(
        [
            str(gmx_binary),
            "trjconv",
            "-f",
            str(exact_trr),
            "-s",
            str(exact_tpr),
            "-o",
            str(out_gro),
            "-dump",
            f"{time_ps:.15g}",
            "-ndec",
            "9",
        ],
        cwd=workdir,
        stdin="0\n",
    )


def run_unsplit_probe(gmx_binary: Path, workdir: Path, coordinate_gro: Path, topol_top: Path, label: str) -> dict[int, dict[str, object]]:
    mdp = workdir / f"{label}.mdp"
    tpr = workdir / f"{label}.tpr"
    dump = workdir / f"{label}_force.tsv"
    write_text(mdp, unsplit_single_point_mdp())
    if dump.exists():
        dump.unlink()

    run_command(
        [str(gmx_binary), "grompp", "-f", str(mdp), "-c", str(coordinate_gro), "-p", str(topol_top), "-o", str(tpr), "-maxwarn", "1"],
        cwd=workdir,
    )
    env = os.environ.copy()
    env["GMX_TOTAL_FORCE_DUMP_FILE"] = str(dump)
    env["GMX_DISABLE_MODULAR_SIMULATOR"] = "ON"
    run_command(
        [str(gmx_binary), "mdrun", "-s", str(tpr), "-deffnm", str(workdir / label), "-ntmpi", "1", "-ntomp", "1"],
        cwd=workdir,
        env=env,
    )
    if dump.exists():
        return parse_force_dump(dump)
    trr = workdir / f"{label}.trr"
    return parse_trr_force_frames(gmx_binary, workdir, trr)


def summarize_system(
    gmx_binary: Path, reference_root: Path, out_root: Path, system_id: str, outer_steps: int, pair14_level: int
) -> dict[str, object]:
    initial_gro, topol_top = system_paths(reference_root, system_id)
    workdir = out_root / system_id
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    exact_tpr, exact_trr, exact_frames = run_exact_case(
        gmx_binary, workdir, initial_gro, topol_top, outer_steps, pair14_level
    )

    frame_summaries: list[dict[str, object]] = []
    for step in sorted(exact_frames):
        frame = exact_frames[step]
        frame_time = float(frame["time_ps"])
        frame_gro = initial_gro if step == 0 else workdir / f"frame_step_{step:04d}.g96"
        if step != 0:
            extract_frame(gmx_binary, workdir, exact_tpr, exact_trr, frame_time, frame_gro)

        unsplit_frames = run_unsplit_probe(gmx_binary, workdir, frame_gro, topol_top, f"unsplit_step_{step:04d}")
        unsplit_step = min(unsplit_frames)

        comparison = compare_force_frames(
            frame["forces"],  # type: ignore[arg-type]
            unsplit_frames[unsplit_step]["forces"],  # type: ignore[arg-type]
        )
        frame_summaries.append(
            {
                "step": step,
                "time_ps": frame_time,
                "highest_active_level": int(frame["highest_active_level"]),
                "unsplit_dump_step": unsplit_step,
                **comparison,
            }
        )

    max_abs_component_delta = max(frame["max_abs_component_delta_kj_mol_nm"] for frame in frame_summaries)
    max_atom_norm_delta = max(frame["max_atom_norm_delta_kj_mol_nm"] for frame in frame_summaries)
    rms_component_delta = max(frame["rms_component_delta_kj_mol_nm"] for frame in frame_summaries)

    summary = {
        "system": system_id,
        "outer_steps": outer_steps,
        "pair14_level": pair14_level,
        "frames": frame_summaries,
        "overall": {
            "max_abs_component_delta_kj_mol_nm": max_abs_component_delta,
            "max_atom_norm_delta_kj_mol_nm": max_atom_norm_delta,
            "max_rms_component_delta_kj_mol_nm": rms_component_delta,
        },
    }
    write_text(workdir / "force_compare_summary.json", json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    args = parse_args()
    gmx_binary = Path(args.gmx).resolve()
    reference_root = Path(args.reference_root).resolve()
    out_root = Path(args.out).resolve()
    systems = args.systems or list(DEFAULT_SYSTEMS)

    if not args.skip_build:
        run_command(
            ["cmake", "--build", str(BUILD_DIR), "--target", args.build_target],
            cwd=REPO_ROOT,
        )

    out_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        summarize_system(gmx_binary, reference_root, out_root, system_id, args.outer_steps, args.pair14_level)
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
    write_text(out_root / "aggregate_force_compare.json", json.dumps(aggregate, indent=2, sort_keys=True))
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
