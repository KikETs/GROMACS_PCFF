#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import IO


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.pcff_fixture_bridge.common import (  # noqa: E402
    KCAL_TO_KJ,
    build_typed_ir,
    dump_json,
    parse_lammps_input,
    render_gromacs_topology,
    system_metadata,
)
from tools.run_m10_0_short_workflow.run_m10_0 import create_gro_from_lammps  # noqa: E402


DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "pcff_charged_expansion" / "m1_m3_dense_salt_polymer"
GROMACS_PCOUPL = {
    "berendsen": "Berendsen",
    "c-rescale": "C-rescale",
    "parrinello-rahman": "Parrinello-Rahman",
    "mttk": "MTTK",
}
BAR_TO_ATM = 0.9869232667160128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the explicit M1-M3 PCFF charged capability subset.")
    parser.add_argument("--system", default="dense_salt_polymer")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixture-manifest", type=Path, help="Optional pre-generated strict-PCFF fixture manifest.")
    parser.add_argument("--milestone-subset", default="M1-M3")
    parser.add_argument("--warmup-ps", type=float, default=0.0)
    parser.add_argument("--warmup-scope", choices=["gmx-only", "paired"], default="gmx-only")
    parser.add_argument("--npt-ps", type=float, default=1000.0)
    parser.add_argument("--analysis-window-ps", type=float, default=500.0)
    parser.add_argument("--nvt-ps", type=float, default=2000.0)
    parser.add_argument("--skip-nvt", action="store_true", help="Skip the long-horizon NVT continuation stage.")
    parser.add_argument("--density-threshold", type=float, default=0.05)
    parser.add_argument("--volume-threshold", type=float, default=0.05)
    parser.add_argument("--stability-mean-temp-tolerance-k", type=float, default=20.0)
    parser.add_argument("--stability-max-temp-k", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--gmx-integrator", choices=["md", "md-vv"], default="md")
    parser.add_argument("--gmx-tcoupl", choices=["v-rescale", "nose-hoover"], default="v-rescale")
    parser.add_argument("--gmx-pcoupl", choices=["berendsen", "c-rescale", "parrinello-rahman", "mttk"], default="c-rescale")
    parser.add_argument("--thermal-start", choices=["generated", "fixture"], default="generated")
    parser.add_argument("--nsttcouple", type=int, help="Optional explicit thermostat coupling frequency.")
    parser.add_argument("--nstpcouple", type=int, help="Optional explicit pressure-coupling frequency.")
    parser.add_argument("--tau-t-ps", type=float, default=0.1)
    parser.add_argument("--tau-p-ps", type=float, default=1.0)
    parser.add_argument("--ref-p-bar", type=float, default=1.0)
    parser.add_argument("--compressibility-bar-inv", type=float, default=4.5e-5)
    parser.add_argument("--lmp-neighbor-skin-angstrom", type=float)
    parser.add_argument("--lmp-neighbor-every", type=int)
    parser.add_argument(
        "--lmp-target-barostat",
        choices=["npt", "berendsen"],
        default="npt",
        help="LAMMPS target pressure control; berendsen is only a weak density-equilibration probe.",
    )
    parser.add_argument("--gmx-threads", type=int, default=1)
    parser.add_argument("--lmp-ranks", type=int, default=1)
    parser.add_argument("--lmp-omp-threads", type=int, default=1)
    return parser.parse_args()


def run_command(
    cmd: list[str],
    work_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    stdin_text: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    result = subprocess.run(
        cmd,
        cwd=work_dir,
        input=stdin_text,
        capture_output=True,
        text=True,
        errors="replace",
        env={**os.environ, "GMX_MAXBACKUP": "-1", **(extra_env or {})},
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def spawn_command(
    cmd: list[str],
    work_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen[str], IO[str], IO[str], list[str]]:
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=work_dir,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        env={**os.environ, "GMX_MAXBACKUP": "-1", **(extra_env or {})},
    )
    return process, stdout_handle, stderr_handle, cmd


def wait_for_process(running: tuple[subprocess.Popen[str], IO[str], IO[str], list[str]]) -> None:
    process, stdout_handle, stderr_handle, cmd = running
    returncode = process.wait()
    stdout_handle.close()
    stderr_handle.close()
    if returncode != 0:
        raise RuntimeError(f"Command failed ({returncode}): {' '.join(cmd)}")


def wait_for_process_pair(
    first: tuple[subprocess.Popen[str], IO[str], IO[str], list[str]],
    second: tuple[subprocess.Popen[str], IO[str], IO[str], list[str]],
) -> None:
    running = [first, second]
    finished: set[int] = set()
    try:
        while len(finished) < len(running):
            for idx, item in enumerate(running):
                if idx in finished:
                    continue
                process, _, _, cmd = item
                returncode = process.poll()
                if returncode is None:
                    continue
                finished.add(idx)
                if returncode != 0:
                    for other_idx, other in enumerate(running):
                        other_process = other[0]
                        if other_idx not in finished and other_process.poll() is None:
                            other_process.terminate()
                    for other_idx, other in enumerate(running):
                        other_process = other[0]
                        if other_idx not in finished:
                            try:
                                other_process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                other_process.kill()
                                other_process.wait()
                            finished.add(other_idx)
                    raise RuntimeError(f"Command failed ({returncode}): {' '.join(cmd)}")
            if len(finished) < len(running):
                time.sleep(1.0)
    finally:
        for _, stdout_handle, stderr_handle, _ in running:
            stdout_handle.close()
            stderr_handle.close()


def validate_args(args: argparse.Namespace) -> None:
    if args.gmx_pcoupl == "mttk":
        if args.gmx_integrator != "md-vv":
            raise ValueError("GROMACS MTTK requires --gmx-integrator md-vv.")
        if args.gmx_tcoupl != "nose-hoover":
            raise ValueError("GROMACS MTTK requires --gmx-tcoupl nose-hoover.")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(read_text(path))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def block_sem(values: list[float], nblocks: int = 5) -> float | None:
    if len(values) < nblocks or nblocks < 2:
        return None
    block_size = len(values) // nblocks
    if block_size == 0:
        return None
    means = []
    for idx in range(nblocks):
        start = idx * block_size
        end = (idx + 1) * block_size if idx < nblocks - 1 else len(values)
        if start >= len(values):
            break
        block = values[start:end]
        if not block:
            continue
        means.append(statistics.fmean(block))
    if len(means) < 2:
        return None
    return statistics.stdev(means) / math.sqrt(len(means))


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def parse_xvg(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw in read_text(path).splitlines():
        if raw.startswith(("#", "@")):
            continue
        stripped = raw.strip()
        if not stripped:
            continue
        rows.append([float(token) for token in stripped.split()])
    return rows


def filter_time_window(rows: list[list[float]], time_index: int, start_ps: float) -> list[list[float]]:
    return [row for row in rows if row[time_index] >= start_ps]


def parse_lammps_blocks(log_path: Path) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    current_header: list[str] | None = None
    current_rows: list[list[float]] = []
    for raw in read_text(log_path).splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("Step "):
            if current_header is not None and current_rows:
                blocks.append({"header": current_header, "rows": current_rows})
            current_header = stripped.split()
            current_rows = []
            continue
        if current_header is None:
            continue
        if stripped.startswith("Loop time"):
            if current_rows:
                blocks.append({"header": current_header, "rows": current_rows})
            current_header = None
            current_rows = []
            continue
        parts = stripped.split()
        if parts and parts[0].lstrip("-").isdigit():
            try:
                current_rows.append([float(token) for token in parts])
            except ValueError:
                continue
    if current_header is not None and current_rows:
        blocks.append({"header": current_header, "rows": current_rows})
    return blocks


def build_gmx_min_mdp() -> str:
    return "\n".join(
        [
            "integrator = steep",
            "emtol = 100.0",
            "nsteps = 500",
            "cutoff-scheme = Verlet",
            "nstlist = 10",
            "verlet-buffer-tolerance = -1",
            "rlist = 0.9",
            "coulombtype = PME",
            "coulomb-modifier = none",
            "rcoulomb = 0.9",
            "vdw-type = Cut-off",
            "vdw-modifier = none",
            "rvdw = 0.9",
            "DispCorr = no",
            "pbc = xyz",
            "",
        ]
    )


def build_gmx_npt_mdp(
    integrator: str,
    nsteps: int,
    seed: int,
    tcoupl: str,
    pcoupl: str,
    thermal_start: str,
    nsttcouple: int | None,
    nstpcouple: int | None,
    tau_t_ps: float,
    tau_p_ps: float,
    ref_p_bar: float,
    compressibility_bar_inv: float,
    continuation: bool = False,
) -> str:
    lines = [
        f"integrator = {integrator}",
        "dt = 0.001",
        f"nsteps = {nsteps}",
        "cutoff-scheme = Verlet",
        "nstlist = 10",
        "verlet-buffer-tolerance = -1",
        "rlist = 0.9",
        "coulombtype = PME",
        "coulomb-modifier = none",
        "rcoulomb = 0.9",
        "pme-order = 4",
        "fourierspacing = 0.12",
        "ewald-rtol = 1e-5",
        "vdw-type = Cut-off",
        "vdw-modifier = none",
        "rvdw = 0.9",
        "DispCorr = no",
        "pbc = xyz",
        "nstenergy = 100",
        "nstlog = 100",
        "nstxout-compressed = 0",
        f"tcoupl = {tcoupl}",
    ]
    if nsttcouple is not None:
        lines.append(f"nsttcouple = {nsttcouple}")
    lines.extend(
        [
            "tc-grps = System",
            f"tau-t = {tau_t_ps}",
            "ref-t = 300",
            f"pcoupl = {pcoupl}",
            "pcoupltype = isotropic",
        ]
    )
    if nstpcouple is not None:
        lines.append(f"nstpcouple = {nstpcouple}")
    lines.extend(
        [
            f"tau-p = {tau_p_ps}",
            f"compressibility = {compressibility_bar_inv}",
            f"ref-p = {ref_p_bar}",
            f"gen-vel = {'yes' if thermal_start == 'generated' else 'no'}",
        ]
    )
    if thermal_start == "generated":
        lines.extend(["gen-temp = 300", f"gen-seed = {seed}"])
    if continuation:
        lines.append("continuation = yes")
    lines.extend(["constraints = none", ""])
    return "\n".join(lines)


def build_gmx_nvt_mdp(integrator: str, tcoupl: str, tau_t_ps: float, nsteps: int) -> str:
    return "\n".join(
        [
            f"integrator = {integrator}",
            "dt = 0.001",
            f"nsteps = {nsteps}",
            "cutoff-scheme = Verlet",
            "nstlist = 10",
            "verlet-buffer-tolerance = -1",
            "rlist = 0.9",
            "coulombtype = PME",
            "coulomb-modifier = none",
            "rcoulomb = 0.9",
            "pme-order = 4",
            "fourierspacing = 0.12",
            "ewald-rtol = 1e-5",
            "vdw-type = Cut-off",
            "vdw-modifier = none",
            "rvdw = 0.9",
            "DispCorr = no",
            "pbc = xyz",
            "nstenergy = 100",
            "nstlog = 100",
            "nstxout-compressed = 0",
            f"tcoupl = {tcoupl}",
            "tc-grps = System",
            f"tau-t = {tau_t_ps}",
            "ref-t = 300",
            "pcoupl = no",
            "gen-vel = no",
            "continuation = yes",
            "constraints = none",
            "",
        ]
    )


def write_lammps_min_input(source_in: Path, output_in: Path) -> None:
    source_lines = read_text(source_in).splitlines()
    rendered = []
    for line in source_lines:
        if line.startswith("read_data"):
            rendered.append("read_data system.data")
        else:
            rendered.append(line)
    rendered.extend(
        [
            "",
            "thermo 1",
            "thermo_style custom step temp pe ke etotal press vol density",
            "minimize 1.0e-4 1.0e-6 100 1000",
            "write_data minimized.data",
            "",
        ]
    )
    write_text(output_in, "\n".join(rendered))


def write_lammps_berendsen_warmup_input(
    source_in: Path,
    output_in: Path,
    seed: int,
    nsteps: int,
    thermal_start: str,
    tau_t_ps: float,
    tau_p_ps: float,
    ref_p_bar: float,
) -> None:
    source_lines = read_text(source_in).splitlines()
    rendered = []
    for line in source_lines:
        if line.startswith("read_data"):
            rendered.append("read_data minimized.data")
        else:
            rendered.append(line)
    rendered.append("")
    if thermal_start == "generated":
        rendered.append(f"velocity all create 300.0 {seed} dist gaussian mom yes rot yes")
    rendered.extend(
        [
            "reset_timestep 0",
            "timestep 1.0",
            "fix 1 all nve",
            f"fix 2 all temp/berendsen 300.0 300.0 {tau_t_ps * 1000.0:.10g}",
            f"fix 3 all press/berendsen iso {ref_p_bar * BAR_TO_ATM:.10g} {ref_p_bar * BAR_TO_ATM:.10g} {tau_p_ps * 1000.0:.10g}",
            "thermo 100",
            "thermo_style custom step temp pe ke etotal press vol density",
            "thermo_modify flush yes",
            f"run {nsteps}",
            "write_data warmup.data",
            "",
        ]
    )
    write_text(output_in, "\n".join(rendered))


def write_lammps_npt_input(
    source_in: Path,
    output_in: Path,
    seed: int,
    nsteps: int,
    thermal_start: str,
    tau_t_ps: float,
    tau_p_ps: float,
    ref_p_bar: float,
    target_barostat: str,
    neighbor_skin_angstrom: float | None = None,
    neighbor_every: int | None = None,
    read_data_name: str = "minimized.data",
) -> None:
    source_lines = read_text(source_in).splitlines()
    rendered = []
    for line in source_lines:
        if line.startswith("read_data"):
            rendered.append(f"read_data {read_data_name}")
        else:
            rendered.append(line)
    rendered.append("")
    if thermal_start == "generated":
        rendered.append(f"velocity all create 300.0 {seed} dist gaussian mom yes rot yes")
    rendered.extend(["reset_timestep 0", "timestep 1.0"])
    if neighbor_skin_angstrom is not None:
        rendered.append(f"neighbor {neighbor_skin_angstrom:.10g} bin")
    if neighbor_every is not None:
        rendered.append(f"neigh_modify delay 0 every {neighbor_every} check yes")
    if target_barostat == "npt":
        rendered.append(
            f"fix 1 all npt temp 300.0 300.0 {tau_t_ps * 1000.0:.10g} iso {ref_p_bar * BAR_TO_ATM:.10g} {ref_p_bar * BAR_TO_ATM:.10g} {tau_p_ps * 1000.0:.10g}"
        )
    elif target_barostat == "berendsen":
        rendered.extend(
            [
                "fix 1 all nve",
                f"fix 2 all temp/berendsen 300.0 300.0 {tau_t_ps * 1000.0:.10g}",
                f"fix 3 all press/berendsen iso {ref_p_bar * BAR_TO_ATM:.10g} {ref_p_bar * BAR_TO_ATM:.10g} {tau_p_ps * 1000.0:.10g}",
            ]
        )
    else:
        raise ValueError(f"Unsupported LAMMPS target barostat: {target_barostat}")
    rendered.extend(
        [
            "thermo 100",
            "thermo_style custom step temp pe ke etotal press vol density",
            "thermo_modify flush yes",
            f"run {nsteps}",
            "write_data final.data",
            "",
        ]
    )
    write_text(output_in, "\n".join(rendered))


def lammps_command(input_name: str, log_name: str, args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    cmd: list[str]
    env = {"OMP_NUM_THREADS": str(args.lmp_omp_threads)}
    if args.lmp_ranks > 1:
        cmd = ["mpirun", "-np", str(args.lmp_ranks), "lmp"]
    else:
        cmd = ["lmp"]
    if args.lmp_omp_threads > 1:
        cmd.extend(["-sf", "omp", "-pk", "omp", str(args.lmp_omp_threads)])
    cmd.extend(["-log", log_name, "-in", input_name])
    return cmd, env


def generate_fixture_bundle(system_id: str, out_root: Path, fixture_manifest_path: Path | None) -> tuple[dict[str, str], dict[str, object] | None]:
    generated_root = out_root / "generated_pair"
    generated_root.mkdir(parents=True, exist_ok=True)

    if fixture_manifest_path is not None:
        fixture_manifest = load_json(fixture_manifest_path)
        artifacts = fixture_manifest["artifacts"]
        shutil.copy(Path(artifacts["typed_system"]), generated_root / "typed_system.json")
        shutil.copy(Path(artifacts["topology"]), generated_root / "system.top")
        shutil.copy(Path(artifacts["gro"]), generated_root / "system.gro")
        shutil.copy(Path(artifacts["system_data"]), generated_root / "system.data")
        shutil.copy(Path(artifacts["system_in"]), generated_root / "system.in")
        shutil.copy(Path(artifacts["system_json"]), generated_root / "system.json")
        return (
            {
                "typed_system": str(generated_root / "typed_system.json"),
                "gromacs_topology": str(generated_root / "system.top"),
                "gromacs_coordinates": str(generated_root / "system.gro"),
                "lammps_data": str(generated_root / "system.data"),
                "lammps_input": str(generated_root / "system.in"),
                "system_json": str(generated_root / "system.json"),
            },
            fixture_manifest,
        )

    corpus_root = REPO_ROOT / "testdata" / "lammps_golden"
    system_root = corpus_root / "systems" / system_id
    typed_ir = build_typed_ir({"id": system_id, "path": f"systems/{system_id}"}, corpus_root)
    dump_json(generated_root / "typed_system.json", typed_ir)
    write_text(generated_root / "system.top", render_gromacs_topology(typed_ir))
    create_gro_from_lammps(system_root / "lammps" / "system.data", generated_root / "system.gro")
    shutil.copy(system_root / "lammps" / "system.data", generated_root / "system.data")
    shutil.copy(system_root / "lammps" / "system.in", generated_root / "system.in")
    shutil.copy(system_root / "system.json", generated_root / "system.json")
    return (
        {
            "typed_system": str(generated_root / "typed_system.json"),
            "gromacs_topology": str(generated_root / "system.top"),
            "gromacs_coordinates": str(generated_root / "system.gro"),
            "lammps_data": str(generated_root / "system.data"),
            "lammps_input": str(generated_root / "system.in"),
            "system_json": str(generated_root / "system.json"),
        },
        None,
    )


def build_pair_manifest(
    system_id: str,
    out_root: Path,
    generated_paths: dict[str, str],
    fixture_manifest: dict[str, object] | None,
    fixture_manifest_path: Path | None,
    milestone_subset: str,
) -> dict[str, object]:
    if fixture_manifest is None:
        corpus_root = REPO_ROOT / "testdata" / "lammps_golden"
        system_record = {"id": system_id, "path": f"systems/{system_id}"}
        meta = system_metadata(system_record, corpus_root)
    else:
        meta = load_json(Path(generated_paths["system_json"]))
    lammps_input = parse_lammps_input(Path(generated_paths["lammps_input"]))
    manifest = {
        "milestone_subset": milestone_subset,
        "system_id": system_id,
        "pair_status": "strict_pcff_qualified",
        "acpype_gaff2_dependency": False,
        "gromacs_preparation": "pcff_fixture_bridge_generated_topology",
        "lammps_reference": "pcff_class2_fixture",
        "chemistry_scope_delta": {
            "previous_boundary": "frozen PT8 SPE subset and small charged fixtures",
            "candidate_expansion": system_id,
        },
        "system_metadata": meta,
        "lammps_styles": lammps_input["styles"],
        "artifacts": generated_paths,
        "fixture_manifest": None if fixture_manifest_path is None else str(fixture_manifest_path),
        "audit_basis": [
            "GROMACS topology is generated directly from the repository PCFF/Class2 LAMMPS golden fixture.",
            "No ACPYPE, GAFF2, or surrogate GROMACS topology path is used.",
            "Typed IR retains per-record source provenance to raw LAMMPS files.",
        ],
    }
    if fixture_manifest is not None:
        manifest["derived_fixture"] = {
            "seed_system": fixture_manifest.get("seed_system"),
            "replicate": fixture_manifest.get("replicate"),
            "natoms": fixture_manifest.get("natoms"),
            "box_nm": fixture_manifest.get("box_nm"),
        }
    dump_json(out_root / "qualified_pair_manifest.json", manifest)
    return manifest


def summarize_series(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean_or_none(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "sem_block5": block_sem(values, nblocks=5),
    }


def run_paired_npt(
    args: argparse.Namespace,
    generated_paths: dict[str, str],
    out_root: Path,
) -> dict[str, object]:
    work_root = out_root / "paired_npt"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)

    gmx_dir = work_root / "gromacs"
    lmp_dir = work_root / "lammps"
    gmx_dir.mkdir()
    lmp_dir.mkdir()

    shutil.copy(generated_paths["gromacs_topology"], gmx_dir / "system.top")
    shutil.copy(generated_paths["gromacs_coordinates"], gmx_dir / "system.gro")
    shutil.copy(generated_paths["lammps_data"], lmp_dir / "system.data")
    shutil.copy(generated_paths["lammps_input"], lmp_dir / "system.in")

    write_text(gmx_dir / "min.mdp", build_gmx_min_mdp())
    write_text(
        gmx_dir / "npt.mdp",
        build_gmx_npt_mdp(
            args.gmx_integrator,
            int(args.npt_ps / 0.001),
            args.seed,
            args.gmx_tcoupl,
            GROMACS_PCOUPL[args.gmx_pcoupl],
            "fixture" if args.warmup_ps > 0.0 else args.thermal_start,
            args.nsttcouple,
            args.nstpcouple,
            args.tau_t_ps,
            args.tau_p_ps,
            args.ref_p_bar,
            args.compressibility_bar_inv,
            continuation=args.warmup_ps > 0.0,
        ),
    )
    run_command(
        [str(REPO_ROOT / "build" / "bin" / "gmx"), "grompp", "-f", "min.mdp", "-c", "system.gro", "-p", "system.top", "-o", "min.tpr", "-po", "min_mdout.mdp"],
        gmx_dir,
        gmx_dir / "grompp_min.stdout",
        gmx_dir / "grompp_min.stderr",
    )
    run_command(
        [str(REPO_ROOT / "build" / "bin" / "gmx"), "mdrun", "-s", "min.tpr", "-deffnm", "min", "-nt", str(args.gmx_threads), "-pin", "off", "-reprod"],
        gmx_dir,
        gmx_dir / "mdrun_min.stdout",
        gmx_dir / "mdrun_min.stderr",
    )
    write_lammps_min_input(lmp_dir / "system.in", lmp_dir / "min.in")
    lmp_min_cmd, lmp_env = lammps_command("min.in", "min.log", args)
    run_command(lmp_min_cmd, lmp_dir, lmp_dir / "min.stdout", lmp_dir / "min.stderr", extra_env=lmp_env)

    gmx_start_conf = "min.gro"
    gmx_checkpoint: str | None = None
    lmp_start_data = "minimized.data"
    lmp_target_thermal_start = args.thermal_start

    if args.warmup_ps > 0.0:
        write_text(
            gmx_dir / "warmup.mdp",
            build_gmx_npt_mdp(
                "md",
                int(args.warmup_ps / 0.001),
                args.seed,
                "v-rescale",
                "Berendsen",
                args.thermal_start,
                None,
                None,
                0.1,
                1.0,
                1.0,
                args.compressibility_bar_inv,
            ),
        )
        run_command(
            [str(REPO_ROOT / "build" / "bin" / "gmx"), "grompp", "-f", "warmup.mdp", "-c", "min.gro", "-p", "system.top", "-o", "warmup.tpr", "-po", "warmup_mdout.mdp", "-maxwarn", "1"],
            gmx_dir,
            gmx_dir / "grompp_warmup.stdout",
            gmx_dir / "grompp_warmup.stderr",
        )
        gmx_warmup = spawn_command(
            [str(REPO_ROOT / "build" / "bin" / "gmx"), "mdrun", "-s", "warmup.tpr", "-deffnm", "warmup", "-nt", str(args.gmx_threads), "-pin", "off", "-reprod"],
            gmx_dir,
            gmx_dir / "mdrun_warmup.stdout",
            gmx_dir / "mdrun_warmup.stderr",
        )
        wait_for_process(gmx_warmup)
        gmx_start_conf = "warmup.gro"
        gmx_checkpoint = "warmup.cpt"
        if args.warmup_scope == "paired":
            write_lammps_berendsen_warmup_input(
                lmp_dir / "system.in",
                lmp_dir / "warmup.in",
                args.seed,
                int(args.warmup_ps * 1000.0),
                args.thermal_start,
                args.tau_t_ps,
                args.tau_p_ps,
                args.ref_p_bar,
            )
            lmp_warmup_cmd, _ = lammps_command("warmup.in", "warmup.log", args)
            run_command(
                lmp_warmup_cmd,
                lmp_dir,
                lmp_dir / "warmup.stdout",
                lmp_dir / "warmup.stderr",
                extra_env=lmp_env,
            )
            lmp_start_data = "warmup.data"
            lmp_target_thermal_start = "fixture"

    grompp_npt_cmd = [
        str(REPO_ROOT / "build" / "bin" / "gmx"),
        "grompp",
        "-f",
        "npt.mdp",
        "-c",
        gmx_start_conf,
        "-p",
        "system.top",
        "-o",
        "npt.tpr",
        "-po",
        "npt_mdout.mdp",
        "-maxwarn",
        "1",
    ]
    if gmx_checkpoint is not None:
        grompp_npt_cmd.extend(["-t", gmx_checkpoint])
    run_command(
        grompp_npt_cmd,
        gmx_dir,
        gmx_dir / "grompp_npt.stdout",
        gmx_dir / "grompp_npt.stderr",
    )
    write_lammps_npt_input(
        lmp_dir / "system.in",
        lmp_dir / "npt.in",
        args.seed,
        int(args.npt_ps * 1000.0),
        lmp_target_thermal_start,
        args.tau_t_ps,
        args.tau_p_ps,
        args.ref_p_bar,
        args.lmp_target_barostat,
        args.lmp_neighbor_skin_angstrom,
        args.lmp_neighbor_every,
        read_data_name=lmp_start_data,
    )
    lmp_npt_cmd, _ = lammps_command("npt.in", "npt.log", args)

    gmx_npt = spawn_command(
        [str(REPO_ROOT / "build" / "bin" / "gmx"), "mdrun", "-s", "npt.tpr", "-deffnm", "npt", "-nt", str(args.gmx_threads), "-pin", "off", "-reprod"],
        gmx_dir,
        gmx_dir / "mdrun_npt.stdout",
        gmx_dir / "mdrun_npt.stderr",
    )
    lmp_npt = spawn_command(
        lmp_npt_cmd,
        lmp_dir,
        lmp_dir / "npt.stdout",
        lmp_dir / "npt.stderr",
        extra_env=lmp_env,
    )
    wait_for_process_pair(gmx_npt, lmp_npt)
    run_command(
        [str(REPO_ROOT / "build" / "bin" / "gmx"), "energy", "-f", "npt.edr", "-o", "npt_energy.xvg"],
        gmx_dir,
        gmx_dir / "energy.stdout",
        gmx_dir / "energy.stderr",
        stdin_text="Potential\nTemperature\nPressure\nVolume\nDensity\n0\n",
    )

    gmx_rows = parse_xvg(gmx_dir / "npt_energy.xvg")
    analysis_start_ps = max(0.0, args.npt_ps - args.analysis_window_ps)
    gmx_window = filter_time_window(gmx_rows, 0, analysis_start_ps)
    gmx_potential = [row[1] for row in gmx_window]
    gmx_temperature = [row[2] for row in gmx_window]
    gmx_pressure = [row[3] for row in gmx_window]
    gmx_volume = [row[4] for row in gmx_window]
    gmx_density = [row[5] for row in gmx_window]

    lmp_blocks = parse_lammps_blocks(lmp_dir / "npt.log")
    if not lmp_blocks:
        raise RuntimeError("LAMMPS NPT run did not produce a thermo block")
    lmp_rows = lmp_blocks[-1]["rows"]  # type: ignore[index]
    lmp_window = [row for row in lmp_rows if (row[0] * 0.001) >= analysis_start_ps]
    lmp_potential = [row[2] * KCAL_TO_KJ for row in lmp_window]
    lmp_temperature = [row[1] for row in lmp_window]
    lmp_pressure = [row[5] for row in lmp_window]
    lmp_volume = [row[6] / 1000.0 for row in lmp_window]
    lmp_density = [row[7] * 1000.0 for row in lmp_window]

    density_rel_diff = abs(statistics.fmean(gmx_density) - statistics.fmean(lmp_density)) / statistics.fmean(lmp_density)
    volume_rel_diff = abs(statistics.fmean(gmx_volume) - statistics.fmean(lmp_volume)) / statistics.fmean(lmp_volume)
    parity_pass = density_rel_diff <= args.density_threshold and volume_rel_diff <= args.volume_threshold

    report = {
        "system_id": args.system,
        "protocol": {
            "ensemble": "NPT",
            "warmup_ps": args.warmup_ps,
            "warmup_protocol": (
                "GROMACS md/v-rescale/Berendsen only"
                if args.warmup_ps > 0.0 and args.warmup_scope == "gmx-only"
                else "GROMACS md/v-rescale/Berendsen + LAMMPS nve+temp/berendsen+press/berendsen"
                if args.warmup_ps > 0.0
                else None
            ),
            "warmup_scope": args.warmup_scope if args.warmup_ps > 0.0 else None,
            "duration_ps": args.npt_ps,
            "analysis_window_ps": args.analysis_window_ps,
            "gmx_integrator": args.gmx_integrator,
            "gmx_temperature_coupling": args.gmx_tcoupl,
            "gmx_pressure_coupling": args.gmx_pcoupl,
            "tau_t_ps": args.tau_t_ps,
            "tau_p_ps": args.tau_p_ps,
            "ref_p_bar": args.ref_p_bar,
            "compressibility_bar_inv": args.compressibility_bar_inv,
            "thermal_start": args.thermal_start,
            "lammps_ranks": args.lmp_ranks,
            "lammps_omp_threads": args.lmp_omp_threads,
            "lammps_neighbor_skin_angstrom": args.lmp_neighbor_skin_angstrom,
            "lammps_neighbor_every": args.lmp_neighbor_every,
            "lammps_target_barostat": args.lmp_target_barostat,
            "gmx_nonbonded_modifiers": {"coulomb-modifier": "none", "vdw-modifier": "none"},
            "lammps_velocity_initialization": (
                "velocity create 300 K with fixed seed"
                if args.thermal_start == "generated"
                else "fixture velocities from system.data"
            ),
            "thresholds": {
                "density_rel_diff_max": args.density_threshold,
                "volume_rel_diff_max": args.volume_threshold,
                "relative_diff_reference": "LAMMPS mean over final analysis window",
            },
        },
        "gromacs": {
            "potential_energy_kj_mol": summarize_series(gmx_potential),
            "temperature_k": summarize_series(gmx_temperature),
            "pressure_bar": summarize_series(gmx_pressure),
            "volume_nm3": summarize_series(gmx_volume),
            "density_kg_m3": summarize_series(gmx_density),
        },
        "lammps": {
            "potential_energy_kj_mol": summarize_series(lmp_potential),
            "temperature_k": summarize_series(lmp_temperature),
            "pressure_atm": summarize_series(lmp_pressure),
            "volume_nm3": summarize_series(lmp_volume),
            "density_kg_m3": summarize_series(lmp_density),
        },
        "parity_metrics": {
            "density_rel_diff": density_rel_diff,
            "volume_rel_diff": volume_rel_diff,
        },
        "status": "PASS" if parity_pass else "FAIL",
        "artifacts": {
            "gromacs_root": str(gmx_dir),
            "lammps_root": str(lmp_dir),
            "gromacs_energy_xvg": str(gmx_dir / "npt_energy.xvg"),
            "lammps_log": str(lmp_dir / "npt.log"),
            "gromacs_warmup_log": None if args.warmup_ps <= 0.0 else str(gmx_dir / "warmup.log"),
            "lammps_warmup_log": None if args.warmup_ps <= 0.0 or args.warmup_scope != "paired" else str(lmp_dir / "warmup.log"),
        },
    }
    dump_json(work_root / "dense_npt_parity_report.json", report)
    return report


def run_long_nvt_stability(args: argparse.Namespace, out_root: Path) -> dict[str, object]:
    src_root = out_root / "paired_npt" / "gromacs"
    work_root = out_root / "long_nvt_stability"
    system_id = getattr(args, "system", None)
    if system_id is None:
        manifest_path = out_root / "qualified_pair_manifest.json"
        if manifest_path.exists():
            system_id = load_json(manifest_path).get("system_id")
    if system_id is None:
        system_id = out_root.name
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)

    shutil.copy(src_root / "system.top", work_root / "system.top")
    shutil.copy(src_root / "npt.gro", work_root / "system.gro")
    shutil.copy(src_root / "npt.cpt", work_root / "system.cpt")
    write_text(
        work_root / "nvt.mdp",
        build_gmx_nvt_mdp(args.gmx_integrator, args.gmx_tcoupl, args.tau_t_ps, int(args.nvt_ps / 0.001)),
    )

    run_command(
        [
            str(REPO_ROOT / "build" / "bin" / "gmx"),
            "grompp",
            "-f",
            "nvt.mdp",
            "-c",
            "system.gro",
            "-t",
            "system.cpt",
            "-p",
            "system.top",
            "-o",
            "nvt.tpr",
            "-po",
            "nvt_mdout.mdp",
            "-maxwarn",
            "1",
        ],
        work_root,
        work_root / "grompp_nvt.stdout",
        work_root / "grompp_nvt.stderr",
    )
    run_command(
        [str(REPO_ROOT / "build" / "bin" / "gmx"), "mdrun", "-s", "nvt.tpr", "-deffnm", "nvt", "-nt", str(args.gmx_threads), "-pin", "off", "-reprod"],
        work_root,
        work_root / "mdrun_nvt.stdout",
        work_root / "mdrun_nvt.stderr",
    )
    run_command(
        [str(REPO_ROOT / "build" / "bin" / "gmx"), "energy", "-f", "nvt.edr", "-o", "nvt_energy.xvg"],
        work_root,
        work_root / "energy.stdout",
        work_root / "energy.stderr",
        stdin_text="Potential\nTemperature\n0\n",
    )

    rows = parse_xvg(work_root / "nvt_energy.xvg")
    analysis_start_ps = max(0.0, args.nvt_ps / 2.0)
    window = filter_time_window(rows, 0, analysis_start_ps)
    potential = [row[1] for row in window]
    temperature = [row[2] for row in window]
    mean_temp = statistics.fmean(temperature)
    max_temp = max(temperature)
    stability_pass = (abs(mean_temp - 300.0) <= args.stability_mean_temp_tolerance_k) and (max_temp <= args.stability_max_temp_k)

    report = {
        "system_id": system_id,
        "protocol": {
            "ensemble": "NVT",
            "duration_ps": args.nvt_ps,
            "analysis_window_ps": args.nvt_ps / 2.0,
            "thresholds": {
                "mean_temperature_target_k": 300.0,
                "mean_temperature_tolerance_k": args.stability_mean_temp_tolerance_k,
                "max_temperature_k": args.stability_max_temp_k,
            },
        },
        "temperature_k": summarize_series(temperature),
        "potential_energy_kj_mol": summarize_series(potential),
        "status": "PASS" if stability_pass else "FAIL",
        "artifacts": {
            "gromacs_root": str(work_root),
            "energy_xvg": str(work_root / "nvt_energy.xvg"),
            "log": str(work_root / "nvt.log"),
            "checkpoint": str(work_root / "nvt.cpt"),
        },
    }
    dump_json(work_root / "long_nvt_stability_report.json", report)
    return report


def main() -> int:
    args = parse_args()
    validate_args(args)
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    generated_paths, fixture_manifest = generate_fixture_bundle(
        args.system,
        out_root,
        None if args.fixture_manifest is None else args.fixture_manifest.resolve(),
    )
    pair_manifest = build_pair_manifest(
        args.system,
        out_root,
        generated_paths,
        fixture_manifest,
        None if args.fixture_manifest is None else args.fixture_manifest.resolve(),
        args.milestone_subset,
    )
    parity_report = run_paired_npt(args, generated_paths, out_root)
    stability_report = None if args.skip_nvt else run_long_nvt_stability(args, out_root)

    overall_pass = parity_report["status"] == "PASS" and (
        args.skip_nvt or (stability_report is not None and stability_report["status"] == "PASS")
    )
    summary = {
        "milestone_subset": args.milestone_subset,
        "system_id": args.system,
        "qualified_pair_manifest_status": pair_manifest["pair_status"],
        "dense_parity_status": parity_report["status"],
        "long_stability_status": "SKIPPED" if stability_report is None else stability_report["status"],
        "capability_subset_status": "PASS" if overall_pass else "FAIL",
        "old_boundary": "frozen PT8 SPE subset and small charged fixtures only",
        "new_candidate_boundary": {
            "qualified_charged_pair": args.system,
            "dense_parity": parity_report["status"],
            "long_horizon_stability": "SKIPPED" if stability_report is None else stability_report["status"],
        },
        "artifacts": {
            "qualified_pair_manifest": str(out_root / "qualified_pair_manifest.json"),
            "dense_npt_parity_report": str(out_root / "paired_npt" / "dense_npt_parity_report.json"),
            "long_nvt_stability_report": None
            if stability_report is None
            else str(out_root / "long_nvt_stability" / "long_nvt_stability_report.json"),
        },
    }
    dump_json(out_root / "m1_m3_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
