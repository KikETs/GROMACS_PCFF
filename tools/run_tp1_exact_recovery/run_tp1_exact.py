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
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.pcff_fixture_bridge.common import build_typed_ir, dump_json, render_gromacs_topology  # noqa: E402
from tools.run_m10_0_short_workflow.run_m10_0 import create_gro_from_lammps  # noqa: E402


DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "tp1_exact_recovery" / "dense_salt_polymer_corrected_npt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the corrected exact TP1 dense_salt_polymer recovery path.")
    parser.add_argument("--system", default="dense_salt_polymer")
    parser.add_argument("--duration-ps", type=float, default=5000.0)
    parser.add_argument("--analysis-window-ps", type=float, default=1000.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--ntmpi", type=int, default=1)
    parser.add_argument("--ntomp", type=int, default=1)
    parser.add_argument("--mean-temp-tolerance-k", type=float, default=20.0)
    parser.add_argument("--max-temp-k", type=float, default=400.0)
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def run_command(cmd: list[str], work_dir: Path, stdout_path: Path, stderr_path: Path, stdin_text: str | None = None) -> None:
    result = subprocess.run(
        cmd,
        cwd=work_dir,
        input=stdin_text,
        capture_output=True,
        text=True,
        errors="replace",
        env={**os.environ, "GMX_MAXBACKUP": "-1"},
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def build_min_mdp() -> str:
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
            "pme-order = 4",
            "fourierspacing = 0.12",
            "ewald-rtol = 1e-5",
            "vdw-type = Cut-off",
            "vdw-modifier = none",
            "rvdw = 0.9",
            "DispCorr = no",
            "pbc = xyz",
            "",
        ]
    )


def build_corrected_npt_mdp(duration_ps: float, seed: int) -> str:
    return "\n".join(
        [
            "; Corrected TP1 exact-blocker NPT protocol.",
            "; This intentionally fixes the old TP1.2 tcouple/pcouple/gen_vel key typo.",
            "integrator = md",
            "dt = 0.001",
            f"nsteps = {int(duration_ps / 0.001)}",
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
            "nstenergy = 1000",
            "nstlog = 1000",
            "nstxout-compressed = 0",
            "tcoupl = v-rescale",
            "tc-grps = System",
            "tau-t = 0.5",
            "ref-t = 300",
            "pcoupl = Berendsen",
            "pcoupltype = isotropic",
            "tau-p = 5.0",
            "compressibility = 4.5e-5",
            "ref-p = 1.0",
            "gen-vel = yes",
            "gen-temp = 300",
            f"gen-seed = {seed}",
            "constraints = none",
            "",
        ]
    )


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
        block = values[start:end]
        if block:
            means.append(statistics.fmean(block))
    if len(means) < 2:
        return None
    return statistics.stdev(means) / math.sqrt(len(means))


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "sem_block5": block_sem(values),
    }


def build_system_bundle(system_id: str, work_dir: Path) -> dict[str, str]:
    corpus_root = REPO_ROOT / "testdata" / "lammps_golden"
    system_root = corpus_root / "systems" / system_id
    typed_ir = build_typed_ir({"id": system_id, "path": f"systems/{system_id}"}, corpus_root)
    dump_json(work_dir / "typed_system.json", typed_ir)
    write_text(work_dir / "system.top", render_gromacs_topology(typed_ir))
    create_gro_from_lammps(system_root / "lammps" / "system.data", work_dir / "system.gro")
    shutil.copy(system_root / "lammps" / "system.data", work_dir / "system.data")
    shutil.copy(system_root / "lammps" / "system.in", work_dir / "system.in")
    shutil.copy(system_root / "system.json", work_dir / "system.json")
    return {
        "typed_system": str(work_dir / "typed_system.json"),
        "topology": str(work_dir / "system.top"),
        "coordinates": str(work_dir / "system.gro"),
        "lammps_data": str(work_dir / "system.data"),
        "lammps_input": str(work_dir / "system.in"),
        "system_json": str(work_dir / "system.json"),
    }


def assert_corrected_mdp(mdout_path: Path) -> dict[str, str | bool]:
    text = read_text(mdout_path)
    checks = {
        "tcoupl_v_rescale": "tcoupl                   = v-rescale" in text,
        "pcoupl_berendsen": "pcoupl                   = Berendsen" in text,
        "gen_vel_yes": "gen-vel                  = yes" in text or "gen_vel                  = yes" in text,
    }
    checks["all_pass"] = all(checks.values())
    if not checks["all_pass"]:
        raise RuntimeError(f"Corrected MDP contract failed: {checks}")
    return checks


def analyze_energy(work_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    run_command(
        [str(REPO_ROOT / "build" / "bin" / "gmx"), "energy", "-f", "tp1_equil.edr", "-o", "tp1_energy.xvg"],
        work_dir,
        work_dir / "energy.stdout",
        work_dir / "energy.stderr",
        stdin_text="Potential\nTemperature\nPressure\nVolume\nDensity\n0\n",
    )
    rows = parse_xvg(work_dir / "tp1_energy.xvg")
    if not rows:
        raise RuntimeError("No TP1 energy rows were extracted.")
    analysis_start_ps = max(0.0, args.duration_ps - args.analysis_window_ps)
    window = [row for row in rows if row[0] >= analysis_start_ps]
    potential = [row[1] for row in window]
    temperature = [row[2] for row in window]
    pressure = [row[3] for row in window]
    volume = [row[4] for row in window]
    density = [row[5] for row in window]
    pass_status = (
        rows[-1][0] >= args.duration_ps
        and abs(statistics.fmean(temperature) - 300.0) <= args.mean_temp_tolerance_k
        and max(temperature) <= args.max_temp_k
    )
    return {
        "duration_completed_ps": rows[-1][0],
        "analysis_start_ps": analysis_start_ps,
        "potential_energy_kj_mol": summarize(potential),
        "temperature_k": summarize(temperature),
        "pressure_bar": summarize(pressure),
        "volume_nm3": summarize(volume),
        "density_kg_m3": summarize(density),
        "status": "PASS" if pass_status else "FAIL",
    }


def main() -> int:
    args = parse_args()
    out_root = args.out.resolve()
    if out_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {out_root}")
    out_root.mkdir(parents=True)
    generated = build_system_bundle(args.system, out_root)

    protocol = {
        "milestone": "TP1 exact blocker recovery",
        "system_id": args.system,
        "source": "testdata/lammps_golden/systems/dense_salt_polymer",
        "duration_ps": args.duration_ps,
        "analysis_window_ps": args.analysis_window_ps,
        "corrected_old_runner_defect": "tcouple/pcouple/gen_vel typo corrected to tcoupl/pcoupl/gen-vel",
        "thresholds": {
            "mean_temperature_target_k": 300.0,
            "mean_temperature_tolerance_k": args.mean_temp_tolerance_k,
            "max_temperature_k": args.max_temp_k,
        },
        "grompp_maxwarn": {
            "tp1_equil": 1,
            "reason": "Berendsen is retained to rerun the historical TP1 intended protocol; GROMACS emits a non-production ensemble warning.",
        },
        "artifacts": generated,
    }
    dump_json(out_root / "tp1_exact_protocol.json", protocol)

    write_text(out_root / "min.mdp", build_min_mdp())
    write_text(out_root / "tp1_equil.mdp", build_corrected_npt_mdp(args.duration_ps, args.seed))
    run_command(
        [
            str(REPO_ROOT / "build" / "bin" / "gmx"),
            "grompp",
            "-f",
            "min.mdp",
            "-c",
            "system.gro",
            "-p",
            "system.top",
            "-o",
            "min.tpr",
            "-po",
            "min_mdout.mdp",
        ],
        out_root,
        out_root / "grompp_min.stdout",
        out_root / "grompp_min.stderr",
    )
    run_command(
        [str(REPO_ROOT / "build" / "bin" / "gmx"), "mdrun", "-s", "min.tpr", "-deffnm", "min", "-ntmpi", "1", "-ntomp", "1", "-pin", "off", "-reprod"],
        out_root,
        out_root / "mdrun_min.stdout",
        out_root / "mdrun_min.stderr",
    )
    run_command(
        [
            str(REPO_ROOT / "build" / "bin" / "gmx"),
            "grompp",
            "-f",
            "tp1_equil.mdp",
            "-c",
            "min.gro",
            "-p",
            "system.top",
            "-o",
            "tp1_equil.tpr",
            "-po",
            "tp1_equil_mdout.mdp",
            "-maxwarn",
            "1",
        ],
        out_root,
        out_root / "grompp_tp1.stdout",
        out_root / "grompp_tp1.stderr",
    )
    mdout_checks = assert_corrected_mdp(out_root / "tp1_equil_mdout.mdp")
    run_command(
        [
            str(REPO_ROOT / "build" / "bin" / "gmx"),
            "mdrun",
            "-s",
            "tp1_equil.tpr",
            "-deffnm",
            "tp1_equil",
            "-ntmpi",
            str(args.ntmpi),
            "-ntomp",
            str(args.ntomp),
            "-pin",
            "off",
            "-reprod",
        ],
        out_root,
        out_root / "mdrun_tp1.stdout",
        out_root / "mdrun_tp1.stderr",
    )
    analysis = analyze_energy(out_root, args)
    raw_artifacts = {
        "log": str(out_root / "tp1_equil.log"),
        "energy": str(out_root / "tp1_equil.edr"),
        "energy_xvg": str(out_root / "tp1_energy.xvg"),
        "checkpoint": str(out_root / "tp1_equil.cpt"),
        "coordinates": str(out_root / "tp1_equil.gro"),
        "run_input": str(out_root / "tp1_equil.tpr"),
        "mdp": str(out_root / "tp1_equil.mdp"),
        "mdout": str(out_root / "tp1_equil_mdout.mdp"),
        "mdrun_stderr": str(out_root / "mdrun_tp1.stderr"),
    }
    report = {
        "milestone": "TP1 exact blocker recovery",
        "system_id": args.system,
        "claim_scope": "historical dense_salt_polymer exact-system corrected NPT stability only",
        "protocol": protocol,
        "mdout_contract": mdout_checks,
        "analysis": analysis,
        "raw_artifacts": raw_artifacts,
        "status": analysis["status"],
        "known_limitations": [
            "This does not establish charged transport readiness.",
            "This does not close dense GROMACS-vs-LAMMPS density parity unless evaluated separately.",
        ],
    }
    dump_json(out_root / "tp1_exact_recovery_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
