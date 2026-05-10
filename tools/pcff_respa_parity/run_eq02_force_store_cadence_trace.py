#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
GMX = REPO / "build_gateb_cuda/bin/gmx"
BASE_WORK = REPO / "output/polygen_pcff_gromacs_initial_em_notebook/gromacs_cpu_openmp_fit_eq01_clean_20260502"
BASE_MDP = BASE_WORK / "02_eq02_npt_compress_100ps_chunk0001.mdp"
START_GRO = BASE_WORK / "01_eq01_nvt_50ps.gro"
TOPOL = BASE_WORK / "topol.top"


def run(cmd: list[str | Path], cwd: Path, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in cmd],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def replace_mdp(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^({re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(rf"\g<1>{value}", text, count=1)
    return text + f"\n{key} = {value}\n"


def write_mdp(path: Path, cadence: int, nsteps: int) -> None:
    text = BASE_MDP.read_text(encoding="utf-8", errors="ignore")
    for key, value in {
        "nsteps": str(nsteps),
        "nstcalcenergy": str(cadence),
        "nstenergy": str(cadence),
        "nstlog": str(cadence),
        "nstxout": "0",
        "nstvout": "0",
        "nstfout": "0",
        "nstxout-compressed": "0",
        "continuation": "yes",
        "gen-vel": "no",
        "nstlist": "1",
    }.items():
        text = replace_mdp(text, key, value)
    path.write_text(text, encoding="utf-8")


def trace_env(trace_dir: Path, trace_steps: str, ntomp: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(ntomp),
            "OMP_PROC_BIND": "true",
            "OMP_PLACES": "cores",
            "GMX_PCFF_EWALD_BETA_INV_A": "0.21160096",
            "GMX_PCFF_EWALD_REAL_ONLY": "1",
            "GMX_PCFF_EXACT_RESPA_FUSED_INITIAL_DRIFT": "1",
            "GMX_PCFF_EXACT_RESPA_PRE_TROTTER": "two",
            "GMX_PCFF_EXACT_RESPA_POST_TROTTER": "three",
            "GMX_PCFF_EXACT_RESPA_NBNXM_OWNER_STEP_SCALAR_FALLBACK": "0",
            "GMX_PCFF_MTTK_MASS_MODE": "lammps_pmass_pchain",
            "GMX_PCFF_MTTK_LAMMPS_NATOMS": "7075",
            "GMX_PCFF_MTTK_LAMMPS_PDAMP_PS": "0.5",
            "GMX_PCFF_MTTK_PRESSURE_MASS_SCALE": "1.21",
            "GMX_PCFF_MTTK_NRESET_STEPS": "80000",
            "GMX_PCFF_EXACT_RESPA_MTTK_EXTENDED_UPDATE": "velocity-lammps-remap",
            "GMX_PCFF_EXACT_RESPA_MTTK_INLINE_BOX_REMAP": "1",
            "GMX_PCFF_EXACT_RESPA_MTTK_VETA_SCALE": "1",
            "GMX_PCFF_REFP_RAMP_START_BAR": "1.01325",
            "GMX_PCFF_REFP_RAMP_END_BAR": "507.13162",
            "GMX_PCFF_REFP_RAMP_DURATION_PS": "50",
            "GMX_PCFF_EXACT_RESPA_FORCESTORE_SUMMARY_DIR": str(trace_dir),
            "GMX_PCFF_EXACT_RESPA_FORCESTORE_SUMMARY_STEPS": trace_steps,
        }
    )
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO / "output/probes/eq02_force_store_cadence_trace_20260503",
    )
    parser.add_argument("--cadences", type=int, nargs="+", default=[40000, 160000])
    parser.add_argument("--nsteps", type=int, default=40004)
    parser.add_argument("--trace-steps", default="0,40000,40004")
    parser.add_argument("--ntomp", type=int, default=12)
    parser.add_argument("--cores", default="0-11")
    args = parser.parse_args()

    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    for required in (GMX, BASE_MDP, START_GRO, TOPOL):
        if not required.exists():
            raise FileNotFoundError(required)

    for cadence in args.cadences:
        work = args.root / f"nstcalcenergy_{cadence}"
        trace_dir = work / "force_store_trace"
        work.mkdir(parents=True, exist_ok=True)
        trace_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(START_GRO, work / "start.gro")
        shutil.copy2(TOPOL, work / "topol.top")
        mdp = work / "eq02_trace.mdp"
        tpr = work / "eq02_trace.tpr"
        deffnm = work / "eq02_trace"
        write_mdp(mdp, cadence, args.nsteps)

        grompp = run([GMX, "grompp", "-f", mdp, "-c", "start.gro", "-p", "topol.top", "-o", tpr, "-maxwarn", "2"], cwd=work)
        (work / "grompp.stdout.log").write_text(grompp.stdout, encoding="utf-8")
        if grompp.returncode != 0:
            print(f"cadence={cadence} grompp_failed {work / 'grompp.stdout.log'}", flush=True)
            continue

        mdrun = run(
            [
                "taskset",
                "-c",
                args.cores,
                GMX,
                "mdrun",
                "-s",
                tpr,
                "-deffnm",
                deffnm,
                "-ntmpi",
                "1",
                "-ntomp",
                str(args.ntomp),
                "-pin",
                "off",
                "-dlb",
                "no",
                "-notunepme",
                "-nb",
                "cpu",
                "-pme",
                "cpu",
                "-bonded",
                "cpu",
                "-update",
                "cpu",
            ],
            cwd=work,
            env=trace_env(trace_dir, args.trace_steps, args.ntomp),
        )
        (work / "mdrun.stdout.log").write_text(mdrun.stdout, encoding="utf-8")
        print(
            f"cadence={cadence} mdrun_returncode={mdrun.returncode} "
            f"trace={trace_dir / 'force_store_update_summary.tsv'}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
