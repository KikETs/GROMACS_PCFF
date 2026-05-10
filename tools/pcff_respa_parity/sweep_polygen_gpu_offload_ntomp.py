#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from polygen_stage_metric_audit import DEFAULT_NOTEBOOK, load_notebook_config


OUT_ROOT = REPO / "output" / "polygen_pcff_gromacs_initial_em_notebook"
GPU_WORK = OUT_ROOT / "gromacs_gpu_hybrid"
CPU_WORK = OUT_ROOT / "gromacs_cpu_openmp"
DEFAULT_GMX = REPO / "build_gateb_cuda" / "bin" / "gmx"

PERFORMANCE_RE = re.compile(
    r"Performance:\s+(?P<ns_per_day>[0-9.eE+-]+)\s+(?P<hours_per_ns>[0-9.eE+-]+)\s+"
    r"(?P<ms_per_step>[0-9.eE+-]+)"
)

WALLCYCLE_LABELS = (
    "eR CPU listed",
    "eR bond H2D X",
    "eR bond list",
    "eR bond clear",
    "eR bond launch",
    "eR bond D2H F",
    "eR bond wait",
    "eR bond add F",
    "eR bond energy",
    "eR GPU wait NB",
    "PME mesh",
    "Update",
    "Total",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Short production-like GPU offload/ntomp sweep for the PolyGen exact r-RESPA PCFF lane. "
            "The sweep reuses the completed eq13 endpoint and does not touch existing equil/prod outputs."
        )
    )
    parser.add_argument("--gmx", type=Path, default=DEFAULT_GMX)
    parser.add_argument("--work", type=Path, default=GPU_WORK)
    parser.add_argument("--cpu-work", type=Path, default=CPU_WORK)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_ROOT / "gromacs_gpu_hybrid_offload_ntomp_sweep_20260507",
    )
    parser.add_argument("--start-stem", default="13_eq13_nvt_fixed_volume_1000ps_chunk0005")
    parser.add_argument("--stage-stem", default="14_prod01_nvt_10000ps_chunk0001")
    parser.add_argument("--nsteps", type=int, default=20000, help="mdrun -nsteps override; 20000 = 10 ps at 0.5 fs.")
    parser.add_argument("--ntomp-list", nargs="+", type=int, default=[8, 12, 16, 20, 24])
    parser.add_argument("--nstlist", type=int, default=None, help="Optional MDP nstlist override for speed probes.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "nb_gpu_pme_cpu_bonded_cpu",
            "nb_gpu_pme_cpu_bonded_pair14",
            "nb_gpu_pme_cpu_bonded_class2_pair14",
            "nb_gpu_pme_cpu_bonded_all",
        ],
    )
    parser.add_argument("--cpuset", default="0-23")
    parser.add_argument("--maxwarn", type=int, default=2)
    parser.add_argument("--timeout-sec", type=float, default=180.0, help="Per-case grompp/mdrun timeout; failed cases are recorded and skipped.")
    return parser.parse_args()


def mode_args_env(mode: str) -> tuple[list[str], dict[str, str]]:
    if mode == "nb_gpu_pme_cpu_bonded_cpu":
        return ["-nb", "gpu", "-pme", "cpu", "-bonded", "cpu", "-update", "cpu"], {}
    if mode == "nb_gpu_pme_cpu_bonded_pair14":
        return ["-nb", "gpu", "-pme", "cpu", "-bonded", "gpu", "-update", "cpu"], {
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_FTYPES": "pair14",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_CPU_LISTED_OVERLAP": "1",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_LIST_CACHE": "1",
        }
    if mode == "nb_gpu_pme_cpu_bonded_class2_pair14":
        return ["-nb", "gpu", "-pme", "cpu", "-bonded", "gpu", "-update", "cpu"], {
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_FTYPES": "class2-pair14",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_CPU_LISTED_OVERLAP": "1",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_LIST_CACHE": "1",
        }
    if mode == "nb_gpu_pme_cpu_bonded_all":
        return ["-nb", "gpu", "-pme", "cpu", "-bonded", "gpu", "-update", "cpu"], {
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_FTYPES": "all",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_CPU_LISTED_OVERLAP": "1",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_LIST_CACHE": "1",
        }
    if mode == "nb_gpu_pme_gpu_bonded_cpu_order4":
        return ["-nb", "gpu", "-pme", "gpu", "-bonded", "cpu", "-update", "cpu"], {}
    if mode == "nb_gpu_pme_gpu_bonded_all_order4":
        return ["-nb", "gpu", "-pme", "gpu", "-bonded", "gpu", "-update", "cpu"], {
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_FTYPES": "all",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_CPU_LISTED_OVERLAP": "1",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_LIST_CACHE": "1",
        }
    raise ValueError(f"Unsupported mode: {mode}")


def force_mdp_key(path: Path, key: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    prefix = f"{key:<24}="
    replaced = False
    for i, raw in enumerate(lines):
        if raw.strip().startswith(f"{key} ") or raw.strip().startswith(f"{key}="):
            comment = ""
            body = raw
            if ";" in raw:
                body, comment = raw.split(";", 1)
                comment = " ;" + comment
            lines[i] = f"{prefix} {value}{comment}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{prefix} {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_runtime_env(config: dict[str, Any], base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    env.setdefault("GMX_PCFF_EXACT_RESPA_WRAP_STATE_IN_BOX", "0")
    if config.get("GMX_MATCH_LAMMPS_TEMPERATURE_DOF"):
        env["GMX_PCFF_EXACT_RESPA_ALLOW_LINEAR_COM_REMOVAL"] = "1"
    nvt_mass_mode = config.get("GMX_PCFF_NVT_MASS_MODE") or config.get("GMX_PCFF_MTTK_MASS_MODE")
    if nvt_mass_mode:
        env["GMX_PCFF_MTTK_MASS_MODE"] = str(nvt_mass_mode)
        natoms = config.get("GMX_PCFF_MTTK_LAMMPS_NATOMS_FALLBACK")
        if natoms:
            env["GMX_PCFF_MTTK_LAMMPS_NATOMS"] = str(natoms)
    if config.get("GMX_PCFF_EXACT_RESPA_PRE_TROTTER") is not None:
        env["GMX_PCFF_EXACT_RESPA_PRE_TROTTER"] = str(config["GMX_PCFF_EXACT_RESPA_PRE_TROTTER"])
    post_trotter = config.get("GMX_PCFF_EXACT_RESPA_POST_TROTTER_BY_ENSEMBLE", {}).get(
        "nvt", config.get("GMX_PCFF_EXACT_RESPA_POST_TROTTER")
    )
    if post_trotter is not None:
        env["GMX_PCFF_EXACT_RESPA_POST_TROTTER"] = str(post_trotter)
    if config.get("GMX_PCFF_NHC_INTEGRATOR"):
        env["GMX_PCFF_NHC_INTEGRATOR"] = str(config["GMX_PCFF_NHC_INTEGRATOR"])
    env["GMX_PCFF_EXACT_RESPA_NBNXM_OWNER_STEP_SCALAR_FALLBACK"] = (
        "1" if config.get("GMX_NBNXM_OWNER_STEP_SCALAR_FALLBACK") else "0"
    )
    class2_floor = config.get("GMX_PCFF_MIXED_CLASS2_LINEAR_ANGLE_SIN_FLOOR")
    if class2_floor and Path(env.get("GMX_BINARY_NAME", "gmx")).name != "gmx_d":
        env["GMX_PCFF_CLASS2_LINEAR_ANGLE_SIN_FLOOR"] = str(class2_floor)
    return env


def run(cmd: list[str | Path], cwd: Path, env: dict[str, str], log: Path, timeout_sec: float | None = None) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [str(item) for item in cmd],
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        captured = exc.stdout or ""
        if isinstance(captured, bytes):
            captured = captured.decode("utf-8", errors="replace")
        log.write_text(
            "$ " + " ".join(str(item) for item in cmd) + f"\nTIMEOUT after {elapsed:.1f} s\n" + captured,
            encoding="utf-8",
        )
        raise RuntimeError(f"Command timed out after {elapsed:.1f} s: {' '.join(str(x) for x in cmd)}\nSee {log}") from exc
    elapsed = time.monotonic() - started
    log.write_text("$ " + " ".join(str(item) for item in cmd) + "\n" + completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(str(x) for x in cmd)}\nSee {log}")
    return elapsed


def parse_wallcycle_seconds(text: str, label: str) -> float | None:
    value: float | None = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped.startswith(label):
            continue
        rest = stripped[len(label) :].split()
        if label == "Total":
            if rest:
                try:
                    value = float(rest[0])
                except ValueError:
                    pass
        elif len(rest) >= 4:
            try:
                value = float(rest[3])
            except ValueError:
                pass
    return value


def parse_mdrun_log(path: Path) -> dict[str, float | None]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    metrics: dict[str, float | None] = {}
    match = PERFORMANCE_RE.search(text)
    metrics["ns_per_day"] = float(match.group("ns_per_day")) if match else None
    metrics["hours_per_ns"] = float(match.group("hours_per_ns")) if match else None
    metrics["ms_per_step"] = float(match.group("ms_per_step")) if match else None
    for label in WALLCYCLE_LABELS:
        key = label.lower().replace(" ", "_").replace(".", "").replace("+", "plus").replace(",", "")
        metrics[f"{key}_s"] = parse_wallcycle_seconds(text, label)
    return metrics


def write_summary(rows: list[dict[str, Any]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "mode",
        "ntomp",
        "ns_per_day",
        "ms_per_step",
        "elapsed_s",
        "eR_CPU_listed_s",
        "eR_bond_total_s",
        "eR_GPU_wait_NB_s",
        "PME_mesh_s",
        "total_wallcycle_s",
        "error",
        "case_dir",
    ]
    with (out / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    args = parse_args()
    gmx = args.gmx.resolve()
    work = args.work.resolve()
    cpu_work = args.cpu_work.resolve()
    out = args.out.resolve()
    config = load_notebook_config(args.notebook.resolve())

    top = work / "topol.top"
    mdp = work / f"{args.stage_stem}.mdp"
    if not mdp.exists():
        mdp = cpu_work / f"{args.stage_stem}.mdp"
    start_gro = work / f"{args.start_stem}.hi.gro"
    if not start_gro.exists():
        start_gro = work / f"{args.start_stem}.gro"
    start_cpt = work / f"{args.start_stem}.cpt"
    for path in (gmx, top, mdp, start_gro, start_cpt):
        if not path.exists():
            raise FileNotFoundError(path)

    base_env = os.environ.copy()
    base_env["GMX_MAXBACKUP"] = "-1"
    base_env["GMX_BINARY_NAME"] = gmx.name

    rows: list[dict[str, Any]] = []
    for mode in args.modes:
        mdrun_extra, mode_env = mode_args_env(mode)
        for ntomp in args.ntomp_list:
            case = out / mode / f"ntomp{ntomp:02d}"
            if case.exists():
                shutil.rmtree(case)
            case.mkdir(parents=True)
            shutil.copy2(top, case / "topol.top")
            shutil.copy2(mdp, case / "prod_probe.mdp")
            if "pme_gpu" in mode and "order4" in mode:
                force_mdp_key(case / "prod_probe.mdp", "pme-order", "4")
            if args.nstlist is not None:
                force_mdp_key(case / "prod_probe.mdp", "nstlist", str(args.nstlist))
            shutil.copy2(start_gro, case / start_gro.name)
            shutil.copy2(start_cpt, case / start_cpt.name)
            env = dict(base_env)
            env["OMP_NUM_THREADS"] = str(ntomp)
            env.update(mode_env)
            env = stage_runtime_env(config, env)

            tpr = case / "prod_probe.tpr"
            deffnm = case / "prod_probe"
            try:
                run(
                    [
                        gmx,
                        "grompp",
                        "-f",
                        "prod_probe.mdp",
                        "-c",
                        start_gro.name,
                        "-t",
                        start_cpt.name,
                        "-p",
                        "topol.top",
                        "-o",
                        tpr.name,
                        "-maxwarn",
                        str(args.maxwarn),
                    ],
                    cwd=case,
                    env=env,
                    log=case / "grompp.stdout.log",
                    timeout_sec=args.timeout_sec,
                )

                prefix: list[str | Path] = []
                if args.cpuset:
                    prefix = ["taskset", "-c", args.cpuset]
                elapsed = run(
                    [
                        *prefix,
                        gmx,
                        "mdrun",
                        "-s",
                        tpr.name,
                        "-deffnm",
                        deffnm.name,
                        "-nsteps",
                        str(args.nsteps),
                        "-ntmpi",
                        "1",
                        "-ntomp",
                        str(ntomp),
                        "-pin",
                        "off",
                        "-dlb",
                        "no",
                        "-notunepme",
                        *mdrun_extra,
                    ],
                    cwd=case,
                    env=env,
                    log=case / "mdrun.stdout.log",
                    timeout_sec=args.timeout_sec,
                )
                metrics = parse_mdrun_log(deffnm.with_suffix(".log"))
                error = None
            except Exception as exc:
                elapsed = None
                metrics = {}
                error = str(exc)
            bond_labels = [
                "er_bond_h2d_x_s",
                "er_bond_list_s",
                "er_bond_clear_s",
                "er_bond_launch_s",
                "er_bond_d2h_f_s",
                "er_bond_wait_s",
                "er_bond_add_f_s",
                "er_bond_energy_s",
            ]
            row = {
                "mode": mode,
                "ntomp": ntomp,
                "ns_per_day": metrics.get("ns_per_day"),
                "ms_per_step": metrics.get("ms_per_step"),
                "elapsed_s": elapsed,
                "error": error,
                "eR_CPU_listed_s": metrics.get("er_cpu_listed_s"),
                "eR_bond_total_s": sum(float(metrics.get(key) or 0.0) for key in bond_labels),
                "eR_GPU_wait_NB_s": metrics.get("er_gpu_wait_nb_s"),
                "PME_mesh_s": metrics.get("pme_mesh_s"),
                "total_wallcycle_s": metrics.get("total_s"),
                "case_dir": str(case),
                "metrics": metrics,
            }
            rows.append(row)
            rows.sort(
                key=lambda item: (
                    item["ns_per_day"] is not None,
                    float(item["ns_per_day"] or -1.0),
                ),
                reverse=True,
            )
            write_summary(rows, out)
            print(
                f"{mode} ntomp={ntomp}: ns/day={row['ns_per_day']} "
                f"eR_CPU_listed_s={row['eR_CPU_listed_s']} eR_bond_total_s={row['eR_bond_total_s']:.3f}"
                + (f" ERROR={error}" if error else ""),
                flush=True,
            )
    best = rows[0] if rows else None
    if best:
        print(
            f"BEST mode={best['mode']} ntomp={best['ntomp']} ns/day={best['ns_per_day']} "
            f"case={best['case_dir']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
