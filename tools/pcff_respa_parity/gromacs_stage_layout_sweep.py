#!/usr/bin/env python3
"""Sweep GROMACS stage layouts per generated PolyGen stage TPR."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

ATM_TO_BAR = 1.01325
GMX_PCFF_RUNTIME_ENV = (
    "GMX_PCFF_EXACT_RESPA_FUSED_INITIAL_DRIFT=1;"
    "GMX_PCFF_MIXED_CLASS2_LINEAR_ANGLE_SIN_FLOOR=0.00038;"
    "GMX_PCFF_MTTK_MASS_MODE=lammps;"
    "GMX_PCFF_NVT_MASS_MODE=lammps_tchain;"
    "GMX_PCFF_MTTK_BOXV_INTEGRATOR=lammps;"
    "GMX_PCFF_MTTK_EXTENDED_UPDATE_MODE=velocity-lammps-remap;"
    "GMX_PCFF_NHC_INTEGRATOR=lammps;"
    "GMX_PCFF_EXACT_RESPA_PRE_TROTTER=two;"
    "GMX_PCFF_EXACT_RESPA_POST_TROTTER=three;"
    "GMX_PCFF_ALLOW_LONG_EXCLUDED=1"
)


def parse_env_assignments(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw.split(";"):
        item = item.strip()
        if item and "=" in item:
            key, value = item.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def parse_perf(text: str) -> float | None:
    matches = re.findall(r"Performance:\s+([0-9.]+)", text)
    return float(matches[-1]) if matches else None


def load_context(md_dir: Path) -> dict[str, Any]:
    path = md_dir / "gromacs_runtime_context.json"
    if not path.exists():
        return {"stages": []}
    return json.loads(path.read_text(encoding="utf-8"))


def infer_gro_atom_count(path: Path) -> int | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) < 2:
            return None
        value = int(lines[1].strip())
    except Exception:
        return None
    return value if value > 0 else None


def apply_project_atom_count_env(env: dict[str, str], md_dir: Path) -> dict[str, str]:
    if env.get("GMX_PCFF_MTTK_LAMMPS_NATOMS"):
        return env
    natoms = infer_gro_atom_count(md_dir / "conf.gro")
    if natoms:
        env["GMX_PCFF_MTTK_LAMMPS_NATOMS"] = str(natoms)
        env.setdefault("GMX_PCFF_MTTK_LAMMPS_NATOMS_FALLBACK", str(natoms))
    return env


def stage_by_name(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(stage.get("name")): stage
        for stage in context.get("stages", [])
        if isinstance(stage, dict) and stage.get("name")
    }


def discover_tprs(md_dir: Path, stages_arg: str) -> dict[str, Path]:
    if stages_arg.strip().lower() == "auto":
        return {
            path.stem: path
            for path in sorted(md_dir.glob("*.tpr"))
            if path.stem.startswith("eq")
        }
    out: dict[str, Path] = {}
    for name in [x.strip() for x in stages_arg.split(",") if x.strip()]:
        path = md_dir / f"{name}.tpr"
        if path.exists():
            out[name] = path
    return out


def divisors(total: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for ntmpi in range(1, total + 1):
        if total % ntmpi == 0:
            out.append((ntmpi, total // ntmpi))
    return out


def layout_combos(totals: list[int], *, allow_domain_decomposition: bool) -> list[tuple[int, int, int]]:
    if allow_domain_decomposition:
        return [(total, ntmpi, ntomp) for total in totals for ntmpi, ntomp in divisors(total)]
    # Exact LAMMPS-style r-RESPA currently rejects domain decomposition.
    # Keep the requested total budgets but realize them as single-rank OpenMP.
    return [(total, 1, total) for total in totals]


def stage_runtime_env(base: dict[str, str], stage: dict[str, Any]) -> dict[str, str]:
    env = dict(base)
    if str(stage.get("kspace_compute", "")).lower() == "no":
        env["GMX_PCFF_EWALD_REAL_ONLY"] = "1"
        env.setdefault("GMX_PCFF_EXACT_RESPA_DISABLE_NBNXM_NARROW", "0")
    if str(stage.get("ensemble", "")) == "npt":
        outer_dt = stage.get("outer_timestep_fs")
        if outer_dt is not None:
            env["GMX_PCFF_MTTK_LAMMPS_PDAMP_PS"] = f"{float(outer_dt):.9g}"
        p0 = stage.get("pressure_start_atm")
        p1 = stage.get("pressure_end_atm")
        if p0 is not None and p1 is not None and abs(float(p0) - float(p1)) > 1.0e-12:
            env["GMX_PCFF_REFP_RAMP_START_BAR"] = f"{float(p0) * ATM_TO_BAR:.9g}"
            env["GMX_PCFF_REFP_RAMP_END_BAR"] = f"{float(p1) * ATM_TO_BAR:.9g}"
            env["GMX_PCFF_REFP_RAMP_DURATION_PS"] = f"{float(stage.get('nsteps', 0)) * float(stage.get('dt_ps', 0.0)):.9g}"
    return env


def lane_extra_args(lane: str, stage_meta: Mapping[str, Any] | None = None) -> list[str]:
    stage_name = str((stage_meta or {}).get("name", ""))
    if lane == "gmx_cpu":
        return ["-nb", "cpu", "-pme", "cpu", "-bonded", "cpu", "-update", "cpu"]
    if lane == "gmx_gpu":
        if "minimize" in stage_name.lower():
            return ["-nb", "cpu", "-pme", "cpu", "-bonded", "cpu", "-update", "cpu"]
        return ["-nb", "gpu", "-pme", "cpu", "-bonded", "gpu", "-update", "cpu", "-dlb", "no", "-notunepme"]
    raise ValueError(f"Unsupported lane={lane!r}")


def run_one(
    *,
    gmx: Path,
    source_tpr: Path,
    stage: str,
    stage_info: dict[str, Any],
    lane: str,
    total_cores: int,
    ntmpi: int,
    ntomp: int,
    outdir: Path,
    steps: int,
    timeout: int,
) -> dict[str, Any]:
    case = outdir / "runs" / f"{stage}_c{total_cores}_ntmpi{ntmpi}_ntomp{ntomp}"
    if case.exists():
        shutil.rmtree(case)
    case.mkdir(parents=True, exist_ok=True)
    short_tpr = case / "speed.tpr"
    env = os.environ.copy()
    env.update(parse_env_assignments(GMX_PCFF_RUNTIME_ENV))
    env = stage_runtime_env(env, stage_info)
    env = apply_project_atom_count_env(env, source_tpr.parent)
    convert = subprocess.run(
        [str(gmx), "convert-tpr", "-s", str(source_tpr), "-o", str(short_tpr), "-nsteps", str(int(steps))],
        cwd=case,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (case / "convert_tpr.stdout.log").write_text(convert.stdout, encoding="utf-8")
    if convert.returncode != 0:
        return {
            "stage": stage,
            "ntomp": ntomp,
            "returncode": convert.returncode,
            "error": "convert_tpr_failed",
            "log": str(case / "convert_tpr.stdout.log"),
        }
    cmd = [
        str(gmx),
        "mdrun",
        "-s",
        str(short_tpr),
        "-deffnm",
        "speed",
        "-ntmpi",
        str(int(ntmpi)),
        "-ntomp",
        str(int(ntomp)),
        "-pin",
        "off",
        "-noconfout",
        *lane_extra_args(lane, stage_info),
    ]
    log_path = case / "mdrun.stdout.log"
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=case,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(timeout),
            check=False,
        )
        text = proc.stdout
        rc = proc.returncode
        error = ""
    except subprocess.TimeoutExpired as exc:
        text = exc.stdout if isinstance(exc.stdout, str) else ""
        rc = -1
        error = f"timeout:{timeout}s"
    wall_s = time.time() - t0
    log_path.write_text(text, encoding="utf-8")
    return {
        "stage": stage,
        "lane": lane,
        "total_cores": int(total_cores),
        "ntmpi": int(ntmpi),
        "ntomp": int(ntomp),
        "returncode": int(rc),
        "wall_s": wall_s,
        "ns_per_day": parse_perf(text),
        "error": error,
        "log": str(log_path),
    }


def choose_best(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for stage in sorted({str(row["stage"]) for row in rows}):
        ok = [r for r in rows if r.get("stage") == stage and int(r.get("returncode", -1)) == 0]
        if not ok:
            continue
        with_perf = [r for r in ok if r.get("ns_per_day") is not None]
        if with_perf:
            chosen = max(with_perf, key=lambda row: float(row["ns_per_day"]))
        else:
            chosen = min(ok, key=lambda row: float(row.get("wall_s") or 1.0e99))
        out[stage] = {
            "ntmpi": int(chosen["ntmpi"]),
            "ntomp": int(chosen["ntomp"]),
            "extra_args": [],
            "source": str(chosen["log"]),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--md-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--gmx", type=Path, required=True)
    p.add_argument("--lane", choices=["gmx_cpu", "gmx_gpu"], required=True)
    p.add_argument("--totals", default="8,10,12")
    p.add_argument("--ntomp", default="")
    p.add_argument("--allow-domain-decomposition", action="store_true")
    p.add_argument("--stages", default="auto")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args()

    md_dir = args.md_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    context = load_context(md_dir)
    stages_info = stage_by_name(context)
    tprs = discover_tprs(md_dir, args.stages)
    if str(args.ntomp).strip():
        combos = [(int(x), 1, int(x)) for x in str(args.ntomp).split(",") if x.strip()]
    else:
        totals = [int(x) for x in str(args.totals).split(",") if x.strip()]
        combos = layout_combos(totals, allow_domain_decomposition=bool(args.allow_domain_decomposition))
    rows: list[dict[str, Any]] = []
    for stage, tpr in tprs.items():
        for total_cores, ntmpi, ntomp in combos:
            row = run_one(
                gmx=args.gmx.resolve(),
                source_tpr=tpr,
                stage=stage,
                stage_info=stages_info.get(stage, {}),
                lane=args.lane,
                total_cores=total_cores,
                ntmpi=ntmpi,
                ntomp=ntomp,
                outdir=outdir,
                steps=args.steps,
                timeout=args.timeout,
            )
            rows.append(row)
            print(row, flush=True)
    result_csv = outdir / "results.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with result_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    best = choose_best(rows)
    (outdir / "best_stage_layouts.json").write_text(
        json.dumps({"stage_layouts": best, "results_csv": str(result_csv)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("results:", result_csv)
    print("best:", outdir / "best_stage_layouts.json")


if __name__ == "__main__":
    main()
