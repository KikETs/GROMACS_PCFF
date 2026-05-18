#!/usr/bin/env python3
"""Run short remote speed sweeps for PolyGen multi-system validation lanes."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


DEFAULT_OUTDIR = Path("GROMACS_PCFF/output/polygen_multisystem_validation_20260512_m1p50")

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


def parse_env_assignments(value: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in value.split(";"):
        if not item.strip():
            continue
        key, _, val = item.partition("=")
        if key:
            env[key] = val
    return env


def default_mpirun_binary() -> str:
    return shutil.which("mpirun") or shutil.which("mpiexec") or "mpirun"


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, stdout: Path | None = None) -> tuple[int, float]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    start = time.monotonic()
    if stdout is None:
        proc = subprocess.run(cmd, cwd=cwd, env=merged_env, text=True)
    else:
        with stdout.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=cwd, env=merged_env, text=True, stdout=fh, stderr=subprocess.STDOUT)
    return proc.returncode, time.monotonic() - start


def parse_gmx_performance(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Performance:\s+([0-9.]+)", text)
    return float(matches[-1]) if matches else None


def parse_lammps_performance(log_path: Path) -> tuple[float | None, float | None]:
    if not log_path.exists():
        return None, None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = [float(x) for x in re.findall(r"Performance:\s+([0-9.]+)\s+ns/day", text)]
    if not matches:
        return None, None
    return matches[0], matches[-1]


def find_one(pattern: str) -> Path:
    hits = sorted(Path(hit) for hit in glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def parse_gmx_tpr_dump(text: str) -> dict[str, object]:
    """Extract the geometry fields that make a speed probe meaningful."""
    out: dict[str, object] = {}
    for key, pattern in {
        "natoms": r"\bnatoms\s*=\s*([0-9]+)",
        "fourier_nx": r"\bfourier-nx\s*=\s*([0-9]+)",
        "fourier_ny": r"\bfourier-ny\s*=\s*([0-9]+)",
        "fourier_nz": r"\bfourier-nz\s*=\s*([0-9]+)",
    }.items():
        match = re.search(pattern, text)
        if match:
            out[key] = int(match.group(1))

    # GROMACS dump prints the diagonal box components as e.g.
    # box[    0]={ 4.23768e+00,  0.00000e+00,  0.00000e+00}
    box: list[float] = []
    for axis in range(3):
        match = re.search(
            rf"box\[\s*{axis}\]\s*=\s*\{{\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)",
            text,
        )
        if match:
            box.append(float(match.group(axis + 1)))
    if len(box) == 3:
        out["box_nm"] = box
        out["box_max_nm"] = max(box)
    grid = [out.get("fourier_nx"), out.get("fourier_ny"), out.get("fourier_nz")]
    if all(isinstance(x, int) for x in grid):
        out["pme_grid"] = grid
        out["pme_grid_max"] = max(int(x) for x in grid)
    return out


def inspect_gmx_tpr(binary: Path, tpr: Path, case: Path) -> dict[str, object]:
    dump_log = case / "tpr_dump.stdout.log"
    rc, wall = run([str(binary), "dump", "-s", str(tpr)], cwd=case, stdout=dump_log)
    info = {
        "tpr_dump_returncode": rc,
        "tpr_dump_wall_s": round(wall, 3),
    }
    if rc != 0:
        return info
    info.update(parse_gmx_tpr_dump(dump_log.read_text(encoding="utf-8", errors="ignore")))
    return info


def reject_tpr_geometry(info: dict[str, object], *, max_box_nm: float, max_pme_grid: int) -> str | None:
    box_max = info.get("box_max_nm")
    if isinstance(box_max, (int, float)) and float(box_max) > max_box_nm:
        return f"box_max_nm>{max_box_nm:g}"
    grid_max = info.get("pme_grid_max")
    if isinstance(grid_max, int) and grid_max > max_pme_grid:
        return f"pme_grid_max>{max_pme_grid}"
    if info.get("tpr_dump_returncode") != 0:
        return "tpr_dump_failed"
    return None


def patch_lammps_input(src_md: Path, dst_md: Path, eqfactor: float) -> Path:
    if dst_md.exists():
        shutil.rmtree(dst_md)
    shutil.copytree(src_md, dst_md)
    path = dst_md / "production.in"
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"^variable\s+eqfactor\s+equal\s+\S+",
        f"variable        eqfactor    equal {eqfactor}",
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(
        r"^variable\s+dumpfile\s+index\s+\S+",
        "variable        dumpfile    index speed_probe.lammpstrj",
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(
        r"(write_restart\s+final\.restart\s*)\n+(\s*# Compute clusters)",
        r"\1\nquit\n\n\2",
        text,
        count=1,
        flags=re.M,
    )
    path.write_text(text, encoding="utf-8")
    return path


def gmx_sweep(args: argparse.Namespace, outroot: Path, rows: list[dict[str, object]]) -> None:
    if str(args.run_group).startswith("smoke_") and not args.allow_smoke_tpr:
        raise ValueError(
            "Refusing to run GROMACS speed sweep from smoke outputs. "
            "Smoke runs deliberately use only a few NPT steps, so their sparse initial boxes "
            "produce inflated PME grids and invalid ns/day numbers. Use --run-group main20 "
            "after equilibration, or pass --allow-smoke-tpr only for executable smoke checks."
        )
    base = Path(args.outdir) / "runs_batch" / str(args.run_group) / args.role
    lanes = [lane.strip() for lane in str(args.gmx_lanes).split(",") if lane.strip()]
    invalid = [lane for lane in lanes if lane not in {"gmx_cpu", "gmx_gpu"}]
    if invalid:
        raise ValueError(f"Invalid --gmx-lanes entries: {invalid}")
    tpr_by_lane = {
        lane: find_one(str(base / lane / "Traj_*" / "MD_GMX" / "prod_nvt.tpr"))
        for lane in lanes
    }
    bin_by_lane = {
        "gmx_cpu": Path(args.gmx_cpu_binary),
        "gmx_gpu": Path(args.gmx_gpu_binary),
    }
    extra_by_lane = {
        "gmx_cpu": ["-nb", "cpu", "-pme", "cpu", "-bonded", "cpu", "-update", "cpu"],
        "gmx_gpu": ["-nb", "gpu", "-pme", "gpu", "-bonded", "gpu", "-update", "cpu"],
    }
    for lane in lanes:
        binary = bin_by_lane[lane]
        if not binary.exists():
            rows.append({"role": args.role, "lane": lane, "status": "missing_binary", "binary": str(binary)})
            continue
        for ntomp in args.gmx_ntomp:
            case = outroot / f"{lane}_ntomp{ntomp}"
            case.mkdir(parents=True, exist_ok=True)
            tpr = case / "speed.tpr"
            convert_rc, convert_wall = run(
                [str(binary), "convert-tpr", "-s", str(tpr_by_lane[lane]), "-o", str(tpr), "-nsteps", str(args.gmx_steps)],
                cwd=case,
                stdout=case / "convert_tpr.stdout.log",
            )
            if convert_rc != 0:
                rows.append({
                    "role": args.role,
                    "lane": lane,
                    "ntomp": ntomp,
                    "status": "convert_tpr_failed",
                    "wall_s": round(convert_wall, 3),
                })
                continue
            tpr_info = inspect_gmx_tpr(binary, tpr, case)
            reject_reason = reject_tpr_geometry(
                tpr_info,
                max_box_nm=float(args.max_box_nm),
                max_pme_grid=int(args.max_pme_grid),
            )
            if reject_reason:
                rows.append({
                    "role": args.role,
                    "lane": lane,
                    "ntomp": ntomp,
                    "steps": args.gmx_steps,
                    "status": "rejected_bad_tpr_geometry",
                    "reason": reject_reason,
                    "source_tpr": str(tpr_by_lane[lane]),
                    "workdir": str(case),
                    "binary": str(binary),
                    **tpr_info,
                })
                continue
            cmd = [
                str(binary),
                "mdrun",
                "-s",
                str(tpr),
                "-deffnm",
                "speed",
                "-ntmpi",
                "1",
                "-ntomp",
                str(ntomp),
                "-pin",
                "on",
                "-noconfout",
                *extra_by_lane[lane],
            ]
            env = {"GMX_MAXBACKUP": "-1", **parse_env_assignments(GMX_PCFF_RUNTIME_ENV)}
            rc, wall = run(cmd, cwd=case, env=env, stdout=case / "mdrun.stdout.log")
            rows.append({
                "role": args.role,
                "lane": lane,
                "ntomp": ntomp,
                "steps": args.gmx_steps,
                "status": "ok" if rc == 0 else "failed",
                "returncode": rc,
                "wall_s": round(wall, 3),
                "ns_per_day": parse_gmx_performance(case / "speed.log"),
                "workdir": str(case),
                "binary": str(binary),
                "source_tpr": str(tpr_by_lane[lane]),
                **tpr_info,
            })


def lammps_sweep(args: argparse.Namespace, outroot: Path, rows: list[dict[str, object]]) -> None:
    base = Path(args.outdir) / "runs_batch" / str(args.run_group) / args.role
    src_md = find_one(str(base / "lammps_cpu" / "Traj_*" / "MD"))
    binary = Path(args.lmp_binary)
    if not binary.exists():
        rows.append({"role": args.role, "lane": "lammps_cpu", "status": "missing_binary", "binary": str(binary)})
        return
    eqfactor = args.lammps_steps * args.lammps_dt_fs / 1_000_000.0
    for nproc in args.lammps_nproc:
        case = outroot / f"lammps_cpu_nproc{nproc}"
        md = case / "MD"
        input_path = patch_lammps_input(src_md, md, eqfactor)
        cmd = [
            str(Path(args.mpirun_binary)),
            "-np",
            str(nproc),
            str(binary),
            "-in",
            str(input_path.name),
        ]
        rc, wall = run(
            cmd,
            cwd=md,
            env={"OMP_NUM_THREADS": "1", "LAMMPS_POTENTIALS": os.environ.get("LAMMPS_POTENTIALS", "")},
            stdout=md / "speed.stdout.log",
        )
        md_ns_day, last_ns_day = parse_lammps_performance(md / "log.lammps")
        rows.append({
            "role": args.role,
            "lane": "lammps_cpu",
            "nproc": nproc,
            "steps": args.lammps_steps,
            "status": "ok" if rc == 0 else "failed",
            "returncode": rc,
            "wall_s": round(wall, 3),
            "ns_per_day": md_ns_day,
            "last_ns_per_day": last_ns_day,
            "workdir": str(md),
            "binary": str(binary),
        })


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", required=True)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--gmx-cpu-binary", default="GROMACS_PCFF/build_gateb_double_cpu/bin/gmx_d")
    p.add_argument("--gmx-gpu-binary", default="GROMACS_PCFF/build_gateb_cuda/bin/gmx")
    p.add_argument("--lmp-binary", default=str(Path.home() / ".local/lammps/lmp"))
    p.add_argument("--mpirun-binary", default=default_mpirun_binary())
    p.add_argument("--gmx-ntomp", type=parse_int_list, default=parse_int_list("2,4,6,8,10,12"))
    p.add_argument("--lammps-nproc", type=parse_int_list, default=parse_int_list("4,6,8,10,12"))
    p.add_argument("--gmx-steps", type=int, default=2000)
    p.add_argument("--run-group", default="main20")
    p.add_argument("--allow-smoke-tpr", action="store_true")
    p.add_argument("--max-box-nm", type=float, default=12.0)
    p.add_argument("--max-pme-grid", type=int, default=96)
    p.add_argument("--gmx-lanes", default="gmx_cpu,gmx_gpu")
    p.add_argument("--lammps-steps", type=int, default=2000)
    p.add_argument("--lammps-dt-fs", type=float, default=2.0)
    p.add_argument("--skip-lammps", action="store_true")
    p.add_argument("--skip-gromacs", action="store_true")
    args = p.parse_args()
    workspace = Path.cwd()
    args.outdir = args.outdir if args.outdir.is_absolute() else workspace / args.outdir
    for attr in ("gmx_cpu_binary", "gmx_gpu_binary", "lmp_binary", "mpirun_binary"):
        value = Path(getattr(args, attr))
        setattr(args, attr, str(value if value.is_absolute() else workspace / value))

    outroot = args.outdir / "speed_sweeps" / f"{args.role}_{args.tag}"
    outroot.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    if not args.skip_lammps:
        lammps_sweep(args, outroot, rows)
    if not args.skip_gromacs:
        gmx_sweep(args, outroot, rows)

    csv_path = outroot / "speed_sweep_summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (outroot / "speed_sweep_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(csv_path)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
