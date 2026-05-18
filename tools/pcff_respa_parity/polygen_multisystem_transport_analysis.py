#!/usr/bin/env python3
"""Run cNE0/NE transport analysis for PolyGen multi-system batch outputs."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analyze_polygen_transport import (
    LaneSpec,
    analyze_cache,
    parse_first_lammps_frame,
    read_lammps_selected_frame,
    read_xvg_matrix,
    save_cache,
    unwrap_gromacs,
    write_index_file,
)


DEFAULT_OUTDIR = Path("GROMACS_PCFF/output/polygen_multisystem_validation_20260512_m1p50")


def batch_traj_id(row: pd.Series) -> int:
    return int(row["trajectory_id"]) * 100 + int(row["replica"])


def analysis_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        temperature=float(args.temperature),
        z=float(args.z),
        max_cluster=int(args.max_cluster),
        cluster_cutoff_angstrom=float(args.cluster_cutoff_angstrom),
        cluster_sample_stride=int(args.cluster_sample_stride),
        msd_lags=int(args.msd_lags),
    )


def load_lammps_project_cache(
    project: Path,
    topology,
    cache_path: Path,
    *,
    lane_key: str,
    prod_dt_fs: float,
) -> int:
    dump_path = project / "MD/traj.lammpstrj"
    if not dump_path.exists():
        raise FileNotFoundError(f"missing LAMMPS production dump: {dump_path}")

    times = []
    boxes = []
    wrapped_frames = []
    unwrapped_frames = []
    first_step = None
    with dump_path.open() as handle:
        while True:
            frame = read_lammps_selected_frame(handle, topology)
            if frame is None:
                break
            step, box, wrapped, unwrapped = frame
            if first_step is None:
                first_step = step
            times.append((step - first_step) * float(prod_dt_fs) / 1000.0)
            boxes.append(box)
            wrapped_frames.append(wrapped)
            unwrapped_frames.append(unwrapped)

    if len(times) < 5:
        raise RuntimeError(f"not enough LAMMPS frames for MSD/cNE0 analysis: {len(times)}")

    save_cache(
        cache_path,
        LaneSpec(lane_key, "lammps", project),
        topology,
        np.asarray(times, dtype=np.float64),
        np.stack(boxes),
        np.stack(wrapped_frames),
        np.stack(unwrapped_frames),
    )
    return len(times)


def load_gromacs_project_cache(
    project: Path,
    topology,
    cache_path: Path,
    *,
    lane_key: str,
    gmx_binary: Path,
) -> int:
    md_dir = project / "MD_GMX"
    traj_path = md_dir / "prod_nvt.xtc"
    if not traj_path.exists():
        traj_path = md_dir / "prod_nvt.trr"
    tpr_path = md_dir / "prod_nvt.tpr"
    if not traj_path.exists() or not tpr_path.exists():
        raise FileNotFoundError(f"missing GROMACS production trajectory/tpr under {md_dir}")
    if not gmx_binary.exists():
        raise FileNotFoundError(f"missing gmx binary: {gmx_binary}")

    with tempfile.TemporaryDirectory(prefix="polygen_multisystem_gmx_") as tmp_s:
        tmp = Path(tmp_s)
        ndx = tmp / "ion_atoms.ndx"
        coord_xvg = tmp / "coord.xvg"
        box_xvg = tmp / "box.xvg"
        write_index_file(ndx, topology.ion_atom_ids)
        proc = subprocess.run(
            [
                str(gmx_binary),
                "traj",
                "-f",
                str(traj_path),
                "-s",
                str(tpr_path),
                "-n",
                str(ndx),
                "-ox",
                str(coord_xvg),
                "-ob",
                str(box_xvg),
                "-xvg",
                "none",
                "-fp",
            ],
            input="0\n",
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"gmx traj failed for {traj_path}\n{proc.stderr[-4000:]}")

        coord = read_xvg_matrix(coord_xvg)
        box = read_xvg_matrix(box_xvg)

    if coord.shape[0] < 5:
        raise RuntimeError(f"not enough GROMACS frames for MSD/cNE0 analysis: {coord.shape[0]}")
    if coord.shape[0] != box.shape[0]:
        raise RuntimeError(f"GROMACS coord/box frame mismatch: {coord.shape[0]} vs {box.shape[0]}")
    if coord.shape[1] != 1 + 3 * len(topology.ion_atom_ids):
        raise RuntimeError(
            f"Unexpected GROMACS coordinate columns: {coord.shape[1]} for {len(topology.ion_atom_ids)} ion atoms"
        )

    wrapped = coord[:, 1:].reshape(coord.shape[0], len(topology.ion_atom_ids), 3).astype(np.float32)
    boxes = box[:, 1:4].astype(np.float32)
    unwrapped = unwrap_gromacs(wrapped, boxes)
    time_ps = coord[:, 0].astype(np.float64)
    time_ps = time_ps - time_ps[0]
    save_cache(cache_path, LaneSpec(lane_key, "gromacs", project, gmx_binary), topology, time_ps, boxes, wrapped, unwrapped)
    return int(coord.shape[0])


def analyze_job(
    job: pd.Series,
    *,
    outdir: Path,
    args: argparse.Namespace,
) -> dict:
    lane = str(job["lane"])
    run_group = str(job["run_group"])
    role = str(job["worker_role"])
    traj = batch_traj_id(job)
    project = outdir / "runs_batch" / run_group / role / lane / f"Traj_{traj}"
    lammps_project = outdir / "runs_batch" / run_group / role / "lammps_cpu" / f"Traj_{traj}"
    topology_dump = lammps_project / "MD/traj.lammpstrj"
    if not topology_dump.exists():
        raise FileNotFoundError(f"missing sibling LAMMPS topology dump: {topology_dump}")

    topology = parse_first_lammps_frame(topology_dump)
    lane_out = project / "transport_analysis"
    cache_path = lane_out / "cache" / f"{lane}_ion_trajectory.npz"
    if args.force or not cache_path.exists():
        if lane == "lammps_cpu":
            frames = load_lammps_project_cache(
                project,
                topology,
                cache_path,
                lane_key=lane,
                prod_dt_fs=float(args.lammps_prod_dt_fs),
            )
        elif lane == "gmx_cpu":
            frames = load_gromacs_project_cache(project, topology, cache_path, lane_key=lane, gmx_binary=args.gmx_cpu)
        elif lane == "gmx_gpu":
            frames = load_gromacs_project_cache(project, topology, cache_path, lane_key=lane, gmx_binary=args.gmx_gpu)
        else:
            raise RuntimeError(f"unsupported lane: {lane}")
    else:
        frames = int(np.load(cache_path)["time_ps"].shape[0])

    summary = analyze_cache(cache_path, topology, analysis_args(args), lane_out)
    return {
        "status": "ok",
        "project_dir": str(project),
        "frames": frames,
        "summary_json": str(lane_out / lane / "summary.json"),
        "time_end_ps": summary.get("time_end_ps"),
        "NE_msd_fit_S_cm": summary["NE_msd_fit"]["conductivity_s_cm"],
        "cNE0_raw_htpmd_S_cm": summary["cNE0_htpmd"]["conductivity_s_cm"],
        "cNE0_msd_fit_S_cm": summary["cNE0_msd_fit"]["conductivity_s_cm"],
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    jobs = pd.read_csv(args.jobs_csv)
    mask = jobs["worker_role"].eq(args.role)
    if args.run_group:
        mask &= jobs["run_group"].eq(args.run_group)
    lanes = {x.strip() for x in args.lanes.split(",") if x.strip()}
    mask &= jobs["lane"].isin(lanes)
    selected = jobs[mask].drop_duplicates(
        ["run_group", "worker_role", "lane", "trajectory_id", "replica"]
    )
    if args.max_jobs:
        selected = selected.head(int(args.max_jobs))

    rows = []
    for _, job in selected.iterrows():
        base = {
            "run_group": job["run_group"],
            "worker_role": job["worker_role"],
            "lane": job["lane"],
            "trajectory_id": job["trajectory_id"],
            "replica": job["replica"],
            "system_key": job["system_key"],
        }
        try:
            result = analyze_job(job, outdir=args.outdir, args=args)
            rows.append({**base, **result, "error": ""})
            print("ok", base)
        except Exception as exc:
            rows.append({**base, "status": "failed", "error": repr(exc)})
            print("failed", base, repr(exc))
            if not args.keep_going:
                raise

    df = pd.DataFrame(rows)
    group_label = args.run_group if args.run_group else "all"
    report_path = args.outdir / "analysis" / f"transport_analysis_status_{args.role}_{group_label}.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_path, index=False)
    print(report_path)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--jobs-csv", type=Path, default=DEFAULT_OUTDIR / "manifest/jobs.csv")
    p.add_argument("--role", default="local_main")
    p.add_argument("--run-group", default="")
    p.add_argument("--lanes", default="lammps_cpu,gmx_cpu,gmx_gpu")
    p.add_argument("--gmx-cpu", type=Path, default=Path("GROMACS_PCFF/build_gateb_double_cpu/bin/gmx_d"))
    p.add_argument("--gmx-gpu", type=Path, default=Path("GROMACS_PCFF/build_gateb_cuda/bin/gmx"))
    p.add_argument("--lammps-prod-dt-fs", type=float, default=2.0)
    p.add_argument("--temperature", type=float, default=353.0)
    p.add_argument("--z", type=float, default=1.0)
    p.add_argument("--max-cluster", type=int, default=10)
    p.add_argument("--cluster-cutoff-angstrom", type=float, default=3.4)
    p.add_argument("--cluster-sample-stride", type=int, default=1)
    p.add_argument("--msd-lags", type=int, default=220)
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
