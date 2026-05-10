#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = REPO / "output/polygen_pcff_gromacs_initial_em_notebook"

GMX_PERF_RE = re.compile(r"Performance:\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")
GMX_TIME_RE = re.compile(r"Time:\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")
LAMMPS_LOOP_RE = re.compile(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs for\s+(\d+)\s+steps")
LAMMPS_PERF_RE = re.compile(r"Performance:\s+([0-9.eE+-]+)\s+ns/day,\s+([0-9.eE+-]+)\s+hours/ns,\s+([0-9.eE+-]+)\s+timesteps/s,\s+([0-9.eE+-]+)\s+Matom-step/s")


def parse_timestamp(text: str | None) -> datetime | None:
    if not text:
        return None
    # Runtime markers use strings like "2026-05-07 13:26:41 KST".
    value = text.rsplit(" ", 1)[0]
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def runtime_elapsed_s(lane_dir: Path, stem: str) -> float | None:
    state_dir = lane_dir / ".resume_state"
    candidates = sorted(state_dir.glob(f"*_{stem}.runtime.json"))
    done_candidates = sorted(state_dir.glob(f"*_{stem}.done.json"))
    if not candidates or not done_candidates:
        return None
    runtime = read_json(candidates[-1])
    done = read_json(done_candidates[-1])
    start = parse_timestamp(str(runtime.get("timestamp") or ""))
    stop = parse_timestamp(str(done.get("timestamp") or ""))
    if start is None or stop is None:
        return None
    return max(0.0, (stop - start).total_seconds())


def collect_gromacs_lane(lane_dir: Path, lane: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stdout_log in sorted(lane_dir.glob("*.mdrun.stdout.log")):
        stem = stdout_log.name.removesuffix(".mdrun.stdout.log")
        text = stdout_log.read_text(encoding="utf-8", errors="replace")
        perf = GMX_PERF_RE.search(text)
        time_match = GMX_TIME_RE.search(text)
        elapsed = runtime_elapsed_s(lane_dir, stem)
        row = {
            "engine": "gromacs",
            "lane": lane,
            "stage": stem,
            "source": str(stdout_log.relative_to(REPO)),
            "ns_day": "",
            "wall_s": elapsed if elapsed is not None else "",
            "steps": "",
            "ms_per_step": "",
            "hours_per_ns": "",
            "matom_step_s": "",
            "note": "",
        }
        if perf:
            row["ns_day"] = float(perf.group(1))
            row["hours_per_ns"] = float(perf.group(2))
            row["ms_per_step"] = float(perf.group(3))
            row["matom_step_s"] = float(perf.group(4))
        else:
            row["note"] = "no_ns_day_in_log"
        if time_match and row["wall_s"] == "":
            row["wall_s"] = float(time_match.group(2))
        rows.append(row)
    return rows


def collect_lammps_log(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending_loop: tuple[float, int, int] | None = None
    stage_index = 0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        loop = LAMMPS_LOOP_RE.search(line)
        if loop:
            pending_loop = (float(loop.group(1)), int(loop.group(2)), int(loop.group(3)))
            continue
        perf = LAMMPS_PERF_RE.search(line)
        if not perf or pending_loop is None:
            continue
        stage_index += 1
        loop_s, nprocs, steps = pending_loop
        rows.append(
            {
                "engine": "lammps",
                "lane": "lammps_openmp",
                "stage": f"{log_path.stem}_run{stage_index:04d}",
                "source": str(log_path.relative_to(REPO)),
                "ns_day": float(perf.group(1)),
                "wall_s": loop_s,
                "steps": steps,
                "ms_per_step": loop_s * 1000.0 / steps if steps else "",
                "hours_per_ns": float(perf.group(2)),
                "matom_step_s": float(perf.group(4)),
                "note": f"{nprocs} procs; sequential LAMMPS run block",
            }
        )
        pending_loop = None
    return rows


def collect(out_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in ("gromacs_cpu_openmp", "gromacs_gpu_hybrid"):
        lane_dir = out_root / lane
        if lane_dir.exists():
            rows.extend(collect_gromacs_lane(lane_dir, lane))
    lammps_dir = out_root / "lammps_openmp"
    for log_name in ("equil_from_em.stdout.log", "prod_from_relaxed.stdout.log"):
        log_path = lammps_dir / log_name
        if log_path.exists():
            rows.extend(collect_lammps_log(log_path))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect stage speed records from GROMACS/LAMMPS logs.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--output", default=None, help="Optional TSV output path. Defaults to stdout.")
    args = parser.parse_args()

    rows = collect(Path(args.out_root).resolve())
    fieldnames = [
        "engine",
        "lane",
        "stage",
        "ns_day",
        "wall_s",
        "steps",
        "ms_per_step",
        "hours_per_ns",
        "matom_step_s",
        "source",
        "note",
    ]
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        handle = out.open("w", encoding="utf-8", newline="")
        close = True
    else:
        import sys

        handle = sys.stdout
        close = False
    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if close:
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
