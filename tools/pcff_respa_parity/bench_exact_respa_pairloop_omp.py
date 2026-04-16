#!/usr/bin/env python3

import argparse
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PAIRLOOP_OMP_ENV = "GMX_PCFF_EXACT_RESPA_PAIRLOOP_OMP"
PAIRLOOP_VECTOR_ENV = "GMX_PCFF_EXACT_RESPA_PAIRLOOP_VECTOR"

WALLCYCLE_LABELS = {
    "Neighbor search": "neighbor_search_seconds",
    "Force": "force_seconds",
    "PME mesh": "pme_mesh_seconds",
    "Update": "update_seconds",
    "Total": "total_wallcycle_seconds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark staged exact-r-RESPA CPU pair-loop OpenMP/vector modes on a fixed TPR."
    )
    parser.add_argument("--gmx", type=Path, default=REPO_ROOT / "build" / "bin" / "gmx")
    parser.add_argument("--tpr", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--ntomp", type=int, nargs="+", default=[1, 2, 6, 12])
    parser.add_argument("--pin", choices=("off", "on", "auto"), default="off")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("baseline", "pairloop_omp", "pairloop_vector", "combined"),
        default=["baseline"],
    )
    parser.add_argument("--fixture-id", default="unspecified")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path, env: dict[str, str], stdout_path: Path) -> None:
    with stdout_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        quoted = " ".join(shlex.quote(part) for part in cmd)
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {quoted}")


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def cpu_model() -> str:
    try:
        output = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return platform.processor() or "unknown"
    for line in output.splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def starts_with_label(line: str, label: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(label) and (len(stripped) == len(label) or stripped[len(label)].isspace())


def extract_seconds(log_text: str, label: str) -> float | None:
    for line in log_text.splitlines():
        if not starts_with_label(line, label):
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            return float(fields[-3])
        except ValueError:
            continue
    return None


def extract_performance(log_text: str) -> tuple[float | None, float | None]:
    match = re.search(r"^Performance:\s+([0-9.eE+-]+)\s+[0-9.eE+-]+\s+([0-9.eE+-]+)", log_text, re.MULTILINE)
    if match is None:
        return None, None
    return float(match.group(1)), float(match.group(2))


def mode_env(mode: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop(PAIRLOOP_OMP_ENV, None)
    env.pop(PAIRLOOP_VECTOR_ENV, None)
    if mode in ("pairloop_omp", "combined"):
        env[PAIRLOOP_OMP_ENV] = "1"
    if mode in ("pairloop_vector", "combined"):
        env[PAIRLOOP_VECTOR_ENV] = "1"
    return env


def benchmark_one(args: argparse.Namespace, mode: str, ntomp: int, repeat_index: int) -> dict:
    run_dir = args.output_dir / mode / f"ntomp_{ntomp}" / f"repeat_{repeat_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    deffnm = run_dir / "run"
    stdout_path = run_dir / "mdrun.stdout.txt"
    env = mode_env(mode)
    env["OMP_NUM_THREADS"] = str(ntomp)
    cmd = [
        str(args.gmx),
        "mdrun",
        "-s",
        str(args.tpr),
        "-deffnm",
        str(deffnm),
        "-nsteps",
        str(args.steps),
        "-ntmpi",
        "1",
        "-ntomp",
        str(ntomp),
        "-dlb",
        "no",
        "-nb",
        "cpu",
        "-pme",
        "cpu",
        "-bonded",
        "cpu",
        "-update",
        "cpu",
        "-pin",
        args.pin,
        "-reprod",
    ]
    started = time.time()
    run(cmd, cwd=REPO_ROOT, env=env, stdout_path=stdout_path)
    elapsed = time.time() - started
    log_path = run_dir / "run.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    ns_per_day, ms_per_step = extract_performance(log_text)
    row = {
        "fixture_id": args.fixture_id,
        "mode": mode,
        "ntmpi": 1,
        "ntomp": ntomp,
        "pin": args.pin,
        "steps": args.steps,
        "repeat_index": repeat_index,
        "elapsed_seconds": elapsed,
        "ns_per_day": ns_per_day,
        "ms_per_step": ms_per_step,
        "tpr": str(args.tpr),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "stdout_path": str(stdout_path),
        "pairloop_omp_env": env.get(PAIRLOOP_OMP_ENV, "0"),
        "pairloop_vector_env": env.get(PAIRLOOP_VECTOR_ENV, "0"),
        "command": cmd,
    }
    for label, key in WALLCYCLE_LABELS.items():
        row[key] = extract_seconds(log_text, label)
    return row


def main() -> None:
    args = parse_args()
    if not args.tpr.exists():
        raise SystemExit(f"TPR not found: {args.tpr}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_name": "exact_respa_pairloop_omp_benchmark",
        "schema_version": 1,
        "created_unix_time": time.time(),
        "host": socket.gethostname(),
        "cpu_model": cpu_model(),
        "platform": platform.platform(),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_branch": git_text(["branch", "--show-current"]),
        "git_status_short": git_text(["status", "--short"]),
        "gmx": str(args.gmx),
        "tpr": str(args.tpr),
        "fixture_id": args.fixture_id,
        "steps": args.steps,
        "repeats": args.repeats,
        "ntomp": args.ntomp,
        "pin": args.pin,
        "modes": args.modes,
    }

    rows = []
    for mode in args.modes:
        for ntomp in args.ntomp:
            for repeat_index in range(1, args.repeats + 1):
                rows.append(benchmark_one(args, mode, ntomp, repeat_index))

    output = {"metadata": metadata, "runs": rows}
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    columns = [
        "fixture_id",
        "mode",
        "ntomp",
        "repeat_index",
        "ns_per_day",
        "ms_per_step",
        "force_seconds",
        "update_seconds",
        "total_wallcycle_seconds",
        "log_path",
    ]
    with (args.output_dir / "summary.tsv").open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")


if __name__ == "__main__":
    main()
