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
FORCE_DUMP_DIR_ENV = "GMX_PCFF_EXACT_RESPA_PAIRLOOP_FORCE_DUMP_DIR"
FORCE_DUMP_LABEL_ENV = "GMX_PCFF_EXACT_RESPA_PAIRLOOP_FORCE_DUMP_LABEL"
FORCE_DUMP_MAX_ENV = "GMX_PCFF_EXACT_RESPA_PAIRLOOP_FORCE_DUMP_MAX"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate exact-r-RESPA pair-loop force-delta parity without disabling the "
            "OpenMP/vector fast path."
        )
    )
    parser.add_argument("--gmx", type=Path, default=REPO_ROOT / "build" / "bin" / "gmx")
    parser.add_argument("--tpr", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dump-max", type=int, default=32)
    parser.add_argument("--ntomp", type=int, default=6)
    parser.add_argument("--pin", choices=("off", "on", "auto"), default="off")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("pairloop_omp", "pairloop_vector", "combined"),
        default=["pairloop_omp", "pairloop_vector", "combined"],
    )
    parser.add_argument("--fixture-id", default="unspecified")
    parser.add_argument("--abs-tol", type=float, default=1.0e-2)
    parser.add_argument("--rel-tol", type=float, default=5.0e-5)
    return parser.parse_args()


def run(cmd: list[str], cwd: Path, env: dict[str, str], stdout_path: Path) -> None:
    with stdout_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        quoted = " ".join(shlex.quote(part) for part in cmd)
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {quoted}")


def capture_output(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


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


def mode_env(mode: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in (PAIRLOOP_OMP_ENV, PAIRLOOP_VECTOR_ENV, FORCE_DUMP_DIR_ENV, FORCE_DUMP_LABEL_ENV, FORCE_DUMP_MAX_ENV):
        env.pop(name, None)
    if mode in ("pairloop_omp", "combined"):
        env[PAIRLOOP_OMP_ENV] = "1"
    if mode in ("pairloop_vector", "combined"):
        env[PAIRLOOP_VECTOR_ENV] = "1"
    return env


def run_mode(args: argparse.Namespace, mode: str) -> dict:
    run_dir = args.output_dir / mode
    dump_dir = run_dir / "force_dumps"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = mode_env(mode)
    env["OMP_NUM_THREADS"] = str(args.ntomp)
    env[FORCE_DUMP_DIR_ENV] = str(dump_dir)
    env[FORCE_DUMP_LABEL_ENV] = mode
    env[FORCE_DUMP_MAX_ENV] = str(args.dump_max)
    deffnm = run_dir / "run"
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
        str(args.ntomp),
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
    run(cmd, cwd=REPO_ROOT, env=env, stdout_path=run_dir / "mdrun.stdout.txt")
    elapsed = time.time() - started
    return {
        "mode": mode,
        "run_dir": str(run_dir),
        "dump_dir": str(dump_dir),
        "elapsed_seconds": elapsed,
        "command": cmd,
        "pairloop_omp_env": env.get(PAIRLOOP_OMP_ENV, "0"),
        "pairloop_vector_env": env.get(PAIRLOOP_VECTOR_ENV, "0"),
    }


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Unexpected boolean value: {value}")


def parse_dump(path: Path) -> dict:
    metadata: dict[str, str] = {}
    rows: dict[tuple[int, int], tuple[float, float, float, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("# "):
                key_value = line[2:].split(maxsplit=1)
                if len(key_value) == 2:
                    metadata[key_value[0]] = key_value[1]
                continue
            if line.startswith("contribution_index\t"):
                continue
            fields = line.split("\t")
            if len(fields) != 6:
                raise ValueError(f"Malformed dump row in {path}: {line}")
            contribution_index = int(fields[0])
            contribution = fields[1]
            atom = int(fields[2])
            rows[(contribution_index, atom)] = (
                float(fields[3]),
                float(fields[4]),
                float(fields[5]),
                contribution,
            )
    return {"path": str(path), "metadata": metadata, "rows": rows}


def dumps_by_ordinal(dump_dir: Path) -> dict[int, dict]:
    result = {}
    for path in sorted(dump_dir.glob("pairloop_force_delta_*.tsv")):
        parsed = parse_dump(path)
        ordinal = int(parsed["metadata"]["ordinal"])
        result[ordinal] = parsed
    return result


def compare_dump_pair(baseline: dict, candidate: dict, abs_tol: float, rel_tol: float) -> dict:
    baseline_rows = baseline["rows"]
    candidate_rows = candidate["rows"]
    if baseline_rows.keys() != candidate_rows.keys():
        missing_in_candidate = sorted(baseline_rows.keys() - candidate_rows.keys())[:10]
        missing_in_baseline = sorted(candidate_rows.keys() - baseline_rows.keys())[:10]
        raise ValueError(
            "Force dump row keys differ: "
            f"missing_in_candidate={missing_in_candidate} missing_in_baseline={missing_in_baseline}"
        )

    max_abs_delta = 0.0
    max_rel_delta_at_max_abs = 0.0
    max_abs_record = None
    max_rel_delta = 0.0
    max_rel_record = None
    failures = 0
    failure_examples = []
    component_names = ("fx", "fy", "fz")
    for key, baseline_values in baseline_rows.items():
        candidate_values = candidate_rows[key]
        for component_index, component_name in enumerate(component_names):
            b_value = baseline_values[component_index]
            c_value = candidate_values[component_index]
            abs_delta = abs(c_value - b_value)
            denom = max(abs(b_value), abs(c_value), 1.0e-30)
            rel_delta = abs_delta / denom
            if abs_delta > max_abs_delta:
                max_abs_delta = abs_delta
                max_rel_delta_at_max_abs = rel_delta
                max_abs_record = {
                    "contribution_index": key[0],
                    "contribution": baseline_values[3],
                    "atom": key[1],
                    "component": component_name,
                    "baseline": b_value,
                    "candidate": c_value,
                    "abs_delta": abs_delta,
                    "rel_delta": rel_delta,
                }
            if rel_delta > max_rel_delta:
                max_rel_delta = rel_delta
                max_rel_record = {
                    "contribution_index": key[0],
                    "contribution": baseline_values[3],
                    "atom": key[1],
                    "component": component_name,
                    "baseline": b_value,
                    "candidate": c_value,
                    "abs_delta": abs_delta,
                    "rel_delta": rel_delta,
                }
            if abs_delta > abs_tol and rel_delta > rel_tol:
                failures += 1
                if len(failure_examples) < 10:
                    failure_examples.append(
                        {
                            "contribution_index": key[0],
                            "contribution": baseline_values[3],
                            "atom": key[1],
                            "component": component_name,
                            "baseline": b_value,
                            "candidate": c_value,
                            "abs_delta": abs_delta,
                            "rel_delta": rel_delta,
                        }
                    )

    return {
        "baseline_path": baseline["path"],
        "candidate_path": candidate["path"],
        "baseline_step": int(baseline["metadata"]["step"]),
        "candidate_step": int(candidate["metadata"]["step"]),
        "candidate_pair_fast_path_used": parse_bool(candidate["metadata"]["pair_fast_path_used"]),
        "candidate_excluded_pair_fast_path_used": parse_bool(
            candidate["metadata"]["excluded_pair_fast_path_used"]
        ),
        "row_count": len(baseline_rows),
        "component_count": len(baseline_rows) * 3,
        "max_abs_delta": max_abs_delta,
        "max_rel_delta_at_max_abs": max_rel_delta_at_max_abs,
        "max_abs_record": max_abs_record,
        "max_rel_delta": max_rel_delta,
        "max_rel_record": max_rel_record,
        "failure_count": failures,
        "failure_examples": failure_examples,
        "passed": failures == 0,
    }


def compare_mode(args: argparse.Namespace, baseline_run: dict, candidate_run: dict) -> dict:
    baseline_dumps = dumps_by_ordinal(Path(baseline_run["dump_dir"]))
    candidate_dumps = dumps_by_ordinal(Path(candidate_run["dump_dir"]))
    common_ordinals = sorted(set(baseline_dumps) & set(candidate_dumps))
    compared = []
    skipped = []
    for ordinal in common_ordinals:
        candidate = candidate_dumps[ordinal]
        pair_fast = parse_bool(candidate["metadata"]["pair_fast_path_used"])
        excluded_fast = parse_bool(candidate["metadata"]["excluded_pair_fast_path_used"])
        if not (pair_fast or excluded_fast):
            skipped.append({"ordinal": ordinal, "reason": "candidate_fast_path_not_used"})
            continue
        compared.append(compare_dump_pair(baseline_dumps[ordinal], candidate, args.abs_tol, args.rel_tol))

    passed = bool(compared) and all(row["passed"] for row in compared)
    return {
        "mode": candidate_run["mode"],
        "baseline_dump_count": len(baseline_dumps),
        "candidate_dump_count": len(candidate_dumps),
        "common_ordinal_count": len(common_ordinals),
        "compared_fast_path_snapshot_count": len(compared),
        "skipped_snapshot_count": len(skipped),
        "skipped_snapshots": skipped,
        "comparisons": compared,
        "passed": passed,
        "failure_reason": None if passed else "no_fast_path_snapshot_or_tolerance_failure",
    }


def extract_performance(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^Performance:\s+([0-9.eE+-]+)\s+[0-9.eE+-]+\s+([0-9.eE+-]+)", text, re.MULTILINE)
    if match is None:
        return {}
    return {"ns_per_day": float(match.group(1)), "ms_per_step": float(match.group(2))}


def main() -> None:
    args = parse_args()
    if not args.tpr.exists():
        raise SystemExit(f"TPR not found: {args.tpr}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_name": "exact_respa_pairloop_force_delta_parity",
        "schema_version": 1,
        "created_unix_time": time.time(),
        "host": socket.gethostname(),
        "cpu_model": cpu_model(),
        "platform": platform.platform(),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_branch": git_text(["branch", "--show-current"]),
        "git_status_short": git_text(["status", "--short"]),
        "gmx": str(args.gmx),
        "gmx_version": capture_output([str(args.gmx), "--version"], REPO_ROOT),
        "tpr": str(args.tpr),
        "fixture_id": args.fixture_id,
        "steps": args.steps,
        "dump_max": args.dump_max,
        "ntmpi": 1,
        "ntomp": args.ntomp,
        "pin": args.pin,
        "abs_tol": args.abs_tol,
        "rel_tol": args.rel_tol,
        "candidate_modes": args.modes,
    }

    baseline_run = run_mode(args, "baseline")
    baseline_run.update(extract_performance(Path(baseline_run["run_dir"]) / "run.log"))
    candidate_runs = []
    comparisons = []
    for mode in args.modes:
        candidate_run = run_mode(args, mode)
        candidate_run.update(extract_performance(Path(candidate_run["run_dir"]) / "run.log"))
        candidate_runs.append(candidate_run)
        comparisons.append(compare_mode(args, baseline_run, candidate_run))

    output = {
        "metadata": metadata,
        "baseline_run": baseline_run,
        "candidate_runs": candidate_runs,
        "comparisons": comparisons,
        "passed": all(comparison["passed"] for comparison in comparisons),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not output["passed"]:
        raise SystemExit("Force-delta parity failed; see summary.json")


if __name__ == "__main__":
    main()
