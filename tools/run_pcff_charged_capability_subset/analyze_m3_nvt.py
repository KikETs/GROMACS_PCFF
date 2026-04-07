#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an M3 NVT stability bundle.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--duration-ps", type=float, required=True)
    parser.add_argument("--analysis-window-ps", type=float, required=True)
    parser.add_argument("--mean-temp-target-k", type=float, default=300.0)
    parser.add_argument("--mean-temp-tolerance-k", type=float, default=20.0)
    parser.add_argument("--max-temp-k", type=float, default=400.0)
    parser.add_argument("--out", type=Path, help="Output report path. Defaults to <bundle>/m3_recovery_report.json.")
    return parser.parse_args()


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


def parse_xvg(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
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


def main() -> int:
    args = parse_args()
    bundle = args.bundle.resolve()
    out_path = (args.out if args.out is not None else bundle / "m3_recovery_report.json").resolve()
    energy_xvg = bundle / "nvt_energy_50ps.xvg"

    run_command(
        [str(REPO_ROOT / "build" / "bin" / "gmx"), "energy", "-f", "nvt.edr", "-o", energy_xvg.name],
        bundle,
        bundle / "energy_50ps.stdout",
        bundle / "energy_50ps.stderr",
        stdin_text="Potential\nTemperature\n0\n",
    )

    rows = parse_xvg(energy_xvg)
    analysis_start_ps = max(0.0, args.duration_ps - args.analysis_window_ps)
    window = [row for row in rows if row[0] >= analysis_start_ps]
    potential = [row[1] for row in window]
    temperature = [row[2] for row in window]
    if not temperature:
        raise RuntimeError("No temperature samples in requested M3 analysis window.")

    mean_temp = statistics.fmean(temperature)
    max_temp = max(temperature)
    required = {
        "protocol": bundle / "m3_recovery_protocol.json",
        "log": bundle / "nvt.log",
        "energy": bundle / "nvt.edr",
        "checkpoint": bundle / "nvt.cpt",
        "coordinates": bundle / "nvt.gro",
        "mdp": bundle / "nvt.mdp",
        "run_input": bundle / "nvt_extend50.tpr",
        "energy_xvg": energy_xvg,
        "convert_tpr_stdout": bundle / "convert_tpr.stdout",
        "convert_tpr_stderr": bundle / "convert_tpr.stderr",
        "mdrun_extend_stderr": bundle / "mdrun_extend.stderr",
        "energy_stdout": bundle / "energy_50ps.stdout",
        "energy_stderr": bundle / "energy_50ps.stderr",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    stability_pass = (
        not missing
        and abs(mean_temp - args.mean_temp_target_k) <= args.mean_temp_tolerance_k
        and max_temp <= args.max_temp_k
    )

    report = {
        "milestone": "M3",
        "system_id": args.system_id,
        "claim_scope": "strict-PCFF-qualified charged 2x2x2 subset NVT stability only",
        "protocol": {
            "ensemble": "NVT",
            "duration_ps": args.duration_ps,
            "analysis_window_ps": args.analysis_window_ps,
            "analysis_start_ps": analysis_start_ps,
            "thresholds": {
                "mean_temperature_target_k": args.mean_temp_target_k,
                "mean_temperature_tolerance_k": args.mean_temp_tolerance_k,
                "max_temperature_k": args.max_temp_k,
            },
        },
        "temperature_k": summarize(temperature),
        "potential_energy_kj_mol": summarize(potential),
        "artifact_checks": {
            "analysis_script": str(Path(__file__).resolve()),
            "required_artifacts": {name: str(path) for name, path in required.items()},
            "missing": missing,
        },
        "status": "PASS" if stability_pass else "FAIL",
        "known_limitations": [
            "This is not an ns-scale replacement for the historical TP1 dense_salt_polymer failure.",
            "This does not establish charged transport readiness.",
        ],
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
