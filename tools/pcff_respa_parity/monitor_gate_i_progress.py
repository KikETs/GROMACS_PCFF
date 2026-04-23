#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_I_ROOT = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_i_charged_long_npt_conditioning_commonpre3000_eq750_prod10000_ntmpi1_ntomp12_pinoff_ownerfallback_updatefastpath_nst20_cpuonly_20260423"
)
DEFAULT_GMX = REPO_ROOT / "build" / "bin" / "gmx"
PERFORMANCE_RE = re.compile(
    r"^Performance:\s+(?P<ns_per_day>[0-9.eE+-]+)\s+(?P<hour_per_ns>[0-9.eE+-]+)\s+(?P<ms_per_step>[0-9.eE+-]+)",
    flags=re.MULTILINE,
)
CONTINUATION_RE = re.compile(
    r"continuing from step\s+\d+,\s+(?P<ps>[0-9.eE+-]+)\s+ps",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RunningProcess:
    pid: int
    elapsed_s: float
    command: str


@dataclass(frozen=True)
class PhaseStatus:
    name: str
    deffnm: str
    target_ps: float
    current_ps: float
    progress_pct: float
    state: str
    running_pid: int | None
    ns_per_day: float | None
    source: str


@dataclass(frozen=True)
class GateIStatus:
    root: str
    timestamp: str
    active_phase: str | None
    active_speed_ns_per_day: float | None
    estimate_speed_ns_per_day: float | None
    remaining_ns: float
    eta: str | None
    phases: list[PhaseStatus]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report Gate I progress, current throughput, remaining wall time, and ETA from logs/checkpoints."
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_GATE_I_ROOT),
        help="Gate I artifact root. Defaults to the current local common-preconditioned 750ps+10ns run.",
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Optional gmx binary for checkpoint fallback parsing.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of the text report.")
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="Refresh interval in seconds. Default 0 prints once.",
    )
    return parser.parse_args()


def load_contract(root: Path) -> dict[str, object]:
    path = root / "gate_i_contract.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_settings(contract: dict[str, object]) -> dict[str, object]:
    settings = contract.get("run_settings")
    return settings if isinstance(settings, dict) else {}


def phase_targets(root: Path, contract: dict[str, object]) -> list[tuple[str, Path, float]]:
    settings = run_settings(contract)
    phases: list[tuple[str, Path, float]] = []
    common_precondition_ps = float(settings.get("common_precondition_ps", 0.0) or 0.0)
    if common_precondition_ps > 0:
        phases.append(("precondition/common", root / "precondition" / "common" / "precondition", common_precondition_ps))

    replica_count = int(settings.get("replicas", 0) or 0)
    if replica_count <= 0:
        replica_count = len(sorted((root / "cpu").glob("replica_*"))) if (root / "cpu").exists() else 0
    equil_ps = float(settings.get("equil_ps", 750.0) or 750.0)
    prod_ps = float(settings.get("prod_ps", 10000.0) or 10000.0)
    for replica_index in range(1, replica_count + 1):
        replica_root = root / "cpu" / f"replica_{replica_index:02d}"
        phases.append((f"replica_{replica_index:02d}/equil", replica_root / "equil", equil_ps))
        phases.append((f"replica_{replica_index:02d}/prod", replica_root / "prod", prod_ps))
    return phases


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def latest_step_time_ps(log_text: str) -> tuple[int | None, float | None]:
    step: int | None = None
    time_ps: float | None = None
    want_next_numeric = False
    for line in log_text.splitlines():
        if "Step" in line and "Time" in line:
            want_next_numeric = True
            continue
        if not want_next_numeric:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) >= 2:
            try:
                candidate_step = int(fields[0])
                candidate_ps = float(fields[1])
            except ValueError:
                want_next_numeric = False
                continue
            step = candidate_step
            time_ps = candidate_ps
        want_next_numeric = False
    return step, time_ps


def final_performance_ns_per_day(log_text: str) -> float | None:
    matches = list(PERFORMANCE_RE.finditer(log_text))
    if not matches:
        return None
    return float(matches[-1].group("ns_per_day"))


def continuation_start_ps(log_text: str) -> float:
    matches = list(CONTINUATION_RE.finditer(log_text))
    if not matches:
        return 0.0
    return float(matches[-1].group("ps"))


def checkpoint_time_ps(gmx: Path, cpt_path: Path) -> float | None:
    if not cpt_path.exists() or cpt_path.stat().st_size == 0 or not gmx.exists():
        return None
    completed = subprocess.run(
        [str(gmx), "dump", "-cp", str(cpt_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    match = re.search(r"^t = (?P<t>[0-9.eE+-]+)$", completed.stdout, flags=re.MULTILINE)
    return float(match.group("t")) if match is not None else None


def parse_running_processes() -> dict[str, RunningProcess]:
    completed = subprocess.run(
        ["pgrep", "-af", r"gmx mdrun"],
        text=True,
        capture_output=True,
        check=False,
    )
    processes: dict[str, RunningProcess] = {}
    if completed.returncode not in (0, 1):
        return processes
    for line in completed.stdout.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        command = fields[1]
        args = command.split()
        if "-deffnm" not in args:
            continue
        index = args.index("-deffnm")
        if index + 1 >= len(args):
            continue
        deffnm = str(Path(args[index + 1]).resolve())
        ps_result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etimes="],
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            elapsed_s = float(ps_result.stdout.strip())
        except ValueError:
            elapsed_s = math.nan
        processes[deffnm] = RunningProcess(pid=pid, elapsed_s=elapsed_s, command=command)
    return processes


def phase_status(
    *,
    name: str,
    deffnm: Path,
    target_ps: float,
    gmx: Path,
    running: dict[str, RunningProcess],
) -> PhaseStatus:
    log_path = deffnm.with_suffix(".log")
    log_text = read_text(log_path)
    _, log_time_ps = latest_step_time_ps(log_text)
    cpt_time = checkpoint_time_ps(gmx, deffnm.with_suffix(".cpt"))
    current_ps = max(value for value in (log_time_ps, cpt_time, 0.0) if value is not None)
    current_ps = min(current_ps, target_ps)
    deffnm_key = str(deffnm.resolve())
    running_process = running.get(deffnm_key)
    completed = (
        deffnm.with_suffix(".gro").exists()
        and deffnm.with_suffix(".cpt").exists()
        and current_ps >= target_ps - 1.0e-6
    )
    final_speed = final_performance_ns_per_day(log_text)
    source = "missing"
    state = "pending"
    speed: float | None = None
    if running_process is not None:
        state = "running"
        segment_ps = max(current_ps - continuation_start_ps(log_text), 0.0)
        if running_process.elapsed_s > 0 and segment_ps > 0:
            speed = (segment_ps / 1000.0) / (running_process.elapsed_s / 86400.0)
            source = "active-log-progress"
        elif final_speed is not None:
            speed = final_speed
            source = "completed-performance-line"
        else:
            source = "active-no-speed-yet"
    elif completed:
        state = "done"
        speed = final_speed
        source = "completed-performance-line" if final_speed is not None else "checkpoint/log-complete"
    elif current_ps > 0:
        state = "partial"
        speed = final_speed
        source = "partial-log-or-checkpoint"

    progress_pct = 100.0 * current_ps / target_ps if target_ps > 0 else 100.0
    return PhaseStatus(
        name=name,
        deffnm=str(deffnm),
        target_ps=target_ps,
        current_ps=current_ps,
        progress_pct=progress_pct,
        state=state,
        running_pid=running_process.pid if running_process is not None else None,
        ns_per_day=speed,
        source=source,
    )


def build_status(root: Path, gmx: Path) -> GateIStatus:
    contract = load_contract(root)
    running = parse_running_processes()
    phases = [
        phase_status(name=name, deffnm=deffnm, target_ps=target_ps, gmx=gmx, running=running)
        for name, deffnm, target_ps in phase_targets(root, contract)
    ]
    active = next((phase for phase in phases if phase.state == "running"), None)
    completed_speeds = [phase.ns_per_day for phase in phases if phase.state == "done" and phase.ns_per_day]
    active_speed = active.ns_per_day if active is not None else None
    estimate_speed = active_speed
    if estimate_speed is None and completed_speeds:
        estimate_speed = completed_speeds[-1]
    remaining_ns = sum(max(phase.target_ps - phase.current_ps, 0.0) for phase in phases) / 1000.0
    eta: str | None = None
    if estimate_speed is not None and estimate_speed > 0:
        remaining_days = remaining_ns / estimate_speed
        eta = (datetime.now() + timedelta(days=remaining_days)).strftime("%Y-%m-%d %H:%M:%S")
    return GateIStatus(
        root=str(root),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        active_phase=active.name if active is not None else None,
        active_speed_ns_per_day=active_speed,
        estimate_speed_ns_per_day=estimate_speed,
        remaining_ns=remaining_ns,
        eta=eta,
        phases=phases,
    )


def duration_label(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    whole = int(round(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def print_text(status: GateIStatus) -> None:
    remaining_seconds = None
    if status.estimate_speed_ns_per_day is not None and status.estimate_speed_ns_per_day > 0:
        remaining_seconds = status.remaining_ns / status.estimate_speed_ns_per_day * 86400.0
    print(f"Gate I root: {status.root}")
    print(f"timestamp: {status.timestamp}")
    print(f"active: {status.active_phase or 'none'}")
    print(f"speed: {format_float(status.active_speed_ns_per_day)} ns/day active, {format_float(status.estimate_speed_ns_per_day)} ns/day estimate")
    print(f"remaining: {status.remaining_ns:.3f} ns, {duration_label(remaining_seconds)}")
    print(f"eta: {status.eta or 'unknown'}")
    print()
    print("phase                          state       progress        speed(ns/day)  pid")
    print("-----------------------------  ----------  --------------  -------------  ----")
    for phase in status.phases:
        progress = f"{phase.current_ps:.1f}/{phase.target_ps:.1f} ps ({phase.progress_pct:.1f}%)"
        pid = str(phase.running_pid) if phase.running_pid is not None else "-"
        print(
            f"{phase.name:<29}  {phase.state:<10}  {progress:<14}  "
            f"{format_float(phase.ns_per_day):>13}  {pid}"
        )


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    gmx = Path(args.gmx).resolve()
    while True:
        status = build_status(root, gmx)
        if args.json:
            print(json.dumps(asdict(status), indent=2, sort_keys=True))
        else:
            print_text(status)
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)
        if not args.json:
            print()


if __name__ == "__main__":
    raise SystemExit(main())
