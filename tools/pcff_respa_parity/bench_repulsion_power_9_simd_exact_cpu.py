#!/usr/bin/env python3

import argparse
import json
import os
import platform
import re
import shlex
import socket
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path


DEFAULT_SYSTEMS = ("small_oligomer", "small_salt_polymer_box")
DEFAULT_NTOMP = (1, 2, 6, 12)
ENERGY_INTERVAL = 200

SIMD_ADMISSION_MARKER = "Enabling the validated CPU SIMD short-range exact LJ 9-6 non-bonded path."
PLAINC_MARKER = "Found environment variable GMX_DISABLE_SIMD_KERNELS."


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "build"
    output_dir = repo_root / "output" / "repulsion_power_9_simd_exact_cpu_perf"

    parser = argparse.ArgumentParser(
        description="Benchmark CPU exact r-RESPA PCFF repulsion-power-9 plain-C versus admitted SIMD paths."
    )
    parser.add_argument("--build-dir", type=Path, default=build_dir)
    parser.add_argument("--gmx", type=Path, default=build_dir / "bin" / "gmx")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--pin", choices=("on", "off", "auto"), default="on")
    parser.add_argument("--ntomp", type=int, nargs="+", default=list(DEFAULT_NTOMP))
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    return parser.parse_args()


def read_cpu_model() -> str:
    try:
        output = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return platform.processor() or "unknown"

    for line in output.splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def make_respa_mdp(steps: int) -> str:
    if steps <= 0 or steps % 4 != 0:
        raise ValueError("--steps must be positive and divisible by 4 for the exact r-RESPA fixture")

    return textwrap.dedent(
        f"""\
        title                   = pcff exact respa perf benchmark
        integrator              = md-vv
        dt                      = 0.0005
        nsteps                  = {steps}
        constraints             = none
        cutoff-scheme           = Verlet
        nstlist                 = 4
        rlist                   = 0.99
        rvdw                    = 0.9
        rcoulomb                = 0.9
        vdwtype                 = Cut-off
        vdw-modifier            = none
        coulombtype             = PME
        coulomb-modifier        = none
        ewald-rtol              = 1e-6
        pme-order               = 4
        fourierspacing          = 0.08
        epsilon-r               = 1
        pbc                     = xyz
        tcoupl                  = no
        pcoupl                  = no
        comm-mode               = none
        verlet-buffer-tolerance = -1
        gen-vel                 = no
        mts                     = yes
        mts-mode                = lammps-respa
        mts-levels              = 3
        mts-level2-factor       = 2
        mts-level3-factor       = 4
        mts-respa-bond-level    = 1
        mts-respa-angle-level   = 1
        mts-respa-dihedral-level = 1
        mts-respa-improper-level = 1
        mts-respa-pair14-level  = 1
        mts-respa-kspace-level  = 3
        mts-respa-inner-level   = 1
        mts-respa-middle-level  = 2
        mts-respa-outer-level   = 3
        mts-respa-inner-off     = 0.30
        mts-respa-inner-on      = 0.45
        mts-respa-outer-on      = 0.60
        mts-respa-outer-off     = 0.80
        nstcalcenergy           = {ENERGY_INTERVAL}
        nstenergy               = {ENERGY_INTERVAL}
        nstlog                  = {ENERGY_INTERVAL}
        nstxout                 = 0
        nstvout                 = 0
        nstfout                 = 0
        nstxout-compressed      = 0
        """
    )


def run_command(cmd: list[str], cwd: Path, env: dict[str, str], stdout_path: Path) -> None:
    with stdout_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(shlex.quote(x) for x in cmd)}")


def starts_with_wallcycle_label(line: str, label: str) -> bool:
    trimmed = line.strip()
    return trimmed.startswith(label) and (len(trimmed) == len(label) or trimmed[len(label)].isspace())


def extract_wallcycle_seconds(log_contents: str, label: str) -> float | None:
    for line in log_contents.splitlines():
        if not starts_with_wallcycle_label(line, label):
            continue
        tokens = line.split()
        if len(tokens) < 4:
            continue
        try:
            return float(tokens[-3])
        except ValueError:
            continue
    return None


def extract_ns_per_day(log_contents: str) -> float | None:
    match = re.search(r"^Performance:\s+([0-9.eE+-]+)", log_contents, flags=re.MULTILINE)
    if match is None:
        return None
    return float(match.group(1))


def load_log(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def benchmark_one_run(
    gmx: Path,
    run_dir: Path,
    tpr_path: Path,
    ntomp: int,
    pin_mode: str,
    disable_simd: bool,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    deffnm = run_dir / "run"
    stdout_path = run_dir / "mdrun.stdout.txt"
    log_path = run_dir / "run.log"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(ntomp)
    if disable_simd:
        env["GMX_DISABLE_SIMD_KERNELS"] = "1"
    else:
        env.pop("GMX_DISABLE_SIMD_KERNELS", None)

    cmd = [
        str(gmx),
        "mdrun",
        "-s",
        str(tpr_path),
        "-deffnm",
        str(deffnm),
        "-pin",
        pin_mode,
        "-nb",
        "cpu",
        "-pme",
        "cpu",
        "-bonded",
        "cpu",
        "-update",
        "cpu",
        "-notunepme",
        "-ntmpi",
        "1",
        "-ntomp",
        str(ntomp),
    ]
    run_command(cmd, cwd=run_dir, env=env, stdout_path=stdout_path)

    log_contents = load_log(log_path)
    ns_per_day = extract_ns_per_day(log_contents)
    force_seconds = extract_wallcycle_seconds(log_contents, "Force")
    pme_seconds = extract_wallcycle_seconds(log_contents, "PME mesh")
    if ns_per_day is None or force_seconds is None:
        raise RuntimeError(f"Could not parse required performance counters from {log_path}")

    expected_marker = PLAINC_MARKER if disable_simd else SIMD_ADMISSION_MARKER
    if expected_marker not in log_contents:
        raise RuntimeError(f"Expected log marker not found in {log_path}: {expected_marker}")

    return {
        "mode": "plainc" if disable_simd else "simd",
        "ntomp": ntomp,
        "ns_per_day": ns_per_day,
        "force_seconds": force_seconds,
        "pme_mesh_seconds": pme_seconds,
        "log_path": str(log_path),
        "stdout_path": str(stdout_path),
    }


def summarize_runs(runs: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, int], dict[str, list[dict]]] = {}
    for run in runs:
        key = (run["system"], run["ntomp"])
        buckets.setdefault(key, {}).setdefault(run["mode"], []).append(run)

    summary = []
    for (system, ntomp), by_mode in sorted(buckets.items()):
        if "plainc" not in by_mode or "simd" not in by_mode:
            continue
        plain_runs = by_mode["plainc"]
        simd_runs = by_mode["simd"]
        plain_ns = statistics.median(item["ns_per_day"] for item in plain_runs)
        simd_ns = statistics.median(item["ns_per_day"] for item in simd_runs)
        plain_force = statistics.median(item["force_seconds"] for item in plain_runs)
        simd_force = statistics.median(item["force_seconds"] for item in simd_runs)
        summary.append(
            {
                "system": system,
                "ntomp": ntomp,
                "plainc_ns_per_day_median": plain_ns,
                "simd_ns_per_day_median": simd_ns,
                "wall_speedup_simd_vs_plainc": simd_ns / plain_ns,
                "plainc_force_seconds_median": plain_force,
                "simd_force_seconds_median": simd_force,
                "force_speedup_simd_vs_plainc": plain_force / simd_force,
            }
        )

    baseline = {}
    for item in summary:
        if item["ntomp"] == 1:
            baseline[item["system"]] = item

    for item in summary:
        system_baseline = baseline.get(item["system"])
        if system_baseline is None:
            item["plainc_scaling_vs_ntomp1"] = None
            item["simd_scaling_vs_ntomp1"] = None
            continue
        item["plainc_scaling_vs_ntomp1"] = (
            item["plainc_ns_per_day_median"] / system_baseline["plainc_ns_per_day_median"]
        )
        item["simd_scaling_vs_ntomp1"] = (
            item["simd_ns_per_day_median"] / system_baseline["simd_ns_per_day_median"]
        )
    return summary


def write_markdown(summary_path: Path, metadata: dict, summary_rows: list[dict]) -> None:
    lines = [
        "# Repulsion-Power-9 CPU Exact SIMD Benchmark",
        "",
        "This file is host-local. It is not a cross-machine claim.",
        "",
        "## Host",
        "",
        f"- hostname: `{metadata['hostname']}`",
        f"- cpu: `{metadata['cpu_model']}`",
        f"- gmx: `{metadata['gmx']}`",
        f"- steps per run: `{metadata['steps']}`",
        f"- repeats per point: `{metadata['repeats']}`",
        f"- pin mode: `{metadata['pin']}`",
        "",
        "## Measurement Notes",
        "",
        "- `ns/day` is the wall-clock campaign metric from the mdrun performance line.",
        "- `Force s` is the `REAL CYCLE AND TIME ACCOUNTING` wallcycle entry for `Force`.",
        "- `Force s` is kernel-adjacent, not isolated nonbonded-only timing. PME and bonded work remain in the same exact r-RESPA campaign.",
        "",
    ]

    systems = sorted({row["system"] for row in summary_rows})
    for system in systems:
        lines.extend(
            [
                f"## {system}",
                "",
                "| ntomp | plain-C ns/day | SIMD ns/day | wall speedup | plain-C Force s | SIMD Force s | force speedup | plain-C scaling | SIMD scaling |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in [item for item in summary_rows if item["system"] == system]:
            lines.append(
                "| {ntomp} | {plain_ns:.3f} | {simd_ns:.3f} | {wall_speedup:.3f} | {plain_force:.6f} | {simd_force:.6f} | {force_speedup:.3f} | {plain_scaling:.3f} | {simd_scaling:.3f} |".format(
                    ntomp=row["ntomp"],
                    plain_ns=row["plainc_ns_per_day_median"],
                    simd_ns=row["simd_ns_per_day_median"],
                    wall_speedup=row["wall_speedup_simd_vs_plainc"],
                    plain_force=row["plainc_force_seconds_median"],
                    simd_force=row["simd_force_seconds_median"],
                    force_speedup=row["force_speedup_simd_vs_plainc"],
                    plain_scaling=row["plainc_scaling_vs_ntomp1"],
                    simd_scaling=row["simd_scaling_vs_ntomp1"],
                )
            )
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture_root = args.build_dir.parent / "tests" / "reference_results" / "m6_respa"
    mdp_contents = make_respa_mdp(args.steps)

    runs = []
    for system in args.systems:
        system_root = fixture_root / system
        if not system_root.exists():
            raise SystemExit(f"Unknown system fixture: {system_root}")

        system_dir = args.output_dir / system
        system_dir.mkdir(parents=True, exist_ok=True)
        mdp_path = system_dir / "exact_respa_perf.mdp"
        tpr_path = system_dir / "exact_respa_perf.tpr"
        grompp_stdout = system_dir / "grompp.stdout.txt"
        mdp_path.write_text(mdp_contents, encoding="utf-8")

        run_command(
            [
                str(args.gmx),
                "grompp",
                "-f",
                str(mdp_path),
                "-p",
                str(system_root / "topol.top"),
                "-c",
                str(system_root / "initial_nve.gro"),
                "-o",
                str(tpr_path),
                "-maxwarn",
                "1",
            ],
            cwd=system_dir,
            env=os.environ.copy(),
            stdout_path=grompp_stdout,
        )

        for ntomp in args.ntomp:
            for repeat in range(args.repeats):
                for disable_simd in (False, True):
                    mode_label = "plainc" if disable_simd else "simd"
                    run_dir = system_dir / f"ntomp{ntomp}" / mode_label / f"repeat{repeat + 1}"
                    row = benchmark_one_run(
                        gmx=args.gmx,
                        run_dir=run_dir,
                        tpr_path=tpr_path,
                        ntomp=ntomp,
                        pin_mode=args.pin,
                        disable_simd=disable_simd,
                    )
                    row["system"] = system
                    row["repeat"] = repeat + 1
                    runs.append(row)

    summary_rows = summarize_runs(runs)
    metadata = {
        "hostname": socket.gethostname(),
        "cpu_model": read_cpu_model(),
        "gmx": str(args.gmx),
        "build_dir": str(args.build_dir),
        "output_dir": str(args.output_dir),
        "steps": args.steps,
        "repeats": args.repeats,
        "pin": args.pin,
        "systems": args.systems,
        "ntomp": args.ntomp,
    }
    summary = {
        "metadata": metadata,
        "runs": runs,
        "summary": summary_rows,
    }

    summary_json = args.output_dir / "summary.json"
    summary_md = args.output_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary_md, metadata, summary_rows)

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
