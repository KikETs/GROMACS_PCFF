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
from collections import OrderedDict
from pathlib import Path


DEFAULT_SYSTEMS = ("gate_h_dense_salt_polymer_2x2x2",)
DEFAULT_NTOMP = (1, 2, 6)
DEFAULT_MODES = ("generic", "specialized")
SPECIALIZED_LABEL = "specialized"
GENERIC_LABEL = "generic"
PLAINC_LABEL = "plainc"

SIMD_ADMISSION_MARKER = "Enabling the validated CPU SIMD short-range exact LJ 9-6 non-bonded path."
SPECIALIZED_MARKER = "Selecting the specialized exact CPU SIMD repulsion-power-9 microkernel path."
GENERIC_ENV_MARKER = "Found environment variable GMX_DISABLE_REPULSION_POWER_9_SIMD_SPECIALIZATION."
GENERIC_MARKER = "Keeping the admitted generic CPU SIMD repulsion-power-9 path for baseline comparison."
PLAINC_MARKER = "Found environment variable GMX_DISABLE_SIMD_KERNELS."
SPECIALIZATION_DISABLE_ENV = "GMX_DISABLE_REPULSION_POWER_9_SIMD_SPECIALIZATION"

MODE_SPECS = {
    PLAINC_LABEL: {
        "env_set": {"GMX_DISABLE_SIMD_KERNELS": "1"},
        "env_unset": [SPECIALIZATION_DISABLE_ENV],
        "required_markers": [PLAINC_MARKER],
    },
    GENERIC_LABEL: {
        "env_set": {SPECIALIZATION_DISABLE_ENV: "1"},
        "env_unset": ["GMX_DISABLE_SIMD_KERNELS"],
        "required_markers": [SIMD_ADMISSION_MARKER, GENERIC_ENV_MARKER, GENERIC_MARKER],
    },
    SPECIALIZED_LABEL: {
        "env_set": {},
        "env_unset": ["GMX_DISABLE_SIMD_KERNELS", SPECIALIZATION_DISABLE_ENV],
        "required_markers": [SIMD_ADMISSION_MARKER, SPECIALIZED_MARKER],
    },
}

WALLCYCLE_LABELS = OrderedDict(
    [
        ("Neighbor search", "neighbor_search_seconds"),
        ("Force", "force_seconds"),
        ("PME mesh", "pme_mesh_seconds"),
        ("Update", "update_seconds"),
        ("Total", "total_wallcycle_seconds"),
    ]
)

SUBCOUNTER_LABELS = OrderedDict(
    [
        ("NB F kernel", "nb_f_kernel_seconds"),
        ("NB F buffer ops.", "nb_f_buffer_ops_seconds"),
        ("NB X buffer ops.", "nb_x_buffer_ops_seconds"),
        ("Listed buffer ops.", "listed_buffer_ops_seconds"),
        ("Bonded F", "bonded_force_seconds"),
        ("Clear force buffer", "clear_force_buffer_seconds"),
    ]
)

SYSTEM_LAYOUTS = {
    "small_oligomer": ("tests/reference_results/m6_respa/small_oligomer/topol.top", "tests/reference_results/m6_respa/small_oligomer/initial_nve.gro"),
    "small_salt_polymer_box": (
        "tests/reference_results/m6_respa/small_salt_polymer_box/topol.top",
        "tests/reference_results/m6_respa/small_salt_polymer_box/initial_nve.gro",
    ),
    "gate_h_dense_salt_polymer_2x2x2": (
        "tests/reference_results/gate_h_fixture_scaffold/gate_h_dense_salt_polymer_2x2x2/generated/topol.top",
        "tests/reference_results/gate_h_fixture_scaffold/gate_h_dense_salt_polymer_2x2x2/generated/system.gro",
    ),
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Benchmark the real CPU SIMD short-range nonbonded kernel path on non-MTS short MD."
    )
    parser.add_argument("--gmx", type=Path, default=repo_root / "build_subcounters" / "bin" / "gmx")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "output" / "repulsion_power_9_simd_shortmd_cpu_perf",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--pin", choices=("on", "off", "auto"), default="on")
    parser.add_argument("--dlb", choices=("auto", "no", "yes"), default="auto")
    parser.add_argument("--ntomp", type=int, nargs="+", default=list(DEFAULT_NTOMP))
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    parser.add_argument("--modes", nargs="+", choices=tuple(MODE_SPECS), default=list(DEFAULT_MODES))
    parser.add_argument("--alternate-mode-order", action="store_true")
    parser.add_argument("--warmup-cycles-per-ntomp", type=int, default=0)
    parser.add_argument("--report-affinity", action="store_true")
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


def make_shortmd_mdp(steps: int) -> str:
    if steps <= 0:
        raise ValueError("--steps must be positive")
    return textwrap.dedent(
        f"""\
        title                   = pcff short md specialized simd perf benchmark
        integrator              = md-vv
        dt                      = 0.0005
        nsteps                  = {steps}
        constraints             = none
        cutoff-scheme           = Verlet
        nstlist                 = 20
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
        nstcalcenergy           = {steps}
        nstenergy               = {steps}
        nstlog                  = {steps}
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


def starts_with_label(line: str, label: str) -> bool:
    trimmed = line.strip()
    return trimmed.startswith(label) and (len(trimmed) == len(label) or trimmed[len(label)].isspace())


def extract_seconds(log_contents: str, label: str) -> float | None:
    for line in log_contents.splitlines():
        if not starts_with_label(line, label):
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
    return None if match is None else float(match.group(1))


def load_log(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def configure_environment_for_mode(mode: str) -> dict[str, str]:
    env = os.environ.copy()
    spec = MODE_SPECS[mode]
    for key in spec["env_unset"]:
        env.pop(key, None)
    for key, value in spec["env_set"].items():
        env[key] = value
    return env


def resolve_system_layout(repo_root: Path, system: str) -> tuple[Path, Path]:
    layout = SYSTEM_LAYOUTS.get(system)
    if layout is None:
        raise SystemExit(f"Unknown system fixture key: {system}")
    top_path = repo_root / layout[0]
    coord_path = repo_root / layout[1]
    if not top_path.exists() or not coord_path.exists():
        raise SystemExit(f"Fixture files missing for {system}: {top_path}, {coord_path}")
    return top_path, coord_path


def benchmark_one_run(
    gmx: Path,
    run_dir: Path,
    tpr_path: Path,
    ntomp: int,
    pin_mode: str,
    dlb_mode: str,
    mode: str,
    report_affinity: bool,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    deffnm = run_dir / "run"
    stdout_path = run_dir / "mdrun.stdout.txt"
    log_path = run_dir / "run.log"

    env = configure_environment_for_mode(mode)
    env["OMP_NUM_THREADS"] = str(ntomp)
    if report_affinity:
        env["GMX_REPORT_CPU_AFFINITY"] = "1"

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
        "-dlb",
        dlb_mode,
        "-ntmpi",
        "1",
        "-ntomp",
        str(ntomp),
    ]
    run_command(cmd, cwd=run_dir, env=env, stdout_path=stdout_path)

    log_contents = load_log(log_path)
    ns_per_day = extract_ns_per_day(log_contents)
    if ns_per_day is None:
        raise RuntimeError(f"Could not parse Performance from {log_path}")
    result = {
        "mode": mode,
        "ntomp": ntomp,
        "ns_per_day": ns_per_day,
        "log_path": str(log_path),
        "stdout_path": str(stdout_path),
        "reported_affinity": None,
    }
    for label, field_name in WALLCYCLE_LABELS.items():
        result[field_name] = extract_seconds(log_contents, label)
    for label, field_name in SUBCOUNTER_LABELS.items():
        result[field_name] = extract_seconds(log_contents, label)
    for marker in MODE_SPECS[mode]["required_markers"]:
        if marker not in log_contents:
            raise RuntimeError(f"Expected log marker not found in {log_path}: {marker}")
    affinity_match = re.search(r"New affinity:\s*(.+)", log_contents)
    if affinity_match is not None:
        result["reported_affinity"] = affinity_match.group(1).strip()
    return result


def median_by_mode(runs: list[dict], field: str, modes: list[str]) -> dict[str, float | None]:
    medians: dict[str, float | None] = {}
    for mode in modes:
        values = [item[field] for item in runs if item["mode"] == mode and item.get(field) is not None]
        medians[mode] = statistics.median(values) if values else None
    return medians


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def summarize_runs(runs: list[dict], modes: list[str]) -> list[dict]:
    buckets: dict[tuple[str, int], list[dict]] = {}
    for run in runs:
        buckets.setdefault((run["system"], run["ntomp"]), []).append(run)

    summary = []
    for (system, ntomp), bucket in sorted(buckets.items()):
        counts_by_mode = {mode: sum(1 for item in bucket if item["mode"] == mode) for mode in modes}
        if any(count == 0 for count in counts_by_mode.values()):
            continue
        row = {"system": system, "ntomp": ntomp, "repeats_by_mode": counts_by_mode}
        for mode in modes:
            row[f"{mode}_ns_per_day_median"] = median_by_mode(bucket, "ns_per_day", [mode])[mode]
        for field_name in list(WALLCYCLE_LABELS.values()) + list(SUBCOUNTER_LABELS.values()):
            medians = median_by_mode(bucket, field_name, modes)
            for mode in modes:
                row[f"{mode}_{field_name}_median"] = medians[mode]
        if GENERIC_LABEL in modes and SPECIALIZED_LABEL in modes:
            row["specialized_wall_speedup_vs_generic"] = safe_ratio(
                row[f"{SPECIALIZED_LABEL}_ns_per_day_median"], row[f"{GENERIC_LABEL}_ns_per_day_median"]
            )
            for field_name in ("force_seconds", "nb_f_kernel_seconds", "total_wallcycle_seconds"):
                row[f"specialized_{field_name}_speedup_vs_generic"] = safe_ratio(
                    row.get(f"{GENERIC_LABEL}_{field_name}_median"),
                    row.get(f"{SPECIALIZED_LABEL}_{field_name}_median"),
                )
        if PLAINC_LABEL in modes and SPECIALIZED_LABEL in modes:
            row["specialized_wall_speedup_vs_plainc"] = safe_ratio(
                row[f"{SPECIALIZED_LABEL}_ns_per_day_median"], row[f"{PLAINC_LABEL}_ns_per_day_median"]
            )
        summary.append(row)
    return summary


def format_float(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_markdown(summary_path: Path, metadata: dict, summary_rows: list[dict], modes: list[str]) -> None:
    lines = [
        "# Repulsion-Power-9 SIMD Short-MD CPU Benchmark",
        "",
        "This benchmark is designed to exercise the admitted CPU short-range nonbonded kernel path without exact r-RESPA pair splitting.",
        "",
        "## Host",
        "",
        f"- hostname: `{metadata['hostname']}`",
        f"- cpu: `{metadata['cpu_model']}`",
        f"- gmx: `{metadata['gmx']}`",
        f"- steps per run: `{metadata['steps']}`",
        f"- repeats per point: `{metadata['repeats']}`",
        f"- pin mode: `{metadata['pin']}`",
        f"- DLB mode: `{metadata['dlb']}`",
        f"- alternate mode order: `{metadata['alternate_mode_order']}`",
        f"- warmup cycles per ntomp: `{metadata['warmup_cycles_per_ntomp']}`",
        f"- modes: `{', '.join(modes)}`",
        "",
        "## Notes",
        "",
        "- This is a non-MTS short-MD shape.",
        "- `NB F kernel` comes from the wallcycle subcounter breakdown when the binary was built with `GMX_CYCLE_SUBCOUNTERS=ON`.",
        "- If `NB F kernel` is `n/a`, the selected binary did not emit subcounters for that run.",
        "",
    ]
    for system in sorted({row["system"] for row in summary_rows}):
        rows = [row for row in summary_rows if row["system"] == system]
        lines.extend(
            [
                f"## {system}",
                "",
                "| ntomp | " + " | ".join(f"{mode} ns/day" for mode in modes) + " | specialized/generic |",
                "|" + " --- |" * (2 + len(modes)),
            ]
        )
        for row in rows:
            values = [str(row["ntomp"])]
            for mode in modes:
                values.append(format_float(row.get(f"{mode}_ns_per_day_median")))
            values.append(format_float(row.get("specialized_wall_speedup_vs_generic")))
            lines.append("| " + " | ".join(values) + " |")
        lines.extend(
            [
                "",
                "| ntomp | component | " + " | ".join(modes) + " | generic/specialized time ratio |",
                "|" + " --- |" * (3 + len(modes)),
            ]
        )
        for row in rows:
            for field_name in ("force_seconds", "nb_f_kernel_seconds", "bonded_force_seconds", "update_seconds", "total_wallcycle_seconds"):
                values = [str(row["ntomp"]), field_name]
                for mode in modes:
                    values.append(format_float(row.get(f"{mode}_{field_name}_median"), 6))
                values.append(format_float(row.get(f"specialized_{field_name}_speedup_vs_generic")))
                lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    repo_root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mdp_contents = make_shortmd_mdp(args.steps)

    runs = []
    for system in args.systems:
        top_path, coord_path = resolve_system_layout(repo_root, system)
        system_dir = args.output_dir / system
        system_dir.mkdir(parents=True, exist_ok=True)
        mdp_path = system_dir / "shortmd_specialized_perf.mdp"
        tpr_path = system_dir / "shortmd_specialized_perf.tpr"
        grompp_stdout = system_dir / "grompp.stdout.txt"
        mdp_path.write_text(mdp_contents, encoding="utf-8")
        run_command(
            [
                str(args.gmx),
                "grompp",
                "-f",
                str(mdp_path),
                "-p",
                str(top_path),
                "-c",
                str(coord_path),
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
            for warmup_cycle in range(args.warmup_cycles_per_ntomp):
                warmup_modes = list(args.modes)
                if args.alternate_mode_order and warmup_cycle % 2 == 1:
                    warmup_modes.reverse()
                for mode in warmup_modes:
                    warmup_dir = system_dir / f"ntomp{ntomp}" / mode / f"warmup{warmup_cycle + 1}"
                    benchmark_one_run(
                        gmx=args.gmx,
                        run_dir=warmup_dir,
                        tpr_path=tpr_path,
                        ntomp=ntomp,
                        pin_mode=args.pin,
                        dlb_mode=args.dlb,
                        mode=mode,
                        report_affinity=args.report_affinity,
                    )
            for repeat in range(args.repeats):
                modes_for_repeat = list(args.modes)
                if args.alternate_mode_order and repeat % 2 == 1:
                    modes_for_repeat.reverse()
                for mode in modes_for_repeat:
                    run_dir = system_dir / f"ntomp{ntomp}" / mode / f"repeat{repeat + 1}"
                    row = benchmark_one_run(
                        gmx=args.gmx,
                        run_dir=run_dir,
                        tpr_path=tpr_path,
                        ntomp=ntomp,
                        pin_mode=args.pin,
                        dlb_mode=args.dlb,
                        mode=mode,
                        report_affinity=args.report_affinity,
                    )
                    row["system"] = system
                    row["repeat"] = repeat + 1
                    runs.append(row)

    summary_rows = summarize_runs(runs, args.modes)
    metadata = {
        "hostname": socket.gethostname(),
        "cpu_model": read_cpu_model(),
        "gmx": str(args.gmx),
        "steps": args.steps,
        "repeats": args.repeats,
        "pin": args.pin,
        "dlb": args.dlb,
        "alternate_mode_order": args.alternate_mode_order,
        "warmup_cycles_per_ntomp": args.warmup_cycles_per_ntomp,
        "ntomp": args.ntomp,
        "systems": args.systems,
        "modes": args.modes,
    }
    summary_json = args.output_dir / "summary.json"
    summary_md = args.output_dir / "summary.md"
    summary_json.write_text(json.dumps({"metadata": metadata, "runs": runs, "summary": summary_rows}, indent=2), encoding="utf-8")
    write_markdown(summary_md, metadata, summary_rows, args.modes)
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
