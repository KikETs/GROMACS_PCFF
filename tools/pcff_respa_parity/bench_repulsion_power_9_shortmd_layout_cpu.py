#!/usr/bin/env python3

import argparse
import json
import os
import re
import socket
import statistics
import sys
from pathlib import Path

from bench_repulsion_power_9_simd_shortmd_cpu import (
    DEFAULT_SYSTEMS,
    GENERIC_LABEL,
    MODE_SPECS,
    SPECIALIZED_LABEL,
    SUBCOUNTER_LABELS,
    SYSTEM_LAYOUTS,
    WALLCYCLE_LABELS,
    configure_environment_for_mode,
    extract_ns_per_day,
    extract_seconds,
    load_log,
    read_cpu_model,
    resolve_system_layout,
    run_command,
)


DEFAULT_LAYOUT_SPECS = (
    "omp2:ntmpi=1,ntomp=2",
    "omp6:ntmpi=1,ntomp=6",
    "omp12:ntmpi=1,ntomp=12",
    "split12_pp6_pme6:ntmpi=2,npme=1,ntomp=6,ntomp_pme=6",
)

PME_ACTIVITY_FIELDS = ("pme_spread_seconds", "pme_gather_seconds", "pme_3d_fft_seconds")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Benchmark pure-OpenMP and PME-split CPU layouts for the repulsion-power-9 short-MD shape."
    )
    parser.add_argument("--gmx", type=Path, default=repo_root / "build_subcounters" / "bin" / "gmx")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "output" / "repulsion_power_9_shortmd_layout_opt",
    )
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pin", choices=("on", "off", "auto"), default="on")
    parser.add_argument("--dlb", choices=("auto", "no", "yes"), default="no")
    parser.add_argument("--modes", nargs="+", choices=tuple(MODE_SPECS), default=[GENERIC_LABEL, SPECIALIZED_LABEL])
    parser.add_argument("--alternate-mode-order", action="store_true")
    parser.add_argument("--warmup-cycles-per-layout", type=int, default=1)
    parser.add_argument("--report-affinity", action="store_true")
    parser.add_argument("--layouts", nargs="+", default=list(DEFAULT_LAYOUT_SPECS))
    return parser.parse_args()


def make_shortmd_mdp(steps: int) -> str:
    if steps <= 0:
        raise ValueError("--steps must be positive")
    return (
        "title                   = pcff short md layout optimization benchmark\n"
        "integrator              = md-vv\n"
        "dt                      = 0.0005\n"
        f"nsteps                  = {steps}\n"
        "constraints             = none\n"
        "cutoff-scheme           = Verlet\n"
        "nstlist                 = 20\n"
        "rlist                   = 0.99\n"
        "rvdw                    = 0.9\n"
        "rcoulomb                = 0.9\n"
        "vdwtype                 = Cut-off\n"
        "vdw-modifier            = none\n"
        "coulombtype             = PME\n"
        "coulomb-modifier        = none\n"
        "ewald-rtol              = 1e-6\n"
        "pme-order               = 4\n"
        "fourierspacing          = 0.08\n"
        "epsilon-r               = 1\n"
        "pbc                     = xyz\n"
        "tcoupl                  = no\n"
        "pcoupl                  = no\n"
        "comm-mode               = none\n"
        "verlet-buffer-tolerance = -1\n"
        "gen-vel                 = no\n"
        f"nstcalcenergy           = {steps}\n"
        f"nstenergy               = {steps}\n"
        f"nstlog                  = {steps}\n"
        "nstxout                 = 0\n"
        "nstvout                 = 0\n"
        "nstfout                 = 0\n"
        "nstxout-compressed      = 0\n"
    )


def parse_layout_spec(text: str) -> dict:
    if ":" not in text:
        raise SystemExit(f"Invalid layout spec '{text}'. Expected name:key=value,...")
    name, raw_params = text.split(":", 1)
    params = {}
    for item in raw_params.split(","):
        key, sep, value = item.partition("=")
        if sep != "=":
            raise SystemExit(f"Invalid layout parameter '{item}' in '{text}'")
        params[key.strip()] = int(value.strip())

    ntmpi = params.get("ntmpi")
    ntomp = params.get("ntomp")
    npme = params.get("npme", 0)
    ntomp_pme = params.get("ntomp_pme", ntomp)
    if ntmpi is None or ntomp is None:
        raise SystemExit(f"Layout '{text}' must define ntmpi and ntomp")
    if ntmpi < 1 or ntomp < 1 or npme < 0 or ntomp_pme < 1:
        raise SystemExit(f"Layout '{text}' has non-positive thread settings")
    if npme >= ntmpi:
        raise SystemExit(f"Layout '{text}' has npme >= ntmpi")

    return {
        "name": name.strip(),
        "ntmpi": ntmpi,
        "ntomp": ntomp,
        "npme": npme,
        "ntomp_pme": ntomp_pme,
        "total_threads": (ntmpi - npme) * ntomp + npme * ntomp_pme,
        "has_separate_pme_ranks": npme > 0,
    }


def extract_real_wall_seconds(log_contents: str) -> float | None:
    match = re.search(r"^\s*Time:\s+[0-9.eE+-]+\s+([0-9.eE+-]+)", log_contents, flags=re.MULTILINE)
    return None if match is None else float(match.group(1))


def extract_rank_affinities(log_contents: str) -> list[str]:
    ranks = []
    for line in log_contents.splitlines():
        if "New affinity:" in line and "Rank" in line:
            ranks.append(line.strip())
    return ranks


def benchmark_one_run(
    gmx: Path,
    run_dir: Path,
    tpr_path: Path,
    layout: dict,
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
    env["OMP_NUM_THREADS"] = str(layout["ntomp"])
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
        str(layout["ntmpi"]),
        "-ntomp",
        str(layout["ntomp"]),
    ]
    if layout["npme"] > 0:
        cmd.extend(["-npme", str(layout["npme"]), "-ntomp_pme", str(layout["ntomp_pme"])])

    run_command(cmd, cwd=run_dir, env=env, stdout_path=stdout_path)

    log_contents = load_log(log_path)
    ns_per_day = extract_ns_per_day(log_contents)
    if ns_per_day is None:
        raise RuntimeError(f"Could not parse Performance from {log_path}")

    result = {
        "layout_name": layout["name"],
        "mode": mode,
        "ns_per_day": ns_per_day,
        "real_wall_seconds": extract_real_wall_seconds(log_contents),
        "log_path": str(log_path),
        "stdout_path": str(stdout_path),
        "reported_affinity": extract_rank_affinities(log_contents),
        "has_separate_pme_ranks": layout["has_separate_pme_ranks"],
        "ntmpi": layout["ntmpi"],
        "ntomp": layout["ntomp"],
        "npme": layout["npme"],
        "ntomp_pme": layout["ntomp_pme"],
        "total_threads": layout["total_threads"],
    }
    for label, field_name in WALLCYCLE_LABELS.items():
        result[field_name] = extract_seconds(log_contents, label)
    for label, field_name in SUBCOUNTER_LABELS.items():
        result[field_name] = extract_seconds(log_contents, label)
    for marker in MODE_SPECS[mode]["required_markers"]:
        if marker not in log_contents:
            raise RuntimeError(f"Expected log marker not found in {log_path}: {marker}")
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
    buckets: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        buckets.setdefault((run["system"], run["layout_name"]), []).append(run)

    summary = []
    fields = (
        ["ns_per_day", "real_wall_seconds"]
        + list(WALLCYCLE_LABELS.values())
        + list(SUBCOUNTER_LABELS.values())
        + list(PME_ACTIVITY_FIELDS)
    )
    for (system, layout_name), bucket in sorted(buckets.items()):
        counts_by_mode = {mode: sum(1 for item in bucket if item["mode"] == mode) for mode in modes}
        if any(count == 0 for count in counts_by_mode.values()):
            continue
        row = {
            "system": system,
            "layout_name": layout_name,
            "layout": {
                "ntmpi": bucket[0]["ntmpi"],
                "ntomp": bucket[0]["ntomp"],
                "npme": bucket[0]["npme"],
                "ntomp_pme": bucket[0]["ntomp_pme"],
                "total_threads": bucket[0]["total_threads"],
                "has_separate_pme_ranks": bucket[0]["has_separate_pme_ranks"],
            },
            "repeats_by_mode": counts_by_mode,
        }
        for field in fields:
            medians = median_by_mode(bucket, field, modes)
            for mode in modes:
                row[f"{mode}_{field}_median"] = medians[mode]
        if GENERIC_LABEL in modes and SPECIALIZED_LABEL in modes:
            row["specialized_speedup_vs_generic"] = safe_ratio(
                row.get(f"{SPECIALIZED_LABEL}_ns_per_day_median"), row.get(f"{GENERIC_LABEL}_ns_per_day_median")
            )
            row["specialized_real_wall_speedup_vs_generic"] = safe_ratio(
                row.get(f"{GENERIC_LABEL}_real_wall_seconds_median"),
                row.get(f"{SPECIALIZED_LABEL}_real_wall_seconds_median"),
            )
        summary.append(row)
    return summary


def format_float(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_markdown(summary_path: Path, metadata: dict, summary_rows: list[dict], modes: list[str]) -> None:
    lines = [
        "# Repulsion-Power-9 Short-MD CPU Layout Sweep",
        "",
        "This benchmark compares pure-OpenMP and PME-split CPU layouts on the non-MTS short-MD shape.",
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
        f"- warmup cycles per layout: `{metadata['warmup_cycles_per_layout']}`",
        f"- modes: `{', '.join(modes)}`",
        "",
        "## Notes",
        "",
        "- `real_wall_seconds` comes from the `Time:` line and is the metric to use for final speed claims.",
        "- For layouts with separate PME ranks, `Force`, `PME mesh`, and related wallcycle rows overlap across ranks and are not additive wall shares.",
        "- `NB F kernel` remains useful for PP-kernel comparison inside the same layout, but not as a total-wall decomposition term for PME-split layouts.",
        "",
    ]

    for system in sorted({row["system"] for row in summary_rows}):
        rows = [row for row in summary_rows if row["system"] == system]
        lines.extend(
            [
                f"## {system}",
                "",
                "| layout | total threads | ntmpi | npme | ntomp | ntomp_pme | "
                + " | ".join(f"{mode} ns/day" for mode in modes)
                + " | specialized/generic |",
                "|" + " --- |" * (7 + len(modes)),
            ]
        )
        for row in rows:
            layout = row["layout"]
            values = [
                row["layout_name"],
                str(layout["total_threads"]),
                str(layout["ntmpi"]),
                str(layout["npme"]),
                str(layout["ntomp"]),
                str(layout["ntomp_pme"]),
            ]
            for mode in modes:
                values.append(format_float(row.get(f"{mode}_ns_per_day_median")))
            values.append(format_float(row.get("specialized_speedup_vs_generic")))
            lines.append("| " + " | ".join(values) + " |")
        lines.extend(
            [
                "",
                "| layout | metric | " + " | ".join(modes) + " | generic/specialized time ratio |",
                "|" + " --- |" * (3 + len(modes)),
            ]
        )
        metrics = ["real_wall_seconds", "force_seconds", "pme_mesh_seconds", "update_seconds", "nb_f_kernel_seconds"] + list(
            PME_ACTIVITY_FIELDS
        )
        for row in rows:
            for metric in metrics:
                generic_value = row.get(f"{GENERIC_LABEL}_{metric}_median")
                specialized_value = row.get(f"{SPECIALIZED_LABEL}_{metric}_median")
                values = [row["layout_name"], metric]
                for mode in modes:
                    values.append(format_float(row.get(f"{mode}_{metric}_median"), 6))
                values.append(format_float(safe_ratio(generic_value, specialized_value)))
                lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def enrich_with_pme_subcounters(runs: list[dict]) -> None:
    for run in runs:
        log_contents = load_log(Path(run["log_path"]))
        run["pme_spread_seconds"] = extract_seconds(log_contents, "PME spread")
        run["pme_gather_seconds"] = extract_seconds(log_contents, "PME gather")
        run["pme_3d_fft_seconds"] = extract_seconds(log_contents, "PME 3D-FFT")


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    layouts = [parse_layout_spec(spec) for spec in args.layouts]
    repo_root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mdp_contents = make_shortmd_mdp(args.steps)

    runs: list[dict] = []
    for system in args.systems:
        top_path, coord_path = resolve_system_layout(repo_root, system)
        system_dir = args.output_dir / system
        system_dir.mkdir(parents=True, exist_ok=True)
        mdp_path = system_dir / "shortmd_layout_opt.mdp"
        tpr_path = system_dir / "shortmd_layout_opt.tpr"
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

        for layout in layouts:
            for warmup_cycle in range(args.warmup_cycles_per_layout):
                warmup_modes = list(args.modes)
                if args.alternate_mode_order and warmup_cycle % 2 == 1:
                    warmup_modes.reverse()
                for mode in warmup_modes:
                    warmup_dir = system_dir / layout["name"] / mode / f"warmup{warmup_cycle + 1}"
                    benchmark_one_run(
                        gmx=args.gmx,
                        run_dir=warmup_dir,
                        tpr_path=tpr_path,
                        layout=layout,
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
                    run_dir = system_dir / layout["name"] / mode / f"repeat{repeat + 1}"
                    row = benchmark_one_run(
                        gmx=args.gmx,
                        run_dir=run_dir,
                        tpr_path=tpr_path,
                        layout=layout,
                        pin_mode=args.pin,
                        dlb_mode=args.dlb,
                        mode=mode,
                        report_affinity=args.report_affinity,
                    )
                    row["system"] = system
                    row["repeat"] = repeat + 1
                    runs.append(row)

    enrich_with_pme_subcounters(runs)
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
        "warmup_cycles_per_layout": args.warmup_cycles_per_layout,
        "systems": args.systems,
        "modes": args.modes,
        "layouts": layouts,
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
