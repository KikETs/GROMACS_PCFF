from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

from freeze_gate_a_oracle import (
    DEFAULT_GMX,
    base_env,
    capture_output,
    command_record,
    env_delta,
    write_commands_script,
    write_text,
)
from validate_gate_c_nb_bonded_gpu import (
    capture_optional_output,
    load_json,
    maybe_build,
    parse_gpu_support,
    parse_precision_mode,
    run_command_allow_failure,
)
from validate_gate_e_update_gpu import parse_layout_report
from validate_gate_g_long_ensemble import DT_PS, EXACT_RESPA_FACTOR, steps_from_ps
from validate_gate_i_charged_long_npt_conditioning import DEFAULT_SCAFFOLD_MANIFEST, make_gate_i_npt_mdp


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "output" / "exact_respa_gpu_hybrid_gate_i_hostlocal_calibration_20260422"
PERFORMANCE_RE = re.compile(
    r"Performance:\s+(?P<ns_per_day>[0-9.]+)\s+(?P<hour_per_ns>[0-9.]+)\s+(?P<ms_per_step>[0-9.]+)"
)
WALLCYCLE_FIELDS = (
    ("Neighbor search", "neighbor_search_seconds"),
    ("Force", "force_seconds"),
    ("PME mesh", "pme_mesh_seconds"),
    ("Update", "update_seconds"),
    ("Total", "total_wallcycle_seconds"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a short Gate-I-shaped host-local performance calibration for the exact r-RESPA "
            "GPU hybrid runtime on the charged gate_h_dense_salt_polymer_2x2x2 scaffold."
        )
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument(
        "--scaffold-manifest",
        default=str(DEFAULT_SCAFFOLD_MANIFEST),
        help="Charged scaffold manifest path.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--build-target", default="gmx", help="CMake target to build.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step.")
    parser.add_argument("--build-dir", default=None, help="Optional explicit build directory.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("cpu", "gpu_hybrid"),
        default=("cpu", "gpu_hybrid"),
        help="Runtime modes to benchmark on the same Gate-I-shaped scaffold.",
    )
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks.")
    parser.add_argument("--ntomp", type=int, default=12, help="OpenMP threads.")
    parser.add_argument("--npme", type=int, default=None, help="Optional explicit -npme value.")
    parser.add_argument(
        "--pin",
        choices=("off", "on", "auto"),
        default="on",
        help="GROMACS mdrun -pin policy for this host-local performance probe.",
    )
    parser.add_argument("--pinoffset", type=int, default=None, help="Optional GROMACS mdrun -pinoffset.")
    parser.add_argument("--pinstride", type=int, default=1, help="Optional GROMACS mdrun -pinstride.")
    parser.add_argument("--equil-ps", type=float, default=20.0, help="Equilibration duration in ps.")
    parser.add_argument("--prod-ps", type=float, default=40.0, help="Production duration in ps.")
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=EXACT_RESPA_FACTOR * 100,
        help="Energy/log sampling interval in base steps. Must be a positive multiple of the exact-r-RESPA factor.",
    )
    parser.add_argument("--temperature-k", type=float, default=300.0, help="Target temperature.")
    parser.add_argument("--pressure-bar", type=float, default=1.0, help="Target pressure.")
    parser.add_argument("--tau-t-ps", type=float, default=0.5, help="Thermostat coupling time.")
    parser.add_argument("--tau-p-ps", type=float, default=5.0, help="Barostat coupling time.")
    parser.add_argument(
        "--compressibility-bar-inv",
        type=float,
        default=4.5e-5,
        help="Isotropic compressibility in bar^-1 for C-rescale pressure coupling.",
    )
    parser.add_argument("--seed", type=int, default=70001, help="Velocity generation seed for equilibration.")
    parser.add_argument(
        "--ld-seed-base",
        type=int,
        default=80001,
        help="Base stochastic seed used for thermostat/barostat coupling during equilibration.",
    )
    parser.add_argument(
        "--prod-ld-seed-offset",
        type=int,
        default=100000,
        help="Offset added to --ld-seed-base for the production stochastic seed.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.ntmpi != 1:
        raise ValueError("This host-local GPU hybrid calibration is single-rank only; --ntmpi must remain 1.")
    if args.sample_interval <= 0 or args.sample_interval % EXACT_RESPA_FACTOR != 0:
        raise ValueError("sample-interval must be a positive multiple of the exact-r-RESPA factor.")
    for name, duration_ps in (("equil-ps", args.equil_ps), ("prod-ps", args.prod_ps)):
        steps = steps_from_ps(duration_ps)
        if not math.isclose(steps * DT_PS, duration_ps, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{name} must be representable as an integer number of base steps.")
        if steps <= 0 or steps % EXACT_RESPA_FACTOR != 0:
            raise ValueError(f"{name} must correspond to a positive multiple of the exact-r-RESPA factor.")


def starts_with_wallcycle_label(line: str, label: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(label) and (len(stripped) == len(label) or stripped[len(label)].isspace())


def extract_wallcycle_seconds(log_text: str, label: str) -> float | None:
    for line in log_text.splitlines():
        if not starts_with_wallcycle_label(line, label):
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            return float(fields[-3])
        except ValueError:
            continue
    return None


def parse_log_metrics(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = PERFORMANCE_RE.search(text)
    payload: dict[str, object] = {
        "log_path": str(log_path),
        "ns_per_day": None,
        "hour_per_ns": None,
        "ms_per_step": None,
    }
    if match is not None:
        payload["ns_per_day"] = float(match.group("ns_per_day"))
        payload["hour_per_ns"] = float(match.group("hour_per_ns"))
        payload["ms_per_step"] = float(match.group("ms_per_step"))
    for label, field_name in WALLCYCLE_FIELDS:
        payload[field_name] = extract_wallcycle_seconds(text, label)
    return payload


def load_scaffold_inputs(scaffold_manifest_path: Path) -> dict[str, object]:
    manifest = load_json(scaffold_manifest_path)
    if str(manifest.get("derived_system")) != "gate_h_dense_salt_polymer_2x2x2":
        raise ValueError("This benchmark is frozen to gate_h_dense_salt_polymer_2x2x2.")
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("Scaffold manifest is missing the artifacts dictionary.")
    gro_path = Path(str(artifacts["gro"]))
    top_path = Path(str(artifacts["topology"]))
    return {
        "manifest": manifest,
        "gro_path": gro_path,
        "top_path": top_path,
    }


def mdrun_args(args: argparse.Namespace, tpr_path: Path, deffnm: Path, mode: str) -> list[str]:
    argv = [
        "-s",
        str(tpr_path),
        "-deffnm",
        str(deffnm),
        "-ntmpi",
        str(args.ntmpi),
        "-ntomp",
        str(args.ntomp),
        "-dlb",
        "no",
        "-nb",
        "gpu" if mode == "gpu_hybrid" else "cpu",
        "-pme",
        "cpu",
        "-bonded",
        "cpu",
        "-update",
        "cpu",
        "-pin",
        args.pin,
    ]
    if args.pin != "off":
        if args.pinoffset is not None:
            argv.extend(["-pinoffset", str(args.pinoffset)])
        if args.pinstride is not None:
            argv.extend(["-pinstride", str(args.pinstride)])
    return argv


def run_mdrun(
    *,
    gmx: Path,
    argv: list[str],
    env: dict[str, str],
    logs_dir: Path,
    commands: list[dict[str, object]],
    label: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    stdout_path = logs_dir / f"{label}.stdout"
    stderr_path = logs_dir / f"{label}.stderr"
    full_argv = [str(gmx), "mdrun", *argv]
    started = time.time()
    result = run_command_allow_failure(full_argv, cwd=REPO_ROOT, env=env, stdout_path=stdout_path, stderr_path=stderr_path)
    elapsed_seconds = time.time() - started
    commands.append(
        command_record(
            label,
            full_argv,
            cwd=REPO_ROOT,
            env_overrides=env_delta(env, os.environ),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    )
    payload = {
        "run_id": label,
        "argv": full_argv,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "returncode": result.returncode,
        "elapsed_seconds": elapsed_seconds,
        "layout_report": parse_layout_report(stdout_path, stderr_path, args),
    }
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed; see {stderr_path}")
    return payload


def combined_throughput_ns_day(total_ps: float, elapsed_seconds: float) -> float | None:
    if elapsed_seconds <= 0.0:
        return None
    total_ns = total_ps / 1000.0
    return total_ns * 86400.0 / elapsed_seconds


def safe_add(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return lhs + rhs


def summarize_mode(
    *,
    mode: str,
    equil_metrics: dict[str, object],
    prod_metrics: dict[str, object],
    equil_run: dict[str, object],
    prod_run: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    total_elapsed = float(equil_run["elapsed_seconds"]) + float(prod_run["elapsed_seconds"])
    combined = {
        "elapsed_seconds": total_elapsed,
        "ns_per_day": combined_throughput_ns_day(args.equil_ps + args.prod_ps, total_elapsed),
        "force_seconds": safe_add(equil_metrics["force_seconds"], prod_metrics["force_seconds"]),
        "pme_mesh_seconds": safe_add(equil_metrics["pme_mesh_seconds"], prod_metrics["pme_mesh_seconds"]),
        "update_seconds": safe_add(equil_metrics["update_seconds"], prod_metrics["update_seconds"]),
        "total_wallcycle_seconds": safe_add(
            equil_metrics["total_wallcycle_seconds"], prod_metrics["total_wallcycle_seconds"]
        ),
    }
    return {
        "mode": mode,
        "ntmpi": args.ntmpi,
        "ntomp": args.ntomp,
        "pin": args.pin,
        "pinoffset": args.pinoffset,
        "pinstride": args.pinstride,
        "equil_ps": args.equil_ps,
        "prod_ps": args.prod_ps,
        "sample_interval": args.sample_interval,
        "equil": {
            "metrics": equil_metrics,
            "runtime": equil_run,
        },
        "prod": {
            "metrics": prod_metrics,
            "runtime": prod_run,
        },
        "combined": combined,
    }


def emit_summary_tsv(rows: list[dict[str, object]], out_path: Path) -> None:
    header = [
        "mode",
        "ntomp",
        "pin",
        "pinstride",
        "equil_ns_day",
        "prod_ns_day",
        "combined_ns_day",
        "equil_force_s",
        "prod_force_s",
        "combined_force_s",
        "equil_update_s",
        "prod_update_s",
        "combined_update_s",
        "combined_elapsed_s",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        combined = row["combined"]
        equil_metrics = row["equil"]["metrics"]
        prod_metrics = row["prod"]["metrics"]
        values = [
            row["mode"],
            str(row["ntomp"]),
            str(row["pin"]),
            "" if row["pinstride"] is None else str(row["pinstride"]),
            "" if equil_metrics["ns_per_day"] is None else f"{equil_metrics['ns_per_day']:.3f}",
            "" if prod_metrics["ns_per_day"] is None else f"{prod_metrics['ns_per_day']:.3f}",
            "" if combined["ns_per_day"] is None else f"{combined['ns_per_day']:.3f}",
            "" if equil_metrics["force_seconds"] is None else f"{equil_metrics['force_seconds']:.3f}",
            "" if prod_metrics["force_seconds"] is None else f"{prod_metrics['force_seconds']:.3f}",
            "" if combined["force_seconds"] is None else f"{combined['force_seconds']:.3f}",
            "" if equil_metrics["update_seconds"] is None else f"{equil_metrics['update_seconds']:.3f}",
            "" if prod_metrics["update_seconds"] is None else f"{prod_metrics['update_seconds']:.3f}",
            "" if combined["update_seconds"] is None else f"{combined['update_seconds']:.3f}",
            f"{combined['elapsed_seconds']:.3f}",
        ]
        lines.append("\t".join(values))
    write_text(out_path, "\n".join(lines) + "\n")


def build_readme(manifest: dict[str, object], mode_rows: list[dict[str, object]]) -> str:
    lines = [
        "# exact r-RESPA GPU Hybrid Gate-I-Shaped Host-Local Calibration",
        "",
        "This artifact is a host-local performance calibration, not an exactness gate.",
        "",
        "Runtime notes:",
        "- The charged scaffold is `gate_h_dense_salt_polymer_2x2x2`.",
        "- The MDP shape matches Gate I style NPT conditioning (`v-rescale` + `c-rescale`) but uses a short horizon for throughput calibration.",
        "- `-reprod` is intentionally omitted because the GPU hybrid path cannot use it; this benchmark is for runtime selection, not exactness certification.",
        "",
        "## Results",
        "",
        "| mode | ntomp | pin | pinstride | equil ns/day | prod ns/day | combined ns/day | combined elapsed s |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in mode_rows:
        combined = row["combined"]
        equil_metrics = row["equil"]["metrics"]
        prod_metrics = row["prod"]["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["mode"],
                    str(row["ntomp"]),
                    str(row["pin"]),
                    "" if row["pinstride"] is None else str(row["pinstride"]),
                    "" if equil_metrics["ns_per_day"] is None else f"{equil_metrics['ns_per_day']:.3f}",
                    "" if prod_metrics["ns_per_day"] is None else f"{prod_metrics['ns_per_day']:.3f}",
                    "" if combined["ns_per_day"] is None else f"{combined['ns_per_day']:.3f}",
                    f"{combined['elapsed_seconds']:.3f}",
                ]
            )
            + " |"
        )

    if len(mode_rows) >= 2:
        by_mode = {row["mode"]: row for row in mode_rows}
        cpu_row = by_mode.get("cpu")
        gpu_row = by_mode.get("gpu_hybrid")
        if cpu_row is not None and gpu_row is not None:
            cpu_ns_day = cpu_row["combined"]["ns_per_day"]
            gpu_ns_day = gpu_row["combined"]["ns_per_day"]
            if cpu_ns_day not in (None, 0) and gpu_ns_day is not None:
                speedup = gpu_ns_day / cpu_ns_day
                lines.extend(
                    [
                        "",
                        "## Host-Local Decision",
                        "",
                        f"- Combined GPU-hybrid speedup vs CPU baseline: `{speedup:.3f}x`",
                        "- This is the runtime-selection basis for host-local experiments only.",
                    ]
                )

    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- GROMACS binary: `{manifest['inputs']['gmx']}`",
            f"- Modes: `{', '.join(manifest['inputs']['modes'])}`",
            f"- Thread shape: `-ntmpi {manifest['inputs']['ntmpi']} -ntomp {manifest['inputs']['ntomp']}`",
            f"- Pinning: `-pin {manifest['inputs']['pin']}`"
            + (
                ""
                if manifest["inputs"]["pinstride"] is None
                else f" `-pinstride {manifest['inputs']['pinstride']}`"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    validate_args(args)

    gmx = Path(args.gmx).resolve()
    build_dir = Path(args.build_dir).resolve() if args.build_dir is not None else None
    maybe_build(args, build_dir)

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    scaffold_manifest_path = Path(args.scaffold_manifest).resolve()
    scaffold_inputs = load_scaffold_inputs(scaffold_manifest_path)
    gro_path = scaffold_inputs["gro_path"]
    top_path = scaffold_inputs["top_path"]

    gmx_version = capture_output([str(gmx), "--version"], cwd=REPO_ROOT)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "kind": "exact_respa_gpu_hybrid_gate_i_hostlocal_calibration",
        "scope": {
            "purpose": "host_local_runtime_selection",
            "claim_boundary": (
                "This artifact does not widen the audited exactness claim. It only fixes the host-local "
                "experiment runtime default for the charged Gate-I-shaped scaffold."
            ),
        },
        "inputs": {
            "gmx": str(gmx),
            "scaffold_manifest": str(scaffold_manifest_path),
            "modes": list(args.modes),
            "ntmpi": args.ntmpi,
            "ntomp": args.ntomp,
            "pin": args.pin,
            "pinoffset": args.pinoffset,
            "pinstride": args.pinstride,
            "equil_ps": args.equil_ps,
            "prod_ps": args.prod_ps,
            "sample_interval": args.sample_interval,
            "seed": args.seed,
            "ld_seed_base": args.ld_seed_base,
            "prod_ld_seed_offset": args.prod_ld_seed_offset,
        },
        "host": {
            "gmx_version": gmx_version,
            "gpu_support": parse_gpu_support(gmx_version),
            "precision": parse_precision_mode(gmx_version),
            "nvidia_smi": capture_optional_output(["nvidia-smi", "-L"]),
        },
        "scaffold": scaffold_inputs["manifest"],
        "runs": [],
    }

    mode_rows: list[dict[str, object]] = []
    commands: list[dict[str, object]] = []

    for mode in args.modes:
        mode_root = out_root / mode
        inputs_dir = mode_root / "inputs"
        logs_dir = mode_root / "logs"
        for directory in (inputs_dir, logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        equil_mdp_path = inputs_dir / "equil.mdp"
        prod_mdp_path = inputs_dir / "prod.mdp"
        equil_tpr_path = inputs_dir / "equil.tpr"
        prod_tpr_path = inputs_dir / "prod.tpr"
        equil_mdout_path = inputs_dir / "equil.mdout.mdp"
        prod_mdout_path = inputs_dir / "prod.mdout.mdp"
        equil_deffnm = mode_root / "equil"
        prod_deffnm = mode_root / "prod"

        write_text(
            equil_mdp_path,
            make_gate_i_npt_mdp(
                duration_ps=args.equil_ps,
                sample_interval=args.sample_interval,
                phase="equil",
                seed=args.seed,
                args=args,
                ld_seed=args.ld_seed_base,
            ),
        )
        write_text(
            prod_mdp_path,
            make_gate_i_npt_mdp(
                duration_ps=args.prod_ps,
                sample_interval=args.sample_interval,
                phase="prod",
                seed=args.seed + 1,
                args=args,
                ld_seed=args.ld_seed_base + args.prod_ld_seed_offset,
            ),
        )

        env = base_env(args)

        equil_grompp_argv = [
            str(gmx),
            "grompp",
            "-f",
            str(equil_mdp_path),
            "-c",
            str(gro_path),
            "-p",
            str(top_path),
            "-o",
            str(equil_tpr_path),
            "-po",
            str(equil_mdout_path),
            "-maxwarn",
            "1",
        ]
        equil_grompp_stdout = logs_dir / "equil_grompp.stdout"
        equil_grompp_stderr = logs_dir / "equil_grompp.stderr"
        equil_grompp = run_command_allow_failure(
            equil_grompp_argv,
            cwd=REPO_ROOT,
            env=env,
            stdout_path=equil_grompp_stdout,
            stderr_path=equil_grompp_stderr,
        )
        commands.append(
            command_record(
                f"{mode}_equil_grompp",
                equil_grompp_argv,
                cwd=REPO_ROOT,
                env_overrides=env_delta(env, os.environ),
                stdout_path=equil_grompp_stdout,
                stderr_path=equil_grompp_stderr,
            )
        )
        if equil_grompp.returncode != 0:
            raise RuntimeError(f"{mode} equilibration grompp failed; see {equil_grompp_stderr}")

        equil_run = run_mdrun(
            gmx=gmx,
            argv=mdrun_args(args, equil_tpr_path, equil_deffnm, mode),
            env=env,
            logs_dir=logs_dir,
            commands=commands,
            label=f"{mode}_equil_mdrun",
            args=args,
        )

        prod_grompp_argv = [
            str(gmx),
            "grompp",
            "-f",
            str(prod_mdp_path),
            "-c",
            str(equil_deffnm.with_suffix(".gro")),
            "-p",
            str(top_path),
            "-o",
            str(prod_tpr_path),
            "-po",
            str(prod_mdout_path),
            "-t",
            str(equil_deffnm.with_suffix(".cpt")),
            "-maxwarn",
            "1",
        ]
        prod_grompp_stdout = logs_dir / "prod_grompp.stdout"
        prod_grompp_stderr = logs_dir / "prod_grompp.stderr"
        prod_grompp = run_command_allow_failure(
            prod_grompp_argv,
            cwd=REPO_ROOT,
            env=env,
            stdout_path=prod_grompp_stdout,
            stderr_path=prod_grompp_stderr,
        )
        commands.append(
            command_record(
                f"{mode}_prod_grompp",
                prod_grompp_argv,
                cwd=REPO_ROOT,
                env_overrides=env_delta(env, os.environ),
                stdout_path=prod_grompp_stdout,
                stderr_path=prod_grompp_stderr,
            )
        )
        if prod_grompp.returncode != 0:
            raise RuntimeError(f"{mode} production grompp failed; see {prod_grompp_stderr}")

        prod_run = run_mdrun(
            gmx=gmx,
            argv=mdrun_args(args, prod_tpr_path, prod_deffnm, mode),
            env=env,
            logs_dir=logs_dir,
            commands=commands,
            label=f"{mode}_prod_mdrun",
            args=args,
        )

        row = summarize_mode(
            mode=mode,
            equil_metrics=parse_log_metrics(equil_deffnm.with_suffix(".log")),
            prod_metrics=parse_log_metrics(prod_deffnm.with_suffix(".log")),
            equil_run=equil_run,
            prod_run=prod_run,
            args=args,
        )
        mode_rows.append(row)
        manifest["runs"].append(row)

    if mode_rows:
        best_row = max(
            (row for row in mode_rows if row["combined"]["ns_per_day"] is not None),
            key=lambda row: row["combined"]["ns_per_day"],
        )
        manifest["decision"] = {
            "best_mode": best_row["mode"],
            "best_combined_ns_day": best_row["combined"]["ns_per_day"],
            "host_local_runtime_default": {
                "ntmpi": args.ntmpi,
                "ntomp": args.ntomp,
                "pin": args.pin,
                "pinoffset": args.pinoffset,
                "pinstride": args.pinstride,
            },
        }
        if {row["mode"] for row in mode_rows} == {"cpu", "gpu_hybrid"}:
            by_mode = {row["mode"]: row for row in mode_rows}
            cpu_ns_day = by_mode["cpu"]["combined"]["ns_per_day"]
            gpu_ns_day = by_mode["gpu_hybrid"]["combined"]["ns_per_day"]
            if cpu_ns_day not in (None, 0) and gpu_ns_day is not None:
                manifest["decision"]["gpu_hybrid_speedup_vs_cpu"] = gpu_ns_day / cpu_ns_day

    manifest["commands"] = commands

    write_text(out_root / "summary.json", json.dumps(mode_rows, indent=2, sort_keys=True) + "\n")
    write_text(out_root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    emit_summary_tsv(mode_rows, out_root / "summary.tsv")
    write_text(out_root / "README.md", build_readme(manifest, mode_rows))
    write_commands_script(out_root / "replay_commands.sh", commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
