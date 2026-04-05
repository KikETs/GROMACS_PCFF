from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from freeze_gate_a_oracle import (
    base_env,
    capture_output,
    command_record,
    energy_summary,
    env_delta,
    make_exact_respa_mdp,
    parse_energy_dump,
    parse_event_trace,
    parse_merge_trace_dir,
    parse_total_force_dump,
    run_command,
    write_commands_script,
    write_text,
    reference_event_trace,
)
from validate_gate_b_nb_gpu import (
    compare_energy_frames,
    compare_event_trace,
    compare_per_level_force_entries,
    compare_total_force_entries,
    parse_gpu_support,
    parse_precision_mode,
    run_command_allow_failure,
)
from validate_gate_e_update_gpu import parse_layout_report


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GPU_GMX = REPO_ROOT / "build_gateb_cuda" / "bin" / "gmx"
DEFAULT_CPU_GMX = REPO_ROOT / "build" / "bin" / "gmx"
DEFAULT_SCAFFOLD_MANIFEST = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_h_fixture_scaffold"
    / "gate_h_dense_oligomer_2x2x2"
    / "fixture_manifest.json"
)
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_h_neutral_scaffold_bringup"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short exact-r-RESPA bring-up for a Gate H scaffold system."
    )
    parser.add_argument("--gmx", default=None, help="Path to the GROMACS CLI binary.")
    parser.add_argument("--scaffold-manifest", default=str(DEFAULT_SCAFFOLD_MANIFEST), help="Scaffold manifest path.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--outer-steps", type=int, default=8, help="Number of exact r-RESPA outer steps.")
    parser.add_argument("--pair14-level", type=int, default=1, help="Exact r-RESPA pair14 level.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks.")
    parser.add_argument("--ntomp", type=int, default=1, help="OpenMP threads.")
    parser.add_argument("--npme", type=int, default=None, help="Optional explicit -npme value; omitted by default.")
    return parser.parse_args()


def choose_default_gmx() -> Path:
    if DEFAULT_GPU_GMX.exists():
        return DEFAULT_GPU_GMX
    return DEFAULT_CPU_GMX


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def trace_env(args: argparse.Namespace, run_root: Path) -> dict[str, str]:
    env = base_env(args)
    env["GMX_EXACT_RESPA_RUNTIME_EVENT_TRACE_FILE"] = str(run_root / "event_trace.tsv")
    env["GMX_PCFF_RESPA_MERGE_TRACE_DIR"] = str(run_root / "merge_trace")
    env["GMX_EXACT_RESPA_TOTAL_FORCE_DUMP_FILE"] = str(run_root / "total_force.tsv")
    return env


def mdrun_args_cpu(args: argparse.Namespace, tpr_path: Path, deffnm: Path) -> list[str]:
    return [
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
        "cpu",
        "-pme",
        "cpu",
        "-bonded",
        "cpu",
        "-update",
        "cpu",
        "-pin",
        "off",
        "-reprod",
    ]


def mdrun_args_gpu(args: argparse.Namespace, tpr_path: Path, deffnm: Path) -> list[str]:
    return [
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
        "gpu",
        "-pme",
        "gpu",
        "-bonded",
        "gpu",
        "-update",
        "gpu",
        "-pin",
        "off",
    ]


def dump_energy_frames(gmx: Path, edr_path: Path) -> list[dict[str, object]]:
    return parse_energy_dump(capture_output([str(gmx), "dump", "-e", str(edr_path)], cwd=REPO_ROOT))


def run_grompp(
    *,
    gmx: Path,
    mdp_path: Path,
    conf_path: Path,
    top_path: Path,
    tpr_path: Path,
    mdout_path: Path,
    env: dict[str, str],
    logs_dir: Path,
    commands: list[dict[str, object]],
) -> None:
    argv = [
        str(gmx),
        "grompp",
        "-f",
        str(mdp_path),
        "-c",
        str(conf_path),
        "-p",
        str(top_path),
        "-o",
        str(tpr_path),
        "-po",
        str(mdout_path),
        "-maxwarn",
        "1",
    ]
    stdout_path = logs_dir / "grompp.stdout"
    stderr_path = logs_dir / "grompp.stderr"
    run_command(argv, cwd=REPO_ROOT, env=env, stdout_path=stdout_path, stderr_path=stderr_path)
    commands.append(
        command_record(
            "grompp",
            argv,
            cwd=REPO_ROOT,
            env_overrides=env_delta(env, os.environ),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    )


def collect_run_outputs(gmx: Path, deffnm_root: Path, trace_root: Path) -> dict[str, object]:
    events = parse_event_trace(trace_root / "event_trace.tsv")
    energy_frames = dump_energy_frames(gmx, deffnm_root.with_suffix(".edr"))
    total_force = parse_total_force_dump(trace_root / "total_force.tsv")
    per_level_force = parse_merge_trace_dir(trace_root / "merge_trace")
    return {
        "events": events,
        "energy_frames": energy_frames,
        "total_force": total_force,
        "per_level_force": per_level_force,
    }


def main() -> int:
    args = parse_args()
    gmx = Path(args.gmx) if args.gmx else choose_default_gmx()
    scaffold_manifest = load_json(Path(args.scaffold_manifest))
    out_root = Path(args.out).resolve()
    system_id = str(scaffold_manifest["derived_system"])
    if out_root.exists():
        shutil.rmtree(out_root)
    inputs_dir = out_root / "inputs"
    cpu_root = out_root / "cpu"
    gpu_root = out_root / "gpu"
    logs_dir = out_root / "logs"
    summaries_dir = out_root / "summaries"
    for path in (inputs_dir, cpu_root, gpu_root, logs_dir, summaries_dir):
        path.mkdir(parents=True, exist_ok=True)

    gro_path = Path(str(scaffold_manifest["artifacts"]["gro"]))
    top_path = Path(str(scaffold_manifest["artifacts"]["topology"]))
    mdp_path = inputs_dir / "exact_respa_short.mdp"
    write_text(mdp_path, make_exact_respa_mdp(args.outer_steps, args.pair14_level))
    tpr_path = inputs_dir / "exact_respa_short.tpr"
    mdout_path = inputs_dir / "mdout.mdp"

    commands: list[dict[str, object]] = []
    base_environment = base_env(args)
    run_grompp(
        gmx=gmx,
        mdp_path=mdp_path,
        conf_path=gro_path,
        top_path=top_path,
        tpr_path=tpr_path,
        mdout_path=mdout_path,
        env=base_environment,
        logs_dir=logs_dir,
        commands=commands,
    )

    cpu_env = trace_env(args, cpu_root)
    cpu_argv = [str(gmx), "mdrun", *mdrun_args_cpu(args, tpr_path, cpu_root / "run")]
    cpu_stdout = logs_dir / "mdrun_cpu.stdout"
    cpu_stderr = logs_dir / "mdrun_cpu.stderr"
    run_command(cpu_argv, cwd=REPO_ROOT, env=cpu_env, stdout_path=cpu_stdout, stderr_path=cpu_stderr)
    commands.append(
        command_record(
            "mdrun_cpu",
            cpu_argv,
            cwd=REPO_ROOT,
            env_overrides=env_delta(cpu_env, os.environ),
            stdout_path=cpu_stdout,
            stderr_path=cpu_stderr,
        )
    )

    gpu_env = trace_env(args, gpu_root)
    gpu_argv = [str(gmx), "mdrun", *mdrun_args_gpu(args, tpr_path, gpu_root / "run")]
    gpu_stdout = logs_dir / "mdrun_gpu.stdout"
    gpu_stderr = logs_dir / "mdrun_gpu.stderr"
    gpu_result = run_command_allow_failure(
        gpu_argv,
        cwd=REPO_ROOT,
        env=gpu_env,
        stdout_path=gpu_stdout,
        stderr_path=gpu_stderr,
    )
    commands.append(
        command_record(
            "mdrun_gpu",
            gpu_argv,
            cwd=REPO_ROOT,
            env_overrides=env_delta(gpu_env, os.environ),
            stdout_path=gpu_stdout,
            stderr_path=gpu_stderr,
        )
    )

    version_text = capture_output([str(gmx), "--version"], cwd=REPO_ROOT)
    reference_events = reference_event_trace(args.outer_steps)
    cpu_outputs = collect_run_outputs(gmx, cpu_root / "run", cpu_root)

    result: dict[str, object] = {
        "schema_version": 1,
        "system_id": system_id,
        "scaffold_manifest": str(Path(args.scaffold_manifest).resolve()),
        "status": "FAIL" if gpu_result.returncode != 0 else "PASS",
        "gmx_binary": str(gmx),
        "gpu_support": parse_gpu_support(version_text),
        "precision_mode": parse_precision_mode(version_text),
        "layout": parse_layout_report(gpu_stdout, gpu_stderr, args),
        "short_run_settings": {
            "outer_steps": args.outer_steps,
            "pair14_level": args.pair14_level,
            "ntmpi": args.ntmpi,
            "ntomp": args.ntomp,
            "dlb": "no",
            "cpu_shape": "nb cpu / bonded cpu / pme cpu / update cpu",
            "gpu_shape": "nb gpu / bonded gpu / pme gpu / update gpu",
        },
        "cpu_reference_event_check": compare_event_trace(cpu_outputs["events"], reference_events),
        "cpu_energy_summary": energy_summary(system_id, args.outer_steps, args.pair14_level, cpu_outputs["energy_frames"]),
        "gpu_run": {
            "returncode": gpu_result.returncode,
            "stdout": str(gpu_stdout),
            "stderr": str(gpu_stderr),
        },
    }

    if gpu_result.returncode == 0:
        gpu_outputs = collect_run_outputs(gmx, gpu_root / "run", gpu_root)
        result["gpu_reference_event_check"] = compare_event_trace(gpu_outputs["events"], reference_events)
        result["cpu_vs_gpu_event_comparison"] = compare_event_trace(gpu_outputs["events"], cpu_outputs["events"])
        result["cpu_vs_gpu_energy_comparison"] = compare_energy_frames(
            gpu_outputs["energy_frames"], cpu_outputs["energy_frames"]
        )
        result["cpu_vs_gpu_total_force_comparison"] = compare_total_force_entries(
            gpu_outputs["total_force"]["per_step_totals"], cpu_outputs["total_force"]["per_step_totals"]
        )
        result["cpu_vs_gpu_per_level_force_comparison"] = compare_per_level_force_entries(
            gpu_outputs["per_level_force"]["entries"], cpu_outputs["per_level_force"]["entries"]
        )
        result["gpu_energy_summary"] = energy_summary(
            system_id, args.outer_steps, args.pair14_level, gpu_outputs["energy_frames"]
        )
        event_ok = bool(result["gpu_reference_event_check"]["matches"]) and bool(
            result["cpu_vs_gpu_event_comparison"]["matches"]
        )
        result["status"] = "PASS" if event_ok else "FAIL"
    else:
        result["failure_reason"] = "GPU exact-r-RESPA run failed during bring-up."

    dump_path = summaries_dir / "bringup_result.json"
    dump_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_commands_script(out_root / "run_commands.sh", commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
