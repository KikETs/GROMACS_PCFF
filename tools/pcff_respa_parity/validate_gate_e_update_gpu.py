from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from freeze_gate_a_oracle import (
    DEFAULT_GMX,
    LEVEL_FACTORS,
    base_env,
    capture_output,
    command_record,
    env_delta,
    parse_event_trace,
    parse_trr_dump,
    restart_summary,
    write_commands_script,
    write_text,
)
from validate_gate_b_nb_gpu import (
    assess_energy_display_resolution,
    compare_energy_frames,
    compare_event_trace,
    compare_per_level_force_entries,
    compare_total_force_entries,
    estimate_noise_floor,
    extract_virial_deltas,
    max_abs_delta_for_terms,
    read_gro_atom_count,
    trace_env_for_run,
)
from validate_gate_c_nb_bonded_gpu import (
    DEFAULT_GATE_A_MANIFEST,
    GPU_REPEAT_COUNT,
    SYSTEMS,
    assess_gate_c_system,
    build_class2_trace_bucket_assessments,
    build_cpu_correction_trace_bucket_assessments,
    build_gate_a_term_coverage,
    build_per_term_comparison_rows,
    capture_optional_output,
    first_term_issue,
    load_json,
    load_run_outputs,
    maybe_build,
    parse_gpu_support,
    parse_precision_mode,
    parse_tpr_inventory,
    run_command_allow_failure,
)
from validate_gate_d_nb_bonded_pme_gpu import (
    RECIPROCAL_FORCE_PASS_STATUSES,
    build_reciprocal_ownership_summary,
    characterize_reciprocal_force_difference,
    compare_force_component_rows,
    estimate_force_component_roundoff_proxy,
    estimate_force_component_noise_floor,
    load_fft_backend_arithmetic_evidence,
    parse_force_component_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_D_MANIFEST = (
    REPO_ROOT / "tests" / "reference_results" / "gate_d_nb_bonded_pme_gpu_validation" / "gate_d_manifest.json"
)
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_e_update_gpu_validation"
GPU_REPRODUCIBILITY_NOTE = (
    "Binary reproducibility (-reprod) is not enabled because GROMACS rejects -nb gpu together with -reprod."
)
TRR_DUMP_COMPONENT_RESOLUTION = 1.0e-6
RESTART_TOLERANCES = {
    "potential_abs_delta_kj_mol": 1e-6,
    "total_abs_delta_kj_mol": 1e-6,
    "max_coordinate_abs_delta_nm": TRR_DUMP_COMPONENT_RESOLUTION * 1.1,
    "max_velocity_abs_delta_nm_ps": TRR_DUMP_COMPONENT_RESOLUTION * 1.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Gate E for standalone exact r-RESPA with nb, bonded, PME, and update on GPU."
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument(
        "--gate-a-manifest",
        default=str(DEFAULT_GATE_A_MANIFEST),
        help="Path to the frozen Gate A CPU oracle manifest.",
    )
    parser.add_argument(
        "--gate-d-manifest",
        default=str(DEFAULT_GATE_D_MANIFEST),
        help="Path to the Gate D manifest used as the upstream evidence gate.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--build-target", default="gmx", help="CMake target to build.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step.")
    parser.add_argument("--build-dir", default=None, help="Optional explicit build directory.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks for mdrun.")
    parser.add_argument("--ntomp", type=int, default=1, help="OpenMP threads for mdrun.")
    parser.add_argument("--npme", type=int, default=None, help="Optional explicit -npme value.")
    parser.add_argument("--outer-steps", type=int, default=5, help="Number of exact r-RESPA outer steps.")
    parser.add_argument(
        "--gpu-repeats",
        type=int,
        default=GPU_REPEAT_COUNT,
        help="Number of repeated GPU runs for noise-floor estimation when the path executes.",
    )
    return parser.parse_args()


def load_optional_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return load_json(path)


def validate_gate_chain(gate_a_manifest: dict[str, object], gate_d_manifest: dict[str, object] | None) -> None:
    if gate_a_manifest.get("status") != "PASS":
        raise ValueError("Gate A manifest is not PASS; Gate E cannot use it as a frozen oracle.")
    if gate_d_manifest is None:
        raise ValueError("Gate D manifest is missing; Gate E must chain from a passed Gate D.")


def mdrun_args_gate_e(
    args: argparse.Namespace,
    tpr_path: Path,
    deffnm: Path,
    *,
    nsteps: int | None = None,
    cpi: Path | None = None,
) -> list[str]:
    if args.ntmpi != 1:
        raise ValueError("Gate E canonical validation is single-rank only; -ntmpi must remain 1.")
    if args.npme is not None:
        raise ValueError("Gate E canonical validation is single-rank only; -npme must be omitted.")

    result = [
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
    if nsteps is not None:
        result.extend(["-nsteps", str(nsteps)])
    if cpi is not None:
        result.extend(["-cpi", str(cpi)])
    return result


def gate_e_trace_env(args: argparse.Namespace, run_root: Path, atom_count: int) -> dict[str, str]:
    env = trace_env_for_run(args, run_root, atom_count=atom_count)
    env["GMX_PCFF_RESPA_M2M_TRACE_DIR"] = str(run_root / "m2m_trace")
    env["GMX_PCFF_RESPA_M2M_MODE"] = "gate_e_update_gpu"
    return env


def parse_layout_report(stdout_path: Path, stderr_path: Path, args: argparse.Namespace) -> dict[str, object]:
    combined_text = ""
    if stdout_path.exists():
        combined_text += stdout_path.read_text(encoding="utf-8", errors="replace")
    if stderr_path.exists():
        combined_text += "\n" + stderr_path.read_text(encoding="utf-8", errors="replace")

    lines = combined_text.splitlines()
    mapping_header = None
    mapping_lines: list[str] = []
    for index, line in enumerate(lines):
        if line.startswith("Mapping of GPU IDs to the "):
            mapping_header = line.strip()
            follow_index = index + 1
            while follow_index < len(lines) and lines[follow_index].startswith("  "):
                mapping_lines.append(lines[follow_index].strip())
                follow_index += 1
            break

    pp_line = next((line.strip() for line in lines if line.startswith("PP tasks will ")), None)
    pme_line = next((line.strip() for line in lines if line.startswith("PME tasks will ")), None)
    update_line = next((line.strip() for line in lines if "update and constrain coordinates" in line), None)

    return {
        "rank_layout": "single-rank colocated PP+PME tasks on rank 0",
        "single_rank_explicit": args.ntmpi == 1,
        "ntmpi": args.ntmpi,
        "ntomp": args.ntomp,
        "npme_flag_used": args.npme is not None,
        "npme_requested": args.npme,
        "npme_equals_one": args.npme == 1,
        "gpu_mapping_header": mapping_header,
        "gpu_mapping_lines": mapping_lines,
        "pp_task_line": pp_line,
        "pme_task_line": pme_line,
        "update_task_line": update_line,
        "pme_gpu_enabled": pme_line == "PME tasks will do all aspects on the GPU",
        "update_gpu_enabled": update_line == "PP task will update and constrain coordinates on the GPU",
        "mapping_explicit": mapping_header is not None and bool(mapping_lines),
        "rank_layout_sensitivity_tested": False,
        "rank_layout_sensitivity_note": "Gate E is restricted to the canonical single-rank layout; no alternate layout tested.",
    }


def parse_mdp_settings(path: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        settings[key.strip().lower()] = value.strip()
    return settings


def build_unsupported_feature_assessment(mdp_settings: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    checks = [
        {
            "field": "single_rank",
            "expected": "ntmpi=1",
            "actual": f"ntmpi={args.ntmpi}",
            "passes": args.ntmpi == 1,
        },
        {
            "field": "npme_unused",
            "expected": "npme omitted",
            "actual": "omitted" if args.npme is None else f"npme={args.npme}",
            "passes": args.npme is None,
        },
        {
            "field": "integrator",
            "expected": "md-vv",
            "actual": mdp_settings.get("integrator"),
            "passes": mdp_settings.get("integrator") == "md-vv",
        },
        {
            "field": "exact_respa",
            "expected": "yes",
            "actual": mdp_settings.get("exact-respa"),
            "passes": mdp_settings.get("exact-respa") == "yes",
        },
        {
            "field": "constraints",
            "expected": "none",
            "actual": mdp_settings.get("constraints"),
            "passes": mdp_settings.get("constraints") == "none",
        },
        {
            "field": "tcoupl",
            "expected": "no",
            "actual": mdp_settings.get("tcoupl"),
            "passes": mdp_settings.get("tcoupl") == "no",
        },
        {
            "field": "pcoupl",
            "expected": "no",
            "actual": mdp_settings.get("pcoupl"),
            "passes": mdp_settings.get("pcoupl") == "no",
        },
        {
            "field": "comm-mode",
            "expected": "none",
            "actual": mdp_settings.get("comm-mode"),
            "passes": mdp_settings.get("comm-mode") == "none",
        },
        {
            "field": "dlb",
            "expected": "no",
            "actual": "no",
            "passes": True,
        },
    ]
    return {
        "checks": checks,
        "all_pass": all(check["passes"] for check in checks),
        "box_relevant": mdp_settings.get("pcoupl") != "no",
        "box_reason": (
            "Pressure coupling is off, so the box is fixed and box state continuity is not a differentiating Gate E signal."
        ),
    }


def extract_failure_markers(stderr_path: Path, stdout_path: Path) -> list[str]:
    markers = []
    combined_text = ""
    if stdout_path.exists():
        combined_text += stdout_path.read_text(encoding="utf-8", errors="replace")
    if stderr_path.exists():
        combined_text += "\n" + stderr_path.read_text(encoding="utf-8", errors="replace")
    for marker in (
        "Standalone exact r-RESPA is not supported on GPUs.",
        "Standalone exact r-RESPA GPU update is only supported for single-rank, unconstrained md-vv with GPU non-bonded and GPU PME.",
        "Only the md integrator and standalone exact r-RESPA md-vv are supported.",
        "Offload features enabled require X/F buffer ops",
        "Trying to consume an event before marking it or after fully consuming it",
    ):
        if marker in combined_text:
            markers.append(marker)
    return markers


def load_run_outputs_gate_e(gmx: Path, run_root: Path, run_id: str) -> dict[str, object]:
    outputs = load_run_outputs(gmx, run_root, run_id)
    trr_path = run_root / "exact_full.trr"
    trr_dump = capture_output([str(gmx), "dump", "-f", str(trr_path)], cwd=REPO_ROOT)
    force_component_path = run_root / "m2p_trace" / "step0_force_component_trace.txt"
    outputs["full_outputs"]["trr"] = str(trr_path)
    outputs["full_outputs"]["force_component_trace_txt"] = str(force_component_path)
    outputs["full_outputs"]["m2m_trace_dir"] = str(run_root / "m2m_trace")
    outputs["full_outputs"]["m2p_force_component_trace_txt"] = str(force_component_path)
    outputs["trr_frames"] = parse_trr_dump(trr_dump)
    outputs["force_component_trace"] = parse_force_component_trace(force_component_path)
    return outputs


def collect_gpu_run(
    *,
    args: argparse.Namespace,
    gmx: Path,
    gate_a_tpr: Path,
    system_root: Path,
    run_label: str,
    trace_atom_count: int,
    commands: list[dict[str, object]],
) -> dict[str, object]:
    run_root = system_root / run_label
    logs_dir = system_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    environment = gate_e_trace_env(args, run_root, trace_atom_count)
    run_env_delta = env_delta(environment, os.environ)
    deffnm = run_root / "exact_full"
    mdrun = [str(gmx), "mdrun", *mdrun_args_gate_e(args, gate_a_tpr, deffnm)]
    stdout_path = logs_dir / f"{run_label}.stdout"
    stderr_path = logs_dir / f"{run_label}.stderr"
    result = run_command_allow_failure(
        mdrun,
        cwd=REPO_ROOT,
        env=environment,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    commands.append(
        command_record(
            run_label,
            mdrun,
            cwd=REPO_ROOT,
            env_overrides=run_env_delta,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    )

    run_data = {
        "run_id": run_label,
        "artifact_root": str(run_root),
        "argv": mdrun,
        "env_overrides": run_env_delta,
        "returncode": result.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "failure_markers": extract_failure_markers(stderr_path, stdout_path),
        "layout_report": parse_layout_report(stdout_path, stderr_path, args),
    }
    if result.returncode == 0:
        run_data.update(load_run_outputs_gate_e(gmx, run_root, run_label))
    return run_data


def load_restart_result(gmx: Path, deffnm: Path) -> dict[str, object]:
    energy_dump = capture_output([str(gmx), "dump", "-e", str(deffnm.with_suffix(".edr"))], cwd=REPO_ROOT)
    trr_dump = capture_output([str(gmx), "dump", "-f", str(deffnm.with_suffix(".trr"))], cwd=REPO_ROOT)
    from validate_gate_c_nb_bonded_gpu import parse_energy_dump  # local import to keep the module graph simple

    return {
        "deffnm": str(deffnm),
        "edr": str(deffnm.with_suffix(".edr")),
        "trr": str(deffnm.with_suffix(".trr")),
        "cpt": str(deffnm.with_suffix(".cpt")),
        "energy_frames": parse_energy_dump(energy_dump),
        "trr_frames": parse_trr_dump(trr_dump),
    }


def collect_restart_continuity(
    *,
    args: argparse.Namespace,
    gmx: Path,
    gate_a_tpr: Path,
    system_root: Path,
    commands: list[dict[str, object]],
) -> dict[str, object]:
    logs_dir = system_root / "logs"
    restart_full_dir = system_root / "restart_full"
    restart_split_dir = system_root / "restart_split"
    restart_full_dir.mkdir(parents=True, exist_ok=True)
    restart_split_dir.mkdir(parents=True, exist_ok=True)

    environment = base_env(args)
    run_env_delta = env_delta(environment, os.environ)
    split_outer_steps = max(1, args.outer_steps // 2)
    split_steps = split_outer_steps * LEVEL_FACTORS[-1]

    restart_full_deffnm = restart_full_dir / "exact_full"
    restart_full_command = [str(gmx), "mdrun", *mdrun_args_gate_e(args, gate_a_tpr, restart_full_deffnm)]
    restart_full_stdout = logs_dir / "restart_full.stdout"
    restart_full_stderr = logs_dir / "restart_full.stderr"
    restart_full_result = run_command_allow_failure(
        restart_full_command,
        cwd=REPO_ROOT,
        env=environment,
        stdout_path=restart_full_stdout,
        stderr_path=restart_full_stderr,
    )
    commands.append(
        command_record(
            "restart_full",
            restart_full_command,
            cwd=REPO_ROOT,
            env_overrides=run_env_delta,
            stdout_path=restart_full_stdout,
            stderr_path=restart_full_stderr,
        )
    )

    restart_split_deffnm = restart_split_dir / "exact_split"
    restart_split_first_command = [
        str(gmx),
        "mdrun",
        *mdrun_args_gate_e(args, gate_a_tpr, restart_split_deffnm, nsteps=split_steps),
    ]
    restart_split_first_stdout = logs_dir / "restart_split_first.stdout"
    restart_split_first_stderr = logs_dir / "restart_split_first.stderr"
    restart_split_first_result = run_command_allow_failure(
        restart_split_first_command,
        cwd=REPO_ROOT,
        env=environment,
        stdout_path=restart_split_first_stdout,
        stderr_path=restart_split_first_stderr,
    )
    commands.append(
        command_record(
            "restart_split_first",
            restart_split_first_command,
            cwd=REPO_ROOT,
            env_overrides=run_env_delta,
            stdout_path=restart_split_first_stdout,
            stderr_path=restart_split_first_stderr,
        )
    )

    restart_split_second_stdout = logs_dir / "restart_split_second.stdout"
    restart_split_second_stderr = logs_dir / "restart_split_second.stderr"
    restart_split_second_command = [
        str(gmx),
        "mdrun",
        *mdrun_args_gate_e(
            args, gate_a_tpr, restart_split_deffnm, cpi=restart_split_deffnm.with_suffix(".cpt")
        ),
    ]
    restart_split_second_result = None
    if restart_split_first_result.returncode == 0:
        restart_split_second_result = run_command_allow_failure(
            restart_split_second_command,
            cwd=REPO_ROOT,
            env=environment,
            stdout_path=restart_split_second_stdout,
            stderr_path=restart_split_second_stderr,
        )
        commands.append(
            command_record(
                "restart_split_second",
                restart_split_second_command,
                cwd=REPO_ROOT,
                env_overrides=run_env_delta,
                stdout_path=restart_split_second_stdout,
                stderr_path=restart_split_second_stderr,
            )
        )

    results = {
        "split_outer_steps": split_outer_steps,
        "split_steps": split_steps,
        "runs": {
            "full": {
                "argv": restart_full_command,
                "stdout": str(restart_full_stdout),
                "stderr": str(restart_full_stderr),
                "returncode": restart_full_result.returncode,
                "failure_markers": extract_failure_markers(restart_full_stderr, restart_full_stdout),
            },
            "split_first": {
                "argv": restart_split_first_command,
                "stdout": str(restart_split_first_stdout),
                "stderr": str(restart_split_first_stderr),
                "returncode": restart_split_first_result.returncode,
                "failure_markers": extract_failure_markers(restart_split_first_stderr, restart_split_first_stdout),
            },
            "split_second": {
                "argv": restart_split_second_command,
                "stdout": str(restart_split_second_stdout),
                "stderr": str(restart_split_second_stderr),
                "returncode": None if restart_split_second_result is None else restart_split_second_result.returncode,
                "failure_markers": (
                    []
                    if restart_split_second_result is None
                    else extract_failure_markers(restart_split_second_stderr, restart_split_second_stdout)
                ),
                "executed": restart_split_second_result is not None,
            },
        },
    }

    if (
        restart_full_result.returncode == 0
        and restart_split_first_result.returncode == 0
        and restart_split_second_result is not None
        and restart_split_second_result.returncode == 0
    ):
        full_outputs = load_restart_result(gmx, restart_full_deffnm)
        split_outputs = load_restart_result(gmx, restart_split_deffnm)
        results["outputs"] = {"full": full_outputs, "split": split_outputs}
    return results


def compare_trr_frames(actual_frames: list[dict[str, object]], expected_frames: list[dict[str, object]]) -> dict[str, object]:
    first_coverage_mismatch = None
    max_coordinate_delta = 0.0
    max_velocity_delta = 0.0
    first_nonzero_coordinate = None
    first_nonzero_velocity = None
    frame_rows = []
    frame_count = min(len(actual_frames), len(expected_frames))
    for index in range(frame_count):
        actual_frame = actual_frames[index]
        expected_frame = expected_frames[index]
        if (
            int(actual_frame["step"]) != int(expected_frame["step"])
            or abs(float(actual_frame["time_ps"]) - float(expected_frame["time_ps"])) > 1e-12
            or int(actual_frame["natoms"]) != int(expected_frame["natoms"])
        ):
            first_coverage_mismatch = {
                "frame_index": index,
                "expected_step": int(expected_frame["step"]),
                "actual_step": int(actual_frame["step"]),
                "expected_time_ps": float(expected_frame["time_ps"]),
                "actual_time_ps": float(actual_frame["time_ps"]),
                "expected_natoms": int(expected_frame["natoms"]),
                "actual_natoms": int(actual_frame["natoms"]),
            }
            break

        frame_coordinate_delta = 0.0
        frame_velocity_delta = 0.0
        for actual_coord, expected_coord in zip(actual_frame["coordinates"], expected_frame["coordinates"]):
            for actual_value, expected_value in zip(actual_coord, expected_coord):
                delta = abs(float(actual_value) - float(expected_value))
                frame_coordinate_delta = max(frame_coordinate_delta, delta)
                max_coordinate_delta = max(max_coordinate_delta, delta)
        for actual_velocity, expected_velocity in zip(actual_frame["velocities"], expected_frame["velocities"]):
            for actual_value, expected_value in zip(actual_velocity, expected_velocity):
                delta = abs(float(actual_value) - float(expected_value))
                frame_velocity_delta = max(frame_velocity_delta, delta)
                max_velocity_delta = max(max_velocity_delta, delta)

        if first_nonzero_coordinate is None and frame_coordinate_delta != 0.0:
            first_nonzero_coordinate = {
                "frame_index": index,
                "step": int(actual_frame["step"]),
                "delta_nm": frame_coordinate_delta,
            }
        if first_nonzero_velocity is None and frame_velocity_delta != 0.0:
            first_nonzero_velocity = {
                "frame_index": index,
                "step": int(actual_frame["step"]),
                "delta_nm_ps": frame_velocity_delta,
            }
        frame_rows.append(
            {
                "frame_index": index,
                "step": int(actual_frame["step"]),
                "time_ps": float(actual_frame["time_ps"]),
                "coordinate_max_abs_delta_nm": frame_coordinate_delta,
                "velocity_max_abs_delta_nm_ps": frame_velocity_delta,
            }
        )

    if first_coverage_mismatch is None and len(actual_frames) != len(expected_frames):
        first_coverage_mismatch = {
            "actual_frame_count": len(actual_frames),
            "expected_frame_count": len(expected_frames),
        }

    return {
        "available": bool(actual_frames) and bool(expected_frames),
        "coverage_matches": first_coverage_mismatch is None,
        "actual_frame_count": len(actual_frames),
        "expected_frame_count": len(expected_frames),
        "first_coverage_mismatch": first_coverage_mismatch,
        "max_coordinate_abs_delta_nm": max_coordinate_delta,
        "max_velocity_abs_delta_nm_ps": max_velocity_delta,
        "first_nonzero_coordinate_delta": first_nonzero_coordinate,
        "first_nonzero_velocity_delta": first_nonzero_velocity,
        "frames": frame_rows,
    }


def unavailable_state_comparison(reference_label: str, reason: str) -> dict[str, object]:
    return {
        "available": False,
        "reference_label": reference_label,
        "reason": reason,
        "coverage_matches": False,
        "actual_frame_count": 0,
        "expected_frame_count": 0,
        "first_coverage_mismatch": None,
        "max_coordinate_abs_delta_nm": None,
        "max_velocity_abs_delta_nm_ps": None,
        "first_nonzero_coordinate_delta": None,
        "first_nonzero_velocity_delta": None,
        "frames": [],
    }


def estimate_state_noise_floor(successful_runs: list[dict[str, object]]) -> dict[str, object]:
    if len(successful_runs) < 2:
        return {
            "available": False,
            "reason": "Fewer than two successful GPU runs are available for state-noise estimation.",
            "successful_run_count": len(successful_runs),
        }

    reference = successful_runs[0]
    max_coordinate_delta = 0.0
    max_velocity_delta = 0.0
    compared_runs = []
    for run in successful_runs[1:]:
        comparison = compare_trr_frames(run["trr_frames"], reference["trr_frames"])
        if not comparison["coverage_matches"]:
            return {
                "available": False,
                "reason": "Repeated GPU runs changed trajectory frame coverage.",
                "successful_run_count": len(successful_runs),
                "first_coverage_mismatch": comparison["first_coverage_mismatch"],
            }
        max_coordinate_delta = max(max_coordinate_delta, float(comparison["max_coordinate_abs_delta_nm"]))
        max_velocity_delta = max(max_velocity_delta, float(comparison["max_velocity_abs_delta_nm_ps"]))
        compared_runs.append(
            {
                "run_id": run["run_id"],
                "max_coordinate_abs_delta_nm": comparison["max_coordinate_abs_delta_nm"],
                "max_velocity_abs_delta_nm_ps": comparison["max_velocity_abs_delta_nm_ps"],
            }
        )

    return {
        "available": True,
        "successful_run_count": len(successful_runs),
        "reference_gpu_run_id": reference["run_id"],
        "max_coordinate_abs_delta_nm": max_coordinate_delta,
        "max_velocity_abs_delta_nm_ps": max_velocity_delta,
        "compared_runs": compared_runs,
    }


def characterize_state_difference(
    state_comparison: dict[str, object],
    state_noise_floor: dict[str, object],
    restart_validation: dict[str, object],
    *,
    reference_label: str,
) -> dict[str, object]:
    if not state_comparison.get("available"):
        return {
            "status": "unavailable",
            "note": f"Step-to-step position/velocity continuity against {reference_label} was not available.",
            "inference": False,
        }
    if not state_comparison.get("coverage_matches"):
        return {
            "status": "coverage_mismatch",
            "note": f"Trajectory frame coverage or frame identity diverged from {reference_label}.",
            "first_coverage_mismatch": state_comparison.get("first_coverage_mismatch"),
            "inference": False,
        }
    if (
        float(state_comparison.get("max_coordinate_abs_delta_nm", 0.0)) == 0.0
        and float(state_comparison.get("max_velocity_abs_delta_nm_ps", 0.0)) == 0.0
    ):
        return {
            "status": "exact_match",
            "note": f"Trajectory frames match {reference_label} exactly at the dump cadence.",
            "inference": False,
        }

    if state_noise_floor.get("available"):
        within_noise = (
            float(state_comparison.get("max_coordinate_abs_delta_nm", 0.0))
            <= float(state_noise_floor.get("max_coordinate_abs_delta_nm", 0.0)) + 1e-12
            and float(state_comparison.get("max_velocity_abs_delta_nm_ps", 0.0))
            <= float(state_noise_floor.get("max_velocity_abs_delta_nm_ps", 0.0)) + 1e-12
        )
        if within_noise:
            return {
                "status": "within_gpu_noise_floor",
                "note": f"Position/velocity deltas versus {reference_label} stay within repeated GPU state noise.",
                "gpu_noise_floor": state_noise_floor,
                "inference": False,
            }

    return {
        "status": "systematic_cpu_gpu_state_difference",
        "max_coordinate_abs_delta_nm": state_comparison.get("max_coordinate_abs_delta_nm"),
        "max_velocity_abs_delta_nm_ps": state_comparison.get("max_velocity_abs_delta_nm_ps"),
        "gpu_noise_floor": state_noise_floor if state_noise_floor.get("available") else None,
        "restart_validation_status": restart_validation.get("status"),
        "reference_label": reference_label,
        "note": (
            "추측입니다. Update gpu keeps event order, force ownership, and restart continuity intact, "
            f"but the trajectory diverges from {reference_label} more than repeated GPU state noise. "
            "That points to a systematic CPU-vs-GPU integration/update path difference rather than "
            "a restart-boundary break."
        ),
        "inference": True,
    }


def max_explicit_energy_delta(per_term_rows: list[dict[str, object]]) -> dict[str, object]:
    explicit_rows = [
        row
        for row in per_term_rows
        if row["comparison_source"] == "energy_dump"
        and row["coverage_status"] == "EXPLICIT"
        and row["max_abs_delta_kj_mol"] is not None
    ]
    if not explicit_rows:
        return {"available": False}

    worst_row = max(explicit_rows, key=lambda row: float(row["max_abs_delta_kj_mol"]))
    return {
        "available": True,
        "bucket": worst_row["bucket"],
        "terms": worst_row["terms"],
        "max_abs_delta_kj_mol": worst_row["max_abs_delta_kj_mol"],
        "first_nonzero_delta": worst_row["first_nonzero_delta"],
    }


def characterize_explicit_energy_difference(
    explicit_term_energy_display_assessment: dict[str, object],
    per_term_rows: list[dict[str, object]],
    gpu_noise_floor: dict[str, object],
) -> dict[str, object]:
    if explicit_term_energy_display_assessment.get("within_bounds"):
        return {
            "status": "within_display_resolution",
            "note": "Explicit energy-dump terms stay within the gmx dump display-resolution bound.",
            "inference": False,
        }

    delta_summary = max_explicit_energy_delta(per_term_rows)
    if delta_summary.get("available") and gpu_noise_floor.get("available"):
        if float(delta_summary["max_abs_delta_kj_mol"]) <= float(gpu_noise_floor["energy_max_abs_delta_kj_mol"]) + 1e-12:
            return {
                "status": "within_gpu_noise_floor",
                "delta_summary": delta_summary,
                "gpu_noise_floor": gpu_noise_floor,
                "note": "Explicit energy-dump term deltas exceed display resolution but stay within repeated GPU energy noise.",
                "inference": False,
            }

    return {
        "status": "systematic_cpu_gpu_energy_difference",
        "delta_summary": delta_summary,
        "gpu_noise_floor": gpu_noise_floor if gpu_noise_floor.get("available") else None,
        "note": (
            "추측입니다. Explicit energy-dump terms exceed both display resolution and repeated GPU energy noise, "
            "while event order and restart semantics remain intact. This points to a systematic CPU-vs-GPU "
            "update-path difference rather than a trace hole."
        ),
        "inference": True,
    }


def characterize_restart_continuity(
    restart_validation: dict[str, object], state_noise_floor: dict[str, object]
) -> dict[str, object]:
    if not restart_validation.get("available"):
        return {
            "status": "unavailable",
            "note": "Restart continuity evidence is unavailable.",
            "inference": False,
        }
    if restart_validation.get("status") == "PASS":
        effective_tolerances = restart_validation.get("effective_tolerances", RESTART_TOLERANCES)
        used_noise_relaxed_energy_tolerance = any(
            float(effective_tolerances.get(field, RESTART_TOLERANCES[field])) > float(RESTART_TOLERANCES[field])
            for field in ("potential_abs_delta_kj_mol", "total_abs_delta_kj_mol")
        )
        return {
            "status": "pass",
            "note": (
                "Restart continuity satisfies the pinned Gate E tolerances."
                if not used_noise_relaxed_energy_tolerance
                else "Restart continuity satisfies the effective Gate E tolerances after repeated GPU energy-noise inflation for restart energy fields."
            ),
            "inference": False,
        }

    if state_noise_floor.get("available"):
        noise_by_field = {
            "max_coordinate_abs_delta_nm": float(state_noise_floor.get("max_coordinate_abs_delta_nm", 0.0)),
            "max_velocity_abs_delta_nm_ps": float(state_noise_floor.get("max_velocity_abs_delta_nm_ps", 0.0)),
        }
        if restart_validation.get("failures"):
            all_failures_within_noise = True
            for failure in restart_validation["failures"]:
                field = failure["field"]
                if field not in noise_by_field or float(failure["value"]) > noise_by_field[field] + 1e-12:
                    all_failures_within_noise = False
                    break
            if all_failures_within_noise:
                return {
                    "status": "within_gpu_noise_floor",
                    "failures": restart_validation["failures"],
                    "state_noise_floor": state_noise_floor,
                    "note": "Restart continuity exceeds the CPU-oracle epsilon but stays within repeated GPU state noise.",
                    "inference": False,
                }

    return {
        "status": "fail",
        "failures": restart_validation.get("failures", []),
        "note": "Restart continuity exceeded the pinned Gate E tolerances by more than repeated GPU state noise.",
        "inference": False,
    }


def first_blocking_term_issue_gate_e(
    per_term_rows: list[dict[str, object]],
    explicit_energy_characterization: dict[str, object],
) -> dict[str, object] | None:
    for row in per_term_rows:
        mismatch_category = row["mismatch_category"]
        if mismatch_category is None:
            continue
        if mismatch_category in {"trace insufficiency", "ownership"}:
            return {
                "bucket": row["bucket"],
                "comparison_status": row["comparison_status"],
                "mismatch_category": mismatch_category,
                "note": row["note"],
                "terms": row["terms"],
                "first_nonzero_delta": row["first_nonzero_delta"],
            }
        if (
            mismatch_category == "ownership_or_reduction"
            and explicit_energy_characterization["status"] == "systematic_cpu_gpu_energy_difference"
        ):
            return {
                "bucket": row["bucket"],
                "comparison_status": row["comparison_status"],
                "mismatch_category": mismatch_category,
                "note": row["note"],
                "terms": row["terms"],
                "first_nonzero_delta": row["first_nonzero_delta"],
            }
    return None


def blocking_ambiguous_terms_gate_e(
    per_term_rows: list[dict[str, object]],
    explicit_energy_characterization: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    for row in per_term_rows:
        mismatch_category = row["mismatch_category"]
        if mismatch_category is None:
            continue
        if mismatch_category in {"trace insufficiency", "ownership"} or (
            mismatch_category == "ownership_or_reduction"
            and explicit_energy_characterization["status"] == "systematic_cpu_gpu_energy_difference"
        ):
            rows.append(
                {
                    "bucket": row["bucket"],
                    "comparison_status": row["comparison_status"],
                    "mismatch_category": mismatch_category,
                    "note": row["note"],
                    "terms": row["terms"],
                }
            )
    return rows


def build_restart_validation(
    restart_results: dict[str, object],
    gate_a_restart_summary: dict[str, object],
    gpu_noise_floor: dict[str, object],
) -> dict[str, object]:
    if "outputs" not in restart_results:
        return {
            "available": False,
            "status": "BLOCKER",
            "reason": "Gate E restart runs did not complete successfully.",
            "runs": restart_results["runs"],
        }

    summary = restart_summary(
        "gate_e",
        int(restart_results["split_outer_steps"]),
        restart_results["outputs"]["full"]["energy_frames"],
        restart_results["outputs"]["split"]["energy_frames"],
        restart_results["outputs"]["full"]["trr_frames"],
        restart_results["outputs"]["split"]["trr_frames"],
    )
    effective_tolerances = dict(RESTART_TOLERANCES)
    if gpu_noise_floor.get("available"):
        energy_noise_floor = float(gpu_noise_floor.get("energy_max_abs_delta_kj_mol", 0.0))
        effective_tolerances["potential_abs_delta_kj_mol"] = max(
            float(effective_tolerances["potential_abs_delta_kj_mol"]), energy_noise_floor
        )
        effective_tolerances["total_abs_delta_kj_mol"] = max(
            float(effective_tolerances["total_abs_delta_kj_mol"]), energy_noise_floor
        )
    failures = []
    first_failure = None
    for field, tolerance in effective_tolerances.items():
        value = float(summary[field])
        if value > tolerance:
            failure = {"field": field, "value": value, "tolerance": tolerance}
            failures.append(failure)
            if first_failure is None:
                first_failure = failure

    comparison_to_gate_a = {
        field: {
            "gate_a": gate_a_restart_summary.get(field),
            "gate_e": summary.get(field),
            "delta": float(summary.get(field, 0.0)) - float(gate_a_restart_summary.get(field, 0.0)),
        }
        for field in RESTART_TOLERANCES
    }
    return {
        "available": True,
        "status": "PASS" if not failures else "FAIL",
        "summary": summary,
        "tolerances": RESTART_TOLERANCES,
        "effective_tolerances": effective_tolerances,
        "failures": failures,
        "first_failure": first_failure,
        "comparison_to_gate_a_restart_summary": comparison_to_gate_a,
        "runs": restart_results["runs"],
        "outputs": restart_results["outputs"],
    }


def summarize_direct_oracle_comparison_gate_e(
    *,
    event_order_comparison: dict[str, object],
    total_force_comparison: dict[str, object],
    per_level_force_comparison: dict[str, object],
    energy_comparison: dict[str, object],
    virial_comparison: dict[str, object],
    gpu_noise_floor: dict[str, object],
    per_term_rows: list[dict[str, object]],
    class2_trace_bucket_assessments: dict[str, dict[str, object]],
    cpu_correction_trace_bucket_assessments: dict[str, dict[str, object]],
    reciprocal_force_comparison: dict[str, object],
    reciprocal_force_noise_floor: dict[str, object],
    reciprocal_force_roundoff_proxy: dict[str, object],
    fft_backend_arithmetic_evidence: dict[str, object],
    state_comparison_to_gate_d: dict[str, object],
    state_comparison_to_gate_a: dict[str, object],
    state_noise_floor: dict[str, object],
    restart_validation: dict[str, object],
) -> dict[str, object]:
    return {
        "event_order": event_order_comparison,
        "total_force": total_force_comparison,
        "per_level_force_totals": per_level_force_comparison,
        "per_term_energies": {
            "max_abs_delta_kj_mol": energy_comparison["max_abs_delta_kj_mol"],
            "first_mismatch": energy_comparison["first_mismatch"],
            "rows": per_term_rows,
        },
        "virial_contributors": virial_comparison,
        "class2_subterm_trace": class2_trace_bucket_assessments,
        "cpu_correction_trace": cpu_correction_trace_bucket_assessments,
        "gpu_noise_floor": gpu_noise_floor,
        "reciprocal_force_trace": reciprocal_force_comparison,
        "reciprocal_force_noise_floor": reciprocal_force_noise_floor,
        "reciprocal_force_roundoff_proxy": reciprocal_force_roundoff_proxy,
        "fft_backend_arithmetic_evidence": fft_backend_arithmetic_evidence,
        "state_continuity_vs_gate_d": state_comparison_to_gate_d,
        "state_continuity_vs_gate_a": state_comparison_to_gate_a,
        "state_noise_floor": state_noise_floor,
        "restart_continuity": restart_validation,
    }


def assess_gate_e_system(
    *,
    main_run: dict[str, object],
    gate_d_manifest_status: str | None,
    gate_a_energy_frames: list[dict[str, object]],
    event_order_comparison: dict[str, object],
    total_force_comparison: dict[str, object],
    per_level_force_comparison: dict[str, object],
    energy_comparison: dict[str, object],
    virial_comparison: dict[str, object],
    gpu_noise_floor: dict[str, object],
    per_term_rows: list[dict[str, object]],
    reciprocal_force_comparison: dict[str, object],
    reciprocal_force_noise_floor: dict[str, object],
    reciprocal_force_roundoff_proxy: dict[str, object],
    fft_backend_arithmetic_evidence: dict[str, object],
    unsupported_feature_assessment: dict[str, object],
    restart_validation: dict[str, object],
    state_comparison_to_gate_d: dict[str, object],
    state_comparison_to_gate_a: dict[str, object],
    state_noise_floor: dict[str, object],
) -> dict[str, object]:
    base_assessment = assess_gate_c_system(
        main_run=main_run,
        gate_a_energy_frames=gate_a_energy_frames,
        event_order_comparison=event_order_comparison,
        total_force_comparison=total_force_comparison,
        per_level_force_comparison=per_level_force_comparison,
        energy_comparison=energy_comparison,
        virial_comparison=virial_comparison,
        gpu_noise_floor=gpu_noise_floor,
        per_term_rows=per_term_rows,
    )

    reasons = list(base_assessment.get("reasons", []))
    layout_report = main_run.get("layout_report", {})
    explicit_energy_characterization = characterize_explicit_energy_difference(
        base_assessment.get("explicit_term_energy_display_assessment", {}),
        per_term_rows,
        gpu_noise_floor,
    )
    reciprocal_force_characterization = characterize_reciprocal_force_difference(
        reciprocal_force_comparison,
        reciprocal_force_noise_floor,
        reciprocal_force_roundoff_proxy,
        fft_backend_arithmetic_evidence,
    )
    blocking_term_issue = first_blocking_term_issue_gate_e(per_term_rows, explicit_energy_characterization)
    restart_characterization = characterize_restart_continuity(restart_validation, state_noise_floor)
    if gate_d_manifest_status != "PASS":
        reasons.append("Gate D upstream evidence is not PASS; Gate E is contaminated by unresolved PME-level mechanics.")
    if not layout_report.get("single_rank_explicit"):
        reasons.append("Gate E is not running in the required single-rank layout.")
    if not layout_report.get("pme_gpu_enabled"):
        reasons.append("PME GPU execution is not explicit in the runtime layout report.")
    if not layout_report.get("update_gpu_enabled"):
        reasons.append("GPU update execution is not explicit in the runtime layout report.")
    if not layout_report.get("mapping_explicit"):
        reasons.append("GPU task mapping is not explicit in the runtime layout report.")
    if not unsupported_feature_assessment.get("all_pass"):
        reasons.append("Unsupported Gate E features were not explicitly absent in the frozen mdp settings.")
    if not virial_comparison.get("available"):
        reasons.append("Virial / pressure deltas were not available for Gate E.")
    if reciprocal_force_comparison["missing_in_actual"] or reciprocal_force_comparison["extra_in_actual"]:
        reasons.append("Reciprocal-force trace coverage diverged from the Gate A oracle.")
    elif reciprocal_force_comparison["first_mismatch"] is not None:
        reasons.append("Reciprocal-force trace availability diverged from the Gate A oracle.")
    elif reciprocal_force_characterization["status"] not in RECIPROCAL_FORCE_PASS_STATUSES:
        reasons.append(reciprocal_force_characterization["note"])
    if not restart_validation.get("available"):
        reasons.append("Restart continuity evidence is unavailable for Gate E.")
    elif restart_characterization["status"] == "fail":
        reasons.append("Restart continuity exceeded the pinned Gate E tolerances.")
    if not state_comparison_to_gate_d.get("available"):
        reasons.append("Step-to-step state continuity against Gate D is unavailable.")
    elif not state_comparison_to_gate_d.get("coverage_matches"):
        reasons.append("Trajectory frame coverage diverged from the Gate D baseline.")
    if not state_comparison_to_gate_a.get("available"):
        reasons.append("Step-to-step state continuity against Gate A is unavailable.")
    elif not state_comparison_to_gate_a.get("coverage_matches"):
        reasons.append("Trajectory frame coverage diverged from the Gate A oracle.")

    if explicit_energy_characterization["status"] != "systematic_cpu_gpu_energy_difference":
        reasons = [
            reason
            for reason in reasons
            if reason != "Explicit per-term energy deltas exceed the gmx dump display-resolution bound."
        ]

    if gate_d_manifest_status != "PASS":
        status = "BLOCKER"
    elif reasons or blocking_term_issue is not None:
        status = "FAIL" if main_run["returncode"] == 0 else "BLOCKER"
    else:
        status = "PASS" if main_run["returncode"] == 0 else "BLOCKER"

    state_characterization = characterize_state_difference(
        state_comparison_to_gate_d,
        state_noise_floor,
        restart_validation,
        reference_label="Gate D",
    )
    strongest_surviving_claim = (
        "Event order, per-level force totals, explicit per-term ownership, and restart-boundary behavior remain "
        "aligned with the frozen Gate A oracle, while the GPU-update trajectory stays consistent with the Gate D "
        "baseline up to a small characterized CPU-vs-GPU update drift."
    )
    if base_assessment.get("virial_characterization", {}).get("status") == "systematic_reduction_difference":
        strongest_surviving_claim += (
            " Aggregate virial drift is tracked separately as a watch field and is not used as the Gate E verdict."
        )
    if state_characterization["status"] == "systematic_cpu_gpu_state_difference":
        strongest_surviving_claim += (
            " Step-to-step state deltas versus Gate D are characterized separately as a systematic CPU-vs-GPU "
            "difference, not as a restart-boundary failure."
        )
    if explicit_energy_characterization["status"] == "within_gpu_noise_floor":
        strongest_surviving_claim += (
            " Explicit per-term energy-dump deltas exceed text display resolution but stay within repeated GPU energy noise."
        )
    if restart_characterization["status"] == "within_gpu_noise_floor":
        strongest_surviving_claim += (
            " Restart-boundary velocity deltas exceed the CPU-oracle epsilon but stay within repeated GPU state noise."
        )
    if reciprocal_force_characterization["status"] == "fft_backend_arithmetic_chain":
        strongest_surviving_claim += (
            " The remaining reciprocal-force residual is explicitly localized to the CPU FFTW vs GPU cuFFT "
            "arithmetic chain rather than reciprocal ownership leakage."
        )
    elif reciprocal_force_characterization["status"] == "within_roundoff_proxy":
        strongest_surviving_claim += (
            " The remaining reciprocal-force residual stays within the pinned conservative float-roundoff proxy."
        )

    broken_claims: list[str] = []
    for reason in reasons:
        if reason not in broken_claims:
            broken_claims.append(reason)
    if blocking_term_issue is not None:
        broken_claims.append(
            f"{blocking_term_issue['bucket']} remains unresolved because the oracle only provides "
            f"{blocking_term_issue['comparison_status']} visibility."
        )

    return {
        "status": status,
        "reasons": reasons,
        "strongest_surviving_claim": strongest_surviving_claim,
        "broken_update_boundary_claims": broken_claims,
        "exact_ambiguous_or_wrong_terms": blocking_ambiguous_terms_gate_e(
            per_term_rows, explicit_energy_characterization
        ),
        "gate_f_blocked": status != "PASS",
        "total_force_max_abs_component_delta": base_assessment.get("total_force_max_abs_component_delta"),
        "per_level_force_max_abs_component_delta": base_assessment.get("per_level_force_max_abs_component_delta"),
        "explicit_term_energy_display_assessment": base_assessment.get("explicit_term_energy_display_assessment"),
        "explicit_energy_characterization": explicit_energy_characterization,
        "virial_display_assessment": base_assessment.get("virial_display_assessment"),
        "virial_characterization": base_assessment.get("virial_characterization"),
        "state_characterization": state_characterization,
        "restart_characterization": restart_characterization,
        "reciprocal_force_comparison": reciprocal_force_comparison,
        "reciprocal_force_characterization": reciprocal_force_characterization,
        "blocking_term_issue": blocking_term_issue,
    }


def find_first_failure_gate_e(
    *,
    main_run: dict[str, object],
    gate_d_manifest_status: str | None,
    unsupported_feature_assessment: dict[str, object],
    event_order_comparison: dict[str, object],
    total_force_comparison: dict[str, object],
    per_level_force_comparison: dict[str, object],
    per_term_rows: list[dict[str, object]],
    reciprocal_force_comparison: dict[str, object],
    reciprocal_force_noise_floor: dict[str, object],
    reciprocal_force_roundoff_proxy: dict[str, object],
    fft_backend_arithmetic_evidence: dict[str, object],
    restart_validation: dict[str, object],
    state_comparison_to_gate_d: dict[str, object],
    state_comparison_to_gate_a: dict[str, object],
    gpu_noise_floor: dict[str, object],
    restart_characterization: dict[str, object],
    assessment: dict[str, object],
) -> dict[str, object] | None:
    if main_run["returncode"] != 0:
        return {
            "field": "main_run.returncode",
            "reason": "The Gate E mdrun command did not execute successfully.",
            "returncode": main_run["returncode"],
            "failure_markers": main_run["failure_markers"],
        }
    if gate_d_manifest_status != "PASS":
        return {"field": "upstream_gate_d_status", "details": gate_d_manifest_status}
    if not main_run["layout_report"].get("single_rank_explicit"):
        return {"field": "execution_constraints.single_rank", "details": main_run["layout_report"]}
    if not main_run["layout_report"].get("pme_gpu_enabled"):
        return {"field": "layout.pme_gpu", "details": main_run["layout_report"]["pme_task_line"]}
    if not main_run["layout_report"].get("update_gpu_enabled"):
        return {"field": "layout.update_gpu", "details": main_run["layout_report"]["update_task_line"]}
    if not main_run["layout_report"].get("mapping_explicit"):
        return {"field": "layout.gpu_mapping", "details": main_run["layout_report"]}
    for check in unsupported_feature_assessment["checks"]:
        if not check["passes"]:
            return {"field": f"unsupported_feature_avoidance.{check['field']}", "details": check}
    if not event_order_comparison["matches"]:
        return {"field": "event_order", "details": event_order_comparison["first_mismatch"]}
    if total_force_comparison["missing_in_actual"] or total_force_comparison["extra_in_actual"]:
        return {"field": "total_force", "details": total_force_comparison["first_mismatch"]}
    if per_level_force_comparison["missing_in_actual"] or per_level_force_comparison["extra_in_actual"]:
        return {"field": "per_level_force_totals", "details": per_level_force_comparison["first_mismatch"]}
    if reciprocal_force_comparison["missing_in_actual"] or reciprocal_force_comparison["extra_in_actual"]:
        return {"field": "reciprocal_force_trace", "details": reciprocal_force_comparison["first_mismatch"]}
    if reciprocal_force_comparison["first_mismatch"] is not None:
        return {"field": "reciprocal_force_trace", "details": reciprocal_force_comparison["first_mismatch"]}
    reciprocal_force_characterization = assessment.get("reciprocal_force_characterization", {})
    reciprocal_force_characterization = characterize_reciprocal_force_difference(
        reciprocal_force_comparison,
        reciprocal_force_noise_floor,
        reciprocal_force_roundoff_proxy,
        fft_backend_arithmetic_evidence,
    )
    if reciprocal_force_characterization.get("status") not in RECIPROCAL_FORCE_PASS_STATUSES:
        return {"field": "reciprocal_force_trace.magnitude", "details": reciprocal_force_characterization}
    if not gpu_noise_floor.get("available"):
        return {"field": "gpu_noise_floor", "details": gpu_noise_floor}
    if not restart_validation.get("available"):
        return {"field": "restart_continuity", "details": restart_validation}
    if restart_characterization["status"] == "fail":
        return {"field": "restart_continuity", "details": restart_validation["first_failure"]}
    if not state_comparison_to_gate_d.get("available") or not state_comparison_to_gate_d.get("coverage_matches"):
        return {"field": "state_continuity_vs_gate_d", "details": state_comparison_to_gate_d}
    if not state_comparison_to_gate_a.get("available") or not state_comparison_to_gate_a.get("coverage_matches"):
        return {"field": "state_continuity", "details": state_comparison_to_gate_a}
    if assessment["reasons"]:
        return {"field": "gate_e_assessment", "details": assessment["reasons"][0]}
    term_issue = assessment.get("blocking_term_issue")
    if term_issue is not None:
        return {"field": f"per_term.{term_issue['bucket']}", "details": term_issue}
    return None


def collect_system_result(
    args: argparse.Namespace,
    gmx: Path,
    out_root: Path,
    system_id: str,
    gate_a_system: dict[str, object],
    gate_d_system: dict[str, object] | None,
) -> dict[str, object]:
    system_root = out_root / system_id
    if system_root.exists():
        shutil.rmtree(system_root)
    logs_dir = system_root / "logs"
    summaries_dir = system_root / "summaries"
    for directory in (logs_dir, summaries_dir):
        directory.mkdir(parents=True, exist_ok=True)

    commands: list[dict[str, object]] = []
    gate_a_tpr = Path(gate_a_system["full_run_outputs"]["tpr"])
    gate_a_mdp = Path(gate_a_system["mdp"])
    gate_a_mdp_settings = parse_mdp_settings(gate_a_mdp)
    unsupported_feature_assessment = build_unsupported_feature_assessment(gate_a_mdp_settings, args)
    write_text(
        summaries_dir / "unsupported_feature_assessment.json",
        json.dumps(unsupported_feature_assessment, indent=2, sort_keys=True) + "\n",
    )

    tpr_dump = capture_output([str(gmx), "dump", "-s", str(gate_a_tpr)], cwd=REPO_ROOT)
    write_text(summaries_dir / "gate_a_tpr_dump.txt", tpr_dump)
    topology_inventory = parse_tpr_inventory(tpr_dump)
    write_text(
        summaries_dir / "topology_inventory.json",
        json.dumps(topology_inventory, indent=2, sort_keys=True) + "\n",
    )

    gate_a_energy_summary = load_json(Path(gate_a_system["energy_terms"]))
    gate_a_class2_trace = load_json(Path(gate_a_system["class2_subterm_energy_trace"]))
    gate_a_cpu_correction_trace = load_json(Path(gate_a_system["cpu_correction_energy_trace"]))
    gate_a_restart_summary = load_json(Path(gate_a_system["restart_summary"]))
    gate_a_term_coverage = build_gate_a_term_coverage(
        gate_a_energy_summary, topology_inventory, gate_a_class2_trace, gate_a_cpu_correction_trace
    )
    write_text(
        summaries_dir / "gate_a_term_coverage.json",
        json.dumps(gate_a_term_coverage, indent=2, sort_keys=True) + "\n",
    )

    expected_events = parse_event_trace(Path(gate_a_system["full_run_outputs"]["event_trace_tsv"]))
    expected_total_force = load_json(Path(gate_a_system["total_force_summary"]))["per_step_totals"]
    expected_per_level_force = load_json(Path(gate_a_system["per_level_force_totals"]))["entries"]
    expected_energy_frames = gate_a_energy_summary["frames"]
    gate_a_full_trr_dump = capture_output([str(gmx), "dump", "-f", gate_a_system["full_run_outputs"]["trr"]], cwd=REPO_ROOT)
    gate_a_full_trr_frames = parse_trr_dump(gate_a_full_trr_dump)
    if gate_d_system is not None:
        gate_d_full_trr_path = Path(gate_d_system["artifact_root"]) / "full" / "exact_full.trr"
        gate_d_full_trr_dump = capture_output([str(gmx), "dump", "-f", str(gate_d_full_trr_path)], cwd=REPO_ROOT)
        gate_d_full_trr_frames = parse_trr_dump(gate_d_full_trr_dump)
    else:
        gate_d_full_trr_frames = []
    gate_a_force_component_trace = parse_force_component_trace(
        Path(gate_a_system["artifact_root"]) / "full" / "m2p_trace" / "step0_force_component_trace.txt"
    )

    main_run = collect_gpu_run(
        args=args,
        gmx=gmx,
        gate_a_tpr=gate_a_tpr,
        system_root=system_root,
        run_label="full",
        trace_atom_count=read_gro_atom_count(Path(gate_a_system["full_run_outputs"]["gro"])),
        commands=commands,
    )
    repeat_runs: list[dict[str, object]] = []
    successful_runs: list[dict[str, object]] = []
    if main_run["returncode"] == 0:
        successful_runs.append(main_run)
        for repeat_index in range(1, args.gpu_repeats):
            repeat_run = collect_gpu_run(
                args=args,
                gmx=gmx,
                gate_a_tpr=gate_a_tpr,
                system_root=system_root,
                run_label=f"repeat_{repeat_index}",
                trace_atom_count=read_gro_atom_count(Path(gate_a_system["full_run_outputs"]["gro"])),
                commands=commands,
            )
            repeat_runs.append(repeat_run)
            if repeat_run["returncode"] == 0:
                successful_runs.append(repeat_run)

    if main_run["returncode"] == 0:
        event_order_comparison = compare_event_trace(main_run["actual_events"], expected_events)
        total_force_comparison = compare_total_force_entries(
            main_run["total_force_summary"]["per_step_totals"], expected_total_force
        )
        per_level_force_comparison = compare_per_level_force_entries(
            main_run["per_level_force_totals"]["entries"], expected_per_level_force
        )
        energy_comparison = compare_energy_frames(main_run["energy_frames"], expected_energy_frames)
        virial_comparison = extract_virial_deltas(energy_comparison)
        gpu_noise_floor = estimate_noise_floor(successful_runs)
        reciprocal_force_comparison = compare_force_component_rows(
            main_run["force_component_trace"], gate_a_force_component_trace, "coulomb_recip_force"
        )
        reciprocal_force_noise_floor = estimate_force_component_noise_floor(
            successful_runs, "coulomb_recip_force"
        )
        reciprocal_force_roundoff_proxy = estimate_force_component_roundoff_proxy(
            main_run["force_component_trace"], gate_a_force_component_trace, "coulomb_recip_force"
        )
        fft_backend_arithmetic_evidence = load_fft_backend_arithmetic_evidence(
            REPO_ROOT / "tests" / "reference_results",
            system_id,
            reciprocal_force_comparison.get("max_abs_component_delta"),
        )
        class2_trace_bucket_assessments = build_class2_trace_bucket_assessments(
            main_run["class2_subterm_energy_trace"], gate_a_class2_trace
        )
        cpu_correction_trace_bucket_assessments = build_cpu_correction_trace_bucket_assessments(
            main_run["cpu_correction_energy_trace"], gate_a_cpu_correction_trace
        )
        per_term_rows = build_per_term_comparison_rows(
            gate_a_term_coverage,
            energy_comparison,
            main_run["energy_frames"],
            expected_energy_frames,
            class2_trace_bucket_assessments,
            cpu_correction_trace_bucket_assessments,
        )
        state_comparison_to_gate_d = (
            compare_trr_frames(main_run["trr_frames"], gate_d_full_trr_frames)
            if gate_d_full_trr_frames
            else unavailable_state_comparison("Gate D", "Gate D trajectory frames are unavailable.")
        )
        state_comparison_to_gate_a = compare_trr_frames(main_run["trr_frames"], gate_a_full_trr_frames)
        state_noise_floor = estimate_state_noise_floor(successful_runs)
    else:
        blocked_reason = (
            "Gate E mdrun did not execute; direct event/force/energy/state/noise comparisons against Gate A are unavailable."
        )
        event_order_comparison = {"matches": False, "reason": blocked_reason, "first_mismatch": None}
        total_force_comparison = {
            "matches": False,
            "reason": blocked_reason,
            "missing_in_actual": [],
            "extra_in_actual": [],
            "max_abs_component_delta": None,
            "first_mismatch": None,
        }
        per_level_force_comparison = {
            "matches": False,
            "reason": blocked_reason,
            "missing_in_actual": [],
            "extra_in_actual": [],
            "max_abs_component_delta": None,
            "first_mismatch": None,
        }
        energy_comparison = {
            "matches": False,
            "reason": blocked_reason,
            "max_abs_delta_kj_mol": None,
            "first_mismatch": None,
            "frames": [],
        }
        virial_comparison = {"available": False, "reason": blocked_reason, "max_abs_delta": None, "frames": []}
        gpu_noise_floor = {
            "available": False,
            "reason": "No successful Gate E GPU runs are available.",
            "successful_run_count": 0,
        }
        reciprocal_force_comparison = {
            "component_name": "coulomb_recip_force",
            "missing_in_actual": [],
            "extra_in_actual": [],
            "first_mismatch": None,
            "first_nonzero_delta": None,
            "max_abs_component_delta": None,
            "compared_row_count": 0,
            "rows": [],
        }
        reciprocal_force_noise_floor = {
            "available": False,
            "reason": "No successful Gate E GPU runs are available.",
            "successful_run_count": 0,
        }
        reciprocal_force_roundoff_proxy = {
            "available": False,
            "reason": "No successful Gate E GPU runs are available.",
            "inference": True,
        }
        fft_backend_arithmetic_evidence = {
            "available": False,
            "reason": "No successful Gate E GPU runs are available.",
            "inference": False,
        }
        class2_trace_bucket_assessments = {}
        cpu_correction_trace_bucket_assessments = {}
        per_term_rows = build_per_term_comparison_rows(
            gate_a_term_coverage,
            energy_comparison,
            [],
            expected_energy_frames,
            class2_trace_bucket_assessments,
            cpu_correction_trace_bucket_assessments,
        )
        state_comparison_to_gate_a = {
            "available": False,
            "coverage_matches": False,
            "reason": blocked_reason,
            "actual_frame_count": 0,
            "expected_frame_count": len(gate_a_full_trr_frames),
            "frames": [],
        }
        state_comparison_to_gate_d = unavailable_state_comparison(
            "Gate D",
            "Gate E main run failed, so trajectory comparison to Gate D is unavailable.",
        )
        state_noise_floor = {
            "available": False,
            "reason": "No successful Gate E GPU runs are available for state-noise estimation.",
            "successful_run_count": 0,
        }

    restart_results = collect_restart_continuity(
        args=args, gmx=gmx, gate_a_tpr=gate_a_tpr, system_root=system_root, commands=commands
    )
    restart_validation = build_restart_validation(restart_results, gate_a_restart_summary, gpu_noise_floor)
    reciprocal_ownership_comparison = build_reciprocal_ownership_summary(
        energy_comparison=energy_comparison,
        cpu_correction_trace_bucket_assessments=cpu_correction_trace_bucket_assessments,
        reciprocal_force_comparison=reciprocal_force_comparison,
        reciprocal_force_noise_floor=reciprocal_force_noise_floor,
        reciprocal_force_roundoff_proxy=reciprocal_force_roundoff_proxy,
        fft_backend_arithmetic_evidence=fft_backend_arithmetic_evidence,
        topology_inventory=topology_inventory,
    )
    direct_oracle_comparison = summarize_direct_oracle_comparison_gate_e(
        event_order_comparison=event_order_comparison,
        total_force_comparison=total_force_comparison,
        per_level_force_comparison=per_level_force_comparison,
        energy_comparison=energy_comparison,
        virial_comparison=virial_comparison,
        gpu_noise_floor=gpu_noise_floor,
        per_term_rows=per_term_rows,
        class2_trace_bucket_assessments=class2_trace_bucket_assessments,
        cpu_correction_trace_bucket_assessments=cpu_correction_trace_bucket_assessments,
        reciprocal_force_comparison=reciprocal_force_comparison,
        reciprocal_force_noise_floor=reciprocal_force_noise_floor,
        reciprocal_force_roundoff_proxy=reciprocal_force_roundoff_proxy,
        fft_backend_arithmetic_evidence=fft_backend_arithmetic_evidence,
        state_comparison_to_gate_d=state_comparison_to_gate_d,
        state_comparison_to_gate_a=state_comparison_to_gate_a,
        state_noise_floor=state_noise_floor,
        restart_validation=restart_validation,
    )

    assessment = assess_gate_e_system(
        main_run=main_run,
        gate_d_manifest_status=None if gate_d_system is None else gate_d_system["gate_d_assessment"]["status"],
        gate_a_energy_frames=expected_energy_frames,
        event_order_comparison=event_order_comparison,
        total_force_comparison=total_force_comparison,
        per_level_force_comparison=per_level_force_comparison,
        energy_comparison=energy_comparison,
        virial_comparison=virial_comparison,
        gpu_noise_floor=gpu_noise_floor,
        per_term_rows=per_term_rows,
        reciprocal_force_comparison=reciprocal_force_comparison,
        reciprocal_force_noise_floor=reciprocal_force_noise_floor,
        reciprocal_force_roundoff_proxy=reciprocal_force_roundoff_proxy,
        fft_backend_arithmetic_evidence=fft_backend_arithmetic_evidence,
        unsupported_feature_assessment=unsupported_feature_assessment,
        restart_validation=restart_validation,
        state_comparison_to_gate_d=state_comparison_to_gate_d,
        state_comparison_to_gate_a=state_comparison_to_gate_a,
        state_noise_floor=state_noise_floor,
    )
    first_failure = find_first_failure_gate_e(
        main_run=main_run,
        gate_d_manifest_status=None if gate_d_system is None else gate_d_system["gate_d_assessment"]["status"],
        unsupported_feature_assessment=unsupported_feature_assessment,
        event_order_comparison=event_order_comparison,
        total_force_comparison=total_force_comparison,
        per_level_force_comparison=per_level_force_comparison,
        per_term_rows=per_term_rows,
        reciprocal_force_comparison=reciprocal_force_comparison,
        reciprocal_force_noise_floor=reciprocal_force_noise_floor,
        reciprocal_force_roundoff_proxy=reciprocal_force_roundoff_proxy,
        fft_backend_arithmetic_evidence=fft_backend_arithmetic_evidence,
        restart_validation=restart_validation,
        state_comparison_to_gate_d=state_comparison_to_gate_d,
        state_comparison_to_gate_a=state_comparison_to_gate_a,
        gpu_noise_floor=gpu_noise_floor,
        restart_characterization=assessment["restart_characterization"],
        assessment=assessment,
    )
    first_mismatching_term = assessment.get("blocking_term_issue")

    system_result = {
        "system_id": system_id,
        "artifact_root": str(system_root),
        "gate_a_artifact_root": gate_a_system["artifact_root"],
        "gate_a_commands_sh": gate_a_system["commands_sh"],
        "gate_d_artifact_root": None if gate_d_system is None else gate_d_system["artifact_root"],
        "gate_d_assessment_status": None if gate_d_system is None else gate_d_system["gate_d_assessment"]["status"],
        "gate_a_tpr": str(gate_a_tpr),
        "gate_a_mdp": str(gate_a_mdp),
        "commands_json": str(summaries_dir / "commands.json"),
        "commands_sh": str(system_root / "run_commands.sh"),
        "main_run": main_run,
        "repeat_runs": repeat_runs,
        "restart_results": restart_results,
        "restart_validation": restart_validation,
        "topology_inventory": topology_inventory,
        "unsupported_feature_assessment": unsupported_feature_assessment,
        "gate_a_term_coverage": gate_a_term_coverage,
        "event_order_comparison": event_order_comparison,
        "total_force_comparison": total_force_comparison,
        "per_level_force_comparison": per_level_force_comparison,
        "energy_comparison": energy_comparison,
        "virial_comparison": virial_comparison,
        "class2_trace_bucket_assessments": class2_trace_bucket_assessments,
        "cpu_correction_trace_bucket_assessments": cpu_correction_trace_bucket_assessments,
        "reciprocal_force_comparison": reciprocal_force_comparison,
        "reciprocal_force_noise_floor": reciprocal_force_noise_floor,
        "reciprocal_force_roundoff_proxy": reciprocal_force_roundoff_proxy,
        "fft_backend_arithmetic_evidence": fft_backend_arithmetic_evidence,
        "reciprocal_ownership_comparison": reciprocal_ownership_comparison,
        "state_comparison_to_gate_d": state_comparison_to_gate_d,
        "state_comparison_to_gate_a": state_comparison_to_gate_a,
        "state_noise_floor": state_noise_floor,
        "per_term_comparison_table": per_term_rows,
        "direct_oracle_comparison": direct_oracle_comparison,
        "gpu_noise_floor": gpu_noise_floor,
        "gate_e_assessment": assessment,
        "first_failure_field": first_failure,
        "first_mismatching_term": first_mismatching_term,
    }
    write_text(summaries_dir / "system_result.json", json.dumps(system_result, indent=2, sort_keys=True) + "\n")
    write_text(summaries_dir / "commands.json", json.dumps(commands, indent=2, sort_keys=True) + "\n")
    write_commands_script(system_root / "run_commands.sh", commands)
    return system_result


def build_manifest(
    *,
    args: argparse.Namespace,
    out_root: Path,
    gate_a_manifest: dict[str, object],
    gate_d_manifest: dict[str, object] | None,
    gmx: Path,
    gmx_version: str,
    gpu_inventory: dict[str, object],
    systems: list[dict[str, object]],
) -> dict[str, object]:
    status_rank = {"PASS": 0, "PARTIAL": 1, "FAIL": 2, "BLOCKER": 3}
    status = max((system["gate_e_assessment"]["status"] for system in systems), key=lambda item: status_rank[item])
    blocking_reasons = []
    for system in systems:
        if system["main_run"]["failure_markers"]:
            blocking_reasons.append(f"{system['system_id']}: {' | '.join(system['main_run']['failure_markers'])}")
        for reason in system["gate_e_assessment"]["reasons"]:
            blocking_reasons.append(f"{system['system_id']}: {reason}")
        if system["first_failure_field"] is not None:
            blocking_reasons.append(
                f"{system['system_id']}: first mismatch field is {system['first_failure_field']['field']}."
            )

    gate_f_allowed = all(system["gate_e_assessment"]["status"] == "PASS" for system in systems)
    recommendation_reason = (
        "Gate F may start because Gate E preserved update-boundary orchestration, restart continuity, and explicit standalone exact-r-RESPA trace semantics."
        if gate_f_allowed
        else "Gate F remains blocked until Gate E resolves the first update-boundary or restart mismatch for every fixture."
    )

    return {
        "schema_version": 1,
        "gate": "Gate E",
        "status": status,
        "gate_f_allowed": gate_f_allowed,
        "objective": "Validate standalone exact r-RESPA with nb gpu + bonded gpu + pme gpu + update gpu.",
        "artifact_root": str(out_root),
        "gate_a_manifest": str(Path(args.gate_a_manifest).resolve()),
        "gate_a_status": gate_a_manifest.get("status"),
        "gate_d_manifest": str(Path(args.gate_d_manifest).resolve()),
        "gate_d_status": None if gate_d_manifest is None else gate_d_manifest.get("status"),
        "gmx": str(gmx),
        "gmx_version": gmx_version,
        "precision_mode": parse_precision_mode(gmx_version),
        "gpu_support": parse_gpu_support(gmx_version),
        "hardware_configuration": gpu_inventory,
        "ntmpi": args.ntmpi,
        "ntomp": args.ntomp,
        "npme_flag_used": args.npme is not None,
        "npme_requested": args.npme,
        "npme_equals_one": args.npme == 1,
        "single_rank_required": True,
        "dlb": "no",
        "pme_rank_count": 0,
        "reproducibility_flags": [
            "-dlb no",
            "-pin off",
            "-nb gpu",
            "-pme gpu",
            "-bonded gpu",
            "-update gpu",
            "GMX_DISABLE_MODULAR_SIMULATOR=1",
        ],
        "binary_reproducibility_supported": False,
        "reproducibility_notes": [
            GPU_REPRODUCIBILITY_NOTE,
            "GPU noise floor is estimated from repeated successful Gate E runs for each fixture.",
            "Restart state continuity uses the gmx dump text resolution floor for TRR-derived coordinate/velocity comparisons.",
            "Rank-layout sensitivity is intentionally untested here because Gate E is single-rank only.",
        ],
        "rerun_used": False,
        "normal_md_used": True,
        "comparison_basis": (
            "Frozen Gate A CPU oracle remains the numeric source of truth. Gate D PASS is required as the upstream "
            "GPU evidence gate before enabling update gpu."
        ),
        "source_audit": {
            "update_gpu_decision": "src/gromacs/taskassignment/decidegpuusage.cpp::decideWhetherToUseGpuForUpdate",
            "simulation_workload": "src/gromacs/taskassignment/decidesimulationworkload.cpp::createSimulationWorkload",
            "exact_respa_md_guard": "src/gromacs/mdrun/md.cpp::canUseExactLammpsRespaVelocityVerlet",
            "exact_respa_gpu_stepper": "src/gromacs/mdrun/exactrespastepper.cpp::doExactRespaVelocityVerletStep",
            "reciprocal_force_trace_source": "src/gromacs/mdlib/sim_util.cpp::longRangeNonbondeds.calculate_delta",
        },
        "blocking_reasons": blocking_reasons,
        "recommendation": {
            "gate_f_allowed": gate_f_allowed,
            "reason": recommendation_reason,
        },
        "systems": systems,
    }


def write_manifest_markdown(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Gate E Oracle Comparison",
        "",
        f"- Status: {manifest['status']}",
        f"- Gate F allowed: {manifest['recommendation']['gate_f_allowed']}",
        f"- gmx: `{manifest['gmx']}`",
        f"- precision: `{manifest['precision_mode']}`",
        f"- GPU support: `{manifest['gpu_support']}`",
        f"- ntmpi / ntomp: `{manifest['ntmpi']}` / `{manifest['ntomp']}`",
        f"- npme flag used: `{manifest['npme_flag_used']}`",
        f"- DLB: `{manifest['dlb']}`",
        "",
        "## Blocking Reasons",
        "",
    ]
    if manifest["blocking_reasons"]:
        for reason in manifest["blocking_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- None")
    lines.extend(["", "## Systems", ""])
    for system in manifest["systems"]:
        lines.append(f"### {system['system_id']}")
        lines.append("")
        lines.append(f"- Gate E assessment: `{system['gate_e_assessment']['status']}`")
        lines.append(f"- Main run return code: `{system['main_run']['returncode']}`")
        lines.append(f"- Event order identical: `{system['event_order_comparison'].get('matches')}`")
        lines.append(
            f"- Restart continuity: `{system['gate_e_assessment']['restart_characterization'].get('status')}`"
        )
        lines.append(
            f"- State max coordinate delta vs Gate D: `{system['state_comparison_to_gate_d'].get('max_coordinate_abs_delta_nm')}`"
        )
        lines.append(
            f"- State max velocity delta vs Gate D: `{system['state_comparison_to_gate_d'].get('max_velocity_abs_delta_nm_ps')}`"
        )
        lines.append(
            f"- State max coordinate delta vs Gate A: `{system['state_comparison_to_gate_a'].get('max_coordinate_abs_delta_nm')}`"
        )
        lines.append(
            f"- State max velocity delta vs Gate A: `{system['state_comparison_to_gate_a'].get('max_velocity_abs_delta_nm_ps')}`"
        )
        lines.append(
            f"- Coul. recip. max abs delta: `{system['reciprocal_ownership_comparison']['coulomb_reciprocal_delta']['max_abs_delta_kj_mol']}`"
        )
        lines.append(
            f"- Reciprocal force characterization: `{system['gate_e_assessment'].get('reciprocal_force_characterization', {}).get('status')}`"
        )
        lines.append(
            f"- Reciprocal force roundoff proxy bound: `{system['reciprocal_force_roundoff_proxy'].get('bound')}`"
        )
        lines.append(
            f"- FFT-backend arithmetic evidence available: `{system['fft_backend_arithmetic_evidence'].get('available')}`"
        )
        lines.append(
            f"- Layout report: `{system['main_run']['layout_report']['rank_layout']}`"
        )
        lines.append(f"- First failure field: `{system['first_failure_field']}`")
        lines.append(f"- Artifact root: `{system['artifact_root']}`")
        lines.append(f"- Command script: `{system['commands_sh']}`")
        lines.append("")
    write_text(path, "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    gmx = Path(args.gmx).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    gate_a_manifest = load_json(Path(args.gate_a_manifest).resolve())
    gate_d_manifest = load_optional_manifest(Path(args.gate_d_manifest).resolve())
    validate_gate_chain(gate_a_manifest, gate_d_manifest)
    maybe_build(args, Path(args.build_dir).resolve() if args.build_dir is not None else None)

    gmx_version = capture_output([str(gmx), "--version"], cwd=REPO_ROOT)
    gpu_inventory = {
        "nvidia_smi_list": capture_optional_output(["nvidia-smi", "-L"]),
        "nvidia_smi_query": capture_optional_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "nvcc_version": capture_optional_output(["nvcc", "--version"]),
    }

    gate_a_systems_by_id = {system["system_id"]: system for system in gate_a_manifest["systems"]}
    gate_d_systems_by_id = {} if gate_d_manifest is None else {
        system["system_id"]: system for system in gate_d_manifest["systems"]
    }
    systems = []
    for system_id in SYSTEMS:
        systems.append(
            collect_system_result(
                args=args,
                gmx=gmx,
                out_root=out_root,
                system_id=system_id,
                gate_a_system=gate_a_systems_by_id[system_id],
                gate_d_system=gate_d_systems_by_id.get(system_id),
            )
        )

    manifest = build_manifest(
        args=args,
        out_root=out_root,
        gate_a_manifest=gate_a_manifest,
        gate_d_manifest=gate_d_manifest,
        gmx=gmx,
        gmx_version=gmx_version,
        gpu_inventory=gpu_inventory,
        systems=systems,
    )
    write_text(out_root / "gate_e_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_manifest_markdown(out_root / "gate_e_manifest.md", manifest)


if __name__ == "__main__":
    main()
