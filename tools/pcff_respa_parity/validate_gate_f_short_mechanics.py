from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

from freeze_gate_a_oracle import (
    DEFAULT_GMX,
    FIXTURE_ROOT,
    base_env,
    capture_output,
    command_record,
    env_delta,
    parse_energy_dump,
    parse_trr_dump,
    write_commands_script,
    write_text,
)
from validate_gate_b_nb_gpu import (
    GPU_REPEAT_COUNT,
    assess_energy_display_resolution,
    compare_energy_frames,
    parse_gpu_support,
    parse_precision_mode,
    run_command_allow_failure,
    trace_env_for_run,
)
from validate_gate_c_nb_bonded_gpu import DEFAULT_GATE_A_MANIFEST, load_json, maybe_build
from validate_gate_e_update_gpu import (
    TRR_DUMP_COMPONENT_RESOLUTION,
    build_unsupported_feature_assessment,
    compare_trr_frames,
    estimate_state_noise_floor,
    parse_layout_report,
    parse_mdp_settings,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_E_MANIFEST = (
    REPO_ROOT / "tests" / "reference_results" / "gate_e_update_gpu_validation" / "gate_e_manifest.json"
)
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_f_short_mechanics_validation"
RESTART_STATE_TOLERANCE = TRR_DUMP_COMPONENT_RESOLUTION * 1.1
DRIFT_TERMS = (
    "Potential",
    "Total Energy",
    "Pressure",
    "Vir-XX",
    "Vir-YY",
    "Vir-ZZ",
    "Pres-XX",
    "Pres-YY",
    "Pres-ZZ",
)
WATCH_VIRIAL_TERMS = (
    "Pressure",
    "Vir-XX",
    "Vir-YY",
    "Vir-ZZ",
    "Pres-XX",
    "Pres-YY",
    "Pres-ZZ",
)
GPU_REPRODUCIBILITY_NOTE = (
    "Binary reproducibility (-reprod) is not enabled because GROMACS rejects -nb gpu together with -reprod."
)
EXACT_RESPA_FACTOR = 4
LOW_LEVEL_TRACE_ATOM_COUNT = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Gate F short-horizon mechanics for standalone exact r-RESPA on the full GPU path."
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument(
        "--gate-a-manifest",
        default=str(DEFAULT_GATE_A_MANIFEST),
        help="Path to the frozen Gate A CPU oracle manifest.",
    )
    parser.add_argument(
        "--gate-e-manifest",
        default=str(DEFAULT_GATE_E_MANIFEST),
        help="Path to the Gate E manifest used as the upstream GPU gate.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--build-target", default="gmx", help="CMake target to build.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step.")
    parser.add_argument("--build-dir", default=None, help="Optional explicit build directory.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks for mdrun.")
    parser.add_argument("--ntomp", type=int, default=1, help="OpenMP threads for mdrun.")
    parser.add_argument("--npme", type=int, default=None, help="Optional explicit -npme value; omitted in Gate F.")
    parser.add_argument("--outer-steps", type=int, default=20, help="Number of exact r-RESPA outer steps.")
    parser.add_argument(
        "--gpu-repeats",
        type=int,
        default=GPU_REPEAT_COUNT,
        help="Total number of GPU runs, including the main run, for noise-floor estimation.",
    )
    parser.add_argument(
        "--exact-gpu-bonded-sequential-ftypes",
        action="store_true",
        help="Enable the exact-r-RESPA validation mode that replaces combined GPU bonded launches with sequential per-ftype launches.",
    )
    return parser.parse_args()


def validate_gate_chain(gate_a_manifest: dict[str, object], gate_e_manifest: dict[str, object]) -> None:
    if gate_a_manifest.get("status") != "PASS":
        raise ValueError("Gate A manifest is not PASS; Gate F cannot use it as a CPU baseline source.")
    if gate_e_manifest.get("status") != "PASS":
        raise ValueError("Gate E manifest is not PASS; Gate F should not proceed.")


def fixture_dir(system_id: str) -> Path:
    return FIXTURE_ROOT / system_id


def make_gate_f_mdp(outer_steps: int) -> str:
    nsteps = outer_steps * EXACT_RESPA_FACTOR
    return (
        "title                   = gate f short exact respa mechanics\n"
        "integrator              = md-vv\n"
        "dt                      = 0.0005\n"
        f"nsteps                  = {nsteps}\n"
        "constraints             = none\n"
        "cutoff-scheme           = Verlet\n"
        f"nstlist                 = {EXACT_RESPA_FACTOR}\n"
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
        "exact-respa             = yes\n"
        "exact-respa-levels      = 3\n"
        "exact-respa-level2-factor = 2\n"
        f"exact-respa-level3-factor = {EXACT_RESPA_FACTOR}\n"
        "exact-respa-bond-level  = 1\n"
        "exact-respa-angle-level = 1\n"
        "exact-respa-dihedral-level = 1\n"
        "exact-respa-improper-level = 1\n"
        "exact-respa-pair14-level = 1\n"
        "exact-respa-pair-level  = 3\n"
        "exact-respa-kspace-level = 3\n"
        "exact-respa-inner-level = 1\n"
        "exact-respa-middle-level = 2\n"
        "exact-respa-outer-level = 3\n"
        "exact-respa-inner-off   = 0.30\n"
        "exact-respa-inner-on    = 0.45\n"
        "exact-respa-outer-on    = 0.60\n"
        "exact-respa-outer-off   = 0.80\n"
        f"nstcalcenergy           = {EXACT_RESPA_FACTOR}\n"
        f"nstenergy               = {EXACT_RESPA_FACTOR}\n"
        f"nstlog                  = {EXACT_RESPA_FACTOR}\n"
        f"nstxout                 = {EXACT_RESPA_FACTOR}\n"
        f"nstvout                 = {EXACT_RESPA_FACTOR}\n"
        "nstfout                 = 0\n"
        "nstxout-compressed      = 0\n"
    )


def mdrun_args_cpu(args: argparse.Namespace, deffnm: Path) -> list[str]:
    return [
        "-s",
        str(deffnm.with_suffix(".tpr")),
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


def mdrun_args_gpu(args: argparse.Namespace, deffnm: Path, *, nsteps: int | None = None, cpi: Path | None = None) -> list[str]:
    argv = [
        "-s",
        str(deffnm.with_suffix(".tpr")),
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
        argv.extend(["-nsteps", str(nsteps)])
    if cpi is not None:
        argv.extend(["-cpi", str(cpi)])
    return argv


def gate_f_gpu_trace_env(args: argparse.Namespace, run_root: Path) -> dict[str, str]:
    env = trace_env_for_run(args, run_root)
    max_base_step = args.outer_steps * EXACT_RESPA_FACTOR
    env["GMX_PCFF_RESPA_TRACE_ATOMS"] = ",".join(str(atom) for atom in range(LOW_LEVEL_TRACE_ATOM_COUNT))
    env["GMX_EXACT_RESPA_FORCESTORE_TRACE_FILE"] = str(run_root / "force_store_trace.tsv")
    env["GMX_EXACT_RESPA_FORCESTORE_TRACE_ATOMS"] = str(LOW_LEVEL_TRACE_ATOM_COUNT)
    env["GMX_EXACT_RESPA_FORCESTORE_TRACE_MAX_BASE_STEP"] = str(max_base_step)
    if args.exact_gpu_bonded_sequential_ftypes:
        env["GMX_PCFF_RESPA_EXACT_GPU_BONDED_SEQUENTIAL_FTYPES"] = "1"
    return env


def extract_failure_markers(stderr_path: Path, stdout_path: Path) -> list[str]:
    markers: list[str] = []
    combined = ""
    if stderr_path.exists():
        combined += stderr_path.read_text(encoding="utf-8", errors="replace")
    if stdout_path.exists():
        combined += "\n" + stdout_path.read_text(encoding="utf-8", errors="replace")
    for token in (
        "Fatal error:",
        "Segmentation fault",
        "Only the md integrator is supported.",
        "Trying to mark event before fully consuming it",
    ):
        if token in combined:
            markers.append(token)
    return markers


def load_run_outputs(gmx: Path, run_root: Path) -> dict[str, object]:
    edr_path = run_root.with_suffix(".edr")
    trr_path = run_root.with_suffix(".trr")
    trace_root = run_root.parent
    energy_dump = capture_output([str(gmx), "dump", "-e", str(edr_path)], cwd=REPO_ROOT)
    trr_dump = capture_output([str(gmx), "dump", "-f", str(trr_path)], cwd=REPO_ROOT)
    return {
        "edr": str(edr_path),
        "trr": str(trr_path),
        "energy_frames": parse_energy_dump(energy_dump),
        "trr_frames": parse_trr_dump(trr_dump),
        "force_store_trace_path": str(trace_root / "force_store_trace.tsv"),
        "bonded_reduction_trace_path": str(
            trace_root / "m2p_trace" / "exact_gpu_bonded_reduction_trace.txt"
        ),
        "force_store_trace": parse_force_store_trace(trace_root / "force_store_trace.tsv"),
        "bonded_reduction_trace": parse_exact_gpu_bonded_reduction_trace(
            trace_root / "m2p_trace" / "exact_gpu_bonded_reduction_trace.txt"
        ),
    }


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
    label: str,
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
    stdout_path = logs_dir / f"{label}.stdout"
    stderr_path = logs_dir / f"{label}.stderr"
    result = run_command_allow_failure(argv, cwd=REPO_ROOT, env=env, stdout_path=stdout_path, stderr_path=stderr_path)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed; see {stderr_path}")
    commands.append(
        command_record(
            label,
            argv,
            cwd=REPO_ROOT,
            env_overrides=env_delta(env, os.environ),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    )


def run_md(
    *,
    gmx: Path,
    argv: list[str],
    env: dict[str, str],
    logs_dir: Path,
    commands: list[dict[str, object]],
    label: str,
    layout_args: argparse.Namespace,
) -> dict[str, object]:
    stdout_path = logs_dir / f"{label}.stdout"
    stderr_path = logs_dir / f"{label}.stderr"
    result = run_command_allow_failure(argv, cwd=REPO_ROOT, env=env, stdout_path=stdout_path, stderr_path=stderr_path)
    record = {
        "run_id": label,
        "argv": argv,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "returncode": result.returncode,
        "failure_markers": extract_failure_markers(stderr_path, stdout_path),
    }
    commands.append(
        command_record(
            label,
            argv,
            cwd=REPO_ROOT,
            env_overrides=env_delta(env, os.environ),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    )
    if result.returncode == 0:
        deffnm = Path(argv[argv.index("-deffnm") + 1])
        record["layout_report"] = parse_layout_report(stdout_path, stderr_path, layout_args)
        record["outputs"] = load_run_outputs(gmx, deffnm)
        record["energy_frames"] = record["outputs"]["energy_frames"]
        record["trr_frames"] = record["outputs"]["trr_frames"]
    return record


def build_series(frames: list[dict[str, object]], term: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    initial_value = None
    for frame in frames:
        terms = frame["terms"]
        if term not in terms:
            continue
        value = float(terms[term])
        if initial_value is None:
            initial_value = value
        rows.append(
            {
                "step": int(frame["step"]),
                "time_ps": float(frame["time_ps"]),
                "value": value,
                "drift": value - initial_value,
            }
        )
    return rows


def compute_envelope(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"available": False}
    drifts = [float(row["drift"]) for row in rows]
    values = [float(row["value"]) for row in rows]
    times = [float(row["time_ps"]) for row in rows]
    steps = [int(row["step"]) for row in rows]
    max_abs_drift = max(abs(value) for value in drifts)
    first_peak = next(
        (
            {
                "step": steps[index],
                "time_ps": times[index],
                "drift": drifts[index],
            }
            for index in range(len(drifts))
            if abs(drifts[index]) == max_abs_drift
        ),
        None,
    )
    duration_ps = times[-1] - times[0]
    slope = 0.0 if duration_ps == 0.0 else (drifts[-1] - drifts[0]) / duration_ps
    return {
        "available": True,
        "frame_count": len(rows),
        "initial_value": values[0],
        "final_value": values[-1],
        "endpoint_drift": drifts[-1],
        "max_abs_drift": max_abs_drift,
        "min_drift": min(drifts),
        "max_drift": max(drifts),
        "drift_slope_per_ps": slope,
        "first_peak": first_peak,
    }


def build_observable_series(frames: list[dict[str, object]], terms: tuple[str, ...]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for term in terms:
        rows = build_series(frames, term)
        result[term] = {
            "rows": rows,
            "envelope": compute_envelope(rows),
        }
    return result


def compare_series(
    actual_rows: list[dict[str, object]],
    expected_rows: list[dict[str, object]],
) -> dict[str, object]:
    if not actual_rows or not expected_rows:
        return {"available": False}
    frame_count = min(len(actual_rows), len(expected_rows))
    max_abs_value_delta = 0.0
    max_abs_drift_delta = 0.0
    first_nonzero = None
    rows = []
    for index in range(frame_count):
        actual = actual_rows[index]
        expected = expected_rows[index]
        value_delta = float(actual["value"]) - float(expected["value"])
        drift_delta = float(actual["drift"]) - float(expected["drift"])
        max_abs_value_delta = max(max_abs_value_delta, abs(value_delta))
        max_abs_drift_delta = max(max_abs_drift_delta, abs(drift_delta))
        if first_nonzero is None and (value_delta != 0.0 or drift_delta != 0.0):
            first_nonzero = {
                "step": int(actual["step"]),
                "time_ps": float(actual["time_ps"]),
                "value_delta": value_delta,
                "drift_delta": drift_delta,
            }
        rows.append(
            {
                "step": int(actual["step"]),
                "time_ps": float(actual["time_ps"]),
                "actual_value": float(actual["value"]),
                "expected_value": float(expected["value"]),
                "actual_drift": float(actual["drift"]),
                "expected_drift": float(expected["drift"]),
                "value_delta": value_delta,
                "drift_delta": drift_delta,
            }
        )
    return {
        "available": True,
        "actual_frame_count": len(actual_rows),
        "expected_frame_count": len(expected_rows),
        "frame_count": frame_count,
        "max_abs_value_delta": max_abs_value_delta,
        "max_abs_drift_delta": max_abs_drift_delta,
        "first_nonzero": first_nonzero,
        "rows": rows,
    }


def estimate_series_noise_floor(
    reference: dict[str, dict[str, object]],
    repeats: list[dict[str, dict[str, object]]],
    terms: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    noise: dict[str, dict[str, object]] = {}
    for term in terms:
        max_abs_value_delta = 0.0
        max_abs_drift_delta = 0.0
        compared_runs = []
        available = True
        for index, repeat in enumerate(repeats, start=1):
            comparison = compare_series(repeat[term]["rows"], reference[term]["rows"])
            if not comparison.get("available"):
                available = False
                break
            max_abs_value_delta = max(max_abs_value_delta, float(comparison["max_abs_value_delta"]))
            max_abs_drift_delta = max(max_abs_drift_delta, float(comparison["max_abs_drift_delta"]))
            compared_runs.append(
                {
                    "run_id": f"repeat_{index}",
                    "max_abs_value_delta": comparison["max_abs_value_delta"],
                    "max_abs_drift_delta": comparison["max_abs_drift_delta"],
                }
            )
        noise[term] = {
            "available": available and bool(repeats),
            "reference_run_id": "gpu_full",
            "max_abs_value_delta": max_abs_value_delta if repeats else None,
            "max_abs_drift_delta": max_abs_drift_delta if repeats else None,
            "compared_runs": compared_runs,
        }
    return noise


def parse_force_store_trace(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError(f"Unexpected force-store trace row: {raw_line}")
        rows.append(
            {
                "base_step": int(fields[0]),
                "phase": fields[1],
                "level": int(fields[2]),
                "atom": int(fields[3]),
                "fx": float(fields[4]),
                "fy": float(fields[5]),
                "fz": float(fields[6]),
            }
        )
    return rows


def parse_exact_gpu_bonded_reduction_trace(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError(f"Unexpected exact GPU bonded reduction trace row: {raw_line}")
        rows.append(
            {
                "step": int(fields[0]),
                "level": int(fields[1]),
                "stage": fields[2],
                "atom": int(fields[3]),
                "fx": float(fields[4]),
                "fy": float(fields[5]),
                "fz": float(fields[6]),
            }
        )
    return rows


def compare_vector_trace_rows(
    actual_rows: list[dict[str, object]],
    expected_rows: list[dict[str, object]],
    *,
    key_fields: tuple[str, ...],
    bucket_fields: tuple[str, ...],
) -> dict[str, object]:
    actual_map = {tuple(row[field] for field in key_fields): row for row in actual_rows}
    expected_map = {tuple(row[field] for field in key_fields): row for row in expected_rows}
    missing_in_actual = sorted([list(key) for key in expected_map.keys() - actual_map.keys()])
    extra_in_actual = sorted([list(key) for key in actual_map.keys() - expected_map.keys()])
    max_abs_component_delta = 0.0
    first_nonzero_delta = None
    max_abs_component_delta_by_bucket: dict[str, float] = {}
    for key in sorted(actual_map.keys() & expected_map.keys()):
        actual_row = actual_map[key]
        expected_row = expected_map[key]
        deltas = {
            axis: float(actual_row[axis]) - float(expected_row[axis]) for axis in ("fx", "fy", "fz")
        }
        local_max = max(abs(delta) for delta in deltas.values())
        max_abs_component_delta = max(max_abs_component_delta, local_max)
        bucket_label = "/".join(str(actual_row[field]) for field in bucket_fields)
        max_abs_component_delta_by_bucket[bucket_label] = max(
            max_abs_component_delta_by_bucket.get(bucket_label, 0.0), local_max
        )
        if first_nonzero_delta is None and any(delta != 0.0 for delta in deltas.values()):
            first_nonzero_delta = {
                "key": list(key),
                "deltas": deltas,
                "expected": {axis: float(expected_row[axis]) for axis in ("fx", "fy", "fz")},
                "actual": {axis: float(actual_row[axis]) for axis in ("fx", "fy", "fz")},
            }
    return {
        "missing_in_actual": missing_in_actual,
        "extra_in_actual": extra_in_actual,
        "max_abs_component_delta": max_abs_component_delta,
        "max_abs_component_delta_by_bucket": max_abs_component_delta_by_bucket,
        "first_nonzero_delta": first_nonzero_delta,
    }


def estimate_vector_trace_noise_floor(
    reference_rows: list[dict[str, object]],
    repeat_runs: list[dict[str, object]],
    *,
    trace_key: str,
    key_fields: tuple[str, ...],
    bucket_fields: tuple[str, ...],
) -> dict[str, object]:
    if not reference_rows:
        return {
            "available": False,
            "reason": f"Reference run did not produce {trace_key}.",
            "successful_run_count": len(repeat_runs) + 1,
        }
    if not repeat_runs:
        return {
            "available": False,
            "reason": "Need at least two successful GPU runs to estimate a low-level noise floor.",
            "successful_run_count": 1,
        }

    max_abs_component_delta = 0.0
    max_abs_component_delta_by_bucket: dict[str, float] = {}
    worst_repeat = None
    for repeat_run in repeat_runs:
        comparison = compare_vector_trace_rows(
            repeat_run["outputs"].get(trace_key, []),
            reference_rows,
            key_fields=key_fields,
            bucket_fields=bucket_fields,
        )
        if comparison["missing_in_actual"] or comparison["extra_in_actual"]:
            return {
                "available": False,
                "reason": f"Repeated GPU runs changed {trace_key} coverage.",
                "successful_run_count": len(repeat_runs) + 1,
                "first_mismatch": comparison["missing_in_actual"][:1] or comparison["extra_in_actual"][:1],
            }
        if float(comparison["max_abs_component_delta"]) > max_abs_component_delta:
            max_abs_component_delta = float(comparison["max_abs_component_delta"])
            worst_repeat = {
                "run_id": repeat_run["run_id"],
                "max_abs_component_delta": max_abs_component_delta,
                "first_nonzero_delta": comparison["first_nonzero_delta"],
            }
        for bucket, value in comparison["max_abs_component_delta_by_bucket"].items():
            max_abs_component_delta_by_bucket[bucket] = max(
                max_abs_component_delta_by_bucket.get(bucket, 0.0), float(value)
            )

    return {
        "available": True,
        "successful_run_count": len(repeat_runs) + 1,
        "reference_run_id": "gpu_full",
        "max_abs_component_delta": max_abs_component_delta,
        "max_abs_component_delta_by_bucket": max_abs_component_delta_by_bucket,
        "worst_repeat": worst_repeat,
    }


def characterize_low_level_noise_resolution(
    *,
    energy_noise_floor: dict[str, object],
    low_level_noise_floor: dict[str, dict[str, object]],
) -> dict[str, object]:
    total_energy_noise = max(
        float(energy_noise_floor.get("max_abs_value_delta") or 0.0),
        float(energy_noise_floor.get("max_abs_drift_delta") or 0.0),
    )
    low_level_max = 0.0
    for trace_name in ("force_store_trace", "bonded_reduction_trace"):
        trace_result = low_level_noise_floor.get(trace_name, {})
        if trace_result.get("available"):
            low_level_max = max(low_level_max, float(trace_result.get("max_abs_component_delta") or 0.0))

    if total_energy_noise == 0.0 and low_level_max > 0.0:
        return {
            "status": "under_resolved_by_energy_terms",
            "note": (
                "Repeated GPU EDR observables are identical at dump resolution, but low-level exact-r-RESPA "
                "bonded/force-store traces still vary across repeats. Energy-only noise floors under-resolve "
                "GPU path noise for Gate F."
            ),
            "energy_noise_floor": total_energy_noise,
            "low_level_max_abs_component_delta": low_level_max,
        }

    return {
        "status": "consistent_with_energy_terms",
        "note": "Energy-observable noise floors are consistent with the measured low-level trace noise.",
        "energy_noise_floor": total_energy_noise,
        "low_level_max_abs_component_delta": low_level_max,
    }


def write_combined_drift_tsv(
    path: Path,
    cpu_series: dict[str, dict[str, object]],
    gpu_series: dict[str, dict[str, object]],
) -> None:
    header = [
        "step",
        "time_ps",
        "cpu_total_energy",
        "cpu_total_energy_drift",
        "gpu_total_energy",
        "gpu_total_energy_drift",
        "cpu_potential",
        "cpu_potential_drift",
        "gpu_potential",
        "gpu_potential_drift",
        "cpu_pressure",
        "cpu_pressure_drift",
        "gpu_pressure",
        "gpu_pressure_drift",
    ]
    cpu_total = cpu_series["Total Energy"]["rows"]
    gpu_total = gpu_series["Total Energy"]["rows"]
    cpu_potential = cpu_series["Potential"]["rows"]
    gpu_potential = gpu_series["Potential"]["rows"]
    cpu_pressure = cpu_series["Pressure"]["rows"]
    gpu_pressure = gpu_series["Pressure"]["rows"]
    lines = ["\t".join(header)]
    frame_count = min(
        len(cpu_total),
        len(gpu_total),
        len(cpu_potential),
        len(gpu_potential),
        len(cpu_pressure),
        len(gpu_pressure),
    )
    for index in range(frame_count):
        lines.append(
            "\t".join(
                [
                    str(cpu_total[index]["step"]),
                    f"{cpu_total[index]['time_ps']:.10f}",
                    f"{cpu_total[index]['value']:.12f}",
                    f"{cpu_total[index]['drift']:.12f}",
                    f"{gpu_total[index]['value']:.12f}",
                    f"{gpu_total[index]['drift']:.12f}",
                    f"{cpu_potential[index]['value']:.12f}",
                    f"{cpu_potential[index]['drift']:.12f}",
                    f"{gpu_potential[index]['value']:.12f}",
                    f"{gpu_potential[index]['drift']:.12f}",
                    f"{cpu_pressure[index]['value']:.12f}",
                    f"{cpu_pressure[index]['drift']:.12f}",
                    f"{gpu_pressure[index]['value']:.12f}",
                    f"{gpu_pressure[index]['drift']:.12f}",
                ]
            )
        )
    write_text(path, "\n".join(lines) + "\n")


def characterize_total_energy(
    *,
    cpu_series: dict[str, dict[str, object]],
    gpu_series: dict[str, dict[str, object]],
    gpu_noise_floor: dict[str, dict[str, object]],
) -> dict[str, object]:
    cpu_envelope = cpu_series["Total Energy"]["envelope"]
    gpu_envelope = gpu_series["Total Energy"]["envelope"]
    series_comparison = compare_series(gpu_series["Total Energy"]["rows"], cpu_series["Total Energy"]["rows"])
    noise_floor = gpu_noise_floor["Total Energy"]
    display_assessment = assess_energy_display_resolution(
        [{"step": row["step"], "time_ps": row["time_ps"], "terms": {"Total Energy": row["value"]}} for row in gpu_series["Total Energy"]["rows"]],
        [{"step": row["step"], "time_ps": row["time_ps"], "terms": {"Total Energy": row["value"]}} for row in cpu_series["Total Energy"]["rows"]],
        selected_terms=("Total Energy",),
    )
    tolerated_drift_delta = max(
        float(display_assessment.get("max_allowed_abs_delta", 0.0)),
        float(noise_floor.get("max_abs_drift_delta") or 0.0),
    )
    envelope_inflation = float(gpu_envelope["max_abs_drift"]) - float(cpu_envelope["max_abs_drift"])
    if envelope_inflation <= tolerated_drift_delta + 1e-12:
        status = "within_noise_floor"
        note = "GPU total-energy drift envelope does not inflate beyond CPU baseline plus measured GPU drift noise."
    else:
        status = "drift_inflation"
        note = "GPU total-energy drift envelope exceeds CPU baseline by more than measured GPU drift noise."
    return {
        "status": status,
        "note": note,
        "cpu_envelope": cpu_envelope,
        "gpu_envelope": gpu_envelope,
        "envelope_inflation": envelope_inflation,
        "tolerated_drift_delta": tolerated_drift_delta,
        "series_comparison": series_comparison,
        "gpu_noise_floor": noise_floor,
        "display_assessment": display_assessment,
    }


def characterize_watch_terms(
    *,
    cpu_series: dict[str, dict[str, object]],
    gpu_series: dict[str, dict[str, object]],
    gpu_noise_floor: dict[str, dict[str, object]],
    terms: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for term in terms:
        comparison = compare_series(gpu_series[term]["rows"], cpu_series[term]["rows"])
        noise_floor = gpu_noise_floor[term]
        display_assessment = assess_energy_display_resolution(
            [{"step": row["step"], "time_ps": row["time_ps"], "terms": {term: row["value"]}} for row in gpu_series[term]["rows"]],
            [{"step": row["step"], "time_ps": row["time_ps"], "terms": {term: row["value"]}} for row in cpu_series[term]["rows"]],
            selected_terms=(term,),
        )
        tolerated = max(
            float(display_assessment.get("max_allowed_abs_delta", 0.0)),
            float(noise_floor.get("max_abs_drift_delta") or 0.0),
        )
        if float(comparison.get("max_abs_drift_delta") or 0.0) <= tolerated + 1e-12:
            status = "within_gpu_noise_floor"
            note = "Observed CPU-vs-GPU drift difference stays within measured GPU noise or dump resolution."
        else:
            status = "systematic_difference"
            note = (
                "추측입니다. Pressure/virial drift differs from the CPU baseline beyond measured GPU noise, "
                "but Gate F records it as an explicit watch item rather than silently accepting it."
            )
        result[term] = {
            "status": status,
            "note": note,
            "comparison": comparison,
            "gpu_noise_floor": noise_floor,
            "display_assessment": display_assessment,
            "cpu_envelope": cpu_series[term]["envelope"],
            "gpu_envelope": gpu_series[term]["envelope"],
        }
    return result


def characterize_restart(
    *,
    full_run: dict[str, object],
    split_run: dict[str, object],
    gpu_state_noise_floor: dict[str, object] | None = None,
) -> dict[str, object]:
    energy_comparison = compare_energy_frames(
        split_run["outputs"]["energy_frames"], full_run["outputs"]["energy_frames"]
    )
    blocking_display = assess_energy_display_resolution(
        split_run["outputs"]["energy_frames"],
        full_run["outputs"]["energy_frames"],
        selected_terms=("Total Energy",),
    )
    watch_display = assess_energy_display_resolution(
        split_run["outputs"]["energy_frames"],
        full_run["outputs"]["energy_frames"],
        selected_terms=("Potential", "Pressure", "Vir-XX", "Vir-YY", "Vir-ZZ", "Pres-XX", "Pres-YY", "Pres-ZZ"),
    )
    state_comparison = compare_trr_frames(split_run["outputs"]["trr_frames"], full_run["outputs"]["trr_frames"])
    tolerated_coordinate_delta = RESTART_STATE_TOLERANCE
    tolerated_velocity_delta = RESTART_STATE_TOLERANCE
    if gpu_state_noise_floor and gpu_state_noise_floor.get("available"):
        tolerated_coordinate_delta = max(
            tolerated_coordinate_delta,
            float(gpu_state_noise_floor.get("max_coordinate_abs_delta_nm") or 0.0),
        )
        tolerated_velocity_delta = max(
            tolerated_velocity_delta,
            float(gpu_state_noise_floor.get("max_velocity_abs_delta_nm_ps") or 0.0),
        )
    failures = []
    if float(state_comparison.get("max_coordinate_abs_delta_nm") or 0.0) > tolerated_coordinate_delta:
        failures.append(
            {
                "field": "max_coordinate_abs_delta_nm",
                "value": state_comparison["max_coordinate_abs_delta_nm"],
                "tolerance": tolerated_coordinate_delta,
            }
        )
    if float(state_comparison.get("max_velocity_abs_delta_nm_ps") or 0.0) > tolerated_velocity_delta:
        failures.append(
            {
                "field": "max_velocity_abs_delta_nm_ps",
                "value": state_comparison["max_velocity_abs_delta_nm_ps"],
                "tolerance": tolerated_velocity_delta,
            }
        )
    if not blocking_display["within_bounds"]:
        failures.append({"field": "energy_display_resolution", "details": blocking_display["first_excess"]})
    note = (
        "Restart continuity uses the larger of the TRR dump-resolution floor and repeated GPU state noise."
        if gpu_state_noise_floor and gpu_state_noise_floor.get("available")
        else "Restart continuity uses the pinned TRR dump-resolution floor."
    )
    return {
        "status": "PASS" if not failures else "FAIL",
        "note": note,
        "energy_comparison": energy_comparison,
        "blocking_energy_display_assessment": blocking_display,
        "watch_energy_display_assessment": watch_display,
        "state_comparison": state_comparison,
        "effective_state_tolerances": {
            "max_coordinate_abs_delta_nm": tolerated_coordinate_delta,
            "max_velocity_abs_delta_nm_ps": tolerated_velocity_delta,
        },
        "failures": failures,
        "first_failure": failures[0] if failures else None,
    }


def build_short_run_summary(
    *,
    cpu_series: dict[str, dict[str, object]],
    gpu_series: dict[str, dict[str, object]],
    total_energy_characterization: dict[str, object],
    virial_pressure_watch: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "cpu_total_energy_envelope": cpu_series["Total Energy"]["envelope"],
        "gpu_total_energy_envelope": gpu_series["Total Energy"]["envelope"],
        "cpu_potential_envelope": cpu_series["Potential"]["envelope"],
        "gpu_potential_envelope": gpu_series["Potential"]["envelope"],
        "energy_drift_characterization": total_energy_characterization,
        "pressure_virial_watch": virial_pressure_watch,
    }


def collect_system_result(
    *,
    args: argparse.Namespace,
    gmx: Path,
    out_root: Path,
    system_id: str,
    gate_a_system: dict[str, object],
) -> dict[str, object]:
    system_root = out_root / system_id
    if system_root.exists():
        shutil.rmtree(system_root)
    inputs_dir = system_root / "inputs"
    cpu_dir = system_root / "cpu_full"
    gpu_dir = system_root / "gpu_full"
    restart_dir = system_root / "gpu_split"
    logs_dir = system_root / "logs"
    summaries_dir = system_root / "summaries"
    for directory in (inputs_dir, cpu_dir, gpu_dir, restart_dir, logs_dir, summaries_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mdp_path = inputs_dir / "gate_f_short_exact_respa.mdp"
    write_text(mdp_path, make_gate_f_mdp(args.outer_steps))
    mdp_settings = parse_mdp_settings(mdp_path)
    unsupported_feature_assessment = build_unsupported_feature_assessment(mdp_settings, args)

    fixture = fixture_dir(system_id)
    commands: list[dict[str, object]] = []
    cpu_env = base_env(args)
    gpu_env = base_env(args)

    cpu_deffnm = cpu_dir / "cpu_full"
    run_grompp(
        gmx=gmx,
        mdp_path=mdp_path,
        conf_path=fixture / "initial_nve.gro",
        top_path=fixture / "topol.top",
        tpr_path=cpu_deffnm.with_suffix(".tpr"),
        mdout_path=inputs_dir / "gate_f_cpu_mdout.mdp",
        env=cpu_env,
        logs_dir=logs_dir,
        commands=commands,
        label="grompp_cpu_full",
    )
    cpu_run = run_md(
        gmx=gmx,
        argv=[str(gmx), "mdrun", *mdrun_args_cpu(args, cpu_deffnm)],
        env=cpu_env,
        logs_dir=logs_dir,
        commands=commands,
        label="mdrun_cpu_full",
        layout_args=args,
    )

    gpu_deffnm = gpu_dir / "gpu_full"
    gpu_trace_env = gate_f_gpu_trace_env(args, gpu_deffnm.parent)
    run_grompp(
        gmx=gmx,
        mdp_path=mdp_path,
        conf_path=fixture / "initial_nve.gro",
        top_path=fixture / "topol.top",
        tpr_path=gpu_deffnm.with_suffix(".tpr"),
        mdout_path=inputs_dir / "gate_f_gpu_mdout.mdp",
        env=gpu_trace_env,
        logs_dir=logs_dir,
        commands=commands,
        label="grompp_gpu_full",
    )
    gpu_run = run_md(
        gmx=gmx,
        argv=[str(gmx), "mdrun", *mdrun_args_gpu(args, gpu_deffnm)],
        env=gpu_trace_env,
        logs_dir=logs_dir,
        commands=commands,
        label="mdrun_gpu_full",
        layout_args=args,
    )

    repeat_runs = []
    for repeat_index in range(1, args.gpu_repeats):
        repeat_deffnm = system_root / f"gpu_repeat_{repeat_index}" / "gpu_repeat"
        repeat_deffnm.parent.mkdir(parents=True, exist_ok=True)
        repeat_trace_env = gate_f_gpu_trace_env(args, repeat_deffnm.parent)
        run_grompp(
            gmx=gmx,
            mdp_path=mdp_path,
            conf_path=fixture / "initial_nve.gro",
            top_path=fixture / "topol.top",
            tpr_path=repeat_deffnm.with_suffix(".tpr"),
            mdout_path=inputs_dir / f"gate_f_gpu_repeat_{repeat_index}_mdout.mdp",
            env=repeat_trace_env,
            logs_dir=logs_dir,
            commands=commands,
            label=f"grompp_gpu_repeat_{repeat_index}",
        )
        repeat_runs.append(
            run_md(
                gmx=gmx,
                argv=[str(gmx), "mdrun", *mdrun_args_gpu(args, repeat_deffnm)],
                env=repeat_trace_env,
                logs_dir=logs_dir,
                commands=commands,
                label=f"mdrun_gpu_repeat_{repeat_index}",
                layout_args=args,
            )
        )

    split_outer_steps = max(1, args.outer_steps // 2)
    split_steps = split_outer_steps * EXACT_RESPA_FACTOR
    split_deffnm = restart_dir / "gpu_split"
    run_grompp(
        gmx=gmx,
        mdp_path=mdp_path,
        conf_path=fixture / "initial_nve.gro",
        top_path=fixture / "topol.top",
        tpr_path=split_deffnm.with_suffix(".tpr"),
        mdout_path=inputs_dir / "gate_f_gpu_split_mdout.mdp",
        env=gpu_env,
        logs_dir=logs_dir,
        commands=commands,
        label="grompp_gpu_split",
    )
    split_first = run_md(
        gmx=gmx,
        argv=[str(gmx), "mdrun", *mdrun_args_gpu(args, split_deffnm, nsteps=split_steps)],
        env=gpu_env,
        logs_dir=logs_dir,
        commands=commands,
        label="mdrun_gpu_split_first",
        layout_args=args,
    )
    split_second = run_md(
        gmx=gmx,
        argv=[str(gmx), "mdrun", *mdrun_args_gpu(args, split_deffnm, cpi=split_deffnm.with_suffix(".cpt"))],
        env=gpu_env,
        logs_dir=logs_dir,
        commands=commands,
        label="mdrun_gpu_split_second",
        layout_args=args,
    )
    split_run = {
        "run_id": "gpu_split",
        "returncode": 0 if split_first["returncode"] == 0 and split_second["returncode"] == 0 else 1,
        "outputs": split_second.get("outputs"),
        "split_first": split_first,
        "split_second": split_second,
    }

    if cpu_run["returncode"] != 0 or gpu_run["returncode"] != 0:
        raise RuntimeError(f"Gate F primary runs failed for {system_id}.")
    successful_repeats = [run for run in repeat_runs if run["returncode"] == 0]

    cpu_series = build_observable_series(cpu_run["outputs"]["energy_frames"], DRIFT_TERMS)
    gpu_series = build_observable_series(gpu_run["outputs"]["energy_frames"], DRIFT_TERMS)
    repeat_series = [build_observable_series(run["outputs"]["energy_frames"], DRIFT_TERMS) for run in successful_repeats]
    gpu_noise_floor = estimate_series_noise_floor(gpu_series, repeat_series, DRIFT_TERMS)
    gpu_low_level_noise_floor = {
        "force_store_trace": estimate_vector_trace_noise_floor(
            gpu_run["outputs"]["force_store_trace"],
            successful_repeats,
            trace_key="force_store_trace",
            key_fields=("base_step", "phase", "level", "atom"),
            bucket_fields=("phase", "level"),
        ),
        "bonded_reduction_trace": estimate_vector_trace_noise_floor(
            gpu_run["outputs"]["bonded_reduction_trace"],
            successful_repeats,
            trace_key="bonded_reduction_trace",
            key_fields=("step", "level", "stage", "atom"),
            bucket_fields=("stage", "level"),
        ),
    }
    low_level_noise_resolution = characterize_low_level_noise_resolution(
        energy_noise_floor=gpu_noise_floor["Total Energy"],
        low_level_noise_floor=gpu_low_level_noise_floor,
    )
    total_energy_characterization = characterize_total_energy(
        cpu_series=cpu_series,
        gpu_series=gpu_series,
        gpu_noise_floor=gpu_noise_floor,
    )
    virial_pressure_watch = characterize_watch_terms(
        cpu_series=cpu_series,
        gpu_series=gpu_series,
        gpu_noise_floor=gpu_noise_floor,
        terms=WATCH_VIRIAL_TERMS,
    )
    cpu_gpu_state_comparison = compare_trr_frames(gpu_run["outputs"]["trr_frames"], cpu_run["outputs"]["trr_frames"])
    gpu_state_noise_floor = estimate_state_noise_floor([gpu_run, *successful_repeats])
    restart_comparison = characterize_restart(
        full_run=gpu_run,
        split_run=split_run,
        gpu_state_noise_floor=gpu_state_noise_floor,
    )
    short_run_summary = build_short_run_summary(
        cpu_series=cpu_series,
        gpu_series=gpu_series,
        total_energy_characterization=total_energy_characterization,
        virial_pressure_watch=virial_pressure_watch,
    )

    first_failure = None
    if total_energy_characterization["status"] == "drift_inflation":
        first_failure = {
            "field": "total_energy_drift_envelope",
            "details": total_energy_characterization,
        }
    elif restart_comparison["status"] != "PASS":
        first_failure = {
            "field": "restart_continuity",
            "details": restart_comparison["first_failure"],
        }

    drift_tsv_path = summaries_dir / "cpu_gpu_total_pressure_drift.tsv"
    write_combined_drift_tsv(drift_tsv_path, cpu_series, gpu_series)

    system_result = {
        "system_id": system_id,
        "artifact_root": str(system_root),
        "gate_a_artifact_root": gate_a_system["artifact_root"],
        "gate_a_commands_sh": gate_a_system["commands_sh"],
        "commands_json": str(summaries_dir / "commands.json"),
        "commands_sh": str(system_root / "run_commands.sh"),
        "short_mdp": str(mdp_path),
        "cpu_run": cpu_run,
        "gpu_run": gpu_run,
        "gpu_repeat_runs": repeat_runs,
        "gpu_split_run": split_run,
        "unsupported_feature_assessment": unsupported_feature_assessment,
        "cpu_series": cpu_series,
        "gpu_series": gpu_series,
        "gpu_noise_floor": gpu_noise_floor,
        "gpu_low_level_noise_floor": gpu_low_level_noise_floor,
        "low_level_noise_resolution": low_level_noise_resolution,
        "cpu_gpu_state_comparison": cpu_gpu_state_comparison,
        "gpu_state_noise_floor": gpu_state_noise_floor,
        "short_run_summary": short_run_summary,
        "total_energy_characterization": total_energy_characterization,
        "virial_pressure_watch": virial_pressure_watch,
        "restart_comparison": restart_comparison,
        "rerun_statement": {
            "rerun_used": False,
            "note": (
                "Gate F does not use mdrun -rerun for mechanical validation. GROMACS documents that rerun computes "
                "coordinate-based potential quantities only and does not report kinetic, total or conserved energy, "
                "temperature, virial, or pressure."
            ),
            "basis": [
                "docs/user-guide/mdrun-features.rst",
                "docs/user-guide/mdp-options.rst",
            ],
        },
        "drift_tsv": str(drift_tsv_path),
        "first_failure_field": first_failure,
        "overall_status": "PASS" if first_failure is None else "FAIL",
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
    gate_e_manifest: dict[str, object],
    gmx: Path,
    gmx_version: str,
    gpu_inventory: dict[str, object],
    systems: list[dict[str, object]],
) -> dict[str, object]:
    status = "PASS" if all(system["overall_status"] == "PASS" for system in systems) else "FAIL"
    blocking_reasons = []
    for system in systems:
        if system["first_failure_field"] is not None:
            blocking_reasons.append(
                f"{system['system_id']}: first failing observable is {system['first_failure_field']['field']}."
            )
    gate_g_allowed = status == "PASS"
    reproducibility_flags = [
        "-dlb no",
        "-pin off",
        "-nb gpu",
        "-pme gpu",
        "-bonded gpu",
        "-update gpu",
        "GMX_DISABLE_MODULAR_SIMULATOR=1",
    ]
    if args.exact_gpu_bonded_sequential_ftypes:
        reproducibility_flags.append("GMX_PCFF_RESPA_EXACT_GPU_BONDED_SEQUENTIAL_FTYPES=1")
    return {
        "schema_version": 1,
        "gate": "Gate F",
        "status": status,
        "gate_g_allowed": gate_g_allowed,
        "objective": "Validate short-horizon mechanical integrity of the standalone exact r-RESPA GPU path.",
        "artifact_root": str(out_root),
        "gate_a_manifest": str(Path(args.gate_a_manifest).resolve()),
        "gate_a_status": gate_a_manifest.get("status"),
        "gate_e_manifest": str(Path(args.gate_e_manifest).resolve()),
        "gate_e_status": gate_e_manifest.get("status"),
        "gmx": str(gmx),
        "gmx_version": gmx_version,
        "precision_mode": parse_precision_mode(gmx_version),
        "gpu_support": parse_gpu_support(gmx_version),
        "hardware_configuration": gpu_inventory,
        "exact_gpu_bonded_validation_mode": (
            "sequential_ftypes" if args.exact_gpu_bonded_sequential_ftypes else "combined_kernel"
        ),
        "ntmpi": args.ntmpi,
        "ntomp": args.ntomp,
        "single_rank_required": True,
        "dlb": "no",
        "reproducibility_flags": reproducibility_flags,
        "reproducibility_notes": [
            GPU_REPRODUCIBILITY_NOTE,
            (
                "Gate F uses nstcalcenergy/nstenergy/nstlog/nstxout/nstvout = exact-respa-factor = 4 because "
                "src/gromacs/mdtypes/exactrespaschedule.cpp requires these periodic intervals to be multiples "
                "of the slowest exact-r-RESPA factor."
            ),
            "GPU drift, restart, and low-level bonded/force-store noise floors are estimated from repeated Gate F GPU runs.",
            "Rerun mode is intentionally not used for Gate F mechanical validation because it does not report kinetic/total/conserved energy, temperature, virial, or pressure.",
        ],
        "rerun_used": False,
        "comparison_basis": (
            "Gate A remains the frozen standalone CPU oracle. Gate F derives a short-horizon NVE-style diagnostic "
            "window from the same fixture inputs and compares the full GPU path against a CPU run of that diagnostic "
            "window plus repeated GPU runs."
        ),
        "blocking_reasons": blocking_reasons,
        "recommendation": {
            "gate_g_allowed": gate_g_allowed,
            "reason": (
                "Gate G may start because Gate F found no unexplained short-window drift inflation or restart discontinuity."
                if gate_g_allowed
                else "Gate G remains blocked until Gate F resolves the first failing short-window observable."
            ),
        },
        "systems": systems,
    }


def write_manifest_markdown(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Gate F Short-Window Mechanics",
        "",
        f"- Status: {manifest['status']}",
        f"- Gate G allowed: {manifest['gate_g_allowed']}",
        f"- gmx: `{manifest['gmx']}`",
        f"- precision: `{manifest['precision_mode']}`",
        f"- GPU support: `{manifest['gpu_support']}`",
        f"- Exact GPU bonded validation mode: `{manifest['exact_gpu_bonded_validation_mode']}`",
        f"- ntmpi / ntomp: `{manifest['ntmpi']}` / `{manifest['ntomp']}`",
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
        total_energy = system["total_energy_characterization"]
        lines.append(f"### {system['system_id']}")
        lines.append("")
        lines.append(f"- Status: `{system['overall_status']}`")
        lines.append(f"- CPU total-energy drift envelope: `{total_energy['cpu_envelope']['max_abs_drift']}`")
        lines.append(f"- GPU total-energy drift envelope: `{total_energy['gpu_envelope']['max_abs_drift']}`")
        lines.append(f"- Envelope inflation: `{total_energy['envelope_inflation']}`")
        lines.append(
            f"- Low-level noise resolution: `{system['low_level_noise_resolution']['status']}`"
        )
        lines.append(
            f"- Force-store noise floor: `{system['gpu_low_level_noise_floor']['force_store_trace']}`"
        )
        lines.append(
            f"- Bonded reduction noise floor: `{system['gpu_low_level_noise_floor']['bonded_reduction_trace']}`"
        )
        lines.append(f"- Restart comparison: `{system['restart_comparison']['status']}`")
        lines.append(f"- First failing observable: `{system['first_failure_field']}`")
        lines.append(f"- Drift TSV: `{system['drift_tsv']}`")
        lines.append(f"- Artifact root: `{system['artifact_root']}`")
        lines.append("")
    write_text(path, "\n".join(lines) + "\n")


def write_blocker_manifest(
    out_root: Path,
    *,
    gate_a_manifest: dict[str, object],
    gate_e_manifest: dict[str, object],
    reason: str,
) -> None:
    manifest = {
        "status": "BLOCKER",
        "gate_g_allowed": False,
        "artifact_root": str(out_root),
        "gate_a_status": gate_a_manifest.get("status"),
        "gate_e_status": gate_e_manifest.get("status"),
        "blocking_reasons": [reason],
        "systems": [],
        "first_failure_field": {"field": "upstream_prerequisites", "details": reason},
    }
    write_text(out_root / "gate_f_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_text(
        out_root / "gate_f_manifest.md",
        "\n".join(
            [
                "# Gate F Short Mechanics",
                "",
                "- Status: BLOCKER",
                "- Gate G allowed: False",
                f"- Gate A status: `{manifest['gate_a_status']}`",
                f"- Gate E status: `{manifest['gate_e_status']}`",
                "",
                "## Blocking Reasons",
                "",
                f"- {reason}",
                "",
            ]
        )
        + "\n",
    )


def main() -> None:
    args = parse_args()
    gmx = Path(args.gmx).resolve()
    out_root = Path(args.out).resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    gate_a_manifest = load_json(Path(args.gate_a_manifest))
    gate_e_manifest = load_json(Path(args.gate_e_manifest))
    try:
        validate_gate_chain(gate_a_manifest, gate_e_manifest)
    except ValueError as exc:
        write_blocker_manifest(
            out_root,
            gate_a_manifest=gate_a_manifest,
            gate_e_manifest=gate_e_manifest,
            reason=str(exc),
        )
        return

    maybe_build(args, Path(args.build_dir).resolve() if args.build_dir is not None else None)

    gmx_version = capture_output([str(gmx), "--version"], cwd=REPO_ROOT)
    gpu_inventory = gate_e_manifest.get("hardware_configuration", {})

    systems = [
        collect_system_result(
            args=args,
            gmx=gmx,
            out_root=out_root,
            system_id=gate_a_system["system_id"],
            gate_a_system=gate_a_system,
        )
        for gate_a_system in gate_a_manifest["systems"]
    ]

    manifest = build_manifest(
        args=args,
        out_root=out_root,
        gate_a_manifest=gate_a_manifest,
        gate_e_manifest=gate_e_manifest,
        gmx=gmx,
        gmx_version=gmx_version,
        gpu_inventory=gpu_inventory,
        systems=systems,
    )
    write_text(out_root / "gate_f_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_manifest_markdown(out_root / "gate_f_manifest.md", manifest)


if __name__ == "__main__":
    main()
