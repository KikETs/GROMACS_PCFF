from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import shutil
from pathlib import Path

from freeze_gate_a_oracle import (
    DEFAULT_GMX,
    FIXTURE_ROOT,
    base_env,
    capture_output,
    command_record,
    env_delta,
    write_commands_script,
    write_text,
)
from validate_gate_b_nb_gpu import (
    load_json,
    parse_gpu_support,
    parse_precision_mode,
    run_command_allow_failure,
)
from validate_gate_c_nb_bonded_gpu import DEFAULT_GATE_A_MANIFEST, maybe_build
from validate_gate_e_update_gpu import parse_layout_report


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_E_MANIFEST = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_e_update_gpu_validation_after_restart_policy_fix"
    / "gate_e_manifest.json"
)
DEFAULT_GATE_F_MANIFEST = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_f_short_mechanics_validation_after_update_fix"
    / "gate_f_manifest.json"
)
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_g_long_ensemble_validation"

DT_PS = 0.0005
EXACT_RESPA_FACTOR = 4
DEFAULT_SAMPLE_INTERVAL = EXACT_RESPA_FACTOR * 10
DEFAULT_EQ_PS = 20.0
DEFAULT_PROD_PS = 40.0
DEFAULT_REPLICAS = 3
DEFAULT_TEMP_K = 300.0
DEFAULT_PRESSURE_BAR = 1.0
DEFAULT_TAU_T_PS = 0.5
DEFAULT_TAU_P_PS = 5.0
DEFAULT_COMPRESSIBILITY_BAR_INV = 4.5e-5
BLOCK_COUNT = 5

SYSTEM_PROTOCOLS: dict[str, dict[str, object]] = {
    "small_oligomer": {
        "ensemble": "nvt",
        "title": "Gate G NVT exact r-RESPA oligomer statistics",
        "required_observables": ("Temperature", "Pressure", "Potential"),
        "optional_observables": (),
        "msd_relevant": False,
        "msd_reason": "Short 40 ps windows on this small oligomer are not long enough for a meaningful diffusion-facing observable.",
    },
    "small_salt_polymer_box": {
        "ensemble": "npt",
        "title": "Gate G NPT exact r-RESPA salt/polymer statistics",
        "required_observables": ("Temperature", "Pressure", "Potential", "Volume", "Density", "Box-X", "Box-Y", "Box-Z"),
        "optional_observables": (),
        "msd_relevant": False,
        "msd_reason": "This small periodic salt/polymer box and the current sub-100 ps horizon are too short for a defensible MSD/diffusion comparison.",
    },
}

REQUESTED_OBSERVABLES = (
    "Temperature",
    "Pressure",
    "Potential",
    "Volume",
    "Density",
    "Box-X",
    "Box-Y",
    "Box-Z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Gate G long-horizon ensemble behavior for standalone exact r-RESPA on the full GPU path."
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
        help="Path to the Gate E manifest used as the full GPU orchestration prerequisite.",
    )
    parser.add_argument(
        "--gate-f-manifest",
        default=str(DEFAULT_GATE_F_MANIFEST),
        help="Path to the Gate F manifest used as the short-horizon mechanical prerequisite.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--build-target", default="gmx", help="CMake target to build.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step.")
    parser.add_argument("--build-dir", default=None, help="Optional explicit build directory.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks for mdrun.")
    parser.add_argument("--ntomp", type=int, default=1, help="OpenMP threads for mdrun.")
    parser.add_argument("--npme", type=int, default=None, help="Optional explicit -npme value; omitted in Gate G.")
    parser.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS, help="Replica count per layout.")
    parser.add_argument("--equil-ps", type=float, default=DEFAULT_EQ_PS, help="Equilibration duration in ps.")
    parser.add_argument("--prod-ps", type=float, default=DEFAULT_PROD_PS, help="Production duration in ps.")
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=DEFAULT_SAMPLE_INTERVAL,
        help="Energy/log sampling interval in base steps. Must be a multiple of the exact-r-RESPA factor.",
    )
    parser.add_argument("--temperature-k", type=float, default=DEFAULT_TEMP_K, help="Target temperature.")
    parser.add_argument("--pressure-bar", type=float, default=DEFAULT_PRESSURE_BAR, help="Target pressure for NPT.")
    parser.add_argument("--tau-t-ps", type=float, default=DEFAULT_TAU_T_PS, help="Thermostat coupling time.")
    parser.add_argument("--tau-p-ps", type=float, default=DEFAULT_TAU_P_PS, help="Barostat coupling time.")
    parser.add_argument(
        "--compressibility-bar-inv",
        type=float,
        default=DEFAULT_COMPRESSIBILITY_BAR_INV,
        help="Isotropic compressibility in bar^-1 for C-rescale pressure coupling.",
    )
    return parser.parse_args()


def validate_gate_chain(
    gate_a_manifest: dict[str, object], gate_e_manifest: dict[str, object], gate_f_manifest: dict[str, object]
) -> None:
    if gate_a_manifest.get("status") != "PASS":
        raise ValueError("Gate A manifest is not PASS; Gate G cannot use it as a mechanically validated CPU source.")
    if gate_e_manifest.get("status") != "PASS":
        raise ValueError("Gate E manifest is not PASS; Gate G should not proceed on an unvalidated GPU update path.")
    if gate_f_manifest.get("status") != "PASS":
        raise ValueError("Gate F manifest is not PASS; Gate G can not excuse unresolved short-horizon mechanical issues.")


def fixture_dir(system_id: str) -> Path:
    return FIXTURE_ROOT / system_id


def validate_args(args: argparse.Namespace) -> None:
    if args.ntmpi != 1:
        raise ValueError("Gate G is restricted to single-rank runs (ntmpi=1).")
    if args.sample_interval <= 0 or args.sample_interval % EXACT_RESPA_FACTOR != 0:
        raise ValueError("sample-interval must be a positive multiple of the exact-r-RESPA factor.")
    for name, value in (("equil-ps", args.equil_ps), ("prod-ps", args.prod_ps)):
        steps = int(round(value / DT_PS))
        if not math.isclose(steps * DT_PS, value, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{name} must be representable as an integer number of base steps.")
        if steps <= 0 or steps % EXACT_RESPA_FACTOR != 0:
            raise ValueError(f"{name} must correspond to a positive multiple of the exact-r-RESPA factor.")


def steps_from_ps(duration_ps: float) -> int:
    return int(round(duration_ps / DT_PS))


def equil_outer_steps(args: argparse.Namespace) -> int:
    return steps_from_ps(args.equil_ps) // EXACT_RESPA_FACTOR


def prod_outer_steps(args: argparse.Namespace) -> int:
    return steps_from_ps(args.prod_ps) // EXACT_RESPA_FACTOR


def exact_respa_common_mdp(nsteps: int, sample_interval: int) -> str:
    return (
        "integrator              = md-vv\n"
        f"dt                      = {DT_PS:.4f}\n"
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
        "comm-mode               = none\n"
        "verlet-buffer-tolerance = -1\n"
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
        f"nstcalcenergy           = {sample_interval}\n"
        f"nstenergy               = {sample_interval}\n"
        f"nstlog                  = {sample_interval}\n"
        "nstxout                 = 0\n"
        "nstvout                 = 0\n"
        "nstfout                 = 0\n"
        "nstxout-compressed      = 0\n"
    )


def make_gate_g_mdp(
    *,
    system_id: str,
    ensemble: str,
    duration_ps: float,
    sample_interval: int,
    phase: str,
    seed: int,
    args: argparse.Namespace,
) -> str:
    nsteps = steps_from_ps(duration_ps)
    title = f"gate g {phase} exact respa {system_id} {ensemble}"
    thermostat = (
        "tcoupl                  = v-rescale\n"
        "tc-grps                 = System\n"
        f"tau-t                   = {args.tau_t_ps:.3f}\n"
        f"ref-t                   = {args.temperature_k:.3f}\n"
        f"nsttcouple              = {EXACT_RESPA_FACTOR}\n"
    )
    if ensemble == "npt":
        barostat = (
            "pcoupl                  = c-rescale\n"
            "pcoupltype              = isotropic\n"
            f"tau-p                   = {args.tau_p_ps:.3f}\n"
            f"ref-p                   = {args.pressure_bar:.3f}\n"
            f"compressibility         = {args.compressibility_bar_inv:.7g}\n"
            f"nstpcouple              = {EXACT_RESPA_FACTOR}\n"
            "refcoord-scaling        = no\n"
        )
    else:
        barostat = "pcoupl                  = no\n"
    if phase == "equil":
        velocity = (
            "gen-vel                 = yes\n"
            f"gen-temp                = {args.temperature_k:.3f}\n"
            f"gen-seed                = {seed}\n"
        )
    else:
        velocity = "gen-vel                 = no\n"
    return f"title                   = {title}\n" + exact_respa_common_mdp(nsteps, sample_interval) + thermostat + barostat + velocity


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


def mdrun_args_gpu(args: argparse.Namespace, deffnm: Path) -> list[str]:
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
    if args.npme is not None:
        argv.extend(["-npme", str(args.npme)])
    return argv


def run_grompp(
    *,
    gmx: Path,
    mdp_path: Path,
    conf_path: Path,
    top_path: Path,
    tpr_path: Path,
    mdout_path: Path,
    logs_dir: Path,
    commands: list[dict[str, object]],
    label: str,
    env: dict[str, str],
    checkpoint_path: Path | None = None,
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
    if checkpoint_path is not None:
        argv.extend(["-t", str(checkpoint_path)])
    stdout_path = logs_dir / f"{label}.stdout"
    stderr_path = logs_dir / f"{label}.stderr"
    result = run_command_allow_failure(argv, cwd=REPO_ROOT, env=env, stdout_path=stdout_path, stderr_path=stderr_path)
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
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed; see {stderr_path}")


def run_md(
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
    result = run_command_allow_failure(argv, cwd=REPO_ROOT, env=env, stdout_path=stdout_path, stderr_path=stderr_path)
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
    payload = {
        "run_id": label,
        "argv": argv,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "returncode": result.returncode,
        "layout_report": parse_layout_report(stdout_path, stderr_path, args),
    }
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed; see {stderr_path}")
    return payload


def energy_terms_input(terms: tuple[str, ...]) -> str:
    return "".join(f"{term}\n" for term in terms) + "0\n"


def extract_energy_series(gmx: Path, edr_path: Path, out_path: Path, terms: tuple[str, ...]) -> dict[str, list[float]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    stdout_path = out_path.with_suffix(".energy.stdout")
    stderr_path = out_path.with_suffix(".energy.stderr")
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(
            [str(gmx), "energy", "-f", str(edr_path), "-o", str(out_path)],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            input=energy_terms_input(terms),
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"gmx energy failed for {edr_path}")
    return parse_xvg_series(out_path)


def parse_xvg_series(path: Path) -> dict[str, list[float]]:
    legends: list[str] = []
    times: list[float] = []
    values_by_legend: dict[str, list[float]] = {}
    legend_re = re.compile(r'^@ s(\d+) legend "(.*)"$')
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("@"):
            match = legend_re.match(line)
            if match:
                legend = match.group(2)
                legends.append(legend)
                values_by_legend.setdefault(legend, [])
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        times.append(float(parts[0]))
        for index, legend in enumerate(legends):
            if index + 1 < len(parts):
                values_by_legend[legend].append(float(parts[index + 1]))
    result: dict[str, list[float]] = {"time_ps": times}
    result.update(values_by_legend)
    return result


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sample_std(values) / math.sqrt(len(values))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def split_blocks(values: list[float], blocks: int) -> list[list[float]]:
    if not values:
        return []
    nblocks = min(blocks, len(values))
    sized_blocks: list[list[float]] = []
    for block_index in range(nblocks):
        start = round(block_index * len(values) / nblocks)
        end = round((block_index + 1) * len(values) / nblocks)
        block = values[start:end]
        if block:
            sized_blocks.append(block)
    return sized_blocks


def summarize_series(values: list[float]) -> dict[str, object]:
    if not values:
        return {"available": False}
    blocks = split_blocks(values, BLOCK_COUNT)
    block_means = [mean(block) for block in blocks]
    abs_block_drift = abs(block_means[-1] - block_means[0]) if len(block_means) >= 2 else 0.0
    return {
        "available": True,
        "frame_count": len(values),
        "mean": mean(values),
        "std": sample_std(values),
        "sem": sem(values),
        "min": min(values),
        "max": max(values),
        "q10": quantile(values, 0.10),
        "q50": quantile(values, 0.50),
        "q90": quantile(values, 0.90),
        "block_count": len(blocks),
        "block_means": block_means,
        "block_sem": sem(block_means),
        "block_span": (max(block_means) - min(block_means)) if block_means else 0.0,
        "abs_block_drift": abs_block_drift,
    }


def ks_distance(cpu_values: list[float], gpu_values: list[float]) -> float:
    if not cpu_values or not gpu_values:
        return 0.0
    left = sorted(cpu_values)
    right = sorted(gpu_values)
    i = 0
    j = 0
    cdf_left = 0.0
    cdf_right = 0.0
    distance = 0.0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            value = left[i]
            while i < len(left) and left[i] == value:
                i += 1
            cdf_left = i / len(left)
        else:
            value = right[j]
            while j < len(right) and right[j] == value:
                j += 1
            cdf_right = j / len(right)
        distance = max(distance, abs(cdf_left - cdf_right))
    distance = max(distance, abs(1.0 - cdf_right), abs(cdf_left - 1.0))
    return distance


def aggregate_replicates(observable_runs: list[dict[str, object]]) -> dict[str, object]:
    if not observable_runs:
        return {"available": False}
    means = [float(run["mean"]) for run in observable_runs]
    blocks = [float(run["block_sem"]) for run in observable_runs]
    block_spans = [float(run["block_span"]) for run in observable_runs]
    abs_block_drifts = [float(run["abs_block_drift"]) for run in observable_runs]
    pooled_values: list[float] = []
    for run in observable_runs:
        pooled_values.extend(run["values"])
    return {
        "available": True,
        "replica_count": len(observable_runs),
        "mean_of_means": mean(means),
        "std_of_means": sample_std(means),
        "sem_of_means": sem(means),
        "mean_block_sem": mean(blocks),
        "mean_block_span": mean(block_spans),
        "mean_abs_block_drift": mean(abs_block_drifts),
        "sem_abs_block_drift": sem(abs_block_drifts),
        "pooled_mean": mean(pooled_values),
        "pooled_std": sample_std(pooled_values),
        "pooled_q10": quantile(pooled_values, 0.10),
        "pooled_q50": quantile(pooled_values, 0.50),
        "pooled_q90": quantile(pooled_values, 0.90),
        "pooled_values": pooled_values,
        "replicas": observable_runs,
    }


def classify_difference(diff: float, combined_uncertainty: float, replica_diffs: list[float]) -> str:
    if diff == 0.0:
        return "stochastic"
    if combined_uncertainty == 0.0:
        return "structural"
    if abs(diff) <= 3.0 * combined_uncertainty:
        return "stochastic"
    nonzero_signs = {math.copysign(1.0, value) for value in replica_diffs if value != 0.0}
    return "structural" if len(nonzero_signs) <= 1 else "stochastic"


def compare_layout_observable(cpu: dict[str, object], gpu: dict[str, object]) -> dict[str, object]:
    if not cpu.get("available") or not gpu.get("available"):
        return {"available": False}
    cpu_uncertainty = max(float(cpu["sem_of_means"]), float(cpu["mean_block_sem"]))
    gpu_uncertainty = max(float(gpu["sem_of_means"]), float(gpu["mean_block_sem"]))
    combined_uncertainty = math.sqrt(cpu_uncertainty * cpu_uncertainty + gpu_uncertainty * gpu_uncertainty)
    diff = float(gpu["mean_of_means"]) - float(cpu["mean_of_means"])
    replica_count = min(len(cpu["replicas"]), len(gpu["replicas"]))
    replica_diffs = [
        float(gpu["replicas"][index]["mean"]) - float(cpu["replicas"][index]["mean"]) for index in range(replica_count)
    ]
    cpu_drift_uncertainty = max(float(cpu["sem_abs_block_drift"]), 1.0e-12)
    gpu_drift_uncertainty = max(float(gpu["sem_abs_block_drift"]), 1.0e-12)
    combined_drift_uncertainty = math.sqrt(
        cpu_drift_uncertainty * cpu_drift_uncertainty + gpu_drift_uncertainty * gpu_drift_uncertainty
    )
    drift_delta = float(gpu["mean_abs_block_drift"]) - float(cpu["mean_abs_block_drift"])
    mean_pass = (combined_uncertainty == 0.0 and diff == 0.0) or abs(diff) <= 3.0 * combined_uncertainty
    drift_pass = drift_delta <= 3.0 * combined_drift_uncertainty
    classification = classify_difference(diff, combined_uncertainty, replica_diffs)
    return {
        "available": True,
        "cpu_mean": cpu["mean_of_means"],
        "gpu_mean": gpu["mean_of_means"],
        "mean_diff": diff,
        "cpu_uncertainty": cpu_uncertainty,
        "gpu_uncertainty": gpu_uncertainty,
        "combined_uncertainty": combined_uncertainty,
        "z_like_ratio": 0.0 if combined_uncertainty == 0.0 and diff == 0.0 else (
            math.inf if combined_uncertainty == 0.0 else abs(diff) / combined_uncertainty
        ),
        "cpu_mean_abs_block_drift": cpu["mean_abs_block_drift"],
        "gpu_mean_abs_block_drift": gpu["mean_abs_block_drift"],
        "block_drift_delta": drift_delta,
        "combined_block_drift_uncertainty": combined_drift_uncertainty,
        "block_drift_ratio": math.inf if combined_drift_uncertainty == 0.0 and drift_delta > 0.0 else (
            0.0 if combined_drift_uncertainty == 0.0 else drift_delta / combined_drift_uncertainty
        ),
        "ks_distance": ks_distance(cpu["pooled_values"], gpu["pooled_values"]),
        "replica_diffs": replica_diffs,
        "classification": classification,
        "passes_mean": mean_pass,
        "passes_block_drift": drift_pass,
        "passes": mean_pass and drift_pass,
    }


def collect_run_observables(gmx: Path, run_deffnm: Path, xvg_path: Path) -> dict[str, dict[str, object]]:
    series = extract_energy_series(gmx, run_deffnm.with_suffix(".edr"), xvg_path, REQUESTED_OBSERVABLES)
    result: dict[str, dict[str, object]] = {}
    for term in REQUESTED_OBSERVABLES:
        values = [float(value) for value in series.get(term, [])]
        summary = summarize_series(values)
        summary["values"] = values
        result[term] = summary
    return result


def sanitize_observable_aggregate(aggregate: dict[str, object]) -> dict[str, object]:
    if not aggregate.get("available"):
        return {"available": False}
    sanitized = {key: value for key, value in aggregate.items() if key != "pooled_values"}
    sanitized["replicas"] = [
        {key: value for key, value in replica.items() if key != "values"} for replica in aggregate["replicas"]
    ]
    return sanitized


def sanitize_layout_aggregates(layout_aggregates: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        layout: {
            "replica_count": payload["replica_count"],
            "observables": {
                term: sanitize_observable_aggregate(aggregate) for term, aggregate in payload["observables"].items()
            },
        }
        for layout, payload in layout_aggregates.items()
    }


def save_observable_table(path: Path, comparisons: dict[str, dict[str, object]]) -> None:
    header = (
        "observable\tcpu_mean\tgpu_mean\tmean_diff\tcombined_uncertainty\tz_like_ratio\t"
        "cpu_mean_abs_block_drift\tgpu_mean_abs_block_drift\tblock_drift_delta\t"
        "combined_block_drift_uncertainty\tks_distance\tclassification\tpasses\n"
    )
    rows = [header]
    for term, comparison in comparisons.items():
        if not comparison.get("available"):
            rows.append(f"{term}\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tmissing\tFalse\n")
            continue
        rows.append(
            f"{term}\t{comparison['cpu_mean']:.10g}\t{comparison['gpu_mean']:.10g}\t"
            f"{comparison['mean_diff']:.10g}\t{comparison['combined_uncertainty']:.10g}\t"
            f"{comparison['z_like_ratio']:.10g}\t{comparison['cpu_mean_abs_block_drift']:.10g}\t"
            f"{comparison['gpu_mean_abs_block_drift']:.10g}\t{comparison['block_drift_delta']:.10g}\t"
            f"{comparison['combined_block_drift_uncertainty']:.10g}\t{comparison['ks_distance']:.10g}\t"
            f"{comparison['classification']}\t{comparison['passes']}\n"
        )
    write_text(path, "".join(rows))


def system_status(
    required_observables: tuple[str, ...], comparisons: dict[str, dict[str, object]]
) -> tuple[str, str | None, list[str]]:
    reasons: list[str] = []
    first_failing = None
    for observable in required_observables:
        comparison = comparisons.get(observable, {})
        if not comparison.get("available"):
            reasons.append(f"{observable} is missing from the production observable set.")
            first_failing = first_failing or observable
            continue
        if not comparison.get("passes"):
            reason = f"{observable} differs beyond the current uncertainty budget ({comparison['classification']})."
            reasons.append(reason)
            first_failing = first_failing or observable
    if first_failing is not None:
        return "FAIL", first_failing, reasons
    return "PASS", None, []


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_protocol(
    *,
    args: argparse.Namespace,
    gmx: Path,
    system_id: str,
    protocol: dict[str, object],
    system_root: Path,
    commands: list[dict[str, object]],
) -> dict[str, object]:
    ensemble = str(protocol["ensemble"])
    fixture_root = fixture_dir(system_id)
    inputs_dir = system_root / "inputs"
    logs_dir = system_root / "logs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = base_env(args)

    per_layout_runs: dict[str, list[dict[str, object]]] = {"cpu": [], "gpu": []}
    for replica_index in range(1, args.replicas + 1):
        seed = 173529 + replica_index - 1
        equil_mdp_path = inputs_dir / f"replica_{replica_index:02d}_{ensemble}_equil.mdp"
        prod_mdp_path = inputs_dir / f"replica_{replica_index:02d}_{ensemble}_prod.mdp"
        write_text(
            equil_mdp_path,
            make_gate_g_mdp(
                system_id=system_id,
                ensemble=ensemble,
                duration_ps=args.equil_ps,
                sample_interval=args.sample_interval,
                phase="equil",
                seed=seed,
                args=args,
            ),
        )
        write_text(
            prod_mdp_path,
            make_gate_g_mdp(
                system_id=system_id,
                ensemble=ensemble,
                duration_ps=args.prod_ps,
                sample_interval=args.sample_interval,
                phase="prod",
                seed=seed,
                args=args,
            ),
        )

        for layout in ("cpu", "gpu"):
            replica_root = system_root / f"replica_{replica_index:02d}" / layout
            replica_root.mkdir(parents=True, exist_ok=True)
            equil_deffnm = replica_root / "equil"
            prod_deffnm = replica_root / "prod"
            run_grompp(
                gmx=gmx,
                mdp_path=equil_mdp_path,
                conf_path=fixture_root / "initial_nve.gro",
                top_path=fixture_root / "topol.top",
                tpr_path=equil_deffnm.with_suffix(".tpr"),
                mdout_path=replica_root / "equil_mdout.mdp",
                logs_dir=logs_dir,
                commands=commands,
                label=f"{system_id}_{layout}_replica_{replica_index:02d}_grompp_equil",
                env=env,
            )
            md_argv = [str(gmx), "mdrun", *(mdrun_args_cpu(args, equil_deffnm) if layout == "cpu" else mdrun_args_gpu(args, equil_deffnm))]
            run_md(
                gmx=gmx,
                argv=md_argv,
                env=env,
                logs_dir=logs_dir,
                commands=commands,
                label=f"{system_id}_{layout}_replica_{replica_index:02d}_mdrun_equil",
                args=args,
            )

            run_grompp(
                gmx=gmx,
                mdp_path=prod_mdp_path,
                conf_path=equil_deffnm.with_suffix(".gro"),
                top_path=fixture_root / "topol.top",
                tpr_path=prod_deffnm.with_suffix(".tpr"),
                mdout_path=replica_root / "prod_mdout.mdp",
                logs_dir=logs_dir,
                commands=commands,
                label=f"{system_id}_{layout}_replica_{replica_index:02d}_grompp_prod",
                env=env,
                checkpoint_path=equil_deffnm.with_suffix(".cpt"),
            )
            prod_result = run_md(
                gmx=gmx,
                argv=[str(gmx), "mdrun", *(mdrun_args_cpu(args, prod_deffnm) if layout == "cpu" else mdrun_args_gpu(args, prod_deffnm))],
                env=env,
                logs_dir=logs_dir,
                commands=commands,
                label=f"{system_id}_{layout}_replica_{replica_index:02d}_mdrun_prod",
                args=args,
            )
            observable_path = replica_root / "production_observables.xvg"
            observables = collect_run_observables(gmx, prod_deffnm, observable_path)
            replica_payload = {
                "replica_index": replica_index,
                "seed": seed,
                "layout": layout,
                "ensemble": ensemble,
                "equil_deffnm": str(equil_deffnm),
                "prod_deffnm": str(prod_deffnm),
                "layout_report": prod_result["layout_report"],
                "observables": observables,
                "production_observables_xvg": str(observable_path),
            }
            write_json(replica_root / "replica_summary.json", replica_payload)
            per_layout_runs[layout].append(replica_payload)

    layout_aggregates: dict[str, dict[str, object]] = {}
    for layout, runs in per_layout_runs.items():
        observables: dict[str, dict[str, object]] = {}
        for term in REQUESTED_OBSERVABLES:
            observable_runs = []
            for run in runs:
                observable = run["observables"][term]
                if observable.get("available"):
                    observable_runs.append(
                        {
                            "replica_index": run["replica_index"],
                            "mean": observable["mean"],
                            "block_sem": observable["block_sem"],
                            "block_span": observable["block_span"],
                            "abs_block_drift": observable["abs_block_drift"],
                            "values": observable["values"],
                        }
                    )
            observables[term] = aggregate_replicates(observable_runs)
        layout_aggregates[layout] = {
            "replica_count": len(runs),
            "observables": observables,
        }

    comparisons = {
        term: compare_layout_observable(layout_aggregates["cpu"]["observables"][term], layout_aggregates["gpu"]["observables"][term])
        for term in REQUESTED_OBSERVABLES
    }
    save_observable_table(system_root / "summaries" / "observable_comparison.tsv", comparisons)

    status, first_failing, reasons = system_status(tuple(protocol["required_observables"]), comparisons)
    replica_variability = {
        layout: {
            term: {
                "std_of_means": aggregate["std_of_means"],
                "sem_of_means": aggregate["sem_of_means"],
                "mean_block_span": aggregate["mean_block_span"],
            }
            for term, aggregate in layout_aggregates[layout]["observables"].items()
            if aggregate.get("available")
        }
        for layout in ("cpu", "gpu")
    }
    system_result = {
        "system_id": system_id,
        "ensemble": ensemble,
        "status": status,
        "first_failing_observable": first_failing,
        "failure_reasons": reasons,
        "required_observables": protocol["required_observables"],
        "optional_observables": protocol["optional_observables"],
        "msd_assessment": {
            "relevant": bool(protocol["msd_relevant"]),
            "status": "not_evaluated" if not protocol["msd_relevant"] else "pending",
            "reason": protocol["msd_reason"],
        },
        "layout_aggregates": sanitize_layout_aggregates(layout_aggregates),
        "observable_comparisons": comparisons,
        "replica_variability": replica_variability,
        "artifacts": {
            "system_root": str(system_root),
            "observable_comparison_tsv": str(system_root / "summaries" / "observable_comparison.tsv"),
        },
    }
    write_json(system_root / "summaries" / "system_result.json", system_result)
    return system_result


def build_manifest_markdown(manifest: dict[str, object]) -> str:
    lines = [
        "# Gate G Long-Horizon Ensemble Validation",
        "",
        f"- Verdict: `{manifest['status']}`",
        f"- Gate H allowed: `{manifest['gate_h_allowed']}`",
        f"- Long-run baseline: `{manifest['baseline_policy']}`",
        f"- Replica count per layout: `{manifest['replicas']}`",
        f"- Equilibration / production: `{manifest['equil_ps']} ps / {manifest['prod_ps']} ps`",
        f"- Sampling limitation note: {manifest['sampling_limitation_note']}",
        "",
        "## Systems",
    ]
    for system in manifest["systems"]:
        lines.append(f"- `{system['system_id']}` `{system['ensemble']}`: `{system['status']}`")
        if system["first_failing_observable"] is not None:
            lines.append(f"  First failing observable: `{system['first_failing_observable']}`")
    return "\n".join(lines) + "\n"


def write_blocker_manifest(
    out_root: Path,
    *,
    gate_a_manifest: dict[str, object],
    gate_e_manifest: dict[str, object],
    gate_f_manifest: dict[str, object],
    reason: str,
) -> None:
    manifest = {
        "status": "BLOCKER",
        "gate_h_allowed": False,
        "artifact_root": str(out_root),
        "gate_a_status": gate_a_manifest.get("status"),
        "gate_e_status": gate_e_manifest.get("status"),
        "gate_f_status": gate_f_manifest.get("status"),
        "blocking_reasons": [reason],
        "systems": [],
        "first_failing_observable": {"field": "upstream_prerequisites", "details": reason},
    }
    write_json(out_root / "gate_g_manifest.json", manifest)
    write_text(
        out_root / "gate_g_manifest.md",
        "\n".join(
            [
                "# Gate G Long Ensemble",
                "",
                "- Status: BLOCKER",
                "- Gate H allowed: False",
                f"- Gate A status: `{manifest['gate_a_status']}`",
                f"- Gate E status: `{manifest['gate_e_status']}`",
                f"- Gate F status: `{manifest['gate_f_status']}`",
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
    validate_args(args)

    out_root = Path(args.out).resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    gate_a_manifest = load_json(Path(args.gate_a_manifest))
    gate_e_manifest = load_json(Path(args.gate_e_manifest))
    gate_f_manifest = load_json(Path(args.gate_f_manifest))
    try:
        validate_gate_chain(gate_a_manifest, gate_e_manifest, gate_f_manifest)
    except ValueError as exc:
        write_blocker_manifest(
            out_root,
            gate_a_manifest=gate_a_manifest,
            gate_e_manifest=gate_e_manifest,
            gate_f_manifest=gate_f_manifest,
            reason=str(exc),
        )
        return

    gmx = Path(args.gmx).resolve()
    maybe_build(args, gmx)

    version_text = capture_output([str(gmx), "--version"], cwd=REPO_ROOT, env=os.environ.copy())

    commands: list[dict[str, object]] = []
    system_results: list[dict[str, object]] = []
    for system_id, protocol in SYSTEM_PROTOCOLS.items():
        system_root = out_root / system_id
        result = run_protocol(
            args=args,
            gmx=gmx,
            system_id=system_id,
            protocol=protocol,
            system_root=system_root,
            commands=commands,
        )
        system_results.append(result)
        write_commands_script(system_root / "run_commands.sh", commands)
        commands.clear()

    status = "PASS" if all(system["status"] == "PASS" for system in system_results) else "FAIL"
    first_failure = next(
        (
            {
                "system_id": system["system_id"],
                "observable": system["first_failing_observable"],
            }
            for system in system_results
            if system["first_failing_observable"] is not None
        ),
        None,
    )
    manifest = {
        "status": status,
        "gate_h_allowed": status == "PASS",
        "baseline_policy": "Controlled CPU exact-r-RESPA long-run baselines derived from the Gate A mechanical path; no trajectory-identity claim is used.",
        "replicas": args.replicas,
        "equil_ps": args.equil_ps,
        "prod_ps": args.prod_ps,
        "sample_interval_base_steps": args.sample_interval,
        "single_rank": True,
        "dlb": "no",
        "npme_flag_used": args.npme is not None,
        "npme_requested": args.npme,
        "precision_mode": parse_precision_mode(version_text),
        "gpu_support": parse_gpu_support(version_text),
        "unsupported_feature_avoidance": {
            "domain_decomposition": False,
            "constraints": "none",
            "comm_mode": "none",
            "rerun_used": False,
            "trajectory_identity_not_required": True,
        },
        "sampling_limitation_note": "Production windows are long enough for temperature/pressure/box/potential summaries, but too short and too small-box to support a defensible MSD/diffusion claim.",
        "systems": system_results,
        "first_failing_observable": first_failure,
        "recommendation": (
            "Gate H is allowed."
            if status == "PASS"
            else "Gate H must remain blocked until the first failing observable is reconciled against the CPU exact-r-RESPA baseline."
        ),
        "artifact_root": str(out_root),
    }
    write_json(out_root / "gate_g_manifest.json", manifest)
    write_text(out_root / "gate_g_manifest.md", build_manifest_markdown(manifest))


if __name__ == "__main__":
    main()
