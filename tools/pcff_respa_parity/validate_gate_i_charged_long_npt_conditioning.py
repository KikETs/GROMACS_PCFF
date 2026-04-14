from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from freeze_gate_a_oracle import DEFAULT_GMX, base_env, command_record, env_delta, write_commands_script, write_text
from validate_gate_c_nb_bonded_gpu import DEFAULT_GATE_A_MANIFEST
from validate_gate_e_update_gpu import parse_layout_report
from validate_gate_g_long_ensemble import (
    BLOCK_COUNT,
    DEFAULT_GATE_E_MANIFEST,
    DEFAULT_GATE_F_MANIFEST,
    aggregate_replicates,
    collect_run_observables,
    exact_respa_common_mdp,
    mean,
    mdrun_args_cpu,
    run_grompp,
    run_md,
    sample_std,
    steps_from_ps,
    validate_gate_chain,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_G_MANIFEST = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_g_long_ensemble_validation"
    / "gate_g_manifest.json"
)
DEFAULT_SCAFFOLD_MANIFEST = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_h_fixture_scaffold"
    / "gate_h_dense_salt_polymer_2x2x2"
    / "fixture_manifest.json"
)
DEFAULT_PERF_REFERENCE_LOG = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_h_transport_validation_large_medium"
    / "gate_h_dense_salt_polymer_2x2x2"
    / "replica_01"
    / "cpu"
    / "prod.log"
)
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_i_charged_long_npt_conditioning"

DT_PS = 0.0005
EXACT_RESPA_FACTOR = 4
DEFAULT_EQ_PS = 250.0
DEFAULT_PROD_PS = 1000.0
DEFAULT_REPLICAS = 3
DEFAULT_SAMPLE_INTERVAL = EXACT_RESPA_FACTOR * 100
DEFAULT_TEMP_K = 300.0
DEFAULT_PRESSURE_BAR = 1.0
DEFAULT_TAU_T_PS = 0.5
DEFAULT_TAU_P_PS = 5.0
DEFAULT_COMPRESSIBILITY_BAR_INV = 4.5e-5

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
PRIMARY_GATE_OBSERVABLES = ("Density", "Volume")
SUPPORT_OBSERVABLES = ("Temperature", "Pressure", "Potential", "Box-X", "Box-Y", "Box-Z")

GATE_ID = "gate_i_cpu_exact_charged_long_npt_conditioning"
SCHEMA_NAME = "gate_i_charged_long_npt_conditioning"
SCHEMA_VERSION = 1
PERFORMANCE_RE = re.compile(
    r"Performance:\s+(?P<ns_per_day>[0-9.]+)\s+(?P<hour_per_ns>[0-9.]+)\s+(?P<ms_per_step>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze and optionally execute the CPU-only exact-r-RESPA charged long-NPT conditioning gate."
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument(
        "--gate-a-manifest",
        default=str(DEFAULT_GATE_A_MANIFEST),
        help="Path to the Gate A CPU oracle manifest.",
    )
    parser.add_argument(
        "--gate-e-manifest",
        default=str(DEFAULT_GATE_E_MANIFEST),
        help="Path to the Gate E manifest.",
    )
    parser.add_argument(
        "--gate-f-manifest",
        default=str(DEFAULT_GATE_F_MANIFEST),
        help="Path to the Gate F manifest.",
    )
    parser.add_argument(
        "--gate-g-manifest",
        default=str(DEFAULT_GATE_G_MANIFEST),
        help="Path to the Gate G manifest.",
    )
    parser.add_argument(
        "--scaffold-manifest",
        default=str(DEFAULT_SCAFFOLD_MANIFEST),
        help="Charged large/medium scaffold manifest path.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks.")
    parser.add_argument("--ntomp", type=int, default=1, help="OpenMP threads.")
    parser.add_argument("--npme", type=int, default=None, help="Optional explicit -npme value; omitted by default.")
    parser.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS, help="Replica count.")
    parser.add_argument("--equil-ps", type=float, default=DEFAULT_EQ_PS, help="Equilibration duration in ps.")
    parser.add_argument("--prod-ps", type=float, default=DEFAULT_PROD_PS, help="Production duration in ps.")
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=DEFAULT_SAMPLE_INTERVAL,
        help="Energy/log sampling interval in base steps. Must be a multiple of the exact-r-RESPA factor.",
    )
    parser.add_argument("--temperature-k", type=float, default=DEFAULT_TEMP_K, help="Target temperature.")
    parser.add_argument("--pressure-bar", type=float, default=DEFAULT_PRESSURE_BAR, help="Target pressure.")
    parser.add_argument("--tau-t-ps", type=float, default=DEFAULT_TAU_T_PS, help="Thermostat time constant.")
    parser.add_argument("--tau-p-ps", type=float, default=DEFAULT_TAU_P_PS, help="Barostat time constant.")
    parser.add_argument(
        "--compressibility-bar-inv",
        type=float,
        default=DEFAULT_COMPRESSIBILITY_BAR_INV,
        help="Compressibility for isotropic C-rescale pressure coupling.",
    )
    parser.add_argument(
        "--density-mean-abs-block-drift-rel-max",
        type=float,
        default=0.05,
        help="Maximum allowed mean relative density block drift across replicas.",
    )
    parser.add_argument(
        "--density-max-replica-abs-block-drift-rel-max",
        type=float,
        default=0.08,
        help="Maximum allowed worst-replica relative density block drift.",
    )
    parser.add_argument(
        "--density-cross-replica-span-rel-max",
        type=float,
        default=0.05,
        help="Maximum allowed relative density span across replica means.",
    )
    parser.add_argument(
        "--volume-mean-abs-block-drift-rel-max",
        type=float,
        default=0.05,
        help="Maximum allowed mean relative volume block drift across replicas.",
    )
    parser.add_argument(
        "--volume-max-replica-abs-block-drift-rel-max",
        type=float,
        default=0.08,
        help="Maximum allowed worst-replica relative volume block drift.",
    )
    parser.add_argument(
        "--volume-cross-replica-span-rel-max",
        type=float,
        default=0.05,
        help="Maximum allowed relative volume span across replica means.",
    )
    parser.add_argument(
        "--temperature-mean-abs-error-k-max",
        type=float,
        default=20.0,
        help="Maximum allowed absolute temperature mean error in kelvin.",
    )
    parser.add_argument(
        "--performance-reference-log",
        default=str(DEFAULT_PERF_REFERENCE_LOG),
        help="Optional checked-in CPU log used only to estimate wall-clock cost.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the predeclared contract and a pending manifest without running the expensive campaign.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed stage artifacts under --out instead of deleting the output root.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if args.ntmpi != 1:
        raise ValueError("Gate I is restricted to single-rank runs (ntmpi=1).")
    if args.replicas < 3:
        raise ValueError("Gate I requires at least 3 replicas; otherwise cross-replica conditioning is too weak.")
    if args.sample_interval <= 0 or args.sample_interval % EXACT_RESPA_FACTOR != 0:
        raise ValueError("sample-interval must be a positive multiple of the exact-r-RESPA factor.")
    for name, duration_ps in (("equil-ps", args.equil_ps), ("prod-ps", args.prod_ps)):
        steps = steps_from_ps(duration_ps)
        if steps <= 0 or not math.isclose(steps * DT_PS, duration_ps, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{name} must be representable as a positive integer number of base steps.")
        if steps % EXACT_RESPA_FACTOR != 0:
            raise ValueError(f"{name} must be a multiple of the exact-r-RESPA factor.")


def validate_prerequisites(
    gate_a_manifest: dict[str, object],
    gate_e_manifest: dict[str, object],
    gate_f_manifest: dict[str, object],
    gate_g_manifest: dict[str, object],
    scaffold_manifest: dict[str, object],
) -> None:
    validate_gate_chain(gate_a_manifest, gate_e_manifest, gate_f_manifest)
    if gate_g_manifest.get("status") != "PASS":
        raise ValueError("Gate G is not PASS; Gate I should not proceed before narrow exact ensemble prerequisites are closed.")
    if str(scaffold_manifest.get("derived_system")) != "gate_h_dense_salt_polymer_2x2x2":
        raise ValueError("Gate I is frozen only for the charged gate_h_dense_salt_polymer_2x2x2 scaffold.")
    if not bool(scaffold_manifest.get("tp0_size_fit")) or not bool(scaffold_manifest.get("tp0_box_fit")):
        raise ValueError("Gate I requires the charged large/medium scaffold to already satisfy TP0 size and box-fit prerequisites.")


def make_gate_i_npt_mdp(*, duration_ps: float, sample_interval: int, phase: str, seed: int, args: argparse.Namespace) -> str:
    nsteps = steps_from_ps(duration_ps)
    thermostat = (
        "tcoupl                  = v-rescale\n"
        "tc-grps                 = System\n"
        f"tau-t                   = {args.tau_t_ps:.3f}\n"
        f"ref-t                   = {args.temperature_k:.3f}\n"
        f"nsttcouple              = {EXACT_RESPA_FACTOR}\n"
    )
    barostat = (
        "pcoupl                  = c-rescale\n"
        "pcoupltype              = isotropic\n"
        f"tau-p                   = {args.tau_p_ps:.3f}\n"
        f"ref-p                   = {args.pressure_bar:.3f}\n"
        f"compressibility         = {args.compressibility_bar_inv:.7g}\n"
        f"nstpcouple              = {EXACT_RESPA_FACTOR}\n"
        "refcoord-scaling        = no\n"
    )
    velocity = (
        "gen-vel                 = yes\n"
        f"gen-temp                = {args.temperature_k:.3f}\n"
        f"gen-seed                = {seed}\n"
        if phase == "equil"
        else "gen-vel                 = no\n"
    )
    return (
        f"title                   = gate i charged long npt {phase} exact respa\n"
        + exact_respa_common_mdp(nsteps, sample_interval)
        + thermostat
        + barostat
        + velocity
    )


def file_exists_and_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def grompp_complete(tpr_path: Path, mdout_path: Path) -> bool:
    return file_exists_and_nonempty(tpr_path) and file_exists_and_nonempty(mdout_path)


def mdrun_complete(deffnm: Path) -> bool:
    required = (
        deffnm.with_suffix(".edr"),
        deffnm.with_suffix(".gro"),
        deffnm.with_suffix(".cpt"),
        deffnm.with_suffix(".log"),
    )
    return all(file_exists_and_nonempty(path) for path in required)


def remove_incomplete_mdrun_outputs(deffnm: Path, preserve_paths: tuple[Path, ...] = ()) -> None:
    preserve = {path.resolve() for path in preserve_paths}
    for suffix in (".cpt", ".edr", ".gro", ".log", ".trr", ".xtc"):
        path = deffnm.with_suffix(suffix)
        if path.exists() and path.resolve() not in preserve:
            path.unlink()


def checkpoint_step(gmx: Path, checkpoint_path: Path) -> int:
    if not file_exists_and_nonempty(checkpoint_path):
        return -1
    completed = subprocess.run(
        [str(gmx), "dump", "-cp", str(checkpoint_path)],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return -1
    match = re.search(r"^step = (\d+)$", completed.stdout, flags=re.MULTILINE)
    return int(match.group(1)) if match is not None else -1


def best_resume_checkpoint(gmx: Path, deffnm: Path) -> Path | None:
    candidates = []
    current = deffnm.with_suffix(".cpt")
    previous = deffnm.with_name(f"{deffnm.name}_prev.cpt")
    for path in (current, previous):
        step = checkpoint_step(gmx, path)
        if step >= 0:
            candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].name))
    return candidates[-1][1]


def run_or_resume_grompp(
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
    if grompp_complete(tpr_path, mdout_path):
        return
    run_grompp(
        gmx=gmx,
        mdp_path=mdp_path,
        conf_path=conf_path,
        top_path=top_path,
        tpr_path=tpr_path,
        mdout_path=mdout_path,
        logs_dir=logs_dir,
        commands=[],
        label=label,
        env=env,
        checkpoint_path=checkpoint_path,
    )


def run_or_resume_md(
    *,
    gmx: Path,
    argv: list[str],
    deffnm: Path,
    env: dict[str, str],
    logs_dir: Path,
    commands: list[dict[str, object]],
    label: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    stdout_path = logs_dir / f"{label}.stdout"
    stderr_path = logs_dir / f"{label}.stderr"
    effective_argv = list(argv)
    resume_checkpoint = best_resume_checkpoint(gmx, deffnm) if args.resume else None
    preserve_partial_outputs = resume_checkpoint == deffnm.with_suffix(".cpt")
    if not mdrun_complete(deffnm):
        if resume_checkpoint is not None:
            effective_argv.extend(["-cpi", str(resume_checkpoint)])
            if not preserve_partial_outputs:
                remove_incomplete_mdrun_outputs(deffnm, preserve_paths=(resume_checkpoint,))
        else:
            remove_incomplete_mdrun_outputs(deffnm)
    commands.append(
        command_record(
            label,
            effective_argv,
            cwd=REPO_ROOT,
            env_overrides=env_delta(env, os.environ),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    )
    if mdrun_complete(deffnm):
        return {
            "run_id": label,
            "argv": effective_argv,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "returncode": 0,
            "layout_report": parse_layout_report(stdout_path, stderr_path, args),
            "resumed_from_existing_artifacts": True,
        }
    payload = run_md(
        gmx=gmx,
        argv=effective_argv,
        env=env,
        logs_dir=logs_dir,
        commands=[],
        label=label,
        args=args,
    )
    payload["resumed_from_existing_artifacts"] = False
    return payload


def performance_reference(path: Path, args: argparse.Namespace) -> dict[str, object]:
    payload = {
        "reference_log": str(path),
        "available": False,
    }
    if not path.exists():
        return payload
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = PERFORMANCE_RE.search(text)
    if match is None:
        return payload
    ns_per_day = float(match.group("ns_per_day"))
    hour_per_ns = float(match.group("hour_per_ns"))
    total_ns = (args.equil_ps + args.prod_ps) / 1000.0
    per_replica_hours = hour_per_ns * total_ns
    payload.update(
        {
            "available": True,
            "ns_per_day": ns_per_day,
            "hour_per_ns": hour_per_ns,
            "estimated_hours_per_replica": per_replica_hours,
            "estimated_hours_serial_for_all_replicas": per_replica_hours * args.replicas,
            "estimated_hours_if_replicas_run_fully_in_parallel": per_replica_hours,
        }
    )
    return payload


def acceptance_criteria(args: argparse.Namespace) -> dict[str, object]:
    return {
        "density_mean_abs_block_drift_rel_max": args.density_mean_abs_block_drift_rel_max,
        "density_max_replica_abs_block_drift_rel_max": args.density_max_replica_abs_block_drift_rel_max,
        "density_cross_replica_span_rel_max": args.density_cross_replica_span_rel_max,
        "volume_mean_abs_block_drift_rel_max": args.volume_mean_abs_block_drift_rel_max,
        "volume_max_replica_abs_block_drift_rel_max": args.volume_max_replica_abs_block_drift_rel_max,
        "volume_cross_replica_span_rel_max": args.volume_cross_replica_span_rel_max,
        "temperature_mean_abs_error_k_max": args.temperature_mean_abs_error_k_max,
        "required_conditioned_state_files": [
            "prod.gro",
            "prod.cpt",
            "prod.tpr",
            "replica_summary.json",
        ],
        "replicas_min": 3,
        "relative_metric_note": (
            "Relative block-drift and cross-replica span metrics are normalized by the absolute aggregate mean of the same observable."
        ),
        "gate_scope_note": (
            "These are predeclared conditioning criteria only. A Gate I PASS closes the density/volume conditioning blocker, not TP0 production or transport readiness."
        ),
    }


def build_contract(
    *,
    args: argparse.Namespace,
    scaffold_manifest: dict[str, object],
    prerequisites: dict[str, object],
    perf_ref: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "gate_id": GATE_ID,
        "status": "DECLARED",
        "execution_policy": {
            "single_rank": True,
            "cpu_only": True,
            "exact_respa": True,
            "ensemble": "npt",
            "resume_supported": True,
            "prepare_only_supported": True,
        },
        "system": {
            "system_id": scaffold_manifest["derived_system"],
            "seed_system": scaffold_manifest["seed_system"],
            "natoms": scaffold_manifest["natoms"],
            "box_nm": scaffold_manifest["box_nm"],
            "fixture_manifest": str(Path(args.scaffold_manifest).resolve()),
            "tp0_size_fit": scaffold_manifest["tp0_size_fit"],
            "tp0_box_fit": scaffold_manifest["tp0_box_fit"],
        },
        "prerequisites": prerequisites,
        "run_settings": {
            "replicas": args.replicas,
            "equil_ps": args.equil_ps,
            "prod_ps": args.prod_ps,
            "sample_interval_base_steps": args.sample_interval,
            "temperature_k": args.temperature_k,
            "pressure_bar": args.pressure_bar,
            "tau_t_ps": args.tau_t_ps,
            "tau_p_ps": args.tau_p_ps,
            "compressibility_bar_inv": args.compressibility_bar_inv,
            "ntmpi": args.ntmpi,
            "ntomp": args.ntomp,
            "mdrun_shape": "nb cpu / bonded cpu / pme cpu / update cpu / reprod / single rank",
        },
        "requested_observables": list(REQUESTED_OBSERVABLES),
        "primary_gate_observables": list(PRIMARY_GATE_OBSERVABLES),
        "support_observables": list(SUPPORT_OBSERVABLES),
        "acceptance_criteria": acceptance_criteria(args),
        "conditioned_state_handoff_rule": {
            "selection": "Choose the replica whose density and volume means are closest to the aggregate conditioning center after the gate passes.",
            "required_files": acceptance_criteria(args)["required_conditioned_state_files"],
            "non_claimable_statement": "The conditioned-state handoff is only an input to later TP0-scale transport campaigns; it is not production approval.",
        },
        "performance_reference": perf_ref,
        "scope_statement": (
            "Gate I is the concrete next gate for the remaining CPU exact-r-RESPA blocker: charged large/medium long-NPT density/volume conditioning on gate_h_dense_salt_polymer_2x2x2."
        ),
        "non_claims": [
            "A declared Gate I contract is not a passed gate.",
            "A Gate I PASS still does not imply conductivity-production readiness.",
            "A Gate I PASS would still not imply LAMMPS-vs-GROMACS transport parity.",
            "A Gate I PASS would still not imply TP0-scale production length or uncertainty closure.",
        ],
    }


def relative_value(value: float, reference: float) -> float:
    if math.isclose(reference, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        return 0.0 if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-12) else math.inf
    return abs(value) / abs(reference)


def aggregate_metric(aggregate: dict[str, object]) -> dict[str, object]:
    if not aggregate.get("available"):
        return {"available": False}
    mean_of_means = float(aggregate["mean_of_means"])
    replica_means = [float(replica["mean"]) for replica in aggregate["replicas"]]
    per_replica_abs_block_drift_rel = [
        relative_value(float(replica["abs_block_drift"]), float(replica["mean"])) for replica in aggregate["replicas"]
    ]
    return {
        "available": True,
        "mean_of_means": mean_of_means,
        "std_of_means": float(aggregate["std_of_means"]),
        "sem_of_means": float(aggregate["sem_of_means"]),
        "mean_abs_block_drift": float(aggregate["mean_abs_block_drift"]),
        "mean_abs_block_drift_rel": relative_value(float(aggregate["mean_abs_block_drift"]), mean_of_means),
        "replica_mean_span": max(replica_means) - min(replica_means) if replica_means else 0.0,
        "cross_replica_span_rel": relative_value(max(replica_means) - min(replica_means), mean_of_means) if replica_means else 0.0,
        "max_replica_abs_block_drift_rel": max(per_replica_abs_block_drift_rel) if per_replica_abs_block_drift_rel else 0.0,
        "per_replica_abs_block_drift_rel": per_replica_abs_block_drift_rel,
        "replica_means": replica_means,
        "replica_count": int(aggregate["replica_count"]),
    }


def sanitize_aggregate(aggregate: dict[str, object]) -> dict[str, object]:
    if not aggregate.get("available"):
        return {"available": False}
    sanitized = {key: value for key, value in aggregate.items() if key != "pooled_values"}
    sanitized["replicas"] = [{key: value for key, value in replica.items() if key != "values"} for replica in aggregate["replicas"]]
    return sanitized


def conditioned_state_candidate(replica_payload: dict[str, object], density_center: float, volume_center: float) -> dict[str, object]:
    observables = replica_payload["observables"]
    density_mean = float(observables["Density"]["mean"])
    volume_mean = float(observables["Volume"]["mean"])
    density_distance = relative_value(density_mean - density_center, density_center)
    volume_distance = relative_value(volume_mean - volume_center, volume_center)
    prod_deffnm = Path(replica_payload["prod_deffnm"])
    files = {
        "prod_gro": str(prod_deffnm.with_suffix(".gro")),
        "prod_cpt": str(prod_deffnm.with_suffix(".cpt")),
        "prod_tpr": str(prod_deffnm.with_suffix(".tpr")),
        "replica_summary": str(replica_payload["replica_summary_json"]),
    }
    return {
        "replica_index": replica_payload["replica_index"],
        "density_mean": density_mean,
        "volume_mean": volume_mean,
        "density_distance_rel": density_distance,
        "volume_distance_rel": volume_distance,
        "selection_score": density_distance + volume_distance,
        "files": files,
        "all_required_files_present": all(file_exists_and_nonempty(Path(path)) for path in files.values()),
    }


def pick_conditioned_state(replica_runs: list[dict[str, object]], density_center: float, volume_center: float) -> dict[str, object]:
    candidates = [conditioned_state_candidate(replica_payload, density_center, volume_center) for replica_payload in replica_runs]
    ordered = sorted(
        candidates,
        key=lambda item: (
            not item["all_required_files_present"],
            item["selection_score"],
            item["replica_index"],
        ),
    )
    return {
        "available": bool(ordered) and ordered[0]["all_required_files_present"],
        "selected": ordered[0] if ordered else None,
        "candidates": ordered,
    }


def failure_reasons_from_metrics(metrics: dict[str, dict[str, object]], args: argparse.Namespace) -> tuple[str | None, list[str]]:
    reasons: list[str] = []
    first_failing: str | None = None
    density = metrics["Density"]
    volume = metrics["Volume"]
    temperature = metrics["Temperature"]

    if not density.get("available"):
        reasons.append("Density observable is missing from the Gate I production output.")
        first_failing = first_failing or "Density.available"
    else:
        if density["mean_abs_block_drift_rel"] > args.density_mean_abs_block_drift_rel_max:
            reasons.append(
                f"Density mean relative block drift {density['mean_abs_block_drift_rel']:.6f} exceeds {args.density_mean_abs_block_drift_rel_max:.6f}."
            )
            first_failing = first_failing or "Density.mean_abs_block_drift_rel"
        if density["max_replica_abs_block_drift_rel"] > args.density_max_replica_abs_block_drift_rel_max:
            reasons.append(
                f"Density worst-replica relative block drift {density['max_replica_abs_block_drift_rel']:.6f} exceeds {args.density_max_replica_abs_block_drift_rel_max:.6f}."
            )
            first_failing = first_failing or "Density.max_replica_abs_block_drift_rel"
        if density["cross_replica_span_rel"] > args.density_cross_replica_span_rel_max:
            reasons.append(
                f"Density cross-replica relative span {density['cross_replica_span_rel']:.6f} exceeds {args.density_cross_replica_span_rel_max:.6f}."
            )
            first_failing = first_failing or "Density.cross_replica_span_rel"

    if not volume.get("available"):
        reasons.append("Volume observable is missing from the Gate I production output.")
        first_failing = first_failing or "Volume.available"
    else:
        if volume["mean_abs_block_drift_rel"] > args.volume_mean_abs_block_drift_rel_max:
            reasons.append(
                f"Volume mean relative block drift {volume['mean_abs_block_drift_rel']:.6f} exceeds {args.volume_mean_abs_block_drift_rel_max:.6f}."
            )
            first_failing = first_failing or "Volume.mean_abs_block_drift_rel"
        if volume["max_replica_abs_block_drift_rel"] > args.volume_max_replica_abs_block_drift_rel_max:
            reasons.append(
                f"Volume worst-replica relative block drift {volume['max_replica_abs_block_drift_rel']:.6f} exceeds {args.volume_max_replica_abs_block_drift_rel_max:.6f}."
            )
            first_failing = first_failing or "Volume.max_replica_abs_block_drift_rel"
        if volume["cross_replica_span_rel"] > args.volume_cross_replica_span_rel_max:
            reasons.append(
                f"Volume cross-replica relative span {volume['cross_replica_span_rel']:.6f} exceeds {args.volume_cross_replica_span_rel_max:.6f}."
            )
            first_failing = first_failing or "Volume.cross_replica_span_rel"

    if not temperature.get("available"):
        reasons.append("Temperature observable is missing from the Gate I production output.")
        first_failing = first_failing or "Temperature.available"
    else:
        temp_error = abs(float(temperature["mean_of_means"]) - args.temperature_k)
        if temp_error > args.temperature_mean_abs_error_k_max:
            reasons.append(
                f"Temperature mean absolute error {temp_error:.6f} K exceeds {args.temperature_mean_abs_error_k_max:.6f} K."
            )
            first_failing = first_failing or "Temperature.mean_abs_error_k"

    return first_failing, reasons


def manifest_markdown(manifest: dict[str, object]) -> str:
    lines = [
        "# Gate I Charged Long-NPT Conditioning",
        "",
        f"- Status: `{manifest['status']}`",
        f"- System: `{manifest['system_id']}`",
        f"- Scope: {manifest['scope']}",
        f"- Contract: `{manifest['contract_path']}`",
        f"- Replicas / horizon: `{manifest['run_settings']['replicas']}` / `{manifest['run_settings']['equil_ps']} ps + {manifest['run_settings']['prod_ps']} ps`",
        "",
        "## Non-Claims",
    ]
    for item in manifest["non_claims"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Failure Reasons")
    for reason in manifest["failure_reasons"] or ["None"]:
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_pending_manifest(out_root: Path, contract: dict[str, object], reason: str) -> None:
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "gate_id": GATE_ID,
        "status": "DECLARED_PENDING_EXECUTION",
        "system_id": contract["system"]["system_id"],
        "scope": "CPU-only single-rank exact-r-RESPA charged large/medium long-NPT conditioning gate",
        "contract_path": str(out_root / "gate_i_contract.json"),
        "prepared_only": True,
        "executed": False,
        "failure_reasons": [reason],
        "first_failing_metric": "execution_state.not_run",
        "run_settings": contract["run_settings"],
        "acceptance_criteria": contract["acceptance_criteria"],
        "performance_reference": contract["performance_reference"],
        "non_claims": contract["non_claims"],
        "conditioned_state_handoff": {
            "available": False,
            "selected": None,
            "candidates": [],
        },
    }
    write_json(out_root / "gate_i_manifest.json", manifest)
    write_text(out_root / "gate_i_manifest.md", manifest_markdown(manifest))


def main() -> int:
    args = parse_args()
    validate_args(args)

    gate_a_manifest = load_json(Path(args.gate_a_manifest))
    gate_e_manifest = load_json(Path(args.gate_e_manifest))
    gate_f_manifest = load_json(Path(args.gate_f_manifest))
    gate_g_manifest = load_json(Path(args.gate_g_manifest))
    scaffold_manifest = load_json(Path(args.scaffold_manifest))
    validate_prerequisites(gate_a_manifest, gate_e_manifest, gate_f_manifest, gate_g_manifest, scaffold_manifest)

    out_root = Path(args.out).resolve()
    if out_root.exists() and not args.resume and not args.prepare_only:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    prereq_payload = {
        "gate_a_status": gate_a_manifest.get("status"),
        "gate_e_status": gate_e_manifest.get("status"),
        "gate_f_status": gate_f_manifest.get("status"),
        "gate_g_status": gate_g_manifest.get("status"),
    }
    perf_ref = performance_reference(Path(args.performance_reference_log), args)
    contract = build_contract(args=args, scaffold_manifest=scaffold_manifest, prerequisites=prereq_payload, perf_ref=perf_ref)
    write_json(out_root / "gate_i_contract.json", contract)

    if args.prepare_only:
        write_pending_manifest(
            out_root,
            contract,
            "Checked-in repository now freezes the Gate I contract, but no completed CPU-only exact long-NPT campaign has been checked in yet.",
        )
        return 0

    inputs_dir = out_root / "inputs"
    logs_dir = out_root / "logs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    gmx = Path(args.gmx)
    top_path = Path(str(scaffold_manifest["artifacts"]["topology"]))
    gro_path = Path(str(scaffold_manifest["artifacts"]["gro"]))
    env = base_env(args)
    commands: list[dict[str, object]] = []
    replica_runs: list[dict[str, object]] = []

    for replica_index in range(1, args.replicas + 1):
        seed = 61001 + replica_index - 1
        replica_input_root = inputs_dir / f"replica_{replica_index:02d}"
        replica_input_root.mkdir(parents=True, exist_ok=True)
        equil_mdp = replica_input_root / "equil.mdp"
        prod_mdp = replica_input_root / "prod.mdp"
        write_text(
            equil_mdp,
            make_gate_i_npt_mdp(
                duration_ps=args.equil_ps,
                sample_interval=args.sample_interval,
                phase="equil",
                seed=seed,
                args=args,
            ),
        )
        write_text(
            prod_mdp,
            make_gate_i_npt_mdp(
                duration_ps=args.prod_ps,
                sample_interval=args.sample_interval,
                phase="prod",
                seed=seed,
                args=args,
            ),
        )

        run_root = out_root / "cpu" / f"replica_{replica_index:02d}"
        run_root.mkdir(parents=True, exist_ok=True)
        equil_deffnm = run_root / "equil"
        prod_deffnm = run_root / "prod"

        run_or_resume_grompp(
            gmx=gmx,
            mdp_path=equil_mdp,
            conf_path=gro_path,
            top_path=top_path,
            tpr_path=equil_deffnm.with_suffix(".tpr"),
            mdout_path=run_root / "equil.mdout.mdp",
            logs_dir=logs_dir,
            commands=commands,
            label=f"cpu_replica_{replica_index:02d}_grompp_equil",
            env=env,
        )
        run_or_resume_md(
            gmx=gmx,
            argv=[str(gmx), "mdrun", *mdrun_args_cpu(args, equil_deffnm)],
            deffnm=equil_deffnm,
            env=env,
            logs_dir=logs_dir,
            commands=commands,
            label=f"cpu_replica_{replica_index:02d}_mdrun_equil",
            args=args,
        )

        run_or_resume_grompp(
            gmx=gmx,
            mdp_path=prod_mdp,
            conf_path=equil_deffnm.with_suffix(".gro"),
            top_path=top_path,
            tpr_path=prod_deffnm.with_suffix(".tpr"),
            mdout_path=run_root / "prod.mdout.mdp",
            logs_dir=logs_dir,
            commands=commands,
            label=f"cpu_replica_{replica_index:02d}_grompp_prod",
            env=env,
            checkpoint_path=equil_deffnm.with_suffix(".cpt"),
        )
        md_payload = run_or_resume_md(
            gmx=gmx,
            argv=[str(gmx), "mdrun", *mdrun_args_cpu(args, prod_deffnm)],
            deffnm=prod_deffnm,
            env=env,
            logs_dir=logs_dir,
            commands=commands,
            label=f"cpu_replica_{replica_index:02d}_mdrun_prod",
            args=args,
        )

        observables = collect_run_observables(gmx, prod_deffnm, run_root / "production_observables.xvg")
        replica_payload = {
            "replica_index": replica_index,
            "seed": seed,
            "prod_deffnm": str(prod_deffnm),
            "layout_report": md_payload["layout_report"],
            "observables": observables,
        }
        replica_summary_json = run_root / "replica_summary.json"
        replica_payload["replica_summary_json"] = str(replica_summary_json)
        write_json(replica_summary_json, replica_payload)
        replica_runs.append(replica_payload)

    aggregates = {
        term: aggregate_replicates([replica["observables"][term] for replica in replica_runs if replica["observables"][term].get("available")])
        for term in REQUESTED_OBSERVABLES
    }
    metrics = {term: aggregate_metric(aggregate) for term, aggregate in aggregates.items()}
    first_failing_metric, failure_reasons = failure_reasons_from_metrics(metrics, args)
    conditioned_state = pick_conditioned_state(
        replica_runs,
        metrics["Density"]["mean_of_means"] if metrics["Density"].get("available") else 0.0,
        metrics["Volume"]["mean_of_means"] if metrics["Volume"].get("available") else 0.0,
    )
    if not conditioned_state["available"]:
        failure_reasons.append("No conditioned-state handoff candidate has the required files.")
        first_failing_metric = first_failing_metric or "conditioned_state_handoff.available"

    status = "PASS" if not failure_reasons else "FAIL"

    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "gate_id": GATE_ID,
        "status": status,
        "system_id": scaffold_manifest["derived_system"],
        "scope": "CPU-only single-rank exact-r-RESPA charged large/medium long-NPT conditioning gate",
        "contract_path": str(out_root / "gate_i_contract.json"),
        "prepared_only": False,
        "executed": True,
        "run_settings": contract["run_settings"],
        "acceptance_criteria": contract["acceptance_criteria"],
        "failure_reasons": failure_reasons,
        "first_failing_metric": first_failing_metric,
        "primary_gate_metrics": {
            "Density": metrics["Density"],
            "Volume": metrics["Volume"],
            "Temperature": {
                **metrics["Temperature"],
                "target_temperature_k": args.temperature_k,
                "mean_abs_error_k": (
                    abs(float(metrics["Temperature"]["mean_of_means"]) - args.temperature_k)
                    if metrics["Temperature"].get("available")
                    else math.inf
                ),
            },
        },
        "support_metrics": {term: metrics[term] for term in SUPPORT_OBSERVABLES if term not in ("Temperature",)},
        "aggregates": {term: sanitize_aggregate(aggregate) for term, aggregate in aggregates.items()},
        "replica_summaries": [str(replica["replica_summary_json"]) for replica in replica_runs],
        "conditioned_state_handoff": conditioned_state,
        "performance_reference": perf_ref,
        "non_claims": contract["non_claims"],
    }
    write_json(out_root / "gate_i_manifest.json", manifest)
    write_text(out_root / "gate_i_manifest.md", manifest_markdown(manifest))
    write_commands_script(out_root / "run_commands.sh", commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
