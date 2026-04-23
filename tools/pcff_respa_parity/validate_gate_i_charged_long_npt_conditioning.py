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
DEFAULT_EQ_PS = 3000.0
DEFAULT_PROD_PS = 1000.0
DEFAULT_REPLICAS = 3
DEFAULT_SAMPLE_INTERVAL = EXACT_RESPA_FACTOR * 100
DEFAULT_TEMP_K = 300.0
DEFAULT_PRESSURE_BAR = 1.0
DEFAULT_TAU_T_PS = 0.5
DEFAULT_TAU_P_PS = 5.0
DEFAULT_COMPRESSIBILITY_BAR_INV = 4.5e-5
NATIVE_MULTI_ENV = "GMX_PCFF_EXACT_RESPA_NATIVE_MULTI"
NATIVE_MULTI_OWNER_FALLBACK_ENV = "GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_OWNER_STEP_FALLBACK"
NATIVE_MULTI_SPLIT_OWNER_ENV = "GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_SPLIT_OWNER_OUTPUTS"
EXACT_RESPA_UPDATE_OMP_ENV = "GMX_PCFF_EXACT_RESPA_UPDATE_OMP"
EXACT_RESPA_UPDATE_OMP_THREADS_ENV = "GMX_PCFF_EXACT_RESPA_UPDATE_OMP_THREADS"
EXACT_RESPA_UPDATE_DIRECT_FASTPATH_ENV = "GMX_PCFF_EXACT_RESPA_UPDATE_DIRECT_FASTPATH"
EXACT_RESPA_FUSED_INITIAL_DRIFT_ENV = "GMX_PCFF_EXACT_RESPA_FUSED_INITIAL_DRIFT"

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


def rehome_repo_artifact_path(path_value: str | Path) -> Path:
    """Map checked-in absolute artifact paths onto the current repository root."""

    candidate = Path(path_value)
    if candidate.exists():
        return candidate

    if candidate.is_absolute():
        parts = candidate.parts
        if "GROMACS_PCFF" in parts:
            repo_index = parts.index("GROMACS_PCFF")
            rebased = REPO_ROOT.joinpath(*parts[repo_index + 1 :])
            if rebased.exists():
                return rebased

    return candidate


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
    parser.add_argument(
        "--start-gro",
        default=None,
        help=(
            "Optional common preconditioned starting structure for the Gate I equilibration stage. "
            "When omitted, the scaffold manifest GRO is used."
        ),
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks.")
    parser.add_argument("--ntomp", type=int, default=1, help="OpenMP threads.")
    parser.add_argument(
        "--pin",
        choices=("off", "on", "auto"),
        default="off",
        help=(
            "GROMACS mdrun -pin policy for CPU-only Gate I runs. Default remains off to preserve "
            "the historical contract; use 'on' only with an explicit measured affinity shape."
        ),
    )
    parser.add_argument(
        "--pinoffset",
        type=int,
        default=None,
        help="Optional GROMACS mdrun -pinoffset value, only emitted when --pin is not off.",
    )
    parser.add_argument(
        "--pinstride",
        type=int,
        default=None,
        help="Optional GROMACS mdrun -pinstride value, only emitted when --pin is not off.",
    )
    parser.add_argument(
        "--native-multi-owner-mode",
        choices=("default", "owner_fallback", "full_owner_native", "split_owner_sidecar"),
        default="default",
        help=(
            "Owner-step native-multi runtime mode. "
            "'default' preserves compiled defaults; other modes pin the exact owner-step env contract explicitly."
        ),
    )
    parser.add_argument(
        "--allow-experimental-native-multi-probe",
        action="store_true",
        help=(
            "Permit full_owner_native or split_owner_sidecar for explicit performance probes. "
            "These modes are not accepted as Gate I exactness evidence until runtime parity is closed."
        ),
    )
    parser.add_argument(
        "--allow-experimental-update-probe",
        action="store_true",
        help=(
            "Permit default-off update experiments such as fused initial drift. "
            "These modes are recorded as probes and are not accepted as claimable Gate I evidence."
        ),
    )
    parser.add_argument(
        "--exact-respa-update-omp-mode",
        choices=("auto", "off", "on"),
        default="auto",
        help=(
            "Exact-r-RESPA CPU update OpenMP policy. The default preserves the runtime auto heuristic; "
            "use 'on' only for thread shapes where probe evidence supports it."
        ),
    )
    parser.add_argument(
        "--exact-respa-update-omp-threads",
        type=int,
        default=None,
        help=(
            "Optional cap for exact-r-RESPA CPU update OpenMP threads. This lets force/nonbonded use "
            "--ntomp while the update loop uses a smaller host-local thread count."
        ),
    )
    parser.add_argument(
        "--exact-respa-update-direct-fastpath-mode",
        choices=("auto", "off", "on"),
        default="auto",
        help=(
            "Exact-r-RESPA direct mobile-atom update fast path. The default preserves the compiled "
            "runtime default; use 'off' to force the conservative per-atom branch path."
        ),
    )
    parser.add_argument(
        "--exact-respa-fused-initial-drift-mode",
        choices=("auto", "off", "on"),
        default="auto",
        help=(
            "Exact-r-RESPA CPU initial half-kick plus drift fusion policy. The default preserves "
            "the runtime default; use 'on' only for thread shapes where probe evidence supports it."
        ),
    )
    parser.add_argument(
        "--nstlist",
        type=int,
        default=EXACT_RESPA_FACTOR,
        help=(
            "Verlet neighbor-list update interval in base steps for explicit performance probes. "
            "The default keeps the historical Gate I exact-r-RESPA shape."
        ),
    )
    parser.add_argument("--npme", type=int, default=None, help="Optional explicit -npme value; omitted by default.")
    parser.add_argument(
        "--ntomp-pme",
        "--ntomp_pme",
        dest="ntomp_pme",
        type=int,
        default=None,
        help="Optional explicit -ntomp_pme value; omitted by default.",
    )
    parser.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS, help="Replica count.")
    parser.add_argument(
        "--ld-seed-base",
        type=int,
        default=None,
        help=(
            "Optional explicit base stochastic seed for thermostat/barostat coupling. "
            "Replica i equilibration uses ld-seed = ld-seed-base + (i - 1), and production "
            "uses ld-seed = ld-seed-base + prod-ld-seed-offset + (i - 1). "
            "When omitted, GROMACS resolves the LD seed at grompp time and the script records the resolved value."
        ),
    )
    parser.add_argument(
        "--prod-ld-seed-offset",
        type=int,
        default=100000,
        help="Offset added to --ld-seed-base for production-phase stochastic seeds.",
    )
    parser.add_argument("--equil-ps", type=float, default=DEFAULT_EQ_PS, help="Equilibration duration in ps.")
    parser.add_argument("--prod-ps", type=float, default=DEFAULT_PROD_PS, help="Production duration in ps.")
    parser.add_argument(
        "--common-precondition-ps",
        type=float,
        default=0.0,
        help=(
            "Optional single common exact-r-RESPA NPT preconditioning duration in ps. "
            "When positive, the generated preconditioned GRO becomes the common start for all replicas."
        ),
    )
    parser.add_argument(
        "--common-precondition-velocity-seed",
        type=int,
        default=60999,
        help="Velocity generation seed for the optional common preconditioning stage.",
    )
    parser.add_argument(
        "--common-precondition-ld-seed",
        type=int,
        default=None,
        help=(
            "Optional LD seed for the common preconditioning stage. When omitted and --ld-seed-base "
            "is set, the script derives a non-overlapping deterministic seed from the Gate I seed contract."
        ),
    )
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
    if args.ntmpi < 1:
        raise ValueError("ntmpi must be positive.")
    if args.ntomp < 1:
        raise ValueError("ntomp must be positive.")
    if args.pin == "off" and (args.pinoffset is not None or args.pinstride is not None):
        raise ValueError("pinoffset/pinstride require --pin on or --pin auto.")
    if args.pinoffset is not None and args.pinoffset < 0:
        raise ValueError("pinoffset must be non-negative when provided.")
    if args.pinstride is not None and args.pinstride < 0:
        raise ValueError("pinstride must be non-negative when provided.")
    if args.npme is not None and (args.npme < 1 or args.npme >= args.ntmpi):
        raise ValueError("npme must be positive and smaller than ntmpi when explicit PME ranks are requested.")
    if args.ntomp_pme is not None and args.ntomp_pme < 1:
        raise ValueError("ntomp-pme must be positive when provided.")
    if args.ntomp_pme is not None and args.npme is None:
        raise ValueError("ntomp-pme requires explicit npme so the PME OpenMP split is unambiguous.")
    if args.replicas < 3:
        raise ValueError("Gate I requires at least 3 replicas; otherwise cross-replica conditioning is too weak.")
    if args.ld_seed_base is not None and args.ld_seed_base < 0:
        raise ValueError("ld-seed-base must be non-negative when provided.")
    if args.ld_seed_base is not None and args.prod_ld_seed_offset < args.replicas:
        raise ValueError("prod-ld-seed-offset must be at least the replica count to avoid equil/prod LD seed overlap.")
    if args.start_gro is not None and not rehome_repo_artifact_path(args.start_gro).is_file():
        raise ValueError(f"start-gro does not exist or is not a file: {args.start_gro}")
    if args.exact_respa_update_omp_threads is not None and args.exact_respa_update_omp_threads <= 0:
        raise ValueError("exact-respa-update-omp-threads must be positive when provided.")
    if args.sample_interval <= 0 or args.sample_interval % EXACT_RESPA_FACTOR != 0:
        raise ValueError("sample-interval must be a positive multiple of the exact-r-RESPA factor.")
    if args.nstlist <= 0:
        raise ValueError("nstlist must be positive.")
    if args.nstlist % EXACT_RESPA_FACTOR != 0:
        raise ValueError("Gate I nstlist probes must use a multiple of the outer exact-r-RESPA factor.")
    if args.common_precondition_ps < 0:
        raise ValueError("common-precondition-ps must be non-negative.")
    if args.common_precondition_velocity_seed < 0:
        raise ValueError("common-precondition-velocity-seed must be non-negative.")
    if args.common_precondition_ld_seed is not None and args.common_precondition_ld_seed < 0:
        raise ValueError("common-precondition-ld-seed must be non-negative when provided.")
    if (
        args.native_multi_owner_mode in ("full_owner_native", "split_owner_sidecar")
        and not args.allow_experimental_native_multi_probe
    ):
        raise ValueError(
            "full_owner_native and split_owner_sidecar are experimental performance probes, "
            "not Gate I exactness evidence. Re-run with --native-multi-owner-mode owner_fallback "
            "for claimable validation, or add --allow-experimental-native-multi-probe for an explicit probe."
        )
    if args.exact_respa_fused_initial_drift_mode == "on" and not args.allow_experimental_update_probe:
        raise ValueError(
            "exact-respa fused initial drift is a default-off update performance probe, not claimable "
            "Gate I evidence. Re-run with --exact-respa-fused-initial-drift-mode off/auto for claimable "
            "validation, or add --allow-experimental-update-probe for an explicit non-claimable probe."
        )
    for name, duration_ps in (("equil-ps", args.equil_ps), ("prod-ps", args.prod_ps)):
        steps = steps_from_ps(duration_ps)
        if steps <= 0 or not math.isclose(steps * DT_PS, duration_ps, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{name} must be representable as a positive integer number of base steps.")
        if steps % EXACT_RESPA_FACTOR != 0:
            raise ValueError(f"{name} must be a multiple of the exact-r-RESPA factor.")
    if args.common_precondition_ps > 0:
        steps = steps_from_ps(args.common_precondition_ps)
        if steps <= 0 or not math.isclose(steps * DT_PS, args.common_precondition_ps, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("common-precondition-ps must be representable as a positive integer number of base steps.")
        if steps % EXACT_RESPA_FACTOR != 0:
            raise ValueError("common-precondition-ps must be a multiple of the exact-r-RESPA factor.")


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


def make_gate_i_npt_mdp(
    *,
    duration_ps: float,
    sample_interval: int,
    phase: str,
    seed: int,
    args: argparse.Namespace,
    ld_seed: int | None = None,
) -> str:
    nsteps = steps_from_ps(duration_ps)
    starts_from_coordinates = phase in ("equil", "precondition")
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
        if starts_from_coordinates
        else "gen-vel                 = no\n"
    )
    continuation = "continuation             = no\n" if starts_from_coordinates else "continuation             = yes\n"
    stochastic_seed = f"ld-seed                 = {ld_seed}\n" if ld_seed is not None else ""
    return (
        f"title                   = gate i charged long npt {phase} exact respa\n"
        + exact_respa_common_mdp(nsteps, sample_interval, nstlist=getattr(args, "nstlist", EXACT_RESPA_FACTOR))
        + thermostat
        + barostat
        + velocity
        + continuation
        + stochastic_seed
    )


def file_exists_and_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def final_box_from_gro(path: Path) -> list[float] | None:
    if not file_exists_and_nonempty(path):
        return None
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return None
    fields = lines[-1].split()
    if len(fields) < 3:
        return None
    try:
        return [float(fields[0]), float(fields[1]), float(fields[2])]
    except ValueError:
        return None


def initial_box_from_observables(observables: dict[str, dict[str, object]]) -> list[float] | None:
    try:
        return [
            float(observables["Box-X"]["values"][0]),
            float(observables["Box-Y"]["values"][0]),
            float(observables["Box-Z"]["values"][0]),
        ]
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def grompp_complete(tpr_path: Path, mdout_path: Path) -> bool:
    return file_exists_and_nonempty(tpr_path) and file_exists_and_nonempty(mdout_path)


def mdrun_complete(gmx: Path, deffnm: Path, expected_steps: int | None = None) -> bool:
    required = (
        deffnm.with_suffix(".edr"),
        deffnm.with_suffix(".gro"),
        deffnm.with_suffix(".cpt"),
        deffnm.with_suffix(".log"),
    )
    if not all(file_exists_and_nonempty(path) for path in required):
        return False
    if expected_steps is None:
        return True
    return checkpoint_step(gmx, deffnm.with_suffix(".cpt")) >= expected_steps


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


def resolved_ld_seed_from_grompp_stdout(stdout_path: Path) -> int | None:
    if not file_exists_and_nonempty(stdout_path):
        return None
    match = re.search(
        r"Setting the LD random seed to\s+(-?\d+)",
        stdout_path.read_text(encoding="utf-8", errors="ignore"),
    )
    return int(match.group(1)) if match is not None else None


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
    expected_steps: int | None = None,
) -> dict[str, object]:
    stdout_path = logs_dir / f"{label}.stdout"
    stderr_path = logs_dir / f"{label}.stderr"
    effective_argv = list(argv)
    resume_checkpoint = best_resume_checkpoint(gmx, deffnm) if args.resume else None
    preserve_partial_outputs = resume_checkpoint == deffnm.with_suffix(".cpt")
    if not mdrun_complete(gmx, deffnm, expected_steps):
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
    if mdrun_complete(gmx, deffnm, expected_steps):
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


def native_multi_owner_mode_env(mode: str) -> dict[str, str]:
    if mode == "default":
        return {}
    if mode == "owner_fallback":
        return {
            NATIVE_MULTI_ENV: "1",
            NATIVE_MULTI_OWNER_FALLBACK_ENV: "1",
            NATIVE_MULTI_SPLIT_OWNER_ENV: "0",
        }
    if mode == "full_owner_native":
        return {
            NATIVE_MULTI_ENV: "1",
            NATIVE_MULTI_OWNER_FALLBACK_ENV: "0",
            NATIVE_MULTI_SPLIT_OWNER_ENV: "0",
        }
    if mode == "split_owner_sidecar":
        return {
            NATIVE_MULTI_ENV: "1",
            NATIVE_MULTI_OWNER_FALLBACK_ENV: "0",
            NATIVE_MULTI_SPLIT_OWNER_ENV: "1",
        }
    raise ValueError(f"Unsupported native-multi owner mode: {mode}")


def exact_respa_update_omp_env(mode: str) -> dict[str, str]:
    if mode == "auto":
        return {}
    if mode == "off":
        return {EXACT_RESPA_UPDATE_OMP_ENV: "0"}
    if mode == "on":
        return {EXACT_RESPA_UPDATE_OMP_ENV: "1"}
    raise ValueError(f"Unsupported exact-r-RESPA update OMP mode: {mode}")


def exact_respa_update_omp_threads_env(num_threads: int | None) -> dict[str, str]:
    if num_threads is None:
        return {}
    return {EXACT_RESPA_UPDATE_OMP_THREADS_ENV: str(num_threads)}


def exact_respa_update_direct_fastpath_env(mode: str) -> dict[str, str]:
    if mode == "auto":
        return {}
    if mode == "off":
        return {EXACT_RESPA_UPDATE_DIRECT_FASTPATH_ENV: "0"}
    if mode == "on":
        return {EXACT_RESPA_UPDATE_DIRECT_FASTPATH_ENV: "1"}
    raise ValueError(f"Unsupported exact-r-RESPA update direct fastpath mode: {mode}")


def exact_respa_fused_initial_drift_env(mode: str) -> dict[str, str]:
    if mode == "auto":
        return {}
    if mode == "off":
        return {EXACT_RESPA_FUSED_INITIAL_DRIFT_ENV: "0"}
    if mode == "on":
        return {EXACT_RESPA_FUSED_INITIAL_DRIFT_ENV: "1"}
    raise ValueError(f"Unsupported exact-r-RESPA fused initial drift mode: {mode}")


def common_precondition_ld_seed(args: argparse.Namespace) -> int | None:
    if args.common_precondition_ld_seed is not None:
        return args.common_precondition_ld_seed
    if args.ld_seed_base is None:
        return None
    return args.ld_seed_base + (2 * args.prod_ld_seed_offset) + args.replicas


def build_contract(
    *,
    args: argparse.Namespace,
    out_root: Path,
    scaffold_manifest: dict[str, object],
    prerequisites: dict[str, object],
    perf_ref: dict[str, object],
) -> dict[str, object]:
    scaffold_gro = rehome_repo_artifact_path(scaffold_manifest["artifacts"]["gro"])
    start_gro = rehome_repo_artifact_path(args.start_gro) if args.start_gro is not None else scaffold_gro
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "gate_id": GATE_ID,
        "status": "DECLARED",
        "execution_policy": {
            "single_rank": args.ntmpi == 1 and args.npme is None,
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
            "ld_seed_base": args.ld_seed_base,
            "prod_ld_seed_offset": args.prod_ld_seed_offset,
            "common_precondition_ps": args.common_precondition_ps,
            "common_precondition_velocity_seed": args.common_precondition_velocity_seed,
            "common_precondition_ld_seed": common_precondition_ld_seed(args),
            "equil_ps": args.equil_ps,
            "prod_ps": args.prod_ps,
            "sample_interval_base_steps": args.sample_interval,
            "nstlist_base_steps": args.nstlist,
            "temperature_k": args.temperature_k,
            "pressure_bar": args.pressure_bar,
            "tau_t_ps": args.tau_t_ps,
            "tau_p_ps": args.tau_p_ps,
            "compressibility_bar_inv": args.compressibility_bar_inv,
            "ntmpi": args.ntmpi,
            "ntomp": args.ntomp,
            "pin": args.pin,
            "pinoffset": args.pinoffset,
            "pinstride": args.pinstride,
            "npme": args.npme,
            "ntomp_pme": args.ntomp_pme,
            "native_multi_owner_mode": args.native_multi_owner_mode,
            "native_multi_owner_mode_env": native_multi_owner_mode_env(args.native_multi_owner_mode),
            "allow_experimental_native_multi_probe": args.allow_experimental_native_multi_probe,
            "allow_experimental_update_probe": args.allow_experimental_update_probe,
            "exact_respa_update_omp_mode": args.exact_respa_update_omp_mode,
            "exact_respa_update_omp_env": exact_respa_update_omp_env(args.exact_respa_update_omp_mode),
            "exact_respa_update_omp_threads": args.exact_respa_update_omp_threads,
            "exact_respa_update_omp_threads_env": exact_respa_update_omp_threads_env(
                args.exact_respa_update_omp_threads
            ),
            "exact_respa_update_direct_fastpath_mode": args.exact_respa_update_direct_fastpath_mode,
            "exact_respa_update_direct_fastpath_env": exact_respa_update_direct_fastpath_env(
                args.exact_respa_update_direct_fastpath_mode
            ),
            "exact_respa_fused_initial_drift_mode": args.exact_respa_fused_initial_drift_mode,
            "exact_respa_fused_initial_drift_env": exact_respa_fused_initial_drift_env(
                args.exact_respa_fused_initial_drift_mode
            ),
            "mdrun_shape": (
                f"ntmpi {args.ntmpi} / ntomp {args.ntomp} / npme {args.npme} / "
                f"ntomp_pme {args.ntomp_pme} / pin {args.pin} / pinoffset {args.pinoffset} / "
                f"pinstride {args.pinstride} / nb cpu / bonded cpu / pme cpu / update cpu / reprod"
            ),
            "stochastic_seed_contract": (
                "explicit_phase_separated_ld_seed_per_replica"
                if args.ld_seed_base is not None
                else "grompp_resolved_phase_ld_seed_recorded_per_replica"
            ),
            "phase_continuation_policy": {
                "equil_gen_vel": True,
                "prod_gen_vel": False,
                "prod_grompp_uses_equil_checkpoint": True,
                "prod_mdp_continuation": True,
                "prod_ld_seed_restarts_from_distinct_seed": args.ld_seed_base is not None,
            },
            "initial_configuration": {
                "source": (
                    "generated_common_preconditioned_start_gro"
                    if args.common_precondition_ps > 0
                    else ("common_preconditioned_start_gro" if args.start_gro is not None else "scaffold_manifest_gro")
                ),
                "input_path": str(start_gro.resolve()),
                "path": (
                    str((out_root / "precondition" / "common" / "precondition.gro").resolve())
                    if args.common_precondition_ps > 0
                    else str(start_gro.resolve())
                ),
                "replica_policy": "all replicas start equilibration from this same coordinate set; replica independence comes from gen-seed and ld-seed",
            },
            "common_preconditioning_policy": {
                "enabled": args.common_precondition_ps > 0,
                "duration_ps": args.common_precondition_ps,
                "velocity_seed": args.common_precondition_velocity_seed,
                "ld_seed": common_precondition_ld_seed(args),
                "output_gro": str((out_root / "precondition" / "common" / "precondition.gro").resolve())
                if args.common_precondition_ps > 0
                else None,
                "claim_note": (
                    "The common preconditioning stage is input preparation only; Gate I acceptance still "
                    "depends on the independent replica production observables."
                ),
            },
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
            "A common preconditioned starting structure is a conditioning input only; it is not production handoff approval.",
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
    contract = build_contract(
        args=args,
        out_root=out_root,
        scaffold_manifest=scaffold_manifest,
        prerequisites=prereq_payload,
        perf_ref=perf_ref,
    )
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
    top_path = rehome_repo_artifact_path(scaffold_manifest["artifacts"]["topology"])
    gro_path = rehome_repo_artifact_path(scaffold_manifest["artifacts"]["gro"])
    initial_gro_path = rehome_repo_artifact_path(args.start_gro) if args.start_gro is not None else gro_path
    if args.start_gro is not None:
        copied_start_gro = inputs_dir / (
            "precondition_input_start.gro" if args.common_precondition_ps > 0 else "common_preconditioned_start.gro"
        )
        shutil.copy2(initial_gro_path, copied_start_gro)
        initial_gro_path = copied_start_gro
    env = base_env(args)
    env.update(native_multi_owner_mode_env(args.native_multi_owner_mode))
    env.update(exact_respa_update_omp_env(args.exact_respa_update_omp_mode))
    env.update(exact_respa_update_omp_threads_env(args.exact_respa_update_omp_threads))
    env.update(exact_respa_update_direct_fastpath_env(args.exact_respa_update_direct_fastpath_mode))
    env.update(exact_respa_fused_initial_drift_env(args.exact_respa_fused_initial_drift_mode))
    commands: list[dict[str, object]] = []
    replica_runs: list[dict[str, object]] = []
    precondition_summary: dict[str, object] = {
        "enabled": False,
        "observables": {},
        "phase_boundary": {},
    }

    if args.common_precondition_ps > 0:
        precondition_input_root = inputs_dir / "precondition"
        precondition_input_root.mkdir(parents=True, exist_ok=True)
        precondition_mdp = precondition_input_root / "precondition.mdp"
        precondition_ld_seed = common_precondition_ld_seed(args)
        write_text(
            precondition_mdp,
            make_gate_i_npt_mdp(
                duration_ps=args.common_precondition_ps,
                sample_interval=args.sample_interval,
                phase="precondition",
                seed=args.common_precondition_velocity_seed,
                args=args,
                ld_seed=precondition_ld_seed,
            ),
        )

        precondition_run_root = out_root / "precondition" / "common"
        precondition_run_root.mkdir(parents=True, exist_ok=True)
        precondition_deffnm = precondition_run_root / "precondition"
        run_or_resume_grompp(
            gmx=gmx,
            mdp_path=precondition_mdp,
            conf_path=initial_gro_path,
            top_path=top_path,
            tpr_path=precondition_deffnm.with_suffix(".tpr"),
            mdout_path=precondition_run_root / "precondition.mdout.mdp",
            logs_dir=logs_dir,
            commands=commands,
            label="common_precondition_grompp",
            env=env,
        )
        resolved_precondition_ld_seed = resolved_ld_seed_from_grompp_stdout(logs_dir / "common_precondition_grompp.stdout")
        precondition_md_payload = run_or_resume_md(
            gmx=gmx,
            argv=[str(gmx), "mdrun", *mdrun_args_cpu(args, precondition_deffnm)],
            deffnm=precondition_deffnm,
            env=env,
            logs_dir=logs_dir,
            commands=commands,
            label="common_precondition_mdrun",
            args=args,
            expected_steps=steps_from_ps(args.common_precondition_ps),
        )
        precondition_observables = collect_run_observables(
            gmx,
            precondition_deffnm,
            precondition_run_root / "precondition_observables.xvg",
        )
        precondition_summary = {
            "enabled": True,
            "duration_ps": args.common_precondition_ps,
            "velocity_seed": args.common_precondition_velocity_seed,
            "ld_seed_requested": precondition_ld_seed,
            "ld_seed_resolved": resolved_precondition_ld_seed,
            "deffnm": str(precondition_deffnm),
            "layout_report": precondition_md_payload["layout_report"],
            "phase_boundary": {
                "input_box_nm": final_box_from_gro(initial_gro_path),
                "final_box_nm": final_box_from_gro(precondition_deffnm.with_suffix(".gro")),
            },
            "observables": precondition_observables,
        }
        write_json(precondition_run_root / "precondition_summary.json", precondition_summary)
        initial_gro_path = precondition_deffnm.with_suffix(".gro")

    for replica_index in range(1, args.replicas + 1):
        seed = 61001 + replica_index - 1
        equil_ld_seed = args.ld_seed_base + replica_index - 1 if args.ld_seed_base is not None else None
        prod_ld_seed = (
            args.ld_seed_base + args.prod_ld_seed_offset + replica_index - 1
            if args.ld_seed_base is not None
            else None
        )
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
                ld_seed=equil_ld_seed,
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
                ld_seed=prod_ld_seed,
            ),
        )

        run_root = out_root / "cpu" / f"replica_{replica_index:02d}"
        run_root.mkdir(parents=True, exist_ok=True)
        equil_deffnm = run_root / "equil"
        prod_deffnm = run_root / "prod"

        run_or_resume_grompp(
            gmx=gmx,
            mdp_path=equil_mdp,
            conf_path=initial_gro_path,
            top_path=top_path,
            tpr_path=equil_deffnm.with_suffix(".tpr"),
            mdout_path=run_root / "equil.mdout.mdp",
            logs_dir=logs_dir,
            commands=commands,
            label=f"cpu_replica_{replica_index:02d}_grompp_equil",
            env=env,
        )
        resolved_equil_ld_seed = resolved_ld_seed_from_grompp_stdout(logs_dir / f"cpu_replica_{replica_index:02d}_grompp_equil.stdout")
        run_or_resume_md(
            gmx=gmx,
            argv=[str(gmx), "mdrun", *mdrun_args_cpu(args, equil_deffnm)],
            deffnm=equil_deffnm,
            env=env,
            logs_dir=logs_dir,
            commands=commands,
            label=f"cpu_replica_{replica_index:02d}_mdrun_equil",
            args=args,
            expected_steps=steps_from_ps(args.equil_ps),
        )
        equil_observables = collect_run_observables(gmx, equil_deffnm, run_root / "equilibration_observables.xvg")

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
        resolved_prod_ld_seed = resolved_ld_seed_from_grompp_stdout(logs_dir / f"cpu_replica_{replica_index:02d}_grompp_prod.stdout")
        md_payload = run_or_resume_md(
            gmx=gmx,
            argv=[str(gmx), "mdrun", *mdrun_args_cpu(args, prod_deffnm)],
            deffnm=prod_deffnm,
            env=env,
            logs_dir=logs_dir,
            commands=commands,
            label=f"cpu_replica_{replica_index:02d}_mdrun_prod",
            args=args,
            expected_steps=steps_from_ps(args.prod_ps),
        )

        observables = collect_run_observables(gmx, prod_deffnm, run_root / "production_observables.xvg")
        phase_boundary = {
            "equil_final_box_nm": final_box_from_gro(equil_deffnm.with_suffix(".gro")),
            "prod_initial_box_nm": initial_box_from_observables(observables),
            "prod_final_box_nm": final_box_from_gro(prod_deffnm.with_suffix(".gro")),
            "prod_grompp_input_checkpoint": str(equil_deffnm.with_suffix(".cpt")),
            "prod_mdp_continuation": True,
        }
        replica_payload = {
            "replica_index": replica_index,
            "seed": seed,
            "velocity_seed": seed,
            "ld_seed_requested": equil_ld_seed,
            "ld_seed_resolved": resolved_equil_ld_seed,
            "equil_ld_seed_requested": equil_ld_seed,
            "equil_ld_seed_resolved": resolved_equil_ld_seed,
            "prod_ld_seed_requested": prod_ld_seed,
            "prod_ld_seed_resolved": resolved_prod_ld_seed,
            "prod_deffnm": str(prod_deffnm),
            "layout_report": md_payload["layout_report"],
            "phase_boundary": phase_boundary,
            "equil_observables": equil_observables,
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
    equil_aggregates = {
        term: aggregate_replicates(
            [replica["equil_observables"][term] for replica in replica_runs if replica["equil_observables"][term].get("available")]
        )
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
        "equilibration_diagnostics": {
            term: aggregate_metric(aggregate) for term, aggregate in equil_aggregates.items() if term in PRIMARY_GATE_OBSERVABLES
        },
        "common_preconditioning": precondition_summary,
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
