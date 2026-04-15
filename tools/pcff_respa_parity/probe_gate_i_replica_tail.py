from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from validate_gate_i_charged_long_npt_conditioning import (
    DEFAULT_COMPRESSIBILITY_BAR_INV,
    DEFAULT_GATE_A_MANIFEST,
    DEFAULT_GATE_E_MANIFEST,
    DEFAULT_GATE_F_MANIFEST,
    DEFAULT_GATE_G_MANIFEST,
    DEFAULT_GMX,
    DEFAULT_PERF_REFERENCE_LOG,
    DEFAULT_PRESSURE_BAR,
    DEFAULT_SCAFFOLD_MANIFEST,
    DEFAULT_SAMPLE_INTERVAL,
    DEFAULT_TAU_P_PS,
    DEFAULT_TAU_T_PS,
    DEFAULT_TEMP_K,
    DT_PS,
    EXACT_RESPA_FACTOR,
    REQUESTED_OBSERVABLES,
    SUPPORT_OBSERVABLES,
    base_env,
    collect_run_observables,
    load_json,
    make_gate_i_npt_mdp,
    mdrun_args_cpu,
    performance_reference,
    run_or_resume_grompp,
    run_or_resume_md,
    steps_from_ps,
    validate_prerequisites,
    write_commands_script,
    write_json,
    write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAILED_GATE_I_ROOT = REPO_ROOT / "tests" / "reference_results" / "gate_i_charged_long_npt_conditioning"
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_i_replica_tail_probe_replica_02_eq750_prod250"

SCHEMA_NAME = "gate_i_replica_tail_probe"
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a focused exact-r-RESPA NPT probe for one Gate I replica seed to test whether "
            "an extended equilibration horizon removes the density/volume conditioning tail."
        )
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument("--gate-a-manifest", default=str(DEFAULT_GATE_A_MANIFEST))
    parser.add_argument("--gate-e-manifest", default=str(DEFAULT_GATE_E_MANIFEST))
    parser.add_argument("--gate-f-manifest", default=str(DEFAULT_GATE_F_MANIFEST))
    parser.add_argument("--gate-g-manifest", default=str(DEFAULT_GATE_G_MANIFEST))
    parser.add_argument("--scaffold-manifest", default=str(DEFAULT_SCAFFOLD_MANIFEST))
    parser.add_argument("--failed-gate-i-root", default=str(DEFAULT_FAILED_GATE_I_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--replica-index", type=int, default=2, help="Original Gate I replica index to replay.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional explicit velocity seed. Defaults to the original Gate I seed mapping.",
    )
    parser.add_argument("--ntmpi", type=int, default=1)
    parser.add_argument("--ntomp", type=int, default=1)
    parser.add_argument("--npme", type=int, default=None)
    parser.add_argument(
        "--equil-ps",
        type=float,
        default=750.0,
        help="Probe equilibration duration in ps. Default extends the original 250 ps horizon materially.",
    )
    parser.add_argument(
        "--prod-ps",
        type=float,
        default=250.0,
        help="Probe production duration in ps. Short scout horizon used only to test whether the early tail persists.",
    )
    parser.add_argument("--sample-interval", type=int, default=DEFAULT_SAMPLE_INTERVAL)
    parser.add_argument("--temperature-k", type=float, default=DEFAULT_TEMP_K)
    parser.add_argument("--pressure-bar", type=float, default=DEFAULT_PRESSURE_BAR)
    parser.add_argument("--tau-t-ps", type=float, default=DEFAULT_TAU_T_PS)
    parser.add_argument("--tau-p-ps", type=float, default=DEFAULT_TAU_P_PS)
    parser.add_argument("--compressibility-bar-inv", type=float, default=DEFAULT_COMPRESSIBILITY_BAR_INV)
    parser.add_argument("--performance-reference-log", default=str(DEFAULT_PERF_REFERENCE_LOG))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.ntmpi != 1:
        raise ValueError("The focused tail probe is restricted to single-rank runs (ntmpi=1).")
    if args.replica_index <= 0:
        raise ValueError("replica-index must be positive.")
    if args.sample_interval <= 0 or args.sample_interval % EXACT_RESPA_FACTOR != 0:
        raise ValueError("sample-interval must be a positive multiple of the exact-r-RESPA factor.")
    for name, duration_ps in (("equil-ps", args.equil_ps), ("prod-ps", args.prod_ps)):
        steps = steps_from_ps(duration_ps)
        if steps <= 0 or not math.isclose(steps * DT_PS, duration_ps, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{name} must be representable as a positive integer number of base steps.")
        if steps % EXACT_RESPA_FACTOR != 0:
            raise ValueError(f"{name} must be a multiple of the exact-r-RESPA factor.")


def original_seed(replica_index: int) -> int:
    return 61001 + replica_index - 1


def sanitize_observable(observable: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in observable.items() if key != "values"}


def relative_value(value: float, reference: float) -> float:
    if math.isclose(reference, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        return 0.0 if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-12) else math.inf
    return abs(value) / abs(reference)


def volume_from_gro(path: Path) -> dict[str, object]:
    fields = path.read_text(encoding="utf-8").splitlines()[-1].split()
    box = [float(value) for value in fields[:3]]
    return {"box_nm": box, "volume_nm3": box[0] * box[1] * box[2]}


def block_span(block_means: list[float]) -> float:
    return max(block_means) - min(block_means) if block_means else 0.0


def monotonic_direction(block_means: list[float], *, increasing: bool) -> bool:
    if len(block_means) < 2:
        return True
    comparator = (lambda a, b: a <= b) if increasing else (lambda a, b: a >= b)
    return all(comparator(left, right) for left, right in zip(block_means, block_means[1:]))


def comparative_metric(original: dict[str, object], probe: dict[str, object]) -> dict[str, object]:
    original_mean = float(original["mean"])
    probe_mean = float(probe["mean"])
    original_abs_drift = float(original["abs_block_drift"])
    probe_abs_drift = float(probe["abs_block_drift"])
    original_block_means = [float(value) for value in original["block_means"]]
    probe_block_means = [float(value) for value in probe["block_means"]]
    return {
        "original_mean": original_mean,
        "probe_mean": probe_mean,
        "mean_delta": probe_mean - original_mean,
        "mean_delta_rel": relative_value(probe_mean - original_mean, original_mean),
        "original_abs_block_drift": original_abs_drift,
        "probe_abs_block_drift": probe_abs_drift,
        "original_abs_block_drift_rel": relative_value(original_abs_drift, original_mean),
        "probe_abs_block_drift_rel": relative_value(probe_abs_drift, probe_mean),
        "improvement_factor": (probe_abs_drift / original_abs_drift) if not math.isclose(original_abs_drift, 0.0) else math.inf,
        "original_block_means": original_block_means,
        "probe_block_means": probe_block_means,
        "original_block_span": block_span(original_block_means),
        "probe_block_span": block_span(probe_block_means),
        "original_first_to_last": original_block_means[-1] - original_block_means[0] if original_block_means else 0.0,
        "probe_first_to_last": probe_block_means[-1] - probe_block_means[0] if probe_block_means else 0.0,
    }


def build_report(
    *,
    args: argparse.Namespace,
    seed: int,
    perf_ref: dict[str, object],
    original_summary: dict[str, object],
    probe_observables: dict[str, dict[str, object]],
    equil_deffnm: Path,
    prod_deffnm: Path,
    md_payloads: dict[str, dict[str, object]],
) -> dict[str, object]:
    original_observables = original_summary["observables"]
    comparisons = {
        term: comparative_metric(original_observables[term], probe_observables[term])
        for term in ("Density", "Volume", "Temperature", "Pressure", "Potential", "Box-X", "Box-Y", "Box-Z")
    }
    density_probe_blocks = comparisons["Density"]["probe_block_means"]
    volume_probe_blocks = comparisons["Volume"]["probe_block_means"]
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "purpose": (
            "Focused replay of one failed Gate I replica seed with extended equilibration and a short scout production "
            "horizon, used only to test whether the early density/volume conditioning tail persists."
        ),
        "non_claims": [
            "This probe is a single-seed diagnostic, not a Gate I rerun.",
            "This probe does not close the charged medium-scale long-NPT blocker.",
            "A favorable probe would only justify a revised follow-up protocol, not a broader CPU completion claim.",
        ],
        "source_gate_i": {
            "failed_root": str(Path(args.failed_gate_i_root).resolve()),
            "replica_index": args.replica_index,
            "seed": seed,
            "original_replica_summary": str(
                (Path(args.failed_gate_i_root) / "cpu" / f"replica_{args.replica_index:02d}" / "replica_summary.json").resolve()
            ),
        },
        "run_settings": {
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
            "npme": args.npme,
            "mdrun_shape": "nb cpu / bonded cpu / pme cpu / update cpu / reprod / single rank",
        },
        "performance_reference": perf_ref,
        "equil_final_box": volume_from_gro(equil_deffnm.with_suffix(".gro")),
        "prod_final_box": volume_from_gro(prod_deffnm.with_suffix(".gro")),
        "layout_reports": {
            "equil": md_payloads["equil"]["layout_report"],
            "prod": md_payloads["prod"]["layout_report"],
        },
        "original_observables": {
            term: sanitize_observable(original_observables[term]) for term in ("Density", "Volume", "Temperature")
        },
        "probe_observables": {
            term: sanitize_observable(probe_observables[term]) for term in ("Density", "Volume", "Temperature")
        },
        "comparisons": comparisons,
        "early_signal_summary": {
            "density_probe_monotonic_increase": monotonic_direction(density_probe_blocks, increasing=True),
            "volume_probe_monotonic_decrease": monotonic_direction(volume_probe_blocks, increasing=False),
            "density_probe_abs_block_drift_rel": comparisons["Density"]["probe_abs_block_drift_rel"],
            "volume_probe_abs_block_drift_rel": comparisons["Volume"]["probe_abs_block_drift_rel"],
            "temperature_probe_abs_block_drift_rel": comparisons["Temperature"]["probe_abs_block_drift_rel"],
        },
        "artifacts": {
            "equil_gro": str(equil_deffnm.with_suffix(".gro")),
            "equil_log": str(equil_deffnm.with_suffix(".log")),
            "prod_gro": str(prod_deffnm.with_suffix(".gro")),
            "prod_log": str(prod_deffnm.with_suffix(".log")),
            "prod_edr": str(prod_deffnm.with_suffix(".edr")),
            "production_observables_xvg": str(prod_deffnm.parent / "production_observables.xvg"),
        },
    }


def report_markdown(report: dict[str, object]) -> str:
    density = report["comparisons"]["Density"]
    volume = report["comparisons"]["Volume"]
    lines = [
        "# Gate I Replica Tail Probe",
        "",
        f"- Replica index: `{report['source_gate_i']['replica_index']}`",
        f"- Seed: `{report['source_gate_i']['seed']}`",
        f"- Probe horizon: `{report['run_settings']['equil_ps']} ps + {report['run_settings']['prod_ps']} ps`",
        f"- Equil final volume (nm^3): `{report['equil_final_box']['volume_nm3']:.6f}`",
        f"- Probe density drift rel: `{density['probe_abs_block_drift_rel']:.6f}`",
        f"- Probe volume drift rel: `{volume['probe_abs_block_drift_rel']:.6f}`",
        "",
        "## Interpretation",
        f"- Density first-to-last block delta: `{density['probe_first_to_last']:.6f}`",
        f"- Volume first-to-last block delta: `{volume['probe_first_to_last']:.6f}`",
        f"- Density monotonic increase: `{report['early_signal_summary']['density_probe_monotonic_increase']}`",
        f"- Volume monotonic decrease: `{report['early_signal_summary']['volume_probe_monotonic_decrease']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    validate_args(args)

    gate_a_manifest = load_json(Path(args.gate_a_manifest))
    gate_e_manifest = load_json(Path(args.gate_e_manifest))
    gate_f_manifest = load_json(Path(args.gate_f_manifest))
    gate_g_manifest = load_json(Path(args.gate_g_manifest))
    scaffold_manifest = load_json(Path(args.scaffold_manifest))
    validate_prerequisites(gate_a_manifest, gate_e_manifest, gate_f_manifest, gate_g_manifest, scaffold_manifest)

    failed_root = Path(args.failed_gate_i_root).resolve()
    original_summary_path = failed_root / "cpu" / f"replica_{args.replica_index:02d}" / "replica_summary.json"
    if not original_summary_path.exists():
        raise FileNotFoundError(f"Missing original replica summary: {original_summary_path}")
    original_summary = load_json(original_summary_path)

    seed = args.seed if args.seed is not None else original_seed(args.replica_index)
    args.replicas = 1
    out_root = Path(args.out).resolve()
    if out_root.exists() and not args.resume:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    perf_ref = performance_reference(Path(args.performance_reference_log), args)
    config = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "DECLARED",
        "purpose": "Replica-specific tail probe for failed Gate I seed with extended equilibration.",
        "source_gate_i_root": str(failed_root),
        "replica_index": args.replica_index,
        "seed": seed,
        "run_settings": {
            "equil_ps": args.equil_ps,
            "prod_ps": args.prod_ps,
            "sample_interval_base_steps": args.sample_interval,
            "ntmpi": args.ntmpi,
            "ntomp": args.ntomp,
            "npme": args.npme,
        },
        "performance_reference": perf_ref,
    }
    write_json(out_root / "probe_config.json", config)

    inputs_dir = out_root / "inputs"
    logs_dir = out_root / "logs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    gmx = Path(args.gmx)
    top_path = Path(str(scaffold_manifest["artifacts"]["topology"]))
    gro_path = Path(str(scaffold_manifest["artifacts"]["gro"]))
    env = base_env(args)
    commands: list[dict[str, object]] = []

    equil_mdp = inputs_dir / "equil.mdp"
    prod_mdp = inputs_dir / "prod.mdp"
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

    run_root = out_root / "cpu" / f"replica_{args.replica_index:02d}"
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
        label=f"replica_{args.replica_index:02d}_probe_grompp_equil",
        env=env,
    )
    equil_payload = run_or_resume_md(
        gmx=gmx,
        argv=[str(gmx), "mdrun", *mdrun_args_cpu(args, equil_deffnm)],
        deffnm=equil_deffnm,
        env=env,
        logs_dir=logs_dir,
        commands=commands,
        label=f"replica_{args.replica_index:02d}_probe_mdrun_equil",
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
        label=f"replica_{args.replica_index:02d}_probe_grompp_prod",
        env=env,
        checkpoint_path=equil_deffnm.with_suffix(".cpt"),
    )
    prod_payload = run_or_resume_md(
        gmx=gmx,
        argv=[str(gmx), "mdrun", *mdrun_args_cpu(args, prod_deffnm)],
        deffnm=prod_deffnm,
        env=env,
        logs_dir=logs_dir,
        commands=commands,
        label=f"replica_{args.replica_index:02d}_probe_mdrun_prod",
        args=args,
    )

    probe_observables = collect_run_observables(gmx, prod_deffnm, run_root / "production_observables.xvg")
    report = build_report(
        args=args,
        seed=seed,
        perf_ref=perf_ref,
        original_summary=original_summary,
        probe_observables=probe_observables,
        equil_deffnm=equil_deffnm,
        prod_deffnm=prod_deffnm,
        md_payloads={"equil": equil_payload, "prod": prod_payload},
    )
    write_json(run_root / "probe_report.json", report)
    write_text(run_root / "probe_report.md", report_markdown(report))
    write_commands_script(out_root / "run_commands.sh", commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
