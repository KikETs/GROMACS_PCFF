from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

from freeze_gate_a_oracle import DEFAULT_GMX, base_env, command_record, env_delta, write_commands_script, write_text
from validate_gate_b_nb_gpu import load_json, parse_gpu_support, parse_precision_mode
from validate_gate_e_update_gpu import parse_layout_report
from validate_gate_g_long_ensemble import (
    aggregate_replicates,
    collect_run_observables,
    compare_layout_observable,
    exact_respa_common_mdp,
    mdrun_args_cpu,
    mdrun_args_gpu,
    run_grompp,
    run_md,
    save_observable_table,
    sanitize_layout_aggregates,
    steps_from_ps,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCAFFOLD_MANIFEST = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_h_fixture_scaffold"
    / "gate_h_dense_oligomer_2x2x2"
    / "fixture_manifest.json"
)
DEFAULT_BRINGUP_RESULT = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_h_neutral_scaffold_bringup"
    / "summaries"
    / "bringup_result.json"
)
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_h_neutral_entry_validation"
DT_PS = 0.0005
EXACT_RESPA_FACTOR = 4
DEFAULT_EQ_PS = 2.0
DEFAULT_PROD_PS = 8.0
DEFAULT_SAMPLE_INTERVAL = EXACT_RESPA_FACTOR * 50
DEFAULT_REPLICAS = 2
DEFAULT_TEMP_K = 300.0
DEFAULT_TAU_T_PS = 0.5
REQUESTED_OBSERVABLES = ("Temperature", "Pressure", "Potential")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-transport large-neutral entry gate for an exact-r-RESPA Gate H scaffold."
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument("--scaffold-manifest", default=str(DEFAULT_SCAFFOLD_MANIFEST), help="Scaffold manifest path.")
    parser.add_argument("--bringup-result", default=str(DEFAULT_BRINGUP_RESULT), help="Bring-up result JSON path.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks.")
    parser.add_argument("--ntomp", type=int, default=1, help="OpenMP threads.")
    parser.add_argument("--npme", type=int, default=None, help="Optional explicit -npme value; omitted by default.")
    parser.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS, help="Replica count per layout.")
    parser.add_argument("--equil-ps", type=float, default=DEFAULT_EQ_PS, help="Equilibration duration in ps.")
    parser.add_argument("--prod-ps", type=float, default=DEFAULT_PROD_PS, help="Production duration in ps.")
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=DEFAULT_SAMPLE_INTERVAL,
        help="Energy/log sampling interval in base steps. Must be a positive multiple of the exact-r-RESPA factor.",
    )
    parser.add_argument("--temperature-k", type=float, default=DEFAULT_TEMP_K, help="Target temperature.")
    parser.add_argument("--tau-t-ps", type=float, default=DEFAULT_TAU_T_PS, help="Thermostat coupling time.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed stage artifacts under --out instead of deleting the output root.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.ntmpi != 1:
        raise ValueError("Neutral entry validation is restricted to single-rank runs.")
    if args.replicas < 2:
        raise ValueError("Neutral entry validation requires replicated runs; use at least 2 replicas.")
    if args.sample_interval <= 0 or args.sample_interval % EXACT_RESPA_FACTOR != 0:
        raise ValueError("sample-interval must be a positive multiple of the exact-r-RESPA factor.")
    for name, duration_ps in (("equil-ps", args.equil_ps), ("prod-ps", args.prod_ps)):
        steps = steps_from_ps(duration_ps)
        if steps <= 0 or not math.isclose(steps * DT_PS, duration_ps, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{name} must be representable as a positive integer number of base steps.")
        if steps % EXACT_RESPA_FACTOR != 0:
            raise ValueError(f"{name} must be a multiple of the exact-r-RESPA factor.")


def validate_prerequisites(scaffold_manifest: dict[str, object], bringup_result: dict[str, object]) -> None:
    if str(scaffold_manifest["derived_system"]) != "gate_h_dense_oligomer_2x2x2":
        raise ValueError("This entry validator is currently frozen only for the neutral dense scaffold.")
    if bringup_result.get("status") != "PASS":
        raise ValueError("Short bring-up is not PASS; longer neutral entry runs should not proceed.")


def make_nvt_mdp(*, duration_ps: float, sample_interval: int, phase: str, seed: int, args: argparse.Namespace) -> str:
    nsteps = steps_from_ps(duration_ps)
    thermostat = (
        "tcoupl                  = v-rescale\n"
        "tc-grps                 = System\n"
        f"tau-t                   = {args.tau_t_ps:.3f}\n"
        f"ref-t                   = {args.temperature_k:.3f}\n"
        f"nsttcouple              = {EXACT_RESPA_FACTOR}\n"
    )
    velocity = (
        "gen-vel                 = yes\n"
        f"gen-temp                = {args.temperature_k:.3f}\n"
        f"gen-seed                = {seed}\n"
        if phase == "equil"
        else "gen-vel                 = no\n"
    )
    return (
        f"title                   = gate h neutral entry {phase} exact respa\n"
        + exact_respa_common_mdp(nsteps, sample_interval)
        + thermostat
        + "pcoupl                  = no\n"
        + velocity
    )


def system_status(comparisons: dict[str, dict[str, object]]) -> tuple[str, str | None, list[str]]:
    reasons: list[str] = []
    first_failing: str | None = None
    for term in REQUESTED_OBSERVABLES:
        payload = comparisons.get(term, {})
        if not payload.get("available"):
            reasons.append(f"{term} is missing.")
            first_failing = first_failing or term
            continue
        if not payload.get("passes"):
            reasons.append(f"{term} exceeds the current combined uncertainty budget ({payload['classification']}).")
            first_failing = first_failing or term
    return ("FAIL", first_failing, reasons) if first_failing is not None else ("PASS", None, [])


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def remove_incomplete_mdrun_outputs(deffnm: Path) -> None:
    for suffix in (".cpt", ".edr", ".gro", ".log", ".trr", ".xtc"):
        path = deffnm.with_suffix(suffix)
        if path.exists():
            path.unlink()


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
    if mdrun_complete(deffnm):
        return {
            "run_id": label,
            "argv": argv,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "returncode": 0,
            "layout_report": parse_layout_report(stdout_path, stderr_path, args),
            "resumed_from_existing_artifacts": True,
        }
    remove_incomplete_mdrun_outputs(deffnm)
    payload = run_md(
        gmx=gmx,
        argv=argv,
        env=env,
        logs_dir=logs_dir,
        commands=[],
        label=label,
        args=args,
    )
    payload["resumed_from_existing_artifacts"] = False
    return payload


def main() -> int:
    args = parse_args()
    validate_args(args)
    gmx = Path(args.gmx)
    scaffold_manifest = load_json(Path(args.scaffold_manifest))
    bringup_result = load_json(Path(args.bringup_result))
    validate_prerequisites(scaffold_manifest, bringup_result)

    out_root = Path(args.out).resolve()
    if out_root.exists() and not args.resume:
        shutil.rmtree(out_root)
    inputs_dir = out_root / "inputs"
    logs_dir = out_root / "logs"
    summaries_dir = out_root / "summaries"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    version_text = Path(args.gmx)
    gmx_version = os.popen(f"{gmx} --version").read()
    top_path = Path(str(scaffold_manifest["artifacts"]["topology"]))
    gro_path = Path(str(scaffold_manifest["artifacts"]["gro"]))
    commands: list[dict[str, object]] = []
    env = base_env(args)
    per_layout_runs: dict[str, list[dict[str, object]]] = {"cpu": [], "gpu": []}

    for replica_index in range(1, args.replicas + 1):
        seed = 41001 + replica_index - 1
        replica_inputs = inputs_dir / f"replica_{replica_index:02d}"
        replica_inputs.mkdir(parents=True, exist_ok=True)
        equil_mdp = replica_inputs / "equil.mdp"
        prod_mdp = replica_inputs / "prod.mdp"
        write_text(equil_mdp, make_nvt_mdp(duration_ps=args.equil_ps, sample_interval=args.sample_interval, phase="equil", seed=seed, args=args))
        write_text(prod_mdp, make_nvt_mdp(duration_ps=args.prod_ps, sample_interval=args.sample_interval, phase="prod", seed=seed, args=args))

        for layout in ("cpu", "gpu"):
            run_root = out_root / layout / f"replica_{replica_index:02d}"
            run_root.mkdir(parents=True, exist_ok=True)
            equil_tpr = run_root / "equil.tpr"
            prod_tpr = run_root / "prod.tpr"
            equil_mdout = run_root / "equil.mdout.mdp"
            prod_mdout = run_root / "prod.mdout.mdp"
            equil_deffnm = run_root / "equil"
            prod_deffnm = run_root / "prod"

            run_or_resume_grompp(
                gmx=gmx,
                mdp_path=equil_mdp,
                conf_path=gro_path,
                top_path=top_path,
                tpr_path=equil_tpr,
                mdout_path=equil_mdout,
                logs_dir=logs_dir,
                commands=commands,
                label=f"{layout}_replica_{replica_index:02d}_grompp_equil",
                env=env,
            )
            md_payload = run_or_resume_md(
                gmx=gmx,
                argv=[str(gmx), "mdrun", *(mdrun_args_cpu(args, equil_deffnm) if layout == "cpu" else mdrun_args_gpu(args, equil_deffnm))],
                deffnm=equil_deffnm,
                env=env,
                logs_dir=logs_dir,
                commands=commands,
                label=f"{layout}_replica_{replica_index:02d}_mdrun_equil",
                args=args,
            )

            run_or_resume_grompp(
                gmx=gmx,
                mdp_path=prod_mdp,
                conf_path=equil_deffnm.with_suffix(".gro"),
                top_path=top_path,
                tpr_path=prod_tpr,
                mdout_path=prod_mdout,
                logs_dir=logs_dir,
                commands=commands,
                label=f"{layout}_replica_{replica_index:02d}_grompp_prod",
                env=env,
                checkpoint_path=equil_deffnm.with_suffix(".cpt"),
            )
            md_payload = run_or_resume_md(
                gmx=gmx,
                argv=[str(gmx), "mdrun", *(mdrun_args_cpu(args, prod_deffnm) if layout == "cpu" else mdrun_args_gpu(args, prod_deffnm))],
                deffnm=prod_deffnm,
                env=env,
                logs_dir=logs_dir,
                commands=commands,
                label=f"{layout}_replica_{replica_index:02d}_mdrun_prod",
                args=args,
            )
            observables = collect_run_observables(gmx, prod_deffnm, run_root / "prod_energy.xvg")
            per_layout_runs[layout].append(
                {
                    "replica_index": replica_index,
                    "layout_report": md_payload["layout_report"],
                    "prod_deffnm": str(prod_deffnm),
                    "observables": observables,
                }
            )

    aggregates: dict[str, dict[str, object]] = {}
    for layout, runs in per_layout_runs.items():
        layout_observables: dict[str, object] = {}
        for term in REQUESTED_OBSERVABLES:
            layout_observables[term] = aggregate_replicates(
                [run["observables"][term] for run in runs if run["observables"][term].get("available")]
            )
        aggregates[layout] = {"replica_count": len(runs), "observables": layout_observables}

    comparisons = {
        term: compare_layout_observable(aggregates["cpu"]["observables"][term], aggregates["gpu"]["observables"][term])
        for term in REQUESTED_OBSERVABLES
    }
    status, first_failing, reasons = system_status(comparisons)

    save_observable_table(summaries_dir / "observable_comparison.tsv", comparisons)
    result = {
        "schema_version": 1,
        "status": status,
        "first_failing_observable": first_failing,
        "failure_reasons": reasons,
        "system_id": str(scaffold_manifest["derived_system"]),
        "scope": "large-neutral exact-r-RESPA entry gate before full Gate H transport runs",
        "limits": [
            "This is not TP0 transport sign-off.",
            "No MSD/diffusion/conductivity estimator is evaluated here.",
            "The purpose is to establish a CPU/GPU variance budget on the large neutral scaffold before long transport runs."
        ],
        "gmx_binary": str(gmx),
        "gpu_support": parse_gpu_support(gmx_version),
        "precision_mode": parse_precision_mode(gmx_version),
        "short_prerequisite": str(Path(args.bringup_result).resolve()),
        "scaffold_manifest": str(Path(args.scaffold_manifest).resolve()),
        "run_settings": {
            "replicas": args.replicas,
            "equil_ps": args.equil_ps,
            "prod_ps": args.prod_ps,
            "sample_interval_base_steps": args.sample_interval,
            "temperature_k": args.temperature_k,
            "tau_t_ps": args.tau_t_ps,
            "ntmpi": args.ntmpi,
            "ntomp": args.ntomp,
            "dlb": "no",
            "cpu_shape": "nb cpu / bonded cpu / pme cpu / update cpu",
            "gpu_shape": "nb gpu / bonded gpu / pme gpu / update gpu",
        },
        "layout_report": per_layout_runs["gpu"][0]["layout_report"],
        "aggregates": sanitize_layout_aggregates(aggregates),
        "comparisons": comparisons,
        "per_layout_runs": per_layout_runs,
        "recommendation": (
            "Proceed to longer neutral Gate G/H-style runs only if this entry gate passes and no structural temperature/pressure/potential drift appears."
            if status == "PASS"
            else "Do not start long neutral transport runs until the first failing observable is reconciled."
        ),
    }
    write_json(summaries_dir / "entry_result.json", result)
    write_commands_script(out_root / "run_commands.sh", commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
