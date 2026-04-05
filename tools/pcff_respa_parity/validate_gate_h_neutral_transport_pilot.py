from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path

from freeze_gate_a_oracle import DEFAULT_GMX, REPO_ROOT, base_env, capture_output, env_delta, write_text
from validate_gate_b_nb_gpu import load_json, parse_gpu_support, parse_precision_mode
from validate_gate_c_nb_bonded_gpu import maybe_build
from validate_gate_e_update_gpu import parse_layout_report
from validate_gate_g_long_ensemble import exact_respa_common_mdp, run_grompp
from validate_gate_h_transport import (
    compare_scalar_aggregates,
    mean,
    record_command,
    run_md,
    sample_std,
    sem,
    summarize_scalar_replicas,
    write_recorded_commands_script,
)


DEFAULT_SCAFFOLD_MANIFEST = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_h_fixture_scaffold"
    / "gate_h_dense_oligomer_2x2x2"
    / "fixture_manifest.json"
)
DEFAULT_ENTRY_RESULT = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_h_neutral_entry_validation_longer"
    / "summaries"
    / "entry_result.json"
)
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_h_neutral_transport_pilot"

DT_PS = 0.0005
EXACT_RESPA_FACTOR = 4
DEFAULT_EQ_PS = 10.0
DEFAULT_PROD_PS = 100.0
DEFAULT_REPLICAS = 2
DEFAULT_TEMP_K = 300.0
DEFAULT_TAU_T_PS = 0.5
DEFAULT_COORD_STRIDE_PS = 0.5
DEFAULT_ENERGY_STRIDE_PS = 1.0
DEFAULT_TRESTART_PS = 2.0
DEFAULT_MAXTAU_PS = 50.0
FIT_WINDOW = (0.10, 0.90)
MOL_DIFF_SCALE = 1.0e-5
R2_THRESHOLD = 0.98
RELATIVE_UNCERTAINTY_THRESHOLD = 0.50
SYSTEM_ID = "gate_h_dense_oligomer_2x2x2"
INDEX_GROUP_NAME = "OLI"
DIFFUSIVITY_METRIC = "oligomer_diffusivity_cm2_s"

MSD_LEGEND_RE = re.compile(
    r'@ s0 legend "D\[\s*(?P<label>[^\]]+)\] = (?P<diff>[-+0-9.eE]+) \(\+/- (?P<err>[-+0-9.eE]+)\) \(1e-5 cm\^2/s\)"'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a large-neutral Gate H transport pilot using molecule-wise MSD on the exact-r-RESPA path."
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument("--scaffold-manifest", default=str(DEFAULT_SCAFFOLD_MANIFEST), help="Large neutral scaffold manifest.")
    parser.add_argument(
        "--entry-result",
        default=str(DEFAULT_ENTRY_RESULT),
        help="Prerequisite large-neutral entry result JSON. Must be PASS.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--build-target", default="gmx", help="CMake target to build.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step.")
    parser.add_argument("--build-dir", default=None, help="Optional explicit build directory.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks.")
    parser.add_argument("--ntomp", type=int, default=1, help="OpenMP threads.")
    parser.add_argument("--npme", type=int, default=None, help="Optional explicit -npme value; omitted by default.")
    parser.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS, help="Replica count per layout.")
    parser.add_argument("--equil-ps", type=float, default=DEFAULT_EQ_PS, help="Equilibration duration in ps.")
    parser.add_argument("--prod-ps", type=float, default=DEFAULT_PROD_PS, help="Production duration in ps.")
    parser.add_argument(
        "--coord-stride-ps",
        type=float,
        default=DEFAULT_COORD_STRIDE_PS,
        help="Compressed-coordinate trajectory stride in ps.",
    )
    parser.add_argument(
        "--energy-stride-ps",
        type=float,
        default=DEFAULT_ENERGY_STRIDE_PS,
        help="Energy/log sampling stride in ps.",
    )
    parser.add_argument("--temperature-k", type=float, default=DEFAULT_TEMP_K, help="Target temperature.")
    parser.add_argument("--tau-t-ps", type=float, default=DEFAULT_TAU_T_PS, help="Thermostat coupling time.")
    parser.add_argument(
        "--trestart-ps",
        type=float,
        default=DEFAULT_TRESTART_PS,
        help="gmx msd restart interval in ps.",
    )
    parser.add_argument(
        "--maxtau-ps",
        type=float,
        default=DEFAULT_MAXTAU_PS,
        help="Maximum time delta for gmx msd in ps.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed stage artifacts under --out instead of deleting the output root.",
    )
    return parser.parse_args()


def steps_from_ps(duration_ps: float) -> int:
    return int(round(duration_ps / DT_PS))


def validate_args(args: argparse.Namespace) -> None:
    if args.ntmpi != 1:
        raise ValueError("Neutral transport pilot is restricted to single-rank runs (ntmpi=1).")
    if args.npme is not None:
        raise ValueError("Neutral transport pilot keeps the canonical single-rank layout; omit -npme.")
    if args.replicas < 2:
        raise ValueError("Neutral transport pilot requires replicated runs; use at least 2 replicas.")
    for name, duration_ps in (
        ("equil-ps", args.equil_ps),
        ("prod-ps", args.prod_ps),
        ("coord-stride-ps", args.coord_stride_ps),
        ("energy-stride-ps", args.energy_stride_ps),
        ("trestart-ps", args.trestart_ps),
        ("maxtau-ps", args.maxtau_ps),
    ):
        steps = steps_from_ps(duration_ps)
        if steps <= 0 or not math.isclose(steps * DT_PS, duration_ps, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{name} must be representable as a positive integer number of base steps.")
        if steps % EXACT_RESPA_FACTOR != 0:
            raise ValueError(f"{name} must be a multiple of the exact-r-RESPA factor.")


def validate_prerequisites(scaffold_manifest: dict[str, object], entry_result: dict[str, object]) -> None:
    if str(scaffold_manifest["derived_system"]) != SYSTEM_ID:
        raise ValueError("Neutral transport pilot is currently frozen only for gate_h_dense_oligomer_2x2x2.")
    if entry_result.get("status") != "PASS":
        raise ValueError("Large-neutral entry gate is not PASS; transport pilot should not proceed.")


def make_transport_pilot_mdp(
    *,
    duration_ps: float,
    coord_stride_ps: float,
    energy_stride_ps: float,
    phase: str,
    seed: int,
    args: argparse.Namespace,
) -> str:
    nsteps = steps_from_ps(duration_ps)
    coord_stride_steps = steps_from_ps(coord_stride_ps)
    energy_stride_steps = steps_from_ps(energy_stride_ps)
    base_exact_mdp = (
        exact_respa_common_mdp(nsteps, energy_stride_steps)
        .replace("nstxout                 = 0\n", "")
        .replace("nstvout                 = 0\n", "")
        .replace("nstfout                 = 0\n", "")
        .replace("nstxout-compressed      = 0\n", "")
    )
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
        f"title                   = gate h neutral transport pilot {phase} exact respa {SYSTEM_ID}\n"
        + base_exact_mdp
        + thermostat
        + "pcoupl                  = no\n"
        + velocity
        + "nstxout                 = 0\n"
        + "nstvout                 = 0\n"
        + "nstfout                 = 0\n"
        + f"nstxout-compressed      = {coord_stride_steps}\n"
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


def mdrun_args_gpu(args: argparse.Namespace, deffnm: Path) -> list[str]:
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
        deffnm.with_suffix(".xtc"),
    )
    return all(file_exists_and_nonempty(path) for path in required)


def analysis_complete(msdout_path: Path, diffmol_path: Path) -> bool:
    return file_exists_and_nonempty(msdout_path) and file_exists_and_nonempty(diffmol_path)


def remove_incomplete_mdrun_outputs(deffnm: Path) -> None:
    for suffix in (".cpt", ".edr", ".gro", ".log", ".xtc", ".trr"):
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
        record_command(
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
        record_command(
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


def write_all_atoms_index(path: Path, natoms: int) -> None:
    lines = [f"[ {INDEX_GROUP_NAME} ]"]
    atoms = list(range(1, natoms + 1))
    for start in range(0, len(atoms), 15):
        lines.append(" ".join(str(atom) for atom in atoms[start : start + 15]))
    write_text(path, "\n".join(lines) + "\n")


def run_or_resume_gmx_msd(
    *,
    gmx: Path,
    traj_path: Path,
    tpr_path: Path,
    ndx_path: Path,
    msdout_path: Path,
    diffmol_path: Path,
    logs_dir: Path,
    commands: list[dict[str, object]],
    label: str,
    env: dict[str, str],
    args: argparse.Namespace,
) -> None:
    argv = [
        str(gmx),
        "msd",
        "-f",
        str(traj_path),
        "-s",
        str(tpr_path),
        "-n",
        str(ndx_path),
        "-o",
        str(msdout_path),
        "-mol",
        str(diffmol_path),
        "-trestart",
        str(args.trestart_ps),
        "-maxtau",
        str(args.maxtau_ps),
    ]
    stdin_text = "0\n"
    stdout_path = logs_dir / f"{label}.stdout"
    stderr_path = logs_dir / f"{label}.stderr"
    commands.append(
        record_command(
            label,
            argv,
            cwd=REPO_ROOT,
            env_overrides=env_delta(env, os.environ),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdin_text=stdin_text,
        )
    )
    if analysis_complete(msdout_path, diffmol_path):
        return
    if msdout_path.exists():
        msdout_path.unlink()
    if diffmol_path.exists():
        diffmol_path.unlink()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            env=env,
            input=stdin_text,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed; see {stderr_path}")


def parse_msd_curve(path: Path) -> tuple[dict[str, object], list[float], list[float]]:
    legend = None
    times_ps: list[float] = []
    msd_nm2: list[float] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("@"):
            match = MSD_LEGEND_RE.match(line)
            if match is not None:
                legend = {
                    "label": match.group("label").strip(),
                    "diffusion_cm2_s": float(match.group("diff")) * MOL_DIFF_SCALE,
                    "fit_error_cm2_s": float(match.group("err")) * MOL_DIFF_SCALE,
                }
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            times_ps.append(float(parts[0]))
            msd_nm2.append(float(parts[1]))
    if legend is None:
        raise ValueError(f"Missing MSD legend in {path}")
    return legend, times_ps, msd_nm2


def parse_diff_mol(path: Path) -> list[float]:
    values: list[float] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("@"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            values.append(float(parts[1]) * MOL_DIFF_SCALE)
    return values


def linear_fit(xs: list[float], ys: list[float]) -> dict[str, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return {"valid": False}
    mean_x = mean(xs)
    mean_y = mean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0.0:
        return {"valid": False}
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "valid": True,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
    }


def determine_fit_window(times_ps: list[float]) -> tuple[float, float]:
    if not times_ps:
        return 0.0, 0.0
    max_tau = times_ps[-1]
    return FIT_WINDOW[0] * max_tau, FIT_WINDOW[1] * max_tau


def save_curve_tsv(path: Path, times_ps: list[float], msd_nm2: list[float]) -> None:
    rows = ["tau_ps\tmsd_nm2\n"]
    rows.extend(f"{tau:.10g}\t{msd:.10g}\n" for tau, msd in zip(times_ps, msd_nm2))
    write_text(path, "".join(rows))


def analyze_replica(
    *,
    gmx: Path,
    natoms: int,
    prod_deffnm: Path,
    replica_root: Path,
    logs_dir: Path,
    commands: list[dict[str, object]],
    env: dict[str, str],
    label_prefix: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    ndx_path = replica_root / "transport_molecules.ndx"
    write_all_atoms_index(ndx_path, natoms)
    msdout_path = replica_root / "msdout.xvg"
    diffmol_path = replica_root / "diff_mol.xvg"
    run_or_resume_gmx_msd(
        gmx=gmx,
        traj_path=prod_deffnm.with_suffix(".xtc"),
        tpr_path=prod_deffnm.with_suffix(".tpr"),
        ndx_path=ndx_path,
        msdout_path=msdout_path,
        diffmol_path=diffmol_path,
        logs_dir=logs_dir,
        commands=commands,
        label=f"{label_prefix}_msd",
        env=env,
        args=args,
    )

    legend, times_ps, msd_nm2 = parse_msd_curve(msdout_path)
    diff_values = parse_diff_mol(diffmol_path)
    fit_start_ps, fit_end_ps = determine_fit_window(times_ps)
    fit_xs = [tau for tau in times_ps if fit_start_ps <= tau <= fit_end_ps]
    fit_ys = [value for tau, value in zip(times_ps, msd_nm2) if fit_start_ps <= tau <= fit_end_ps]
    fit = linear_fit(fit_xs, fit_ys)
    curve_tsv = replica_root / "msd_curve.tsv"
    save_curve_tsv(curve_tsv, times_ps, msd_nm2)

    molecule_mean = mean(diff_values)
    molecule_sem = sem(diff_values)
    molecule_std = sample_std(diff_values)
    relative_fit_error = math.inf if legend["diffusion_cm2_s"] == 0.0 else abs(legend["fit_error_cm2_s"] / legend["diffusion_cm2_s"])
    relative_molecule_sem = math.inf if molecule_mean == 0.0 else abs(molecule_sem / molecule_mean)

    convergence_flags: list[str] = []
    if not fit.get("valid", False):
        convergence_flags.append("MSD fit window is invalid.")
    elif fit["r_squared"] < R2_THRESHOLD:
        convergence_flags.append(f"MSD linearity is weak (R^2={fit['r_squared']:.6f} < {R2_THRESHOLD:.2f}).")
    if relative_fit_error >= RELATIVE_UNCERTAINTY_THRESHOLD:
        convergence_flags.append(
            f"MSD fit uncertainty is large (relative fit error {relative_fit_error:.3f} >= {RELATIVE_UNCERTAINTY_THRESHOLD:.2f})."
        )
    if relative_molecule_sem >= RELATIVE_UNCERTAINTY_THRESHOLD:
        convergence_flags.append(
            f"Per-molecule diffusivity spread is large (molecule SEM / mean = {relative_molecule_sem:.3f} >= {RELATIVE_UNCERTAINTY_THRESHOLD:.2f})."
        )

    return {
        "transport_index": str(ndx_path),
        "msdout_xvg": str(msdout_path),
        "diff_mol_xvg": str(diffmol_path),
        "msd_curve_tsv": str(curve_tsv),
        "diffusion_cm2_s": legend["diffusion_cm2_s"],
        "fit_error_cm2_s": legend["fit_error_cm2_s"],
        "fit_start_ps": fit_start_ps,
        "fit_end_ps": fit_end_ps,
        "fit_r_squared": fit.get("r_squared"),
        "molecule_count": len(diff_values),
        "molecule_mean_cm2_s": molecule_mean,
        "molecule_std_cm2_s": molecule_std,
        "molecule_sem_cm2_s": molecule_sem,
        "molecule_values_cm2_s": diff_values,
        "relative_fit_error": relative_fit_error,
        "relative_molecule_sem": relative_molecule_sem,
        "convergence_flags": convergence_flags,
    }


def build_manifest_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Gate H Neutral Transport Pilot",
        "",
        f"- Pilot status: `{result['status']}`",
        f"- Replica count per layout: `{result['run_settings']['replicas']}`",
        f"- Equilibration / production: `{result['run_settings']['equil_ps']} ps / {result['run_settings']['prod_ps']} ps`",
        f"- Coord stride: `{result['run_settings']['coord_stride_ps']} ps`",
        f"- MSD estimator: `{result['analysis_settings']['estimator']}`",
        f"- Recommendation: {result['recommendation']}",
        "",
        "## Observable",
        f"- `{DIFFUSIVITY_METRIC}`: `{result['comparison']['classification']}` / passes=`{result['comparison']['passes']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    validate_args(args)

    scaffold_manifest = load_json(Path(args.scaffold_manifest))
    entry_result = load_json(Path(args.entry_result))
    validate_prerequisites(scaffold_manifest, entry_result)

    gmx = Path(args.gmx).resolve()
    maybe_build(args, Path(args.build_dir).resolve() if args.build_dir is not None else None)

    out_root = Path(args.out).resolve()
    if out_root.exists() and not args.resume:
        import shutil

        shutil.rmtree(out_root)
    inputs_dir = out_root / "inputs"
    logs_dir = out_root / "logs"
    summaries_dir = out_root / "summaries"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    version_text = capture_output([str(gmx), "--version"], cwd=REPO_ROOT, env=os.environ.copy())
    env = base_env(args)
    top_path = Path(str(scaffold_manifest["artifacts"]["topology"]))
    gro_path = Path(str(scaffold_manifest["artifacts"]["gro"]))
    natoms = int(scaffold_manifest["natoms"])

    commands: list[dict[str, object]] = []
    per_layout_replicas: dict[str, list[dict[str, object]]] = {"cpu": [], "gpu": []}
    convergence_issues: list[str] = []

    for replica_index in range(1, args.replicas + 1):
        seed = 63101 + replica_index - 1
        replica_inputs = inputs_dir / f"replica_{replica_index:02d}"
        replica_inputs.mkdir(parents=True, exist_ok=True)
        equil_mdp = replica_inputs / "equil.mdp"
        prod_mdp = replica_inputs / "prod.mdp"
        write_text(
            equil_mdp,
            make_transport_pilot_mdp(
                duration_ps=args.equil_ps,
                coord_stride_ps=args.coord_stride_ps,
                energy_stride_ps=args.energy_stride_ps,
                phase="equil",
                seed=seed,
                args=args,
            ),
        )
        write_text(
            prod_mdp,
            make_transport_pilot_mdp(
                duration_ps=args.prod_ps,
                coord_stride_ps=args.coord_stride_ps,
                energy_stride_ps=args.energy_stride_ps,
                phase="prod",
                seed=seed,
                args=args,
            ),
        )

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
            analysis = analyze_replica(
                gmx=gmx,
                natoms=natoms,
                prod_deffnm=prod_deffnm,
                replica_root=run_root,
                logs_dir=logs_dir,
                commands=commands,
                env=env,
                label_prefix=f"{layout}_replica_{replica_index:02d}",
                args=args,
            )
            replica_summary = {
                "replica_index": replica_index,
                "seed": seed,
                "layout": layout,
                "layout_report": md_payload["layout_report"],
                "equil_deffnm": str(equil_deffnm),
                "prod_deffnm": str(prod_deffnm),
                "analysis": analysis,
            }
            write_json(run_root / "replica_summary.json", replica_summary)
            per_layout_replicas[layout].append(replica_summary)
            convergence_issues.extend(f"{layout} replica {replica_index:02d}: {issue}" for issue in analysis["convergence_flags"])

    aggregates: dict[str, dict[str, object]] = {}
    for layout in ("cpu", "gpu"):
        scalar_replicas = []
        for replica in per_layout_replicas[layout]:
            analysis = replica["analysis"]
            scalar_replicas.append(
                {
                    "replica_index": replica["replica_index"],
                    "value": analysis["diffusion_cm2_s"],
                    "block_sem": max(float(analysis["fit_error_cm2_s"]), float(analysis["molecule_sem_cm2_s"])),
                    "r_squared": analysis["fit_r_squared"],
                    "fit_error_cm2_s": analysis["fit_error_cm2_s"],
                    "molecule_sem_cm2_s": analysis["molecule_sem_cm2_s"],
                }
            )
        aggregates[layout] = summarize_scalar_replicas(scalar_replicas)
    comparison = compare_scalar_aggregates(aggregates["cpu"], aggregates["gpu"])

    transport_table = (
        "observable\tcpu_mean\tgpu_mean\tmean_diff\tcombined_uncertainty\tclassification\tpasses\n"
        + (
            f"{DIFFUSIVITY_METRIC}\t{comparison['cpu_mean']:.10g}\t{comparison['gpu_mean']:.10g}\t"
            f"{comparison['mean_diff']:.10g}\t{comparison['combined_uncertainty']:.10g}\t"
            f"{comparison['classification']}\t{comparison['passes']}\n"
            if comparison.get("available")
            else f"{DIFFUSIVITY_METRIC}\tNA\tNA\tNA\tNA\tmissing\tFalse\n"
        )
    )
    write_text(summaries_dir / "transport_comparison.tsv", transport_table)

    failure_reasons: list[str] = []
    first_failing = None
    if not comparison.get("available"):
        failure_reasons.append(f"{DIFFUSIVITY_METRIC} comparison is unavailable.")
        first_failing = first_failing or DIFFUSIVITY_METRIC
    elif not comparison.get("passes"):
        failure_reasons.append(
            f"{DIFFUSIVITY_METRIC} exceeds the current combined uncertainty budget ({comparison['classification']})."
        )
        first_failing = first_failing or DIFFUSIVITY_METRIC
    if convergence_issues:
        failure_reasons.extend(convergence_issues)
        first_failing = first_failing or "finite_sampling"

    status = "PASS" if first_failing is None else "FAIL"
    result = {
        "schema_version": 1,
        "scope": "Large-neutral Gate H molecule-wise MSD/diffusivity pilot only; not TP0 production sign-off.",
        "system_id": SYSTEM_ID,
        "status": status,
        "first_failing_observable": first_failing,
        "failure_reasons": failure_reasons,
        "gmx_binary": str(gmx),
        "gpu_support": parse_gpu_support(version_text),
        "precision_mode": parse_precision_mode(version_text),
        "prerequisites": {
            "entry_result": str(Path(args.entry_result).resolve()),
            "entry_status": entry_result.get("status"),
            "scaffold_manifest": str(Path(args.scaffold_manifest).resolve()),
        },
        "run_settings": {
            "replicas": args.replicas,
            "equil_ps": args.equil_ps,
            "prod_ps": args.prod_ps,
            "coord_stride_ps": args.coord_stride_ps,
            "energy_stride_ps": args.energy_stride_ps,
            "temperature_k": args.temperature_k,
            "tau_t_ps": args.tau_t_ps,
            "trestart_ps": args.trestart_ps,
            "maxtau_ps": args.maxtau_ps,
            "ntmpi": args.ntmpi,
            "ntomp": args.ntomp,
            "dlb": "no",
            "cpu_shape": "nb cpu / bonded cpu / pme cpu / update cpu",
            "gpu_shape": "nb gpu / bonded gpu / pme gpu / update gpu",
        },
        "analysis_settings": {
            "estimator": "gmx msd -mol on the all-oligomer atom group, split into molecules by topology.",
            "fit_window_ratio": list(FIT_WINDOW),
            "r_squared_threshold": R2_THRESHOLD,
            "relative_uncertainty_threshold": RELATIVE_UNCERTAINTY_THRESHOLD,
        },
        "aggregates": aggregates,
        "comparison": comparison,
        "per_layout_replicas": per_layout_replicas,
        "recommendation": (
            "Neutral MSD pilot is internally consistent enough to justify longer neutral Gate H scaling, but this is still not TP0 production sign-off."
            if status == "PASS"
            else "Do not treat neutral transport as validated yet; fix the first failing observable or finite-sampling issue first."
        ),
        "production_recommendation": "NO-GO",
    }
    write_json(summaries_dir / "pilot_result.json", result)
    write_text(summaries_dir / "pilot_result.md", build_manifest_markdown(result))
    write_recorded_commands_script(out_root / "run_commands.sh", commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
