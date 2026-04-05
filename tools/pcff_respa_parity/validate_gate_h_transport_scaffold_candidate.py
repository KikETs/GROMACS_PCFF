from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from freeze_gate_a_oracle import DEFAULT_GMX, REPO_ROOT, base_env, capture_output, env_delta, write_text
from validate_gate_b_nb_gpu import load_json, parse_gpu_support, parse_precision_mode
from validate_gate_c_nb_bonded_gpu import maybe_build
from validate_gate_e_update_gpu import parse_layout_report
from validate_gate_g_long_ensemble import exact_respa_common_mdp, extract_energy_series, run_grompp
from validate_gate_h_neutral_transport_pilot import (
    DEFAULT_TRESTART_PS,
    DEFAULT_MAXTAU_PS,
    FIT_WINDOW,
    R2_THRESHOLD,
    RELATIVE_UNCERTAINTY_THRESHOLD,
    linear_fit,
    parse_diff_mol,
    parse_msd_curve,
    run_or_resume_grompp,
    run_or_resume_md,
    save_curve_tsv,
)
from validate_gate_h_transport import (
    DEFAULT_GATE_A_MANIFEST,
    DEFAULT_GATE_E_MANIFEST,
    DEFAULT_GATE_F_MANIFEST,
    DEFAULT_GATE_G_MANIFEST,
    DT_PS,
    EXACT_RESPA_FACTOR,
    E_CHARGE_C,
    K_BOLTZMANN_J_K,
    M3_PER_NM3,
    compare_scalar_aggregates,
    mean,
    parse_box_from_gro,
    record_command,
    sample_std,
    sem,
    summarize_scalar_replicas,
    validate_gate_chain,
)


DEFAULT_NEUTRAL_SCAFFOLD = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_h_fixture_scaffold"
    / "gate_h_dense_oligomer_2x2x2"
    / "fixture_manifest.json"
)
DEFAULT_CHARGED_SCAFFOLD = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_h_fixture_scaffold"
    / "gate_h_dense_salt_polymer_2x2x2"
    / "fixture_manifest.json"
)
DEFAULT_NEUTRAL_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_h_neutral_transport_candidate"
DEFAULT_CHARGED_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_h_charged_transport_candidate"

DEFAULT_EQ_PS = 10.0
DEFAULT_PROD_PS = 100.0
DEFAULT_REPLICAS = 2
DEFAULT_TEMP_K = 300.0
DEFAULT_TAU_T_PS = 0.5
DEFAULT_COORD_STRIDE_PS = 0.5
DEFAULT_ENERGY_STRIDE_PS = 1.0
NM2_PS_TO_CM2_S = 1.0e-2
FIT_WINDOW_CANDIDATES = (
    (0.10, 0.90),
    (0.20, 0.80),
    (0.20, 0.70),
    (0.30, 0.80),
    (0.30, 0.70),
    (0.30, 0.60),
    (0.40, 0.80),
    (0.40, 0.70),
    (0.50, 0.90),
)
MIN_FIT_WINDOW_SPAN_RATIO = 0.30
MIN_FIT_POINTS = 40
NM2_PS_TO_M2_S = 1.0e-6

ENERGY_TERMS = ("Temperature", "Volume", "Box-X", "Box-Y", "Box-Z")

PRESETS: dict[str, dict[str, object]] = {
    "neutral-large": {
        "manifest": DEFAULT_NEUTRAL_SCAFFOLD,
        "out": DEFAULT_NEUTRAL_OUT,
        "system_id": "gate_h_dense_oligomer_2x2x2",
        "species": (
            {
                "name": "oligomer",
                "metric": "oligomer_diffusivity_cm2_s",
                "molecule_types": ("MOL1",),
                "charge": 0,
                "primary": True,
            },
        ),
        "derived_metrics": (),
    },
    "charged-large": {
        "manifest": DEFAULT_CHARGED_SCAFFOLD,
        "out": DEFAULT_CHARGED_OUT,
        "system_id": "gate_h_dense_salt_polymer_2x2x2",
        "species": (
            {
                "name": "polymer",
                "metric": "polymer_diffusivity_cm2_s",
                "molecule_types": ("POL", "POL10", "POL28"),
                "charge": 0,
                "primary": False,
            },
            {
                "name": "cation",
                "metric": "cation_diffusivity_cm2_s",
                "molecule_types": ("CAT",),
                "charge": 1,
                "primary": True,
            },
            {
                "name": "anion",
                "metric": "anion_diffusivity_cm2_s",
                "molecule_types": ("ANI",),
                "charge": -1,
                "primary": True,
            },
        ),
        "derived_metrics": ("conductivity_cne_s_cm", "transference_ne"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Gate H transport candidate on large scaffold fixtures using molecule-wise MSD per species."
    )
    parser.add_argument("--preset", choices=tuple(PRESETS.keys()), required=True, help="Candidate preset to run.")
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument("--gate-a-manifest", default=str(DEFAULT_GATE_A_MANIFEST))
    parser.add_argument("--gate-e-manifest", default=str(DEFAULT_GATE_E_MANIFEST))
    parser.add_argument("--gate-f-manifest", default=str(DEFAULT_GATE_F_MANIFEST))
    parser.add_argument("--gate-g-manifest", default=str(DEFAULT_GATE_G_MANIFEST))
    parser.add_argument("--scaffold-manifest", default=None, help="Override scaffold manifest path.")
    parser.add_argument("--out", default=None, help="Override output root.")
    parser.add_argument("--build-target", default="gmx")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-dir", default=None)
    parser.add_argument("--ntmpi", type=int, default=1)
    parser.add_argument("--ntomp", type=int, default=1)
    parser.add_argument("--npme", type=int, default=None)
    parser.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS)
    parser.add_argument("--equil-ps", type=float, default=DEFAULT_EQ_PS)
    parser.add_argument("--prod-ps", type=float, default=DEFAULT_PROD_PS)
    parser.add_argument("--coord-stride-ps", type=float, default=DEFAULT_COORD_STRIDE_PS)
    parser.add_argument("--energy-stride-ps", type=float, default=DEFAULT_ENERGY_STRIDE_PS)
    parser.add_argument("--temperature-k", type=float, default=DEFAULT_TEMP_K)
    parser.add_argument("--tau-t-ps", type=float, default=DEFAULT_TAU_T_PS)
    parser.add_argument("--trestart-ps", type=float, default=DEFAULT_TRESTART_PS)
    parser.add_argument("--maxtau-ps", type=float, default=DEFAULT_MAXTAU_PS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def steps_from_ps(duration_ps: float) -> int:
    return int(round(duration_ps / DT_PS))


def validate_args(args: argparse.Namespace) -> None:
    if args.ntmpi != 1:
        raise ValueError("Gate H scaffold candidate is restricted to single-rank runs (ntmpi=1).")
    if args.npme is not None:
        raise ValueError("Gate H scaffold candidate keeps the canonical single-rank layout; omit -npme.")
    if args.replicas < 2:
        raise ValueError("Gate H scaffold candidate requires replicated runs.")
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


def validate_gate_prerequisites(
    gate_a_manifest: dict[str, object],
    gate_e_manifest: dict[str, object],
    gate_f_manifest: dict[str, object],
    gate_g_manifest: dict[str, object],
) -> None:
    validate_gate_chain(gate_a_manifest, gate_e_manifest, gate_f_manifest)
    if gate_g_manifest.get("status") != "PASS":
        raise ValueError("Gate G manifest is not PASS; scaffold candidate should not proceed.")


def make_transport_mdp(*, system_id: str, duration_ps: float, phase: str, seed: int, args: argparse.Namespace) -> str:
    nsteps = steps_from_ps(duration_ps)
    energy_stride_steps = steps_from_ps(args.energy_stride_ps)
    coord_stride_steps = steps_from_ps(args.coord_stride_ps)
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
        f"title                   = gate h scaffold candidate {phase} exact respa {system_id}\n"
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


def parse_topology_layout(top_path: Path) -> tuple[dict[str, int], list[tuple[str, int]]]:
    lines = top_path.read_text(encoding="utf-8").splitlines()
    current_section = None
    current_moleculetype = None
    atom_counts: dict[str, int] = {}
    molecules: list[tuple[str, int]] = []
    pending_moleculetype: str | None = None
    for raw_line in lines:
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line.strip("[]").strip().lower()
            if current_section != "atoms":
                current_moleculetype = None
            continue
        if current_section == "moleculetype":
            fields = line.split()
            if fields:
                pending_moleculetype = fields[0]
                atom_counts.setdefault(pending_moleculetype, 0)
                current_moleculetype = pending_moleculetype
                current_section = None
            continue
        if raw_line.split(";", 1)[0].strip().lower() == "[ atoms ]":
            continue
        if current_section == "atoms":
            if current_moleculetype is None and pending_moleculetype is not None:
                current_moleculetype = pending_moleculetype
            if current_moleculetype is not None:
                atom_counts[current_moleculetype] += 1
            continue
        if current_section == "molecules":
            fields = line.split()
            if len(fields) >= 2:
                molecules.append((fields[0], int(fields[1])))
    if not atom_counts or not molecules:
        raise ValueError(f"Failed to parse molecule layout from {top_path}")
    return atom_counts, molecules


def build_species_groups(
    *,
    top_path: Path,
    species_defs: tuple[dict[str, object], ...],
) -> tuple[dict[str, list[int]], dict[str, int]]:
    atom_counts, molecules = parse_topology_layout(top_path)
    species_atoms: dict[str, list[int]] = {str(spec["name"]): [] for spec in species_defs}
    species_counts: dict[str, int] = {str(spec["name"]): 0 for spec in species_defs}
    species_lookup = {
        str(spec["name"]): set(str(name) for name in spec["molecule_types"]) for spec in species_defs
    }
    atom_index = 1
    for molecule_type, count in molecules:
        natoms = atom_counts[molecule_type]
        for _ in range(count):
            atom_span = list(range(atom_index, atom_index + natoms))
            for species_name, allowed_types in species_lookup.items():
                if molecule_type in allowed_types:
                    species_atoms[species_name].extend(atom_span)
                    species_counts[species_name] += 1
            atom_index += natoms
    return species_atoms, species_counts


def write_index_file(path: Path, group_name: str, atoms: list[int]) -> None:
    lines = [f"[ {group_name} ]"]
    for start in range(0, len(atoms), 15):
        lines.append(" ".join(str(atom) for atom in atoms[start : start + 15]))
    write_text(path, "\n".join(lines) + "\n")


def file_exists_and_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def analysis_complete(msdout_path: Path, diffmol_path: Path) -> bool:
    return file_exists_and_nonempty(msdout_path) and file_exists_and_nonempty(diffmol_path)


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
    beginfit_ps: float | None = None,
    endfit_ps: float | None = None,
    force: bool = False,
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
    if beginfit_ps is not None:
        argv.extend(["-beginfit", f"{beginfit_ps:.10g}"])
    if endfit_ps is not None:
        argv.extend(["-endfit", f"{endfit_ps:.10g}"])
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
    if not force and analysis_complete(msdout_path, diffmol_path):
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


def compute_half_interval_fit_error_cm2_s(xs: list[float], ys: list[float]) -> float:
    midpoint = len(xs) // 2
    if midpoint < 2 or len(xs) - midpoint < 2:
        return math.inf
    first_fit = linear_fit(xs[:midpoint], ys[:midpoint])
    second_fit = linear_fit(xs[midpoint:], ys[midpoint:])
    if not first_fit.get("valid", False) or not second_fit.get("valid", False):
        return math.inf
    first_diff = first_fit["slope"] / 6.0 * NM2_PS_TO_CM2_S
    second_diff = second_fit["slope"] / 6.0 * NM2_PS_TO_CM2_S
    return abs(first_diff - second_diff)


def select_fit_window(times_ps: list[float], msd_nm2: list[float]) -> dict[str, object]:
    if not times_ps or len(times_ps) != len(msd_nm2):
        return {"valid": False}
    max_tau = times_ps[-1]
    best: dict[str, object] | None = None
    for start_ratio, end_ratio in FIT_WINDOW_CANDIDATES:
        if end_ratio - start_ratio < MIN_FIT_WINDOW_SPAN_RATIO:
            continue
        fit_start_ps = start_ratio * max_tau
        fit_end_ps = end_ratio * max_tau
        fit_xs = [tau for tau in times_ps if fit_start_ps <= tau <= fit_end_ps]
        fit_ys = [value for tau, value in zip(times_ps, msd_nm2) if fit_start_ps <= tau <= fit_end_ps]
        if len(fit_xs) < MIN_FIT_POINTS:
            continue
        fit = linear_fit(fit_xs, fit_ys)
        if not fit.get("valid", False):
            continue
        diffusion_cm2_s = fit["slope"] / 6.0 * NM2_PS_TO_CM2_S
        fit_error_cm2_s = compute_half_interval_fit_error_cm2_s(fit_xs, fit_ys)
        relative_fit_error = math.inf if diffusion_cm2_s == 0.0 else abs(fit_error_cm2_s / diffusion_cm2_s)
        candidate = {
            "valid": True,
            "fit_start_ps": fit_start_ps,
            "fit_end_ps": fit_end_ps,
            "fit_points": len(fit_xs),
            "fit": fit,
            "diffusion_cm2_s": diffusion_cm2_s,
            "fit_error_cm2_s": fit_error_cm2_s,
            "relative_fit_error": relative_fit_error,
            "start_ratio": start_ratio,
            "end_ratio": end_ratio,
            "score": (fit["r_squared"], -relative_fit_error, end_ratio - start_ratio),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    if best is None:
        return {"valid": False}
    return best


def analyze_species_replica(
    *,
    gmx: Path,
    species_def: dict[str, object],
    species_atoms: list[int],
    prod_deffnm: Path,
    replica_root: Path,
    logs_dir: Path,
    commands: list[dict[str, object]],
    env: dict[str, str],
    label_prefix: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    species_name = str(species_def["name"])
    ndx_path = replica_root / f"{species_name}.ndx"
    write_index_file(ndx_path, species_name.upper(), species_atoms)
    msdout_path = replica_root / f"{species_name}_msdout.xvg"
    diffmol_path = replica_root / f"{species_name}_diff_mol.xvg"
    run_or_resume_gmx_msd(
        gmx=gmx,
        traj_path=prod_deffnm.with_suffix(".xtc"),
        tpr_path=prod_deffnm.with_suffix(".tpr"),
        ndx_path=ndx_path,
        msdout_path=msdout_path,
        diffmol_path=diffmol_path,
        logs_dir=logs_dir,
        commands=commands,
        label=f"{label_prefix}_{species_name}_msd_prefit",
        env=env,
        args=args,
    )
    _, times_ps, msd_nm2 = parse_msd_curve(msdout_path)
    fit_window = select_fit_window(times_ps, msd_nm2)
    if not fit_window.get("valid", False):
        raise RuntimeError(f"{species_name}: unable to determine a valid MSD fit window")
    run_or_resume_gmx_msd(
        gmx=gmx,
        traj_path=prod_deffnm.with_suffix(".xtc"),
        tpr_path=prod_deffnm.with_suffix(".tpr"),
        ndx_path=ndx_path,
        msdout_path=msdout_path,
        diffmol_path=diffmol_path,
        logs_dir=logs_dir,
        commands=commands,
        label=f"{label_prefix}_{species_name}_msd_refit",
        env=env,
        args=args,
        beginfit_ps=float(fit_window["fit_start_ps"]),
        endfit_ps=float(fit_window["fit_end_ps"]),
        force=True,
    )
    legend, times_ps, msd_nm2 = parse_msd_curve(msdout_path)
    diff_values = parse_diff_mol(diffmol_path)
    fit_start_ps = float(fit_window["fit_start_ps"])
    fit_end_ps = float(fit_window["fit_end_ps"])
    fit_xs = [tau for tau in times_ps if fit_start_ps <= tau <= fit_end_ps]
    fit_ys = [value for tau, value in zip(times_ps, msd_nm2) if fit_start_ps <= tau <= fit_end_ps]
    fit = linear_fit(fit_xs, fit_ys)
    curve_tsv = replica_root / f"{species_name}_msd_curve.tsv"
    save_curve_tsv(curve_tsv, times_ps, msd_nm2)

    molecule_mean = mean(diff_values)
    molecule_sem = sem(diff_values)
    molecule_std = sample_std(diff_values)
    relative_fit_error = math.inf if legend["diffusion_cm2_s"] == 0.0 else abs(legend["fit_error_cm2_s"] / legend["diffusion_cm2_s"])
    relative_molecule_sem = math.inf if molecule_mean == 0.0 else abs(molecule_sem / molecule_mean)
    convergence_flags: list[str] = []
    if not fit.get("valid", False):
        convergence_flags.append(f"{species_name}: MSD fit window is invalid.")
    elif fit["r_squared"] < R2_THRESHOLD:
        convergence_flags.append(f"{species_name}: MSD linearity is weak (R^2={fit['r_squared']:.6f}).")
    if relative_fit_error >= RELATIVE_UNCERTAINTY_THRESHOLD:
        convergence_flags.append(
            f"{species_name}: MSD fit uncertainty is large ({relative_fit_error:.3f} >= {RELATIVE_UNCERTAINTY_THRESHOLD:.2f})."
        )
    if relative_molecule_sem >= RELATIVE_UNCERTAINTY_THRESHOLD:
        convergence_flags.append(
            f"{species_name}: per-molecule diffusivity spread is large ({relative_molecule_sem:.3f} >= {RELATIVE_UNCERTAINTY_THRESHOLD:.2f})."
        )
    return {
        "metric": str(species_def["metric"]),
        "charge": int(species_def["charge"]),
        "primary": bool(species_def["primary"]),
        "molecule_types": [str(name) for name in species_def["molecule_types"]],
        "molecule_count": len(diff_values),
        "diffusion_cm2_s": legend["diffusion_cm2_s"],
        "fit_error_cm2_s": legend["fit_error_cm2_s"],
        "fit_r_squared": fit.get("r_squared"),
        "fit_start_ps": fit_start_ps,
        "fit_end_ps": fit_end_ps,
        "fit_points": len(fit_xs),
        "molecule_mean_cm2_s": molecule_mean,
        "molecule_std_cm2_s": molecule_std,
        "molecule_sem_cm2_s": molecule_sem,
        "molecule_values_cm2_s": diff_values,
        "relative_fit_error": relative_fit_error,
        "relative_molecule_sem": relative_molecule_sem,
        "msdout_xvg": str(msdout_path),
        "diff_mol_xvg": str(diffmol_path),
        "msd_curve_tsv": str(curve_tsv),
        "transport_index": str(ndx_path),
        "convergence_flags": convergence_flags,
    }


def compute_conductivity_cne_s_cm(*, temperature_k: float, volume_nm3: float, species_payloads: dict[str, dict[str, object]], species_defs: tuple[dict[str, object], ...]) -> float | None:
    if temperature_k <= 0.0 or volume_nm3 <= 0.0:
        return None
    sum_term = 0.0
    for species_def in species_defs:
        charge = float(species_def["charge"])
        if charge == 0.0:
            continue
        metric_name = str(species_def["metric"])
        payload = species_payloads.get(metric_name)
        if payload is None:
            continue
        diffusion_m2_s = float(payload["value"]) * 1.0e-4
        count = int(payload["count"])
        sum_term += count * charge * charge * diffusion_m2_s
    sigma_s_m = (E_CHARGE_C * E_CHARGE_C) * sum_term / (volume_nm3 * M3_PER_NM3 * K_BOLTZMANN_J_K * temperature_k)
    return sigma_s_m / 100.0


def analyze_replica(
    *,
    gmx: Path,
    preset: dict[str, object],
    top_path: Path,
    prod_deffnm: Path,
    replica_root: Path,
    logs_dir: Path,
    commands: list[dict[str, object]],
    env: dict[str, str],
    label_prefix: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    species_defs = tuple(preset["species"])
    species_groups, species_counts = build_species_groups(top_path=top_path, species_defs=species_defs)
    species_results: dict[str, dict[str, object]] = {}
    convergence_flags: list[str] = []
    for species_def in species_defs:
        species_name = str(species_def["name"])
        payload = analyze_species_replica(
            gmx=gmx,
            species_def=species_def,
            species_atoms=species_groups[species_name],
            prod_deffnm=prod_deffnm,
            replica_root=replica_root,
            logs_dir=logs_dir,
            commands=commands,
            env=env,
            label_prefix=label_prefix,
            args=args,
        )
        payload["count"] = species_counts[species_name]
        species_results[str(species_def["metric"])] = payload
        convergence_flags.extend(payload["convergence_flags"])

    energy_xvg = replica_root / "production_observables.xvg"
    energy_series = extract_energy_series(gmx, prod_deffnm.with_suffix(".edr"), energy_xvg, ENERGY_TERMS)
    temp_series = [float(value) for value in energy_series.get("Temperature", [])]
    volume_series = [float(value) for value in energy_series.get("Volume", [])]
    if not volume_series:
        box_x = [float(value) for value in energy_series.get("Box-X", [])]
        box_y = [float(value) for value in energy_series.get("Box-Y", [])]
        box_z = [float(value) for value in energy_series.get("Box-Z", [])]
        count = min(len(box_x), len(box_y), len(box_z))
        volume_series = [box_x[index] * box_y[index] * box_z[index] for index in range(count)]
    if not volume_series:
        box = parse_box_from_gro(prod_deffnm.with_suffix(".gro"))
        volume_series = [box[0] * box[1] * box[2]]
    if not temp_series:
        temp_series = [DEFAULT_TEMP_K]
    avg_temp = mean(temp_series)
    avg_volume = mean(volume_series)

    if "conductivity_cne_s_cm" in tuple(preset["derived_metrics"]):
        conductivity = compute_conductivity_cne_s_cm(
            temperature_k=avg_temp,
            volume_nm3=avg_volume,
            species_payloads={k: {"value": v["diffusion_cm2_s"], "count": v["count"]} for k, v in species_results.items()},
            species_defs=species_defs,
        )
        cat = species_results["cation_diffusivity_cm2_s"]["diffusion_cm2_s"]
        an = species_results["anion_diffusivity_cm2_s"]["diffusion_cm2_s"]
        denom = cat + an
        species_results["conductivity_cne_s_cm"] = {
            "metric": "conductivity_cne_s_cm",
            "available": conductivity is not None,
            "value": conductivity,
            "count": species_results["cation_diffusivity_cm2_s"]["count"] + species_results["anion_diffusivity_cm2_s"]["count"],
            "block_sem": 0.0,
        }
        species_results["transference_ne"] = {
            "metric": "transference_ne",
            "available": denom != 0.0,
            "value": (cat / denom) if denom != 0.0 else math.nan,
            "count": 1,
            "block_sem": 0.0,
        }

    return {
        "average_temperature_k": avg_temp,
        "average_volume_nm3": avg_volume,
        "energy_xvg": str(energy_xvg),
        "metrics": species_results,
        "convergence_flags": convergence_flags,
        "species_groups": {name: {"atom_count": len(atoms), "molecule_count": species_counts[name]} for name, atoms in species_groups.items()},
    }


def build_manifest_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Gate H Scaffold Transport Candidate",
        "",
        f"- Status: `{result['status']}`",
        f"- Preset: `{result['preset']}`",
        f"- Scope: {result['scope']}",
        f"- Replica count per layout: `{result['run_settings']['replicas']}`",
        f"- Equilibration / production: `{result['run_settings']['equil_ps']} ps / {result['run_settings']['prod_ps']} ps`",
        "",
        "## Observables",
    ]
    for metric_name, comparison in result["observable_comparisons"].items():
        if not comparison.get("available"):
            lines.append(f"- `{metric_name}`: missing")
            continue
        lines.append(
            f"- `{metric_name}`: `{comparison['classification']}` / passes=`{comparison['passes']}` / "
            f"CPU=`{comparison['cpu_mean']:.10g}` / GPU=`{comparison['gpu_mean']:.10g}` / "
            f"unc=`{comparison['combined_uncertainty']:.10g}`"
        )
    if result["failure_reasons"]:
        lines.extend(["", "## Failure Reasons"])
        lines.extend(f"- {reason}" for reason in result["failure_reasons"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    validate_args(args)

    gate_a_manifest = load_json(Path(args.gate_a_manifest))
    gate_e_manifest = load_json(Path(args.gate_e_manifest))
    gate_f_manifest = load_json(Path(args.gate_f_manifest))
    gate_g_manifest = load_json(Path(args.gate_g_manifest))
    validate_gate_prerequisites(gate_a_manifest, gate_e_manifest, gate_f_manifest, gate_g_manifest)

    preset = PRESETS[args.preset]
    scaffold_manifest_path = Path(args.scaffold_manifest) if args.scaffold_manifest is not None else Path(str(preset["manifest"]))
    scaffold_manifest = load_json(scaffold_manifest_path)
    if str(scaffold_manifest["derived_system"]) != str(preset["system_id"]):
        raise ValueError("Preset and scaffold manifest disagree on system_id.")

    gmx = Path(args.gmx).resolve()
    maybe_build(args, Path(args.build_dir).resolve() if args.build_dir is not None else None)
    version_text = capture_output([str(gmx), "--version"], cwd=REPO_ROOT, env=os.environ.copy())

    out_root = Path(args.out).resolve() if args.out is not None else Path(str(preset["out"])).resolve()
    if out_root.exists() and not args.resume:
        shutil.rmtree(out_root)
    inputs_dir = out_root / "inputs"
    logs_dir = out_root / "logs"
    summaries_dir = out_root / "summaries"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    env = base_env(args)
    top_path = Path(str(scaffold_manifest["artifacts"]["topology"]))
    gro_path = Path(str(scaffold_manifest["artifacts"]["gro"]))

    commands: list[dict[str, object]] = []
    per_layout_replicas: dict[str, list[dict[str, object]]] = {"cpu": [], "gpu": []}
    convergence_issues: list[str] = []

    for replica_index in range(1, args.replicas + 1):
        seed = 74101 + replica_index - 1
        replica_inputs = inputs_dir / f"replica_{replica_index:02d}"
        replica_inputs.mkdir(parents=True, exist_ok=True)
        equil_mdp = replica_inputs / "equil.mdp"
        prod_mdp = replica_inputs / "prod.mdp"
        write_text(
            equil_mdp,
            make_transport_mdp(system_id=str(preset["system_id"]), duration_ps=args.equil_ps, phase="equil", seed=seed, args=args),
        )
        write_text(
            prod_mdp,
            make_transport_mdp(system_id=str(preset["system_id"]), duration_ps=args.prod_ps, phase="prod", seed=seed, args=args),
        )

        for layout in ("cpu", "gpu"):
            run_root = out_root / layout / f"replica_{replica_index:02d}"
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
                tpr_path=prod_deffnm.with_suffix(".tpr"),
                mdout_path=run_root / "prod.mdout.mdp",
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
                preset=preset,
                top_path=top_path,
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
            (run_root / "replica_summary.json").write_text(json.dumps(replica_summary, indent=2, sort_keys=True), encoding="utf-8")
            per_layout_replicas[layout].append(replica_summary)
            convergence_issues.extend(f"{layout} replica {replica_index:02d}: {issue}" for issue in analysis["convergence_flags"])

    metric_names = [str(spec["metric"]) for spec in preset["species"]] + list(preset["derived_metrics"])
    per_layout_aggregates: dict[str, dict[str, object]] = {"cpu": {}, "gpu": {}}
    for layout in ("cpu", "gpu"):
        for metric_name in metric_names:
            scalar_replicas = []
            for replica in per_layout_replicas[layout]:
                metric = replica["analysis"]["metrics"].get(metric_name, {})
                if metric_name in preset["derived_metrics"]:
                    if not metric.get("available"):
                        continue
                    scalar_replicas.append(
                        {
                            "replica_index": replica["replica_index"],
                            "value": metric["value"],
                            "block_sem": float(metric.get("block_sem", 0.0)),
                        }
                    )
                    continue
                if not metric:
                    continue
                scalar_replicas.append(
                    {
                        "replica_index": replica["replica_index"],
                        "value": metric["diffusion_cm2_s"],
                        "block_sem": max(float(metric["fit_error_cm2_s"]), float(metric["molecule_sem_cm2_s"])),
                        "r_squared": metric.get("fit_r_squared"),
                    }
                )
            per_layout_aggregates[layout][metric_name] = summarize_scalar_replicas(scalar_replicas)

    observable_comparisons = {
        metric_name: compare_scalar_aggregates(per_layout_aggregates["cpu"][metric_name], per_layout_aggregates["gpu"][metric_name])
        for metric_name in metric_names
    }
    transport_table = "observable\tcpu_mean\tgpu_mean\tmean_diff\tcombined_uncertainty\tclassification\tpasses\n"
    for metric_name, comparison in observable_comparisons.items():
        if not comparison.get("available"):
            transport_table += f"{metric_name}\tNA\tNA\tNA\tNA\tmissing\tFalse\n"
            continue
        transport_table += (
            f"{metric_name}\t{comparison['cpu_mean']:.10g}\t{comparison['gpu_mean']:.10g}\t"
            f"{comparison['mean_diff']:.10g}\t{comparison['combined_uncertainty']:.10g}\t"
            f"{comparison['classification']}\t{comparison['passes']}\n"
        )
    write_text(summaries_dir / "transport_comparison.tsv", transport_table)

    failure_reasons: list[str] = []
    first_failing = None
    primary_metrics = [str(spec["metric"]) for spec in preset["species"] if bool(spec["primary"])]
    for metric_name in primary_metrics:
        comparison = observable_comparisons.get(metric_name, {})
        if not comparison.get("available"):
            failure_reasons.append(f"{metric_name} comparison is unavailable.")
            first_failing = first_failing or metric_name
        elif not comparison.get("passes"):
            failure_reasons.append(f"{metric_name} exceeds the current combined uncertainty budget ({comparison['classification']}).")
            first_failing = first_failing or metric_name
    if convergence_issues:
        failure_reasons.extend(convergence_issues)
        first_failing = first_failing or "finite_sampling"

    status = "PASS" if first_failing is None else "FAIL"
    result = {
        "schema_version": 1,
        "preset": args.preset,
        "scope": "Large-scaffold Gate H transport candidate using molecule-wise MSD per species; candidate evidence only.",
        "status": status,
        "first_failing_observable": first_failing,
        "failure_reasons": failure_reasons,
        "gmx_binary": str(gmx),
        "gpu_support": parse_gpu_support(version_text),
        "precision_mode": parse_precision_mode(version_text),
        "mechanical_prerequisites": {
            "gate_a_status": gate_a_manifest.get("status"),
            "gate_e_status": gate_e_manifest.get("status"),
            "gate_f_status": gate_f_manifest.get("status"),
            "gate_g_status": gate_g_manifest.get("status"),
        },
        "scaffold_manifest": str(scaffold_manifest_path.resolve()),
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
        "species": preset["species"],
        "per_layout_aggregates": per_layout_aggregates,
        "observable_comparisons": observable_comparisons,
        "per_layout_replicas": per_layout_replicas,
        "recommendation": (
            "Candidate transport evidence is internally consistent enough to justify longer production-scale transport runs on this scaffold."
            if status == "PASS"
            else "Do not treat this scaffold as transport-ready yet; resolve the first failing observable or finite-sampling issue first."
        ),
    }
    (summaries_dir / "candidate_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_text(summaries_dir / "candidate_result.md", build_manifest_markdown(result))
    script_lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"# preset: {args.preset}", ""]
    script_lines.extend(["# recorded commands are embedded in the logs for this candidate runner"])
    write_text(out_root / "run_commands.sh", "\n".join(script_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
