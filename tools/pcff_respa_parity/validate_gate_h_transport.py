from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from freeze_gate_a_oracle import (
    DEFAULT_GMX,
    FIXTURE_ROOT,
    REPO_ROOT,
    base_env,
    capture_output,
    env_delta,
    write_text,
)
from validate_gate_b_nb_gpu import load_json, parse_gpu_support, parse_precision_mode, run_command_allow_failure
from validate_gate_c_nb_bonded_gpu import DEFAULT_GATE_A_MANIFEST, maybe_build
from validate_gate_e_update_gpu import parse_layout_report
from validate_gate_g_long_ensemble import (
    DEFAULT_GATE_E_MANIFEST,
    DEFAULT_GATE_F_MANIFEST,
    extract_energy_series,
    exact_respa_common_mdp,
    mean,
    parse_xvg_series,
    run_grompp,
    sample_std,
    sem,
    validate_gate_chain,
)


DEFAULT_GATE_G_MANIFEST = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_g_long_ensemble_validation"
    / "gate_g_manifest.json"
)
DEFAULT_TP0_METADATA = REPO_ROOT / "tests" / "reference_results" / "transport_protocol_metadata" / "tp0.json"
DEFAULT_TP1_STATUS = REPO_ROOT / "tests" / "reference_results" / "transport_protocol_metadata" / "tp1_status.json"
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_h_transport_validation"
DEFAULT_LARGE_NEUTRAL_SCAFFOLD = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_h_fixture_scaffold"
    / "gate_h_dense_oligomer_2x2x2"
    / "fixture_manifest.json"
)
DEFAULT_LARGE_CHARGED_SCAFFOLD = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_h_fixture_scaffold"
    / "gate_h_dense_salt_polymer_2x2x2"
    / "fixture_manifest.json"
)

DT_PS = 0.0005
EXACT_RESPA_FACTOR = 4
DEFAULT_EQ_PS = 200.0
DEFAULT_PROD_PS = 500.0
DEFAULT_REPLICAS = 3
DEFAULT_TEMP_K = 300.0
DEFAULT_TAU_T_PS = 0.5
DEFAULT_COORD_STRIDE_PS = 1.0
DEFAULT_ENERGY_STRIDE_PS = 1.0
FIT_WINDOW = (0.2, 0.8)
MOL_FIT_WINDOW_CANDIDATES = (
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
MIN_MOL_FIT_WINDOW_SPAN_RATIO = 0.30
MIN_MOL_FIT_POINTS = 40
BLOCK_COUNT = 5
TP0_NEUTRAL_MIN_PROD_PS = 10_000.0
TP0_CHARGED_MIN_PROD_PS = 20_000.0
TP0_MIN_ATOMS = 1000
TP0_MIN_BOX_NM = 3.0
E_CHARGE_C = 1.602176634e-19
K_BOLTZMANN_J_K = 1.380649e-23
NM2_PS_TO_CM2_S = 1.0e-2
NM2_PS_TO_M2_S = 1.0e-6
M3_PER_NM3 = 1.0e-27

ENERGY_TERMS = ("Temperature", "Pressure", "Potential", "Volume", "Density", "Box-X", "Box-Y", "Box-Z")
GROUP_LEGEND_RE = re.compile(r'^@ s(\d+) legend "(.+) ([XYZ])"$')
MSD_LEGEND_RE = re.compile(
    r'@ s0 legend "D\[\s*(?P<label>[^\]]+)\] = (?P<diff>[-+0-9.eE]+) \(\+/- (?P<err>[-+0-9.eE]+)\) \(1e-5 cm\^2/s\)"'
)
DEFAULT_TRESTART_PS = 2.0
DEFAULT_MAXTAU_PS = 50.0
R2_THRESHOLD = 0.98
RELATIVE_UNCERTAINTY_THRESHOLD = 0.50

SYSTEM_CONFIGS_SMALL: dict[str, dict[str, object]] = {
    "small_oligomer": {
        "system_class": "neutral_dense",
        "ensemble": "nvt",
        "expected_natoms": 6,
        "group_order": ("SYSTEM", "OLI"),
        "group_atoms": {
            "SYSTEM": tuple(range(1, 7)),
            "OLI": tuple(range(1, 7)),
        },
        "diffusion_observables": (
            {
                "metric": "oligomer_diffusivity_cm2_s",
                "group": "OLI",
                "species": "oligomer",
                "count": 1,
                "charge": 0,
                "primary": True,
            },
        ),
        "derived_observables": (),
        "primary_observables": ("oligomer_diffusivity_cm2_s",),
    },
    "small_salt_polymer_box": {
        "system_class": "charged_salt_polymer",
        "ensemble": "nvt",
        "expected_natoms": 10,
        "group_order": ("SYSTEM", "POL", "CAT", "ANI"),
        "group_atoms": {
            "SYSTEM": tuple(range(1, 11)),
            "POL": tuple(range(1, 9)),
            "CAT": (9,),
            "ANI": (10,),
        },
        "diffusion_observables": (
            {
                "metric": "polymer_diffusivity_cm2_s",
                "group": "POL",
                "species": "polymer",
                "count": 1,
                "charge": 0,
                "primary": False,
            },
            {
                "metric": "cation_diffusivity_cm2_s",
                "group": "CAT",
                "species": "cation",
                "count": 1,
                "charge": 1,
                "primary": True,
            },
            {
                "metric": "anion_diffusivity_cm2_s",
                "group": "ANI",
                "species": "anion",
                "count": 1,
                "charge": -1,
                "primary": True,
            },
        ),
        "derived_observables": (
            "conductivity_cne_s_cm",
            "transference_ne",
        ),
        "primary_observables": (
            "cation_diffusivity_cm2_s",
            "anion_diffusivity_cm2_s",
            "conductivity_cne_s_cm",
            "transference_ne",
        ),
    },
}

SYSTEM_CONFIGS_LARGE_TEMPLATE: dict[str, dict[str, object]] = {
    "gate_h_dense_oligomer_2x2x2": {
        "system_class": "neutral_dense",
        "ensemble": "nvt",
        "analysis_mode": "molecule_wise",
        "expected_natoms": 3072,
        "diffusion_observables": (
            {
                "metric": "oligomer_diffusivity_cm2_s",
                "species": "oligomer",
                "molecule_types": ("MOL1",),
                "count": 512,
                "charge": 0,
                "primary": True,
            },
        ),
        "derived_observables": (),
        "primary_observables": ("oligomer_diffusivity_cm2_s",),
    },
    "gate_h_dense_salt_polymer_2x2x2": {
        "system_class": "charged_salt_polymer",
        "ensemble": "nvt",
        "analysis_mode": "molecule_wise",
        "expected_natoms": 2160,
        "diffusion_observables": (
            {
                "metric": "polymer_diffusivity_cm2_s",
                "species": "polymer",
                "molecule_types": ("POL", "POL10", "POL28"),
                "count": 216,
                "charge": 0,
                "primary": False,
            },
            {
                "metric": "cation_diffusivity_cm2_s",
                "species": "cation",
                "molecule_types": ("CAT",),
                "count": 216,
                "charge": 1,
                "primary": True,
            },
            {
                "metric": "anion_diffusivity_cm2_s",
                "species": "anion",
                "molecule_types": ("ANI",),
                "count": 216,
                "charge": -1,
                "primary": True,
            },
        ),
        "derived_observables": (
            "conductivity_cne_s_cm",
            "transference_ne",
        ),
        "primary_observables": (
            "cation_diffusivity_cm2_s",
            "anion_diffusivity_cm2_s",
            "conductivity_cne_s_cm",
            "transference_ne",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Gate H transport-facing observables for standalone exact r-RESPA on the full GPU path."
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument("--gate-a-manifest", default=str(DEFAULT_GATE_A_MANIFEST), help="Path to the Gate A manifest.")
    parser.add_argument("--gate-e-manifest", default=str(DEFAULT_GATE_E_MANIFEST), help="Path to the Gate E manifest.")
    parser.add_argument("--gate-f-manifest", default=str(DEFAULT_GATE_F_MANIFEST), help="Path to the Gate F manifest.")
    parser.add_argument("--gate-g-manifest", default=str(DEFAULT_GATE_G_MANIFEST), help="Path to the Gate G manifest.")
    parser.add_argument("--tp0-metadata", default=str(DEFAULT_TP0_METADATA), help="Path to the frozen TP0 metadata JSON.")
    parser.add_argument("--tp1-status", default=str(DEFAULT_TP1_STATUS), help="Path to the TP1 status JSON.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument(
        "--fixture-mode",
        choices=("small", "large"),
        default="small",
        help="Use the legacy small fixture set or the large scaffold fixture set.",
    )
    parser.add_argument(
        "--neutral-scaffold-manifest",
        default=str(DEFAULT_LARGE_NEUTRAL_SCAFFOLD),
        help="Large neutral scaffold manifest for fixture-mode=large.",
    )
    parser.add_argument(
        "--charged-scaffold-manifest",
        default=str(DEFAULT_LARGE_CHARGED_SCAFFOLD),
        help="Large charged scaffold manifest for fixture-mode=large.",
    )
    parser.add_argument("--build-target", default="gmx", help="CMake target to build.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step.")
    parser.add_argument("--build-dir", default=None, help="Optional explicit build directory.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks for mdrun.")
    parser.add_argument("--ntomp", type=int, default=1, help="OpenMP threads for mdrun.")
    parser.add_argument("--npme", type=int, default=None, help="Optional explicit -npme value; omitted in Gate H.")
    parser.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS, help="Replica count per layout.")
    parser.add_argument("--equil-ps", type=float, default=DEFAULT_EQ_PS, help="Equilibration duration in ps.")
    parser.add_argument("--prod-ps", type=float, default=DEFAULT_PROD_PS, help="Production duration in ps.")
    parser.add_argument(
        "--coord-stride-ps",
        type=float,
        default=DEFAULT_COORD_STRIDE_PS,
        help="Coordinate/trajectory output stride in ps.",
    )
    parser.add_argument(
        "--energy-stride-ps",
        type=float,
        default=DEFAULT_ENERGY_STRIDE_PS,
        help="Energy/log output stride in ps.",
    )
    parser.add_argument("--temperature-k", type=float, default=DEFAULT_TEMP_K, help="Target temperature.")
    parser.add_argument("--tau-t-ps", type=float, default=DEFAULT_TAU_T_PS, help="Thermostat coupling time.")
    parser.add_argument("--trestart-ps", type=float, default=DEFAULT_TRESTART_PS, help="Restart window for gmx msd -mol in large mode.")
    parser.add_argument("--maxtau-ps", type=float, default=DEFAULT_MAXTAU_PS, help="Maximum lag time for gmx msd -mol in large mode.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed artifacts under --out instead of deleting the output root.")
    return parser.parse_args()


def steps_from_ps(duration_ps: float) -> int:
    return int(round(duration_ps / DT_PS))


def validate_args(args: argparse.Namespace) -> None:
    if args.ntmpi != 1:
        raise ValueError("Gate H is restricted to single-rank runs (ntmpi=1).")
    if args.replicas < 2:
        raise ValueError("Gate H requires replicated runs; use at least 2 replicas.")
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
        raise ValueError("Gate G manifest is not PASS; Gate H cannot proceed on an unvalidated long-run ensemble path.")


def fixture_dir(system_id: str) -> Path:
    return FIXTURE_ROOT / system_id


def build_system_configs(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    if args.fixture_mode == "small":
        return SYSTEM_CONFIGS_SMALL
    large_configs = {key: dict(value) for key, value in SYSTEM_CONFIGS_LARGE_TEMPLATE.items()}
    large_configs["gate_h_dense_oligomer_2x2x2"]["scaffold_manifest"] = str(Path(args.neutral_scaffold_manifest).resolve())
    large_configs["gate_h_dense_salt_polymer_2x2x2"]["scaffold_manifest"] = str(Path(args.charged_scaffold_manifest).resolve())
    return large_configs


def parse_topology_layout(top_path: Path) -> tuple[dict[str, int], list[tuple[str, int]]]:
    lines = top_path.read_text(encoding="utf-8").splitlines()
    current_section = None
    current_moleculetype = None
    pending_moleculetype: str | None = None
    atom_counts: dict[str, int] = {}
    molecules: list[tuple[str, int]] = []
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
    species_atoms: dict[str, list[int]] = {str(spec["species"]): [] for spec in species_defs}
    species_counts: dict[str, int] = {str(spec["species"]): 0 for spec in species_defs}
    species_lookup = {
        str(spec["species"]): set(str(name) for name in spec["molecule_types"]) for spec in species_defs
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


def run_gmx_msd_mol(
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
    stdout_path = logs_dir / f"{label}.stdout"
    stderr_path = logs_dir / f"{label}.stderr"
    stdin_text = "0\n"
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
    if force:
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
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed; see {stderr_path}")


def file_exists_and_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def grompp_complete(tpr_path: Path, mdout_path: Path) -> bool:
    return file_exists_and_nonempty(tpr_path) and file_exists_and_nonempty(mdout_path)


def choose_traj_path(deffnm: Path) -> Path:
    xtc = deffnm.with_suffix(".xtc")
    if file_exists_and_nonempty(xtc):
        return xtc
    trr = deffnm.with_suffix(".trr")
    if file_exists_and_nonempty(trr):
        return trr
    return xtc


def mdrun_complete(deffnm: Path) -> bool:
    required = (
        deffnm.with_suffix(".edr"),
        deffnm.with_suffix(".gro"),
        deffnm.with_suffix(".cpt"),
        deffnm.with_suffix(".log"),
    )
    return all(file_exists_and_nonempty(path) for path in required) and file_exists_and_nonempty(choose_traj_path(deffnm))


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
            if match:
                legend = {
                    "label": match.group("label").strip(),
                    "diffusion_cm2_s": float(match.group("diff")) * 1.0e-5,
                    "fit_error_cm2_s": float(match.group("err")) * 1.0e-5,
                }
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            times_ps.append(float(parts[0]))
            msd_nm2.append(float(parts[1]))
    if legend is None:
        raise ValueError(f"Failed to parse MSD legend from {path}")
    if len(times_ps) < 2:
        raise ValueError(f"MSD curve in {path} is too short")
    return legend, times_ps, msd_nm2


def parse_diff_mol(path: Path) -> list[float]:
    values: list[float] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "@")):
            continue
        parts = line.split()
        values.append(float(parts[-1]) * 1.0e-5)
    if not values:
        raise ValueError(f"No molecule-wise diffusion values parsed from {path}")
    return values


def save_curve_tsv(path: Path, times_ps: list[float], msd_nm2: list[float]) -> None:
    rows = ["tau_ps\tmsd_nm2\n"]
    for tau_ps, value in zip(times_ps, msd_nm2):
        rows.append(f"{tau_ps:.10g}\t{value:.10g}\n")
    write_text(path, "".join(rows))


def make_transport_mdp(
    *,
    system_id: str,
    duration_ps: float,
    energy_stride_ps: float,
    coord_stride_ps: float,
    phase: str,
    seed: int,
    args: argparse.Namespace,
) -> str:
    nsteps = steps_from_ps(duration_ps)
    energy_stride_steps = steps_from_ps(energy_stride_ps)
    coord_stride_steps = steps_from_ps(coord_stride_ps)
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
    # Keep the already validated exact-r-RESPA NVT path instead of opening a new
    # barostat/integrator surface here.
    return (
        f"title                   = gate h transport {phase} exact respa {system_id}\n"
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


def record_command(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    env_overrides: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    stdin_text: str | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "cwd": str(cwd),
        "argv": command,
        "env_overrides": env_overrides,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "stdin_text": stdin_text,
    }


def write_recorded_commands_script(path: Path, records: list[dict[str, object]]) -> None:
    def shell_single_quote(payload: str) -> str:
        return "'" + payload.replace("'", "'\"'\"'") + "'"

    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for record in records:
        lines.append(f"# {record['name']}")
        env_prefix = "env"
        for key, value in record["env_overrides"].items():
            env_prefix += f" {shell_single_quote(f'{key}={value}')}"
        command = " ".join(shell_single_quote(str(arg)) for arg in record["argv"])
        if record.get("stdin_text") is None:
            lines.append(
                f"(cd {shell_single_quote(record['cwd'])} && {env_prefix} {command} > {shell_single_quote(record['stdout'])} 2> {shell_single_quote(record['stderr'])})"
            )
        else:
            lines.append(
                f"(cd {shell_single_quote(record['cwd'])} && printf %s {shell_single_quote(record['stdin_text'])} | {env_prefix} {command} > {shell_single_quote(record['stdout'])} 2> {shell_single_quote(record['stderr'])})"
            )
        lines.append("")
    write_text(path, "\n".join(lines))
    path.chmod(0o755)


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
        record_command(
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


def group_selection_input(system_id: str) -> str:
    group_count = len(SYSTEM_CONFIGS[system_id]["group_order"])
    return "".join(f"{index}\n" for index in range(group_count))


def run_gmx_traj_com(
    *,
    gmx: Path,
    traj_path: Path,
    tpr_path: Path,
    ndx_path: Path,
    ngroups: int,
    out_path: Path,
    logs_dir: Path,
    commands: list[dict[str, object]],
    label: str,
    env: dict[str, str],
    system_id: str,
) -> None:
    stdout_path = logs_dir / f"{label}.stdout"
    stderr_path = logs_dir / f"{label}.stderr"
    stdin_text = group_selection_input(system_id)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(
            [
                str(gmx),
                "traj",
                "-f",
                str(traj_path),
                "-s",
                str(tpr_path),
                "-n",
                str(ndx_path),
                "-com",
                "-nojump",
                "-fp",
                "-ng",
                str(ngroups),
                "-ox",
                str(out_path),
            ],
            cwd=REPO_ROOT,
            env=env,
            input=stdin_text,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    argv = [
        str(gmx),
        "traj",
        "-f",
        str(traj_path),
        "-s",
        str(tpr_path),
        "-n",
        str(ndx_path),
        "-com",
        "-nojump",
        "-fp",
        "-ng",
        str(ngroups),
        "-ox",
        str(out_path),
    ]
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
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed; see {stderr_path}")


def write_index_file(path: Path, group_atoms: dict[str, tuple[int, ...]], group_order: tuple[str, ...]) -> None:
    lines: list[str] = []
    for group_name in group_order:
        lines.append(f"[ {group_name} ]")
        atoms = group_atoms[group_name]
        for start in range(0, len(atoms), 15):
            chunk = atoms[start : start + 15]
            lines.append(" ".join(str(atom) for atom in chunk))
        lines.append("")
    write_text(path, "\n".join(lines).rstrip() + "\n")


def parse_gro_metadata(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    natoms = int(lines[1].strip())
    box_parts = [float(value) for value in lines[-1].split()]
    box = box_parts[:3]
    return {
        "natoms": natoms,
        "box_nm": box,
        "min_box_nm": min(box),
        "volume_nm3": box[0] * box[1] * box[2],
    }


def parse_group_coord_xvg(path: Path) -> tuple[list[float], dict[str, list[list[float]]]]:
    legends: dict[int, tuple[str, str]] = {}
    times: list[float] = []
    vectors: dict[str, list[list[float]]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("@"):
            match = GROUP_LEGEND_RE.match(line)
            if match:
                index = int(match.group(1))
                legends[index] = (match.group(2), match.group(3))
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        time_ps = float(parts[0])
        values = [float(value) for value in parts[1:]]
        times.append(time_ps)
        for column_index, value in enumerate(values):
            legend_index = column_index
            group_name, axis = legends[legend_index]
            axis_index = {"X": 0, "Y": 1, "Z": 2}[axis]
            group_vectors = vectors.setdefault(group_name, [])
            while len(group_vectors) < len(times):
                group_vectors.append([0.0, 0.0, 0.0])
            group_vectors[len(times) - 1][axis_index] = value
    return times, vectors


def subtract_system_drift(times: list[float], vectors: dict[str, list[list[float]]], group_name: str) -> list[list[float]]:
    if group_name not in vectors or "SYSTEM" not in vectors:
        return []
    coords: list[list[float]] = []
    for index in range(len(times)):
        target = vectors[group_name][index]
        system = vectors["SYSTEM"][index]
        coords.append([target[0] - system[0], target[1] - system[1], target[2] - system[2]])
    return coords


def split_blocks(values: list[object], blocks: int) -> list[list[object]]:
    if not values:
        return []
    nblocks = min(blocks, len(values))
    result: list[list[object]] = []
    for block_index in range(nblocks):
        start = round(block_index * len(values) / nblocks)
        end = round((block_index + 1) * len(values) / nblocks)
        block = values[start:end]
        if block:
            result.append(block)
    return result


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


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
        "ss_res": ss_res,
    }


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


def select_molecule_fit_window(times_ps: list[float], msd_nm2: list[float]) -> dict[str, object]:
    if not times_ps or len(times_ps) != len(msd_nm2):
        return {"valid": False}
    max_tau = times_ps[-1]
    best: dict[str, object] | None = None
    for start_ratio, end_ratio in MOL_FIT_WINDOW_CANDIDATES:
        if end_ratio - start_ratio < MIN_MOL_FIT_WINDOW_SPAN_RATIO:
            continue
        fit_start_ps = start_ratio * max_tau
        fit_end_ps = end_ratio * max_tau
        fit_xs = [tau for tau in times_ps if fit_start_ps <= tau <= fit_end_ps]
        fit_ys = [value for tau, value in zip(times_ps, msd_nm2) if fit_start_ps <= tau <= fit_end_ps]
        if len(fit_xs) < MIN_MOL_FIT_POINTS:
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
            "score": (fit["r_squared"], -relative_fit_error, end_ratio - start_ratio),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    if best is None:
        return {"valid": False}
    return best


def compute_msd_curve(times_ps: list[float], coords_nm: list[list[float]]) -> dict[str, object]:
    if len(times_ps) != len(coords_nm) or len(times_ps) < 3:
        return {"available": False, "reason": "not_enough_frames"}
    lags_ps: list[float] = []
    msd_nm2: list[float] = []
    nframes = len(times_ps)
    for lag in range(1, nframes):
        tau = times_ps[lag] - times_ps[0]
        accum = 0.0
        count = nframes - lag
        for origin in range(count):
            dx = coords_nm[origin + lag][0] - coords_nm[origin][0]
            dy = coords_nm[origin + lag][1] - coords_nm[origin][1]
            dz = coords_nm[origin + lag][2] - coords_nm[origin][2]
            accum += dx * dx + dy * dy + dz * dz
        lags_ps.append(tau)
        msd_nm2.append(accum / count)
    max_tau = lags_ps[-1]
    fit_start_ps = FIT_WINDOW[0] * max_tau
    fit_end_ps = FIT_WINDOW[1] * max_tau
    fit_xs = [tau for tau in lags_ps if fit_start_ps <= tau <= fit_end_ps]
    fit_ys = [value for tau, value in zip(lags_ps, msd_nm2) if fit_start_ps <= tau <= fit_end_ps]
    fit = linear_fit(fit_xs, fit_ys)
    if not fit.get("valid"):
        return {"available": False, "reason": "invalid_fit_window"}
    diffusion_nm2_ps = fit["slope"] / 6.0
    return {
        "available": True,
        "lags_ps": lags_ps,
        "msd_nm2": msd_nm2,
        "fit_start_ps": fit_start_ps,
        "fit_end_ps": fit_end_ps,
        "fit_points": len(fit_xs),
        "slope_nm2_ps": fit["slope"],
        "intercept_nm2": fit["intercept"],
        "diffusion_nm2_ps": diffusion_nm2_ps,
        "diffusion_cm2_s": diffusion_nm2_ps * NM2_PS_TO_CM2_S,
        "r_squared": fit["r_squared"],
    }


def compute_block_estimates(times_ps: list[float], coords_nm: list[list[float]]) -> dict[str, object]:
    frame_pairs = list(zip(times_ps, coords_nm))
    blocks = split_blocks(frame_pairs, BLOCK_COUNT)
    estimates = []
    for block in blocks:
        block_times = [entry[0] for entry in block]
        block_coords = [entry[1] for entry in block]
        curve = compute_msd_curve(block_times, block_coords)
        if curve.get("available"):
            estimates.append(float(curve["diffusion_cm2_s"]))
    if not estimates:
        return {"available": False}
    mean_estimate = mean(estimates)
    block_sem_value = sem(estimates)
    relative_uncertainty = math.inf if mean_estimate == 0.0 else abs(block_sem_value / mean_estimate)
    return {
        "available": True,
        "estimates_cm2_s": estimates,
        "mean_cm2_s": mean_estimate,
        "std_cm2_s": sample_std(estimates),
        "sem_cm2_s": block_sem_value,
        "relative_uncertainty": relative_uncertainty,
    }


def volume_nm3_from_box_terms(box_x: list[float], box_y: list[float], box_z: list[float]) -> list[float]:
    count = min(len(box_x), len(box_y), len(box_z))
    return [box_x[index] * box_y[index] * box_z[index] for index in range(count)]


def fallback_constant_series(value: float, count: int) -> list[float]:
    return [value for _ in range(count)]


def summarize_scalar_replicas(replicas: list[dict[str, object]]) -> dict[str, object]:
    if not replicas:
        return {"available": False}
    values = [float(replica["value"]) for replica in replicas]
    block_sems = [float(replica.get("block_sem", 0.0)) for replica in replicas]
    return {
        "available": True,
        "replica_count": len(replicas),
        "mean": mean(values),
        "std": sample_std(values),
        "sem": sem(values),
        "mean_block_sem": mean(block_sems),
        "replicas": replicas,
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


def compare_scalar_aggregates(cpu: dict[str, object], gpu: dict[str, object]) -> dict[str, object]:
    if not cpu.get("available") or not gpu.get("available"):
        return {"available": False}
    cpu_uncertainty = max(float(cpu["sem"]), float(cpu["mean_block_sem"]))
    gpu_uncertainty = max(float(gpu["sem"]), float(gpu["mean_block_sem"]))
    combined_uncertainty = math.sqrt(cpu_uncertainty * cpu_uncertainty + gpu_uncertainty * gpu_uncertainty)
    diff = float(gpu["mean"]) - float(cpu["mean"])
    replica_count = min(len(cpu["replicas"]), len(gpu["replicas"]))
    replica_diffs = [
        float(gpu["replicas"][index]["value"]) - float(cpu["replicas"][index]["value"]) for index in range(replica_count)
    ]
    classification = classify_difference(diff, combined_uncertainty, replica_diffs)
    passes = (combined_uncertainty == 0.0 and diff == 0.0) or (
        combined_uncertainty > 0.0 and abs(diff) <= 3.0 * combined_uncertainty
    )
    return {
        "available": True,
        "cpu_mean": cpu["mean"],
        "gpu_mean": gpu["mean"],
        "mean_diff": diff,
        "cpu_uncertainty": cpu_uncertainty,
        "gpu_uncertainty": gpu_uncertainty,
        "combined_uncertainty": combined_uncertainty,
        "classification": classification,
        "replica_diffs": replica_diffs,
        "passes": passes,
    }


def relative_error(observed: float, reference: float) -> float | None:
    if reference == 0.0:
        return None
    return abs(observed - reference) / abs(reference)


def compute_conductivity_cne_s_cm(
    *,
    temperature_k: float,
    volume_nm3: float,
    terms: tuple[dict[str, object], ...],
    replica_metrics: dict[str, dict[str, object]],
) -> float | None:
    if temperature_k <= 0.0 or volume_nm3 <= 0.0:
        return None
    sum_term = 0.0
    for term in terms:
        metric_name = str(term["metric"])
        if metric_name not in replica_metrics:
            continue
        diffusion_cm2_s = float(replica_metrics[metric_name]["value"])
        diffusion_m2_s = diffusion_cm2_s * 1.0e-4
        charge = float(term["charge"])
        count = int(term["count"])
        if charge == 0.0 or count == 0:
            continue
        sum_term += count * charge * charge * diffusion_m2_s
    sigma_s_m = (E_CHARGE_C * E_CHARGE_C) * sum_term / (volume_nm3 * M3_PER_NM3 * K_BOLTZMANN_J_K * temperature_k)
    return sigma_s_m / 100.0


def fit_protocol_expectation(system_class: str, prod_ps: float) -> dict[str, object]:
    min_prod_ps = TP0_NEUTRAL_MIN_PROD_PS if system_class == "neutral_dense" else TP0_CHARGED_MIN_PROD_PS
    return {
        "minimum_prod_ps": min_prod_ps,
        "meets_minimum": prod_ps >= min_prod_ps,
    }


def sanitize_scalar_aggregate(payload: dict[str, object]) -> dict[str, object]:
    if not payload.get("available"):
        return {"available": False}
    return {
        "available": True,
        "replica_count": payload["replica_count"],
        "mean": payload["mean"],
        "std": payload["std"],
        "sem": payload["sem"],
        "mean_block_sem": payload["mean_block_sem"],
        "replicas": payload["replicas"],
    }


def save_scalar_table(path: Path, comparisons: dict[str, dict[str, object]]) -> None:
    header = "observable\tcpu_mean\tgpu_mean\tmean_diff\tcombined_uncertainty\tclassification\tpasses\n"
    rows = [header]
    for name, comparison in comparisons.items():
        if not comparison.get("available"):
            rows.append(f"{name}\tNA\tNA\tNA\tNA\tmissing\tFalse\n")
            continue
        rows.append(
            f"{name}\t{comparison['cpu_mean']:.10g}\t{comparison['gpu_mean']:.10g}\t{comparison['mean_diff']:.10g}\t"
            f"{comparison['combined_uncertainty']:.10g}\t{comparison['classification']}\t{comparison['passes']}\n"
        )
    write_text(path, "".join(rows))


def system_status(
    *,
    config: dict[str, object],
    comparisons: dict[str, dict[str, object]],
    convergence_issues: list[str],
    protocol_issues: list[str],
    run_failures: list[str],
) -> tuple[str, str | None, list[str]]:
    reasons: list[str] = []
    first_failing = None
    for failure in run_failures:
        reasons.append(failure)
        first_failing = first_failing or "mechanical_pathology"
    for observable in config["primary_observables"]:
        comparison = comparisons.get(observable, {})
        if not comparison.get("available"):
            reasons.append(f"{observable} is missing.")
            first_failing = first_failing or observable
            continue
        if not comparison.get("passes"):
            reasons.append(f"{observable} exceeds the current combined uncertainty budget ({comparison['classification']}).")
            first_failing = first_failing or observable
    for issue in convergence_issues:
        reasons.append(issue)
        first_failing = first_failing or "finite_sampling"
    for issue in protocol_issues:
        reasons.append(issue)
        first_failing = first_failing or "protocol_scope"
    if first_failing is None:
        return "PASS", None, []
    return "FAIL", first_failing, reasons


def save_msd_curve(path: Path, curve: dict[str, object]) -> None:
    if not curve.get("available"):
        write_text(path, "tau_ps\tmsd_nm2\n")
        return
    rows = ["tau_ps\tmsd_nm2\n"]
    for tau_ps, value in zip(curve["lags_ps"], curve["msd_nm2"]):
        rows.append(f"{tau_ps:.10g}\t{value:.10g}\n")
    write_text(path, "".join(rows))


def parse_box_from_gro(path: Path) -> tuple[float, float, float]:
    values = [float(value) for value in path.read_text(encoding="utf-8").splitlines()[-1].split()]
    return values[0], values[1], values[2]


def analyze_replica_transport(
    *,
    gmx: Path,
    system_id: str,
    config: dict[str, object],
    prod_deffnm: Path,
    replica_root: Path,
    logs_dir: Path,
    commands: list[dict[str, object]],
    env: dict[str, str],
    label_prefix: str,
) -> dict[str, object]:
    ndx_path = replica_root / "transport_groups.ndx"
    write_index_file(ndx_path, config["group_atoms"], config["group_order"])
    coord_xvg = replica_root / "transport_group_com.xvg"
    run_gmx_traj_com(
        gmx=gmx,
        traj_path=prod_deffnm.with_suffix(".trr"),
        tpr_path=prod_deffnm.with_suffix(".tpr"),
        ndx_path=ndx_path,
        ngroups=len(config["group_order"]),
        out_path=coord_xvg,
        logs_dir=logs_dir,
        commands=commands,
        label=f"{label_prefix}_traj_com",
        env=env,
        system_id=system_id,
    )
    times_ps, vectors = parse_group_coord_xvg(coord_xvg)
    energy_xvg = replica_root / "production_observables.xvg"
    energy_series = extract_energy_series(gmx, prod_deffnm.with_suffix(".edr"), energy_xvg, ENERGY_TERMS)
    temp_series = [float(value) for value in energy_series.get("Temperature", [])]
    volume_series = [float(value) for value in energy_series.get("Volume", [])]
    if not volume_series:
        box_x = [float(value) for value in energy_series.get("Box-X", [])]
        box_y = [float(value) for value in energy_series.get("Box-Y", [])]
        box_z = [float(value) for value in energy_series.get("Box-Z", [])]
        volume_series = volume_nm3_from_box_terms(box_x, box_y, box_z)
    if not volume_series:
        box = parse_box_from_gro(prod_deffnm.with_suffix(".gro"))
        volume_series = fallback_constant_series(box[0] * box[1] * box[2], max(1, len(temp_series)))
    if not temp_series:
        temp_series = fallback_constant_series(DEFAULT_TEMP_K, len(volume_series))

    metrics: dict[str, dict[str, object]] = {}
    convergence_flags: list[str] = []
    for observable in config["diffusion_observables"]:
        metric_name = str(observable["metric"])
        group_name = str(observable["group"])
        if tuple(config["group_atoms"][group_name]) == tuple(config["group_atoms"]["SYSTEM"]):
            metrics[metric_name] = {
                "available": False,
                "reason": "com_drift_removal_collapses_single_group_measurement",
            }
            convergence_flags.append(
                f"{metric_name} is not measurable here: the selected group equals the full system, so mandatory COM-drift removal collapses the MSD to zero."
            )
            continue
        coords_nm = subtract_system_drift(times_ps, vectors, group_name)
        curve = compute_msd_curve(times_ps, coords_nm)
        curve_path = replica_root / f"{metric_name}_msd.tsv"
        save_msd_curve(curve_path, curve)
        if not curve.get("available"):
            metrics[metric_name] = {
                "available": False,
                "reason": curve.get("reason", "msd_unavailable"),
                "msd_curve_tsv": str(curve_path),
            }
            convergence_flags.append(f"{metric_name} MSD fit is unavailable.")
            continue
        block_stats = compute_block_estimates(times_ps, coords_nm)
        metric_payload = {
            "available": True,
            "value": curve["diffusion_cm2_s"],
            "value_nm2_ps": curve["diffusion_nm2_ps"],
            "r_squared": curve["r_squared"],
            "fit_start_ps": curve["fit_start_ps"],
            "fit_end_ps": curve["fit_end_ps"],
            "fit_points": curve["fit_points"],
            "block_available": block_stats.get("available", False),
            "block_values_cm2_s": block_stats.get("estimates_cm2_s", []),
            "block_sem": block_stats.get("sem_cm2_s", 0.0),
            "block_relative_uncertainty": block_stats.get("relative_uncertainty", math.inf),
            "msd_curve_tsv": str(curve_path),
        }
        metrics[metric_name] = metric_payload
        if curve["r_squared"] < 0.99:
            convergence_flags.append(f"{metric_name} MSD linearity is weak (R^2={curve['r_squared']:.6f}).")
        if not block_stats.get("available"):
            convergence_flags.append(f"{metric_name} has no valid 5-block estimate.")
        elif block_stats["relative_uncertainty"] >= 0.10:
            convergence_flags.append(
                f"{metric_name} 5-block relative uncertainty is {block_stats['relative_uncertainty']:.3f}, above the 0.10 frozen target."
            )

    avg_temp = mean(temp_series)
    avg_volume = mean(volume_series)
    if "small_salt_polymer_box" == system_id:
        conductivity_value = compute_conductivity_cne_s_cm(
            temperature_k=avg_temp,
            volume_nm3=avg_volume,
            terms=tuple(config["diffusion_observables"]),
            replica_metrics=metrics,
        )
        if conductivity_value is None:
            metrics["conductivity_cne_s_cm"] = {"available": False}
            convergence_flags.append("conductivity_cne_s_cm is unavailable.")
        else:
            cat = metrics["cation_diffusivity_cm2_s"]["value"]
            an = metrics["anion_diffusivity_cm2_s"]["value"]
            denom = cat + an
            metrics["conductivity_cne_s_cm"] = {
                "available": True,
                "value": conductivity_value,
                "block_sem": 0.0,
            }
            metrics["transference_ne"] = {
                "available": denom != 0.0,
                "value": (cat / denom) if denom != 0.0 else math.nan,
                "block_sem": 0.0,
            }
            if denom == 0.0:
                convergence_flags.append("transference_ne denominator collapsed to zero.")

    return {
        "group_coord_xvg": str(coord_xvg),
        "transport_index": str(ndx_path),
        "energy_xvg": str(energy_xvg),
        "average_temperature_k": avg_temp,
        "average_volume_nm3": avg_volume,
        "metrics": metrics,
        "convergence_flags": convergence_flags,
        "energy_series": {
            "Temperature": temp_series,
            "Pressure": [float(value) for value in energy_series.get("Pressure", [])],
            "Potential": [float(value) for value in energy_series.get("Potential", [])],
            "Volume": volume_series,
            "Density": [float(value) for value in energy_series.get("Density", [])],
        },
    }


def analyze_replica_transport_molecule_wise(
    *,
    gmx: Path,
    config: dict[str, object],
    top_path: Path,
    prod_deffnm: Path,
    replica_root: Path,
    logs_dir: Path,
    commands: list[dict[str, object]],
    env: dict[str, str],
    label_prefix: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    species_defs = tuple(config["diffusion_observables"])
    species_groups, species_counts = build_species_groups(top_path=top_path, species_defs=species_defs)
    metrics: dict[str, dict[str, object]] = {}
    convergence_flags: list[str] = []
    for species_def in species_defs:
        species_name = str(species_def["species"])
        metric_name = str(species_def["metric"])
        ndx_path = replica_root / f"{species_name}.ndx"
        group_name = species_name.upper()
        write_index_file(
            ndx_path,
            {group_name: tuple(species_groups[species_name])},
            (group_name,),
        )
        msdout_path = replica_root / f"{species_name}_msdout.xvg"
        diffmol_path = replica_root / f"{species_name}_diff_mol.xvg"
        traj_path = choose_traj_path(prod_deffnm)
        run_gmx_msd_mol(
            gmx=gmx,
            traj_path=traj_path,
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
        fit_window = select_molecule_fit_window(times_ps, msd_nm2)
        if not fit_window.get("valid", False):
            convergence_flags.append(f"{metric_name} MSD fit window is invalid.")
            metrics[metric_name] = {"available": False, "reason": "invalid_fit_window"}
            continue
        run_gmx_msd_mol(
            gmx=gmx,
            traj_path=traj_path,
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
        relative_fit_error = (
            math.inf if legend["diffusion_cm2_s"] == 0.0 else abs(legend["fit_error_cm2_s"] / legend["diffusion_cm2_s"])
        )
        relative_molecule_sem = math.inf if molecule_mean == 0.0 else abs(molecule_sem / molecule_mean)
        if not fit.get("valid", False):
            convergence_flags.append(f"{metric_name} MSD fit window is invalid.")
        elif fit["r_squared"] < R2_THRESHOLD:
            convergence_flags.append(f"{metric_name} MSD linearity is weak (R^2={fit['r_squared']:.6f} < {R2_THRESHOLD:.2f}).")
        if relative_fit_error >= RELATIVE_UNCERTAINTY_THRESHOLD:
            convergence_flags.append(
                f"{metric_name} MSD fit uncertainty is large ({relative_fit_error:.3f} >= {RELATIVE_UNCERTAINTY_THRESHOLD:.2f})."
            )
        if relative_molecule_sem >= RELATIVE_UNCERTAINTY_THRESHOLD:
            convergence_flags.append(
                f"{metric_name} per-molecule diffusivity spread is large ({relative_molecule_sem:.3f} >= {RELATIVE_UNCERTAINTY_THRESHOLD:.2f})."
            )
        metrics[metric_name] = {
            "available": True,
            "value": legend["diffusion_cm2_s"],
            "count": species_counts[species_name],
            "molecule_count": len(diff_values),
            "molecule_types": [str(name) for name in species_def["molecule_types"]],
            "fit_error_cm2_s": legend["fit_error_cm2_s"],
            "block_sem": max(float(legend["fit_error_cm2_s"]), float(molecule_sem)),
            "r_squared": fit.get("r_squared"),
            "fit_r_squared": fit.get("r_squared"),
            "fit_start_ps": fit_start_ps,
            "fit_end_ps": fit_end_ps,
            "molecule_mean_cm2_s": molecule_mean,
            "molecule_std_cm2_s": molecule_std,
            "molecule_sem_cm2_s": molecule_sem,
            "relative_fit_error": relative_fit_error,
            "relative_molecule_sem": relative_molecule_sem,
            "msdout_xvg": str(msdout_path),
            "diff_mol_xvg": str(diffmol_path),
            "msd_curve_tsv": str(curve_tsv),
            "transport_index": str(ndx_path),
        }

    energy_xvg = replica_root / "production_observables.xvg"
    energy_series = extract_energy_series(gmx, prod_deffnm.with_suffix(".edr"), energy_xvg, ENERGY_TERMS)
    temp_series = [float(value) for value in energy_series.get("Temperature", [])]
    volume_series = [float(value) for value in energy_series.get("Volume", [])]
    if not volume_series:
        box_x = [float(value) for value in energy_series.get("Box-X", [])]
        box_y = [float(value) for value in energy_series.get("Box-Y", [])]
        box_z = [float(value) for value in energy_series.get("Box-Z", [])]
        volume_series = volume_nm3_from_box_terms(box_x, box_y, box_z)
    if not volume_series:
        box = parse_box_from_gro(prod_deffnm.with_suffix(".gro"))
        volume_series = fallback_constant_series(box[0] * box[1] * box[2], max(1, len(temp_series)))
    if not temp_series:
        temp_series = fallback_constant_series(DEFAULT_TEMP_K, len(volume_series))
    avg_temp = mean(temp_series)
    avg_volume = mean(volume_series)

    if "conductivity_cne_s_cm" in tuple(config["derived_observables"]):
        conductivity_value = compute_conductivity_cne_s_cm(
            temperature_k=avg_temp,
            volume_nm3=avg_volume,
            terms=species_defs,
            replica_metrics=metrics,
        )
        if conductivity_value is None:
            metrics["conductivity_cne_s_cm"] = {"available": False}
            convergence_flags.append("conductivity_cne_s_cm is unavailable.")
        else:
            cat = metrics["cation_diffusivity_cm2_s"]["value"]
            an = metrics["anion_diffusivity_cm2_s"]["value"]
            denom = cat + an
            metrics["conductivity_cne_s_cm"] = {
                "available": True,
                "value": conductivity_value,
                "count": metrics["cation_diffusivity_cm2_s"]["count"] + metrics["anion_diffusivity_cm2_s"]["count"],
                "block_sem": 0.0,
            }
            metrics["transference_ne"] = {
                "available": denom != 0.0,
                "value": (cat / denom) if denom != 0.0 else math.nan,
                "count": 1,
                "block_sem": 0.0,
            }
            if denom == 0.0:
                convergence_flags.append("transference_ne denominator collapsed to zero.")

    return {
        "energy_xvg": str(energy_xvg),
        "average_temperature_k": avg_temp,
        "average_volume_nm3": avg_volume,
        "metrics": metrics,
        "convergence_flags": convergence_flags,
        "species_groups": {name: {"atom_count": len(atoms), "molecule_count": species_counts[name]} for name, atoms in species_groups.items()},
        "energy_series": {
            "Temperature": temp_series,
            "Pressure": [float(value) for value in energy_series.get("Pressure", [])],
            "Potential": [float(value) for value in energy_series.get("Potential", [])],
            "Volume": volume_series,
            "Density": [float(value) for value in energy_series.get("Density", [])],
        },
    }


def run_system(
    *,
    args: argparse.Namespace,
    gmx: Path,
    system_id: str,
    config: dict[str, object],
    tp0: dict[str, object],
    system_root: Path,
) -> dict[str, object]:
    inputs_dir = system_root / "inputs"
    logs_dir = system_root / "logs"
    summaries_dir = system_root / "summaries"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    env = base_env(args)
    analysis_mode = str(config.get("analysis_mode", "group_com"))
    scaffold_manifest = None
    if "scaffold_manifest" in config:
        scaffold_manifest = load_json(Path(str(config["scaffold_manifest"])))
        top_path = Path(str(scaffold_manifest["artifacts"]["topology"]))
        conf_path = Path(str(scaffold_manifest["artifacts"]["gro"]))
        fixture_meta = parse_gro_metadata(conf_path)
    else:
        fixture_root = fixture_dir(system_id)
        top_path = fixture_root / "topol.top"
        conf_path = fixture_root / "initial_nve.gro"
        fixture_meta = parse_gro_metadata(conf_path)

    commands: list[dict[str, object]] = []
    per_layout_replicas: dict[str, list[dict[str, object]]] = {"cpu": [], "gpu": []}
    run_failures: list[str] = []
    convergence_issues: list[str] = []

    for replica_index in range(1, args.replicas + 1):
        seed = 86113 + replica_index - 1
        equil_mdp = inputs_dir / f"replica_{replica_index:02d}_equil.mdp"
        prod_mdp = inputs_dir / f"replica_{replica_index:02d}_prod.mdp"
        write_text(
            equil_mdp,
            make_transport_mdp(
                system_id=system_id,
                duration_ps=args.equil_ps,
                energy_stride_ps=args.energy_stride_ps,
                coord_stride_ps=args.coord_stride_ps,
                phase="equil",
                seed=seed,
                args=args,
            ),
        )
        write_text(
            prod_mdp,
            make_transport_mdp(
                system_id=system_id,
                duration_ps=args.prod_ps,
                energy_stride_ps=args.energy_stride_ps,
                coord_stride_ps=args.coord_stride_ps,
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
            run_or_resume_grompp(
                gmx=gmx,
                mdp_path=equil_mdp,
                conf_path=conf_path,
                top_path=top_path,
                tpr_path=equil_deffnm.with_suffix(".tpr"),
                mdout_path=replica_root / "equil_mdout.mdp",
                logs_dir=logs_dir,
                commands=commands,
                label=f"{system_id}_{layout}_replica_{replica_index:02d}_grompp_equil",
                env=env,
            )
            run_or_resume_md(
                gmx=gmx,
                argv=[str(gmx), "mdrun", *(mdrun_args_cpu(args, equil_deffnm) if layout == "cpu" else mdrun_args_gpu(args, equil_deffnm))],
                deffnm=equil_deffnm,
                env=env,
                logs_dir=logs_dir,
                commands=commands,
                label=f"{system_id}_{layout}_replica_{replica_index:02d}_mdrun_equil",
                args=args,
            )
            run_or_resume_grompp(
                gmx=gmx,
                mdp_path=prod_mdp,
                conf_path=equil_deffnm.with_suffix(".gro"),
                top_path=top_path,
                tpr_path=prod_deffnm.with_suffix(".tpr"),
                mdout_path=replica_root / "prod_mdout.mdp",
                logs_dir=logs_dir,
                commands=commands,
                label=f"{system_id}_{layout}_replica_{replica_index:02d}_grompp_prod",
                env=env,
                checkpoint_path=equil_deffnm.with_suffix(".cpt"),
            )
            md_result = run_or_resume_md(
                gmx=gmx,
                argv=[str(gmx), "mdrun", *(mdrun_args_cpu(args, prod_deffnm) if layout == "cpu" else mdrun_args_gpu(args, prod_deffnm))],
                deffnm=prod_deffnm,
                env=env,
                logs_dir=logs_dir,
                commands=commands,
                label=f"{system_id}_{layout}_replica_{replica_index:02d}_mdrun_prod",
                args=args,
            )
            if analysis_mode == "molecule_wise":
                analysis = analyze_replica_transport_molecule_wise(
                    gmx=gmx,
                    config=config,
                    top_path=top_path,
                    prod_deffnm=prod_deffnm,
                    replica_root=replica_root,
                    logs_dir=logs_dir,
                    commands=commands,
                    env=env,
                    label_prefix=f"{system_id}_{layout}_replica_{replica_index:02d}",
                    args=args,
                )
            else:
                analysis = analyze_replica_transport(
                    gmx=gmx,
                    system_id=system_id,
                    config=config,
                    prod_deffnm=prod_deffnm,
                    replica_root=replica_root,
                    logs_dir=logs_dir,
                    commands=commands,
                    env=env,
                    label_prefix=f"{system_id}_{layout}_replica_{replica_index:02d}",
                )
            replica_summary = {
                "replica_index": replica_index,
                "seed": seed,
                "layout": layout,
                "layout_report": md_result["layout_report"],
                "equil_deffnm": str(equil_deffnm),
                "prod_deffnm": str(prod_deffnm),
                "analysis": analysis,
            }
            write_text(replica_root / "replica_summary.json", json.dumps(replica_summary, indent=2, sort_keys=True))
            per_layout_replicas[layout].append(replica_summary)
            convergence_issues.extend(analysis["convergence_flags"])

    aggregated: dict[str, dict[str, object]] = {"cpu": {}, "gpu": {}}
    all_metric_names = [str(entry["metric"]) for entry in config["diffusion_observables"]] + list(config["derived_observables"])
    for layout in ("cpu", "gpu"):
        for metric_name in all_metric_names:
            scalar_replicas = []
            for replica in per_layout_replicas[layout]:
                metric = replica["analysis"]["metrics"].get(metric_name, {})
                if not metric.get("available"):
                    continue
                scalar_replicas.append(
                    {
                        "replica_index": replica["replica_index"],
                        "value": metric["value"],
                        "block_sem": metric.get("block_sem", 0.0),
                        "r_squared": metric.get("r_squared"),
                    }
                )
            aggregated[layout][metric_name] = summarize_scalar_replicas(scalar_replicas)

    comparisons = {
        metric_name: compare_scalar_aggregates(aggregated["cpu"][metric_name], aggregated["gpu"][metric_name])
        for metric_name in all_metric_names
    }
    save_scalar_table(summaries_dir / "transport_comparison.tsv", comparisons)

    system_class = str(config["system_class"])
    duration_expectation = fit_protocol_expectation(system_class, args.prod_ps)
    protocol_issues: list[str] = []
    if fixture_meta["natoms"] != config["expected_natoms"]:
        protocol_issues.append(
            f"Fixture natom mismatch: expected {config['expected_natoms']}, observed {fixture_meta['natoms']}."
        )
    if fixture_meta["natoms"] < TP0_MIN_ATOMS:
        protocol_issues.append(
            f"Fixture size is out of TP0 scope: {fixture_meta['natoms']} atoms < {TP0_MIN_ATOMS} atom minimum."
        )
    if float(fixture_meta["min_box_nm"]) <= TP0_MIN_BOX_NM:
        protocol_issues.append(
            f"Fixture box is out of TP0 scope: min box {fixture_meta['min_box_nm']:.3f} nm <= {TP0_MIN_BOX_NM:.3f} nm."
        )
    if not duration_expectation["meets_minimum"]:
        protocol_issues.append(
            f"Production duration {args.prod_ps:.1f} ps is below the TP0 minimum {duration_expectation['minimum_prod_ps']:.1f} ps."
        )
    status, first_failing, reasons = system_status(
        config=config,
        comparisons=comparisons,
        convergence_issues=dedupe_preserve_order(convergence_issues),
        protocol_issues=dedupe_preserve_order(protocol_issues),
        run_failures=run_failures,
    )

    system_result = {
        "system_id": system_id,
        "status": status,
        "first_failing_observable": first_failing,
        "failure_reasons": reasons,
        "fixture_metadata": fixture_meta,
        "system_class": system_class,
        "analysis_settings": {
            "replicas": args.replicas,
            "equil_ps": args.equil_ps,
            "prod_ps": args.prod_ps,
            "coord_stride_ps": args.coord_stride_ps,
            "energy_stride_ps": args.energy_stride_ps,
            "fit_window_ratio": list(FIT_WINDOW),
            "block_count": BLOCK_COUNT,
            "estimator": "Einstein MSD with COM-drift removal; conductivity from cNE; transference from NE ratio.",
        },
        "tp0_protocol_expectation": duration_expectation,
        "tp0_scope_issues": protocol_issues,
        "per_layout_aggregates": {
            layout: {metric_name: sanitize_scalar_aggregate(payload) for metric_name, payload in metrics.items()}
            for layout, metrics in aggregated.items()
        },
        "observable_comparisons": comparisons,
        "replica_variability": {
            layout: {
                metric_name: {
                    "std": payload["std"],
                    "sem": payload["sem"],
                    "mean_block_sem": payload["mean_block_sem"],
                }
                for metric_name, payload in metrics.items()
                if payload.get("available")
            }
            for layout, metrics in aggregated.items()
        },
        "artifacts": {
            "system_root": str(system_root),
            "transport_comparison_tsv": str(summaries_dir / "transport_comparison.tsv"),
            **({"scaffold_manifest": str(Path(str(config["scaffold_manifest"])).resolve())} if "scaffold_manifest" in config else {}),
        },
    }
    write_text(summaries_dir / "system_result.json", json.dumps(system_result, indent=2, sort_keys=True))
    write_recorded_commands_script(system_root / "run_commands.sh", commands)
    return system_result


def build_manifest_markdown(manifest: dict[str, object]) -> str:
    lines = [
        "# Gate H Transport Validation",
        "",
        f"- Verdict: `{manifest['status']}`",
        f"- Production recommendation: `{manifest['production_recommendation']}`",
        f"- Replica count per layout: `{manifest['replicas']}`",
        f"- Equilibration / production: `{manifest['equil_ps']} ps / {manifest['prod_ps']} ps`",
        f"- Protocol caveat: {manifest['protocol_caveat']}",
        "",
        "## Systems",
    ]
    for system in manifest["systems"]:
        lines.append(f"- `{system['system_id']}`: `{system['status']}`")
        if system["first_failing_observable"] is not None:
            lines.append(f"  First failing observable: `{system['first_failing_observable']}`")
    return "\n".join(lines) + "\n"


def write_blocker_manifest(
    out_root: Path,
    *,
    gate_a_manifest: dict[str, object],
    gate_e_manifest: dict[str, object],
    gate_f_manifest: dict[str, object],
    gate_g_manifest: dict[str, object],
    reason: str,
) -> None:
    manifest = {
        "status": "BLOCKER",
        "production_recommendation": "NO-GO",
        "artifact_root": str(out_root),
        "mechanical_prerequisites": {
            "gate_a_status": gate_a_manifest.get("status"),
            "gate_e_status": gate_e_manifest.get("status"),
            "gate_f_status": gate_f_manifest.get("status"),
            "gate_g_status": gate_g_manifest.get("status"),
        },
        "blocking_reasons": [reason],
        "systems": [],
        "first_failure": {"field": "upstream_prerequisites", "details": reason},
    }
    write_text(out_root / "gate_h_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    write_text(
        out_root / "gate_h_manifest.md",
        "\n".join(
            [
                "# Gate H Transport",
                "",
                "- Status: BLOCKER",
                "- Production recommendation: NO-GO",
                f"- Gate A status: `{manifest['mechanical_prerequisites']['gate_a_status']}`",
                f"- Gate E status: `{manifest['mechanical_prerequisites']['gate_e_status']}`",
                f"- Gate F status: `{manifest['mechanical_prerequisites']['gate_f_status']}`",
                f"- Gate G status: `{manifest['mechanical_prerequisites']['gate_g_status']}`",
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
    if out_root.exists() and not args.resume:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    gate_a_manifest = load_json(Path(args.gate_a_manifest))
    gate_e_manifest = load_json(Path(args.gate_e_manifest))
    gate_f_manifest = load_json(Path(args.gate_f_manifest))
    gate_g_manifest = load_json(Path(args.gate_g_manifest))
    tp0_metadata = load_json(Path(args.tp0_metadata))
    tp1_status = load_json(Path(args.tp1_status))
    try:
        validate_gate_prerequisites(gate_a_manifest, gate_e_manifest, gate_f_manifest, gate_g_manifest)
    except ValueError as exc:
        write_blocker_manifest(
            out_root,
            gate_a_manifest=gate_a_manifest,
            gate_e_manifest=gate_e_manifest,
            gate_f_manifest=gate_f_manifest,
            gate_g_manifest=gate_g_manifest,
            reason=str(exc),
        )
        return

    gmx = Path(args.gmx).resolve()
    maybe_build(args, gmx)

    version_text = capture_output([str(gmx), "--version"], cwd=REPO_ROOT, env=os.environ.copy())

    system_results = []
    system_configs = build_system_configs(args)
    for system_id, config in system_configs.items():
        system_results.append(
            run_system(
                args=args,
                gmx=gmx,
                system_id=system_id,
                config=config,
                tp0=tp0_metadata,
                system_root=out_root / system_id,
            )
        )

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
        "production_recommendation": "GO" if status == "PASS" else "NO-GO",
        "replicas": args.replicas,
        "equil_ps": args.equil_ps,
        "prod_ps": args.prod_ps,
        "coord_stride_ps": args.coord_stride_ps,
        "energy_stride_ps": args.energy_stride_ps,
        "trestart_ps": args.trestart_ps,
        "maxtau_ps": args.maxtau_ps,
        "fixture_mode": args.fixture_mode,
        "single_rank": True,
        "dlb": "no",
        "precision_mode": parse_precision_mode(version_text),
        "gpu_support": parse_gpu_support(version_text),
        "protocol_caveat": (
            "Gate H reuses the mechanically validated exact-r-RESPA NVT path; small mode is out of TP0 transport scope by size/box/duration,"
            " while large mode fixes size/box but still requires TP0-scale production length and charged-side long NPT density conditioning."
        ),
        "mechanical_prerequisites": {
            "gate_a_status": gate_a_manifest.get("status"),
            "gate_e_status": gate_e_manifest.get("status"),
            "gate_f_status": gate_f_manifest.get("status"),
            "gate_g_status": gate_g_manifest.get("status"),
        },
        "tp0_metadata": {
            "path": str(Path(args.tp0_metadata).resolve()),
            "protocol": tp0_metadata.get("protocol", {}),
            "acceptance_criteria": tp0_metadata.get("acceptance_criteria", {}),
        },
        "tp1_status": {
            "path": str(Path(args.tp1_status).resolve()),
            "status": tp1_status.get("status"),
            "audit_reason": tp1_status.get("audit_reason"),
        },
        "systems": system_results,
        "first_failure": first_failure,
        "artifacts": {
            "root": str(out_root),
            "manifest_json": str(out_root / "gate_h_manifest.json"),
            "manifest_md": str(out_root / "gate_h_manifest.md"),
        },
    }
    write_text(out_root / "gate_h_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    write_text(out_root / "gate_h_manifest.md", build_manifest_markdown(manifest))


if __name__ == "__main__":
    main()
