#!/usr/bin/env python3
"""Worker-side runner for the PolyGen multi-system validation manifests."""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


DEFAULT_OUTDIR = Path("GROMACS_PCFF/output/polygen_multisystem_validation_20260512_m1p50")
DEFAULT_MANIFEST = DEFAULT_OUTDIR / "manifest/jobs.csv"
DEFAULT_SYSTEMS = DEFAULT_OUTDIR / "manifest/systems.csv"
LANES = ("lammps_cpu", "gmx_cpu", "gmx_gpu")
DEFAULT_SMOKE_EQFACTOR = 0.00005
MOLALITY_BASIS_MIXTURE = "salt_mol_per_kg_total_mixture"
GMX_PCFF_RUNTIME_ENV = (
    # Single-rank CPU PME otherwise enables a 1x1x1 DD grid whose initial
    # distribution wraps every atom into the primary cell.  The no-DD force
    # setup has a separate put_atoms_in_box path, so disable that as well.
    # Together these retain the LAMMPS image-unwrapped representation from
    # Eq01 through the second `velocity ... mom yes rot yes` at Eq04.
    "GMX_DD_SINGLE_RANK=0;"
    "GMX_PCFF_LAMMPS_CG_EM_SKIP_PUT_ATOMS_IN_BOX=1;"
    "GMX_PCFF_EXACT_RESPA_FUSED_INITIAL_DRIFT=1;"
    "GMX_PCFF_MIXED_CLASS2_LINEAR_ANGLE_SIN_FLOOR=0.00038;"
    "GMX_PCFF_MTTK_MASS_MODE=lammps;"
    "GMX_PCFF_NVT_MASS_MODE=lammps_tchain;"
    "GMX_PCFF_MTTK_BOXV_INTEGRATOR=lammps;"
    # Match LAMMPS FixNH velocity scaling and box remapping. The earlier
    # apparent box explosion came from invalid 1-4 pairs and a mismatched
    # r-RESPA contribution layout, not from this remap path.
    "GMX_PCFF_EXACT_RESPA_MTTK_EXTENDED_UPDATE=velocity-lammps-remap;"
    "GMX_PCFF_EXACT_RESPA_MTTK_INLINE_BOX_REMAP=1;"
    "GMX_PCFF_EXACT_RESPA_MTTK_INLINE_BOX_REMAP_PME=1;"
    "GMX_PCFF_NHC_INTEGRATOR=lammps;"
    # `comm-mode = Linear` supplies the LAMMPS 3N-3 thermostat DOF.  The MDP
    # uses an effectively one-shot nstcomm, and exact r-RESPA requires an
    # explicit opt-in for this supported linear COM-removal path.
    "GMX_PCFF_EXACT_RESPA_ALLOW_LINEAR_COM_REMOVAL=1;"
    "GMX_PCFF_EXACT_RESPA_PRE_TROTTER=two;"
    "GMX_PCFF_EXACT_RESPA_POST_TROTTER=three;"
    "GMX_PCFF_ALLOW_LONG_EXCLUDED=1"
)


def require_pandas():
    if pd is None:
        raise ModuleNotFoundError(
            "pandas is required for manifest/selected CSV operations. "
            "Install the conda package set from `bootstrap-plan` first."
        )
    return pd


def repo_workspace_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "GROMACS_PCFF").is_dir() and (candidate / "MY_PAPER_RELATED").is_dir():
            return candidate
    return Path.cwd().resolve()


def prepend_sys_path(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def which(name: str) -> str | None:
    return shutil.which(name)


def detect_nvidia_smi() -> str | None:
    return shutil.which("nvidia-smi")


def check_import(module: str) -> str:
    try:
        importlib.import_module(module)
        return "ok"
    except Exception as exc:
        return f"missing:{type(exc).__name__}:{exc}"


def existing_binary(candidates: list[Path]) -> str | None:
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def default_gmx_binary(workspace: Path, lane: str) -> str | None:
    repo = workspace / "GROMACS_PCFF"
    if lane == "gmx_cpu":
        return existing_binary(
            [
                repo / "build_gateb_double_cpu/bin/gmx_d",
                repo / "build-znver4/bin/gmx",
                repo / "build/bin/gmx",
                repo / "build_desktop/bin/gmx",
            ]
        )
    if lane == "gmx_gpu":
        return existing_binary(
            [
                repo / "build_gateb_cuda/bin/gmx",
                repo / "build/bin/gmx",
                repo / "build_desktop/bin/gmx",
            ]
        )
    return None


def resolve_binary_path(binary: str | None, workspace: Path) -> str | None:
    if not binary:
        return None
    path = Path(binary).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    if len(path.parts) > 1:
        return str((workspace / path).resolve())
    found = shutil.which(binary)
    return found or str((workspace / path).resolve())


def lane_mdrun_settings(lane: str, workspace: Path, gmx_binary: str | None) -> dict[str, str]:
    pair14_level = os.environ.get("GROMACS_BATCH_EXACT_RESPA_PAIR14_LEVEL", "2").strip()
    if pair14_level not in {"1", "2"}:
        raise ValueError(
            "GROMACS_BATCH_EXACT_RESPA_PAIR14_LEVEL must be 1 or 2; "
            f"got {pair14_level!r}"
        )
    if lane == "gmx_cpu":
        binary = resolve_binary_path(gmx_binary, workspace) or default_gmx_binary(workspace, lane) or ""
        return {
            "GROMACS_BATCH_GMX_BINARY": binary,
            "GROMACS_BATCH_SCHEDULE": "polygen_em_handoff",
            "GROMACS_BATCH_EXACT_RESPA_PAIR14_LEVEL": pair14_level,
            "GROMACS_BATCH_GROMPP_EXTRA_ARGS": "-maxwarn 10",
            "GROMACS_BATCH_MDRUN_EXTRA_ARGS": "-nb cpu -pme cpu -bonded cpu -update cpu -pin off",
            "GROMACS_BATCH_MDRUN_ENV": GMX_PCFF_RUNTIME_ENV,
        }
    if lane == "gmx_gpu":
        binary = resolve_binary_path(gmx_binary, workspace) or default_gmx_binary(workspace, lane) or ""
        return {
            "GROMACS_BATCH_GMX_BINARY": binary,
            "GROMACS_BATCH_SCHEDULE": "polygen_em_handoff",
            "GROMACS_BATCH_EXACT_RESPA_PAIR14_LEVEL": pair14_level,
            "GROMACS_BATCH_PME_ORDER": "5",
            "GROMACS_BATCH_GROMPP_EXTRA_ARGS": "-maxwarn 10",
            "GROMACS_BATCH_MDRUN_EXTRA_ARGS": "-nb gpu -pme cpu -bonded gpu -update cpu -pin off -dlb no -notunepme",
            "GROMACS_BATCH_MDRUN_EM_EXTRA_ARGS": "-nb cpu -pme cpu -bonded cpu -update cpu -pin off",
            "GROMACS_BATCH_MDRUN_ENV": GMX_PCFF_RUNTIME_ENV,
        }
    return {}


def batch_project_id(raw_value: object, fallback: int = 0) -> int:
    try:
        return int(float(raw_value))
    except Exception:
        return int(fallback)


def lammps_lane_root(outdir: Path, run_group: str, role: str) -> Path:
    return outdir / "runs_batch" / run_group / role / "lammps_cpu"


def load_lammps_stage_layouts(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    layout_path = Path(path).expanduser().resolve()
    if not layout_path.exists():
        raise FileNotFoundError(f"LAMMPS stage layout file not found: {layout_path}")
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    if "stage_layouts" in data:
        data = data["stage_layouts"]
    if not isinstance(data, dict):
        raise ValueError(f"LAMMPS stage layout must be a JSON object: {layout_path}")
    layouts: dict[str, dict[str, object]] = {}
    for stage, raw in data.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid layout for stage {stage!r}: expected object")
        layouts[str(stage)] = {
            "mpi_ranks": int(raw.get("mpi_ranks", 1)),
            "omp_threads": int(raw.get("omp_threads", 1)),
            "openmp": bool(raw.get("openmp", False)),
            "source": str(layout_path),
        }
    return layouts


def load_gromacs_stage_layouts(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    layout_path = Path(path).expanduser().resolve()
    if not layout_path.exists():
        raise FileNotFoundError(f"GROMACS stage layout file not found: {layout_path}")
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    if "stage_layouts" in data:
        data = data["stage_layouts"]
    if not isinstance(data, dict):
        raise ValueError(f"GROMACS stage layout must be a JSON object: {layout_path}")
    layouts: dict[str, dict[str, object]] = {}
    for stage, raw in data.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid GROMACS layout for stage {stage!r}: expected object")
        extra_args = raw.get("extra_args", [])
        if isinstance(extra_args, str):
            extra_norm: object = extra_args
        elif isinstance(extra_args, list):
            extra_norm = [str(x) for x in extra_args]
        else:
            extra_norm = []
        env_raw = raw.get("env", {})
        env_norm = {str(k): str(v) for k, v in dict(env_raw).items()} if isinstance(env_raw, dict) else {}
        layouts[str(stage)] = {
            "ntomp": int(raw.get("ntomp", raw.get("omp_threads", 1))),
            "ntmpi": int(raw.get("ntmpi", 1)),
            "extra_args": extra_norm,
            "env": env_norm,
            "source": str(layout_path),
        }
    return layouts


def bridge_script(workspace: Path) -> Path:
    # The bridge is part of the engine branch.  Resolving it relative to this
    # worker prevents a remote run from silently using an unrelated dirty
    # checkout that cannot be reproduced by pulling the branch.
    del workspace
    path = Path(__file__).resolve().parents[2] / "tools/pcff_fixture_bridge/lammps_data_bridge.py"
    if not path.exists():
        raise FileNotFoundError(f"LAMMPS-data bridge script not found: {path}")
    return path


def _done(path: Path) -> bool:
    return path.exists() and (not path.is_file() or path.stat().st_size > 0)


def _float_matches(a: object, b: float, *, tol: float = 1.0e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def _cached_pysoftk_context_matches(
    cached: dict[str, object] | None,
    *,
    replica_seed: int,
    n_salt: int,
    m_target: float,
    dp: int,
    monomer_smiles: str,
) -> bool:
    if cached is None or cached.get("n_chains") is None:
        return False
    try:
        if int(cached.get("seed_offset", -1)) != int(replica_seed):
            return False
        if int(cached.get("n_salt", -1)) != int(n_salt):
            return False
        if int(cached.get("dp", -1)) != int(dp):
            return False
    except Exception:
        return False
    if str(cached.get("monomer_smiles", "")) != str(monomer_smiles):
        return False
    if str(cached.get("molality_basis", "")) != MOLALITY_BASIS_MIXTURE:
        return False
    return _float_matches(cached.get("molality_target"), float(m_target))


def _remove_lammps_downstream_outputs(proj: Path) -> None:
    for rel in ("build/cell", "build/ion_remap", "MD", "logs"):
        path = proj / rel
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    for rel in ("prepared_lammps_inputs.json",):
        (proj / rel).unlink(missing_ok=True)


def prepare_lammps_input_projects(
    selected_csv: Path,
    *,
    workspace: Path,
    outdir: Path,
    role: str,
    run_group: str,
    nproc: int,
    eqfactor: float,
    production_ns: float,
    resume_existing: bool,
    force_restart: bool,
    lmp_binary: str | None,
    mpirun_binary: str | None = None,
) -> pd.DataFrame:
    pd = require_pandas()
    """Build pysoftk/LUNAR/ion-remapped LAMMPS project inputs without running MD."""
    selected_df = pd.read_csv(selected_csv).reset_index(drop=True)
    output_root = lammps_lane_root(outdir, run_group, role)
    output_root.mkdir(parents=True, exist_ok=True)
    if selected_df.empty:
        return pd.DataFrame()

    batch_dir = workspace / "MY_PAPER_RELATED/LAMMPS_BATCH"
    prepend_sys_path(batch_dir)
    from batch_utils.batch_run_utils import psmiles_to_monomer_smiles
    from batch_utils.lunar_utils import run_lunar_pipeline
    from batch_utils.md_utils import prepare_ion_remap, setup_md_environment
    from batch_utils.pysoftk_utils import build_polymer_inputs, load_cached_pysoftk_context

    rows: list[dict[str, object]] = []
    for row_idx, row in selected_df.iterrows():
            traj_id = batch_project_id(row.get("Trajectory ID"), row_idx)
            proj = output_root / f"Traj_{traj_id}"
            if force_restart and proj.exists():
                shutil.rmtree(proj)
            proj.mkdir(parents=True, exist_ok=True)
            resume_effective = bool(resume_existing) and not bool(force_restart)
            skipped: list[str] = []
            status = "ok"
            error = ""
            try:
                replica_seed = int(row.get("replica_seed", traj_id))
                psmiles = str(row["SMILES"])
                monomer_smiles = psmiles_to_monomer_smiles(psmiles)
                dp_val = row.get("Degree of Polymerization", 19)
                dp = max(1, int(round(float(dp_val)))) if pd.notna(dp_val) else 19
                m_val = row.get("Molality", 1.45)
                m_target = float(m_val) if pd.notna(m_val) else 1.45
                m_min = max(0.01, m_target - 0.05)
                m_max = m_target + 0.05

                py_ctx = None
                pysoftk_reran = False
                lunar_reran = False
                if resume_effective and _done(proj / "build/chain_fixed.mol2") and _done(proj / "build/pysoftk_context.json"):
                    cached = load_cached_pysoftk_context(proj)
                    if _cached_pysoftk_context_matches(
                        cached,
                        replica_seed=replica_seed,
                        n_salt=100,
                        m_target=m_target,
                        dp=dp,
                        monomer_smiles=monomer_smiles,
                    ):
                        py_ctx = cached
                        skipped.append("pysoftk")
                    elif cached is not None:
                        _remove_lammps_downstream_outputs(proj)
                        print(
                            "[pysoftk] cache mismatch; rebuilding "
                            f"(cached molality={cached.get('molality_target')}, requested={m_target})"
                        )
                if py_ctx is None:
                    py_ctx = build_polymer_inputs(
                        proj,
                        monomer_smiles=monomer_smiles,
                        placeholder="Br",
                        dp=dp,
                        n_salt=100,
                        m_min=m_min,
                        m_max=m_max,
                        m_target=m_target,
                        seed_offset=replica_seed,
                    )
                    pysoftk_reran = True

                if (not pysoftk_reran) and resume_effective and _done(proj / "build/cell/polymer_cell.data"):
                    skipped.append("lunar")
                else:
                    run_lunar_pipeline(
                        proj,
                        n_chains=int(py_ctx["n_chains"]),
                        lunar_dir=batch_dir / "extern/LUNAR",
                        force_field="PCFF",
                        seed=replica_seed,
                    )
                    lunar_reran = True

                if (
                    (not pysoftk_reran)
                    and (not lunar_reran)
                    and
                    resume_effective
                    and _done(proj / "build/ion_remap/type_map.json")
                    and (proj / "build/ion_remap/ion_parameters").is_dir()
                    and (proj / "build/ion_remap/molecular_templates").is_dir()
                ):
                    skipped.append("ion_remap")
                else:
                    prepare_ion_remap(proj, base_dir=batch_dir / "BASE")

                ctx = setup_md_environment(
                    proj,
                    base_dir=batch_dir / "BASE",
                    start_phase="analysis",
                    resume_existing=resume_effective and not (pysoftk_reran or lunar_reran),
                    force_rerun_from_start_phase=False,
                    eqfactor=float(eqfactor),
                    production_total_ns=float(production_ns),
                    nproc=int(nproc),
                    use_kokkos=False,
                    lammps_binary=lmp_binary,
                    mpirun_binary=mpirun_binary,
                    lammps_thermo_flush=True,
                    replica_seed=replica_seed,
                )
                manifest = {
                    "schema_name": "polygen_multisystem_prepared_lammps_input",
                    "schema_version": 1,
                    "trajectory_id": int(traj_id),
                    "run_group": run_group,
                    "role": role,
                    "production_total_ns": float(production_ns),
                    "eqfactor": float(eqfactor),
                    "replica_seed": int(replica_seed),
                    "md_dir": str(ctx["md_dir"]),
                    "prepared_files": [
                        "MD/in.data",
                        "MD/equilibration.in",
                        "MD/production.in",
                        "MD/equil_stage00_pre_em.in",
                        "MD/equil_stage01_em.in",
                        "MD/equil_stage02_dynamics.in",
                        "MD/resume_inputs",
                        "MD/.resume_state",
                        "MD/ion_parameters",
                        "MD/molecular_templates",
                    ],
                }
                (proj / "prepared_lammps_inputs.json").write_text(
                    json.dumps(manifest, indent=2) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:
                status = "failed"
                error = repr(exc)
            rows.append(
                {
                    "Trajectory ID": traj_id,
                    "status": status,
                    "project_dir": str(proj),
                    "skipped_phases": ",".join(skipped),
                    "error": error,
                }
            )
            print(f"[prepare-inputs] Traj_{traj_id}: {status}")

    status_df = pd.DataFrame(rows)
    status_df.to_csv(output_root / "prepared_input_status.csv", index=False)
    print("prepared status:", output_root / "prepared_input_status.csv")
    return status_df


def _run_bridge_command(*, workspace: Path, source_data: Path, bridge_out: Path, traj_id: int) -> None:
    bridge_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(bridge_script(workspace)),
        "--data",
        str(source_data),
        "--out",
        str(bridge_out),
        "--system-id",
        f"Traj_{traj_id}",
        "--display-name",
        f"Traj_{traj_id}",
        "--category",
        "polymer_box",
        "--pair-style-arg",
        "9.5",
        "--kspace-style",
        "pppm 0.0001",
    ]
    print("$", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=str(workspace), check=True)


def _resolve_lammps_binary(lmp_binary: str | None = None) -> str:
    candidates = []
    if lmp_binary:
        candidates.append(str(Path(lmp_binary).expanduser()))
    env_bin = os.environ.get("LAMMPS_BATCH_LMP_BINARY")
    if env_bin:
        candidates.append(str(Path(env_bin).expanduser()))
    found = shutil.which("lmp")
    if found:
        candidates.append(found)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(str(Path(conda_prefix) / "bin/lmp"))
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path.resolve())
    raise FileNotFoundError(
        "LAMMPS binary is required to materialize MD/em.lmp from the EM restart. "
        "Pass --lmp-binary or run inside the conda env that provides `lmp`."
    )


def _ensure_lammps_em_data(lmp_proj: Path, *, lmp_binary: str | None = None) -> Path:
    md_dir = lmp_proj / "MD"
    em_data = md_dir / "em.lmp"
    if em_data.exists() and em_data.stat().st_size > 0:
        return em_data

    em_restart = md_dir / "equil_stage01_em.restart"
    if not em_restart.exists():
        raise FileNotFoundError(
            "GROMACS parity bridge must start from the LAMMPS EM endpoint, but neither "
            f"{em_data} nor {em_restart} exists. Run the sibling lammps_cpu lane through "
            "equil_stage01_em first."
        )

    lmp = _resolve_lammps_binary(lmp_binary)
    script = md_dir / "write_em_data_from_restart.in"
    script.write_text(
        "\n".join(
            [
                "echo both",
                "units real",
                "boundary p p p",
                "atom_style full",
                "read_restart equil_stage01_em.restart",
                "kspace_style pppm 0.0001",
                "write_data em.lmp",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log = md_dir / "write_em_data_from_restart.stdout.log"
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run([lmp, "-nonbuf", "-in", script.name], cwd=str(md_dir), stdout=handle, stderr=subprocess.STDOUT, check=True)
    if not em_data.exists() or em_data.stat().st_size <= 0:
        raise FileNotFoundError(f"LAMMPS restart-to-data conversion finished but did not create {em_data}")
    return em_data


_LAMMPS_THERMO_HEADER_RE = re.compile(r"^\s*Step\s+(?:v_time|Time)\b")
_LAMMPS_THERMO_ROW_RE = re.compile(
    r"^\s*([0-9]+)\s+([0-9.eE+-]+)(?:\s|$)"
)


def _lammps_production_log_endpoint(log_path: Path) -> tuple[float | None, bool]:
    """Return the largest production time in fs and normal-exit marker state."""

    if not log_path.is_file():
        return None, False
    max_time_fs: float | None = None
    in_thermo = False
    normal_exit = False
    with log_path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if _LAMMPS_THERMO_HEADER_RE.match(line):
                in_thermo = True
                continue
            if "Total wall time:" in line:
                normal_exit = True
            if not in_thermo:
                continue
            if line.lstrip().startswith("Loop time of"):
                in_thermo = False
                continue
            match = _LAMMPS_THERMO_ROW_RE.match(line)
            if match is None:
                continue
            try:
                time_fs = float(match.group(2))
            except ValueError:
                continue
            if math.isfinite(time_fs):
                max_time_fs = (
                    time_fs if max_time_fs is None else max(max_time_fs, time_fs)
                )
    return max_time_fs, normal_exit


def _declared_lammps_production_ns(lmp_proj: Path) -> float | None:
    for metadata_path in (
        lmp_proj / "prepared_lammps_inputs.json",
        lmp_proj / "MD/meta.json",
    ):
        if not metadata_path.is_file():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            value = float(payload["production_total_ns"])
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return None


def _assert_sibling_lammps_lane_ok(
    lmp_proj: Path,
    traj_id: int,
    *,
    expected_production_ns: float,
) -> None:
    """Require terminal sibling LAMMPS production evidence before bridging.

    A lane-level ``batch_status.csv`` is written only after every trajectory in
    that worker has returned.  Therefore a completed leading trajectory in an
    active remote batch legitimately has no status row yet.  It may proceed
    only when its own production log proves the requested endpoint and the
    terminal restart exists.  Conversely, a stale ``status=ok`` row alone is
    never completion evidence.
    """

    try:
        expected_ns = float(expected_production_ns)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid expected LAMMPS production duration: {expected_production_ns!r}"
        ) from exc
    if not math.isfinite(expected_ns) or expected_ns <= 0.0:
        raise ValueError(
            f"Expected LAMMPS production duration must be positive: {expected_ns!r}"
        )

    status_path = lmp_proj.parent / "batch_status.csv"
    if status_path.exists():
        with status_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        matched = next(
            (
                row
                for row in rows
                if batch_project_id(row.get("Trajectory ID"), -1) == traj_id
            ),
            None,
        )
        if matched is None:
            raise RuntimeError(
                f"Refusing GROMACS bridge for Traj_{traj_id}: {status_path} exists "
                "but has no matching trajectory row"
            )
        status = str(matched.get("status", "")).strip().lower()
        if status != "ok":
            failed_phase = str(matched.get("failed_phase", "")).strip()
            error = str(matched.get("error", "")).strip()
            raise RuntimeError(
                f"Refusing GROMACS bridge for Traj_{traj_id}: sibling LAMMPS lane "
                f"is not ok (status={matched.get('status')!r}, "
                f"failed_phase={failed_phase!r}, error={error[:500]!r})"
            )

    final_restart = lmp_proj / "MD/final.restart"
    if not final_restart.is_file() or final_restart.stat().st_size <= 0:
        raise RuntimeError(
            f"Refusing GROMACS bridge for Traj_{traj_id}: missing non-empty "
            f"LAMMPS terminal restart {final_restart}"
        )

    expected_time_fs = expected_ns * 1.0e6
    stage_log = lmp_proj / "logs/lammps_prod_stage00_nvt_cpu.log"
    # The dedicated stage log is authoritative when present.  Do not let an
    # older aggregate MD/log.lammps mask an incomplete production rerun.
    log_candidates = (
        (stage_log,)
        if stage_log.is_file()
        else (lmp_proj / "MD/log.lammps",)
    )
    seen_logs: list[str] = []
    latest_log_mtime_ns = -1
    for log_path in log_candidates:
        if not log_path.is_file():
            continue
        latest_log_mtime_ns = max(latest_log_mtime_ns, log_path.stat().st_mtime_ns)
        endpoint_fs, normal_exit = _lammps_production_log_endpoint(log_path)
        seen_logs.append(
            f"{log_path}:endpoint_fs={endpoint_fs!r},normal_exit={normal_exit}"
        )
        if (
            normal_exit
            and endpoint_fs is not None
            and endpoint_fs + 0.5 >= expected_time_fs
        ):
            return

    completion_flags = (
        lmp_proj / "MD/production_complete.flag",
        lmp_proj / "MD/lammps_production_complete.flag",
    )
    declared_ns = _declared_lammps_production_ns(lmp_proj)
    for flag_path in completion_flags:
        if not flag_path.is_file() or flag_path.stat().st_size <= 0:
            continue
        # A newer production log can invalidate a stale completion flag when a
        # trajectory has been restarted.  Accept a flag only when it is at
        # least as recent as every production log that is present, and only
        # when persisted metadata proves which duration the flag represents.
        if (
            declared_ns is not None
            and declared_ns + 1.0e-12 >= expected_ns
            and flag_path.stat().st_mtime_ns >= latest_log_mtime_ns
        ):
            return

    evidence = "; ".join(seen_logs) if seen_logs else "no production log"
    raise RuntimeError(
        f"Refusing GROMACS bridge for Traj_{traj_id}: sibling LAMMPS production "
        f"has no terminal evidence for {expected_ns:g} ns ({evidence})"
    )


_LAMMPS_G_VECTOR_RE = re.compile(
    r"G vector \(1/distance\) =\s*([0-9.eE+-]+)"
)


def _first_lammps_g_vector(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    match = _LAMMPS_G_VECTOR_RE.search(
        log_path.read_text(encoding="utf-8", errors="ignore")
    )
    return float(match.group(1)) if match is not None else None


def _lammps_log_gromacs_stage_keys(log_path: Path) -> tuple[str, ...]:
    name = log_path.name
    if name == "lammps_lammps_equil_01_eq01_nvt_0p5fs_50ps_chunk0001_cpu.log":
        return ("eq01_soft_langevin_10ps", "eq01_nvt_40ps")
    if name == "lammps_lammps_equil_03_minimize_cpu.log":
        return ("eq03_pre_2fs_minimize",)
    if name == "lammps_lammps_equil_12_npt_avg_cell_1200ps_cpu.log":
        return ("eq12_npt_1200ps",)
    if name == "lammps_prod_stage00_nvt_cpu.log":
        return ("prod_nvt",)

    match = re.fullmatch(
        r"lammps_lammps_equil_(?:02|04|05|06|07|08|09|10|11|13)_"
        r"(eq\d{2}_.+?)_cpu\.log",
        name,
    )
    if match is None or "_retry" in name:
        return ()
    stage = match.group(1).replace("_0p5fs_", "_")
    if stage.startswith("eq02_npt_100ps_"):
        stage = stage.replace("eq02_npt_100ps_", "eq02_npt_compress_100ps_", 1)
    return (stage,)


def lammps_beta_stage_layouts(
    lmp_proj: Path,
    base_layouts: dict[str, dict[str, object]] | None,
) -> dict[str, dict[str, object]]:
    """Attach each trajectory's static LAMMPS PPPM beta to its GROMACS stage.

    LAMMPS reinitializes PPPM at the start of every generated stage.  A single
    host-level layout is therefore insufficient when several replicas have
    different cells.  Preserve explicit run-specific values and fill only
    missing values from the sibling trajectory's stage logs.
    """

    layouts = json.loads(json.dumps(base_layouts or {}))
    logs_dir = lmp_proj / "logs"
    derived: dict[str, float] = {}
    if logs_dir.is_dir():
        for log_path in sorted(logs_dir.glob("*.log")):
            stage_keys = _lammps_log_gromacs_stage_keys(log_path)
            if not stage_keys:
                continue
            beta = _first_lammps_g_vector(log_path)
            if beta is None:
                continue
            for stage_key in stage_keys:
                derived[stage_key] = beta

    for stage_key, beta in derived.items():
        entry = dict(layouts.get(stage_key, {}))
        stage_env = dict(entry.get("env", {}))
        stage_env.setdefault("GMX_PCFF_EWALD_BETA_INV_A", f"{beta:.9g}")
        entry["env"] = stage_env
        layouts[stage_key] = entry

    # A default beta is an explicit stage-layout decision and is valid
    # coverage.  Materialize it into any stage-specific entry that otherwise
    # shadows ``default`` in the runtime layout selector.
    default_entry = layouts.get("default", {})
    default_env = (
        dict(default_entry.get("env", {}))
        if isinstance(default_entry, dict)
        else {}
    )
    default_beta = default_env.get("GMX_PCFF_EWALD_BETA_INV_A")
    if default_beta is not None:
        for stage_key, raw_entry in list(layouts.items()):
            if stage_key == "default" or not isinstance(raw_entry, dict):
                continue
            entry = dict(raw_entry)
            stage_env = dict(entry.get("env", {}))
            stage_env.setdefault("GMX_PCFF_EWALD_BETA_INV_A", str(default_beta))
            entry["env"] = stage_env
            layouts[stage_key] = entry
    return layouts


def _layout_lammps_beta(
    layouts: object,
    stage_name: str,
) -> tuple[object | None, str]:
    if not isinstance(layouts, dict):
        return None, "missing layouts"
    raw = layouts.get(stage_name) or layouts.get("default")
    if not isinstance(raw, dict):
        return None, "missing stage/default layout"
    env = raw.get("env", {})
    if not isinstance(env, dict):
        return None, "layout env is not an object"
    value = env.get("GMX_PCFF_EWALD_BETA_INV_A")
    return value, "stage" if layouts.get(stage_name) else "default"


def _assert_gromacs_beta_coverage(runtime_ctx: dict[str, object]) -> None:
    """Fail before mdrun if any exact-rRESPA/CG stage lacks LAMMPS beta."""

    if str(runtime_ctx.get("schedule", "")) != "polygen_em_handoff":
        return
    layouts = runtime_ctx.get("gromacs_stage_layouts", {})
    stages = runtime_ctx.get("stages", [])
    if not isinstance(stages, list):
        raise RuntimeError("GROMACS runtime context has no generated stage list")

    missing: list[str] = []
    invalid: list[str] = []
    for raw_stage in stages:
        if not isinstance(raw_stage, dict):
            continue
        if str(raw_stage.get("kind", "")) not in {"md", "em"}:
            continue
        stage_name = str(raw_stage.get("name", "")).strip()
        value, source = _layout_lammps_beta(layouts, stage_name)
        if value is None or not str(value).strip():
            missing.append(stage_name or "<unnamed>")
            continue
        try:
            beta = float(value)
        except (TypeError, ValueError):
            invalid.append(f"{stage_name}={value!r}({source})")
            continue
        if not math.isfinite(beta) or beta <= 0.0:
            invalid.append(f"{stage_name}={value!r}({source})")

    if missing or invalid:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if invalid:
            details.append("invalid=" + ",".join(invalid))
        raise RuntimeError(
            "Refusing exact-rRESPA/CG GROMACS start: every generated stage "
            "requires a finite positive LAMMPS G-vector beta; "
            + "; ".join(details)
        )


def _synthetic_gromacs_prepare_report(
    *,
    proj: Path,
    gmx_binary: str,
    bridge_dir: Path,
) -> dict[str, object]:
    return {
        "schema_name": "polygen_multisystem_bridge_prepare_report",
        "schema_version": 1,
        "workflow": {
            "overall_status": "ready_for_md",
            "failure_reason": None,
            "preparation_mode": "lammps_data_bridge_from_lammps_lane_em_lmp",
        },
        "binary": {
            "batch_binary": str(gmx_binary),
            "gmxlib": os.environ.get("GMXLIB"),
        },
        "bridge": {
            "bridge_dir": str(bridge_dir),
            "source": "sibling lammps_cpu lane MD/em.lmp",
        },
        "outputs": {
            "conf_gro": str(proj / "MD_GMX/conf.gro"),
            "topol_top": str(proj / "MD_GMX/topol.top"),
        },
    }


def _patch_smoke_mdp_lengths(context: dict[str, object]) -> None:
    for stage in context.get("stages", []):
        if not isinstance(stage, dict):
            continue
        mdp_path = Path(str(stage["mdp_path"]))
        text = mdp_path.read_text(encoding="utf-8")
        if str(stage.get("kind", "")) == "em" or str(stage.get("name", "")) == "00_em":
            nsteps = 20
        else:
            nsteps = 4
        text = re_sub_line(text, "nsteps", f"nsteps                  = {nsteps}")
        mdp_path.write_text(text, encoding="utf-8")
        stage["nsteps"] = nsteps


def re_sub_line(text: str, key: str, replacement: str) -> str:
    import re

    pattern = rf"^(\s*{re.escape(key)}\s*=).*$"
    out, n = re.subn(pattern, replacement, text, count=1, flags=re.M)
    if n == 0:
        out += f"\n{replacement}\n"
    return out


def run_gromacs_bridge_lane(
    selected_csv: Path,
    *,
    workspace: Path,
    outdir: Path,
    lane: str,
    role: str,
    run_group: str,
    nproc: int,
    eqfactor: float,
    production_ns: float,
    resume_existing: bool,
    force_restart: bool,
    gmx_binary: str | None,
    lmp_binary: str | None = None,
    gromacs_stage_layouts: dict[str, dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pd = require_pandas()
    selected_df = pd.read_csv(selected_csv).reset_index(drop=True)
    output_root = outdir / "runs_batch" / run_group / role / lane
    output_root.mkdir(parents=True, exist_ok=True)
    if selected_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    settings = lane_mdrun_settings(lane, workspace, gmx_binary)
    binary = settings.get("GROMACS_BATCH_GMX_BINARY", "")
    if not binary:
        raise FileNotFoundError(f"No GROMACS binary found for lane={lane}")
    if not Path(binary).exists():
        raise FileNotFoundError(f"GROMACS binary does not exist for lane={lane}: {binary}")

    batch_dir = workspace / "MY_PAPER_RELATED/GROMACS_PCFF_BATCH"
    prepend_sys_path(batch_dir)
    from batch_utils.gromacs_analysis_utils import run_gromacs_analysis
    # Keep the validation protocol versioned with the patched engine.  The
    # surrounding batch workspace still supplies assembly/analysis helpers,
    # but stage generation and checkpoint semantics must come from this branch
    # so local and remote workers execute the same protocol after git pull.
    from polygen_gromacs_runtime import (
        run_gromacs_equilibration,
        run_gromacs_production,
        setup_gromacs_environment,
        write_gromacs_meta_json,
    )

    old_env = os.environ.copy()
    try:
        for key, value in settings.items():
            if value:
                os.environ[key] = value

        statuses: list[dict[str, object]] = []
        metrics: list[dict[str, object]] = []
        for row_idx, row in selected_df.iterrows():
            traj_id = batch_project_id(row.get("Trajectory ID"), row_idx)
            proj = output_root / f"Traj_{traj_id}"
            if force_restart and proj.exists():
                shutil.rmtree(proj)
            proj.mkdir(parents=True, exist_ok=True)
            failed_phase = ""
            skipped: list[str] = []
            try:
                replica_seed = int(row.get("replica_seed", traj_id))
                os.environ["GROMACS_BATCH_GEN_SEED"] = str(replica_seed)
                lmp_proj = lammps_lane_root(outdir, run_group, role) / f"Traj_{traj_id}"
                failed_phase = "lammps_source_status"
                _assert_sibling_lammps_lane_ok(
                    lmp_proj,
                    traj_id,
                    expected_production_ns=float(production_ns),
                )
                source_data = _ensure_lammps_em_data(lmp_proj, lmp_binary=lmp_binary)

                bridge_dir = proj / "build/lammps_data_bridge"
                md_dir = proj / "MD_GMX"
                md_dir.mkdir(parents=True, exist_ok=True)
                if (
                    resume_existing
                    and not force_restart
                    and (bridge_dir / "system.gro").exists()
                    and (bridge_dir / "topol.top").exists()
                ):
                    skipped.append("bridge")
                else:
                    failed_phase = "bridge"
                    _run_bridge_command(
                        workspace=workspace,
                        source_data=source_data,
                        bridge_out=bridge_dir,
                        traj_id=traj_id,
                    )

                for src_name, dst_name in (("system.gro", "conf.gro"), ("topol.top", "topol.top")):
                    src = bridge_dir / src_name
                    dst = md_dir / dst_name
                    if not src.exists():
                        raise FileNotFoundError(f"Bridge output missing: {src}")
                    if (not resume_existing) or force_restart or not dst.exists():
                        shutil.copy2(src, dst)

                failed_phase = "gromacs_setup"
                prepare_report = _synthetic_gromacs_prepare_report(
                    proj=proj,
                    gmx_binary=binary,
                    bridge_dir=bridge_dir,
                )
                report_path = proj / "build/gromacs_pcff/gromacs_pcff_prepare_report.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(prepare_report, indent=2) + "\n", encoding="utf-8")
                trajectory_stage_layouts = lammps_beta_stage_layouts(
                    lmp_proj,
                    dict(gromacs_stage_layouts or {}),
                )
                runtime_ctx = setup_gromacs_environment(
                    proj,
                    base_dir=batch_dir / "BASE",
                    prepare_report=prepare_report,
                    temperature_k=353.0,
                    eqfactor=float(eqfactor),
                    production_total_ns=float(production_ns),
                    nproc=int(nproc),
                    gromacs_stage_layouts=trajectory_stage_layouts,
                )
                failed_phase = "gromacs_beta_coverage"
                _assert_gromacs_beta_coverage(runtime_ctx)
                runtime_ctx["resume_existing_effective"] = bool(resume_existing and not force_restart)
                (md_dir / "gromacs_runtime_context.json").write_text(
                    json.dumps(runtime_ctx, indent=2) + "\n",
                    encoding="utf-8",
                )
                if float(production_ns) <= 0.01:
                    _patch_smoke_mdp_lengths(runtime_ctx)
                    (md_dir / "gromacs_runtime_context.json").write_text(
                        json.dumps(runtime_ctx, indent=2) + "\n",
                        encoding="utf-8",
                    )

                failed_phase = "gromacs_equil"
                if resume_existing and not force_restart and (md_dir / "equilibration_complete.flag").exists():
                    skipped.append("gromacs_equil")
                else:
                    run_gromacs_equilibration(runtime_ctx)

                failed_phase = "gromacs_prod"
                if resume_existing and not force_restart and (md_dir / "production_complete.flag").exists():
                    skipped.append("gromacs_prod")
                else:
                    run_gromacs_production(runtime_ctx)

                failed_phase = "meta"
                if resume_existing and not force_restart and (md_dir / "meta.json").exists():
                    skipped.append("meta")
                else:
                    write_gromacs_meta_json(
                        proj,
                        monomer_smiles=str(row.get("SMILES", "")),
                        placeholder="Br",
                        force_field="PCFF",
                        temperature_k=353.0,
                        production_total_ns=float(production_ns),
                        prepare_report=prepare_report,
                        runtime_context=runtime_ctx,
                    )

                failed_phase = "analysis"
                if float(production_ns) <= 0.01:
                    skipped.append("analysis_smoke_short_window")
                    metrics.append(
                        {
                            "Trajectory ID": traj_id,
                            "sigma_NE_htpmd_S_cm": float("nan"),
                            "c_tn_htpmd": float("nan"),
                            "D_Li_cm2s": float("nan"),
                            "D_an_cm2s": float("nan"),
                            "bucket": row.get("_bucket"),
                            "rank_value": row.get("CONDUCTIVITY"),
                            "analysis_note": "skipped for smoke window",
                        }
                    )
                else:
                    analysis_report = run_gromacs_analysis(
                        proj,
                        analysis_begin_ns=0.0,
                        analysis_end_ns=float(production_ns),
                        temperature_k=353.0,
                        resume_existing_effective=bool(resume_existing and not force_restart),
                    )
                    metrics.append(
                        {
                            "Trajectory ID": traj_id,
                            "sigma_NE_htpmd_S_cm": analysis_report["conductivity"]["sigma_NE_htpmd_S_cm"],
                            "c_tn_htpmd": analysis_report["conductivity"]["c_tn_htpmd"],
                            "D_Li_cm2s": analysis_report["diffusion"]["D_Li_cm2s"],
                            "D_an_cm2s": analysis_report["diffusion"]["D_an_cm2s"],
                            "bucket": row.get("_bucket"),
                            "rank_value": row.get("CONDUCTIVITY"),
                        }
                    )
                statuses.append(
                    {
                        "Trajectory ID": traj_id,
                        "status": "ok",
                        "project_dir": str(proj),
                        "bucket": row.get("_bucket"),
                        "rank_value": row.get("CONDUCTIVITY"),
                        "skipped_phases": ",".join(skipped),
                    }
                )
                print(f"=== GROMACS bridge {lane} Traj_{traj_id} done ===")
            except Exception as exc:
                statuses.append(
                    {
                        "Trajectory ID": traj_id,
                        "status": "failed",
                        "failed_phase": failed_phase,
                        "project_dir": str(proj),
                        "bucket": row.get("_bucket"),
                        "rank_value": row.get("CONDUCTIVITY"),
                        "skipped_phases": ",".join(skipped),
                        "error": repr(exc),
                    }
                )
                print(f"=== GROMACS bridge {lane} Traj_{traj_id} failed at {failed_phase}: {exc!r} ===")

        status_df = pd.DataFrame(statuses)
        metrics_df = pd.DataFrame(metrics)
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    status_df.to_csv(output_root / "batch_status.csv", index=False)
    metrics_df.to_csv(output_root / "batch_metrics.csv", index=False)
    print("status:", output_root / "batch_status.csv")
    print("metrics:", output_root / "batch_metrics.csv")
    return status_df, metrics_df


def role_jobs(jobs_csv: Path, role: str, lane: str | None, run_group: str | None) -> pd.DataFrame:
    pd = require_pandas()
    jobs = pd.read_csv(jobs_csv)
    out = jobs.copy()
    if role != "all":
        out = out[out["worker_role"] == role]
    if lane:
        out = out[out["lane"] == lane]
    if run_group:
        out = out[out["run_group"] == run_group]
    return out.reset_index(drop=True)


def selected_rows_for_lane(
    jobs: pd.DataFrame,
    systems_csv: Path,
    *,
    molality_override: float | None = None,
) -> pd.DataFrame:
    pd = require_pandas()
    systems = pd.read_csv(systems_csv)
    keep = (
        jobs[
            [
                "run_group",
                "duration_ns",
                "system_key",
                "replica",
                "replica_seed",
                "trajectory_id",
                "category",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    merged = keep.merge(systems, on=["system_key", "category"], how="left", suffixes=("", "_system"))
    if merged["SMILES"].isna().any():
        missing = merged[merged["SMILES"].isna()]["system_key"].tolist()
        raise RuntimeError(f"systems.csv is missing rows for system_key={missing}")

    out = pd.DataFrame()
    out["Trajectory ID"] = merged["trajectory_id"].astype(int) * 100 + merged["replica"].astype(int)
    out["Source Trajectory ID"] = merged["trajectory_id"].astype(int)
    out["system_key"] = merged["system_key"]
    out["replica"] = merged["replica"].astype(int)
    out["replica_seed"] = merged["replica_seed"].astype(int)
    out["run_group"] = merged["run_group"]
    out["duration_ns"] = merged["duration_ns"].astype(float)
    out["SMILES"] = merged["SMILES"].astype(str)
    out["CONDUCTIVITY"] = pd.to_numeric(merged["CONDUCTIVITY"], errors="coerce")
    out["_bucket"] = merged["category"]
    for col, default in (
        ("Degree of Polymerization", 19),
        ("Molality", 1.45),
        ("Density", 1.0),
    ):
        out[col] = pd.to_numeric(merged.get(col), errors="coerce").fillna(default)
    if molality_override is not None:
        out["Source Molality"] = out["Molality"]
        out["Molality"] = float(molality_override)
        out["Molality Override"] = float(molality_override)
    return out


def choose_smoke_jobs(jobs: pd.DataFrame, systems_csv: Path) -> pd.DataFrame:
    """Pick a cheap representative row for smoke tests.

    The manifest order is optimized for study partitioning, not for short
    validation.  SMILES length is kept first because pysoftk setup cost can
    dominate smoke tests; DP/density only break ties among similarly cheap
    candidates.
    """
    pd = require_pandas()
    if jobs.empty or not systems_csv.exists():
        return jobs.drop_duplicates(["system_key", "replica", "lane"]).head(1)
    systems = pd.read_csv(systems_csv)
    wanted_cols = [
        "system_key",
        "category",
        "SMILES",
        "Degree of Polymerization",
        "Density",
        "Molality",
    ]
    systems_cols = [col for col in wanted_cols if col in systems.columns]
    merged = jobs.merge(
        systems[systems_cols],
        on=["system_key", "category"],
        how="left",
    )
    merged["_smiles_len"] = merged["SMILES"].fillna("").astype(str).str.len()
    merged["_dp"] = pd.to_numeric(merged.get("Degree of Polymerization"), errors="coerce").fillna(9999.0)
    merged["_density"] = pd.to_numeric(merged.get("Density"), errors="coerce").fillna(9999.0)
    merged["_molality"] = pd.to_numeric(merged.get("Molality"), errors="coerce").fillna(9999.0)
    merged = merged.sort_values(
        ["_smiles_len", "_dp", "_density", "_molality", "system_key", "replica"]
    ).drop(columns=["SMILES", "_smiles_len", "_dp", "_density", "_molality"])
    return merged.drop_duplicates(["system_key", "replica", "lane"]).head(1)


def write_selected_csv(
    jobs_csv: Path,
    systems_csv: Path,
    outdir: Path,
    *,
    role: str,
    lane: str,
    run_group: str | None,
    molality_override: float | None = None,
) -> Path:
    jobs = role_jobs(jobs_csv, role, lane, run_group)
    selected_dir = outdir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    group_label = run_group or "allgroups"
    path = selected_dir / f"selected_{role}_{group_label}_{lane}.csv"
    if lane in {"gmx_cpu", "gmx_gpu"}:
        lammps_path = selected_dir / f"selected_{role}_{group_label}_lammps_cpu.csv"
        if lammps_path.exists():
            selected = pd.read_csv(lammps_path)
            if molality_override is not None:
                selected = selected.copy()
                selected["Molality"] = float(molality_override)
                selected["Molality Override"] = float(molality_override)
            selected.to_csv(path, index=False)
            print(path)
            return path
    selected = selected_rows_for_lane(jobs, systems_csv, molality_override=molality_override)
    selected.to_csv(path, index=False)
    print(path)
    return path


def bootstrap_plan(workspace: Path) -> str:
    repo = workspace / "GROMACS_PCFF"
    return "\n".join(
        [
            "# Run inside the conda env you prepared for this worker.",
            "# Minimal remote package set when LAMMPS input projects were prepared locally and rsynced:",
            "conda install -y -c conda-forge numpy pandas scipy tqdm matplotlib nbformat ipykernel cmake ninja make pkg-config compilers fftw openmpi hwloc gsl docopt mdtraj pymatgen",
            "# Only install these on a worker if it must regenerate pysoftk/LUNAR/Packmol inputs itself:",
            "conda install -y -c conda-forge rdkit openbabel packmol",
            "python -m pip install pysoftk",
            f"cd {repo}",
            "cmake -S . -B build_gateb_double_cpu -DGMX_DOUBLE=ON -DGMX_OPENMP=ON -DGMX_GPU=OFF -DGMX_MPI=OFF -DGMX_BUILD_OWN_FFTW=ON -DGMX_BUILD_UNITTESTS=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release",
            "cmake --build build_gateb_double_cpu -j4",
            "# GPU build requires a working CUDA toolkit/compiler on that machine.",
            "cmake -S . -B build_gateb_cuda -DGMX_DOUBLE=OFF -DGMX_OPENMP=ON -DGMX_GPU=CUDA -DGMX_MPI=OFF -DGMX_BUILD_OWN_FFTW=ON -DGMX_BUILD_UNITTESTS=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release",
            "cmake --build build_gateb_cuda -j4",
        ]
    )


def doctor(workspace: Path) -> dict[str, object]:
    gmx_cpu = default_gmx_binary(workspace, "gmx_cpu")
    gmx_gpu = default_gmx_binary(workspace, "gmx_gpu")
    result = {
        "workspace": str(workspace),
        "executables": {
            name: which(name)
            for name in ["python", "python3", "packmol", "obabel", "lmp", "mpirun", "mpiexec", "cmake", "ninja", "nvidia-smi"]
        },
        "nvidia_smi_detected": detect_nvidia_smi(),
        "python_modules": {
            name: check_import(name)
            for name in ["numpy", "pandas", "scipy", "tqdm", "rdkit", "pysoftk", "nbformat", "docopt", "mdtraj", "pymatgen"]
        },
        "gromacs_binaries": {
            "gmx_cpu": gmx_cpu,
            "gmx_gpu": gmx_gpu,
        },
        "batch_dirs": {
            "lammps": str(workspace / "MY_PAPER_RELATED/LAMMPS_BATCH"),
            "gromacs": str(workspace / "MY_PAPER_RELATED/GROMACS_PCFF_BATCH"),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _module_ok(module: str) -> bool:
    return check_import(module) == "ok"


def _binary_ok(path_or_name: str | None) -> bool:
    if not path_or_name:
        return False
    path = Path(path_or_name).expanduser()
    if path.is_file():
        return True
    return shutil.which(path_or_name) is not None


def _selected_csv_path(outdir: Path, role: str, run_group: str, lane: str) -> Path:
    return outdir / "selected" / f"selected_{role}_{run_group}_{lane}.csv"


def preflight(
    *,
    workspace: Path,
    outdir: Path,
    jobs_csv: Path | None = None,
    systems_csv: Path | None = None,
    role: str,
    run_group: str,
    lanes: list[str],
    lmp_binary: str | None = None,
    mpirun_binary: str | None = None,
    gmx_cpu_binary: str | None = None,
    gmx_gpu_binary: str | None = None,
) -> dict[str, object]:
    """Check whether a worker is ready to run the selected lane jobs."""
    pd = require_pandas()
    jobs_csv = (
        Path(jobs_csv).expanduser().resolve()
        if jobs_csv is not None
        else outdir / "manifest/jobs.csv"
    )
    systems_csv = (
        Path(systems_csv).expanduser().resolve()
        if systems_csv is not None
        else outdir / "manifest/systems.csv"
    )

    checks: list[dict[str, object]] = []

    def add(name: str, ok: bool, *, severity: str = "error", detail: object = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "severity": severity, "detail": detail})

    add("workspace_has_GROMACS_PCFF", (workspace / "GROMACS_PCFF").is_dir(), detail=str(workspace / "GROMACS_PCFF"))
    add("workspace_has_MY_PAPER_RELATED", (workspace / "MY_PAPER_RELATED").is_dir(), detail=str(workspace / "MY_PAPER_RELATED"))
    add("jobs_csv_exists", jobs_csv.is_file(), detail=str(jobs_csv))
    add("systems_csv_exists", systems_csv.is_file(), detail=str(systems_csv))
    add("bridge_script_exists", bridge_script(workspace).is_file())

    if jobs_csv.is_file():
        jobs = pd.read_csv(jobs_csv)
    else:
        jobs = pd.DataFrame()

    lane_details: dict[str, object] = {}
    for lane in lanes:
        lane_jobs = role_jobs(jobs_csv, role, lane, run_group) if jobs_csv.is_file() else pd.DataFrame()
        selected_csv = _selected_csv_path(outdir, role, run_group, lane)
        selected_exists = selected_csv.is_file()
        selected_rows = None
        replica_values: list[int] = []
        if selected_exists:
            selected = pd.read_csv(selected_csv)
            selected_rows = int(len(selected))
            if "replica" in selected:
                replica_values = sorted(int(x) for x in selected["replica"].dropna().unique().tolist())
        expected_rows = int(
            lane_jobs[
                ["system_key", "replica", "lane"]
            ].drop_duplicates().shape[0]
        ) if not lane_jobs.empty else 0

        add(
            f"{lane}_selected_csv",
            selected_exists and selected_rows == expected_rows and expected_rows > 0,
            detail={
                "path": str(selected_csv),
                "selected_rows": selected_rows,
                "expected_rows": expected_rows,
                "replicas": replica_values,
            },
        )
        lane_details[lane] = {
            "selected_csv": str(selected_csv),
            "selected_rows": selected_rows,
            "expected_rows": expected_rows,
        }

    if "lammps_cpu" in lanes:
        lmp = lmp_binary or shutil.which("lmp")
        mpi = mpirun_binary or shutil.which("mpirun") or shutil.which("mpiexec")
        add("lammps_binary", _binary_ok(lmp), detail=lmp)
        add(
            "mpi_launcher",
            _binary_ok(mpi),
            severity="warning",
            detail={
                "selected": mpi,
                "mpirun": shutil.which("mpirun"),
                "mpiexec": shutil.which("mpiexec"),
                "note": "LAMMPS smoke can run serial without MPI, but production performance needs MPI.",
            },
        )
        for module in ["numpy", "pandas", "scipy", "tqdm", "rdkit", "pysoftk", "docopt", "mdtraj", "pymatgen"]:
            add(f"python_module_{module}", _module_ok(module), detail=check_import(module))
        add("openbabel_or_obabel", bool(shutil.which("obabel") or _module_ok("openbabel")), detail={"obabel": shutil.which("obabel"), "openbabel": check_import("openbabel")})

    if "gmx_cpu" in lanes:
        gmx_cpu = gmx_cpu_binary or default_gmx_binary(workspace, "gmx_cpu")
        add("gmx_cpu_binary", _binary_ok(gmx_cpu), detail=gmx_cpu)
        add("gmx_cpu_after_lammps_note", True, severity="info", detail="gmx_cpu bridge requires sibling lammps_cpu MD/em.lmp; run lammps_cpu through EM first.")

    if "gmx_gpu" in lanes:
        gmx_gpu = gmx_gpu_binary or default_gmx_binary(workspace, "gmx_gpu")
        nvidia_smi = detect_nvidia_smi()
        cuda_device_hint = any(Path(path).exists() for path in ("/dev/nvidia0", "/dev/nvidiactl", "/dev/dxg"))
        add("gmx_gpu_binary", _binary_ok(gmx_gpu), detail=gmx_gpu)
        add(
            "cuda_visibility",
            nvidia_smi is not None or cuda_device_hint,
            severity="warning",
            detail={"nvidia_smi": nvidia_smi, "device_hint": cuda_device_hint},
        )
        add("gmx_gpu_after_lammps_note", True, severity="info", detail="gmx_gpu bridge requires sibling lammps_cpu MD/em.lmp; run lammps_cpu through EM first.")

    ok = all(c["ok"] or c["severity"] in {"warning", "info"} for c in checks)
    result = {
        "ok": ok,
        "workspace": str(workspace),
        "outdir": str(outdir),
        "role": role,
        "run_group": run_group,
        "lanes": lanes,
        "lane_details": lane_details,
        "checks": checks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_batch_lane(
    selected_csv: Path,
    *,
    workspace: Path,
    outdir: Path,
    lane: str,
    role: str,
    run_group: str,
    nproc: int,
    eqfactor: float,
    production_ns: float,
    resume_existing: bool,
    force_restart: bool,
    gmx_binary: str | None,
    lmp_binary: str | None,
    mpirun_binary: str | None = None,
    lammps_cpu_openmp: bool = False,
    lammps_cpu_mpi_ranks: int | None = None,
    lammps_cpu_omp_threads: int = 1,
    lammps_thermo_flush: bool = True,
    lammps_resume_from_checkpoint: bool = False,
    lammps_stop_after_stage: str | None = None,
    lammps_stage_layouts: dict[str, dict[str, object]] | None = None,
    gromacs_stage_layouts: dict[str, dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pd = require_pandas()
    selected_df = pd.read_csv(selected_csv)
    if selected_df.empty:
        print("No selected rows for this lane.")
        return pd.DataFrame(), pd.DataFrame()

    output_root = outdir / "runs_batch" / run_group / role / lane
    analysis_begin_ns = 0.0
    analysis_end_ns = float(production_ns)

    if lane == "lammps_cpu":
        lammps_cpu_mpi_ranks = int(lammps_cpu_mpi_ranks or (1 if lammps_cpu_openmp else nproc))
        lammps_cpu_omp_threads = int(max(1, lammps_cpu_omp_threads))
        print(
            "[lammps_cpu] cfg layout:",
            f"openmp={bool(lammps_cpu_openmp)}",
            f"mpi_ranks={lammps_cpu_mpi_ranks}",
            f"omp_threads={lammps_cpu_omp_threads}",
            f"stage_layouts={sorted((lammps_stage_layouts or {}).keys())}",
        )
        batch_dir = workspace / "MY_PAPER_RELATED/LAMMPS_BATCH"
        prepend_sys_path(batch_dir)
        from batch_utils import BatchRunConfig, run_batch_pipeline

        cfg_kwargs = {
            "base_dir": batch_dir / "BASE",
            "lunar_dir": batch_dir / "extern/LUNAR",
            "output_root": output_root,
            "production_total_ns": float(production_ns),
            "analysis_begin_ns": analysis_begin_ns,
            "analysis_end_ns": analysis_end_ns,
            "nproc": int(nproc),
            "use_kokkos": False,
            "resume_existing": bool(resume_existing),
            "force_restart": bool(force_restart),
            "scan_incomplete_only": False,
            "eqfactor": float(eqfactor),
            "lammps_binary": str(Path(lmp_binary).expanduser().resolve()) if lmp_binary else None,
            "mpirun_binary": str(Path(mpirun_binary).expanduser().resolve()) if mpirun_binary else None,
            "lammps_thermo_flush": bool(lammps_thermo_flush),
            "lammps_resume_from_checkpoint": bool(lammps_resume_from_checkpoint),
            "lammps_stop_after_stage": str(lammps_stop_after_stage).strip() if lammps_stop_after_stage else None,
            "lammps_cpu_openmp": bool(lammps_cpu_openmp),
            "lammps_cpu_mpi_ranks": int(lammps_cpu_mpi_ranks),
            "lammps_cpu_omp_threads": int(lammps_cpu_omp_threads),
            "lammps_stage_layouts": dict(lammps_stage_layouts or {}),
        }
        supported_cfg_args = set(inspect.signature(BatchRunConfig).parameters)
        skipped_cfg_args = sorted(k for k in cfg_kwargs if k not in supported_cfg_args)
        if skipped_cfg_args:
            print(f"[lammps_cpu] BatchRunConfig does not support args; skipping: {skipped_cfg_args}")
        cfg = BatchRunConfig(**{k: v for k, v in cfg_kwargs.items() if k in supported_cfg_args})
    elif lane in {"gmx_cpu", "gmx_gpu"}:
        return run_gromacs_bridge_lane(
            selected_csv,
            workspace=workspace,
            outdir=outdir,
            lane=lane,
            role=role,
            run_group=run_group,
            nproc=nproc,
            eqfactor=eqfactor,
            production_ns=production_ns,
            resume_existing=resume_existing,
            force_restart=force_restart,
            gmx_binary=gmx_binary,
            lmp_binary=lmp_binary,
            gromacs_stage_layouts=gromacs_stage_layouts,
        )
    else:
        raise ValueError(f"Unsupported lane: {lane}")

    status_df, metrics_df = run_batch_pipeline(selected_df, config=cfg, rank_col="CONDUCTIVITY")

    output_root.mkdir(parents=True, exist_ok=True)
    status_df.to_csv(output_root / "batch_status.csv", index=False)
    metrics_df.to_csv(output_root / "batch_metrics.csv", index=False)
    print("status:", output_root / "batch_status.csv")
    print("metrics:", output_root / "batch_metrics.csv")
    return status_df, metrics_df


def raise_for_failed_lane_status(status_df: object, lane: str) -> None:
    if getattr(status_df, "empty", True) or "status" not in status_df.columns:
        return
    failed_rows = status_df[status_df["status"].astype(str).str.lower() == "failed"]
    if failed_rows.empty:
        return
    failed_ids = ",".join(str(value) for value in failed_rows["Trajectory ID"].tolist())
    print(
        f"run-lane failed: lane={lane} count={len(failed_rows)} trajectory_ids={failed_ids}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common: dict[str, object] = {}
    for name in ["doctor", "bootstrap-plan", "preflight", "make-selected", "prepare-inputs", "run-lane", "smoke"]:
        sp = sub.add_parser(name)
        sp.add_argument("--workspace", type=Path, default=None)
        sp.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
        if name in {"preflight", "make-selected", "prepare-inputs", "run-lane", "smoke"}:
            sp.add_argument("--jobs-csv", type=Path, default=DEFAULT_MANIFEST)
            sp.add_argument("--systems-csv", type=Path, default=DEFAULT_SYSTEMS)
            sp.add_argument("--role", default="local_main")
            sp.add_argument("--run-group", default="main20")
        if name in {"make-selected", "prepare-inputs", "run-lane", "smoke"}:
            sp.add_argument("--lane", choices=LANES, default="lammps_cpu")
            sp.add_argument(
                "--molality-override",
                type=float,
                default=None,
                help=(
                    "Override selected-row Molality before building inputs. "
                    "Use this to force one molality basis across a validation cohort "
                    "without editing the source CSV."
                ),
            )
        if name == "preflight":
            sp.add_argument("--lanes", default=",".join(LANES))
            sp.add_argument("--gmx-cpu-binary", default=None)
            sp.add_argument("--gmx-gpu-binary", default=None)
            sp.add_argument("--lmp-binary", default=None)
            sp.add_argument("--mpirun-binary", default=None)
        if name in {"prepare-inputs", "run-lane", "smoke"}:
            sp.add_argument("--nproc", type=int, default=4)
            sp.add_argument("--eqfactor", type=float, default=0.83)
            sp.add_argument("--production-ns", type=float, default=None)
            sp.add_argument("--resume-existing", action="store_true")
            sp.add_argument("--force-restart", action="store_true")
            sp.add_argument("--gmx-binary", default=None)
            sp.add_argument("--lmp-binary", default=None)
            sp.add_argument("--mpirun-binary", default=None)
            sp.add_argument("--lammps-cpu-openmp", action="store_true")
            sp.add_argument("--lammps-cpu-mpi-ranks", type=int, default=None)
            sp.add_argument("--lammps-cpu-omp-threads", type=int, default=1)
            sp.add_argument("--lammps-stage-layout-file", type=Path, default=None)
            sp.add_argument("--gromacs-stage-layout-file", type=Path, default=None)
            sp.add_argument("--no-lammps-thermo-flush", action="store_true")
            sp.add_argument("--lammps-resume-from-checkpoint", action="store_true")
            sp.add_argument(
                "--lammps-stop-after-stage",
                default=None,
                help="Stop a LAMMPS CPU lane after this exact equilibration stage and skip production/analysis.",
            )
            sp.add_argument("--max-systems", type=int, default=None)
        if name in {"make-selected", "prepare-inputs"}:
            sp.add_argument(
                "--smoke-select",
                action="store_true",
                help="Select the same cheap representative subset used by the smoke command without running the lane.",
            )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve() if args.workspace else repo_workspace_root(Path.cwd())
    if hasattr(args, "jobs_csv") and args.jobs_csv == DEFAULT_MANIFEST:
        args.jobs_csv = args.outdir / "manifest/jobs.csv"
    if hasattr(args, "systems_csv") and args.systems_csv == DEFAULT_SYSTEMS:
        args.systems_csv = args.outdir / "manifest/systems.csv"

    if args.cmd == "doctor":
        doctor(workspace)
        return
    if args.cmd == "bootstrap-plan":
        print(bootstrap_plan(workspace))
        return
    if args.cmd == "preflight":
        lanes = [x.strip() for x in str(args.lanes).split(",") if x.strip()]
        invalid = [x for x in lanes if x not in LANES]
        if invalid:
            raise ValueError(f"Invalid lanes: {invalid}; valid={LANES}")
        result = preflight(
            workspace=workspace,
            outdir=args.outdir,
            jobs_csv=args.jobs_csv,
            systems_csv=args.systems_csv,
            role=args.role,
            run_group=args.run_group,
            lanes=lanes,
            lmp_binary=args.lmp_binary,
            mpirun_binary=args.mpirun_binary,
            gmx_cpu_binary=args.gmx_cpu_binary,
            gmx_gpu_binary=args.gmx_gpu_binary,
        )
        if not result["ok"]:
            raise SystemExit(1)
        return

    if args.cmd in {"make-selected", "prepare-inputs", "run-lane", "smoke"}:
        jobs = role_jobs(args.jobs_csv, args.role, args.lane, args.run_group)
        smoke_selection = args.cmd == "smoke" or bool(getattr(args, "smoke_select", False))
        smoke_run_group = str(args.run_group)
        if smoke_selection:
            smoke_run_group = smoke_run_group if smoke_run_group.startswith("smoke_") else f"smoke_{smoke_run_group}"
            jobs = choose_smoke_jobs(jobs, args.systems_csv)
            jobs = jobs.copy()
            jobs["run_group"] = smoke_run_group
            smoke_jobs = args.outdir / "manifest" / f"smoke_{args.role}_{args.run_group}_{args.lane}.csv"
            smoke_jobs.parent.mkdir(parents=True, exist_ok=True)
            jobs.to_csv(smoke_jobs, index=False)
            jobs_csv = smoke_jobs
        elif getattr(args, "max_systems", None):
            keys = jobs["system_key"].drop_duplicates().head(int(args.max_systems)).tolist()
            jobs = jobs[jobs["system_key"].isin(keys)]
            tmp_jobs = args.outdir / "manifest" / f"subset_{args.role}_{args.run_group}_{args.lane}.csv"
            tmp_jobs.parent.mkdir(parents=True, exist_ok=True)
            jobs.to_csv(tmp_jobs, index=False)
            jobs_csv = tmp_jobs
        else:
            jobs_csv = args.jobs_csv

        selected_run_group = smoke_run_group if smoke_selection else args.run_group
        selected_csv = write_selected_csv(
            jobs_csv,
            args.systems_csv,
            args.outdir,
            role=args.role,
            lane=args.lane,
            run_group=selected_run_group,
            molality_override=getattr(args, "molality_override", None),
        )
        if args.cmd == "make-selected":
            return

        production_ns = args.production_ns
        if production_ns is None:
            production_ns = 0.002 if smoke_selection else float(jobs["duration_ns"].max())
        eqfactor = (
            float(os.environ.get("POLYGEN_SMOKE_EQFACTOR", str(DEFAULT_SMOKE_EQFACTOR)))
            if smoke_selection
            else float(args.eqfactor)
        )
        if args.cmd == "prepare-inputs":
            if args.lane != "lammps_cpu":
                raise ValueError("prepare-inputs currently prepares the shared lammps_cpu project only")
            prepare_lammps_input_projects(
                selected_csv,
                workspace=workspace,
                outdir=args.outdir,
                role=args.role,
                run_group=selected_run_group,
                nproc=args.nproc,
                eqfactor=eqfactor,
                production_ns=float(production_ns),
                resume_existing=bool(args.resume_existing),
                force_restart=bool(args.force_restart),
                lmp_binary=args.lmp_binary,
                mpirun_binary=args.mpirun_binary,
            )
            return
        output_run_group = smoke_run_group if smoke_selection else args.run_group
        lammps_stage_layouts = load_lammps_stage_layouts(args.lammps_stage_layout_file)
        gromacs_stage_layouts = load_gromacs_stage_layouts(args.gromacs_stage_layout_file)
        status_df, _ = run_batch_lane(
            selected_csv,
            workspace=workspace,
            outdir=args.outdir,
            lane=args.lane,
            role=args.role,
            run_group=output_run_group,
            nproc=args.nproc,
            eqfactor=eqfactor,
            production_ns=float(production_ns),
            resume_existing=bool(args.resume_existing),
            force_restart=bool(args.force_restart),
            gmx_binary=args.gmx_binary,
            lmp_binary=args.lmp_binary,
            mpirun_binary=args.mpirun_binary,
            lammps_cpu_openmp=bool(args.lammps_cpu_openmp),
            lammps_cpu_mpi_ranks=args.lammps_cpu_mpi_ranks,
            lammps_cpu_omp_threads=args.lammps_cpu_omp_threads,
            lammps_thermo_flush=not bool(args.no_lammps_thermo_flush),
            lammps_resume_from_checkpoint=bool(args.lammps_resume_from_checkpoint),
            lammps_stop_after_stage=args.lammps_stop_after_stage,
            lammps_stage_layouts=lammps_stage_layouts,
            gromacs_stage_layouts=gromacs_stage_layouts,
        )
        raise_for_failed_lane_status(status_df, args.lane)


if __name__ == "__main__":
    main()
