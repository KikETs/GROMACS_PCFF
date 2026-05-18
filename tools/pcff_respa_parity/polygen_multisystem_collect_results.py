#!/usr/bin/env python3
"""Collect multi-system validation outputs into one CSV.

This collector is conservative: if a lane does not expose a raw or MSD-fit
cNE0 estimator, the output cell is left empty and a missing_reason is recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import pandas as pd


DEFAULT_OUTDIR = Path("GROMACS_PCFF/output/polygen_multisystem_validation_20260512_m1p50")
METRIC_COLUMNS = [
    "cNE0_raw_htpmd_S_cm",
    "cNE0_raw_t_plus",
    "cNE0_msd_fit_S_cm",
    "cNE0_msd_fit_t_plus",
    "NE_msd_fit_S_cm",
    "D_Li_cm2s",
    "D_anion_cm2s",
    "t_plus_NE",
    "analysis_backend",
]


def float_or_nan(value) -> float:
    try:
        if value is None:
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def batch_traj_id(row: pd.Series) -> int:
    return int(row["trajectory_id"]) * 100 + int(row["replica"])


def load_lammps_result(project: Path) -> tuple[dict, str]:
    transport = load_transport_result(project, "lammps_cpu")
    if transport:
        return transport, "ok"
    pkl_path = project / "MD/analysis_results_htpmd.pkl"
    if not pkl_path.exists():
        return {}, f"missing {pkl_path}"
    with pkl_path.open("rb") as handle:
        result = pickle.load(handle)
    return {
        "cNE0_raw_htpmd_S_cm": float_or_nan(result.get("conductivity")),
        "NE_msd_fit_S_cm": float_or_nan(result.get("conductivity_ne")),
        "D_Li_cm2s": float_or_nan(result.get("li_diffusivity")),
        "D_anion_cm2s": float_or_nan(result.get("tfsi_diffusivity")),
        "t_plus_NE": float_or_nan(result.get("transference_number_ne")),
        "analysis_backend": "LAMMPS_BATCH htpmd.analysis",
    }, "ok"


def load_gromacs_result(project: Path) -> tuple[dict, str]:
    lane = project.parent.name
    transport = load_transport_result(project, lane)
    if transport:
        return transport, "ok"
    report_path = project / "MD_GMX/analysis/gromacs_analysis_report.json"
    if not report_path.exists():
        return {}, f"missing {report_path}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    diffusion = report.get("diffusion", {})
    conductivity = report.get("conductivity", {})
    return {
        "cNE0_raw_htpmd_S_cm": math.nan,
        "cNE0_msd_fit_S_cm": math.nan,
        "NE_msd_fit_S_cm": float_or_nan(conductivity.get("sigma_NE_htpmd_S_cm")),
        "D_Li_cm2s": float_or_nan(diffusion.get("D_Li_cm2s")),
        "D_anion_cm2s": float_or_nan(diffusion.get("D_an_cm2s")),
        "t_plus_NE": float_or_nan(conductivity.get("c_tn_htpmd")),
        "analysis_backend": "GROMACS_PCFF_BATCH gmx msd NE-only",
    }, "gromacs_batch_ne_only_no_cne0_population_matrix"


def load_transport_result(project: Path, lane: str) -> dict | None:
    summary_path = project / "transport_analysis" / lane / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "cNE0_raw_htpmd_S_cm": float_or_nan(summary.get("cNE0_htpmd", {}).get("conductivity_s_cm")),
        "cNE0_raw_t_plus": float_or_nan(summary.get("cNE0_htpmd", {}).get("t_plus")),
        "cNE0_msd_fit_S_cm": float_or_nan(summary.get("cNE0_msd_fit", {}).get("conductivity_s_cm")),
        "cNE0_msd_fit_t_plus": float_or_nan(summary.get("cNE0_msd_fit", {}).get("t_plus")),
        "NE_msd_fit_S_cm": float_or_nan(summary.get("NE_msd_fit", {}).get("conductivity_s_cm")),
        "D_Li_cm2s": float_or_nan(
            summary.get("diffusion_msd_fit_drift_removed", {}).get("cation", {}).get("diffusion_cm2_s")
        ),
        "D_anion_cm2s": float_or_nan(
            summary.get("diffusion_msd_fit_drift_removed", {}).get("anion", {}).get("diffusion_cm2_s")
        ),
        "t_plus_NE": float_or_nan(summary.get("NE_msd_fit", {}).get("t_plus")),
        "analysis_backend": "polygen_multisystem_transport_analysis cNE0_raw/cNE0_msd_fit/NE",
    }


def collect(outdir: Path, jobs_csv: Path) -> pd.DataFrame:
    jobs = pd.read_csv(jobs_csv)
    rows = []
    key_cols = [
        "run_group",
        "duration_ns",
        "worker_role",
        "lane",
        "system_key",
        "category",
        "category_rank",
        "trajectory_id",
        "psmiles",
        "reference_conductivity_s_cm",
        "replica",
    ]
    for _, job in jobs[key_cols].drop_duplicates().iterrows():
        traj = batch_traj_id(job)
        project = outdir / "runs_batch" / str(job["run_group"]) / str(job["worker_role"]) / str(job["lane"]) / f"Traj_{traj}"
        if job["lane"] == "lammps_cpu":
            metrics, status = load_lammps_result(project)
        elif job["lane"] in {"gmx_cpu", "gmx_gpu"}:
            metrics, status = load_gromacs_result(project)
        else:
            metrics, status = {}, f"unsupported lane {job['lane']}"

        row = {col: job[col] for col in key_cols}
        row["project_dir"] = str(project)
        row["analysis_status"] = "ok" if status == "ok" else "partial_or_missing"
        row["missing_reason"] = "" if status == "ok" else status
        for col in METRIC_COLUMNS:
            row[col] = "" if col == "analysis_backend" else math.nan
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    out = outdir / "analysis/multisystem_transport_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(out)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--jobs-csv", type=Path, default=DEFAULT_OUTDIR / "manifest/jobs.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    collect(args.outdir, args.jobs_csv)


if __name__ == "__main__":
    main()
