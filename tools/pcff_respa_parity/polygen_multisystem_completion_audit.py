#!/usr/bin/env python3
"""Audit the PolyGen multi-system validation workflow artifacts.

This is an evidence gate, not a production runner.  It checks that manifests,
selected CSVs, notebooks, smoke statuses, and transport-analysis output schema
cover the requested multi-system workflow.  It also records known blockers such
as remote GPU lanes without CUDA visibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
import pandas as pd


DEFAULT_OUTDIR = Path("GROMACS_PCFF/output/polygen_multisystem_validation_20260512_m1p50")
LANES = ("lammps_cpu", "gmx_cpu", "gmx_gpu")
REQUIRED_TRANSPORT_COLUMNS = [
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


def ok_item(name: str, evidence: object = None) -> dict[str, object]:
    return {"name": name, "status": "ok", "evidence": evidence}


def fail_item(name: str, evidence: object = None) -> dict[str, object]:
    return {"name": name, "status": "missing_or_failed", "evidence": evidence}


def blocked_item(name: str, evidence: object = None) -> dict[str, object]:
    return {"name": name, "status": "blocked", "evidence": evidence}


def csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(len(pd.read_csv(path)))


def all_status_ok(path: Path) -> tuple[bool, object]:
    if not path.exists():
        return False, f"missing {path}"
    df = pd.read_csv(path)
    if "status" not in df.columns:
        return False, f"missing status column in {path}"
    return bool(df["status"].eq("ok").all()), df.to_dict(orient="records")


def notebook_contains(path: Path, needles: list[str]) -> tuple[bool, object]:
    if not path.exists():
        return False, f"missing {path}"
    nb = nbformat.read(path, as_version=4)
    text = "\n".join("".join(cell.get("source", "")) for cell in nb.cells)
    missing = [needle for needle in needles if needle not in text]
    return not missing, {"path": str(path), "cells": len(nb.cells), "missing": missing}


def load_gpu_diagnostic(outdir: Path, role: str) -> dict[str, object] | None:
    path = outdir / "analysis" / f"gpu_diagnostic_{role}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_blocker_evidence(outdir: Path, role: str) -> dict[str, object]:
    diagnostic = load_gpu_diagnostic(outdir, role)
    if diagnostic is None:
        return {
            "selected_csv": str(outdir / "selected" / f"selected_{role}_main20_gmx_gpu.csv"),
            "reason": "remote gmx_gpu smoke is not validated and no GPU diagnostic JSON was found",
        }
    checks = diagnostic.get("checks", {})
    if not checks.get("cuda_runtime_visible"):
        reason = "CUDA runtime is not visible on this remote worker"
    elif not checks.get("gmx_gpu_binary_exists"):
        reason = "CUDA runtime is visible, but build_gateb_cuda/bin/gmx is missing"
    elif not checks.get("gmx_reports_cuda"):
        reason = "GROMACS GPU binary exists but does not report CUDA support"
    else:
        reason = "remote gmx_gpu smoke has not passed yet"
    return {
        "selected_csv": str(outdir / "selected" / f"selected_{role}_main20_gmx_gpu.csv"),
        "diagnostic_json": str(outdir / "analysis" / f"gpu_diagnostic_{role}.json"),
        "reason": reason,
        "checks": checks,
        "recommendation": diagnostic.get("recommendation", ""),
    }


def audit(outdir: Path, repo: Path) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []

    systems_path = outdir / "manifest/systems.csv"
    long_path = outdir / "manifest/long_window_systems.csv"
    jobs_path = outdir / "manifest/jobs.csv"

    if systems_path.exists():
        systems = pd.read_csv(systems_path)
        counts = systems.groupby("category").size().to_dict()
        expected = {"high": 4, "middle": 8, "low": 4}
        checks.append(
            ok_item("main_system_selection_16_systems_4_8_4", counts)
            if counts == expected and len(systems) == 16
            else fail_item("main_system_selection_16_systems_4_8_4", {"rows": len(systems), "counts": counts})
        )
    else:
        checks.append(fail_item("main_system_selection_16_systems_4_8_4", f"missing {systems_path}"))

    if long_path.exists():
        long = pd.read_csv(long_path)
        counts = long.groupby("category").size().to_dict()
        expected = {"high": 1, "middle": 2, "low": 1}
        checks.append(
            ok_item("long_window_selection_4_systems_1_2_1", counts)
            if counts == expected and len(long) == 4
            else fail_item("long_window_selection_4_systems_1_2_1", {"rows": len(long), "counts": counts})
        )
    else:
        checks.append(fail_item("long_window_selection_4_systems_1_2_1", f"missing {long_path}"))

    if jobs_path.exists():
        jobs = pd.read_csv(jobs_path)
        job_counts = jobs.groupby(["run_group", "lane"]).size().to_dict()
        checks.append(ok_item("jobs_csv_180_lane_jobs", {"rows": len(jobs), "by_group_lane": {str(k): int(v) for k, v in job_counts.items()}}) if len(jobs) == 180 else fail_item("jobs_csv_180_lane_jobs", len(jobs)))
        checks.append(ok_item("main20_144_lane_jobs", int(jobs["run_group"].eq("main20").sum())) if int(jobs["run_group"].eq("main20").sum()) == 144 else fail_item("main20_144_lane_jobs", int(jobs["run_group"].eq("main20").sum())))
        checks.append(ok_item("long50_36_lane_jobs", int(jobs["run_group"].eq("long50").sum())) if int(jobs["run_group"].eq("long50").sum()) == 36 else fail_item("long50_36_lane_jobs", int(jobs["run_group"].eq("long50").sum())))
    else:
        checks.append(fail_item("jobs_csv_180_lane_jobs", f"missing {jobs_path}"))

    selected_expectations = {
        ("local_main", "main20"): 24,
        ("local_main", "long50"): 12,
        ("remote_mid_a", "main20"): 12,
        ("remote_mid_b", "main20"): 12,
    }
    for (role, group), expected_rows in selected_expectations.items():
        for lane in LANES:
            path = outdir / "selected" / f"selected_{role}_{group}_{lane}.csv"
            rows = csv_rows(path)
            checks.append(
                ok_item(f"selected_{role}_{group}_{lane}", {"path": str(path), "rows": rows})
                if rows == expected_rows
                else fail_item(f"selected_{role}_{group}_{lane}", {"path": str(path), "rows": rows, "expected": expected_rows})
            )

    notebook_expectations = {
        "output/jupyter-notebook/polygen_multisystem_validation_local_main.ipynb": [
            "ROLE = 'local_main'",
            "RUN_GROUPS = ['main20', 'long50']",
            "polygen_multisystem_transport_analysis.py",
            "polygen_multisystem_collect_results.py",
        ],
        "output/jupyter-notebook/polygen_multisystem_validation_remote_mid_a.ipynb": [
            "ROLE = 'remote_mid_a'",
            "RUN_GROUPS = ['main20']",
            "LANES = ['lammps_cpu', 'gmx_cpu', 'gmx_gpu']",
            "build_gateb_cuda/bin/gmx",
            "MD/em.lmp",
        ],
        "output/jupyter-notebook/polygen_multisystem_validation_remote_mid_b.ipynb": [
            "ROLE = 'remote_mid_b'",
            "RUN_GROUPS = ['main20']",
            "LANES = ['lammps_cpu', 'gmx_cpu', 'gmx_gpu']",
            "build_gateb_cuda/bin/gmx",
            "MD/em.lmp",
        ],
        "output/jupyter-notebook/polygen_multisystem_deploy_remotes.ipynb": [
            "RUN_PREPARE_INPUTS",
            "deploy",
        ],
        "output/jupyter-notebook/polygen_multisystem_remote_setup_remote_mid_a.ipynb": [
            "ROLE = 'remote_mid_a'",
            "RUN_GPU_DIAGNOSTIC",
            "RUN_CUDA_CONDA_INSTALL",
            "polygen_remote_gpu_diagnose.py",
        ],
        "output/jupyter-notebook/polygen_multisystem_remote_setup_remote_mid_b.ipynb": [
            "ROLE = 'remote_mid_b'",
            "RUN_GPU_DIAGNOSTIC",
            "RUN_CUDA_CONDA_INSTALL",
            "polygen_remote_gpu_diagnose.py",
        ],
    }
    for rel, needles in notebook_expectations.items():
        ok, evidence = notebook_contains(repo / rel, needles)
        checks.append(ok_item(f"notebook_{Path(rel).name}", evidence) if ok else fail_item(f"notebook_{Path(rel).name}", evidence))

    smoke_expectations = [
        ("local_main", "lammps_cpu"),
        ("local_main", "gmx_cpu"),
        ("local_main", "gmx_gpu"),
        ("remote_mid_a", "lammps_cpu"),
        ("remote_mid_a", "gmx_cpu"),
        ("remote_mid_b", "lammps_cpu"),
        ("remote_mid_b", "gmx_cpu"),
    ]
    for role, lane in smoke_expectations:
        path = outdir / "runs_batch/smoke_main20" / role / lane / "batch_status.csv"
        ok, evidence = all_status_ok(path)
        checks.append(ok_item(f"smoke_{role}_{lane}", evidence) if ok else fail_item(f"smoke_{role}_{lane}", evidence))

    for role in ("remote_mid_a", "remote_mid_b"):
        path = outdir / "runs_batch/smoke_main20" / role / "gmx_gpu" / "batch_status.csv"
        ok, status_evidence = all_status_ok(path)
        if ok:
            checks.append(ok_item(f"smoke_{role}_gmx_gpu", status_evidence))
            continue
        evidence = gpu_blocker_evidence(outdir, role)
        diagnostic = load_gpu_diagnostic(outdir, role)
        ready = bool(diagnostic and diagnostic.get("checks", {}).get("gmx_gpu_ready"))
        if ready:
            checks.append(fail_item(f"remote_gpu_smoke_{role}_not_passed", evidence))
        else:
            item = blocked_item(f"remote_gpu_smoke_{role}_blocked", evidence)
            checks.append(item)
            blockers.append(item)

    for role in ("local_main", "remote_mid_a", "remote_mid_b"):
        path = outdir / "analysis" / f"gpu_diagnostic_{role}.json"
        checks.append(
            ok_item(f"gpu_diagnostic_{role}", str(path))
            if path.exists()
            else fail_item(f"gpu_diagnostic_{role}", str(path))
        )

    summary_path = outdir / "analysis/multisystem_transport_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        missing_cols = [col for col in REQUIRED_TRANSPORT_COLUMNS if col not in summary.columns]
        evidence = {"path": str(summary_path), "rows": len(summary), "missing_columns": missing_cols}
        checks.append(ok_item("transport_summary_schema", evidence) if len(summary) == 180 and not missing_cols else fail_item("transport_summary_schema", evidence))
    else:
        checks.append(fail_item("transport_summary_schema", f"missing {summary_path}"))

    for rel in [
        "tools/pcff_respa_parity/polygen_multisystem_worker.py",
        "tools/pcff_respa_parity/polygen_multisystem_manifest.py",
        "tools/pcff_respa_parity/polygen_multisystem_deploy.py",
        "tools/pcff_respa_parity/polygen_multisystem_transport_analysis.py",
        "tools/pcff_respa_parity/polygen_multisystem_collect_results.py",
        "tools/pcff_respa_parity/polygen_remote_gpu_diagnose.py",
    ]:
        path = repo / rel
        checks.append(ok_item(f"script_{Path(rel).name}", str(path)) if path.exists() else fail_item(f"script_{Path(rel).name}", str(path)))

    failed = [item for item in checks if item["status"] == "missing_or_failed"]
    verdict = "ready_except_remote_gpu_blocked" if not failed and blockers else "ready" if not failed else "not_ready"
    return {
        "objective": "48 x 20 ns main workflow plus 4 x 3 x 50 ns long-window sanity across local/remotes with LAMMPS CPU, GROMACS CPU/GPU, raw/MSD-fit cNE0 analysis, notebooks, and smoke gates.",
        "verdict": verdict,
        "complete": verdict == "ready",
        "checks": checks,
        "failed": failed,
        "blockers": blockers,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path("GROMACS_PCFF"))
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--json-out", type=Path, default=DEFAULT_OUTDIR / "analysis/completion_audit.json")
    p.add_argument("--strict", action="store_true", help="Exit non-zero unless the audit is fully complete.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.outdir, args.repo)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.json_out)
    print(json.dumps({"verdict": result["verdict"], "complete": result["complete"], "failed": len(result["failed"]), "blockers": len(result["blockers"])}, indent=2))
    if args.strict and not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
