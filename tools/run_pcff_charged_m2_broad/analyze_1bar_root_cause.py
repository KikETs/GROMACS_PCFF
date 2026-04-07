#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "tests/reference_results/pcff_charged_expansion/m2_1bar_root_cause"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize non-formal M2 1 bar root-cause probes.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def summarize_formal_report(path: Path) -> dict[str, object]:
    report = load_json(path)
    return {
        "path": str(path),
        "status": report["status"],
        "protocol": {
            "duration_ps": report["protocol"]["duration_ps"],
            "analysis_window_ps": report["protocol"]["analysis_window_ps"],
            "ref_p_bar": report["protocol"]["ref_p_bar"],
            "warmup_scope": report["protocol"].get("warmup_scope"),
            "warmup_ps": report["protocol"].get("warmup_ps"),
            "lammps_neighbor_skin_angstrom": report["protocol"].get("lammps_neighbor_skin_angstrom"),
            "lammps_neighbor_every": report["protocol"].get("lammps_neighbor_every"),
        },
        "parity_metrics": report["parity_metrics"],
        "gromacs": {
            "density_kg_m3_mean": report["gromacs"]["density_kg_m3"]["mean"],
            "volume_nm3_mean": report["gromacs"]["volume_nm3"]["mean"],
            "pressure_bar_mean": report["gromacs"]["pressure_bar"]["mean"],
        },
        "lammps": {
            "density_kg_m3_mean": report["lammps"]["density_kg_m3"]["mean"],
            "volume_nm3_mean": report["lammps"]["volume_nm3"]["mean"],
            "pressure_atm_mean": report["lammps"]["pressure_atm"]["mean"],
        },
    }


def parse_lammps_thermo(log_path: Path) -> list[dict[str, float]]:
    lines = log_path.read_text(errors="replace").splitlines()
    header: list[str] | None = None
    start: int | None = None
    for idx, line in enumerate(lines):
        if "Step" in line and "Temp" in line and "Density" in line:
            header = line.split()
            start = idx + 1
            break
    if header is None or start is None:
        raise RuntimeError(f"No LAMMPS thermo header found in {log_path}")

    rows: list[dict[str, float]] = []
    for line in lines[start:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Loop":
            break
        if parts[0].isdigit() and len(parts) >= len(header):
            rows.append({key: float(value) for key, value in zip(header, parts)})
    if not rows:
        raise RuntimeError(f"No LAMMPS thermo rows found in {log_path}")
    return rows


def summarize_lammps(log_path: Path, window_start_ps: float, window_end_ps: float | None = None) -> dict[str, object]:
    rows = parse_lammps_thermo(log_path)
    window_start_step = window_start_ps * 1000.0
    window_end_step = None if window_end_ps is None else window_end_ps * 1000.0
    window = [
        row
        for row in rows
        if row["Step"] >= window_start_step and (window_end_step is None or row["Step"] <= window_end_step)
    ]
    if not window:
        raise RuntimeError(f"No LAMMPS rows after {window_start_ps} ps in {log_path}")

    def convert(row: dict[str, float]) -> dict[str, float]:
        return {
            "time_ps": row["Step"] * 0.001,
            "temperature_k": row["Temp"],
            "pressure_atm": row["Press"],
            "volume_nm3": row["Volume"] / 1000.0,
            "density_kg_m3": row["Density"] * 1000.0,
        }

    return {
        "path": str(log_path),
        "rows": len(rows),
        "first": convert(rows[0]),
        "last": convert(rows[-1]),
        "window_start_ps": window_start_ps,
        "window_end_ps": window_end_ps,
        "window_mean": {
            "temperature_k": mean([row["Temp"] for row in window]),
            "pressure_atm": mean([row["Press"] for row in window]),
            "volume_nm3": mean([row["Volume"] / 1000.0 for row in window]),
            "density_kg_m3": mean([row["Density"] * 1000.0 for row in window]),
        },
    }


def parse_xvg(xvg_path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in xvg_path.read_text(errors="replace").splitlines():
        if not line or line[0] in "#@":
            continue
        parts = line.split()
        if len(parts) >= 6:
            rows.append([float(value) for value in parts[:6]])
    if not rows:
        raise RuntimeError(f"No XVG rows found in {xvg_path}")
    return rows


def summarize_gromacs_xvg(xvg_path: Path, window_start_ps: float, window_end_ps: float | None = None) -> dict[str, object]:
    rows = parse_xvg(xvg_path)
    window = [row for row in rows if row[0] >= window_start_ps and (window_end_ps is None or row[0] <= window_end_ps)]
    if not window:
        raise RuntimeError(f"No GROMACS rows after {window_start_ps} ps in {xvg_path}")

    def convert(row: list[float]) -> dict[str, float]:
        return {
            "time_ps": row[0],
            "temperature_k": row[2],
            "pressure_bar": row[3],
            "volume_nm3": row[4],
            "density_kg_m3": row[5],
        }

    return {
        "path": str(xvg_path),
        "rows": len(rows),
        "first": convert(rows[0]),
        "last": convert(rows[-1]),
        "window_start_ps": window_start_ps,
        "window_end_ps": window_end_ps,
        "window_mean": {
            "temperature_k": mean([row[2] for row in window]),
            "pressure_bar": mean([row[3] for row in window]),
            "volume_nm3": mean([row[4] for row in window]),
            "density_kg_m3": mean([row[5] for row in window]),
        },
    }


def write_sha_manifest(root: Path) -> Path:
    manifest = root / "sha256_manifest.txt"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != manifest.name)
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    formal_1bar = summarize_formal_report(
        REPO_ROOT
        / "tests/reference_results/pcff_charged_expansion/m2_broad/systems/monoglyme_ethane_litfsi_1to1_dense18/paired_npt/dense_npt_parity_report.json"
    )
    formal_250bar = summarize_formal_report(
        REPO_ROOT
        / "tests/reference_results/pcff_charged_expansion/m2_broad_v3_250bar_mttk_skin4_lmp1/systems/monoglyme_ethane_litfsi_1to1_dense18/paired_npt/dense_npt_parity_report.json"
    )
    lmp_1bar_initial = summarize_lammps(
        root / "m5_1bar_lmp1_skin4_100ps_lammps_only/lammps/npt.log",
        window_start_ps=50.0,
    )
    lmp_250_to_1bar = summarize_lammps(
        root / "m5_250bar_endpoint_to_1bar_lmp_skin4_100ps/lammps/npt_1bar_from_250bar_final.log",
        window_start_ps=50.0,
    )
    lmp_250_to_1bar_10_to_20 = summarize_lammps(
        root / "m5_250bar_endpoint_to_1bar_lmp_skin4_100ps/lammps/npt_1bar_from_250bar_final.log",
        window_start_ps=10.0,
        window_end_ps=20.0,
    )
    gmx_250_to_1bar_20ps = summarize_gromacs_xvg(
        root / "m5_250bar_endpoint_to_1bar_gmx_mttk_20ps/gromacs/npt_1bar_from_250bar_final_energy.xvg",
        window_start_ps=10.0,
        window_end_ps=20.0,
    )

    summary = {
        "system": "monoglyme_ethane_litfsi_1to1_dense18",
        "purpose": "ambient 1 bar dense-parity root-cause probes; non-formal evidence, not M2 PASS evidence",
        "interpretation": {
            "ambient_1bar_broader_m2_status": "UNRESOLVED",
            "neighbor_skin_not_primary_cause": True,
            "gmx_only_warmup_not_primary_cause": True,
            "one_bar_behavior_is_path_dependent": True,
            "most_supported_current_failure_mode": (
                "The direct 1 bar release from the generated dense18 state expands/cavitates to gas-like "
                "volumes, while 250 bar preconditioned coordinates remain condensed in short 1 bar probes. "
                "The likely blocker is pressure-path/initial-basin dependence rather than LAMMPS neighbor-list "
                "settings or parser/emitter provenance."
            ),
        },
        "probes": {
            "formal_v1_1bar_initial_100ps": formal_1bar,
            "formal_v3_250bar_100ps": formal_250bar,
            "lammps_1bar_initial_skin4_100ps": lmp_1bar_initial,
            "lammps_250bar_endpoint_to_1bar_skin4_100ps": lmp_250_to_1bar,
            "lammps_250bar_endpoint_to_1bar_skin4_20ps_window_from10ps": lmp_250_to_1bar_10_to_20,
            "gromacs_250bar_endpoint_to_1bar_mttk_20ps_window_from10ps": gmx_250_to_1bar_20ps,
        },
        "caveats": [
            "GROMACS endpoint-to-1bar probe is 20 ps only; it is root-cause evidence, not formal parity evidence.",
            "The 250 bar precondition path cannot be presented as ambient 1 bar equilibrium unless a predeclared staged ambient protocol is validated.",
            "Pressure means differ and fluctuate; density retention alone is not transport or production readiness.",
        ],
    }
    write_json(root / "m2_1bar_root_cause_summary.json", summary)
    write_sha_manifest(root)
    print(root / "m2_1bar_root_cause_summary.json")


if __name__ == "__main__":
    main()
