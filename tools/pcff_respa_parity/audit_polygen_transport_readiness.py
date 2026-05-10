#!/usr/bin/env python3
"""Audit whether current PolyGen production outputs are transport-analysis ready.

This checker is intentionally conservative.  It does not claim conductivity or
cNE parity; it only checks whether the trajectory artifacts match the current
notebook contract closely enough to feed structure/transport/cNE analysis.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = REPO / "output" / "polygen_pcff_gromacs_initial_em_notebook"
DEFAULT_REPORT_ROOT = REPO / "output" / "polygen_transport_readiness_audits"
DEFAULT_GROMACS_LANES = {
    "gromacs_cpu_strict": "gromacs_cpu_openmp",
    "gromacs_gpu_strict": "gromacs_gpu_hybrid_strict_pme5",
}


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Check whether LAMMPS/GROMACS PolyGen production trajectories match "
            "the current transport-ready artifact contract."
        )
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_ROOT / f"transport_ready_{stamp}")
    parser.add_argument("--expected-prod-chunks", type=int, default=50)
    parser.add_argument("--expected-lammps-dump-stride", type=int, default=1000)
    parser.add_argument("--expected-gmx-xtc-stride", type=int, default=4000)
    parser.add_argument(
        "--gromacs-lane",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help=(
            "GROMACS production lane to audit relative to --out-root. "
            "May be passed more than once. Defaults to CPU strict and GPU strict lanes."
        ),
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def extract_mdp_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    key_re = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(.*?)\s*$")
    for raw in read_text(path).splitlines():
        match = key_re.match(raw)
        if match:
            return match.group(1)
    return None


def pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


@dataclass
class Check:
    item: str
    status: str
    evidence: str

    def as_dict(self) -> dict[str, str]:
        return {"item": self.item, "status": self.status, "evidence": self.evidence}


def audit_lammps(out_root: Path, expected_chunks: int, expected_stride: int) -> list[Check]:
    work = out_root / "lammps_openmp"
    manifest = work / "prod_trajectory_manifest.json"
    checks: list[Check] = []

    if manifest.exists():
        try:
            payload = json.loads(read_text(manifest))
        except json.JSONDecodeError as exc:
            payload = {}
            checks.append(Check("lammps_manifest_parse", "FAIL", f"{manifest}: {exc}"))
    else:
        payload = {}
        checks.append(Check("lammps_manifest_exists", "FAIL", str(manifest)))

    if payload:
        stride = payload.get("stride_steps")
        checks.append(
            Check(
                "lammps_dump_stride_matches_polygen_production",
                pass_fail(stride == expected_stride),
                f"{manifest}: stride_steps={stride}, expected={expected_stride}",
            )
        )
        fields = payload.get("fields", [])
        required_fields = ["id", "mol", "type", "mass", "q", "x", "y", "z", "ix", "iy", "iz"]
        missing = [field for field in required_fields if field not in fields]
        checks.append(
            Check(
                "lammps_dump_fields_include_transport_minimum",
                pass_fail(not missing),
                f"{manifest}: missing={missing}, fields={fields}",
            )
        )
        files = [Path(path) for path in payload.get("files", [])]
        if files and not files[0].is_absolute():
            files = [(work / path).resolve() for path in files]
        present = [path for path in files if nonempty(path)]
        checks.append(
            Check(
                "lammps_dump_chunk_count",
                pass_fail(len(present) >= expected_chunks),
                f"nonempty={len(present)}, expected>={expected_chunks}",
            )
        )

    first_input = work / "resume_inputs" / "lammps_prod_chunk0001.in"
    run_style_ok = first_input.exists() and "run_style       respa 2 4" in read_text(first_input)
    checks.append(
        Check(
            "lammps_production_run_style_matches_polygen_production_in",
            pass_fail(run_style_ok),
            str(first_input),
        )
    )
    return checks


def parse_gromacs_lanes(raw_lanes: list[str]) -> dict[str, str]:
    if not raw_lanes:
        return dict(DEFAULT_GROMACS_LANES)
    lanes: dict[str, str] = {}
    for raw in raw_lanes:
        if "=" not in raw:
            raise ValueError(f"--gromacs-lane must be LABEL=DIR, got: {raw}")
        label, rel_dir = raw.split("=", 1)
        label = label.strip()
        rel_dir = rel_dir.strip()
        if not label or not rel_dir:
            raise ValueError(f"--gromacs-lane must be LABEL=DIR, got: {raw}")
        lanes[label] = rel_dir
    return lanes


def audit_gromacs_lane(
    out_root: Path, label: str, rel_dir: str, expected_chunks: int, expected_xtc_stride: int
) -> list[Check]:
    work = out_root / rel_dir
    checks: list[Check] = []
    mdp = work / "14_prod01_nvt_10000ps_chunk0001.mdp"

    mdp_expectations = {
        "exact-respa-levels": "2",
        "exact-respa-level2-factor": "4",
        "exact-respa-pair-level": "2",
        "exact-respa-kspace-level": "2",
        "nstxout-compressed": str(expected_xtc_stride),
    }
    for key, expected in mdp_expectations.items():
        value = extract_mdp_value(mdp, key)
        checks.append(
            Check(
                f"{label}_prod_mdp_{key}",
                pass_fail(value == expected),
                f"{mdp}: {key}={value}, expected={expected}",
            )
        )

    xtc = sorted(work.glob("14_prod01_nvt_10000ps_chunk*.xtc"))
    tpr = sorted(work.glob("14_prod01_nvt_10000ps_chunk*.tpr"))
    checks.append(
        Check(
            f"{label}_prod_xtc_chunk_count",
            pass_fail(len([path for path in xtc if nonempty(path)]) >= expected_chunks),
            f"nonempty={len([path for path in xtc if nonempty(path)])}, expected>={expected_chunks}",
        )
    )
    checks.append(
        Check(
            f"{label}_prod_tpr_chunk_count",
            pass_fail(len([path for path in tpr if nonempty(path)]) >= expected_chunks),
            f"nonempty={len([path for path in tpr if nonempty(path)])}, expected>={expected_chunks}",
        )
    )
    return checks


def audit_gromacs(out_root: Path, expected_chunks: int, expected_xtc_stride: int, raw_lanes: list[str]) -> list[Check]:
    checks: list[Check] = []
    for label, rel_dir in parse_gromacs_lanes(raw_lanes).items():
        checks.extend(audit_gromacs_lane(out_root, label, rel_dir, expected_chunks, expected_xtc_stride))
    return checks


def write_report(report_dir: Path, checks: list[Check]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [check.as_dict() for check in checks]
    overall = "PASS" if all(check.status == "PASS" for check in checks) else "FAIL"
    payload: dict[str, Any] = {
        "overall": overall,
        "scope": "Artifact readiness only. This is not a cNE/transport parity result.",
        "checks": rows,
    }
    (report_dir / "transport_readiness_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        f"# PolyGen Transport Readiness Audit",
        "",
        f"Overall: **{overall}**",
        "",
        "This checks artifact readiness only. It does not compute cNE, NE, MSD, diffusion, or conductivity.",
        "",
        "| Item | Status | Evidence |",
        "|---|---:|---|",
    ]
    for check in checks:
        evidence = check.evidence.replace("|", "\\|")
        lines.append(f"| {check.item} | {check.status} | `{evidence}` |")
    (report_dir / "transport_readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_root = args.out_root.resolve()
    checks = [
        *audit_lammps(out_root, args.expected_prod_chunks, args.expected_lammps_dump_stride),
        *audit_gromacs(out_root, args.expected_prod_chunks, args.expected_gmx_xtc_stride, args.gromacs_lane),
    ]
    write_report(args.report_dir.resolve(), checks)
    overall = "PASS" if all(check.status == "PASS" for check in checks) else "FAIL"
    print(f"transport readiness {overall}: {args.report_dir.resolve()}", flush=True)
    return 0 if overall == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
