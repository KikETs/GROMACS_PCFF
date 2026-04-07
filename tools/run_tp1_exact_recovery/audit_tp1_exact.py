#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "tp1_exact_recovery"
    / "dense_salt_polymer_corrected_npt_5ns"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit corrected exact TP1 recovery artifacts.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_final_box_nm(gro_path: Path) -> list[float]:
    last = gro_path.read_text(encoding="utf-8", errors="replace").splitlines()[-1]
    return [float(token) for token in last.split()[:3]]


def parse_mdp_value(mdout_text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}\s*=\s*(\S+)", mdout_text, re.MULTILINE)
    return match.group(1) if match else None


def parse_float_mdp(mdout_text: str, key: str) -> float | None:
    value = parse_mdp_value(mdout_text, key)
    return float(value) if value is not None else None


def repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def main() -> int:
    args = parse_args()
    bundle = args.bundle.resolve()
    report_path = bundle / "tp1_exact_recovery_report.json"
    report = load_json(report_path)
    raw_paths = [Path(path) for path in report["raw_artifacts"].values()]
    generated_paths = [Path(path) for path in report["protocol"]["artifacts"].values()]
    required_paths = raw_paths + generated_paths + [
        bundle / "tp1_exact_protocol.json",
        bundle / "grompp_tp1.stderr",
        bundle / "grompp_tp1.stdout",
        bundle / "mdrun_tp1.stderr",
        bundle / "tp1_equil.log",
    ]
    missing = [repo_rel(path) for path in required_paths if not path.exists()]

    mdout_text = (bundle / "tp1_equil_mdout.mdp").read_text(encoding="utf-8", errors="replace")
    final_box_nm = parse_final_box_nm(bundle / "tp1_equil.gro")
    cutoffs_nm = {
        "rlist": parse_float_mdp(mdout_text, "rlist"),
        "rcoulomb": parse_float_mdp(mdout_text, "rcoulomb"),
        "rvdw": parse_float_mdp(mdout_text, "rvdw"),
    }
    max_cutoff_nm = max(value for value in cutoffs_nm.values() if value is not None)
    min_half_box_nm = min(final_box_nm) / 2.0
    half_box_margin_nm = min_half_box_nm - max_cutoff_nm

    analysis = report["analysis"]
    temperature = analysis["temperature_k"]
    thresholds = report["protocol"]["thresholds"]
    thermal_pass = (
        report["status"] == "PASS"
        and analysis["duration_completed_ps"] >= report["protocol"]["duration_ps"]
        and abs(temperature["mean"] - thresholds["mean_temperature_target_k"])
        <= thresholds["mean_temperature_tolerance_k"]
        and temperature["max"] <= thresholds["max_temperature_k"]
    )
    corrected_contract_pass = bool(report["mdout_contract"]["all_pass"])
    artifacts_pass = not missing
    endpoint_cutoff_margin_pass = half_box_margin_nm >= 0.0

    audit = {
        "schema_name": "tp1_exact_recovery_audit",
        "schema_version": 1,
        "system_id": report["system_id"],
        "bundle": repo_rel(bundle),
        "source_report": repo_rel(report_path),
        "historical_blocker_addressed": (
            "The old TP1.2 3.017 ns thermal runaway is superseded only for the corrected "
            "dense_salt_polymer NPT protocol with tcoupl/pcoupl/gen-vel applied."
        ),
        "verdicts": {
            "thermal_runaway_exact_blocker": "PASS" if thermal_pass else "FAIL",
            "corrected_protocol_contract": "PASS" if corrected_contract_pass else "FAIL",
            "raw_artifact_bundle": "PASS" if artifacts_pass else "FAIL",
            "endpoint_cutoff_margin": "PASS" if endpoint_cutoff_margin_pass else "FAIL",
            "transport_or_production_readiness": "FAIL",
        },
        "duration_completed_ps": analysis["duration_completed_ps"],
        "analysis_window_ps": report["protocol"]["analysis_window_ps"],
        "temperature_k": temperature,
        "density_kg_m3": analysis["density_kg_m3"],
        "volume_nm3": analysis["volume_nm3"],
        "mdout_contract": report["mdout_contract"],
        "final_box_nm": final_box_nm,
        "cutoffs_nm": cutoffs_nm,
        "min_half_box_nm": min_half_box_nm,
        "half_box_margin_nm": half_box_margin_nm,
        "missing_artifacts": missing,
        "claimable_statement": (
            "The exact TP1 thermal-runaway blocker is resolved for the corrected 5 ns "
            "dense_salt_polymer NPT rerun."
        ),
        "non_claimable_statement": (
            "This does not establish dense GROMACS-vs-LAMMPS parity, transport readiness, "
            "or endpoint continuation safety; the final box is smaller than twice the 0.9 nm cutoff."
        ),
    }
    out_path = args.out or (bundle / "tp1_exact_recovery_audit.json")
    write_json(out_path, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
