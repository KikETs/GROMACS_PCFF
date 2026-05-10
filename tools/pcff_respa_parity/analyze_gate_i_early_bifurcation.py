from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from freeze_gate_a_oracle import DEFAULT_GMX, write_text
from validate_gate_g_long_ensemble import extract_energy_series


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_I_ROOT = REPO_ROOT / "tests" / "reference_results" / "gate_i_charged_long_npt_conditioning"
DEFAULT_OUT_PREFIX = "equil_early_bifurcation"
REQUESTED_TERMS = ("Density", "Volume", "Temperature", "Pressure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Gate I equilibration observables for early replica bifurcation without changing the original "
            "Gate I contract or rerunning the campaign."
        )
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument("--gate-i-root", default=str(DEFAULT_GATE_I_ROOT), help="Completed or running Gate I root.")
    parser.add_argument(
        "--horizon-ps",
        type=float,
        default=250.0,
        help="Early equilibration horizon to analyze for replica bifurcation.",
    )
    parser.add_argument(
        "--density-thresholds",
        default="100,200,300,400",
        help="Comma-separated density gap thresholds in kg/m^3.",
    )
    parser.add_argument(
        "--volume-thresholds",
        default="2,5,8,10",
        help="Comma-separated volume gap thresholds in nm^3.",
    )
    parser.add_argument(
        "--out-prefix",
        default=DEFAULT_OUT_PREFIX,
        help="Prefix for the JSON/Markdown report under the Gate I root.",
    )
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    thresholds: list[float] = []
    for field in raw.split(","):
        value = field.strip()
        if not value:
            continue
        thresholds.append(float(value))
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def replica_dirs(gate_i_root: Path) -> list[Path]:
    cpu_root = gate_i_root / "cpu"
    if not cpu_root.exists():
        raise FileNotFoundError(f"Missing Gate I cpu root: {cpu_root}")
    replicas = sorted(path for path in cpu_root.glob("replica_*") if path.is_dir())
    if not replicas:
        raise FileNotFoundError(f"No replica directories found under {cpu_root}")
    return replicas


def extract_replica_series(gmx: Path, replica_dir: Path) -> dict[str, list[float]]:
    equil_edr = replica_dir / "equil.edr"
    if not equil_edr.exists():
        raise FileNotFoundError(f"Missing equilibration EDR: {equil_edr}")
    return extract_energy_series(gmx, equil_edr, replica_dir / "equil_observables.xvg", REQUESTED_TERMS)


def aligned_time_index(series_by_replica: list[dict[str, list[float]]], horizon_ps: float) -> tuple[list[float], list[int]]:
    if horizon_ps <= 0.0:
        raise ValueError("horizon-ps must be positive.")
    min_len = min(len(series["time_ps"]) for series in series_by_replica)
    if min_len == 0:
        raise ValueError("At least one equilibration series is empty.")
    reference = [float(value) for value in series_by_replica[0]["time_ps"][:min_len]]
    for series in series_by_replica[1:]:
        current = [float(value) for value in series["time_ps"][:min_len]]
        for index, (left, right) in enumerate(zip(reference, current)):
            if not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(
                    f"Replica equilibration times are misaligned at sample {index}: {left} ps vs {right} ps."
                )
    indices = [index for index, time_ps in enumerate(reference) if time_ps <= horizon_ps + 1.0e-9]
    if not indices:
        raise ValueError(f"No equilibration frames fall within horizon {horizon_ps} ps.")
    return reference, indices


def first_crossing_time(times: list[float], values: list[float], threshold: float) -> float | None:
    for time_ps, value in zip(times, values):
        if value >= threshold:
            return time_ps
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def build_gap_series(values_by_replica: list[list[float]], target_index: int) -> list[float]:
    if len(values_by_replica) < 2:
        return [0.0 for _ in values_by_replica[target_index]]
    target = values_by_replica[target_index]
    others = [values for index, values in enumerate(values_by_replica) if index != target_index]
    return [abs(target[sample] - mean([values[sample] for values in others])) for sample in range(len(target))]


def rounded_mapping(thresholds: list[float], values: list[float]) -> dict[str, float | None]:
    return {
        f"{threshold:g}": (None if value is None else round(value, 6)) for threshold, value in zip(thresholds, values)
    }


def build_report(
    *,
    gate_i_root: Path,
    gmx: Path,
    horizon_ps: float,
    density_thresholds: list[float],
    volume_thresholds: list[float],
    replica_series: list[dict[str, object]],
) -> dict[str, object]:
    series_payloads = [payload["series"] for payload in replica_series]
    times, horizon_indices = aligned_time_index(series_payloads, horizon_ps)
    horizon_times = [times[index] for index in horizon_indices]

    density_by_replica = [[float(value) for value in payload["series"]["Density"]] for payload in replica_series]
    volume_by_replica = [[float(value) for value in payload["series"]["Volume"]] for payload in replica_series]
    density_horizon = [[values[index] for index in horizon_indices] for values in density_by_replica]
    volume_horizon = [[values[index] for index in horizon_indices] for values in volume_by_replica]

    density_span = [max(samples) - min(samples) for samples in zip(*density_horizon)]
    volume_span = [max(samples) - min(samples) for samples in zip(*volume_horizon)]

    replica_reports = []
    for index, payload in enumerate(replica_series):
        density_gap = build_gap_series(density_horizon, index)
        volume_gap = build_gap_series(volume_horizon, index)
        density_crossings = [first_crossing_time(horizon_times, density_gap, threshold) for threshold in density_thresholds]
        volume_crossings = [first_crossing_time(horizon_times, volume_gap, threshold) for threshold in volume_thresholds]
        replica_reports.append(
            {
                "replica_name": payload["replica_name"],
                "replica_dir": str(payload["replica_dir"]),
                "horizon_end": {
                    "time_ps": round(horizon_times[-1], 6),
                    "density": density_horizon[index][-1],
                    "volume": volume_horizon[index][-1],
                },
                "equil_end": {
                    "time_ps": round(times[-1], 6),
                    "density": density_by_replica[index][-1],
                    "volume": volume_by_replica[index][-1],
                },
                "gap_to_other_mean": {
                    "Density": {
                        "horizon_end_gap": density_gap[-1],
                        "max_gap_within_horizon": max(density_gap),
                        "threshold_crossing_ps": rounded_mapping(density_thresholds, density_crossings),
                    },
                    "Volume": {
                        "horizon_end_gap": volume_gap[-1],
                        "max_gap_within_horizon": max(volume_gap),
                        "threshold_crossing_ps": rounded_mapping(volume_thresholds, volume_crossings),
                    },
                },
            }
        )

    density_span_crossings = [first_crossing_time(horizon_times, density_span, threshold) for threshold in density_thresholds]
    volume_span_crossings = [first_crossing_time(horizon_times, volume_span, threshold) for threshold in volume_thresholds]
    median_equil_density = median([replica["equil_end"]["density"] for replica in replica_reports])
    median_equil_volume = median([replica["equil_end"]["volume"] for replica in replica_reports])
    sparse_basin_scores: list[tuple[float, str]] = []
    for replica in replica_reports:
        score = 0.0
        if density_thresholds:
            score += max(0.0, median_equil_density - float(replica["equil_end"]["density"])) / density_thresholds[0]
        if volume_thresholds:
            score += max(0.0, float(replica["equil_end"]["volume"]) - median_equil_volume) / volume_thresholds[0]
        replica["equil_end"]["sparse_basin_score"] = score
        sparse_basin_scores.append((score, replica["replica_name"]))
    suspected_sparse_basin = max(sparse_basin_scores)[1] if sparse_basin_scores and max(sparse_basin_scores)[0] > 0.0 else None

    detected = any(value is not None for value in density_span_crossings + volume_span_crossings)
    return {
        "schema_name": "gate_i_early_bifurcation",
        "schema_version": 1,
        "status": "EARLY_BIFURCATION_DETECTED" if detected else "NO_EARLY_BIFURCATION_DETECTED",
        "purpose": (
            "Detect early replica bifurcation during Gate I equilibration so density/volume failures are not hidden "
            "behind longer production summaries."
        ),
        "non_claims": [
            "This report is a diagnostic on equilibration observables only.",
            "This report does not convert partial NPT diagnostics into a Gate I PASS.",
            "A dense early horizon does not imply transport readiness or production handoff readiness.",
        ],
        "gate_i_root": str(gate_i_root),
        "gmx": str(gmx),
        "replica_count": len(replica_series),
        "horizon_ps_requested": horizon_ps,
        "horizon_ps_analyzed": round(horizon_times[-1], 6),
        "equil_end_ps": round(times[-1], 6),
        "cross_replica_span": {
            "Density": {
                "horizon_end_span": density_span[-1],
                "max_span_within_horizon": max(density_span),
                "threshold_crossing_ps": rounded_mapping(density_thresholds, density_span_crossings),
            },
            "Volume": {
                "horizon_end_span": volume_span[-1],
                "max_span_within_horizon": max(volume_span),
                "threshold_crossing_ps": rounded_mapping(volume_thresholds, volume_span_crossings),
            },
        },
        "suspected_sparse_basin_replica": suspected_sparse_basin,
        "replicas": replica_reports,
    }


def report_markdown(report: dict[str, object]) -> str:
    density_span = report["cross_replica_span"]["Density"]
    volume_span = report["cross_replica_span"]["Volume"]
    lines = [
        "# Gate I Early Bifurcation Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Gate I root: `{report['gate_i_root']}`",
        f"- Replicas analyzed: `{report['replica_count']}`",
        f"- Horizon analyzed: `{report['horizon_ps_analyzed']} ps`",
        f"- Equil end: `{report['equil_end_ps']} ps`",
        f"- Suspected sparse-basin replica: `{report['suspected_sparse_basin_replica']}`",
        "",
        "## Cross-Replica Span",
        f"- Density horizon-end span: `{density_span['horizon_end_span']:.6f}`",
        f"- Density max span within horizon: `{density_span['max_span_within_horizon']:.6f}`",
        f"- Density threshold crossings (ps): `{density_span['threshold_crossing_ps']}`",
        f"- Volume horizon-end span: `{volume_span['horizon_end_span']:.6f}`",
        f"- Volume max span within horizon: `{volume_span['max_span_within_horizon']:.6f}`",
        f"- Volume threshold crossings (ps): `{volume_span['threshold_crossing_ps']}`",
        "",
        "## Per Replica",
    ]
    for replica in report["replicas"]:
        density = replica["gap_to_other_mean"]["Density"]
        volume = replica["gap_to_other_mean"]["Volume"]
        lines.extend(
            [
                f"- {replica['replica_name']}: horizon density `{replica['horizon_end']['density']:.6f}`, "
                f"horizon volume `{replica['horizon_end']['volume']:.6f}`, "
                f"equil-end density `{replica['equil_end']['density']:.6f}`, "
                f"equil-end volume `{replica['equil_end']['volume']:.6f}`",
                f"- {replica['replica_name']} density gap-to-other-mean max `{density['max_gap_within_horizon']:.6f}`, "
                f"crossings `{density['threshold_crossing_ps']}`",
                f"- {replica['replica_name']} volume gap-to-other-mean max `{volume['max_gap_within_horizon']:.6f}`, "
                f"crossings `{volume['threshold_crossing_ps']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    density_thresholds = parse_thresholds(args.density_thresholds)
    volume_thresholds = parse_thresholds(args.volume_thresholds)
    gate_i_root = Path(args.gate_i_root).resolve()
    gmx = Path(args.gmx).resolve()

    replicas = []
    for replica_dir in replica_dirs(gate_i_root):
        replicas.append(
            {
                "replica_name": replica_dir.name,
                "replica_dir": replica_dir,
                "series": extract_replica_series(gmx, replica_dir),
            }
        )

    report = build_report(
        gate_i_root=gate_i_root,
        gmx=gmx,
        horizon_ps=args.horizon_ps,
        density_thresholds=density_thresholds,
        volume_thresholds=volume_thresholds,
        replica_series=replicas,
    )
    write_json(gate_i_root / f"{args.out_prefix}_report.json", report)
    write_text(gate_i_root / f"{args.out_prefix}_report.md", report_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
