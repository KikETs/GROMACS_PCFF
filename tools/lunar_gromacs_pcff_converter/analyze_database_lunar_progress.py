#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_WINDOWS = (10, 20, 50, 100, 200)
DEFAULT_SAMPLE_SIZE = 24


@dataclass(frozen=True)
class CaseRecord:
    trajectory_id: str
    row: dict[str, str]
    report_path: Path
    report: dict
    preferred_data_path: Path
    report_mtime: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def preferred_data_path(batch_root: Path, trajectory_id: str) -> Path:
    return batch_root / f"Traj_{trajectory_id}" / "build" / "lunar_pcff" / "chain_fixed_typed_nodup_IFF_nodup.data"


def report_path(batch_root: Path, trajectory_id: str) -> Path:
    return batch_root / f"Traj_{trajectory_id}" / "build" / "lunar_pcff_generation_report.json"


def isoformat_ts(epoch_seconds: float | None) -> str | None:
    if epoch_seconds is None:
        return None
    return datetime.fromtimestamp(epoch_seconds).isoformat(timespec="seconds")


def quantile_bins(values: list[float], bins: int, prefix: str) -> tuple[list[float], callable]:
    sorted_values = sorted(values)
    if not sorted_values:
        return [], lambda value: f"{prefix}1"
    cut_points: list[float] = []
    for idx in range(1, bins):
        pos = int(round(idx * (len(sorted_values) - 1) / bins))
        cut_points.append(sorted_values[pos])

    def classify(value: float) -> str:
        for idx, cut in enumerate(cut_points, start=1):
            if value <= cut:
                return f"{prefix}{idx}"
        return f"{prefix}{bins}"

    return cut_points, classify


def chemistry_tokens(smiles: str) -> list[str]:
    tokens = {"chem:base"}
    if any(ch.islower() for ch in smiles):
        tokens.add("chem:aromatic_or_lowercase")
    if "N" in smiles or "n" in smiles:
        tokens.add("chem:N")
    if "O" in smiles or "o" in smiles:
        tokens.add("chem:O")
    if "S" in smiles or "s" in smiles:
        tokens.add("chem:S")
    if "F" in smiles:
        tokens.add("chem:F")
    if "#" in smiles:
        tokens.add("chem:triple_bond")
    if any(ch.isdigit() for ch in smiles):
        tokens.add("chem:ring_digit")
    return sorted(tokens)


def chemistry_family(smiles: str) -> str:
    has_n = "N" in smiles or "n" in smiles
    has_o = "O" in smiles or "o" in smiles
    has_s = "S" in smiles or "s" in smiles
    has_f = "F" in smiles
    has_aromatic = any(ch.islower() for ch in smiles)
    has_triple = "#" in smiles

    if has_f and has_n and has_o:
        return "family:F/N/O"
    if has_s and has_n and has_o:
        return "family:S/N/O"
    if has_s and has_o:
        return "family:S/O"
    if has_aromatic and has_n and has_o:
        return "family:aromatic/N/O"
    if has_triple and has_n and has_o:
        return "family:nitrile/N/O"
    if has_n and has_o:
        return "family:N/O"
    if has_o:
        return "family:O_only"
    return "family:other"


def warning_bucket(warning_count: int | None) -> str:
    if warning_count is None:
        return "warn:unknown"
    if warning_count < 100:
        return "warn:<100"
    if warning_count < 300:
        return "warn:100-299"
    return "warn:>=300"


def rate_windows(paths: list[Path], windows: Iterable[int]) -> list[dict]:
    results: list[dict] = []
    ordered = sorted(paths, key=lambda path: path.stat().st_mtime)
    for window in windows:
        if len(ordered) < window:
            continue
        start = ordered[-window].stat().st_mtime
        end = ordered[-1].stat().st_mtime
        hours = (end - start) / 3600.0
        rate_per_hour = (window - 1) / hours if hours > 0 else math.inf
        results.append(
            {
                "window_size": window,
                "hours_spanned": round(hours, 3),
                "rate_per_hour": round(rate_per_hour, 3),
                "end_timestamp": isoformat_ts(end),
            }
        )
    return results


def active_generation_processes() -> list[dict]:
    cmd = [
        "ps",
        "-eo",
        "pid=,etimes=,pcpu=,pmem=,cmd=",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return []

    matches: list[dict] = []
    needles = (
        "generate_lunar_pcff_data_from_csv.py",
        "/extern/LUNAR/atom_typing.py",
        "/extern/LUNAR/all2lmp.py",
        "/extern/LUNAR/cell_builder.py",
    )
    for raw_line in result.stdout.splitlines():
        if not any(needle in raw_line for needle in needles):
            continue
        parts = raw_line.strip().split(maxsplit=4)
        if len(parts) < 5:
            continue
        pid, elapsed, pcpu, pmem, command = parts
        matches.append(
            {
                "pid": int(pid),
                "elapsed_seconds": int(elapsed),
                "pcpu": float(pcpu),
                "pmem": float(pmem),
                "command": command,
            }
        )
    return matches


def load_case_records(rows: list[dict[str, str]], batch_root: Path) -> tuple[list[CaseRecord], list[str]]:
    records: list[CaseRecord] = []
    failures: list[str] = []
    for row in rows:
        trajectory_id = row["Trajectory ID"].strip()
        preferred = preferred_data_path(batch_root, trajectory_id)
        report = report_path(batch_root, trajectory_id)
        if not report.is_file():
            continue
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except Exception:
            failures.append(trajectory_id)
            continue
        records.append(
            CaseRecord(
                trajectory_id=trajectory_id,
                row=row,
                report_path=report,
                report=payload,
                preferred_data_path=preferred,
                report_mtime=report.stat().st_mtime,
            )
        )
    return records, failures


def phase_breakdown(case: CaseRecord) -> dict | None:
    report = case.report
    if report.get("status") != "generated" or "elapsed_seconds" not in report:
        return None

    build = case.report_path.parent
    chain_fixed = build / "chain_fixed.mol2"
    typed_nodup = build / "lunar_typed" / "chain_fixed_typed_nodup.data"
    pcff_nodup = build / "lunar_pcff" / "chain_fixed_typed_nodup_IFF_nodup.data"
    cell_data = build / "cell" / "polymer_cell.data"
    if not (chain_fixed.exists() and typed_nodup.exists() and pcff_nodup.exists() and cell_data.exists()):
        return None

    start = case.report_mtime - float(report["elapsed_seconds"])
    t1 = chain_fixed.stat().st_mtime
    t2 = typed_nodup.stat().st_mtime
    t3 = pcff_nodup.stat().st_mtime
    t4 = cell_data.stat().st_mtime
    t5 = case.report_mtime
    total = float(report["elapsed_seconds"])

    def nonneg(value: float) -> float:
        return max(0.0, value)

    pre = nonneg(t1 - start)
    atom = nonneg(t2 - t1)
    all2 = nonneg(t3 - t2)
    cell = nonneg(t4 - t3)
    post = nonneg(t5 - t4)
    return {
        "trajectory_id": case.trajectory_id,
        "total_seconds": round(total, 3),
        "pre_lunar_seconds": round(pre, 3),
        "atom_typing_seconds": round(atom, 3),
        "all2lmp_seconds": round(all2, 3),
        "cell_builder_seconds": round(cell, 3),
        "postprocess_seconds": round(post, 3),
        "pre_lunar_share": round(pre / total, 6) if total > 0 else None,
        "lunar_all2lmp_warning_count": case.report.get("lunar_all2lmp_warning_count"),
        "n_chains": case.report.get("n_chains"),
        "degree_of_polymerization": case.report.get("degree_of_polymerization"),
    }


def summarize_phase_breakdown(records: list[CaseRecord]) -> dict:
    breakdowns = [item for item in (phase_breakdown(record) for record in records) if item is not None]
    if not breakdowns:
        return {"count": 0, "top_slow_cases": [], "top_all2lmp_outliers": []}

    def metric_summary(name: str) -> dict:
        values = [float(item[name]) for item in breakdowns]
        values_sorted = sorted(values)
        return {
            "median_seconds": round(statistics.median(values), 3),
            "mean_seconds": round(statistics.mean(values), 3),
            "p90_seconds": round(values_sorted[int(0.9 * (len(values_sorted) - 1))], 3),
            "max_seconds": round(values_sorted[-1], 3),
        }

    top_slow = sorted(breakdowns, key=lambda item: item["total_seconds"], reverse=True)[:15]
    top_all2 = sorted(breakdowns, key=lambda item: item["all2lmp_seconds"], reverse=True)[:15]
    return {
        "count": len(breakdowns),
        "pre_lunar_seconds": metric_summary("pre_lunar_seconds"),
        "atom_typing_seconds": metric_summary("atom_typing_seconds"),
        "all2lmp_seconds": metric_summary("all2lmp_seconds"),
        "cell_builder_seconds": metric_summary("cell_builder_seconds"),
        "postprocess_seconds": metric_summary("postprocess_seconds"),
        "pre_lunar_share": metric_summary("pre_lunar_share"),
        "top_slow_cases": top_slow,
        "top_all2lmp_outliers": top_all2,
    }


def elapsed_summary(records: list[CaseRecord]) -> dict:
    elapsed = [
        float(record.report["elapsed_seconds"])
        for record in records
        if record.report.get("status") == "generated" and "elapsed_seconds" in record.report
    ]
    if not elapsed:
        return {"count": 0}
    elapsed_sorted = sorted(elapsed)
    return {
        "count": len(elapsed_sorted),
        "mean_seconds": round(statistics.mean(elapsed_sorted), 3),
        "median_seconds": round(statistics.median(elapsed_sorted), 3),
        "p90_seconds": round(elapsed_sorted[int(0.9 * (len(elapsed_sorted) - 1))], 3),
        "max_seconds": round(elapsed_sorted[-1], 3),
    }


def generated_case_tokens(
    case: CaseRecord,
    dp_bin_for: callable,
    molality_bin_for: callable,
) -> list[str]:
    report = case.report
    row = case.row
    tokens = [
        dp_bin_for(float(row["Degree of Polymerization"])),
        molality_bin_for(float(row["Molality"])),
        warning_bucket(report.get("lunar_all2lmp_warning_count")),
        chemistry_family(row["SMILES"]),
    ]
    tokens.extend(chemistry_tokens(row["SMILES"]))
    return sorted(set(tokens))


def greedy_sample_plan(
    records: list[CaseRecord],
    rows: list[dict[str, str]],
    sample_size: int,
) -> dict:
    if not records:
        return {
            "sample_size_target": sample_size,
            "selected_count": 0,
            "selected_cases": [],
            "coverage": {},
            "note": "No generated cases are available for sampling.",
        }

    dp_values = [float(row["Degree of Polymerization"]) for row in rows]
    molality_values = [float(row["Molality"]) for row in rows]
    dp_cuts, dp_bin_for = quantile_bins(dp_values, 4, "dp:q")
    molality_cuts, molality_bin_for = quantile_bins(molality_values, 3, "molality:q")

    token_map: dict[str, list[str]] = {}
    strata_by_id: dict[str, tuple[str, str, str, str]] = {}
    for case in records:
        dp_bin = dp_bin_for(float(case.row["Degree of Polymerization"]))
        molality_bin = molality_bin_for(float(case.row["Molality"]))
        family = chemistry_family(case.row["SMILES"])
        warn = warning_bucket(case.report.get("lunar_all2lmp_warning_count"))
        token_map[case.trajectory_id] = generated_case_tokens(case, dp_bin_for, molality_bin_for)
        strata_by_id[case.trajectory_id] = (dp_bin, molality_bin, family, warn)

    universe = sorted({token for tokens in token_map.values() for token in tokens})
    strata_counts = Counter(strata_by_id.values())
    chosen_ids: set[str] = set()
    selected: list[CaseRecord] = []

    def pick_best(candidates: list[CaseRecord]) -> CaseRecord | None:
        available = [case for case in candidates if case.trajectory_id not in chosen_ids]
        if not available:
            return None
        available.sort(key=lambda case: (float(case.report.get("elapsed_seconds", math.inf)), int(case.trajectory_id)))
        return available[0]

    all_dp_bins = sorted({stratum[0] for stratum in strata_counts})
    all_molality_bins = sorted({stratum[1] for stratum in strata_counts})
    all_families = sorted({stratum[2] for stratum in strata_counts})
    all_warn_buckets = sorted({stratum[3] for stratum in strata_counts})

    for dp_bin in all_dp_bins:
        if len(selected) >= sample_size:
            break
        pick = pick_best([case for case in records if strata_by_id[case.trajectory_id][0] == dp_bin])
        if pick is not None:
            selected.append(pick)
            chosen_ids.add(pick.trajectory_id)

    for molality_bin in all_molality_bins:
        if len(selected) >= sample_size:
            break
        pick = pick_best([case for case in records if strata_by_id[case.trajectory_id][1] == molality_bin])
        if pick is not None:
            selected.append(pick)
            chosen_ids.add(pick.trajectory_id)

    for family in all_families:
        if len(selected) >= sample_size:
            break
        pick = pick_best([case for case in records if strata_by_id[case.trajectory_id][2] == family])
        if pick is not None:
            selected.append(pick)
            chosen_ids.add(pick.trajectory_id)

    for warn in all_warn_buckets:
        if len(selected) >= sample_size:
            break
        pick = pick_best([case for case in records if strata_by_id[case.trajectory_id][3] == warn])
        if pick is not None:
            selected.append(pick)
            chosen_ids.add(pick.trajectory_id)

    unique_strata = sorted(strata_counts.items(), key=lambda item: (item[1], item[0]))
    for stratum, _count in unique_strata:
        if len(selected) >= sample_size:
            break
        candidates = [
            case
            for case in records
            if strata_by_id[case.trajectory_id] == stratum and case.trajectory_id not in chosen_ids
        ]
        if not candidates:
            continue
        pick = pick_best(candidates)
        if pick is None:
            continue
        selected.append(pick)
        chosen_ids.add(pick.trajectory_id)

    remaining = [
        case
        for case in sorted(records, key=lambda case: (float(case.report.get("elapsed_seconds", math.inf)), int(case.trajectory_id)))
        if case.trajectory_id not in chosen_ids
    ]
    selected_strata_counts = Counter(strata_by_id[case.trajectory_id] for case in selected)
    for case in remaining:
        if len(selected) >= sample_size:
            break
        stratum = strata_by_id[case.trajectory_id]
        if selected_strata_counts[stratum] > 0:
            continue
        selected.append(case)
        chosen_ids.add(case.trajectory_id)
        selected_strata_counts[stratum] += 1

    for case in remaining:
        if len(selected) >= sample_size:
            break
        if case.trajectory_id in chosen_ids:
            continue
        selected.append(case)
        chosen_ids.add(case.trajectory_id)

    selected_cases = []
    for case in selected:
        selected_cases.append(
            {
                "trajectory_id": case.trajectory_id,
                "elapsed_seconds": case.report.get("elapsed_seconds"),
                "degree_of_polymerization": float(case.row["Degree of Polymerization"]),
                "molality": float(case.row["Molality"]),
                "n_chains": case.report.get("n_chains"),
                "warning_bucket": warning_bucket(case.report.get("lunar_all2lmp_warning_count")),
                "lunar_all2lmp_warning_count": case.report.get("lunar_all2lmp_warning_count"),
                "chemistry_family": chemistry_family(case.row["SMILES"]),
                "stratum": list(strata_by_id[case.trajectory_id]),
                "tokens": token_map[case.trajectory_id],
                "smiles": case.row["SMILES"],
            }
        )

    selected_token_coverage = Counter(token for case in selected_cases for token in case["tokens"])
    selected_dp_bins = sorted({case["stratum"][0] for case in selected_cases})
    selected_molality_bins = sorted({case["stratum"][1] for case in selected_cases})
    selected_families = sorted({case["stratum"][2] for case in selected_cases})
    selected_warn_buckets = sorted({case["stratum"][3] for case in selected_cases})
    return {
        "sample_size_target": sample_size,
        "selected_count": len(selected_cases),
        "selected_cases": selected_cases,
        "coverage": {
            "universe_token_count": len(universe),
            "selected_token_count": len(selected_token_coverage),
            "uncovered_tokens": sorted(set(universe) - set(selected_token_coverage)),
            "token_counts": dict(sorted(selected_token_coverage.items())),
            "dp_cut_points": dp_cuts,
            "molality_cut_points": molality_cuts,
            "selected_dp_bins": selected_dp_bins,
            "selected_molality_bins": selected_molality_bins,
            "selected_families": selected_families,
            "selected_warning_buckets": selected_warn_buckets,
            "selected_unique_strata": len({tuple(case["stratum"]) for case in selected_cases}),
            "available_unique_strata": len(strata_counts),
        },
        "recommended_claim_boundary": (
            "Do not claim full frozen-database completion. "
            "Use the selected generated-case sample as a stratified smoke set, and only claim "
            "parser->mapping->emission->grompp support for cases whose artifacts are actually generated and validated."
        ),
    }


def report_failures(records: list[CaseRecord]) -> list[dict]:
    failures = []
    for case in records:
        if case.report.get("status") != "failure":
            continue
        failures.append(
            {
                "trajectory_id": case.trajectory_id,
                "elapsed_seconds": case.report.get("elapsed_seconds"),
                "error": case.report.get("error"),
            }
        )
    return sorted(failures, key=lambda item: (item["elapsed_seconds"] or 0), reverse=True)


def make_markdown(summary: dict) -> str:
    lines: list[str] = []
    progress = summary["progress"]
    lines.append("# LUNAR Database Progress Analysis")
    lines.append("")
    lines.append(f"- Generated `.data`: {progress['generated_preferred_data_count']} / {progress['csv_row_count']}")
    lines.append(f"- Remaining: {progress['remaining_missing_count']}")
    lines.append(f"- Completion: {progress['completion_percent']:.2f}%")
    lines.append(f"- Active generation-related processes: {progress['active_process_count']}")
    lines.append("")

    lines.append("## Throughput")
    lines.append("")
    for item in summary["throughput"]["windows"]:
        eta_hours = item.get("eta_hours")
        eta_days = item.get("eta_days")
        lines.append(
            f"- Last {item['window_size']} outputs: {item['rate_per_hour']:.2f}/h over {item['hours_spanned']:.2f} h; "
            f"ETA {eta_hours:.1f} h ({eta_days:.1f} d)"
        )
    lines.append("")

    phase = summary["phase_breakdown"]
    if phase.get("count", 0):
        lines.append("## Bottleneck")
        lines.append("")
        lines.append(
            f"- Pre-LUNAR stage share median: {phase['pre_lunar_share']['median_seconds']:.3f}; "
            f"mean: {phase['pre_lunar_share']['mean_seconds']:.3f}"
        )
        lines.append(
            f"- Median phase times (s): pre={phase['pre_lunar_seconds']['median_seconds']}, "
            f"atom={phase['atom_typing_seconds']['median_seconds']}, "
            f"all2lmp={phase['all2lmp_seconds']['median_seconds']}, "
            f"cell={phase['cell_builder_seconds']['median_seconds']}"
        )
        lines.append("- Interpretation: most elapsed time is consumed before `chain_fixed.mol2` appears; LUNAR stages are usually sub-second after that.")
        lines.append("")
        lines.append("### Slow Cases")
        lines.append("")
        for item in phase["top_slow_cases"][:10]:
            lines.append(
                f"- Traj_{item['trajectory_id']}: total={item['total_seconds']:.1f}s, "
                f"pre={item['pre_lunar_seconds']:.1f}s, all2lmp={item['all2lmp_seconds']:.1f}s, "
                f"warnings={item['lunar_all2lmp_warning_count']}"
            )
        lines.append("")

    failures = summary["failures"]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for item in failures[:10]:
            lines.append(f"- Traj_{item['trajectory_id']}: {item['error']}")
        lines.append("")

    sampling = summary["sampling_strategy"]
    lines.append("## Sampling Claim Path")
    lines.append("")
    lines.append(f"- Target sample size: {sampling['sample_size_target']}")
    lines.append(f"- Selected generated cases: {sampling['selected_count']}")
    lines.append(f"- Uncovered tokens after selection: {len(sampling['coverage'].get('uncovered_tokens', []))}")
    lines.append(f"- Claim boundary: {sampling['recommended_claim_boundary']}")
    lines.append("")
    lines.append("### Selected Cases")
    lines.append("")
    for item in sampling["selected_cases"][:24]:
        lines.append(
            f"- Traj_{item['trajectory_id']}: DP={item['degree_of_polymerization']}, molality={item['molality']}, "
            f"warnings={item['lunar_all2lmp_warning_count']}, tokens={', '.join(item['tokens'])}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    rows = read_csv_rows(args.csv.resolve())
    batch_root = args.batch_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    preferred_paths = [preferred_data_path(batch_root, row["Trajectory ID"].strip()) for row in rows]
    existing_preferred = [path for path in preferred_paths if path.is_file()]
    records, bad_reports = load_case_records(rows, batch_root)

    record_map = {record.trajectory_id: record for record in records}
    failures = report_failures(records)
    phase = summarize_phase_breakdown(records)
    throughput = {
        "windows": [],
    }
    for item in rate_windows(existing_preferred, DEFAULT_WINDOWS):
        rate = item["rate_per_hour"]
        if rate and math.isfinite(rate) and rate > 0:
            item["eta_hours"] = round((len(rows) - len(existing_preferred)) / rate, 2)
            item["eta_days"] = round(item["eta_hours"] / 24.0, 2)
        throughput["windows"].append(item)

    active = active_generation_processes()
    sampling = greedy_sample_plan(
        [
            record
            for record in records
            if record.report.get("status") == "generated" and record.preferred_data_path.is_file()
        ],
        rows,
        sample_size=args.sample_size,
    )

    summary = {
        "generated_at": isoformat_ts(datetime.now().timestamp()),
        "inputs": {
            "csv": str(args.csv.resolve()),
            "batch_root": str(batch_root),
            "out_dir": str(out_dir),
            "sample_size": args.sample_size,
        },
        "progress": {
            "csv_row_count": len(rows),
            "generated_preferred_data_count": len(existing_preferred),
            "remaining_missing_count": len(rows) - len(existing_preferred),
            "completion_percent": round(100.0 * len(existing_preferred) / len(rows), 3) if rows else 0.0,
            "report_count": len(records),
            "bad_report_count": len(bad_reports),
            "active_process_count": len(active),
            "active_processes": active,
        },
        "throughput": throughput,
        "elapsed_summary": elapsed_summary(records),
        "phase_breakdown": phase,
        "failures": failures,
        "sampling_strategy": sampling,
    }

    json_path = out_dir / "database_lunar_progress_analysis.json"
    md_path = out_dir / "database_lunar_progress_analysis.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(make_markdown(summary), encoding="utf-8")

    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
