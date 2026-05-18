#!/usr/bin/env python3
"""Create PolyGen multi-system validation manifests.

The manifest is intentionally explicit: systems, replicas, lanes, durations,
and worker assignment are written to CSV before any expensive MD starts.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE_CSV = Path(
    "MY_PAPER_RELATED/Decoder_Only/data/simulation-trajectory-aggregate.csv"
)
DEFAULT_OUTDIR = Path(
    "GROMACS_PCFF/output/polygen_multisystem_validation_20260512_m1p50"
)

LANES = ("lammps_cpu", "gmx_cpu", "gmx_gpu")

WORKERS = {
    "local_main": {"host": "localhost", "ssh_port": "", "ssh_target": ""},
    "remote_mid_a": {
        "host": "100.110.123.78",
        "ssh_port": "",
        "ssh_target": "user@100.110.123.78",
    },
    "remote_mid_b": {
        "host": "100.120.161.20",
        "ssh_port": "",
        "ssh_target": "user@100.120.161.20",
    },
}


@dataclass(frozen=True)
class SelectionConfig:
    high_n: int
    middle_n: int
    low_n: int
    replicas: int
    main_ns: float
    long_ns: float
    long_high_n: int
    long_middle_n: int
    long_low_n: int
    random_state: int


def slug(text: object) -> str:
    raw = str(text)
    raw = raw.replace("[*]", "star").replace("*", "star")
    raw = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()
    return raw[:64] or "system"


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def unique_by_smiles(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    work = df.copy()
    work["_smiles_key"] = work[smiles_col].astype(str).str.strip()
    work = work[work["_smiles_key"] != ""]
    return work.drop_duplicates("_smiles_key", keep="first").drop(columns=["_smiles_key"])


def select_systems(
    source_csv: Path,
    *,
    config: SelectionConfig,
    id_col: str = "Trajectory ID",
    smiles_col: str = "SMILES",
    rank_col: str = "CONDUCTIVITY",
) -> pd.DataFrame:
    df = pd.read_csv(source_csv)
    require_columns(df, [id_col, smiles_col, rank_col], source_csv)

    work = df.dropna(subset=[id_col, smiles_col, rank_col]).copy()
    work[rank_col] = pd.to_numeric(work[rank_col], errors="coerce")
    work = work.dropna(subset=[rank_col])
    work = work.sort_values(rank_col, ascending=False)
    work = unique_by_smiles(work, smiles_col)

    if len(work) < config.high_n + config.middle_n + config.low_n:
        raise ValueError(
            f"Need at least {config.high_n + config.middle_n + config.low_n} unique systems, "
            f"found {len(work)}"
        )

    high = work.head(config.high_n).copy()
    low = work.tail(config.low_n).sort_values(rank_col, ascending=True).copy()

    used_smiles = set(high[smiles_col].astype(str)) | set(low[smiles_col].astype(str))
    middle_pool = work[~work[smiles_col].astype(str).isin(used_smiles)].copy()
    median = float(middle_pool[rank_col].median())
    middle_pool["_median_distance"] = (middle_pool[rank_col] - median).abs()
    middle = (
        middle_pool.sort_values(["_median_distance", rank_col], ascending=[True, False])
        .head(config.middle_n)
        .drop(columns=["_median_distance"])
        .copy()
    )

    blocks = []
    for category, block in (("high", high), ("middle", middle), ("low", low)):
        block = block.reset_index(drop=True).copy()
        block["category"] = category
        block["category_rank"] = range(1, len(block) + 1)
        blocks.append(block)

    systems = pd.concat(blocks, ignore_index=True)
    systems["source_csv"] = str(source_csv)
    systems["selection_rule"] = (
        "high=top unique SMILES by CONDUCTIVITY; "
        "low=bottom unique SMILES by CONDUCTIVITY; "
        "middle=unique SMILES nearest global median CONDUCTIVITY after excluding high/low"
    )
    systems["system_key"] = [
        f"{row.category}_{int(row.category_rank):03d}_tid{int(float(row[id_col]))}_{slug(row[smiles_col])}"
        for _, row in systems.iterrows()
    ]

    front = [
        "system_key",
        "category",
        "category_rank",
        id_col,
        smiles_col,
        rank_col,
        "selection_rule",
        "source_csv",
    ]
    rest = [c for c in systems.columns if c not in front]
    return systems[front + rest]


def worker_for_main(category: str, category_rank: int) -> str:
    if category == "middle":
        return "remote_mid_a" if int(category_rank) <= 4 else "remote_mid_b"
    return "local_main"


def build_jobs(
    systems: pd.DataFrame,
    *,
    config: SelectionConfig,
    outdir: Path,
    id_col: str = "Trajectory ID",
    smiles_col: str = "SMILES",
    rank_col: str = "CONDUCTIVITY",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_blocks = []
    for category, n in (
        ("high", config.long_high_n),
        ("middle", config.long_middle_n),
        ("low", config.long_low_n),
    ):
        block = (
            systems[systems["category"] == category]
            .sort_values("category_rank")
            .head(int(n))
            .copy()
        )
        long_blocks.append(block)
    long_systems = pd.concat(long_blocks, ignore_index=True)

    rows: list[dict[str, object]] = []

    def add_job_rows(system_row: pd.Series, run_group: str, duration_ns: float, worker_role: str) -> None:
        worker = WORKERS[worker_role]
        for replica in range(1, int(config.replicas) + 1):
            replica_seed = int(float(system_row[id_col])) * 100 + replica
            for lane in LANES:
                rows.append(
                    {
                        "job_id": (
                            f"{run_group}__{system_row['system_key']}__r{replica:02d}__{lane}"
                        ),
                        "run_group": run_group,
                        "duration_ns": float(duration_ns),
                        "system_key": system_row["system_key"],
                        "category": system_row["category"],
                        "category_rank": int(system_row["category_rank"]),
                        "trajectory_id": int(float(system_row[id_col])),
                        "psmiles": system_row[smiles_col],
                        "reference_conductivity_s_cm": float(system_row[rank_col]),
                        "replica": replica,
                        "replica_seed": replica_seed,
                        "lane": lane,
                        "worker_role": worker_role,
                        "worker_host": worker["host"],
                        "ssh_port": worker["ssh_port"],
                        "ssh_target": worker["ssh_target"],
                        "local_run_root": str(
                            outdir
                            / "runs"
                            / run_group
                            / worker_role
                            / str(system_row["system_key"])
                            / f"replica_{replica:02d}"
                            / lane
                        ),
                        "status": "planned",
                    }
                )

    for _, row in systems.iterrows():
        add_job_rows(
            row,
            run_group="main20",
            duration_ns=config.main_ns,
            worker_role=worker_for_main(str(row["category"]), int(row["category_rank"])),
        )

    for _, row in long_systems.iterrows():
        add_job_rows(
            row,
            run_group="long50",
            duration_ns=config.long_ns,
            worker_role="local_main",
        )

    jobs = pd.DataFrame(rows)
    return jobs, long_systems


def write_manifests(
    source_csv: Path,
    outdir: Path,
    *,
    config: SelectionConfig,
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_dir = outdir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    systems = select_systems(source_csv, config=config)
    jobs, long_systems = build_jobs(systems, config=config, outdir=outdir)

    paths = {
        "systems": manifest_dir / "systems.csv",
        "jobs": manifest_dir / "jobs.csv",
        "long_systems": manifest_dir / "long_window_systems.csv",
        "config": manifest_dir / "manifest_config.json",
    }
    systems.to_csv(paths["systems"], index=False)
    jobs.to_csv(paths["jobs"], index=False)
    long_systems.to_csv(paths["long_systems"], index=False)

    for role in sorted(WORKERS):
        role_jobs = jobs[jobs["worker_role"] == role].copy()
        role_path = manifest_dir / f"jobs_{role}.csv"
        role_jobs.to_csv(role_path, index=False)
        paths[f"jobs_{role}"] = role_path

    payload = {
        "source_csv": str(source_csv),
        "outdir": str(outdir),
        "lanes": list(LANES),
        "workers": WORKERS,
        "selection": config.__dict__,
        "counts": {
            "systems": int(len(systems)),
            "main20_lane_jobs": int(len(jobs[jobs["run_group"] == "main20"])),
            "long50_lane_jobs": int(len(jobs[jobs["run_group"] == "long50"])),
            "all_lane_jobs": int(len(jobs)),
        },
    }
    paths["config"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {key: str(path) for key, path in paths.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--high", type=int, default=4)
    p.add_argument("--middle", type=int, default=8)
    p.add_argument("--low", type=int, default=4)
    p.add_argument("--replicas", type=int, default=3)
    p.add_argument("--main-ns", type=float, default=20.0)
    p.add_argument("--long-ns", type=float, default=50.0)
    p.add_argument("--long-high", type=int, default=1)
    p.add_argument("--long-middle", type=int, default=2)
    p.add_argument("--long-low", type=int, default=1)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = SelectionConfig(
        high_n=args.high,
        middle_n=args.middle,
        low_n=args.low,
        replicas=args.replicas,
        main_ns=args.main_ns,
        long_ns=args.long_ns,
        long_high_n=args.long_high,
        long_middle_n=args.long_middle,
        long_low_n=args.long_low,
        random_state=args.random_state,
    )
    paths = write_manifests(args.source_csv, args.outdir, config=config)
    print(json.dumps(paths, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
