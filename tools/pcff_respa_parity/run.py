from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
DEFAULT_BINARY = BUILD_DIR / "bin" / "mdrun-non-integrator-test"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "tests" / "reference_results" / "m6_respa"
DEFAULT_ACTUAL_ROOT = DEFAULT_REFERENCE_ROOT / "last_run_actual"
DEFAULT_SUMMARY_DIR = DEFAULT_REFERENCE_ROOT / "last_run_compare"
LEGACY_COMPARE_SUBDIR_NAME = "legacy_m6_parity"
OFFLINE_ORACLE_SCRIPT = REPO_ROOT / "tools" / "pcff_respa_parity" / "offline_oracle_compare_v1.py"
OFFLINE_ORACLE_DEFAULT_FIXTURE = "dense_oligomer"
OFFLINE_ORACLE_DEFAULT_DT_LABEL = "dt_0p0005"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LAMMPS run_style respa parity harness.")
    parser.add_argument(
        "--build-target",
        default="mdrun-non-integrator-test",
        help="CMake target to build before running the GROMACS side of the harness.",
    )
    parser.add_argument(
        "--binary",
        default=str(DEFAULT_BINARY),
        help="Test binary that dumps exact r-RESPA observables.",
    )
    parser.add_argument(
        "--gtest-filter",
        default="PcffRespaObservableDump/*",
        help="GTest filter for the exact r-RESPA observable dump tests.",
    )
    parser.add_argument(
        "--reference-root",
        default=str(DEFAULT_REFERENCE_ROOT),
        help="Directory containing or receiving frozen LAMMPS reference artifacts.",
    )
    parser.add_argument(
        "--actual-root",
        default=str(DEFAULT_ACTUAL_ROOT),
        help="Directory for actual GROMACS exact r-RESPA JSON summaries.",
    )
    parser.add_argument(
        "--summary-dir",
        default=str(DEFAULT_SUMMARY_DIR),
        help="Directory for comparison JSON outputs.",
    )
    parser.add_argument(
        "--prepare-reference",
        action="store_true",
        help="Regenerate the frozen LAMMPS run_style respa reference before comparison.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the build step and run the binary directly.",
    )
    parser.add_argument(
        "--lammps-cmd",
        default="/home/user/.local/bin/lmp",
        help="LAMMPS executable forwarded to prepare_reference.py when requested.",
    )
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="System id to process. Repeat to select multiple systems.",
    )
    parser.add_argument(
        "--outer-steps",
        type=int,
        default=5,
        help="Number of outer r-RESPA steps to run in both LAMMPS and GROMACS diagnostics.",
    )
    parser.add_argument(
        "--pair14-level",
        type=int,
        default=1,
        help="GROMACS mts-respa-pair14-level used for the actual run.",
    )
    parser.add_argument(
        "--nested-prototype",
        action="store_true",
        help="Enable the CPU nested exact-r-RESPA prototype via GMX_EXACT_RESPA_NESTED_PROTOTYPE=1.",
    )
    parser.add_argument(
        "--offline-oracle-mode",
        choices=("auto", "off", "only"),
        default="auto",
        help=(
            "Control the current-fixture authoritative offline comparator workflow. "
            "'auto' runs it after the regular parity harness, 'only' skips the parity harness "
            "and runs only the authoritative dense_oligomer/dt_0p0005 comparator, "
            "'off' disables it."
        ),
    )
    parser.add_argument(
        "--offline-oracle-fixture",
        default=OFFLINE_ORACLE_DEFAULT_FIXTURE,
        help=(
            "Fixture forwarded to offline_oracle_compare_v1.py. "
            "The authoritative rule is validated only for dense_oligomer."
        ),
    )
    parser.add_argument(
        "--offline-oracle-dt-label",
        default=OFFLINE_ORACLE_DEFAULT_DT_LABEL,
        help="Timestep label forwarded to offline_oracle_compare_v1.py.",
    )
    parser.add_argument(
        "--offline-oracle-out",
        default=None,
        help=(
            "Output directory for authoritative offline comparator artifacts. "
            "Default: <summary-dir>/authoritative_dense_oligomer_dt_0p0005"
        ),
    )
    return parser.parse_args()


def default_offline_oracle_out(summary_dir: Path, fixture: str, dt_label: str) -> Path:
    return summary_dir / f"authoritative_{fixture}_{dt_label}"


def default_truth_source_pointer(summary_dir: Path) -> Path:
    return summary_dir / "plain_facing_truth_source.json"


def default_legacy_compare_out(summary_dir: Path) -> Path:
    return summary_dir / LEGACY_COMPARE_SUBDIR_NAME


def is_current_authoritative_fixture(args: argparse.Namespace) -> bool:
    return (
        args.offline_oracle_fixture == OFFLINE_ORACLE_DEFAULT_FIXTURE
        and args.offline_oracle_dt_label == OFFLINE_ORACLE_DEFAULT_DT_LABEL
    )


def run_offline_oracle_compare(args: argparse.Namespace) -> dict:
    summary_dir = Path(args.summary_dir).resolve()
    out_dir = (
        Path(args.offline_oracle_out).resolve()
        if args.offline_oracle_out
        else default_offline_oracle_out(summary_dir, args.offline_oracle_fixture, args.offline_oracle_dt_label)
    )
    command = [
        sys.executable,
        str(OFFLINE_ORACLE_SCRIPT),
        "--fixture",
        args.offline_oracle_fixture,
        "--dt-label",
        args.offline_oracle_dt_label,
        "--out",
        str(out_dir),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    summary_path = out_dir / "offline_oracle_compare_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Expected offline oracle summary at {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["authoritative_output_root"] = str(out_dir)
    summary["entrypoint"] = str(Path(__file__).resolve())
    summary["mode"] = args.offline_oracle_mode
    return summary


def persist_authoritative_truth_source(summary_dir: Path, offline_summary: dict) -> None:
    truth_source = {
        "schema_version": 1,
        "truth_source_type": "authoritative_plain_facing_offline_compare",
        "fixture": offline_summary["fixture"],
        "dt_label": offline_summary["dt_label"],
        "current_fixture_only_rule": offline_summary["current_fixture_only_rule"],
        "authoritative_generator": str(OFFLINE_ORACLE_SCRIPT),
        "authoritative_entrypoint": str(Path(__file__).resolve()),
        "authoritative_output_root": offline_summary["authoritative_output_root"],
        "decisive_row": offline_summary["decisive_row"],
        "legacy_compare_blocking": {
            "compare_py_requires_override": True,
            "legacy_step3_step4_outputs_non_authoritative": True,
        },
        "notes": [
            "This file is the single machine-visible plain-facing truth source for dense_oligomer/dt_0p0005 in the default workflow root.",
            "compare.py outputs remain valid for M6 parity diagnostics but are not the authoritative dense plain-facing comparator outputs.",
        ],
    }
    path = default_truth_source_pointer(summary_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(truth_source, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def deambiguate_default_root(summary_dir: Path) -> None:
    truth_pointer = default_truth_source_pointer(summary_dir)
    legacy_dir = default_legacy_compare_out(summary_dir)
    moved_files: list[str] = []
    legacy_dir.mkdir(parents=True, exist_ok=True)
    for path in summary_dir.glob("*.json"):
        if path.name == truth_pointer.name:
            continue
        destination = legacy_dir / path.name
        if destination.exists():
            destination.unlink()
        shutil.move(str(path), str(destination))
        moved_files.append(path.name)

    manifest = {
        "schema_version": 1,
        "artifact_role": "legacy_m6_parity_output_root",
        "non_authoritative": True,
        "authoritative_pointer": str(truth_pointer),
        "moved_files": sorted(moved_files),
        "notes": [
            "Files in this directory are legacy M6 parity outputs and are not the authoritative dense_oligomer/dt_0p0005 plain-facing truth source.",
            "Use the authoritative pointer in the default workflow root instead.",
        ],
    }
    (legacy_dir / "legacy_output_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir).resolve()
    if args.offline_oracle_mode == "only":
        offline_summary = run_offline_oracle_compare(args)
        persist_authoritative_truth_source(summary_dir, offline_summary)
        if is_current_authoritative_fixture(args):
            deambiguate_default_root(summary_dir)
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "mode": "offline_oracle_only",
                    "offline_oracle_compare": offline_summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    reference_root = Path(args.reference_root).resolve()
    actual_root = Path(args.actual_root).resolve()
    legacy_compare_out = default_legacy_compare_out(summary_dir) if is_current_authoritative_fixture(args) else summary_dir

    if args.prepare_reference:
        command = [
            sys.executable,
            str(REPO_ROOT / "tools" / "pcff_respa_parity" / "prepare_reference.py"),
            "--out",
            str(reference_root),
            "--lammps-cmd",
            args.lammps_cmd,
            "--outer-steps",
            str(args.outer_steps),
        ]
        for system_id in args.systems or []:
            command.extend(["--system", system_id])
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    if not args.skip_build:
        subprocess.run(
            ["cmake", "--build", str(BUILD_DIR), "--target", args.build_target],
            cwd=REPO_ROOT,
            check=True,
        )

    if actual_root.exists():
        shutil.rmtree(actual_root)
    actual_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["GMX_PCFF_RESPA_ACTUAL_DIR"] = str(actual_root)
    env["GMX_PCFF_RESPA_OUTER_STEPS"] = str(args.outer_steps)
    env["GMX_PCFF_RESPA_PAIR14_LEVEL"] = str(args.pair14_level)
    if args.nested_prototype:
        env["GMX_EXACT_RESPA_NESTED_PROTOTYPE"] = "1"
    subprocess.run(
        [args.binary, f"--gtest_filter={args.gtest_filter}"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    compare_command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "pcff_respa_parity" / "compare.py"),
        "--allow-non-authoritative-plain-facing-use",
        "--authoritative-pointer",
        str(default_truth_source_pointer(summary_dir)),
        "--reference-root",
        str(reference_root),
        "--actual-root",
        str(actual_root),
        "--out",
        str(legacy_compare_out),
    ]
    for system_id in args.systems or []:
        compare_command.extend(["--system", system_id])
    subprocess.run(compare_command, cwd=REPO_ROOT, check=True)

    summary = json.loads((legacy_compare_out / "comparison_summary.json").read_text(encoding="utf-8"))
    if args.offline_oracle_mode == "auto":
        summary["offline_oracle_compare"] = run_offline_oracle_compare(args)
        persist_authoritative_truth_source(summary_dir, summary["offline_oracle_compare"])
        if is_current_authoritative_fixture(args):
            deambiguate_default_root(summary_dir)
            summary["legacy_compare_output_root"] = str(default_legacy_compare_out(summary_dir))
            summary["plain_facing_truth_source"] = str(default_truth_source_pointer(summary_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
