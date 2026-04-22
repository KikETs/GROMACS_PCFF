from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

from freeze_gate_a_oracle import (
    DEFAULT_GMX,
    FIXTURE_ROOT,
    LEVEL_FACTORS,
    RESPA_ENERGY_INTERVAL,
    base_env,
    build_trace_atom_indices,
    capture_output,
    command_record,
    env_delta,
    make_exact_respa_mdp,
    parse_event_trace,
    parse_merge_trace_dir,
    read_gro_atom_count,
    write_commands_script,
    write_text,
)
from validate_gate_b_nb_gpu import (
    compare_event_trace,
    parse_gpu_support,
    parse_precision_mode,
    run_command_allow_failure,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "exact_respa_gpu_hybrid_force_only"
SYSTEMS = ("small_oligomer", "small_salt_polymer_box")
GPU_REPRODUCIBILITY_NOTE = (
    "Binary reproducibility (-reprod) is not enabled because GROMACS rejects -nb gpu together with -reprod."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the admitted exact r-RESPA hybrid OpenMP+GPU nonbonded force-only shape."
        )
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step.")
    parser.add_argument("--build-dir", default=None, help="Optional explicit build directory.")
    parser.add_argument("--build-target", default="gmx", help="CMake target to build.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks for mdrun.")
    parser.add_argument(
        "--ntomp",
        type=int,
        default=2,
        help="OpenMP threads for mdrun. The current audited GPU admission shape requires 2.",
    )
    parser.add_argument("--outer-steps", type=int, default=5, help="Number of exact r-RESPA outer steps.")
    parser.add_argument("--pair14-level", type=int, default=1, help="Exact r-RESPA pair14 level.")
    parser.add_argument(
        "--force-tol",
        type=float,
        default=1.0e-3,
        help="Maximum allowed per-atom force component delta between CPU and GPU force-only runs.",
    )
    parser.add_argument(
        "--aggregate-force-tol",
        type=float,
        default=None,
        help=(
            "Maximum allowed component delta for per-level aggregate merge-trace vector sums. "
            "Defaults to force-tol multiplied by the fixture atom count."
        ),
    )
    return parser.parse_args()


def maybe_build(args: argparse.Namespace, gmx: Path) -> None:
    if args.skip_build:
        return
    build_dir = Path(args.build_dir).resolve() if args.build_dir is not None else gmx.parents[1]
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", args.build_target, "-j4"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
    )


def make_force_only_mdp(outer_steps: int, pair14_level: int) -> str:
    nsteps = outer_steps * RESPA_ENERGY_INTERVAL
    disabled_interval = nsteps + RESPA_ENERGY_INTERVAL
    mdp = make_exact_respa_mdp(outer_steps, pair14_level)
    replacements = {
        "title": "title                   = exact respa hybrid gpu force-only admission",
        "nstcalcenergy": f"nstcalcenergy           = {disabled_interval}",
        "nstenergy": f"nstenergy               = {disabled_interval}",
        "nstlog": f"nstlog                  = {disabled_interval}",
        "nstxout": f"nstxout                 = {disabled_interval}",
        "nstvout": f"nstvout                 = {disabled_interval}",
    }
    output_lines = []
    for raw_line in mdp.splitlines():
        key = raw_line.split("=", 1)[0].strip() if "=" in raw_line else ""
        output_lines.append(replacements.get(key, raw_line))
    return "\n".join(output_lines) + "\n"


def trace_env(args: argparse.Namespace, run_root: Path, atom_count: int) -> dict[str, str]:
    env = base_env(args)
    all_steps = ",".join(str(step) for step in range(args.outer_steps * LEVEL_FACTORS[-1] + 1))
    env["GMX_EXACT_RESPA_RUNTIME_EVENT_TRACE_FILE"] = str(run_root / "event_trace.tsv")
    env["GMX_EXACT_RESPA_TOTAL_FORCE_DUMP_FILE"] = str(run_root / "total_force.tsv")
    env["GMX_PCFF_RESPA_MERGE_TRACE_DIR"] = str(run_root / "merge_trace")
    env["GMX_PCFF_RESPA_TRACE_ATOMS"] = build_trace_atom_indices(atom_count)
    env["GMX_PCFF_RESPA_TRACE_FORCE_COMPONENTS"] = "1"
    env["GMX_PCFF_RESPA_TRACE_FORCE_COMPONENTS_STEPS"] = all_steps
    env["GMX_PCFF_RESPA_TRACE_REALSPACE_FORCE_SUBCOMPONENTS"] = "1"
    env["GMX_PCFF_RESPA_TRACE_REALSPACE_FORCE_SUBCOMPONENTS_STEPS"] = all_steps
    env["GMX_PCFF_RESPA_M2P_TRACE_DIR"] = str(run_root / "m2p_trace")
    env["GMX_PCFF_RESPA_M2P_CASE_LABEL"] = run_root.name
    return env


def mdrun_args(args: argparse.Namespace, tpr_path: Path, deffnm: Path, *, nb: str) -> list[str]:
    result = [
        "mdrun",
        "-s",
        str(tpr_path),
        "-deffnm",
        str(deffnm),
        "-ntmpi",
        str(args.ntmpi),
        "-ntomp",
        str(args.ntomp),
        "-dlb",
        "no",
        "-nb",
        nb,
        "-pme",
        "cpu",
        "-bonded",
        "cpu",
        "-update",
        "cpu",
        "-pin",
        "off",
    ]
    if nb == "cpu":
        result.append("-reprod")
    return result


def compare_force_rows(
    actual_entries: list[dict[str, object]],
    expected_entries: list[dict[str, object]],
    tolerance: float,
) -> dict[str, object]:
    def key(row: dict[str, object]) -> tuple[int, int, int]:
        return (int(row["step"]), int(row["highest_active_level"]), int(row["atom"]))

    actual_map = {key(row): row for row in actual_entries}
    expected_map = {key(row): row for row in expected_entries}
    missing = sorted([f"{step}:{level}:{atom}" for step, level, atom in expected_map.keys() - actual_map.keys()])
    extra = sorted([f"{step}:{level}:{atom}" for step, level, atom in actual_map.keys() - expected_map.keys()])

    max_abs_component_delta = 0.0
    sum_sq = 0.0
    component_count = 0
    first_over_tolerance = None
    for row_key in sorted(actual_map.keys() & expected_map.keys()):
        actual_force = list(actual_map[row_key]["force"])
        expected_force = list(expected_map[row_key]["force"])
        deltas = [float(a) - float(e) for a, e in zip(actual_force, expected_force)]
        for delta in deltas:
            abs_delta = abs(delta)
            max_abs_component_delta = max(max_abs_component_delta, abs_delta)
            sum_sq += delta * delta
            component_count += 1
        if first_over_tolerance is None and any(abs(delta) > tolerance for delta in deltas):
            first_over_tolerance = {
                "step": row_key[0],
                "highest_active_level": row_key[1],
                "atom": row_key[2],
                "expected_force": expected_force,
                "actual_force": actual_force,
                "component_deltas": deltas,
            }

    rms_component_delta = math.sqrt(sum_sq / component_count) if component_count else 0.0
    return {
        "matches": not missing and not extra and first_over_tolerance is None,
        "tolerance": tolerance,
        "missing_in_actual": missing[:20],
        "extra_in_actual": extra[:20],
        "missing_count": len(missing),
        "extra_count": len(extra),
        "compared_row_count": len(actual_map.keys() & expected_map.keys()),
        "max_abs_component_delta": max_abs_component_delta,
        "rms_component_delta": rms_component_delta,
        "first_over_tolerance": first_over_tolerance,
    }


def compare_per_level_force_entries_with_tolerance(
    actual_entries: list[dict[str, object]],
    expected_entries: list[dict[str, object]],
    tolerance: float,
) -> dict[str, object]:
    def key_map(entries: list[dict[str, object]]) -> dict[str, dict[str, object]]:
        return {str(entry["relative_path"]): entry for entry in entries}

    actual_map = key_map(actual_entries)
    expected_map = key_map(expected_entries)
    missing_in_actual = sorted(expected_map.keys() - actual_map.keys())
    extra_in_actual = sorted(actual_map.keys() - expected_map.keys())
    max_abs_component_delta = 0.0
    first_over_tolerance = None
    for row_key in sorted(actual_map.keys() & expected_map.keys()):
        actual_vector = list(actual_map[row_key]["vector_sum"])
        expected_vector = list(expected_map[row_key]["vector_sum"])
        component_deltas = [float(a) - float(e) for a, e in zip(actual_vector, expected_vector)]
        component_abs_deltas = [abs(delta) for delta in component_deltas]
        if component_abs_deltas:
            max_abs_component_delta = max(max_abs_component_delta, max(component_abs_deltas))
        if first_over_tolerance is None and any(delta > tolerance for delta in component_abs_deltas):
            first_over_tolerance = {
                "relative_path": row_key,
                "expected_vector_sum": expected_vector,
                "actual_vector_sum": actual_vector,
                "component_deltas": component_deltas,
                "component_abs_deltas": component_abs_deltas,
            }

    return {
        "matches": not missing_in_actual and not extra_in_actual and first_over_tolerance is None,
        "tolerance": tolerance,
        "missing_in_actual": missing_in_actual,
        "extra_in_actual": extra_in_actual,
        "missing_count": len(missing_in_actual),
        "extra_count": len(extra_in_actual),
        "compared_row_count": len(actual_map.keys() & expected_map.keys()),
        "max_abs_component_delta": max_abs_component_delta,
        "first_over_tolerance": first_over_tolerance,
    }


def parse_force_only_total_force_dump(path: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    grouped: dict[tuple[int, int], dict[str, object]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t")
        if len(parts) == 7:
            step_str, time_str, highest_level_str, atom_str, fx_str, fy_str, fz_str = parts
            local_atom = int(atom_str)
            atom = local_atom
        elif len(parts) == 8:
            step_str, time_str, highest_level_str, local_atom_str, atom_str, fx_str, fy_str, fz_str = parts
            local_atom = int(local_atom_str)
            atom = int(atom_str)
        else:
            raise ValueError(f"Unexpected force dump line in {path}: {raw_line}")

        step = int(step_str)
        highest_level = int(highest_level_str)
        fx = float(fx_str)
        fy = float(fy_str)
        fz = float(fz_str)
        entries.append(
            {
                "step": step,
                "time_ps": float(time_str),
                "highest_active_level": highest_level,
                "local_atom": local_atom,
                "atom": atom,
                "force": [fx, fy, fz],
            }
        )

        bucket = grouped.setdefault(
            (step, highest_level),
            {
                "step": step,
                "time_ps": float(time_str),
                "highest_active_level": highest_level,
                "atom_count": 0,
                "vector_sum": [0.0, 0.0, 0.0],
            },
        )
        bucket["atom_count"] += 1
        bucket["vector_sum"][0] += fx
        bucket["vector_sum"][1] += fy
        bucket["vector_sum"][2] += fz

    return {
        "schema_version": 1,
        "entries": entries,
        "per_step_totals": list(grouped.values()),
    }


def load_force_only_outputs(run_root: Path) -> dict[str, object]:
    return {
        "event_trace": parse_event_trace(run_root / "event_trace.tsv"),
        "total_force_summary": parse_force_only_total_force_dump(run_root / "total_force.tsv"),
        "per_level_force_totals": parse_merge_trace_dir(run_root / "merge_trace"),
    }


def collect_system_result(args: argparse.Namespace, gmx: Path, out_root: Path, system_id: str) -> dict[str, object]:
    fixture = FIXTURE_ROOT / system_id
    system_root = out_root / system_id
    inputs_dir = system_root / "inputs"
    logs_dir = system_root / "logs"
    cpu_dir = system_root / "cpu_force_only"
    gpu_dir = system_root / "gpu_force_only"
    summaries_dir = system_root / "summaries"
    for directory in (inputs_dir, logs_dir, cpu_dir, gpu_dir, summaries_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mdp_path = inputs_dir / "exact_respa_force_only.mdp"
    write_text(mdp_path, make_force_only_mdp(args.outer_steps, args.pair14_level))

    tpr_path = inputs_dir / "exact_respa_force_only.tpr"
    grompp_command = [
        str(gmx),
        "grompp",
        "-f",
        str(mdp_path),
        "-c",
        str(fixture / "initial_nve.gro"),
        "-p",
        str(fixture / "topol.top"),
        "-o",
        str(tpr_path),
        "-po",
        str(inputs_dir / "exact_respa_force_only_mdout.mdp"),
        "-maxwarn",
        "1",
    ]
    commands: list[dict[str, object]] = []
    base_environment = base_env(args)
    grompp_stdout = logs_dir / "grompp.stdout"
    grompp_stderr = logs_dir / "grompp.stderr"
    grompp_result = run_command_allow_failure(
        grompp_command,
        cwd=REPO_ROOT,
        env=base_environment,
        stdout_path=grompp_stdout,
        stderr_path=grompp_stderr,
    )
    commands.append(
        command_record(
            "grompp",
            grompp_command,
            cwd=REPO_ROOT,
            env_overrides=env_delta(base_environment, os.environ),
            stdout_path=grompp_stdout,
            stderr_path=grompp_stderr,
        )
    )

    if grompp_result.returncode != 0:
        result = {
            "system_id": system_id,
            "status": "BLOCKER",
            "artifact_root": str(system_root),
            "grompp_returncode": grompp_result.returncode,
            "blocking_reason": "grompp failed before CPU/GPU force-only comparison.",
        }
        write_text(summaries_dir / "system_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        write_commands_script(system_root / "run_commands.sh", commands)
        return result

    atom_count = read_gro_atom_count(fixture / "initial_nve.gro")
    run_results = {}
    for label, nb, run_root in (("cpu", "cpu", cpu_dir), ("gpu", "gpu", gpu_dir)):
        environment = trace_env(args, run_root, atom_count)
        deffnm = run_root / "exact_force_only"
        command = [str(gmx), *mdrun_args(args, tpr_path, deffnm, nb=nb)]
        stdout_path = logs_dir / f"mdrun_{label}.stdout"
        stderr_path = logs_dir / f"mdrun_{label}.stderr"
        completed = run_command_allow_failure(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        commands.append(
            command_record(
                f"mdrun_{label}",
                command,
                cwd=REPO_ROOT,
                env_overrides=env_delta(environment, os.environ),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        )
        run_results[label] = {
            "returncode": completed.returncode,
            "run_root": str(run_root),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }

    if run_results["cpu"]["returncode"] != 0 or run_results["gpu"]["returncode"] != 0:
        result = {
            "system_id": system_id,
            "status": "BLOCKER",
            "artifact_root": str(system_root),
            "runs": run_results,
            "blocking_reason": "CPU or GPU force-only mdrun failed before comparison artifacts were produced.",
        }
        write_text(summaries_dir / "system_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        write_commands_script(system_root / "run_commands.sh", commands)
        return result

    cpu_outputs = load_force_only_outputs(cpu_dir)
    gpu_outputs = load_force_only_outputs(gpu_dir)
    event_comparison = compare_event_trace(gpu_outputs["event_trace"], cpu_outputs["event_trace"])
    force_comparison = compare_force_rows(
        gpu_outputs["total_force_summary"]["entries"],
        cpu_outputs["total_force_summary"]["entries"],
        args.force_tol,
    )
    aggregate_force_tolerance = (
        args.aggregate_force_tol if args.aggregate_force_tol is not None else args.force_tol * atom_count
    )
    per_level_force_comparison = compare_per_level_force_entries_with_tolerance(
        gpu_outputs["per_level_force_totals"]["entries"],
        cpu_outputs["per_level_force_totals"]["entries"],
        aggregate_force_tolerance,
    )

    status = "PASS" if event_comparison["matches"] and force_comparison["matches"] else "FAIL"
    if not per_level_force_comparison["matches"]:
        status = "FAIL"

    result = {
        "schema_version": 1,
        "system_id": system_id,
        "status": status,
        "artifact_root": str(system_root),
        "admission_shape": {
            "exact_respa": True,
            "force_only": True,
            "ntmpi": args.ntmpi,
            "ntomp": args.ntomp,
            "nb_gpu": True,
            "pme": "cpu",
            "bonded": "cpu",
            "update": "cpu",
        },
        "runs": run_results,
        "event_comparison": event_comparison,
        "force_comparison": force_comparison,
        "aggregate_force_tolerance": aggregate_force_tolerance,
        "aggregate_force_tolerance_policy": (
            "explicit --aggregate-force-tol"
            if args.aggregate_force_tol is not None
            else "force_tol multiplied by fixture atom count"
        ),
        "per_level_force_comparison": per_level_force_comparison,
    }
    write_text(summaries_dir / "system_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_commands_script(system_root / "run_commands.sh", commands)
    return result


def main() -> None:
    args = parse_args()
    if args.ntmpi != 1:
        raise ValueError("The audited exact r-RESPA GPU force-only admission shape is single-rank only.")
    if args.ntomp != 2:
        raise ValueError("The audited exact r-RESPA GPU force-only admission shape currently requires ntomp=2.")

    gmx = Path(args.gmx).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    maybe_build(args, gmx)

    gmx_version = capture_output([str(gmx), "--version"], cwd=REPO_ROOT)
    systems = [collect_system_result(args, gmx, out_root, system_id) for system_id in SYSTEMS]
    all_pass = all(system["status"] == "PASS" for system in systems)
    any_blocker = any(system["status"] == "BLOCKER" for system in systems)
    status = "PASS" if all_pass else "BLOCKER" if any_blocker else "FAIL"
    manifest = {
        "schema_version": 1,
        "gate": "exact-r-RESPA GPU hybrid force-only smoke",
        "status": status,
        "objective": (
            "Validate the currently admitted exact r-RESPA hybrid OpenMP+GPU nonbonded force-only "
            "runtime shape before any broader GPU hybrid claim."
        ),
        "artifact_root": str(out_root),
        "gmx": str(gmx),
        "gmx_version": gmx_version,
        "precision_mode": parse_precision_mode(gmx_version),
        "gpu_support": parse_gpu_support(gmx_version),
        "ntmpi": args.ntmpi,
        "ntomp": args.ntomp,
        "outer_steps": args.outer_steps,
        "pair14_level": args.pair14_level,
        "force_tolerance": args.force_tol,
        "aggregate_force_tolerance": args.aggregate_force_tol,
        "aggregate_force_tolerance_policy": (
            "explicit --aggregate-force-tol"
            if args.aggregate_force_tol is not None
            else "per-system force_tol multiplied by fixture atom count"
        ),
        "reproducibility_notes": [
            GPU_REPRODUCIBILITY_NOTE,
            "This is a force-only validation shape; energy, virial, density, volume, and transport claims remain out of scope.",
        ],
        "systems": systems,
        "claim_boundary": {
            "allowed_if_pass": (
                "single-host, single-rank force-only exact r-RESPA nonbonded GPU offload preserves "
                "CPU event order and force dumps on the tested small fixtures"
            ),
            "not_allowed": [
                "Gate I GPU density/volume readiness",
                "energy or virial GPU exactness",
                "GPU update readiness",
                "transport or production readiness",
                "broad GPU speedup",
            ],
        },
    }
    write_text(out_root / "force_only_gpu_hybrid_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    lines = [
        "# exact r-RESPA GPU Hybrid Force-Only Smoke",
        "",
        f"- Status: {status}",
        f"- gmx: `{gmx}`",
        f"- GPU support: `{manifest['gpu_support']}`",
        f"- ntmpi / ntomp: `{args.ntmpi}` / `{args.ntomp}`",
        f"- force tolerance: `{args.force_tol}`",
        f"- aggregate force tolerance: `{args.aggregate_force_tol if args.aggregate_force_tol is not None else 'per-system force_tol * atom_count'}`",
        "",
        "## Systems",
        "",
    ]
    for system in systems:
        lines.append(f"- {system['system_id']}: `{system['status']}`")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- This validates only the admitted force-only nonbonded GPU shape if status is PASS.",
            "- It does not validate energy, virial, density, volume, transport, GPU update, or production readiness.",
        ]
    )
    write_text(out_root / "force_only_gpu_hybrid_manifest.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
