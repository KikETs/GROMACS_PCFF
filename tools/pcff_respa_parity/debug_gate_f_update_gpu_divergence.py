from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from freeze_gate_a_oracle import (
    DEFAULT_GMX,
    FIXTURE_ROOT,
    REPO_ROOT,
    base_env,
    capture_output,
    command_record,
    env_delta,
    parse_energy_dump,
    parse_merge_trace_dir,
    parse_total_force_dump,
    write_commands_script,
    write_text,
)
from validate_gate_b_nb_gpu import (
    compare_per_level_force_entries,
    compare_total_force_entries,
    run_command_allow_failure,
    trace_env_for_run,
)
from validate_gate_d_nb_bonded_pme_gpu import compare_force_component_rows, parse_force_component_trace
from validate_gate_f_short_mechanics import EXACT_RESPA_FACTOR, make_gate_f_mdp


DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_f_update_gpu_divergence_debug"
TRACE_STAGES = (
    "pre_initial_kick_velocity",
    "post_initial_kick_velocity",
    "post_drift_position",
    "post_final_kick_velocity",
)
SELECTED_TERMS = (
    "Class2 Bond",
    "Class2 Angle",
    "Class2 Dih.",
    "Coul. recip.",
    "Coulomb (SR)",
    "Potential",
    "Total Energy",
    "Pressure",
    "Vir-ZZ",
)
FORCE_COMPONENTS = (
    "bonded_force",
    "realspace_nonbonded_combined_force",
    "coulomb_recip_force",
    "total_force",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short Gate D/Gate E exact-r-RESPA divergence probe with state trace artifacts."
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--system-id", default="small_salt_polymer_box", help="Fixture system id.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks.")
    parser.add_argument("--ntomp", type=int, default=1, help="OpenMP threads.")
    parser.add_argument(
        "--outer-steps",
        type=int,
        default=5,
        help="Number of outer steps for the debug probe. outer_steps=5 yields output at 0,4,8,12,16.",
    )
    parser.add_argument("--trace-atoms", type=int, default=8, help="Number of atoms to record in the state trace.")
    parser.add_argument(
        "--trace-max-base-step",
        type=int,
        default=16,
        help="Maximum exact-r-RESPA base step to record in the state trace.",
    )
    parser.add_argument(
        "--exact-gpu-bonded-sequential-ftypes",
        action="store_true",
        help="Enable the exact-r-RESPA validation mode that replaces combined GPU bonded launches with sequential per-ftype launches.",
    )
    return parser.parse_args()


def fixture_dir(system_id: str) -> Path:
    return FIXTURE_ROOT / system_id


def run_grompp(
    *,
    gmx: Path,
    mdp_path: Path,
    conf_path: Path,
    top_path: Path,
    tpr_path: Path,
    mdout_path: Path,
    env: dict[str, str],
    logs_dir: Path,
    commands: list[dict[str, object]],
    label: str,
) -> None:
    argv = [
        str(gmx),
        "grompp",
        "-f",
        str(mdp_path),
        "-c",
        str(conf_path),
        "-p",
        str(top_path),
        "-o",
        str(tpr_path),
        "-po",
        str(mdout_path),
        "-maxwarn",
        "1",
    ]
    stdout_path = logs_dir / f"{label}.stdout"
    stderr_path = logs_dir / f"{label}.stderr"
    result = run_command_allow_failure(argv, cwd=REPO_ROOT, env=env, stdout_path=stdout_path, stderr_path=stderr_path)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed; see {stderr_path}")
    commands.append(
        command_record(
            label,
            argv,
            cwd=REPO_ROOT,
            env_overrides=env_delta(env, os.environ),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    )


def mdrun_args(args: argparse.Namespace, deffnm: Path, *, update: str) -> list[str]:
    return [
        str(Path(args.gmx)),
        "mdrun",
        "-s",
        str(deffnm.with_suffix(".tpr")),
        "-deffnm",
        str(deffnm),
        "-ntmpi",
        str(args.ntmpi),
        "-ntomp",
        str(args.ntomp),
        "-dlb",
        "no",
        "-nb",
        "gpu",
        "-pme",
        "gpu",
        "-bonded",
        "gpu",
        "-update",
        update,
        "-pin",
        "off",
    ]


def run_md(
    *,
    gmx: Path,
    argv: list[str],
    env: dict[str, str],
    logs_dir: Path,
    commands: list[dict[str, object]],
    label: str,
) -> dict[str, object]:
    stdout_path = logs_dir / f"{label}.stdout"
    stderr_path = logs_dir / f"{label}.stderr"
    result = run_command_allow_failure(argv, cwd=REPO_ROOT, env=env, stdout_path=stdout_path, stderr_path=stderr_path)
    commands.append(
        command_record(
            label,
            argv,
            cwd=REPO_ROOT,
            env_overrides=env_delta(env, os.environ),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    )
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed; see {stderr_path}")

    deffnm = Path(argv[argv.index("-deffnm") + 1])
    return {
        "run_id": label,
        "deffnm": str(deffnm),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "state_trace": str(deffnm.parent / "state_trace.tsv"),
        "force_store_trace": str(deffnm.parent / "force_store_trace.tsv"),
        "total_force_tsv": str(deffnm.parent / "total_force.tsv"),
        "merge_trace_dir": str(deffnm.parent / "merge_trace"),
        "force_component_trace_txt": str(deffnm.parent / "m2p_trace" / "step0_force_component_trace.txt"),
        "energy_frames": parse_energy_dump(capture_output([str(gmx), "dump", "-e", str(deffnm.with_suffix(".edr"))], cwd=REPO_ROOT)),
        "total_force_summary": parse_total_force_dump(deffnm.parent / "total_force.tsv"),
        "per_level_force_totals": parse_merge_trace_dir(deffnm.parent / "merge_trace"),
        "force_component_trace": parse_force_component_trace(deffnm.parent / "m2p_trace" / "step0_force_component_trace.txt"),
    }


def parse_state_trace(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(f"Unexpected state trace row: {raw_line}")
        rows.append(
            {
                "base_step": int(fields[0]),
                "stage": fields[1],
                "atom": int(fields[2]),
                "components": tuple(float(value) for value in fields[3:6]),
            }
        )
    return rows


def compare_state_traces(expected_rows: list[dict[str, object]], actual_rows: list[dict[str, object]]) -> dict[str, object]:
    expected = {
        (row["base_step"], row["stage"], row["atom"]): row["components"] for row in expected_rows
    }
    actual = {(row["base_step"], row["stage"], row["atom"]): row["components"] for row in actual_rows}
    expected_keys = sorted(expected)
    actual_keys = sorted(actual)
    if expected_keys != actual_keys:
        return {
            "coverage_matches": False,
            "missing_keys": [list(key) for key in expected_keys if key not in actual][:10],
            "extra_keys": [list(key) for key in actual_keys if key not in expected][:10],
        }

    stage_order = {stage: index for index, stage in enumerate(TRACE_STAGES)}
    max_abs_delta_by_stage = {stage: 0.0 for stage in TRACE_STAGES}
    first_nonzero = None

    for key in sorted(expected_keys, key=lambda item: (item[0], stage_order.get(item[1], 999), item[2])):
        expected_components = expected[key]
        actual_components = actual[key]
        abs_deltas = [abs(actual_components[i] - expected_components[i]) for i in range(3)]
        max_abs_delta = max(abs_deltas)
        stage = key[1]
        max_abs_delta_by_stage[stage] = max(max_abs_delta_by_stage.get(stage, 0.0), max_abs_delta)
        if first_nonzero is None and max_abs_delta > 0.0:
            first_nonzero = {
                "base_step": key[0],
                "stage": stage,
                "atom": key[2],
                "expected": expected_components,
                "actual": actual_components,
                "abs_deltas": abs_deltas,
                "max_abs_delta": max_abs_delta,
            }

    return {
        "coverage_matches": True,
        "first_nonzero": first_nonzero,
        "max_abs_delta_by_stage": max_abs_delta_by_stage,
    }


def build_energy_step_table(
    expected_frames: list[dict[str, object]],
    actual_frames: list[dict[str, object]],
    *,
    max_step: int,
) -> list[dict[str, object]]:
    expected_by_step = {int(frame["step"]): frame["terms"] for frame in expected_frames}
    actual_by_step = {int(frame["step"]): frame["terms"] for frame in actual_frames}
    rows: list[dict[str, object]] = []
    for step in range(0, max_step + 1, EXACT_RESPA_FACTOR):
        expected_terms = expected_by_step.get(step)
        actual_terms = actual_by_step.get(step)
        if expected_terms is None or actual_terms is None:
            rows.append({"step": step, "missing": True})
            continue
        term_deltas = {}
        for term in SELECTED_TERMS:
            if term in expected_terms and term in actual_terms:
                term_deltas[term] = float(actual_terms[term]) - float(expected_terms[term])
        rows.append({"step": step, "term_deltas": term_deltas})
    return rows


def first_energy_mismatch(rows: list[dict[str, object]]) -> dict[str, object] | None:
    for row in rows:
        if row.get("missing"):
            return {"step": row["step"], "reason": "missing_frame"}
        for term in SELECTED_TERMS:
            delta = row["term_deltas"].get(term)
            if delta is not None and delta != 0.0:
                return {"step": row["step"], "term": term, "delta": delta}
    return None


def write_summary_md(path: Path, summary: dict[str, object]) -> None:
    lines = ["# Gate F Update Divergence Debug", ""]
    lines.append(f"- System: `{summary['system_id']}`")
    lines.append(f"- Artifact root: `{summary['artifact_root']}`")
    lines.append(f"- First state mismatch: `{summary['state_trace_comparison']['first_nonzero']}`")
    lines.append(f"- First energy mismatch: `{summary['first_energy_mismatch']}`")
    lines.append(f"- First total-force mismatch: `{summary['first_total_force_mismatch']}`")
    lines.append(f"- First per-level-force mismatch: `{summary['first_per_level_force_mismatch']}`")
    lines.append("")
    lines.append("## Max State Delta By Stage")
    lines.append("")
    for stage, value in summary["state_trace_comparison"]["max_abs_delta_by_stage"].items():
        lines.append(f"- `{stage}`: `{value}`")
    lines.append("")
    lines.append("## Force Trace Deltas")
    lines.append("")
    lines.append(
        f"- total-force max abs component delta: `{summary['total_force_comparison']['max_abs_component_delta']}`"
    )
    lines.append(
        f"- per-level-force max abs component delta: `{summary['per_level_force_comparison']['max_abs_component_delta']}`"
    )
    for component_name, comparison in summary["force_component_comparisons"].items():
        lines.append(
            f"- {component_name}: max abs component delta `{comparison['max_abs_component_delta']}`, first nonzero `{comparison['first_nonzero_delta']}`"
        )
    lines.append("")
    lines.append("## Energy Deltas")
    lines.append("")
    for row in summary["energy_step_deltas"]:
        if row.get("missing"):
            lines.append(f"- step {row['step']}: missing frame")
            continue
        lines.append(f"- step {row['step']}: `{row['term_deltas']}`")
    write_text(path, "\n".join(lines))


def main() -> None:
    args = parse_args()
    gmx = Path(args.gmx).resolve()
    out_root = Path(args.out).resolve()
    system_root = out_root / args.system_id
    inputs_dir = system_root / "inputs"
    logs_dir = system_root / "logs"
    summaries_dir = system_root / "summaries"
    gate_d_dir = system_root / "gate_d_probe"
    gate_e_dir = system_root / "gate_e_probe"
    for directory in (inputs_dir, logs_dir, summaries_dir, gate_d_dir, gate_e_dir):
        directory.mkdir(parents=True, exist_ok=True)

    fixture = fixture_dir(args.system_id)
    mdp_path = inputs_dir / "gate_f_debug_exact_respa.mdp"
    write_text(mdp_path, make_gate_f_mdp(args.outer_steps))

    baseline_env = base_env(args)
    commands: list[dict[str, object]] = []
    traced_atom_list = ",".join(str(atom) for atom in range(args.trace_atoms))
    all_steps = ",".join(str(step) for step in range(args.outer_steps * EXACT_RESPA_FACTOR + 1))

    for run_id, run_dir, update_mode in (
        ("gate_d_probe", gate_d_dir, "cpu"),
        ("gate_e_probe", gate_e_dir, "gpu"),
    ):
        run_env = trace_env_for_run(args, run_dir)
        run_env.update(baseline_env)
        run_env["GMX_PCFF_RESPA_TRACE_ATOMS"] = traced_atom_list
        run_env["GMX_PCFF_RESPA_TRACE_COORD_HANDOFF"] = "1"
        run_env["GMX_PCFF_RESPA_TRACE_REALSPACE_FORCE_SUBCOMPONENTS_STEPS"] = all_steps
        run_env["GMX_PCFF_RESPA_TRACE_STEP1_SUBSET01_FORCEGROUP_AUDIT_STEPS"] = all_steps
        run_env["GMX_EXACT_RESPA_STATE_TRACE_FILE"] = str(run_dir / "state_trace.tsv")
        run_env["GMX_EXACT_RESPA_FORCESTORE_TRACE_FILE"] = str(run_dir / "force_store_trace.tsv")
        run_env["GMX_EXACT_RESPA_STATE_TRACE_ATOMS"] = str(args.trace_atoms)
        run_env["GMX_EXACT_RESPA_STATE_TRACE_MAX_BASE_STEP"] = str(args.trace_max_base_step)
        if args.exact_gpu_bonded_sequential_ftypes:
            run_env["GMX_PCFF_RESPA_EXACT_GPU_BONDED_SEQUENTIAL_FTYPES"] = "1"
        run_grompp(
            gmx=gmx,
            mdp_path=mdp_path,
            conf_path=fixture / "initial_nve.gro",
            top_path=fixture / "topol.top",
            tpr_path=run_dir / run_id,
            mdout_path=inputs_dir / f"{run_id}_mdout.mdp",
            env=run_env,
            logs_dir=logs_dir,
            commands=commands,
            label=f"grompp_{run_id}",
        )
        run_info = run_md(
            gmx=gmx,
            argv=mdrun_args(args, run_dir / run_id, update=update_mode),
            env=run_env,
            logs_dir=logs_dir,
            commands=commands,
            label=f"mdrun_{run_id}",
        )
        write_text(summaries_dir / f"{run_id}.json", json.dumps(run_info, indent=2))

    gate_d_info = json.loads((summaries_dir / "gate_d_probe.json").read_text(encoding="utf-8"))
    gate_e_info = json.loads((summaries_dir / "gate_e_probe.json").read_text(encoding="utf-8"))

    gate_d_trace = parse_state_trace(Path(gate_d_info["state_trace"]))
    gate_e_trace = parse_state_trace(Path(gate_e_info["state_trace"]))
    state_comparison = compare_state_traces(gate_d_trace, gate_e_trace)
    energy_step_deltas = build_energy_step_table(
        gate_d_info["energy_frames"],
        gate_e_info["energy_frames"],
        max_step=args.trace_max_base_step,
    )
    total_force_comparison = compare_total_force_entries(
        gate_e_info["total_force_summary"]["per_step_totals"], gate_d_info["total_force_summary"]["per_step_totals"]
    )
    per_level_force_comparison = compare_per_level_force_entries(
        gate_e_info["per_level_force_totals"]["entries"], gate_d_info["per_level_force_totals"]["entries"]
    )
    force_component_comparisons = {
        component_name: compare_force_component_rows(
            gate_e_info["force_component_trace"], gate_d_info["force_component_trace"], component_name
        )
        for component_name in FORCE_COMPONENTS
    }

    summary = {
        "system_id": args.system_id,
        "artifact_root": str(system_root),
        "gmx": str(gmx),
        "outer_steps": args.outer_steps,
        "trace_atoms": args.trace_atoms,
        "trace_max_base_step": args.trace_max_base_step,
        "exact_gpu_bonded_validation_mode": (
            "sequential_ftypes" if args.exact_gpu_bonded_sequential_ftypes else "combined_kernel"
        ),
        "gate_d_probe": gate_d_info,
        "gate_e_probe": gate_e_info,
        "state_trace_comparison": state_comparison,
        "total_force_comparison": total_force_comparison,
        "per_level_force_comparison": per_level_force_comparison,
        "force_component_comparisons": force_component_comparisons,
        "energy_step_deltas": energy_step_deltas,
        "first_energy_mismatch": first_energy_mismatch(energy_step_deltas),
        "first_total_force_mismatch": total_force_comparison["first_mismatch"],
        "first_per_level_force_mismatch": per_level_force_comparison["first_mismatch"],
    }

    write_text(summaries_dir / "debug_summary.json", json.dumps(summary, indent=2))
    write_summary_md(summaries_dir / "debug_summary.md", summary)
    write_commands_script(system_root / "run_commands.sh", commands)


if __name__ == "__main__":
    main()
