from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from common import (
    OBSERVABLE_ORDER,
    CORPUS_ROOT,
    dump_json,
    enabled_observables,
    iter_system_records,
    parse_dump_custom,
    parse_thermo_table,
    system_metadata,
    system_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage or run the LAMMPS golden corpus generator for M1."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("stage", "run"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--out", required=True, help="Output directory for staged or generated artifacts.")
        subparser.add_argument(
            "--system",
            action="append",
            dest="systems",
            help="System id to include. Repeat to select multiple systems. Default: all systems.",
        )
        if name == "run":
            subparser.add_argument(
                "--lammps-cmd",
                default="lmp",
                help="LAMMPS command to execute, for example `lmp` or `lmp_serial`.",
            )

    return parser.parse_args()


def observable_paths(stage_dir: Path, system_meta: dict, observable: str) -> dict:
    paths = {
        "input": stage_dir / f"{observable}.in",
        "raw_log": stage_dir / "raw" / f"{observable}.log",
        "normalized": stage_dir / "normalized" / system_meta["expected_observables"][observable]["normalized_output"],
    }

    if observable in {"forces", "nve_drift", "nvt_snapshot"}:
        paths["raw_dump"] = stage_dir / "raw" / f"{observable}.dump"
    return paths


def render_single_point(system_meta: dict, paths: dict) -> str:
    fields = " ".join(system_meta["expected_observables"]["single_point"]["thermo_fields"])
    return "\n".join(
        [
            f"log {paths['raw_log'].relative_to(paths['input'].parent)}",
            "include system.in",
            "reset_timestep 0",
            "thermo 1",
            f"thermo_style custom {fields}",
            "thermo_modify flush yes",
            "run 0",
        ]
    )


def render_forces(system_meta: dict, paths: dict) -> str:
    fields = " ".join(system_meta["expected_observables"]["single_point"]["thermo_fields"])
    dump_fields = " ".join(system_meta["expected_observables"]["forces"]["dump_fields"])
    return "\n".join(
        [
            f"log {paths['raw_log'].relative_to(paths['input'].parent)}",
            "include system.in",
            "reset_timestep 0",
            "thermo 1",
            f"thermo_style custom {fields}",
            "thermo_modify flush yes",
            f"dump force_dump all custom 1 {paths['raw_dump'].relative_to(paths['input'].parent)} {dump_fields}",
            "dump_modify force_dump sort id",
            "run 0",
            "undump force_dump",
        ]
    )


def render_nve_drift(system_meta: dict, paths: dict) -> str:
    config = system_meta["expected_observables"]["nve_drift"]
    fields = " ".join(config["thermo_fields"])
    dump_fields = " ".join(system_meta["expected_observables"]["forces"]["dump_fields"])
    return "\n".join(
        [
            f"log {paths['raw_log'].relative_to(paths['input'].parent)}",
            "include system.in",
            "reset_timestep 0",
            f"timestep {config['timestep_fs']}",
            f"velocity all create {config['initial_temperature_K']} {config['velocity_seed']} mom yes rot yes dist gaussian",
            "fix integ all nve",
            "thermo 1",
            f"thermo_style custom {fields}",
            "thermo_modify flush yes",
            f"dump drift_dump all custom {config['nsteps']} {paths['raw_dump'].relative_to(paths['input'].parent)} {dump_fields}",
            "dump_modify drift_dump sort id",
            f"run {config['nsteps']}",
            "undump drift_dump",
            "unfix integ",
        ]
    )


def render_nvt_snapshot(system_meta: dict, paths: dict) -> str:
    config = system_meta["expected_observables"]["nvt_snapshot"]
    fields = " ".join(config["thermo_fields"])
    dump_fields = " ".join(system_meta["expected_observables"]["forces"]["dump_fields"])
    return "\n".join(
        [
            f"log {paths['raw_log'].relative_to(paths['input'].parent)}",
            "include system.in",
            "reset_timestep 0",
            f"timestep {config['timestep_fs']}",
            f"fix nvtfix all nvt temp {config['temperature_start_K']} {config['temperature_stop_K']} {config['tdamp_fs']}",
            "thermo 1",
            f"thermo_style custom {fields}",
            "thermo_modify flush yes",
            f"dump snapshot_dump all custom {config['nsteps']} {paths['raw_dump'].relative_to(paths['input'].parent)} {dump_fields}",
            "dump_modify snapshot_dump sort id",
            f"run {config['nsteps']}",
            "undump snapshot_dump",
            "unfix nvtfix",
        ]
    )


def render_fd_script(system_meta: dict, atom_id: int, component: str, delta: float, sign: str, raw_log_name: str) -> str:
    fields = system_meta["expected_observables"]["finite_difference"]["energy_field"]
    dx = delta if component == "x" else 0.0
    dy = delta if component == "y" else 0.0
    dz = delta if component == "z" else 0.0
    if sign == "minus":
        dx = -dx
        dy = -dy
        dz = -dz
    return "\n".join(
        [
            f"log raw/{raw_log_name}",
            "include system.in",
            "reset_timestep 0",
            f"group fd_atom id {atom_id}",
            f"displace_atoms fd_atom move {dx} {dy} {dz} units box",
            "thermo 1",
            f"thermo_style custom step {fields}",
            "thermo_modify flush yes",
            "run 0",
        ]
    )


def write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents.strip() + "\n", encoding="utf-8")


def stage_system(record: dict, out_root: Path) -> dict:
    source_root = system_root(record)
    system_meta = system_metadata(record)
    stage_dir = out_root / record["id"]
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    (stage_dir / "raw").mkdir(parents=True)
    (stage_dir / "normalized").mkdir(parents=True)

    shutil.copy2(source_root / "system.json", stage_dir / "system.json")
    shutil.copy2(source_root / "lammps" / "system.data", stage_dir / "system.data")
    shutil.copy2(source_root / "lammps" / "system.in", stage_dir / "system.in")

    operations = []
    for observable in OBSERVABLE_ORDER:
        config = system_meta["expected_observables"][observable]
        if not config["enabled"]:
            continue

        paths = observable_paths(stage_dir, system_meta, observable)
        if observable == "single_point":
            write_text(paths["input"], render_single_point(system_meta, paths))
        elif observable == "forces":
            write_text(paths["input"], render_forces(system_meta, paths))
        elif observable == "nve_drift":
            write_text(paths["input"], render_nve_drift(system_meta, paths))
        elif observable == "nvt_snapshot":
            write_text(paths["input"], render_nvt_snapshot(system_meta, paths))

        if observable != "finite_difference":
            operations.append(
                {
                    "name": observable,
                    "input": paths["input"].name,
                    "raw_log": str(paths["raw_log"].relative_to(stage_dir)),
                    "normalized_output": str(paths["normalized"].relative_to(stage_dir)),
                    "raw_dump": (
                        str(paths["raw_dump"].relative_to(stage_dir)) if "raw_dump" in paths else None
                    ),
                }
            )

    fd_config = system_meta["expected_observables"]["finite_difference"]
    if fd_config["enabled"]:
        fd_tasks = []
        for atom_id in fd_config["atoms"]:
            for component in fd_config["components"]:
                for sign in ("plus", "minus"):
                    script_name = f"finite_difference_atom{atom_id}_{component}_{sign}.in"
                    raw_log_name = f"finite_difference_atom{atom_id}_{component}_{sign}.log"
                    write_text(
                        stage_dir / script_name,
                        render_fd_script(
                            system_meta,
                            atom_id,
                            component,
                            fd_config["delta"],
                            sign,
                            raw_log_name,
                        ),
                    )
                    fd_tasks.append(
                        {
                            "atom_id": atom_id,
                            "component": component,
                            "sign": sign,
                            "input": script_name,
                            "raw_log": f"raw/{raw_log_name}",
                        }
                    )
        operations.append(
            {
                "name": "finite_difference",
                "input": None,
                "raw_log": None,
                "normalized_output": str((stage_dir / "normalized" / fd_config["normalized_output"]).relative_to(stage_dir)),
                "raw_dump": None,
                "tasks": fd_tasks,
            }
        )

    run_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'LAMMPS_CMD="${LAMMPS_CMD:-lmp}"',
    ]
    for operation in operations:
        if operation["name"] == "finite_difference":
            for task in operation["tasks"]:
                run_lines.append('"$LAMMPS_CMD" -in ' + task["input"])
        else:
            run_lines.append('"$LAMMPS_CMD" -in ' + operation["input"])
    write_text(stage_dir / "run_all.sh", "\n".join(run_lines))

    stage_manifest = {
        "schema_version": 1,
        "corpus_root": str(CORPUS_ROOT.relative_to(CORPUS_ROOT)),
        "system_id": record["id"],
        "enabled_observables": enabled_observables(system_meta),
        "operations": operations,
    }
    dump_json(stage_dir / "stage_manifest.json", stage_manifest)
    return stage_manifest


def normalize_single_point(stage_dir: Path, system_meta: dict) -> None:
    config = system_meta["expected_observables"]["single_point"]
    rows = parse_thermo_table(stage_dir / "raw" / "single_point.log", config["thermo_fields"])
    payload = {
        "schema_version": 1,
        "system_id": system_meta["id"],
        "observable": "single_point",
        "units": {"energy": "kcal/mol"},
        "fields": rows[-1],
    }
    dump_json(stage_dir / "normalized" / config["normalized_output"], payload)


def normalize_forces(stage_dir: Path, system_meta: dict) -> None:
    config = system_meta["expected_observables"]["forces"]
    frames = parse_dump_custom(stage_dir / "raw" / "forces.dump")
    payload = {
        "schema_version": 1,
        "system_id": system_meta["id"],
        "observable": "forces",
        "units": {
            "distance": "angstrom",
            "force": "kcal/mol/angstrom",
            "charge": "e",
        },
        "frame": frames[-1],
    }
    dump_json(stage_dir / "normalized" / config["normalized_output"], payload)


def normalize_trace(stage_dir: Path, system_meta: dict, observable: str) -> None:
    config = system_meta["expected_observables"][observable]
    rows = parse_thermo_table(stage_dir / "raw" / f"{observable}.log", config["thermo_fields"])
    frames = parse_dump_custom(stage_dir / "raw" / f"{observable}.dump")
    payload = {
        "schema_version": 1,
        "system_id": system_meta["id"],
        "observable": observable,
        "units": {
            "time": "fs",
            "energy": "kcal/mol",
            "temperature": "K",
            "pressure": "atm",
            "force": "kcal/mol/angstrom",
        },
        "trace": rows,
        "final_frame": frames[-1],
    }
    dump_json(stage_dir / "normalized" / config["normalized_output"], payload)


def normalize_finite_difference(stage_dir: Path, system_meta: dict) -> None:
    config = system_meta["expected_observables"]["finite_difference"]
    forces_payload = load_normalized(stage_dir / "normalized" / system_meta["expected_observables"]["forces"]["normalized_output"])
    force_frame = forces_payload["frame"]
    force_lookup = {atom["id"]: atom for atom in force_frame["atoms"]}

    checks = []
    energy_field = config["energy_field"]
    for atom_id in config["atoms"]:
        for component in config["components"]:
            plus_log = stage_dir / "raw" / f"finite_difference_atom{atom_id}_{component}_plus.log"
            minus_log = stage_dir / "raw" / f"finite_difference_atom{atom_id}_{component}_minus.log"
            plus_rows = parse_thermo_table(plus_log, ["step", energy_field])
            minus_rows = parse_thermo_table(minus_log, ["step", energy_field])
            plus_energy = plus_rows[-1][energy_field]
            minus_energy = minus_rows[-1][energy_field]
            finite_difference_force = -(plus_energy - minus_energy) / (2.0 * config["delta"])
            analytic_force = force_lookup[atom_id][f"f{component}"]
            checks.append(
                {
                    "atom_id": atom_id,
                    "component": component,
                    "delta": config["delta"],
                    "analytic_force": analytic_force,
                    "finite_difference_force": finite_difference_force,
                    "residual": analytic_force - finite_difference_force,
                }
            )

    payload = {
        "schema_version": 1,
        "system_id": system_meta["id"],
        "observable": "finite_difference",
        "units": {
            "energy": "kcal/mol",
            "distance": "angstrom",
            "force": "kcal/mol/angstrom",
        },
        "checks": checks,
    }
    dump_json(stage_dir / "normalized" / config["normalized_output"], payload)


def load_normalized(path: Path) -> dict:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def lammps_command_and_env(lammps_cmd: str) -> tuple[list[str], dict[str, str]]:
    tokens = shlex.split(lammps_cmd)
    env = os.environ.copy()

    executable_index = 0
    while executable_index < len(tokens):
        token = tokens[executable_index]
        if "=" not in token or token.startswith("-"):
            break
        key, value = token.split("=", 1)
        if not key.isidentifier():
            break
        env[key] = value
        executable_index += 1

    if executable_index >= len(tokens):
        raise ValueError(f"Could not determine LAMMPS executable from command: {lammps_cmd!r}")

    return tokens[executable_index:], env


def run_system(record: dict, out_root: Path, lammps_cmd: str) -> None:
    system_meta = system_metadata(record)
    stage_dir = out_root / record["id"]
    manifest = load_normalized(stage_dir / "stage_manifest.json")
    command, env = lammps_command_and_env(lammps_cmd)

    for operation in manifest["operations"]:
        if operation["name"] == "finite_difference":
            for task in operation["tasks"]:
                subprocess.run(
                    command + ["-in", task["input"]],
                    cwd=stage_dir,
                    env=env,
                    check=True,
                )
        else:
            subprocess.run(
                command + ["-in", operation["input"]],
                cwd=stage_dir,
                env=env,
                check=True,
            )

    normalize_single_point(stage_dir, system_meta)
    normalize_forces(stage_dir, system_meta)
    normalize_finite_difference(stage_dir, system_meta)
    for observable in ("nve_drift", "nvt_snapshot"):
        if system_meta["expected_observables"][observable]["enabled"]:
            normalize_trace(stage_dir, system_meta, observable)


def main() -> int:
    args = parse_args()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    records = iter_system_records(args.systems)
    for record in records:
        stage_system(record, out_root)

    if args.command == "run":
        for record in records:
            run_system(record, out_root, args.lammps_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
