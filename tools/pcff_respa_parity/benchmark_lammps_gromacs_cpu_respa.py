#!/usr/bin/env python3
"""Run a bounded CPU-only LAMMPS-vs-GROMACS exact-r-RESPA speed benchmark.

The benchmark is intentionally narrow:
  * same paired fixture artifacts
  * same target simulated duration
  * matched r-RESPA hierarchy by default:
      LAMMPS outer timestep = GROMACS base timestep * exact-respa-level3-factor
  * speed evidence only, not thermostat/ensemble equivalence evidence
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_ROOT = (
    REPO_ROOT
    / "tests/reference_results/gate_h_fixture_scaffold/gate_h_dense_salt_polymer_2x2x2"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "tests/reference_results/lammps_gromacs_cpu_respa_speed_protocol_20260424"
)
DEFAULT_LMP = Path(os.environ.get("LMP_BIN", "lmp"))


def parse_env_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got {value!r}")
    key, env_value = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"Expected non-empty KEY in {value!r}")
    return key, env_value


@dataclass(frozen=True)
class Fixture:
    root: Path
    gro: Path
    top: Path
    lammps_input_header: Path
    lammps_data: Path
    manifest: Path


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    log_path: Path,
    timeout_s: float | None = None,
) -> tuple[int, float]:
    start = time.monotonic()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=merged_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
    return proc.returncode, time.monotonic() - start


def parse_gmx_performance(log_text: str) -> float | None:
    matches = re.findall(r"Performance:\s*([0-9.+\-eE]+)", log_text)
    return float(matches[-1]) if matches else None


def parse_lammps_loop(log_text: str, timestep_fs: float) -> dict[str, Any]:
    match = re.search(
        r"Loop time of\s+([0-9.+\-eE]+)\s+on\s+(\d+)\s+procs\s+for\s+(\d+)\s+steps\s+with\s+(\d+)\s+atoms",
        log_text,
    )
    if not match:
        return {}
    loop_s = float(match.group(1))
    nprocs = int(match.group(2))
    nsteps = int(match.group(3))
    natoms = int(match.group(4))
    simulated_ns = nsteps * timestep_fs * 1.0e-6
    ns_per_day = simulated_ns / loop_s * 86400.0 if loop_s > 0 else None
    return {
        "loop_s": loop_s,
        "nprocs": nprocs,
        "nsteps": nsteps,
        "natoms": natoms,
        "simulated_ns": simulated_ns,
        "ns_per_day_from_loop": ns_per_day,
    }


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_fixture(root: Path) -> Fixture:
    manifest = root / "fixture_manifest.json"
    manifest_data = json.loads(read_text(manifest))
    artifacts = manifest_data["artifacts"]
    return Fixture(
        root=root,
        gro=Path(artifacts["gro"]),
        top=Path(artifacts["topology"]),
        lammps_input_header=Path(artifacts["system_in"]),
        lammps_data=Path(artifacts["system_data"]),
        manifest=manifest,
    )


def write_gmx_mdp(
    path: Path,
    *,
    duration_ps: float,
    base_dt_ps: float,
    temperature_k: float,
    tau_t_ps: float,
    level3_factor: int,
    pair_splitting: str,
    nstlist: int,
    nstcalcenergy: int,
    nstenergy: int,
    nstlog: int,
) -> int:
    if pair_splitting not in {"split", "none"}:
        raise ValueError(f"Unsupported pair splitting mode: {pair_splitting}")

    pair_splitting_lines = [
        "exact-respa-inner-level = 1",
        "exact-respa-middle-level = 2",
        "exact-respa-outer-level = 3",
        "exact-respa-inner-off   = 0.30",
        "exact-respa-inner-on    = 0.45",
        "exact-respa-outer-on    = 0.60",
        "exact-respa-outer-off   = 0.80",
    ] if pair_splitting == "split" else []

    nsteps = int(round(duration_ps / base_dt_ps))
    path.write_text(
        "\n".join(
            [
                "title                   = lammps gromacs cpu respa speed protocol nvt",
                "integrator              = md-vv",
                f"dt                      = {base_dt_ps:.9f}",
                f"nsteps                  = {nsteps}",
                "constraints             = none",
                "cutoff-scheme           = Verlet",
                f"nstlist                 = {nstlist}",
                "rlist                   = 0.99",
                "rvdw                    = 0.9",
                "rcoulomb                = 0.9",
                "vdwtype                 = Cut-off",
                "vdw-modifier            = none",
                "coulombtype             = PME",
                "coulomb-modifier        = none",
                "ewald-rtol              = 1e-6",
                "pme-order               = 4",
                "fourierspacing          = 0.08",
                "epsilon-r               = 1",
                "pbc                     = xyz",
                "comm-mode               = none",
                "verlet-buffer-tolerance = -1",
                "exact-respa             = yes",
                "exact-respa-levels      = 3",
                "exact-respa-level2-factor = 2",
                f"exact-respa-level3-factor = {level3_factor}",
                "exact-respa-bond-level  = 1",
                "exact-respa-angle-level = 1",
                "exact-respa-dihedral-level = 1",
                "exact-respa-improper-level = 1",
                "exact-respa-pair14-level = 1",
                "exact-respa-pair-level  = 3",
                "exact-respa-kspace-level = 3",
                *pair_splitting_lines,
                f"nstcalcenergy           = {nstcalcenergy}",
                f"nstenergy               = {nstenergy}",
                f"nstlog                  = {nstlog}",
                "nstxout                 = 0",
                "nstvout                 = 0",
                "nstfout                 = 0",
                "nstxout-compressed      = 0",
                "tcoupl                  = v-rescale",
                "tc-grps                 = System",
                f"tau-t                   = {tau_t_ps:.6f}",
                f"ref-t                   = {temperature_k:.6f}",
                f"nsttcouple              = {level3_factor}",
                "pcoupl                  = no",
                "gen-vel                 = yes",
                f"gen-temp                = {temperature_k:.6f}",
                "gen-seed                = 248320",
                "continuation            = no",
                "ld-seed                 = 192645",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return nsteps


def write_lammps_input(
    path: Path,
    *,
    header_name: str,
    duration_ps: float,
    outer_dt_fs: float,
    level3_factor: int,
    temperature_k: float,
    tau_t_ps: float,
    thermo_every: int,
) -> int:
    if level3_factor < 1:
        raise ValueError(f"Expected positive r-RESPA factor, got {level3_factor}")

    nsteps = int(round(duration_ps * 1000.0 / outer_dt_fs))
    tdamp_fs = tau_t_ps * 1000.0
    path.write_text(
        "\n".join(
            [
                f"include {header_name}",
                "",
                "neighbor 3.0 bin",
                "neigh_modify delay 0 every 1 check yes",
                "reset_timestep 0",
                f"timestep {outer_dt_fs:.9f}",
                f"run_style respa 2 {level3_factor}",
                "thermo_style custom step temp press pe ke etotal",
                "thermo_modify flush yes",
                f"thermo {thermo_every}",
                "velocity all create "
                f"{temperature_k:.6f} 248320 dist gaussian mom yes rot yes",
                "fix nvtbench all nvt temp "
                f"{temperature_k:.6f} {temperature_k:.6f} {tdamp_fs:.6f}",
                "timer full",
                f"run {nsteps}",
                "unfix nvtbench",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return nsteps


def copy_fixture_inputs(fixture: Fixture, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fixture.gro, run_dir / "system.gro")
    shutil.copy2(fixture.top, run_dir / "topol.top")
    shutil.copy2(fixture.lammps_input_header, run_dir / "system.in")
    shutil.copy2(fixture.lammps_data, run_dir / "system.data")


def run_gromacs(
    *,
    out_root: Path,
    fixture: Fixture,
    gmx: Path,
    duration_ps: float,
    base_dt_ps: float,
    level3_factor: int,
    temperature_k: float,
    tau_t_ps: float,
    ntomp: int,
    ntmpi: int,
    npme: int | None,
    ntomp_pme: int | None,
    pin: str,
    reprod: bool,
    pair_splitting: str,
    update_omp_mode: str,
    native_multi_owner_mode: str,
    extra_env: dict[str, str],
    nstlist: int,
    nstcalcenergy: int,
    nstenergy: int,
    nstlog: int,
) -> dict[str, Any]:
    case_dir = out_root / f"gromacs_cpu_ntomp{ntomp}"
    copy_fixture_inputs(fixture, case_dir)
    nsteps = write_gmx_mdp(
        case_dir / "bench.mdp",
        duration_ps=duration_ps,
        base_dt_ps=base_dt_ps,
        temperature_k=temperature_k,
        tau_t_ps=tau_t_ps,
        level3_factor=level3_factor,
        pair_splitting=pair_splitting,
        nstlist=nstlist,
        nstcalcenergy=nstcalcenergy,
        nstenergy=nstenergy,
        nstlog=nstlog,
    )
    grompp_rc, grompp_elapsed = run_command(
        [
            str(gmx),
            "grompp",
            "-f",
            "bench.mdp",
            "-c",
            "system.gro",
            "-p",
            "topol.top",
            "-o",
            "bench.tpr",
            "-maxwarn",
            "10",
        ],
        cwd=case_dir,
        log_path=case_dir / "grompp.stdout",
    )
    result: dict[str, Any] = {
        "case": f"gromacs_cpu_ntomp{ntomp}",
        "engine": "gromacs",
        "status": "grompp_failed" if grompp_rc else "not_run",
        "grompp_returncode": grompp_rc,
        "grompp_elapsed_s": grompp_elapsed,
        "ntmpi": ntmpi,
        "ntomp": ntomp,
        "npme": npme,
        "ntomp_pme": ntomp_pme,
        "pin": pin,
        "reprod": reprod,
        "pair_splitting": pair_splitting,
        "exact_respa_update_omp_mode": update_omp_mode,
        "native_multi_owner_mode": native_multi_owner_mode,
        "base_dt_ps": base_dt_ps,
        "outer_dt_fs": base_dt_ps * 1000.0 * level3_factor,
        "level3_factor": level3_factor,
        "duration_ps": duration_ps,
        "nsteps": nsteps,
        "run_dir": str(case_dir),
    }
    if grompp_rc != 0:
        return result
    mdrun_cmd = [
        str(gmx),
        "mdrun",
        "-s",
        "bench.tpr",
        "-deffnm",
        "bench",
        "-ntmpi",
        str(ntmpi),
        "-ntomp",
        str(ntomp),
        "-pin",
        pin,
    ]
    if npme is not None:
        mdrun_cmd.extend(["-npme", str(npme)])
    if ntomp_pme is not None:
        mdrun_cmd.extend(["-ntomp_pme", str(ntomp_pme)])
    if reprod:
        mdrun_cmd.append("-reprod")

    env = {"OMP_NUM_THREADS": str(ntomp)}
    if update_omp_mode == "on":
        env["GMX_PCFF_EXACT_RESPA_UPDATE_OMP"] = "1"
    elif update_omp_mode == "off":
        env["GMX_PCFF_EXACT_RESPA_UPDATE_OMP"] = "0"
    elif update_omp_mode != "auto":
        raise ValueError(f"Unsupported update OMP mode: {update_omp_mode}")

    if native_multi_owner_mode == "owner_fallback":
        env.update(
            {
                "GMX_PCFF_EXACT_RESPA_NATIVE_MULTI": "1",
                "GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_OWNER_STEP_FALLBACK": "1",
                "GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_SPLIT_OWNER_OUTPUTS": "0",
            }
        )
    elif native_multi_owner_mode == "full_owner_native":
        env.update(
            {
                "GMX_PCFF_EXACT_RESPA_NATIVE_MULTI": "1",
                "GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_OWNER_STEP_FALLBACK": "0",
                "GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_SPLIT_OWNER_OUTPUTS": "0",
            }
        )
    elif native_multi_owner_mode == "split_owner_sidecar":
        env.update(
            {
                "GMX_PCFF_EXACT_RESPA_NATIVE_MULTI": "1",
                "GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_OWNER_STEP_FALLBACK": "0",
                "GMX_PCFF_EXACT_RESPA_NATIVE_MULTI_SPLIT_OWNER_OUTPUTS": "1",
            }
        )
    elif native_multi_owner_mode != "default":
        raise ValueError(f"Unsupported native-multi owner mode: {native_multi_owner_mode}")
    env.update(extra_env)

    mdrun_rc, mdrun_elapsed = run_command(
        mdrun_cmd,
        cwd=case_dir,
        env=env,
        log_path=case_dir / "mdrun.stdout",
    )
    log_text = read_text(case_dir / "bench.log") if (case_dir / "bench.log").exists() else ""
    result.update(
        {
            "status": "pass" if mdrun_rc == 0 else "mdrun_failed",
            "mdrun_returncode": mdrun_rc,
            "mdrun_elapsed_s": mdrun_elapsed,
            "ns_per_day": parse_gmx_performance(log_text),
            "exact_respa_log_markers": {
                "has_exact_respa": "exact-respa" in log_text or "exact r-RESPA" in log_text,
                "has_mdrun_performance": "Performance:" in log_text,
            },
        }
    )
    return result


def run_lammps(
    *,
    out_root: Path,
    fixture: Fixture,
    lmp: Path,
    duration_ps: float,
    outer_dt_fs: float,
    level3_factor: int,
    temperature_k: float,
    tau_t_ps: float,
    label: str,
    omp_threads: int,
    thermo_every: int,
) -> dict[str, Any]:
    case_dir = out_root / label
    copy_fixture_inputs(fixture, case_dir)
    nsteps = write_lammps_input(
        case_dir / "bench.in",
        header_name="system.in",
        duration_ps=duration_ps,
        outer_dt_fs=outer_dt_fs,
        level3_factor=level3_factor,
        temperature_k=temperature_k,
        tau_t_ps=tau_t_ps,
        thermo_every=thermo_every,
    )
    cmd = [str(lmp), "-sf", "omp", "-pk", "omp", str(omp_threads), "-in", "bench.in"]
    rc, elapsed = run_command(
        cmd,
        cwd=case_dir,
        env={
            "OMP_NUM_THREADS": str(omp_threads),
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
        },
        log_path=case_dir / "lammps.stdout",
    )
    log_text = read_text(case_dir / "log.lammps") if (case_dir / "log.lammps").exists() else ""
    if not log_text and (case_dir / "lammps.stdout").exists():
        log_text = read_text(case_dir / "lammps.stdout")
    loop = parse_lammps_loop(log_text, outer_dt_fs)
    return {
        "case": label,
        "engine": "lammps",
        "status": "pass" if rc == 0 else "failed",
        "returncode": rc,
        "elapsed_s": elapsed,
        "omp_threads_env": omp_threads,
        "lammps_acceleration": "OPENMP package via -sf omp -pk omp",
        "outer_dt_fs": outer_dt_fs,
        "duration_ps": duration_ps,
        "nsteps_requested": nsteps,
        "run_dir": str(case_dir),
        **loop,
    }


def write_protocol_docs(
    *,
    out_root: Path,
    fixture: Fixture,
    args: argparse.Namespace,
    lammps_outer_dt_fs: float,
    gmx_outer_dt_fs: float,
    results: list[dict[str, Any]],
) -> None:
    protocol = {
        "schema_name": "lammps_gromacs_cpu_respa_speed_protocol",
        "schema_version": 1,
        "created_at_kst": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "claim_boundary": {
            "allowed": [
                "host-local CPU-only speed measurements for the named paired fixture",
                "workflow-speed evidence for the configured 50 ps NVT-style equilibration run",
            ],
            "forbidden": [
                "transport-readiness claim",
                "broad CPU scaling claim",
                "ensemble-equivalence claim between LAMMPS fix nvt and GROMACS v-rescale",
                "comparison using mismatched LAMMPS outer timestep and GROMACS base timestep hierarchy",
                "GPU, KOKKOS, or hybrid inference",
            ],
        },
        "fixture": {
            "id": fixture.root.name,
            "manifest": str(fixture.manifest),
            "gromacs_gro": str(fixture.gro),
            "gromacs_topology": str(fixture.top),
            "lammps_input_header": str(fixture.lammps_input_header),
            "lammps_data": str(fixture.lammps_data),
        },
        "runtime_contract": {
            "duration_ps": args.duration_ps,
            "temperature_k": args.temperature_k,
            "tau_t_ps": args.tau_t_ps,
            "gromacs_base_dt_ps": args.gmx_base_dt_ps,
            "gromacs_level3_factor": args.level3_factor,
            "gromacs_outer_dt_fs": gmx_outer_dt_fs,
            "gromacs_ntmpi": args.gmx_ntmpi,
            "gromacs_ntomp": args.gmx_ntomp,
            "gromacs_npme": args.gmx_npme,
            "gromacs_ntomp_pme": args.gmx_ntomp_pme,
            "gromacs_pin": args.gmx_pin,
            "gromacs_reprod": args.gmx_reprod,
            "gromacs_pair_splitting": args.gmx_pair_splitting,
            "gromacs_update_omp_mode": args.gmx_update_omp_mode,
            "gromacs_native_multi_owner_mode": args.gmx_native_multi_owner_mode,
            "gromacs_nstlist": args.gmx_nstlist,
            "gromacs_nstcalcenergy": args.gmx_nstcalcenergy,
            "gromacs_nstenergy": args.gmx_nstenergy,
            "gromacs_nstlog": args.gmx_nstlog,
            "lammps_outer_dt_fs": lammps_outer_dt_fs,
            "hierarchy_match": abs(gmx_outer_dt_fs - lammps_outer_dt_fs) < 1.0e-9,
            "lammps_run_style": f"respa 2 {args.level3_factor}",
            "lammps_respa_inner_factor": args.level3_factor,
            "gromacs_exact_respa": True,
            "output_cadence_is_speed_oriented": True,
        },
        "lammps_binary": str(args.lmp),
        "gromacs_binary": str(args.gmx),
        "results": results,
    }
    (out_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = [
        [
            "case",
            "status",
            "engine",
            "threads",
            "outer_dt_fs",
            "nsteps",
            "elapsed_s",
            "ns_per_day",
            "run_dir",
        ]
    ]
    for item in results:
        threads = item.get("ntomp") or item.get("omp_threads_env")
        ns_per_day = item.get("ns_per_day", item.get("ns_per_day_from_loop"))
        elapsed = item.get("mdrun_elapsed_s", item.get("elapsed_s"))
        nsteps = item.get("nsteps", item.get("nsteps_requested", item.get("nsteps")))
        rows.append(
            [
                str(item.get("case")),
                str(item.get("status")),
                str(item.get("engine")),
                str(threads),
                str(item.get("outer_dt_fs")),
                str(nsteps),
                "" if elapsed is None else f"{float(elapsed):.6f}",
                "" if ns_per_day is None else f"{float(ns_per_day):.6f}",
                str(item.get("run_dir")),
            ]
        )
    (out_root / "cpu_speed_results.tsv").write_text(
        "\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8"
    )

    readme = f"""# LAMMPS vs GROMACS CPU r-RESPA Speed Protocol

This artifact is speed evidence only for the paired fixture
`{fixture.root.name}`. It does not claim transport readiness or broad CPU
scaling.

## Fixed comparison boundary

- GROMACS uses exact r-RESPA with base `dt = {args.gmx_base_dt_ps}` ps and
  `exact-respa-level3-factor = {args.level3_factor}`.
- GROMACS runtime flags are `-ntmpi {args.gmx_ntmpi}`,
  `-ntomp {args.gmx_ntomp}`, `-pin {args.gmx_pin}`,
  `reprod={args.gmx_reprod}`, `pair_splitting={args.gmx_pair_splitting}`,
  `update_omp={args.gmx_update_omp_mode}`,
  `native_multi_owner_mode={args.gmx_native_multi_owner_mode}`, and
  `nstlist={args.gmx_nstlist}`.
- GROMACS output/accounting cadences are `nstcalcenergy={args.gmx_nstcalcenergy}`,
  `nstenergy={args.gmx_nstenergy}`, and `nstlog={args.gmx_nstlog}`.
- LAMMPS uses `run_style respa 2 {args.level3_factor}` with outer `timestep =
  {lammps_outer_dt_fs}` fs.
- The hierarchy match check is
  `LAMMPS outer timestep == GROMACS base dt * level3 factor`:
  `{lammps_outer_dt_fs} fs == {gmx_outer_dt_fs} fs`.
- The run is `{args.duration_ps}` ps, temperature `{args.temperature_k}` K,
  speed-oriented output cadence, no trajectory output.

## Non-claims

- LAMMPS `fix nvt` and GROMACS `v-rescale` are not the same thermostat.
- This is not density/volume convergence evidence.
- This is not transport-readiness evidence.
- This is not GPU/hybrid evidence.
- LAMMPS 1-thread and 12-thread CPU cases use the OPENMP package path:
  `-sf omp -pk omp N`. A KOKKOS/GPU-enabled binary is not used for the CPU-only
  baseline.

See `protocol.json` and `cpu_speed_results.tsv`.
"""
    (out_root / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gmx", type=Path, default=REPO_ROOT / "build/bin/gmx")
    parser.add_argument("--lmp", type=Path, default=DEFAULT_LMP)
    parser.add_argument("--duration-ps", type=float, default=50.0)
    parser.add_argument("--temperature-k", type=float, default=353.0)
    parser.add_argument("--tau-t-ps", type=float, default=0.2)
    parser.add_argument("--gmx-base-dt-ps", type=float, default=0.0005)
    parser.add_argument("--level3-factor", type=int, default=4)
    parser.add_argument("--gmx-ntomp", type=int, default=12)
    parser.add_argument("--gmx-ntmpi", type=int, default=1)
    parser.add_argument("--gmx-npme", type=int)
    parser.add_argument("--gmx-ntomp-pme", type=int)
    parser.add_argument("--gmx-pin", choices=("off", "on", "auto"), default="off")
    parser.add_argument("--gmx-reprod", dest="gmx_reprod", action="store_true", default=True)
    parser.add_argument("--no-gmx-reprod", dest="gmx_reprod", action="store_false")
    parser.add_argument(
        "--gmx-pair-splitting",
        choices=("split", "none"),
        default="split",
        help=(
            "GROMACS exact-r-RESPA real-space pair routing. 'split' uses inner/middle/outer "
            "short-range contributions; 'none' computes the full pair contribution only at "
            "exact-respa-pair-level, matching the simple LAMMPS 'pair kspace' respa layout."
        ),
    )
    parser.add_argument("--gmx-update-omp-mode", choices=("auto", "on", "off"), default="on")
    parser.add_argument(
        "--gmx-native-multi-owner-mode",
        choices=("default", "owner_fallback", "full_owner_native", "split_owner_sidecar"),
        default="default",
        help=(
            "Owner-step native-multi mode. full_owner_native and split_owner_sidecar are "
            "experimental speed probes unless separate exactness evidence is provided."
        ),
    )
    parser.add_argument("--gmx-env", type=parse_env_assignment, action="append", default=[])
    parser.add_argument("--lammps-omp-list", type=int, nargs="+", default=[1, 12])
    parser.add_argument("--gmx-nstlist", type=int, default=20)
    parser.add_argument("--gmx-nstcalcenergy", type=int, default=1000)
    parser.add_argument("--gmx-nstenergy", type=int, default=1000)
    parser.add_argument("--gmx-nstlog", type=int, default=1000)
    parser.add_argument("--lammps-thermo-every", type=int, default=1000)
    parser.add_argument("--skip-gromacs", action="store_true")
    parser.add_argument("--skip-lammps", action="store_true")
    return parser.parse_args()


def require_positive_multiple(*, name: str, value: int, factor: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    if value % factor != 0:
        raise ValueError(f"{name}={value} must be a multiple of --level3-factor={factor}")


def validate_args(args: argparse.Namespace) -> None:
    if args.level3_factor < 1:
        raise ValueError(f"--level3-factor must be positive, got {args.level3_factor}")
    base_steps = int(round(args.duration_ps / args.gmx_base_dt_ps))
    if abs(base_steps * args.gmx_base_dt_ps - args.duration_ps) > 1.0e-9:
        raise ValueError(
            f"--duration-ps={args.duration_ps} must be an integer multiple of "
            f"--gmx-base-dt-ps={args.gmx_base_dt_ps}"
        )
    if base_steps % args.level3_factor != 0:
        raise ValueError(
            f"duration/base_dt gives {base_steps} base steps, which must be a multiple "
            f"of --level3-factor={args.level3_factor} so the final step lands on an "
            "outer-force boundary"
        )
    require_positive_multiple(name="--gmx-nstlist", value=args.gmx_nstlist, factor=args.level3_factor)
    require_positive_multiple(
        name="--gmx-nstcalcenergy", value=args.gmx_nstcalcenergy, factor=args.level3_factor
    )
    require_positive_multiple(name="--gmx-nstenergy", value=args.gmx_nstenergy, factor=args.level3_factor)
    require_positive_multiple(name="--gmx-nstlog", value=args.gmx_nstlog, factor=args.level3_factor)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    fixture = load_fixture(args.fixture_root.resolve())
    out_root = args.out.resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    lammps_outer_dt_fs = args.gmx_base_dt_ps * 1000.0 * args.level3_factor
    gmx_outer_dt_fs = lammps_outer_dt_fs
    results: list[dict[str, Any]] = []

    if not args.skip_gromacs:
        results.append(
            run_gromacs(
                out_root=out_root,
                fixture=fixture,
                gmx=args.gmx.resolve(),
                duration_ps=args.duration_ps,
                base_dt_ps=args.gmx_base_dt_ps,
                level3_factor=args.level3_factor,
                temperature_k=args.temperature_k,
                tau_t_ps=args.tau_t_ps,
                ntomp=args.gmx_ntomp,
                ntmpi=args.gmx_ntmpi,
                npme=args.gmx_npme,
                ntomp_pme=args.gmx_ntomp_pme,
                pin=args.gmx_pin,
                reprod=args.gmx_reprod,
                pair_splitting=args.gmx_pair_splitting,
                update_omp_mode=args.gmx_update_omp_mode,
                native_multi_owner_mode=args.gmx_native_multi_owner_mode,
                extra_env=dict(args.gmx_env),
                nstlist=args.gmx_nstlist,
                nstcalcenergy=args.gmx_nstcalcenergy,
                nstenergy=args.gmx_nstenergy,
                nstlog=args.gmx_nstlog,
            )
        )

    if not args.skip_lammps:
        for omp_threads in args.lammps_omp_list:
            results.append(
                run_lammps(
                    out_root=out_root,
                    fixture=fixture,
                    lmp=args.lmp.resolve(),
                    duration_ps=args.duration_ps,
                    outer_dt_fs=lammps_outer_dt_fs,
                    level3_factor=args.level3_factor,
                    temperature_k=args.temperature_k,
                    tau_t_ps=args.tau_t_ps,
                    label=f"lammps_omp{omp_threads}",
                    omp_threads=omp_threads,
                    thermo_every=args.lammps_thermo_every,
                )
            )

    write_protocol_docs(
        out_root=out_root,
        fixture=fixture,
        args=args,
        lammps_outer_dt_fs=lammps_outer_dt_fs,
        gmx_outer_dt_fs=gmx_outer_dt_fs,
        results=results,
    )
    print(out_root / "cpu_speed_results.tsv")
    failed = [item for item in results if item.get("status") != "pass"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
