#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from polygen_stage_metric_audit import (
    ATM_TO_BAR,
    DEFAULT_NOTEBOOK,
    KJ_MOL_NM3_TO_BAR,
    expected_signature_fragment,
    gmx_stage_key_to_stem,
    lammps_input_to_gmx_stage_key,
    load_notebook_config,
    parse_gmx_energy_terms,
)
from run_polygen_same_state_probe import (
    GMX,
    KCAL_TO_KJ,
    LMP,
    OUT_ROOT,
    bridge_lammps_data,
    gro_volume_nm3,
    parse_energy_xvg,
    replace_mdp_value,
    write_gro_with_velocities,
)


REPO = Path(__file__).resolve().parents[2]
LAMMPS_WORK = OUT_ROOT / "lammps_openmp"
GMX_CPU_WORK = OUT_ROOT / "gromacs_cpu_openmp"
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
TENSOR_TERMS = (
    "c_p_full",
    "c_p_full[1]",
    "c_p_full[2]",
    "c_p_full[3]",
    "c_p_full[4]",
    "c_p_full[5]",
    "c_p_full[6]",
    "c_p_vir",
    "c_p_vir[1]",
    "c_p_vir[2]",
    "c_p_vir[3]",
    "c_p_vir[4]",
    "c_p_vir[5]",
    "c_p_vir[6]",
)
FIX_NH_VECTOR_TERMS = tuple(f"f_1[{i}]" for i in range(1, 29))


def fix_nh_thermo_terms_for_ensemble(ensemble: str) -> tuple[str, ...]:
    if ensemble == "npt":
        return FIX_NH_VECTOR_TERMS
    if ensemble == "nvt":
        return tuple(f"f_1[{i}]" for i in range(1, 13))
    return ()


def fix_nh_restore_terms_for_ensemble(ensemble: str) -> tuple[str, ...]:
    if ensemble == "npt":
        return tuple(f"f_1[{i}]" for i in range(1, 15))
    if ensemble == "nvt":
        return tuple(f"f_1[{i}]" for i in range(1, 7))
    return ()


def run(
    cmd: list[str | Path],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        env=env,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def run_live_to_file(
    cmd: list[str | Path],
    cwd: Path,
    stdout_path: Path,
    *,
    env: dict[str, str] | None = None,
    label: str,
) -> int:
    heartbeat_s = max(0.0, float(os.environ.get("PCFF_STAGE_PROBE_HEARTBEAT_S", "15")))
    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        next_heartbeat = 0.0
        while proc.poll() is None:
            elapsed = time.monotonic() - started
            if heartbeat_s > 0 and elapsed >= next_heartbeat:
                print(f"{label}: running pid={proc.pid} elapsed_s={elapsed:.1f}", flush=True)
                next_heartbeat = elapsed + heartbeat_s
            time.sleep(0.25)
        return proc.returncode


def gmx_mode_args_env(mode: str) -> tuple[list[str], dict[str, str]]:
    if mode == "cpu":
        return ["-nb", "cpu", "-pme", "cpu", "-bonded", "cpu", "-update", "cpu"], {}
    if mode in {"nb_gpu_pme_cpu", "nb_gpu_pme_cpu_bonded_cpu"}:
        return ["-nb", "gpu", "-pme", "cpu", "-bonded", "cpu", "-update", "cpu"], {}
    if mode == "nb_gpu_pme_gpu_bonded_cpu":
        return ["-nb", "gpu", "-pme", "gpu", "-bonded", "cpu", "-update", "cpu"], {}
    if mode == "nb_gpu_pme_cpu_bonded_pair14":
        return ["-nb", "gpu", "-pme", "cpu", "-bonded", "gpu", "-update", "cpu"], {
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_FTYPES": "pair14",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_CPU_LISTED_OVERLAP": "1",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_LIST_CACHE": "1",
        }
    if mode == "nb_gpu_pme_cpu_bonded_class2_pair14":
        return ["-nb", "gpu", "-pme", "cpu", "-bonded", "gpu", "-update", "cpu"], {
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_FTYPES": "class2-pair14",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_CPU_LISTED_OVERLAP": "1",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_LIST_CACHE": "1",
        }
    if mode == "nb_gpu_pme_cpu_bonded_all":
        return ["-nb", "gpu", "-pme", "cpu", "-bonded", "gpu", "-update", "cpu"], {
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_FTYPES": "all",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_CPU_LISTED_OVERLAP": "1",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_LIST_CACHE": "1",
        }
    if mode == "nb_gpu_pme_gpu_bonded_all":
        return ["-nb", "gpu", "-pme", "gpu", "-bonded", "gpu", "-update", "cpu"], {
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_FTYPES": "all",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_CPU_LISTED_OVERLAP": "1",
            "GMX_PCFF_EXACT_RESPA_GPU_BONDED_LIST_CACHE": "1",
        }
    raise ValueError(f"Unsupported GROMACS probe mode: {mode}")


def gmx_mode_uses_pme_gpu(mode: str) -> bool:
    return mode in {"nb_gpu_pme_gpu_bonded_cpu", "nb_gpu_pme_gpu_bonded_all"}


def gmx_effective_mode_for_stage(mode: str, original_text: str) -> str:
    if gmx_mode_uses_pme_gpu(mode) and re.search(r"^\s*kspace_modify\s+compute\s+no\b", original_text, flags=re.M):
        return "cpu"
    return mode


def gmx_binary_looks_double_precision(path: Path) -> bool:
    return path.name == "gmx_d" or path.name.endswith("_d")


def apply_mixed_precision_class2_guard(env: dict[str, str], config: dict[str, Any]) -> None:
    floor = config.get("GMX_PCFF_MIXED_CLASS2_LINEAR_ANGLE_SIN_FLOOR")
    if floor not in (None, "") and not gmx_binary_looks_double_precision(GMX):
        env["GMX_PCFF_CLASS2_LINEAR_ANGLE_SIN_FLOOR"] = str(floor)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def first_match(pattern: str, text: str) -> re.Match[str] | None:
    return re.search(pattern, text, flags=re.M)


def parse_lammps_equal_variables(text: str) -> dict[str, str]:
    return {
        match.group(1): match.group(2).strip()
        for match in re.finditer(r"^\s*variable\s+(\w+)\s+equal\s+(.+?)\s*$", text, flags=re.M)
    }


def evaluate_lammps_equal_expression(text: str, name: str) -> float:
    variables = parse_lammps_equal_variables(text)
    timestep = parse_lammps_timestep_fs(text)
    evaluating: set[str] = set()

    def eval_name(var_name: str) -> float:
        if var_name in evaluating:
            raise ValueError(f"Recursive LAMMPS equal variable: {var_name}")
        if var_name not in variables:
            raise ValueError(f"Unknown LAMMPS equal variable: {var_name}")
        evaluating.add(var_name)
        try:
            return eval_expr(variables[var_name])
        finally:
            evaluating.remove(var_name)

    def eval_expr(expr: str) -> float:
        expr = re.sub(r"\bv_(\w+)\b", lambda match: str(eval_name(match.group(1))), expr)
        expr = re.sub(r"\bdt\b", str(timestep), expr)
        if not re.fullmatch(r"[0-9eE+\-*/().,\s_a-zA-Z]+", expr):
            raise ValueError(f"Unsupported LAMMPS equal expression: {expr!r}")
        return float(
            eval(
                expr,
                {"__builtins__": {}},
                {"floor": math.floor, "ceil": math.ceil, "round": round, "sqrt": math.sqrt},
            )
        )

    return eval_name(name)


def parse_lammps_run_steps(text: str) -> int:
    match = first_match(r"^\s*run\s+(\S+)", text)
    if not match:
        raise ValueError("LAMMPS input has no run command")
    token = match.group(1)
    if token.isdigit():
        return int(token)
    variable_match = re.fullmatch(r"\$\{(\w+)\}|\$(\w+)", token)
    if variable_match:
        variable_name = variable_match.group(1) or variable_match.group(2)
        return int(evaluate_lammps_equal_expression(text, variable_name))
    raise ValueError(f"Unsupported LAMMPS run step token: {token!r}")


def parse_lammps_minimize_maxiter(text: str) -> int:
    match = first_match(r"^\s*minimize\s+\S+\s+\S+\s+(\d+)\s+(\d+)\b", text)
    if not match:
        raise ValueError("LAMMPS input has no minimize command")
    return int(match.group(1))


def parse_lammps_timestep_fs(text: str) -> float:
    match = first_match(r"^\s*timestep\s+([0-9.eE+-]+)\b", text)
    if not match:
        raise ValueError("LAMMPS input has no timestep command")
    return float(match.group(1))


def parse_lammps_read_restart(text: str) -> Path:
    match = first_match(r"^\s*read_restart\s+(.+?)\s*$", text)
    if not match:
        raise ValueError("LAMMPS input has no read_restart command")
    raw = match.group(1).strip()
    path = Path(raw)
    return path if path.is_absolute() else (LAMMPS_WORK / path).resolve()


def parse_lammps_fix_line(text: str) -> str | None:
    match = first_match(r"^\s*fix\s+1\s+all\s+(.+?)\s*$", text)
    return match.group(1).strip() if match else None


def parse_npt_pressure_ramp_atm(fix_line: str | None) -> tuple[float, float] | None:
    if not fix_line or not fix_line.startswith("npt "):
        return None
    match = re.search(r"\biso\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+\S+", fix_line)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def stage_ensemble_from_fix(fix_line: str | None) -> str:
    if fix_line is None:
        return "unknown"
    if fix_line.startswith("npt "):
        return "npt"
    if fix_line.startswith("nvt "):
        return "nvt"
    if fix_line.startswith("nve"):
        return "nve"
    return "unknown"


def lammps_input_for_stage(stem_or_input: str) -> Path:
    candidate = Path(stem_or_input)
    if candidate.exists():
        return candidate.resolve()
    if stem_or_input.endswith(".in"):
        path = LAMMPS_WORK / "resume_inputs" / stem_or_input
        if path.exists():
            return path.resolve()
    stage_key = stem_or_input
    if stage_key.startswith("0") and "_" in stage_key:
        stage_key = re.sub(r"^\d+_", "", stage_key)
    for path in sorted((LAMMPS_WORK / "resume_inputs").glob("lammps_equil_*.in")):
        mapped = lammps_input_to_gmx_stage_key(f"resume_inputs/{path.name}")
        if mapped == stage_key or gmx_stage_key_to_stem(mapped or "") == stem_or_input:
            return path.resolve()
    raise FileNotFoundError(f"No LAMMPS resume input found for {stem_or_input!r}")


def lammps_inputs_for_default_stages() -> list[Path]:
    out: list[Path] = []
    for path in sorted((LAMMPS_WORK / "resume_inputs").glob("lammps_equil_*.in")):
        stage_key = lammps_input_to_gmx_stage_key(f"resume_inputs/{path.name}")
        if stage_key and not ("minimize" in stage_key or stage_key == "gromacs_initial_em"):
            out.append(path.resolve())
    return out


def stage_chunk_index(stage_key: str) -> int | None:
    match = re.search(r"_chunk(\d{4})$", stage_key)
    return int(match.group(1)) if match else None


def root_cause_gate_status(stage_key: str, ensemble: str, fix_vector_restored: bool = False) -> tuple[str, str]:
    chunk_index = stage_chunk_index(stage_key)
    if chunk_index is not None and chunk_index > 1 and ensemble in {"nvt", "npt"}:
        if fix_vector_restored:
            return "valid", "same-start x/v/box plus LAMMPS FixNH extended state are comparable"
        return (
            "diagnostic_only",
            "LAMMPS read_restart restores FixNH extended state for split chunks, but the same-start data/gro bridge cannot carry that state into GROMACS.",
        )
    return "valid", "same-start x/v/box and fresh coupling state are comparable"


def absolute_restart_text(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        path = Path(match.group(1).strip())
        resolved = path if path.is_absolute() else (LAMMPS_WORK / path).resolve()
        return f"read_restart    {resolved}"

    return re.sub(r"^\s*read_restart\s+(.+?)\s*$", repl, text, count=1, flags=re.M)


def lammps_density_variable_block() -> str:
    return (
        "variable        sysvol      equal vol\n"
        "variable        sysmass     equal mass(all)/6.0221367e+23\n"
        "variable        sysdensity  equal v_sysmass/v_sysvol/1.0e-24\n"
    )


def has_lammps_sysdensity_variable(text: str) -> bool:
    return re.search(r"^\s*variable\s+sysdensity\s+equal\b", text, flags=re.M) is not None


def replace_thermo_block(text: str, thermo_every: int) -> str:
    fix_line = parse_lammps_fix_line(text)
    fix_terms = " ".join(fix_nh_thermo_terms_for_ensemble(stage_ensemble_from_fix(fix_line)))
    fix_terms = f" {fix_terms}" if fix_terms else ""
    density_block = "" if has_lammps_sysdensity_variable(text) else lammps_density_variable_block()
    compute_block = (
        "compute         p_full all pressure thermo_temp\n"
        "compute         p_vir all pressure NULL virial\n"
        f"{density_block}"
    )
    thermo_style = (
        "thermo_style    custom step v_time press "
        "c_p_full c_p_full[1] c_p_full[2] c_p_full[3] c_p_full[4] c_p_full[5] c_p_full[6] "
        "c_p_vir c_p_vir[1] c_p_vir[2] c_p_vir[3] c_p_vir[4] c_p_vir[5] c_p_vir[6] "
        f"vol v_sysdensity temp pe ke etotal{fix_terms}\n"
        f"thermo          {max(1, thermo_every)}"
    )
    text = re.sub(r"^\s*compute\s+p_full\s+.+$", "", text, flags=re.M)
    text = re.sub(r"^\s*compute\s+p_vir\s+.+$", "", text, flags=re.M)
    text = re.sub(r"^\s*thermo_style\s+custom.+$", "", text, flags=re.M)
    text = re.sub(r"^\s*thermo\s+\d+\s*$", "", text, flags=re.M)
    thermo_block = compute_block + thermo_style
    if re.search(r"^\s*fix\s+1\s+all\s+.+$", text, flags=re.M):
        text = re.sub(r"^(\s*fix\s+1\s+all\s+.+)$", r"\1\n" + thermo_block, text, count=1, flags=re.M)
    else:
        text = re.sub(r"^\s*run\s+\d+\b", thermo_block + "\n\\g<0>", text, count=1, flags=re.M)
    return text


def replace_minimize_thermo_block(text: str, thermo_every: int) -> str:
    density_block = "" if has_lammps_sysdensity_variable(text) else lammps_density_variable_block()
    compute_block = (
        "compute         p_full all pressure thermo_temp\n"
        "compute         p_vir all pressure NULL virial\n"
        f"{density_block}"
    )
    thermo_style = (
        "thermo_style    custom step fmax fnorm press "
        "c_p_full c_p_full[1] c_p_full[2] c_p_full[3] c_p_full[4] c_p_full[5] c_p_full[6] "
        "c_p_vir c_p_vir[1] c_p_vir[2] c_p_vir[3] c_p_vir[4] c_p_vir[5] c_p_vir[6] "
        "vol v_sysdensity temp pe ke etotal\n"
        f"thermo          {max(1, thermo_every)}"
    )
    text = re.sub(r"^\s*compute\s+p_full\s+.+$", "", text, flags=re.M)
    text = re.sub(r"^\s*compute\s+p_vir\s+.+$", "", text, flags=re.M)
    text = re.sub(r"^\s*thermo_style\s+custom.+$", "", text, flags=re.M)
    text = re.sub(r"^\s*thermo\s+\d+\s*$", "", text, flags=re.M)
    return re.sub(r"^\s*minimize\s+\S+.+$", compute_block + thermo_style + "\n\\g<0>", text, count=1, flags=re.M)


def write_lammps_start_data_input(original_text: str, out_input: Path, data_out: Path) -> None:
    text = absolute_restart_text(original_text)
    stop = (
        first_match(r"^\s*fix\s+1\s+all\s+", text)
        or first_match(r"^\s*run\s+\S+", text)
        or first_match(r"^\s*minimize\s+", text)
    )
    if stop is None:
        raise ValueError("Cannot find fix/run boundary for start-data input")
    prefix = text[: stop.start()].rstrip()
    if not has_lammps_sysdensity_variable(prefix):
        prefix = prefix + "\n" + lammps_density_variable_block().rstrip()
    body = (
        prefix
        + "\n"
        + "thermo_style    custom step temp press vol v_sysdensity pe ke etotal\n"
        + "thermo          1\n"
        + "thermo_modify   flush yes\n"
        + f"write_data      {data_out}\n"
        + "run             0 post no\n"
    )
    write_text(out_input, body)


def apply_ensemble_override_to_lammps(text: str, ensemble_override: str) -> str:
    if ensemble_override == "original":
        return text
    fix_line = first_match(r"^\s*fix\s+1\s+all\s+.+$", text)
    temp_args = "${tlo} ${tlo} ${tdamp}"
    if fix_line is not None:
        temp_match = re.search(r"\btemp\s+(\S+)\s+(\S+)\s+(\S+)", fix_line.group(0))
        if temp_match:
            temp_args = " ".join(temp_match.groups())
    fix_map = {
        "nve": "fix             1 all nve",
        "nvt": f"fix             1 all nvt temp {temp_args}",
        "npt": None,
    }
    replacement = fix_map.get(ensemble_override)
    if replacement is None:
        raise ValueError(f"Unsupported LAMMPS ensemble override: {ensemble_override}")
    return re.sub(r"^\s*fix\s+1\s+all\s+.+$", replacement, text, count=1, flags=re.M)


def write_lammps_short_input(
    original_text: str,
    out_input: Path,
    restart_out: Path,
    outer_steps: int,
    sample_outer_steps: int,
    ensemble_override: str,
    trace_state: bool,
) -> tuple[int, str]:
    original_run_steps = parse_lammps_run_steps(original_text)
    text = absolute_restart_text(original_text)
    text = apply_ensemble_override_to_lammps(text, ensemble_override)
    text = replace_thermo_block(text, sample_outer_steps)
    trace_block = ""
    if trace_state:
        trace_block = (
            "dump            __pcff_trace all custom 1 lammps_outer_trace.lammpstrj "
            "id type x y z vx vy vz fx fy fz\n"
            "dump_modify     __pcff_trace sort id\n"
        )
    short_run_block = (
        trace_block
        + "variable        __pcff_run_start equal step\n"
        f"variable        __pcff_run_stop equal step+{original_run_steps}\n"
        f"run             {outer_steps} start ${{__pcff_run_start}} stop ${{__pcff_run_stop}}"
    )
    text = re.sub(
        r"^\s*run\s+\S+.*$",
        short_run_block,
        text,
        count=1,
        flags=re.M,
    )
    run_end = text.find(short_run_block)
    if run_end < 0:
        raise ValueError("Could not locate shortened LAMMPS run block")
    text = text[: run_end + len(short_run_block)].rstrip() + f"\nwrite_restart   {restart_out}\n"
    write_text(out_input, text)
    return original_run_steps, text


def write_lammps_minimize_input(
    original_text: str,
    out_input: Path,
    restart_out: Path,
    sample_every: int,
) -> tuple[int, str]:
    maxiter = parse_lammps_minimize_maxiter(original_text)
    text = absolute_restart_text(original_text)
    text = replace_minimize_thermo_block(text, sample_every)
    match = first_match(r"^\s*minimize\s+\S+.+$", text)
    if not match:
        raise ValueError("Could not locate LAMMPS minimize command")
    text = text[: match.end()].rstrip() + f"\nwrite_restart   {restart_out}\n"
    write_text(out_input, text)
    return maxiter, text


def parse_lammps_rows(log_path: Path) -> list[dict[str, float]]:
    header: list[str] | None = None
    rows: list[dict[str, float]] = []
    for raw in read_text(log_path).splitlines():
        fields = raw.split()
        if fields and fields[0] == "Step" and ("v_time" in fields or "Fmax" in fields) and "PotEng" in fields:
            header = fields
            continue
        if header is None:
            continue
        vals = FLOAT_RE.findall(raw)
        if len(vals) == len(header):
            rows.append({key: float(value) for key, value in zip(header, vals)})
    if not rows:
        raise RuntimeError(f"No LAMMPS thermo rows found in {log_path}")
    return rows


def parse_lammps_initial_fix1_vector(log_path: Path, ensemble: str) -> list[float] | None:
    terms = fix_nh_restore_terms_for_ensemble(ensemble)
    if not terms:
        return None
    rows = parse_lammps_rows(log_path)
    first = rows[0]
    if not all(term in first for term in terms):
        return None
    return [first[term] for term in terms]


def summarize_lammps(rows: list[dict[str, float]]) -> dict[str, float | int]:
    first = rows[0]
    final = rows[-1]

    def avg(key: str) -> float | None:
        values = [row[key] for row in rows if key in row]
        return sum(values) / len(values) if values else None

    out: dict[str, float | int] = {
        "lammps_step": final.get("Step", 0.0),
        "lammps_sample_count": len(rows),
    }
    if "v_time" in final:
        out["lammps_time_ps"] = final["v_time"] / 1000.0
    if "Press" in final:
        out["lammps_pressure_bar"] = final["Press"] * ATM_TO_BAR
        if "Press" in first:
            out["lammps_pressure_initial_bar"] = first["Press"] * ATM_TO_BAR
        mean = avg("Press")
        if mean is not None:
            out["lammps_pressure_mean_bar"] = mean * ATM_TO_BAR
    if "Volume" in final:
        out["lammps_volume_nm3"] = final["Volume"] / 1000.0
        if "Volume" in first:
            out["lammps_volume_initial_nm3"] = first["Volume"] / 1000.0
        mean = avg("Volume")
        if mean is not None:
            out["lammps_volume_mean_nm3"] = mean / 1000.0
    if "v_sysdensity" in final:
        out["lammps_density_g_cm3"] = final["v_sysdensity"]
        if "v_sysdensity" in first:
            out["lammps_density_initial_g_cm3"] = first["v_sysdensity"]
        mean = avg("v_sysdensity")
        if mean is not None:
            out["lammps_density_mean_g_cm3"] = mean
    if "Temp" in final:
        out["lammps_temperature_k"] = final["Temp"]
        if "Temp" in first:
            out["lammps_temperature_initial_k"] = first["Temp"]
        mean = avg("Temp")
        if mean is not None:
            out["lammps_temperature_mean_k"] = mean
    if "Fmax" in final:
        out["lammps_fmax"] = final["Fmax"]
    if "Fnorm" in final:
        out["lammps_fnorm"] = final["Fnorm"]
    for source, target in (("PotEng", "potential"), ("KinEng", "kinetic"), ("TotEng", "total")):
        if source in final:
            out[f"lammps_{target}_kj_mol"] = final[source] * KCAL_TO_KJ
            if source in first:
                out[f"lammps_{target}_initial_kj_mol"] = first[source] * KCAL_TO_KJ
            mean = avg(source)
            if mean is not None:
                out[f"lammps_{target}_mean_kj_mol"] = mean * KCAL_TO_KJ
    for term in TENSOR_TERMS:
        if term in final:
            key = term.replace("c_p_full", "pressure_full").replace("c_p_vir", "pressure_virial")
            key = key.replace("[", "_").replace("]", "")
            out[f"lammps_{key}_bar"] = final[term] * ATM_TO_BAR
            if term in first:
                out[f"lammps_{key}_initial_bar"] = first[term] * ATM_TO_BAR
            mean = avg(term)
            if mean is not None:
                out[f"lammps_{key}_mean_bar"] = mean * ATM_TO_BAR
    for term in FIX_NH_VECTOR_TERMS:
        if term in final:
            key = term.replace("f_1[", "fix1_").replace("]", "")
            out[f"lammps_{key}"] = final[term]
            mean = avg(term)
            if mean is not None:
                out[f"lammps_{key}_mean"] = mean
    return out


def make_gmx_short_mdp(
    base_mdp: Path,
    out_mdp: Path,
    base_steps: int,
    sample_base_steps: int,
    ensemble_override: str,
    gmx_nstlist: int | None,
    pme_order_override: int | None = None,
) -> None:
    text = read_text(base_mdp)
    for key, value in {
        "nsteps": str(base_steps),
        "nstcalcenergy": str(max(4, sample_base_steps)),
        "nstenergy": str(max(4, sample_base_steps)),
        "nstlog": str(max(4, sample_base_steps)),
        "nstxout": "0",
        "nstvout": "0",
        "nstfout": "0",
        "nstxout-compressed": "0",
        "continuation": "yes",
        "gen-vel": "no",
    }.items():
        text = replace_mdp_value(text, key, value)
    if ensemble_override == "nve":
        text = replace_mdp_value(text, "tcoupl", "no")
        text = replace_mdp_value(text, "pcoupl", "no")
        text = replace_mdp_value(text, "annealing", "no")
    elif ensemble_override == "nvt":
        text = replace_mdp_value(text, "pcoupl", "no")
    if gmx_nstlist is not None:
        text = replace_mdp_value(text, "nstlist", str(gmx_nstlist))
    if pme_order_override is not None:
        text = replace_mdp_value(text, "pme-order", str(pme_order_override))
    write_text(out_mdp, text)


def make_gmx_minimize_mdp(base_mdp: Path, out_mdp: Path, maxiter: int, sample_steps: int, gmx_nstlist: int | None) -> None:
    text = read_text(base_mdp)
    for key, value in {
        "nsteps": str(maxiter),
        "nstenergy": str(max(1, sample_steps)),
        "nstlog": str(max(1, sample_steps)),
        "nstxout": "0",
        "nstvout": "0",
        "nstfout": "0",
        "nstxout-compressed": "0",
    }.items():
        text = replace_mdp_value(text, key, value)
    if gmx_nstlist is not None:
        text = replace_mdp_value(text, "nstlist", str(gmx_nstlist))
    write_text(out_mdp, text)


def flatten_gmx_energy_terms(terms: dict[str, dict[str, float]]) -> dict[str, float | int]:
    def term(name: str, stat: str) -> float | None:
        value = terms.get(name, {}).get(stat)
        return float(value) if value is not None else None

    out: dict[str, float | int] = {}
    scalar_terms = {
        "Potential": "Potential",
        "Kinetic En.": "Kinetic En.",
        "Total Energy": "Total Energy",
        "Temperature": "Temperature",
        "Pressure": "Pressure",
        "PressureNoDispCorr": "PressureNoDispCorr",
        "Volume": "Volume",
        "Density": "Density",
        "Pres. DC": "Pres. DC",
    }
    for source, target in scalar_terms.items():
        first = term(source, "first")
        last = term(source, "last")
        mean = term(source, "mean")
        if first is not None:
            out[f"{target}_initial"] = first
        if last is not None:
            out[target] = last
        if mean is not None:
            out[f"{target}_mean"] = mean

    component_names = ("XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ")
    for comp in component_names:
        suffix = comp.lower()
        for source_prefix, target_prefix in (
            ("Pres", "gmx_pressure"),
            ("VirialPress", "gmx_virial_pressure"),
            ("Vir", "gmx_virial"),
        ):
            source = f"{source_prefix}-{comp}"
            first = term(source, "first")
            last = term(source, "last")
            mean = term(source, "mean")
            unit_suffix = "bar" if source_prefix != "Vir" else "kj_mol"
            if first is not None:
                out[f"{target_prefix}_{suffix}_initial_{unit_suffix}"] = first
            if last is not None:
                out[f"{target_prefix}_{suffix}_{unit_suffix}"] = last
            if mean is not None:
                out[f"{target_prefix}_{suffix}_mean_{unit_suffix}"] = mean

    for prefix in ("gmx_pressure", "gmx_virial_pressure"):
        values = [out.get(f"{prefix}_{comp}_bar") for comp in ("xx", "yy", "zz")]
        mean_values = [out.get(f"{prefix}_{comp}_mean_bar") for comp in ("xx", "yy", "zz")]
        if all(value is not None for value in values):
            out[f"{prefix}_bar"] = sum(float(value) for value in values) / 3.0
        if all(value is not None for value in mean_values):
            out[f"{prefix}_mean_bar"] = sum(float(value) for value in mean_values) / 3.0

    if terms:
        sample_count = next(iter(terms.values())).get("sample_count")
        if sample_count is not None:
            out["gmx_sample_count"] = int(sample_count)
    return out


def fill_gmx_virial_pressure_from_volume(out: dict[str, Any], volume_nm3: float) -> None:
    if volume_nm3 <= 0:
        return
    for comp in ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"):
        virial_key = f"gmx_virial_{comp}_kj_mol"
        initial_virial_key = f"gmx_virial_{comp}_initial_kj_mol"
        mean_virial_key = f"gmx_virial_{comp}_mean_kj_mol"
        pressure_key = f"gmx_virial_pressure_{comp}_bar"
        initial_pressure_key = f"gmx_virial_pressure_{comp}_initial_bar"
        mean_pressure_key = f"gmx_virial_pressure_{comp}_mean_bar"
        if pressure_key not in out and virial_key in out:
            out[pressure_key] = -2.0 * float(out[virial_key]) / volume_nm3 * KJ_MOL_NM3_TO_BAR
        if initial_pressure_key not in out and initial_virial_key in out:
            out[initial_pressure_key] = -2.0 * float(out[initial_virial_key]) / volume_nm3 * KJ_MOL_NM3_TO_BAR
        if mean_pressure_key not in out and mean_virial_key in out:
            out[mean_pressure_key] = -2.0 * float(out[mean_virial_key]) / volume_nm3 * KJ_MOL_NM3_TO_BAR
    for prefix in ("gmx_virial_pressure",):
        values = [out.get(f"{prefix}_{comp}_bar") for comp in ("xx", "yy", "zz")]
        initial_values = [out.get(f"{prefix}_{comp}_initial_bar") for comp in ("xx", "yy", "zz")]
        mean_values = [out.get(f"{prefix}_{comp}_mean_bar") for comp in ("xx", "yy", "zz")]
        if f"{prefix}_bar" not in out and all(value is not None for value in values):
            out[f"{prefix}_bar"] = sum(float(value) for value in values) / 3.0
        if f"{prefix}_initial_bar" not in out and all(value is not None for value in initial_values):
            out[f"{prefix}_initial_bar"] = sum(float(value) for value in initial_values) / 3.0
        if f"{prefix}_mean_bar" not in out and all(value is not None for value in mean_values):
            out[f"{prefix}_mean_bar"] = sum(float(value) for value in mean_values) / 3.0


def gmx_stage_env(
    stage_key: str,
    original_text: str,
    original_run_steps: int,
    timestep_fs: float,
    config: dict[str, Any],
    lammps_log: Path,
    lammps_fix_vector_log: Path,
) -> dict[str, str]:
    expected = expected_signature_fragment(config, lammps_log)
    beta_by_stage = expected.get("gmx_ewald_beta_inv_a_by_stage", {})
    env: dict[str, str] = {
        "OMP_NUM_THREADS": "12",
        "OMP_PROC_BIND": "true",
        "OMP_PLACES": "cores",
        "GMX_PCFF_EXACT_RESPA_ALLOW_LINEAR_COM_REMOVAL": "1",
        "GMX_PCFF_EXACT_RESPA_NBNXM_OWNER_STEP_SCALAR_FALLBACK": (
            "1" if bool(config.get("GMX_NBNXM_OWNER_STEP_SCALAR_FALLBACK", False)) else "0"
        ),
        "GMX_PCFF_EXACT_RESPA_FUSED_INITIAL_DRIFT": "1",
        "GMX_PCFF_EXACT_RESPA_PRE_TROTTER": str(config.get("GMX_PCFF_EXACT_RESPA_PRE_TROTTER", "two")),
        "GMX_PCFF_EXACT_RESPA_POST_TROTTER": str(config.get("GMX_PCFF_EXACT_RESPA_POST_TROTTER", "three")),
        "GMX_PCFF_NHC_INTEGRATOR": str(config.get("GMX_PCFF_NHC_INTEGRATOR", "lammps")),
    }
    fix_line = parse_lammps_fix_line(original_text)
    ensemble = stage_ensemble_from_fix(fix_line)
    fix_vector = parse_lammps_initial_fix1_vector(lammps_fix_vector_log, ensemble)
    if fix_vector is not None:
        env["GMX_PCFF_RESTORE_NH_MTTK_STATE_FROM_LAMMPS_FIX_VECTOR"] = ",".join(
            f"{value:.17g}" for value in fix_vector
        )
    if ensemble == "npt":
        pressure_mass_scale = dict(config.get("GMX_PCFF_MTTK_PRESSURE_MASS_SCALE_BY_STAGE", {})).get(
            stage_key, config.get("GMX_PCFF_MTTK_PRESSURE_MASS_SCALE", 1.0)
        )
        veta_scale = dict(config.get("GMX_PCFF_MTTK_VETA_SCALE_BY_STAGE", {})).get(
            stage_key, config.get("GMX_PCFF_MTTK_VETA_SCALE", 1.0)
        )
        env.update(
            {
                "GMX_PCFF_MTTK_MASS_MODE": str(
                    dict(config.get("GMX_PCFF_MTTK_MASS_MODE_BY_STAGE", {})).get(
                        stage_key, config.get("GMX_PCFF_MTTK_MASS_MODE", "lammps_pmass")
                    )
                ),
                "GMX_PCFF_MTTK_LAMMPS_NATOMS": "7075",
                "GMX_PCFF_MTTK_LAMMPS_PDAMP_PS": f"{timestep_fs:.9g}",
                "GMX_PCFF_MTTK_PRESSURE_MASS_SCALE": f"{float(pressure_mass_scale):.9g}",
                "GMX_PCFF_MTTK_NRESET_STEPS": str(config.get("GMX_MTTK_NRESET_STEPS", 80000)),
                "GMX_PCFF_EXACT_RESPA_MTTK_EXTENDED_UPDATE": str(
                    config.get("GMX_PCFF_MTTK_EXTENDED_UPDATE_MODE", "velocity-lammps-remap")
                ),
                "GMX_PCFF_EXACT_RESPA_MTTK_INLINE_BOX_REMAP": "1",
                "GMX_PCFF_EXACT_RESPA_MTTK_VETA_SCALE": f"{float(veta_scale):.9g}",
            }
        )
        boxv_integrator = config.get("GMX_PCFF_MTTK_BOXV_INTEGRATOR")
        if boxv_integrator:
            env["GMX_PCFF_MTTK_BOXV_INTEGRATOR"] = str(boxv_integrator)
        if not re.search(r"^\s*kspace_modify\s+compute\s+no\b", original_text, flags=re.M):
            env["GMX_PCFF_EXACT_RESPA_MTTK_INLINE_BOX_REMAP_PME"] = "1"
        ramp = parse_npt_pressure_ramp_atm(fix_line)
        if ramp is not None:
            env["GMX_PCFF_REFP_RAMP_START_BAR"] = f"{ramp[0] * ATM_TO_BAR:.9g}"
            env["GMX_PCFF_REFP_RAMP_END_BAR"] = f"{ramp[1] * ATM_TO_BAR:.9g}"
            env["GMX_PCFF_REFP_RAMP_DURATION_PS"] = f"{original_run_steps * timestep_fs / 1000.0:.9g}"
    else:
        env["GMX_PCFF_MTTK_MASS_MODE"] = str(config.get("GMX_PCFF_NVT_MASS_MODE", "lammps_tchain"))
    if re.search(r"^\s*kspace_modify\s+compute\s+no\b", original_text, flags=re.M):
        beta = beta_by_stage.get(stage_key)
        if beta is not None:
            env["GMX_PCFF_EWALD_BETA_INV_A"] = f"{float(beta):.9g}"
        env["GMX_PCFF_EWALD_REAL_ONLY"] = "1"
    apply_mixed_precision_class2_guard(env, config)
    return env


def run_gmx_short(
    stage_key: str,
    stage_stem: str,
    root: Path,
    gro: Path,
    topol: Path,
    original_text: str,
    original_run_steps: int,
    outer_steps: int,
    sample_outer_steps: int,
    config: dict[str, Any],
    lammps_log: Path,
    lammps_fix_vector_log: Path,
    ntomp: int,
    ensemble_override: str,
    trace_mttk: bool,
    trace_stride: int,
    trace_state: bool,
    trace_atoms: int,
    trace_base_steps: int,
    gmx_nstlist: int | None,
    gmx_mode: str,
    cpuset: str,
) -> dict[str, Any]:
    work = root / f"gmx_{stage_stem}_outer{outer_steps}"
    work.mkdir(parents=True, exist_ok=True)
    timestep_fs = parse_lammps_timestep_fs(original_text)
    base_steps = outer_steps * 4
    sample_base_steps = max(1, sample_outer_steps) * 4
    effective_gmx_mode = gmx_effective_mode_for_stage(gmx_mode, original_text)
    mdp = work / f"{stage_stem}.mdp"
    tpr = work / f"{stage_stem}.tpr"
    deffnm = work / stage_stem
    base_mdp = GMX_CPU_WORK / f"{stage_stem}.mdp"
    if not base_mdp.exists():
        raise FileNotFoundError(base_mdp)
    make_gmx_short_mdp(
        base_mdp,
        mdp,
        base_steps,
        sample_base_steps,
        ensemble_override,
        gmx_nstlist,
        pme_order_override=4 if gmx_mode_uses_pme_gpu(effective_gmx_mode) else None,
    )
    shutil.copy2(gro, work / "system.gro")
    shutil.copy2(topol, work / "topol.top")
    grompp = run([GMX, "grompp", "-f", mdp, "-c", "system.gro", "-p", "topol.top", "-o", tpr, "-maxwarn", "5"], cwd=work)
    write_text(work / "grompp.stdout.log", grompp.stdout)
    if grompp.returncode != 0:
        return {"gmx_status": "grompp_failed", "gmx_returncode": grompp.returncode}
    env = os.environ.copy()
    mdrun_device_args, mdrun_device_env = gmx_mode_args_env(effective_gmx_mode)
    env.update(
        gmx_stage_env(
            stage_key,
            original_text,
            original_run_steps,
            timestep_fs,
            config,
            lammps_log,
            lammps_fix_vector_log,
        )
    )
    env.update(mdrun_device_env)
    env["OMP_NUM_THREADS"] = str(ntomp)
    if trace_mttk and stage_ensemble_from_fix(parse_lammps_fix_line(original_text)) == "npt":
        env["GMX_PCFF_MTTK_STATE_TRACE_FILE"] = str(work / "mttk_state_trace.csv")
        env["GMX_PCFF_MTTK_BOXV_TRACE_FILE"] = str(work / "mttk_boxv_trace.csv")
        env["GMX_PCFF_MTTK_STATE_TRACE_STRIDE"] = str(max(1, trace_stride))
    if trace_state:
        env["GMX_EXACT_RESPA_STATE_TRACE_FILE"] = str(work / "exact_respa_state_trace.tsv")
        env["GMX_EXACT_RESPA_STATE_TRACE_ATOMS"] = str(max(1, trace_atoms))
        env["GMX_EXACT_RESPA_STATE_TRACE_MAX_BASE_STEP"] = str(max(0, trace_base_steps))
        env["GMX_EXACT_RESPA_STATE_TRACE_INCLUDE_POSITIONS"] = "1"
        env["GMX_EXACT_RESPA_FORCESTORE_TRACE_FILE"] = str(work / "exact_respa_forcestore_trace.tsv")
        env["GMX_EXACT_RESPA_FORCESTORE_TRACE_ATOMS"] = str(max(1, trace_atoms))
        env["GMX_EXACT_RESPA_FORCESTORE_TRACE_MAX_BASE_STEP"] = str(max(0, trace_base_steps))
    mdrun_cmd: list[str | Path] = [
        "taskset",
        "-c",
        cpuset,
        GMX,
        "mdrun",
        "-s",
        tpr,
        "-deffnm",
        deffnm,
        "-ntmpi",
        "1",
        "-ntomp",
        str(ntomp),
        "-pin",
        "off",
        "-dlb",
        "no",
        "-notunepme",
        *mdrun_device_args,
    ]
    returncode = run_live_to_file(
        mdrun_cmd,
        cwd=work,
        env=env,
        stdout_path=work / "mdrun.stdout.log",
        label=f"gmx_{stage_stem}_outer{outer_steps}",
    )
    out: dict[str, Any] = {"gmx_status": "ok" if returncode == 0 else "mdrun_failed", "gmx_returncode": returncode}
    if trace_mttk:
        for trace_name in ("mttk_state_trace.csv", "mttk_boxv_trace.csv"):
            trace_path = work / trace_name
            if trace_path.exists():
                out[f"gmx_{trace_name.replace('.csv', '')}"] = str(trace_path)
    if trace_state:
        for trace_name in (
            "exact_respa_state_trace.tsv",
            "exact_respa_state_trace.atom_order.tsv",
            "exact_respa_forcestore_trace.tsv",
            "exact_respa_forcestore_trace.atom_order.tsv",
        ):
            trace_path = work / trace_name
            if trace_path.exists():
                out[f"gmx_{trace_name.replace('.', '_')}"] = str(trace_path)
    if returncode != 0:
        return out
    terms = parse_gmx_energy_terms(GMX, deffnm.with_suffix(".edr"), work)
    out.update(flatten_gmx_energy_terms(terms))
    if deffnm.with_suffix(".gro").exists():
        out["gmx_gro_volume_nm3"] = gro_volume_nm3(deffnm.with_suffix(".gro"))
        fill_gmx_virial_pressure_from_volume(out, float(out["gmx_gro_volume_nm3"]))
    return out


def run_gmx_minimize(
    stage_key: str,
    stage_stem: str,
    root: Path,
    gro: Path,
    topol: Path,
    maxiter: int,
    sample_steps: int,
    ntomp: int,
    gmx_nstlist: int | None,
    gmx_mode: str,
    cpuset: str,
) -> dict[str, Any]:
    work = root / f"gmx_{stage_stem}_minimize"
    work.mkdir(parents=True, exist_ok=True)
    mdp = work / f"{stage_stem}.mdp"
    tpr = work / f"{stage_stem}.tpr"
    deffnm = work / stage_stem
    base_mdp = GMX_CPU_WORK / f"{stage_stem}.mdp"
    if not base_mdp.exists():
        raise FileNotFoundError(base_mdp)
    make_gmx_minimize_mdp(base_mdp, mdp, maxiter, sample_steps, gmx_nstlist)
    shutil.copy2(gro, work / "system.gro")
    shutil.copy2(topol, work / "topol.top")
    grompp = run([GMX, "grompp", "-f", mdp, "-c", "system.gro", "-p", "topol.top", "-o", tpr, "-maxwarn", "5"], cwd=work)
    write_text(work / "grompp.stdout.log", grompp.stdout)
    if grompp.returncode != 0:
        return {"gmx_status": "grompp_failed", "gmx_returncode": grompp.returncode}
    env = os.environ.copy()
    effective_gmx_mode = "cpu" if gmx_mode != "cpu" else gmx_mode
    mdrun_device_args, mdrun_device_env = gmx_mode_args_env(effective_gmx_mode)
    env.update(
        {
            "OMP_NUM_THREADS": str(ntomp),
            "OMP_PROC_BIND": "true",
            "OMP_PLACES": "cores",
            "GMX_PCFF_LAMMPS_CG_EM": "1",
        }
    )
    env.update(mdrun_device_env)
    returncode = run_live_to_file(
        [
            "taskset",
            "-c",
            cpuset,
            GMX,
            "mdrun",
            "-s",
            tpr,
            "-deffnm",
            deffnm,
            "-ntmpi",
            "1",
            "-ntomp",
            str(ntomp),
            "-pin",
            "off",
            *mdrun_device_args,
            "-reprod",
        ],
        cwd=work,
        env=env,
        stdout_path=work / "mdrun.stdout.log",
        label=f"gmx_{stage_stem}_minimize",
    )
    out: dict[str, Any] = {
        "gmx_status": "ok" if returncode == 0 else "mdrun_failed",
        "gmx_returncode": returncode,
        "gmx_minimize_effective_mode": effective_gmx_mode,
    }
    if returncode != 0:
        return out
    terms = parse_gmx_energy_terms(GMX, deffnm.with_suffix(".edr"), work)
    out.update(flatten_gmx_energy_terms(terms))
    if deffnm.with_suffix(".gro").exists():
        out["gmx_gro_volume_nm3"] = gro_volume_nm3(deffnm.with_suffix(".gro"))
        fill_gmx_virial_pressure_from_volume(out, float(out["gmx_gro_volume_nm3"]))
    return out


def add_deltas(row: dict[str, Any]) -> None:
    pairs = {
        "Temperature": "lammps_temperature_k",
        "Temperature_initial": "lammps_temperature_initial_k",
        "Pressure": "lammps_pressure_bar",
        "Pressure_initial": "lammps_pressure_initial_bar",
        "Volume": "lammps_volume_nm3",
        "Volume_initial": "lammps_volume_initial_nm3",
        "Potential": "lammps_potential_kj_mol",
        "Potential_initial": "lammps_potential_initial_kj_mol",
        "Kinetic En.": "lammps_kinetic_kj_mol",
        "Kinetic En._initial": "lammps_kinetic_initial_kj_mol",
        "Total Energy": "lammps_total_kj_mol",
        "Total Energy_initial": "lammps_total_initial_kj_mol",
        "Temperature_mean": "lammps_temperature_mean_k",
        "Pressure_mean": "lammps_pressure_mean_bar",
        "Volume_mean": "lammps_volume_mean_nm3",
        "Potential_mean": "lammps_potential_mean_kj_mol",
        "Kinetic En._mean": "lammps_kinetic_mean_kj_mol",
        "Total Energy_mean": "lammps_total_mean_kj_mol",
    }
    for gmx_key, lmp_key in pairs.items():
        if gmx_key in row and lmp_key in row:
            label = gmx_key.lower().replace(" ", "_").replace(".", "")
            row[f"{label}_delta"] = float(row[gmx_key]) - float(row[lmp_key])
    if "Density" in row and "lammps_density_g_cm3" in row:
        row["density_g_cm3_delta"] = float(row["Density"]) * 0.001 - float(row["lammps_density_g_cm3"])
    if "gmx_gro_volume_nm3" in row and "lammps_volume_nm3" in row:
        gmx_volume = float(row["gmx_gro_volume_nm3"])
        lammps_volume = float(row["lammps_volume_nm3"])
        row["gro_volume_delta_nm3"] = gmx_volume - lammps_volume
        if gmx_volume > 0 and "lammps_density_g_cm3" in row:
            row["gmx_density_from_gro_g_cm3"] = float(row["lammps_density_g_cm3"]) * lammps_volume / gmx_volume
            row["gro_density_g_cm3_delta"] = (
                float(row["gmx_density_from_gro_g_cm3"]) - float(row["lammps_density_g_cm3"])
            )
    if "gmx_virial_pressure_bar" in row and "lammps_pressure_virial_bar" in row:
        row["virial_pressure_delta_bar"] = (
            float(row["gmx_virial_pressure_bar"]) - float(row["lammps_pressure_virial_bar"])
        )
    if "gmx_virial_pressure_initial_bar" in row and "lammps_pressure_virial_initial_bar" in row:
        row["virial_pressure_initial_delta_bar"] = (
            float(row["gmx_virial_pressure_initial_bar"]) - float(row["lammps_pressure_virial_initial_bar"])
        )
    if "gmx_virial_pressure_mean_bar" in row and "lammps_pressure_virial_mean_bar" in row:
        row["virial_pressure_mean_delta_bar"] = (
            float(row["gmx_virial_pressure_mean_bar"]) - float(row["lammps_pressure_virial_mean_bar"])
        )
    component_pairs = {
        "xx": "1",
        "yy": "2",
        "zz": "3",
        "xy": "4",
        "xz": "5",
        "yz": "6",
    }
    for comp, lmp_index in component_pairs.items():
        for gmx_prefix, lmp_prefix, delta_prefix in (
            ("gmx_pressure", "lammps_pressure_full", "pressure"),
            ("gmx_virial_pressure", "lammps_pressure_virial", "virial_pressure"),
        ):
            gmx_key = f"{gmx_prefix}_{comp}_bar"
            lmp_key = f"{lmp_prefix}_{lmp_index}_bar"
            if gmx_key in row and lmp_key in row:
                row[f"{delta_prefix}_{comp}_delta_bar"] = float(row[gmx_key]) - float(row[lmp_key])
            gmx_mean_key = f"{gmx_prefix}_{comp}_mean_bar"
            lmp_mean_key = f"{lmp_prefix}_{lmp_index}_mean_bar"
            if gmx_mean_key in row and lmp_mean_key in row:
                row[f"{delta_prefix}_{comp}_mean_delta_bar"] = (
                    float(row[gmx_mean_key]) - float(row[lmp_mean_key])
                )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_stage_probe(
    lammps_input: Path,
    root: Path,
    outer_steps: int,
    sample_outer_steps: int,
    config: dict[str, Any],
    ntomp: int,
    lammps_ntomp: int,
    ensemble_override: str,
    trace_mttk: bool,
    trace_stride: int,
    trace_state: bool,
    trace_atoms: int,
    trace_base_steps: int,
    gmx_nstlist: int | None,
    gmx_mode: str,
    cpuset: str,
) -> dict[str, Any]:
    rel_input = f"resume_inputs/{lammps_input.name}"
    stage_key = lammps_input_to_gmx_stage_key(rel_input)
    if stage_key is None:
        raise ValueError(f"Cannot map {lammps_input} to a GROMACS stage")
    stage_stem = gmx_stage_key_to_stem(stage_key)
    original_text = read_text(lammps_input)
    stage_root = root / (stage_stem if ensemble_override == "original" else f"{stage_stem}_{ensemble_override}")
    stage_root.mkdir(parents=True, exist_ok=True)
    lammps_log = LAMMPS_WORK / "equil_from_em.stdout.log"

    data_out = stage_root / "same_start.lmp"
    start_input = stage_root / "lammps_write_same_start.in"
    write_lammps_start_data_input(original_text, start_input, data_out)
    lmp_env = os.environ.copy()
    lmp_env.update({"OMP_NUM_THREADS": str(lammps_ntomp), "OMP_PROC_BIND": "true", "OMP_PLACES": "cores"})
    start_return = run_live_to_file(
        [LMP, "-nonbuf", "-sf", "omp", "-pk", "omp", str(lammps_ntomp), "-in", start_input],
        cwd=stage_root,
        env=lmp_env,
        stdout_path=stage_root / "lammps_write_same_start.stdout.log",
        label=f"lammps_start_{stage_stem}",
    )
    if start_return != 0:
        raise RuntimeError(f"LAMMPS same-start write_data failed for {stage_stem}")
    bridge_out = stage_root / "bridge"
    bridge_lammps_data(data_out, bridge_out)
    gro = bridge_out / "system_with_velocities.gro"
    write_gro_with_velocities(bridge_out / "system.gro", data_out, gro)
    topol = bridge_out / "topol.top"

    if first_match(r"^\s*minimize\s+", original_text):
        minimize_input = stage_root / "lammps_minimize.in"
        minimize_restart = stage_root / "lammps_minimize.restart"
        maxiter, lammps_probe_text = write_lammps_minimize_input(
            original_text,
            minimize_input,
            minimize_restart,
            sample_outer_steps,
        )
        lmp_return = run_live_to_file(
            [LMP, "-nonbuf", "-sf", "omp", "-pk", "omp", str(lammps_ntomp), "-in", minimize_input],
            cwd=stage_root,
            env=lmp_env,
            stdout_path=stage_root / "lammps_minimize.stdout.log",
            label=f"lammps_{stage_stem}_minimize",
        )
        row: dict[str, Any] = {
            "stage_key": stage_key,
            "stage_stem": stage_stem,
            "lammps_input": str(lammps_input),
            "stage_kind": "minimize",
            "lammps_minimize_maxiter": maxiter,
            "sample_steps": sample_outer_steps,
            "lammps_timestep_fs": parse_lammps_timestep_fs(original_text),
            "ensemble": "minimize",
            "ensemble_override": ensemble_override,
            "lammps_returncode": lmp_return,
            "lammps_status": "ok" if lmp_return == 0 else "failed",
            "gmx_nstlist_override": gmx_nstlist if gmx_nstlist is not None else "",
            "gmx_probe_mode": gmx_mode,
            "gmx_probe_cpuset": cpuset,
            "root_cause_gate": "valid",
            "root_cause_gate_reason": "same-start x/box minimization from the LAMMPS restart is comparable",
        }
        if lmp_return == 0:
            row.update(summarize_lammps(parse_lammps_rows(stage_root / "lammps_minimize.stdout.log")))
            row.update(
                run_gmx_minimize(
                    stage_key,
                    stage_stem,
                    stage_root,
                    gro,
                    topol,
                    maxiter,
                    sample_outer_steps,
                    ntomp,
                    gmx_nstlist,
                    gmx_mode,
                    cpuset,
                )
            )
            add_deltas(row)
        return row

    short_input = stage_root / "lammps_short.in"
    short_restart = stage_root / "lammps_short.restart"
    original_run_steps, lammps_probe_text = write_lammps_short_input(
        original_text,
        short_input,
        short_restart,
        outer_steps,
        sample_outer_steps,
        ensemble_override,
        trace_state,
    )
    # Production inputs keep their dump command.  The probe runs in an isolated
    # stage directory, so create the relative dump directory before LAMMPS opens it.
    (stage_root / "prod_traj").mkdir(exist_ok=True)
    lmp_return = run_live_to_file(
        [LMP, "-nonbuf", "-sf", "omp", "-pk", "omp", str(lammps_ntomp), "-in", short_input],
        cwd=stage_root,
        env=lmp_env,
        stdout_path=stage_root / "lammps_short.stdout.log",
        label=f"lammps_{stage_stem}_outer{outer_steps}",
    )
    row: dict[str, Any] = {
        "stage_key": stage_key,
        "stage_stem": stage_stem,
        "lammps_input": str(lammps_input),
        "outer_steps": outer_steps,
        "base_steps": outer_steps * 4,
        "sample_outer_steps": sample_outer_steps,
        "original_run_steps": original_run_steps,
        "lammps_timestep_fs": parse_lammps_timestep_fs(original_text),
        "ensemble": stage_ensemble_from_fix(parse_lammps_fix_line(lammps_probe_text)),
        "ensemble_override": ensemble_override,
        "lammps_returncode": lmp_return,
        "lammps_status": "ok" if lmp_return == 0 else "failed",
        "gmx_nstlist_override": gmx_nstlist if gmx_nstlist is not None else "",
        "gmx_probe_mode": gmx_mode,
        "gmx_probe_cpuset": cpuset,
        "gmx_pcff_mttk_boxv_integrator": config.get("GMX_PCFF_MTTK_BOXV_INTEGRATOR", ""),
        "gmx_lammps_fix_vector_restore_available": False,
    }
    gate_status, gate_reason = root_cause_gate_status(stage_key, str(row["ensemble"]))
    row["root_cause_gate"] = gate_status
    row["root_cause_gate_reason"] = gate_reason
    if lmp_return == 0:
        lammps_rows = parse_lammps_rows(stage_root / "lammps_short.stdout.log")
        fix_vector = parse_lammps_initial_fix1_vector(stage_root / "lammps_short.stdout.log", str(row["ensemble"]))
        row["gmx_lammps_fix_vector_restore_available"] = fix_vector is not None
        gate_status, gate_reason = root_cause_gate_status(
            stage_key, str(row["ensemble"]), fix_vector is not None
        )
        row["root_cause_gate"] = gate_status
        row["root_cause_gate_reason"] = gate_reason
        row.update(summarize_lammps(lammps_rows))
        row.update(
            run_gmx_short(
                stage_key,
                stage_stem,
                stage_root,
                gro,
                topol,
                lammps_probe_text,
                original_run_steps,
                outer_steps,
                sample_outer_steps,
                config,
                lammps_log,
                stage_root / "lammps_short.stdout.log",
                ntomp,
                ensemble_override,
                trace_mttk,
                trace_stride,
                trace_state,
                trace_atoms,
                trace_base_steps,
                gmx_nstlist,
                gmx_mode,
                cpuset,
            )
        )
        add_deltas(row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run same-start short propagation probes for PolyGen LAMMPS/GROMACS stages. "
            "This is a root-cause gate; full-stage endpoint agreement is checked separately."
        )
    )
    parser.add_argument("--root", type=Path, default=REPO / "output/probes/polygen_stage_short_probe_20260504")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help="Stage keys, GROMACS stems, LAMMPS input names, or input paths. Default: all MD equilibration chunks.",
    )
    parser.add_argument("--outer-steps", type=int, default=2000, help="LAMMPS outer timesteps to run per probe.")
    parser.add_argument("--sample-outer-steps", type=int, default=2000)
    parser.add_argument("--ensemble-override", choices=["original", "nve", "nvt"], default="original")
    parser.add_argument(
        "--mttk-mass-mode",
        default=None,
        help="Override GMX_PCFF_MTTK_MASS_MODE for NPT probes, e.g. lammps_pmass_pchain.",
    )
    parser.add_argument(
        "--extended-update-mode",
        default=None,
        help="Override GMX_PCFF_EXACT_RESPA_MTTK_EXTENDED_UPDATE, e.g. velocity-lammps-remap or lammps-remap.",
    )
    parser.add_argument("--veta-scale", type=float, default=None, help="Override GMX_PCFF_MTTK_VETA_SCALE.")
    parser.add_argument(
        "--pressure-mass-scale",
        type=float,
        default=None,
        help="Override GMX_PCFF_MTTK_PRESSURE_MASS_SCALE.",
    )
    parser.add_argument("--ntomp", type=int, default=12)
    parser.add_argument("--lammps-ntomp", type=int, default=12)
    parser.add_argument("--cpuset", default="0-11", help="CPU affinity passed to taskset for the GROMACS probe.")
    parser.add_argument(
        "--gmx-mode",
        default="cpu",
        choices=[
            "cpu",
            "nb_gpu_pme_cpu",
            "nb_gpu_pme_cpu_bonded_cpu",
            "nb_gpu_pme_gpu_bonded_cpu",
            "nb_gpu_pme_cpu_bonded_pair14",
            "nb_gpu_pme_cpu_bonded_class2_pair14",
            "nb_gpu_pme_cpu_bonded_all",
            "nb_gpu_pme_gpu_bonded_all",
        ],
        help="mdrun offload mode for the GROMACS probe.",
    )
    parser.add_argument("--gmx", type=Path, default=GMX, help="GROMACS binary to use for grompp/mdrun/energy.")
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument(
        "--trace-mttk",
        action="store_true",
        default=bool(os.environ.get("PCFF_STAGE_PROBE_TRACE_MTTK")),
        help="Write GROMACS MTTK/Nose-Hoover internal traces for NPT probes.",
    )
    parser.add_argument("--trace-stride", type=int, default=1, help="GROMACS MTTK state trace stride in MD steps.")
    parser.add_argument(
        "--trace-state",
        action="store_true",
        default=bool(os.environ.get("PCFF_STAGE_PROBE_TRACE_STATE")),
        help="Write LAMMPS outer-step dumps and GROMACS exact-rRESPA x/v/force-store traces.",
    )
    parser.add_argument("--trace-atoms", type=int, default=128, help="Number of local/canonical atoms to trace.")
    parser.add_argument(
        "--trace-base-steps",
        type=int,
        default=None,
        help="Maximum GROMACS base step for exact-rRESPA traces. Default: outer_steps*4.",
    )
    parser.add_argument("--gmx-nstlist", type=int, default=None, help="Override GROMACS nstlist in the probe MDP.")
    parser.add_argument(
        "--mixed-class2-linear-angle-sin-floor",
        default=None,
        help=(
            "Override GMX_PCFF_MIXED_CLASS2_LINEAR_ANGLE_SIN_FLOOR for non-gmx_d probes. "
            "Use only for mixed/GPU eq01 stability checks."
        ),
    )
    parser.add_argument(
        "--owner-scalar-fallback",
        action="store_true",
        help="Set GMX_PCFF_EXACT_RESPA_NBNXM_OWNER_STEP_SCALAR_FALLBACK=1 for the GROMACS probe.",
    )
    args = parser.parse_args()

    globals()["GMX"] = args.gmx.resolve()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    config = load_notebook_config(args.notebook.resolve())
    if args.mttk_mass_mode:
        config["GMX_PCFF_MTTK_MASS_MODE"] = args.mttk_mass_mode
    if args.extended_update_mode:
        config["GMX_PCFF_MTTK_EXTENDED_UPDATE_MODE"] = args.extended_update_mode
    if args.veta_scale is not None:
        config["GMX_PCFF_MTTK_VETA_SCALE"] = args.veta_scale
    if args.pressure_mass_scale is not None:
        config["GMX_PCFF_MTTK_PRESSURE_MASS_SCALE"] = args.pressure_mass_scale
    if args.mixed_class2_linear_angle_sin_floor is not None:
        config["GMX_PCFF_MIXED_CLASS2_LINEAR_ANGLE_SIN_FLOOR"] = args.mixed_class2_linear_angle_sin_floor
    if args.owner_scalar_fallback:
        config["GMX_NBNXM_OWNER_STEP_SCALAR_FALLBACK"] = True
    inputs = [lammps_input_for_stage(stage) for stage in args.stages] if args.stages else lammps_inputs_for_default_stages()
    rows: list[dict[str, Any]] = []
    summary = args.root / "stage_short_probe_summary.csv"
    for lammps_input in inputs:
        row = run_stage_probe(
            lammps_input,
            args.root,
            args.outer_steps,
            args.sample_outer_steps,
            config,
            args.ntomp,
            args.lammps_ntomp,
            args.ensemble_override,
            args.trace_mttk,
            args.trace_stride,
            args.trace_state,
            args.trace_atoms,
            args.trace_base_steps if args.trace_base_steps is not None else args.outer_steps * 4,
            args.gmx_nstlist,
            args.gmx_mode,
            args.cpuset,
        )
        rows.append(row)
        write_rows(summary, rows)
        print(row, flush=True)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
