#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GMX = REPO_ROOT / "build" / "bin" / "gmx"
NATIVE_MULTI_ENV = "GMX_PCFF_EXACT_RESPA_NATIVE_MULTI"
TOTAL_FORCE_DUMP_ENV = "GMX_EXACT_RESPA_TOTAL_FORCE_DUMP_FILE"
PER_LEVEL_FORCE_DUMP_ENV = "GMX_EXACT_RESPA_PER_LEVEL_FORCE_DUMP_FILE"
DISABLE_SIMD_KERNELS_ENV = "GMX_DISABLE_SIMD_KERNELS"
DISABLE_SIMD_KERNELS_MARKERS = (
    "Found environment variable GMX_DISABLE_SIMD_KERNELS.",
    "Using plain-C-4x4 4x4 nonbonded short-range kernels",
)
ENERGY_FRAME_RE = re.compile(r"time:\s*([\-+0-9.eE]+)\s+step:\s*(\d+)")
ENERGY_TERM_RE = re.compile(r"^\s{2,}(.+?)\s+([\-+0-9.]+(?:e[\-+0-9]+)?)\s*$", re.IGNORECASE)
CHECKPOINT_STEP_RE = re.compile(r"^step = (\d+)$", re.MULTILINE)
STEP_FACTOR_RE = re.compile(r"^\s*(?:exact-respa|mts)-level\d+-factor\s*=\s*(\d+)\s*$", re.MULTILINE)
ENERGY_TERMS = (
    "Potential",
    "Total Energy",
    "Pressure",
    "Vir-XX",
    "Vir-XY",
    "Vir-XZ",
    "Vir-YX",
    "Vir-YY",
    "Vir-YZ",
    "Vir-ZX",
    "Vir-ZY",
    "Vir-ZZ",
    "Pres-XX",
    "Pres-XY",
    "Pres-XZ",
    "Pres-YX",
    "Pres-YY",
    "Pres-YZ",
    "Pres-ZX",
    "Pres-ZY",
    "Pres-ZZ",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact-r-RESPA per-contribution launch and native multi-contribution "
            "single-launch behavior inside the real runtime."
        )
    )
    parser.add_argument("--gmx", type=Path, default=DEFAULT_GMX)
    parser.add_argument("--tpr", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--ntomp", type=int, default=6)
    parser.add_argument("--pin", choices=("off", "on", "auto"), default="off")
    parser.add_argument("--fixture-id", default="unspecified")
    parser.add_argument("--baseline-mode-name", default="per_launch")
    parser.add_argument("--candidate-mode-name", default="native_multi")
    parser.add_argument(
        "--baseline-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment override for the baseline mode. Repeatable.",
    )
    parser.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment override for the candidate mode. Repeatable.",
    )
    parser.add_argument(
        "--mdp",
        type=Path,
        help="Deprecated. Same-coordinate probe now reuses the original TPR plus checkpoint continuation.",
    )
    parser.add_argument(
        "--topol",
        type=Path,
        help="Deprecated. Same-coordinate probe now reuses the original TPR plus checkpoint continuation.",
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=1,
        help="Number of continuation steps for same-coordinate probe. Use 0 to skip the probe.",
    )
    return parser.parse_args()


def parse_env_assignments(assignments: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"Invalid environment assignment {assignment!r}; expected KEY=VALUE")
        key, value = assignment.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid environment assignment {assignment!r}; empty key")
        env[key] = value
    return env


def run_command(command: list[str], *, env: dict[str, str], stdout_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        quoted = " ".join(shlex.quote(part) for part in command)
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {quoted}")


def capture_stdout(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        quoted = " ".join(shlex.quote(part) for part in command)
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {quoted}\n{completed.stdout}\n{completed.stderr}")
    return completed.stdout


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_output_file(deffnm: Path, suffix: str) -> Path:
    direct_path = deffnm.with_suffix(suffix)
    if direct_path.exists():
        return direct_path

    part_paths = sorted(deffnm.parent.glob(f"{deffnm.name}.part*{suffix}"))
    if part_paths:
        return part_paths[-1]

    return direct_path


def checkpoint_step(gmx: Path, cpt_path: Path) -> int:
    dump_text = capture_stdout([str(gmx), "dump", "-cp", str(cpt_path)])
    match = CHECKPOINT_STEP_RE.search(dump_text)
    if match is None:
        raise ValueError(f"Could not parse checkpoint step from {cpt_path}")
    return int(match.group(1))


def slowest_step_factor_from_tpr(gmx: Path, tpr_path: Path) -> int:
    dump_text = capture_stdout([str(gmx), "dump", "-s", str(tpr_path)])
    factors = [int(match.group(1)) for match in STEP_FACTOR_RE.finditer(dump_text)]
    if factors:
        return max(factors)
    return 1


def parse_log_metrics(log_path: Path) -> dict[str, float | None]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    performance_match = re.search(
        r"^Performance:\s+([0-9.eE+-]+)\s+[0-9.eE+-]+\s+([0-9.eE+-]+)",
        text,
        re.MULTILINE,
    )
    metrics: dict[str, float | None] = {
        "ns_per_day": float(performance_match.group(1)) if performance_match else None,
        "ms_per_step": float(performance_match.group(2)) if performance_match else None,
        "force_seconds": None,
        "update_seconds": None,
        "total_seconds": None,
    }
    for label, field in (("Force", "force_seconds"), ("Update", "update_seconds"), ("Total", "total_seconds")):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(label) and (len(stripped) == len(label) or stripped[len(label)].isspace()):
                fields = stripped.split()
                if len(fields) >= 4:
                    try:
                        metrics[field] = float(fields[-3])
                    except ValueError:
                        pass
                break
    return metrics


def log_contains_any_marker(log_path: Path, markers: tuple[str, ...]) -> bool:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return any(marker in text for marker in markers)


def parse_energy_dump(dump_text: str) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in dump_text.splitlines():
        frame_match = ENERGY_FRAME_RE.search(line)
        if frame_match is not None:
            if current is not None:
                frames.append(current)
            current = {"time_ps": float(frame_match.group(1)), "step": int(frame_match.group(2)), "terms": {}}
            continue
        if current is None or ":" in line:
            continue
        term_match = ENERGY_TERM_RE.match(line)
        if term_match is None:
            continue
        current["terms"][term_match.group(1).strip()] = float(term_match.group(2))
    if current is not None:
        frames.append(current)
    if not frames:
        raise ValueError("No energy frames parsed from gmx dump output")
    return frames


def dump_energy_frames(gmx: Path, edr_path: Path, dump_path: Path) -> list[dict[str, object]]:
    dump_text = capture_stdout([str(gmx), "dump", "-e", str(edr_path)])
    dump_path.write_text(dump_text, encoding="utf-8")
    return parse_energy_dump(dump_text)


def parse_total_force_dump(path: Path) -> dict[int, dict[str, object]]:
    frames: dict[int, dict[str, object]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) == 7:
            step_str, time_str, highest_level_str, atom_str, fx_str, fy_str, fz_str = fields
            canonical_atom_str = atom_str
        elif len(fields) == 8:
            (
                step_str,
                time_str,
                highest_level_str,
                atom_str,
                canonical_atom_str,
                fx_str,
                fy_str,
                fz_str,
            ) = fields
        else:
            raise ValueError(f"Unexpected total force dump row with {len(fields)} columns: {line}")
        step = int(step_str)
        frame = frames.setdefault(
            step,
            {
                "time_ps": float(time_str),
                "highest_active_level": int(highest_level_str),
                "forces": {},
            },
        )
        frame["forces"][int(canonical_atom_str)] = [float(fx_str), float(fy_str), float(fz_str)]
    return frames


def parse_per_level_force_dump(path: Path) -> dict[tuple[int, int], dict[str, object]]:
    frames: dict[tuple[int, int], dict[str, object]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) == 8:
            step_str, time_str, highest_level_str, level_str, atom_str, fx_str, fy_str, fz_str = fields
            canonical_atom_str = atom_str
        elif len(fields) == 9:
            (
                step_str,
                time_str,
                highest_level_str,
                level_str,
                atom_str,
                canonical_atom_str,
                fx_str,
                fy_str,
                fz_str,
            ) = fields
        else:
            raise ValueError(f"Unexpected per-level force dump row with {len(fields)} columns: {line}")
        key = (int(step_str), int(level_str))
        frame = frames.setdefault(
            key,
            {
                "time_ps": float(time_str),
                "highest_active_level": int(highest_level_str),
                "mts_level": int(level_str),
                "forces": {},
            },
        )
        frame["forces"][int(canonical_atom_str)] = [float(fx_str), float(fy_str), float(fz_str)]
    return frames


def compare_vector_frames(
    candidate_frames: dict[object, dict[str, object]],
    baseline_frames: dict[object, dict[str, object]],
) -> dict[str, object]:
    candidate_keys = sorted(candidate_frames.keys())
    baseline_keys = sorted(baseline_frames.keys())
    common_keys = [key for key in baseline_keys if key in candidate_frames]
    max_abs_delta = 0.0
    first_mismatch = None
    frame_summaries = []

    for key in common_keys:
        candidate_frame = candidate_frames[key]
        baseline_frame = baseline_frames[key]
        candidate_forces = candidate_frame["forces"]
        baseline_forces = baseline_frame["forces"]
        candidate_atoms = sorted(candidate_forces.keys())
        baseline_atoms = sorted(baseline_forces.keys())
        common_atoms = [atom for atom in baseline_atoms if atom in candidate_forces]
        frame_max_abs_delta = 0.0
        for atom in common_atoms:
            for dimension in range(3):
                delta = float(candidate_forces[atom][dimension]) - float(baseline_forces[atom][dimension])
                abs_delta = abs(delta)
                if abs_delta > max_abs_delta:
                    max_abs_delta = abs_delta
                if abs_delta > frame_max_abs_delta:
                    frame_max_abs_delta = abs_delta
                if first_mismatch is None and delta != 0.0:
                    first_mismatch = {
                        "frame_key": list(key) if isinstance(key, tuple) else key,
                        "atom": atom,
                        "dimension": dimension,
                        "baseline": float(baseline_forces[atom][dimension]),
                        "candidate": float(candidate_forces[atom][dimension]),
                        "delta": delta,
                    }
        frame_summary = {
            "frame_key": list(key) if isinstance(key, tuple) else key,
            "candidate_atom_count": len(candidate_atoms),
            "baseline_atom_count": len(baseline_atoms),
            "common_atom_count": len(common_atoms),
            "max_abs_component_delta": frame_max_abs_delta,
        }
        if "time_ps" in baseline_frame:
            frame_summary["time_ps"] = float(baseline_frame["time_ps"])
        if "highest_active_level" in baseline_frame:
            frame_summary["highest_active_level"] = int(baseline_frame["highest_active_level"])
        if "mts_level" in baseline_frame:
            frame_summary["mts_level"] = int(baseline_frame["mts_level"])
        frame_summaries.append(frame_summary)

    return {
        "matches": baseline_keys == candidate_keys and first_mismatch is None,
        "candidate_frame_count": len(candidate_keys),
        "baseline_frame_count": len(baseline_keys),
        "common_frame_count": len(common_keys),
        "candidate_only_frames": [list(key) if isinstance(key, tuple) else key for key in candidate_keys if key not in baseline_frames],
        "baseline_only_frames": [list(key) if isinstance(key, tuple) else key for key in baseline_keys if key not in candidate_frames],
        "max_abs_component_delta": max_abs_delta,
        "first_mismatch": first_mismatch,
        "frames": frame_summaries,
    }


def compare_energy_frames(
    candidate_frames: list[dict[str, object]],
    baseline_frames: list[dict[str, object]],
    selected_terms: tuple[str, ...],
) -> dict[str, object]:
    first_mismatch = None
    max_abs_delta = 0.0
    frame_count = min(len(candidate_frames), len(baseline_frames))
    frames = []

    for index in range(frame_count):
        candidate_terms = dict(candidate_frames[index]["terms"])
        baseline_terms = dict(baseline_frames[index]["terms"])
        deltas = {}
        for term in selected_terms:
            if term not in candidate_terms or term not in baseline_terms:
                continue
            delta = float(candidate_terms[term]) - float(baseline_terms[term])
            deltas[term] = delta
            if abs(delta) > max_abs_delta:
                max_abs_delta = abs(delta)
            if first_mismatch is None and delta != 0.0:
                first_mismatch = {
                    "frame_index": index,
                    "step": int(baseline_frames[index]["step"]),
                    "term": term,
                    "baseline": float(baseline_terms[term]),
                    "candidate": float(candidate_terms[term]),
                    "delta": delta,
                }
        frames.append(
            {
                "frame_index": index,
                "step": int(baseline_frames[index]["step"]),
                "time_ps": float(baseline_frames[index]["time_ps"]),
                "term_deltas": deltas,
            }
        )

    return {
        "matches": len(candidate_frames) == len(baseline_frames) and first_mismatch is None,
        "candidate_frame_count": len(candidate_frames),
        "baseline_frame_count": len(baseline_frames),
        "common_frame_count": frame_count,
        "max_abs_delta": max_abs_delta,
        "first_mismatch": first_mismatch,
        "frames": frames,
    }


def parse_gro(path: Path) -> tuple[list[list[float]], list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    natoms = int(lines[1].strip())
    coordinates: list[list[float]] = []
    for line in lines[2 : 2 + natoms]:
        coordinates.append([float(line[20:28]), float(line[28:36]), float(line[36:44])])
    box = [float(value) for value in lines[2 + natoms].split()]
    return coordinates, box


def compare_gro(candidate_path: Path, baseline_path: Path) -> dict[str, object]:
    candidate_coords, candidate_box = parse_gro(candidate_path)
    baseline_coords, baseline_box = parse_gro(baseline_path)
    max_abs_coord_delta = 0.0
    first_mismatch = None
    for atom, (candidate_coord, baseline_coord) in enumerate(zip(candidate_coords, baseline_coords)):
        for dimension, (candidate_value, baseline_value) in enumerate(zip(candidate_coord, baseline_coord)):
            delta = candidate_value - baseline_value
            abs_delta = abs(delta)
            if abs_delta > max_abs_coord_delta:
                max_abs_coord_delta = abs_delta
            if first_mismatch is None and delta != 0.0:
                first_mismatch = {
                    "atom": atom,
                    "dimension": dimension,
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "delta": delta,
                }
    max_abs_box_delta = max(abs(candidate - baseline) for candidate, baseline in zip(candidate_box, baseline_box))
    return {
        "matches": candidate_coords == baseline_coords and candidate_box == baseline_box,
        "atom_count": len(baseline_coords),
        "max_abs_coord_delta_nm": max_abs_coord_delta,
        "max_abs_box_delta_nm": max_abs_box_delta,
        "first_mismatch": first_mismatch,
    }


def extract_first_frame_comparison(comparison: dict[str, object]) -> dict[str, object] | None:
    frames = comparison["frames"]
    return frames[0] if frames else None


def extract_first_level_comparison(comparison: dict[str, object], level: int) -> dict[str, object] | None:
    for frame in comparison["frames"]:
        if frame.get("mts_level") == level:
            return frame
    return None


def run_mode(
    args: argparse.Namespace,
    mode: str,
    native_multi_value: str,
    *,
    extra_env: dict[str, str] | None = None,
    tpr_override: Path | None = None,
    cpi_path: Path | None = None,
    noappend: bool = False,
    steps_override: int | None = None,
) -> dict[str, object]:
    run_dir = args.output_dir / mode
    total_force_path = run_dir / "total_force.tsv"
    per_level_force_path = run_dir / "per_level_force.tsv"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env["OMP_NUM_THREADS"] = str(args.ntomp)
    env[NATIVE_MULTI_ENV] = native_multi_value
    env[TOTAL_FORCE_DUMP_ENV] = str(total_force_path)
    env[PER_LEVEL_FORCE_DUMP_ENV] = str(per_level_force_path)

    deffnm = run_dir / "run"
    tpr_path = args.tpr if tpr_override is None else tpr_override
    command = [
        str(args.gmx),
        "mdrun",
        "-s",
        str(tpr_path),
        "-deffnm",
        str(deffnm),
        "-ntmpi",
        "1",
        "-ntomp",
        str(args.ntomp),
        "-dlb",
        "no",
        "-nb",
        "cpu",
        "-pme",
        "cpu",
        "-bonded",
        "cpu",
        "-update",
        "cpu",
        "-pin",
        args.pin,
        "-reprod",
    ]
    if cpi_path is not None:
        command.extend(["-cpi", str(cpi_path)])
    if noappend:
        command.append("-noappend")
    effective_steps = args.steps if steps_override is None else steps_override
    if effective_steps is not None:
        command.extend(["-nsteps", str(effective_steps)])

    started = time.time()
    run_command(command, env=env, stdout_path=run_dir / "mdrun.stdout.txt")
    elapsed = time.time() - started

    log_path = find_output_file(deffnm, ".log")
    edr_path = find_output_file(deffnm, ".edr")
    gro_path = find_output_file(deffnm, ".gro")
    output_cpt_path = find_output_file(deffnm, ".cpt")
    energy_dump_path = run_dir / "run.energy.dump.txt"

    return {
        "mode": mode,
        "native_multi_env": native_multi_value,
        "extra_env": dict(sorted((extra_env or {}).items())),
        "disable_simd_kernels_env": env.get(DISABLE_SIMD_KERNELS_ENV, "0"),
        "command": command,
        "elapsed_seconds": elapsed,
        "log": str(log_path),
        "edr": str(edr_path),
        "gro": str(gro_path),
        "cpt": str(output_cpt_path),
        "total_force_dump": str(total_force_path),
        "per_level_force_dump": str(per_level_force_path),
        "metrics": parse_log_metrics(log_path),
        "disable_simd_kernels_marker_seen": log_contains_any_marker(log_path, DISABLE_SIMD_KERNELS_MARKERS),
        "energy_frames": dump_energy_frames(args.gmx, edr_path, energy_dump_path),
        "total_force_frames": parse_total_force_dump(total_force_path),
        "per_level_force_frames": parse_per_level_force_dump(per_level_force_path),
        "gro_sha256": file_sha256(gro_path),
        "edr_sha256": file_sha256(edr_path),
        "cpt_sha256": file_sha256(output_cpt_path),
    }


def run_same_coordinate_probe(
    args: argparse.Namespace,
    baseline: dict[str, object],
    baseline_mode_name: str,
    candidate_mode_name: str,
    baseline_extra_env: dict[str, str],
    candidate_extra_env: dict[str, str],
) -> dict[str, object]:
    if args.probe_steps <= 0:
        return {
            "available": False,
            "reason": "probe_steps<=0",
            "probe_steps": 0,
        }

    probe_root = args.output_dir / "same_coordinate_probe"
    probe_root.mkdir(parents=True, exist_ok=True)
    probe_tpr = probe_root / "same_coordinate_probe.tpr"
    baseline_cpt = Path(baseline["cpt"])
    probe_start_step = checkpoint_step(args.gmx, baseline_cpt)
    outer_step_factor = slowest_step_factor_from_tpr(args.gmx, args.tpr)
    probe_end_step = probe_start_step + max(args.probe_steps, 1)
    while probe_end_step % outer_step_factor != 0:
        probe_end_step += 1
    convert_tpr_command = [
        str(args.gmx),
        "convert-tpr",
        "-s",
        str(args.tpr),
        "-nsteps",
        str(probe_end_step),
        "-o",
        str(probe_tpr),
    ]
    run_command(convert_tpr_command, env=os.environ.copy(), stdout_path=probe_root / "convert_tpr.stdout.txt")

    probe_args = argparse.Namespace(
        gmx=args.gmx,
        tpr=probe_tpr,
        output_dir=probe_root / "runs",
        steps=None,
        ntomp=args.ntomp,
        pin=args.pin,
        fixture_id=f"{args.fixture_id}_same_coordinate_probe",
        mdp=args.mdp,
        topol=args.topol,
        probe_steps=max(args.probe_steps, 1),
    )
    probe_baseline = run_mode(
        probe_args,
        baseline_mode_name,
        "0",
        extra_env=baseline_extra_env,
        tpr_override=probe_tpr,
        cpi_path=baseline_cpt,
        noappend=True,
    )
    probe_candidate = run_mode(
        probe_args,
        candidate_mode_name,
        "1",
        extra_env=candidate_extra_env,
        tpr_override=probe_tpr,
        cpi_path=baseline_cpt,
        noappend=True,
    )
    total_force = compare_vector_frames(probe_candidate["total_force_frames"], probe_baseline["total_force_frames"])
    per_level_force = compare_vector_frames(
        probe_candidate["per_level_force_frames"], probe_baseline["per_level_force_frames"]
    )
    energy = compare_energy_frames(probe_candidate["energy_frames"], probe_baseline["energy_frames"], ENERGY_TERMS)
    gro = compare_gro(Path(probe_candidate["gro"]), Path(probe_baseline["gro"]))

    return {
        "available": True,
        "probe_tpr": str(probe_tpr),
        "probe_start_step": probe_start_step,
        "probe_end_step": probe_end_step,
        "outer_step_factor": outer_step_factor,
        "probe_steps": probe_end_step - probe_start_step,
        "convert_tpr_command": convert_tpr_command,
        "baseline": {
            key: value
            for key, value in probe_baseline.items()
            if key not in ("energy_frames", "total_force_frames", "per_level_force_frames")
        },
        "candidate": {
            key: value
            for key, value in probe_candidate.items()
            if key not in ("energy_frames", "total_force_frames", "per_level_force_frames")
        },
        "comparisons": {
            "total_force": total_force,
            "per_level_force": per_level_force,
            "energy": energy,
            "gro": gro,
            "first_total_force_frame": extract_first_frame_comparison(total_force),
            "first_level0_force_frame": extract_first_level_comparison(per_level_force, 0),
            "first_level1_force_frame": extract_first_level_comparison(per_level_force, 1),
            "first_level2_force_frame": extract_first_level_comparison(per_level_force, 2),
        },
    }


def main() -> None:
    args = parse_args()
    args.gmx = args.gmx.resolve()
    args.tpr = args.tpr.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.mdp is not None:
        args.mdp = args.mdp.resolve()
    if args.topol is not None:
        args.topol = args.topol.resolve()

    baseline_extra_env = parse_env_assignments(args.baseline_env)
    candidate_extra_env = parse_env_assignments(args.candidate_env)

    baseline = run_mode(
        args,
        args.baseline_mode_name,
        "0",
        extra_env=baseline_extra_env,
    )
    candidate = run_mode(
        args,
        args.candidate_mode_name,
        "1",
        extra_env=candidate_extra_env,
    )

    total_force_comparison = compare_vector_frames(
        candidate["total_force_frames"], baseline["total_force_frames"]
    )
    per_level_force_comparison = compare_vector_frames(
        candidate["per_level_force_frames"], baseline["per_level_force_frames"]
    )
    energy_comparison = compare_energy_frames(
        candidate["energy_frames"], baseline["energy_frames"], ENERGY_TERMS
    )
    gro_comparison = compare_gro(Path(candidate["gro"]), Path(baseline["gro"]))
    same_coordinate_probe = run_same_coordinate_probe(
        args,
        baseline,
        args.baseline_mode_name,
        args.candidate_mode_name,
        baseline_extra_env,
        candidate_extra_env,
    )

    report = {
        "schema_name": "exact_respa_native_multi_runtime_parity",
        "schema_version": 1,
        "fixture_id": args.fixture_id,
        "tpr": str(args.tpr),
        "steps": args.steps,
        "ntmpi": 1,
        "ntomp": args.ntomp,
        "pin": args.pin,
        "baseline_mode_name": args.baseline_mode_name,
        "candidate_mode_name": args.candidate_mode_name,
        "baseline_env": dict(sorted(baseline_extra_env.items())),
        "candidate_env": dict(sorted(candidate_extra_env.items())),
        "baseline": {
            key: value
            for key, value in baseline.items()
            if key not in ("energy_frames", "total_force_frames", "per_level_force_frames")
        },
        "candidate": {
            key: value
            for key, value in candidate.items()
            if key not in ("energy_frames", "total_force_frames", "per_level_force_frames")
        },
        "comparisons": {
            "total_force": total_force_comparison,
            "per_level_force": per_level_force_comparison,
            "energy": energy_comparison,
            "gro": gro_comparison,
            "same_coordinate_probe": same_coordinate_probe,
            "hashes": {
                "gro_sha256_equal": baseline["gro_sha256"] == candidate["gro_sha256"],
                "edr_sha256_equal": baseline["edr_sha256"] == candidate["edr_sha256"],
                "cpt_sha256_equal": baseline["cpt_sha256"] == candidate["cpt_sha256"],
            },
            "performance": {
                "baseline_ns_per_day": baseline["metrics"]["ns_per_day"],
                "candidate_ns_per_day": candidate["metrics"]["ns_per_day"],
                "speedup": (
                    float(candidate["metrics"]["ns_per_day"]) / float(baseline["metrics"]["ns_per_day"])
                    if baseline["metrics"]["ns_per_day"] and candidate["metrics"]["ns_per_day"]
                    else None
                ),
                "baseline_force_seconds": baseline["metrics"]["force_seconds"],
                "candidate_force_seconds": candidate["metrics"]["force_seconds"],
                "baseline_update_seconds": baseline["metrics"]["update_seconds"],
                "candidate_update_seconds": candidate["metrics"]["update_seconds"],
            },
        },
        "notes": [
            "Hashes are reported separately from force/energy/virial deltas.",
            "A hash mismatch alone is not sufficient evidence of semantic failure or success.",
            "Current harness compares runtime total-force dumps, per-level force dumps, EDR dump terms, and final GRO coordinates.",
            "The same-coordinate probe reuses the original TPR plus checkpoint continuation instead of rebuilding a new TPR from MDP/topology inputs.",
        ],
    }

    report_path = args.output_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary_path = args.output_dir / "report.tsv"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "\t".join(
                (
                    "fixture_id",
                    "ntomp",
                    "steps",
                    "baseline_ns_per_day",
                    "candidate_ns_per_day",
                    "speedup",
                    "total_force_max_abs_component_delta",
                    "per_level_force_max_abs_component_delta",
                    "energy_max_abs_delta",
                    "gro_max_abs_coord_delta_nm",
                    "gro_sha256_equal",
                    "edr_sha256_equal",
                    "cpt_sha256_equal",
                )
            )
            + "\n"
        )
        handle.write(
            "\t".join(
                (
                    args.fixture_id,
                    str(args.ntomp),
                    str(args.steps),
                    str(baseline["metrics"]["ns_per_day"]),
                    str(candidate["metrics"]["ns_per_day"]),
                    str(report["comparisons"]["performance"]["speedup"]),
                    str(total_force_comparison["max_abs_component_delta"]),
                    str(per_level_force_comparison["max_abs_component_delta"]),
                    str(energy_comparison["max_abs_delta"]),
                    str(gro_comparison["max_abs_coord_delta_nm"]),
                    str(report["comparisons"]["hashes"]["gro_sha256_equal"]).lower(),
                    str(report["comparisons"]["hashes"]["edr_sha256_equal"]).lower(),
                    str(report["comparisons"]["hashes"]["cpt_sha256_equal"]).lower(),
                )
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
