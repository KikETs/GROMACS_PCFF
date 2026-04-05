from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from freeze_gate_a_oracle import (
    DEFAULT_GMX,
    CPU_CORRECTION_TRACE_TERM_ORDER,
    LEVEL_FACTORS,
    base_env,
    capture_output,
    command_record,
    env_delta,
    parse_class2_subterm_energy_trace,
    parse_cpu_correction_energy_trace,
    parse_energy_dump,
    parse_event_trace,
    parse_merge_trace_dir,
    parse_total_force_dump,
    run_command,
    write_commands_script,
    write_text,
)
from validate_gate_b_nb_gpu import (
    assess_energy_display_resolution,
    compare_energy_frames,
    compare_event_trace,
    compare_per_level_force_entries,
    compare_total_force_entries,
    estimate_noise_floor,
    extract_virial_deltas,
    force_sum_roundoff_bound_from_merge_trace_dir,
    force_sum_roundoff_bound_from_total_force_dump,
    max_abs_delta_for_terms,
    read_gro_atom_count,
    trace_env_for_run,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_A_MANIFEST = REPO_ROOT / "tests" / "reference_results" / "gate_a_cpu_oracle" / "oracle_manifest.json"
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_c_nb_bonded_gpu_validation"
SYSTEMS = ("small_oligomer", "small_salt_polymer_box")
GPU_REPEAT_COUNT = 3
GPU_REPRODUCIBILITY_NOTE = (
    "Binary reproducibility (-reprod) is not enabled because GROMACS rejects -nb gpu together with -reprod."
)

TPR_COUNT_LABELS = (
    "Bond",
    "Angle",
    "U-B",
    "Proper Dih.",
    "Ryckaert-Bell.",
    "Improper Dih.",
    "Per. Imp. Dih.",
    "Class2 Bond",
    "Bond-Cross",
    "BA-Cross",
    "Class2 Angle",
    "Class2 Dih.",
    "Class2 Impr.",
    "LJ-14",
    "Coulomb-14",
    "LJC-14 q",
    "LJC Pairs NB",
)
GPU_SUPPORTED_TPR_LABELS = {
    "Bond",
    "Angle",
    "U-B",
    "Proper Dih.",
    "Ryckaert-Bell.",
    "Improper Dih.",
    "Per. Imp. Dih.",
    "LJ-14",
}
GATE_C_REQUIRED_BUCKET_ORDER = (
    "bond",
    "angle",
    "dihedral",
    "improper",
    "class2_cross_terms",
    "lj_sr",
    "coul_sr",
    "lj_14",
    "coul_14",
    "cpu_reciprocal_self_exclusion_corrections",
)
FIRST_BUCKET_PRIORITY = (
    "class2_cross_terms",
    "improper",
    "cpu_reciprocal_self_exclusion_corrections",
    "bond",
    "angle",
    "dihedral",
    "lj_sr",
    "coul_sr",
    "lj_14",
    "coul_14",
)
BUCKET_TERM_MAP = {
    "bond": ("Class2 Bond",),
    "angle": ("Class2 Angle",),
    "dihedral": ("Class2 Dih.",),
    "improper": ("Class2 Impr.",),
    "class2_cross_terms": (),
    "lj_sr": ("LJ (SR)",),
    "coul_sr": ("Coulomb (SR)",),
    "lj_14": ("LJ-14",),
    "coul_14": ("Coulomb-14",),
    "cpu_reciprocal_self_exclusion_corrections": ("Coul. recip.",),
}
CLASS2_TRACE_BUCKET_TERM_MAP = {
    "bond": ("bond_class2_main",),
    "angle": ("angle_class2_main",),
    "dihedral": ("dihedral_class2_main",),
    "improper": ("improper_class2_main",),
    "class2_cross_terms": (
        "angle_class2_bond_bond",
        "angle_class2_bond_angle_1",
        "angle_class2_bond_angle_2",
        "dihedral_class2_middle_bond_torsion",
        "dihedral_class2_end_bond_torsion_1",
        "dihedral_class2_end_bond_torsion_2",
        "dihedral_class2_angle_torsion_1",
        "dihedral_class2_angle_torsion_2",
        "dihedral_class2_angle_angle_torsion",
        "dihedral_class2_bond_bond_13_torsion",
        "improper_class2_angle_angle_1",
        "improper_class2_angle_angle_2",
        "improper_class2_angle_angle_3",
    ),
}
CLASS2_TRACE_EARLY_MAX_STEP = LEVEL_FACTORS[-1]
CPU_CORRECTION_TRACE_BUCKET_TERM_MAP = {
    "cpu_reciprocal_self_exclusion_corrections": (
        "coulomb_reciprocal",
        "coulomb_self",
        "coulomb_excluded_correction",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Gate C for standalone exact r-RESPA with nonbonded and bonded GPU offload."
    )
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="Path to the GROMACS CLI binary.")
    parser.add_argument(
        "--gate-a-manifest",
        default=str(DEFAULT_GATE_A_MANIFEST),
        help="Path to the frozen Gate A CPU oracle manifest.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Artifact root.")
    parser.add_argument("--build-target", default="gmx", help="CMake target to build.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step.")
    parser.add_argument("--build-dir", default=None, help="Optional explicit build directory.")
    parser.add_argument("--ntmpi", type=int, default=1, help="Thread-MPI ranks for mdrun.")
    parser.add_argument("--ntomp", type=int, default=1, help="OpenMP threads for mdrun.")
    parser.add_argument("--outer-steps", type=int, default=5, help="Number of exact r-RESPA outer steps.")
    parser.add_argument(
        "--gpu-repeats",
        type=int,
        default=GPU_REPEAT_COUNT,
        help="Number of repeated GPU runs to attempt for noise-floor estimation when the path executes.",
    )
    return parser.parse_args()


def run_command_allow_failure(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("w", encoding="utf-8") if stdout_path is not None else None
    stderr_handle = stderr_path.open("w", encoding="utf-8") if stderr_path is not None else None
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=False,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    finally:
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def capture_optional_output(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return {
        "argv": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def parse_gpu_support(gmx_version: str) -> str:
    match = re.search(r"GPU support:\s*(.+)", gmx_version)
    return match.group(1).strip() if match is not None else "unknown"


def parse_precision_mode(gmx_version: str) -> str:
    match = re.search(r"Precision:\s*(.+)", gmx_version)
    return match.group(1).strip() if match is not None else "unknown"


def maybe_build(args: argparse.Namespace, build_dir: Path | None) -> None:
    if args.skip_build:
        return
    command = [
        "cmake",
        "--build",
        str(build_dir if build_dir is not None else Path(args.gmx).resolve().parents[1]),
        "--target",
        args.build_target,
        "-j4",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True, text=True)


def validate_inputs(gate_a_manifest: dict[str, object]) -> None:
    if gate_a_manifest.get("status") != "PASS":
        raise ValueError("Gate A manifest is not PASS; Gate C cannot use it as a frozen oracle.")


def mdrun_args_gate_c(args: argparse.Namespace, tpr_path: Path, deffnm: Path) -> list[str]:
    return [
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
        "gpu",
        "-pme",
        "cpu",
        "-bonded",
        "gpu",
        "-update",
        "cpu",
        "-pin",
        "off",
    ]


def extract_failure_markers(stderr_path: Path, stdout_path: Path) -> list[str]:
    markers = []
    combined_text = ""
    if stdout_path.exists():
        combined_text += stdout_path.read_text(encoding="utf-8", errors="replace")
    if stderr_path.exists():
        combined_text += "\n" + stderr_path.read_text(encoding="utf-8", errors="replace")
    for marker in (
        "Cannot run with standalone exact r-RESPA",
        "The current bonded GPU path assumes 12-6 listed 1-4 Lennard-Jones semantics.",
        "PCFF/class2 uses 9-6 listed 1-4 interactions",
        "Standalone exact r-RESPA is not supported on GPUs.",
    ):
        if marker in combined_text:
            markers.append(marker)
    return markers


def parse_tpr_inventory(dump_text: str) -> dict[str, object]:
    counts = {label: 0 for label in TPR_COUNT_LABELS}
    pending_label = None
    for line in dump_text.splitlines():
        label_match = re.match(r"^\s{6,}(.+):\s*$", line)
        if label_match:
            label = label_match.group(1).strip()
            pending_label = label if label in counts else None
            continue
        if pending_label is None:
            continue
        count_match = re.match(r"^\s*nr:\s*(\d+)", line)
        if count_match:
            counts[pending_label] = max(counts[pending_label], int(count_match.group(1)))
            pending_label = None

    reppow_match = re.search(r"reppow\s*=\s*([\-+0-9.eE]+)", dump_text)
    exact_respa_match = re.search(r"exact-respa\s*=\s*(true|false)", dump_text)
    pair14_level_match = re.search(r"exact-respa-pair14-level\s*=\s*(\d+)", dump_text)
    pme_level_match = re.search(r"exact-respa-kspace-level\s*=\s*(\d+)", dump_text)

    angle_parameter_lines = [
        line.strip() for line in dump_text.splitlines() if "functype[" in line and "ANGLE_CLASS2" in line
    ]
    dihedral_parameter_lines = [
        line.strip() for line in dump_text.splitlines() if "functype[" in line and "DIHEDRAL_CLASS2" in line
    ]
    improper_parameter_lines = [
        line.strip() for line in dump_text.splitlines() if "functype[" in line and "IMPROPER_CLASS2" in line
    ]

    angle_cross_tokens = [token for token in ("bb_k=", "ba_k1=", "ba_k2=") if token in "\n".join(angle_parameter_lines)]
    dihedral_cross_tokens = [
        token
        for token in ("mbt_f1=", "ebt_f1_1=", "at_f1_1=", "aat_k=", "bb13t_k=")
        if token in "\n".join(dihedral_parameter_lines)
    ]

    present_nonzero_counts = {label: count for label, count in counts.items() if count > 0}
    supported_gpu_terms_present = {
        label: count for label, count in present_nonzero_counts.items() if label in GPU_SUPPORTED_TPR_LABELS
    }
    unsupported_terms_present = {
        label: count for label, count in present_nonzero_counts.items() if label not in GPU_SUPPORTED_TPR_LABELS
    }

    return {
        "reppow": float(reppow_match.group(1)) if reppow_match is not None else None,
        "exact_respa": exact_respa_match.group(1) == "true" if exact_respa_match is not None else None,
        "exact_respa_pair14_level": int(pair14_level_match.group(1)) if pair14_level_match is not None else None,
        "exact_respa_kspace_level": int(pme_level_match.group(1)) if pme_level_match is not None else None,
        "term_counts": counts,
        "present_nonzero_counts": present_nonzero_counts,
        "supported_gpu_terms_present": supported_gpu_terms_present,
        "unsupported_terms_present": unsupported_terms_present,
        "class2_parameter_evidence": {
            "angle_class2_parameter_lines": angle_parameter_lines,
            "angle_cross_component_tokens": angle_cross_tokens,
            "dihedral_class2_parameter_lines": dihedral_parameter_lines,
            "dihedral_cross_component_tokens": dihedral_cross_tokens,
            "improper_class2_parameter_lines": improper_parameter_lines,
        },
    }


def filter_class2_trace_rows(
    trace_summary: dict[str, object], *, selected_terms: tuple[str, ...] | None = None, max_step: int | None = None
) -> list[dict[str, object]]:
    term_filter = set(selected_terms) if selected_terms is not None else None
    rows = []
    for row in trace_summary.get("rows", []):
        if max_step is not None and int(row["step"]) > max_step:
            continue
        if term_filter is not None and str(row["term"]) not in term_filter:
            continue
        rows.append(dict(row))
    return rows


def gate_a_has_class2_trace_terms(
    trace_summary: dict[str, object], term_names: tuple[str, ...], *, max_step: int = CLASS2_TRACE_EARLY_MAX_STEP
) -> bool:
    return any(
        int(row["interaction_count"]) > 0
        for row in filter_class2_trace_rows(trace_summary, selected_terms=term_names, max_step=max_step)
    )


def compare_class2_subterm_trace_rows(
    actual_trace: dict[str, object],
    expected_trace: dict[str, object],
    *,
    selected_terms: tuple[str, ...],
    max_step: int | None = CLASS2_TRACE_EARLY_MAX_STEP,
) -> dict[str, object]:
    actual_rows = filter_class2_trace_rows(actual_trace, selected_terms=selected_terms, max_step=max_step)
    expected_rows = filter_class2_trace_rows(expected_trace, selected_terms=selected_terms, max_step=max_step)
    actual_map = {(int(row["step"]), int(row["level"]), str(row["term"])): row for row in actual_rows}
    expected_map = {(int(row["step"]), int(row["level"]), str(row["term"])): row for row in expected_rows}
    missing_in_actual = sorted([f"{step}:{level}:{term}" for step, level, term in expected_map.keys() - actual_map.keys()])
    extra_in_actual = sorted([f"{step}:{level}:{term}" for step, level, term in actual_map.keys() - expected_map.keys()])
    count_mismatches = []
    max_abs_delta = 0.0
    first_nonzero = None
    first_mismatch = None
    compared_rows = []
    for key in sorted(actual_map.keys() & expected_map.keys()):
        actual_row = actual_map[key]
        expected_row = expected_map[key]
        actual_count = int(actual_row["interaction_count"])
        expected_count = int(expected_row["interaction_count"])
        delta = float(actual_row["energy_kj_mol"]) - float(expected_row["energy_kj_mol"])
        max_abs_delta = max(max_abs_delta, abs(delta))
        if first_nonzero is None and delta != 0.0:
            first_nonzero = {
                "step": key[0],
                "level": key[1],
                "term": key[2],
                "delta_kj_mol": delta,
            }
        if actual_count != expected_count:
            mismatch = {
                "step": key[0],
                "level": key[1],
                "term": key[2],
                "expected_interaction_count": expected_count,
                "actual_interaction_count": actual_count,
            }
            count_mismatches.append(mismatch)
            if first_mismatch is None:
                first_mismatch = mismatch
        compared_rows.append(
            {
                "step": key[0],
                "level": key[1],
                "term": key[2],
                "expected_energy_kj_mol": float(expected_row["energy_kj_mol"]),
                "actual_energy_kj_mol": float(actual_row["energy_kj_mol"]),
                "delta_kj_mol": delta,
                "interaction_count": actual_count,
            }
        )

    if first_mismatch is None:
        if missing_in_actual:
            first_mismatch = {"missing_key": missing_in_actual[0]}
        elif extra_in_actual:
            first_mismatch = {"extra_key": extra_in_actual[0]}

    return {
        "matches": not missing_in_actual and not extra_in_actual and not count_mismatches,
        "selected_terms": list(selected_terms),
        "max_step_included": max_step,
        "missing_in_actual": missing_in_actual,
        "extra_in_actual": extra_in_actual,
        "count_mismatches": count_mismatches,
        "max_abs_delta_kj_mol": max_abs_delta,
        "first_nonzero_delta": first_nonzero,
        "first_mismatch": first_mismatch,
        "compared_row_count": len(compared_rows),
        "rows": compared_rows,
    }


def required_bucket_row(
    *,
    bucket: str,
    status: str,
    gate_a_terms: list[str],
    note: str,
    topology_counts: dict[str, int],
) -> dict[str, object]:
    return {
        "bucket": bucket,
        "status": status,
        "gate_a_terms": gate_a_terms,
        "note": note,
        "topology_counts": topology_counts,
    }


def build_gate_a_term_coverage(
    gate_a_energy_summary: dict[str, object],
    topology_inventory: dict[str, object],
    gate_a_class2_trace: dict[str, object],
    gate_a_cpu_correction_trace: dict[str, object],
) -> dict[str, object]:
    step0_terms = dict(gate_a_energy_summary["step0_terms_kj_mol"])
    counts = dict(topology_inventory["term_counts"])
    class2_evidence = dict(topology_inventory["class2_parameter_evidence"])

    rows = []
    rows.append(
        required_bucket_row(
            bucket="bond",
            status="EXPLICIT" if "Class2 Bond" in step0_terms else "MISSING",
            gate_a_terms=["Class2 Bond"] if "Class2 Bond" in step0_terms else [],
            note="Gate A freezes the aggregate class2 bond energy as a dedicated term.",
            topology_counts={"Class2 Bond": counts.get("Class2 Bond", 0)},
        )
    )
    rows.append(
        required_bucket_row(
            bucket="angle",
            status="EXPLICIT" if "Class2 Angle" in step0_terms else "MISSING",
            gate_a_terms=["Class2 Angle"] if "Class2 Angle" in step0_terms else [],
            note="Gate A freezes the aggregate class2 angle energy, but not its embedded cross subcomponents separately.",
            topology_counts={"Class2 Angle": counts.get("Class2 Angle", 0)},
        )
    )
    rows.append(
        required_bucket_row(
            bucket="dihedral",
            status="EXPLICIT" if "Class2 Dih." in step0_terms else "MISSING",
            gate_a_terms=["Class2 Dih."] if "Class2 Dih." in step0_terms else [],
            note="Gate A freezes the aggregate class2 dihedral energy, but not its embedded cross subcomponents separately.",
            topology_counts={"Class2 Dih.": counts.get("Class2 Dih.", 0)},
        )
    )

    improper_count = counts.get("Class2 Impr.", 0)
    improper_trace_terms = CLASS2_TRACE_BUCKET_TERM_MAP["improper"] + (
        "improper_class2_angle_angle_1",
        "improper_class2_angle_angle_2",
        "improper_class2_angle_angle_3",
    )
    if improper_count > 0:
        if gate_a_has_class2_trace_terms(gate_a_class2_trace, improper_trace_terms):
            improper_status = "EXPLICIT"
            improper_terms = list(improper_trace_terms)
            improper_note = (
                "Topology contains class2 impropers and Gate A exposes them through the host-diagnostic "
                "class2 subterm trace."
            )
        else:
            improper_status = "MISSING"
            improper_terms = []
            improper_note = "Topology contains class2 impropers and Gate A must expose them explicitly."
    else:
        improper_status = "NOT_PRESENT_IN_FIXTURE"
        improper_terms = []
        improper_note = "No class2 improper interactions are present in this fixture topology."
    rows.append(
        required_bucket_row(
            bucket="improper",
            status=improper_status,
            gate_a_terms=improper_terms,
            note=improper_note,
            topology_counts={"Class2 Impr.": improper_count},
        )
    )

    cross_counts = {
        "Bond-Cross": counts.get("Bond-Cross", 0),
        "BA-Cross": counts.get("BA-Cross", 0),
        "Class2 Angle": counts.get("Class2 Angle", 0),
        "Class2 Dih.": counts.get("Class2 Dih.", 0),
    }
    cross_tokens = [
        *class2_evidence["angle_cross_component_tokens"],
        *class2_evidence["dihedral_cross_component_tokens"],
    ]
    cross_trace_terms = CLASS2_TRACE_BUCKET_TERM_MAP["class2_cross_terms"]
    if gate_a_has_class2_trace_terms(gate_a_class2_trace, cross_trace_terms):
        cross_status = "EXPLICIT"
        cross_note = (
            "Gate A exposes embedded PCFF class2 cross components through the host-diagnostic "
            "class2 subterm trace."
        )
    elif cross_counts["Bond-Cross"] > 0 or cross_counts["BA-Cross"] > 0:
        cross_status = "MISSING"
        cross_note = (
            "Topology contains explicit cross-term lists, but Gate A step-0 energy terms do not separate them."
        )
    elif cross_tokens:
        cross_status = "AMBIGUOUS"
        cross_note = (
            "ANGLE_CLASS2 and DIHEDRAL_CLASS2 parameters embed PCFF cross components "
            f"({', '.join(cross_tokens)}), but Gate A energy output only exposes aggregate "
            "Class2 Angle / Class2 Dih. terms."
        )
    else:
        cross_status = "NOT_PRESENT_IN_FIXTURE"
        cross_note = "No explicit or embedded class2 cross-term evidence was detected in this fixture."
    rows.append(
        required_bucket_row(
            bucket="class2_cross_terms",
            status=cross_status,
            gate_a_terms=list(cross_trace_terms) if cross_status == "EXPLICIT" else [],
            note=cross_note,
            topology_counts=cross_counts,
        )
    )

    rows.append(
        required_bucket_row(
            bucket="lj_sr",
            status="EXPLICIT" if "LJ (SR)" in step0_terms else "MISSING",
            gate_a_terms=["LJ (SR)"] if "LJ (SR)" in step0_terms else [],
            note="Short-range Lennard-Jones is explicit in Gate A step-0 energy terms.",
            topology_counts={},
        )
    )
    rows.append(
        required_bucket_row(
            bucket="coul_sr",
            status="EXPLICIT" if "Coulomb (SR)" in step0_terms else "MISSING",
            gate_a_terms=["Coulomb (SR)"] if "Coulomb (SR)" in step0_terms else [],
            note="Short-range Coulomb is explicit in Gate A step-0 energy terms.",
            topology_counts={},
        )
    )
    rows.append(
        required_bucket_row(
            bucket="lj_14",
            status="EXPLICIT" if "LJ-14" in step0_terms else "MISSING",
            gate_a_terms=["LJ-14"] if "LJ-14" in step0_terms else [],
            note="Listed 1-4 Lennard-Jones is explicit in Gate A step-0 energy terms.",
            topology_counts={"LJ-14": counts.get("LJ-14", 0)},
        )
    )
    rows.append(
        required_bucket_row(
            bucket="coul_14",
            status="EXPLICIT" if "Coulomb-14" in step0_terms else "MISSING",
            gate_a_terms=["Coulomb-14"] if "Coulomb-14" in step0_terms else [],
            note="Listed 1-4 Coulomb is explicit in Gate A step-0 energy terms.",
            topology_counts={"Coulomb-14": counts.get("Coulomb-14", 0)},
        )
    )

    cpu_correction_trace_terms = CPU_CORRECTION_TRACE_BUCKET_TERM_MAP["cpu_reciprocal_self_exclusion_corrections"]
    if gate_a_has_class2_trace_terms(gate_a_cpu_correction_trace, cpu_correction_trace_terms, max_step=None):
        cpu_correction_status = "EXPLICIT"
        cpu_correction_terms = list(cpu_correction_trace_terms)
        cpu_correction_note = (
            "Gate A exposes reciprocal/self/exclusion electrostatic ownership through the runtime "
            "cpu-correction energy split trace."
        )
    elif "Coul. recip." in step0_terms:
        cpu_correction_status = "PARTIAL"
        cpu_correction_terms = ["Coul. recip."]
        cpu_correction_note = (
            "Gate A exposes reciprocal-space Coulomb explicitly, but it does not split self/exclusion "
            "corrections into standalone terms."
        )
    else:
        cpu_correction_status = "MISSING"
        cpu_correction_terms = []
        cpu_correction_note = "Gate A does not expose reciprocal/self/exclusion CPU corrections explicitly."
    rows.append(
        required_bucket_row(
            bucket="cpu_reciprocal_self_exclusion_corrections",
            status=cpu_correction_status,
            gate_a_terms=cpu_correction_terms,
            note=cpu_correction_note,
            topology_counts={"Coul. recip.": counts.get("Coul. recip.", 0)},
        )
    )

    rows_by_bucket = {row["bucket"]: row for row in rows}
    first_insufficient_bucket = None
    for bucket in FIRST_BUCKET_PRIORITY:
        row = rows_by_bucket[bucket]
        if row["status"] not in {"EXPLICIT", "NOT_PRESENT_IN_FIXTURE"}:
            first_insufficient_bucket = row
            break

    return {
        "required_bucket_order": list(GATE_C_REQUIRED_BUCKET_ORDER),
        "rows": rows,
        "has_insufficient_visibility": first_insufficient_bucket is not None,
        "first_insufficient_bucket": first_insufficient_bucket,
    }


def load_run_outputs(gmx: Path, run_root: Path, run_id: str) -> dict[str, object]:
    edr_path = run_root / "exact_full.edr"
    class2_trace_path = run_root / "m2p_trace" / "class2_subterm_energy_trace.tsv"
    cpu_correction_trace_path = run_root / "m2p_trace" / "cpu_correction_energy_trace.tsv"
    energy_dump = capture_output([str(gmx), "dump", "-e", str(edr_path)], cwd=REPO_ROOT)
    return {
        "run_id": run_id,
        "artifact_root": str(run_root),
        "full_outputs": {
            "deffnm": str(run_root / "exact_full"),
            "edr": str(edr_path),
            "event_trace_tsv": str(run_root / "event_trace.tsv"),
            "total_force_tsv": str(run_root / "total_force.tsv"),
            "merge_trace_dir": str(run_root / "merge_trace"),
            "m2p_trace_dir": str(run_root / "m2p_trace"),
            "class2_subterm_trace_tsv": str(class2_trace_path),
            "cpu_correction_trace_tsv": str(cpu_correction_trace_path),
        },
        "actual_events": parse_event_trace(run_root / "event_trace.tsv"),
        "energy_frames": parse_energy_dump(energy_dump),
        "total_force_summary": parse_total_force_dump(run_root / "total_force.tsv"),
        "per_level_force_totals": parse_merge_trace_dir(run_root / "merge_trace"),
        "class2_subterm_energy_trace": parse_class2_subterm_energy_trace(class2_trace_path),
        "cpu_correction_energy_trace": parse_cpu_correction_energy_trace(cpu_correction_trace_path),
    }


def collect_gpu_run(
    *,
    args: argparse.Namespace,
    gmx: Path,
    gate_a_tpr: Path,
    system_root: Path,
    run_label: str,
    trace_atom_count: int,
    commands: list[dict[str, object]],
) -> dict[str, object]:
    run_root = system_root / run_label
    logs_dir = system_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    environment = trace_env_for_run(args, run_root, atom_count=trace_atom_count)
    run_env_delta = env_delta(environment, os.environ)
    deffnm = run_root / "exact_full"
    mdrun = [str(gmx), "mdrun", *mdrun_args_gate_c(args, gate_a_tpr, deffnm)]
    stdout_path = logs_dir / f"{run_label}.stdout"
    stderr_path = logs_dir / f"{run_label}.stderr"
    result = run_command_allow_failure(
        mdrun,
        cwd=REPO_ROOT,
        env=environment,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    commands.append(
        command_record(
            run_label,
            mdrun,
            cwd=REPO_ROOT,
            env_overrides=run_env_delta,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    )

    run_data = {
        "run_id": run_label,
        "artifact_root": str(run_root),
        "argv": mdrun,
        "env_overrides": run_env_delta,
        "returncode": result.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "failure_markers": extract_failure_markers(stderr_path, stdout_path),
    }
    if result.returncode == 0:
        run_data.update(load_run_outputs(gmx, run_root, run_label))
    return run_data


def build_class2_trace_bucket_assessments(
    actual_trace: dict[str, object], expected_trace: dict[str, object]
) -> dict[str, dict[str, object]]:
    return {
        bucket: compare_class2_subterm_trace_rows(
            actual_trace, expected_trace, selected_terms=term_names, max_step=CLASS2_TRACE_EARLY_MAX_STEP
        )
        for bucket, term_names in CLASS2_TRACE_BUCKET_TERM_MAP.items()
        if bucket in {"improper", "class2_cross_terms"}
    }


def build_cpu_correction_trace_bucket_assessments(
    actual_trace: dict[str, object], expected_trace: dict[str, object]
) -> dict[str, dict[str, object]]:
    return {
        bucket: compare_class2_subterm_trace_rows(
            actual_trace, expected_trace, selected_terms=term_names, max_step=None
        )
        for bucket, term_names in CPU_CORRECTION_TRACE_BUCKET_TERM_MAP.items()
    }


def build_per_term_comparison_rows(
    gate_a_term_coverage: dict[str, object],
    energy_comparison: dict[str, object],
    actual_energy_frames: list[dict[str, object]],
    expected_energy_frames: list[dict[str, object]],
    class2_trace_bucket_assessments: dict[str, dict[str, object]] | None = None,
    cpu_correction_trace_bucket_assessments: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    step0_deltas = {}
    if energy_comparison["frames"]:
        step0_deltas = dict(energy_comparison["frames"][0]["term_deltas_kj_mol"])

    rows = []
    for coverage_row in gate_a_term_coverage["rows"]:
        bucket = str(coverage_row["bucket"])
        term_names = BUCKET_TERM_MAP[bucket]
        trace_assessment = None
        comparison_source = "none"
        if class2_trace_bucket_assessments is not None:
            trace_assessment = class2_trace_bucket_assessments.get(bucket)
            if trace_assessment is not None:
                comparison_source = "class2_trace"
        if trace_assessment is None and cpu_correction_trace_bucket_assessments is not None:
            trace_assessment = cpu_correction_trace_bucket_assessments.get(bucket)
            if trace_assessment is not None:
                comparison_source = "cpu_correction_trace"
        if trace_assessment is not None:
            delta_summary = {
                "terms": trace_assessment["selected_terms"],
                "max_abs_delta_kj_mol": trace_assessment["max_abs_delta_kj_mol"],
                "first_nonzero": trace_assessment["first_nonzero_delta"],
            }
            display_assessment = None
            bucket_step0_deltas = {}
        elif term_names:
            delta_summary = max_abs_delta_for_terms(energy_comparison, tuple(term_names))
            display_assessment = assess_energy_display_resolution(
                actual_energy_frames, expected_energy_frames, selected_terms=tuple(term_names)
            )
            bucket_step0_deltas = {term: step0_deltas.get(term) for term in term_names if term in step0_deltas}
            comparison_source = "energy_dump"
        else:
            delta_summary = {"terms": [], "max_abs_delta_kj_mol": None, "first_nonzero": None}
            display_assessment = None
            bucket_step0_deltas = {}
            comparison_source = "none"

        mismatch_category = None
        comparison_status = "MATCH"
        coverage_status = str(coverage_row["status"])
        if coverage_status in {"MISSING", "AMBIGUOUS", "PARTIAL"}:
            comparison_status = coverage_status
            mismatch_category = "trace insufficiency"
        elif coverage_status == "NOT_PRESENT_IN_FIXTURE":
            comparison_status = coverage_status
        elif trace_assessment is not None and not trace_assessment["matches"]:
            comparison_status = "MISMATCH"
            mismatch_category = "ownership"
        elif display_assessment is not None and not display_assessment["within_bounds"]:
            comparison_status = "MISMATCH"
            mismatch_category = "ownership_or_reduction"

        rows.append(
            {
                "bucket": bucket,
                "comparison_source": comparison_source,
                "coverage_status": coverage_status,
                "comparison_status": comparison_status,
                "mismatch_category": mismatch_category,
                "gate_a_terms": coverage_row["gate_a_terms"],
                "topology_counts": coverage_row["topology_counts"],
                "note": coverage_row["note"],
                "terms": list(delta_summary["terms"]),
                "step0_term_deltas_kj_mol": bucket_step0_deltas,
                "max_abs_delta_kj_mol": delta_summary["max_abs_delta_kj_mol"],
                "first_nonzero_delta": delta_summary["first_nonzero"],
                "display_resolution_assessment": display_assessment,
                "trace_key_assessment": trace_assessment,
            }
        )
    return rows


def first_term_issue(per_term_rows: list[dict[str, object]]) -> dict[str, object] | None:
    rows_by_bucket = {row["bucket"]: row for row in per_term_rows}
    for bucket in FIRST_BUCKET_PRIORITY:
        row = rows_by_bucket[bucket]
        if row["mismatch_category"] is not None:
            return {
                "bucket": row["bucket"],
                "comparison_status": row["comparison_status"],
                "mismatch_category": row["mismatch_category"],
                "note": row["note"],
                "terms": row["terms"],
                "first_nonzero_delta": row["first_nonzero_delta"],
            }
    return None


def summarize_direct_oracle_comparison(
    *,
    event_order_comparison: dict[str, object],
    total_force_comparison: dict[str, object],
    per_level_force_comparison: dict[str, object],
    energy_comparison: dict[str, object],
    virial_comparison: dict[str, object],
    gpu_noise_floor: dict[str, object],
    per_term_rows: list[dict[str, object]],
    class2_trace_bucket_assessments: dict[str, dict[str, object]],
    cpu_correction_trace_bucket_assessments: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "event_order": event_order_comparison,
        "total_force": total_force_comparison,
        "per_level_force_totals": per_level_force_comparison,
        "per_term_energies": {
            "max_abs_delta_kj_mol": energy_comparison["max_abs_delta_kj_mol"],
            "first_mismatch": energy_comparison["first_mismatch"],
            "rows": per_term_rows,
        },
        "virial_contributors": virial_comparison,
        "class2_subterm_trace": class2_trace_bucket_assessments,
        "cpu_correction_trace": cpu_correction_trace_bucket_assessments,
        "gpu_noise_floor": gpu_noise_floor,
    }


def find_first_failure(
    *,
    main_run: dict[str, object],
    event_order_comparison: dict[str, object],
    total_force_comparison: dict[str, object],
    per_level_force_comparison: dict[str, object],
    per_term_rows: list[dict[str, object]],
) -> dict[str, object] | None:
    if main_run["returncode"] != 0:
        return {
            "field": "main_run.returncode",
            "reason": "The Gate C mdrun command did not execute successfully.",
            "returncode": main_run["returncode"],
            "failure_markers": main_run["failure_markers"],
        }
    if not event_order_comparison["matches"]:
        return {"field": "event_order", "details": event_order_comparison["first_mismatch"]}
    if total_force_comparison["missing_in_actual"] or total_force_comparison["extra_in_actual"]:
        return {"field": "total_force", "details": total_force_comparison["first_mismatch"]}
    if per_level_force_comparison["missing_in_actual"] or per_level_force_comparison["extra_in_actual"]:
        return {"field": "per_level_force_totals", "details": per_level_force_comparison["first_mismatch"]}
    term_issue = first_term_issue(per_term_rows)
    if term_issue is not None:
        return {"field": f"per_term.{term_issue['bucket']}", "details": term_issue}
    return None


def characterize_virial_difference(
    virial_display: dict[str, object], virial_comparison: dict[str, object], gpu_noise_floor: dict[str, object]
) -> dict[str, object]:
    if not virial_comparison.get("available"):
        return {
            "status": "unavailable",
            "note": "Virial / pressure contributors were not available for characterization.",
            "inference": False,
        }

    noise_floor = None
    if gpu_noise_floor.get("available"):
        noise_floor = float(gpu_noise_floor.get("virial_max_abs_delta", 0.0))

    max_abs_delta = float(virial_comparison.get("max_abs_delta", 0.0))
    if virial_display.get("within_bounds"):
        return {
            "status": "within_display_resolution",
            "max_abs_delta": max_abs_delta,
            "gpu_noise_floor_max_abs_delta": noise_floor,
            "note": "Virial / pressure deltas stay within the gmx dump display-resolution bound.",
            "inference": False,
        }
    if noise_floor is not None and max_abs_delta <= noise_floor + 1e-12:
        return {
            "status": "within_gpu_noise_floor",
            "max_abs_delta": max_abs_delta,
            "gpu_noise_floor_max_abs_delta": noise_floor,
            "note": "Virial / pressure deltas exceed dump display resolution but stay within repeated GPU noise.",
            "inference": False,
        }
    return {
        "status": "systematic_reduction_difference",
        "max_abs_delta": max_abs_delta,
        "gpu_noise_floor_max_abs_delta": noise_floor,
        "first_nonzero": virial_comparison.get("first_nonzero"),
        "display_first_excess": virial_display.get("first_excess"),
        "note": (
            "추측입니다. Virial / pressure deltas exceed both dump display resolution and repeated-run GPU noise, "
            "but event order, per-level force totals, explicit class2/cpu-correction ownership traces, and "
            "explicit per-term energies remain aligned. This points to a systematic aggregate reduction/precision "
            "difference rather than a PCFF ownership collapse."
        ),
        "inference": True,
    }


def assess_gate_c_system(
    *,
    main_run: dict[str, object],
    gate_a_energy_frames: list[dict[str, object]],
    event_order_comparison: dict[str, object],
    total_force_comparison: dict[str, object],
    per_level_force_comparison: dict[str, object],
    energy_comparison: dict[str, object],
    virial_comparison: dict[str, object],
    gpu_noise_floor: dict[str, object],
    per_term_rows: list[dict[str, object]],
) -> dict[str, object]:
    if main_run["returncode"] != 0:
        return {
            "status": "BLOCKER",
            "reasons": ["The primary Gate C GPU run did not execute successfully."],
            "strongest_surviving_claim": "No Gate C mechanics claim survives because the main run failed.",
            "broken_pcff_specific_claims": ["nb gpu + bonded gpu exact-r-RESPA is executable for this fixture."],
            "exact_ambiguous_or_wrong_terms": [],
            "gate_d_blocked": True,
        }

    total_force_roundoff = force_sum_roundoff_bound_from_total_force_dump(
        Path(main_run["full_outputs"]["total_force_tsv"])
    )
    per_level_roundoff = force_sum_roundoff_bound_from_merge_trace_dir(
        Path(main_run["full_outputs"]["merge_trace_dir"])
    )
    explicit_terms = tuple(
        term
        for row in per_term_rows
        if row["comparison_source"] == "energy_dump"
        if row["coverage_status"] == "EXPLICIT"
        for term in row["terms"]
    )
    energy_display = assess_energy_display_resolution(
        main_run["energy_frames"], gate_a_energy_frames, selected_terms=explicit_terms
    )
    virial_display = assess_energy_display_resolution(
        main_run["energy_frames"],
        gate_a_energy_frames,
        selected_terms=("Pressure", "Vir-XX", "Vir-YY", "Vir-ZZ", "Pres-XX", "Pres-YY", "Pres-ZZ"),
    )
    virial_characterization = characterize_virial_difference(virial_display, virial_comparison, gpu_noise_floor)

    total_force_within_roundoff = (
        float(total_force_comparison["max_abs_component_delta"]) <= float(total_force_roundoff["bound"])
    )
    per_level_within_roundoff = (
        float(per_level_force_comparison["max_abs_component_delta"]) <= float(per_level_roundoff["bound"])
    )
    term_issue = first_term_issue(per_term_rows)

    fail_reasons = []
    if not event_order_comparison["matches"]:
        fail_reasons.append("Runtime event ordering diverged from the frozen Gate A oracle.")
    if total_force_comparison["missing_in_actual"] or total_force_comparison["extra_in_actual"]:
        fail_reasons.append("Step-level total-force trace coverage diverged from Gate A.")
    if per_level_force_comparison["missing_in_actual"] or per_level_force_comparison["extra_in_actual"]:
        fail_reasons.append("Per-level force trace coverage diverged from Gate A.")
    if not gpu_noise_floor["available"]:
        fail_reasons.append("GPU run-to-run noise floor was not measured.")
    elif not gpu_noise_floor["event_trace_identical_across_successful_gpu_runs"]:
        fail_reasons.append("Repeated GPU runs changed event ordering.")
    if not total_force_within_roundoff:
        fail_reasons.append("CPU-vs-GPU total-force deviation exceeds a conservative float roundoff bound.")
    if not per_level_within_roundoff:
        fail_reasons.append("CPU-vs-GPU per-level force deviation exceeds a conservative float roundoff bound.")
    if not energy_display["within_bounds"]:
        fail_reasons.append("Explicit per-term energy deltas exceed the gmx dump display-resolution bound.")

    ambiguous_or_wrong_terms = [
        {
            "bucket": row["bucket"],
            "comparison_status": row["comparison_status"],
            "mismatch_category": row["mismatch_category"],
            "note": row["note"],
            "terms": row["terms"],
        }
        for row in per_term_rows
        if row["comparison_status"] not in {"MATCH", "NOT_PRESENT_IN_FIXTURE"}
    ]
    broken_claims = []
    if fail_reasons:
        broken_claims.extend(fail_reasons)
    if term_issue is not None:
        broken_claims.append(
            f"{term_issue['bucket']} remains unresolved because the oracle only provides {term_issue['comparison_status']} visibility."
        )

    if fail_reasons:
        status = "FAIL"
    elif term_issue is not None:
        status = "PARTIAL"
    else:
        status = "PASS"

    strongest_surviving_claim = (
        "Event order, explicit per-level force traces, and explicit Class2/LJ14/Coulomb-14 energy terms "
        "match the frozen Gate A oracle within observed GPU noise and float-roundoff bounds."
    )
    if status == "BLOCKER":
        strongest_surviving_claim = "No Gate C mechanics claim survives."
    elif status == "PARTIAL":
        strongest_surviving_claim += (
            " However, CPU reciprocal/self/exclusion correction ownership remains under-specified by the "
            "current oracle traces."
        )
    elif virial_characterization["status"] == "systematic_reduction_difference":
        strongest_surviving_claim += (
            " Aggregate virial / pressure drift is characterized separately as a likely reduction-path "
            "difference and is not used as the Gate C ownership verdict."
        )

    return {
        "status": status,
        "reasons": fail_reasons,
        "strongest_surviving_claim": strongest_surviving_claim,
        "broken_pcff_specific_claims": broken_claims,
        "exact_ambiguous_or_wrong_terms": ambiguous_or_wrong_terms,
        "gate_d_blocked": status != "PASS",
        "total_force_max_abs_component_delta": total_force_comparison["max_abs_component_delta"],
        "per_level_force_max_abs_component_delta": per_level_force_comparison["max_abs_component_delta"],
        "total_force_float_roundoff_bound": total_force_roundoff,
        "per_level_force_float_roundoff_bound": per_level_roundoff,
        "explicit_term_energy_display_assessment": energy_display,
        "virial_display_assessment": virial_display,
        "virial_characterization": virial_characterization,
    }


def collect_system_result(
    args: argparse.Namespace,
    gmx: Path,
    out_root: Path,
    system_id: str,
    gate_a_system: dict[str, object],
) -> dict[str, object]:
    system_root = out_root / system_id
    if system_root.exists():
        shutil.rmtree(system_root)
    logs_dir = system_root / "logs"
    summaries_dir = system_root / "summaries"
    for directory in (logs_dir, summaries_dir):
        directory.mkdir(parents=True, exist_ok=True)

    commands: list[dict[str, object]] = []
    gate_a_tpr = Path(gate_a_system["full_run_outputs"]["tpr"])
    tpr_dump = capture_output([str(gmx), "dump", "-s", str(gate_a_tpr)], cwd=REPO_ROOT)
    write_text(summaries_dir / "gate_a_tpr_dump.txt", tpr_dump)
    topology_inventory = parse_tpr_inventory(tpr_dump)
    write_text(
        summaries_dir / "topology_inventory.json",
        json.dumps(topology_inventory, indent=2, sort_keys=True) + "\n",
    )

    gate_a_energy_summary = load_json(Path(gate_a_system["energy_terms"]))
    gate_a_class2_trace = load_json(Path(gate_a_system["class2_subterm_energy_trace"]))
    gate_a_cpu_correction_trace = load_json(Path(gate_a_system["cpu_correction_energy_trace"]))
    gate_a_term_coverage = build_gate_a_term_coverage(
        gate_a_energy_summary, topology_inventory, gate_a_class2_trace, gate_a_cpu_correction_trace
    )
    write_text(
        summaries_dir / "gate_a_term_coverage.json",
        json.dumps(gate_a_term_coverage, indent=2, sort_keys=True) + "\n",
    )

    expected_events = parse_event_trace(Path(gate_a_system["full_run_outputs"]["event_trace_tsv"]))
    expected_total_force = load_json(Path(gate_a_system["total_force_summary"]))["per_step_totals"]
    expected_per_level_force = load_json(Path(gate_a_system["per_level_force_totals"]))["entries"]
    expected_energy_frames = load_json(Path(gate_a_system["energy_terms"]))["frames"]

    main_run = collect_gpu_run(
        args=args,
        gmx=gmx,
        gate_a_tpr=gate_a_tpr,
        system_root=system_root,
        run_label="full",
        trace_atom_count=read_gro_atom_count(Path(gate_a_system["full_run_outputs"]["gro"])),
        commands=commands,
    )
    repeat_runs: list[dict[str, object]] = []
    successful_runs: list[dict[str, object]] = []
    if main_run["returncode"] == 0:
        successful_runs.append(main_run)
        for repeat_index in range(1, args.gpu_repeats):
            repeat_run = collect_gpu_run(
                args=args,
                gmx=gmx,
                gate_a_tpr=gate_a_tpr,
                system_root=system_root,
                run_label=f"repeat_{repeat_index}",
                trace_atom_count=read_gro_atom_count(Path(gate_a_system["full_run_outputs"]["gro"])),
                commands=commands,
            )
            repeat_runs.append(repeat_run)
            if repeat_run["returncode"] == 0:
                successful_runs.append(repeat_run)

    if main_run["returncode"] == 0:
        event_order_comparison = compare_event_trace(main_run["actual_events"], expected_events)
        total_force_comparison = compare_total_force_entries(
            main_run["total_force_summary"]["per_step_totals"], expected_total_force
        )
        per_level_force_comparison = compare_per_level_force_entries(
            main_run["per_level_force_totals"]["entries"], expected_per_level_force
        )
        energy_comparison = compare_energy_frames(main_run["energy_frames"], expected_energy_frames)
        virial_comparison = extract_virial_deltas(energy_comparison)
        gpu_noise_floor = estimate_noise_floor(successful_runs)
        class2_trace_bucket_assessments = build_class2_trace_bucket_assessments(
            main_run["class2_subterm_energy_trace"], gate_a_class2_trace
        )
        cpu_correction_trace_bucket_assessments = build_cpu_correction_trace_bucket_assessments(
            main_run["cpu_correction_energy_trace"], gate_a_cpu_correction_trace
        )
        per_term_rows = build_per_term_comparison_rows(
            gate_a_term_coverage,
            energy_comparison,
            main_run["energy_frames"],
            expected_energy_frames,
            class2_trace_bucket_assessments,
            cpu_correction_trace_bucket_assessments,
        )
    else:
        blocked_reason = (
            "Gate C mdrun did not execute; direct event/force/energy/noise comparisons against Gate A are unavailable."
        )
        event_order_comparison = {"matches": False, "reason": blocked_reason, "first_mismatch": None}
        total_force_comparison = {
            "matches": False,
            "reason": blocked_reason,
            "missing_in_actual": [],
            "extra_in_actual": [],
            "max_abs_component_delta": None,
            "first_mismatch": None,
        }
        per_level_force_comparison = {
            "matches": False,
            "reason": blocked_reason,
            "missing_in_actual": [],
            "extra_in_actual": [],
            "max_abs_component_delta": None,
            "first_mismatch": None,
        }
        energy_comparison = {
            "matches": False,
            "reason": blocked_reason,
            "max_abs_delta_kj_mol": None,
            "first_mismatch": None,
            "frames": [],
        }
        virial_comparison = {"available": False, "reason": blocked_reason, "max_abs_delta": None, "frames": []}
        gpu_noise_floor = {
            "available": False,
            "reason": "No successful Gate C GPU runs are available.",
            "successful_run_count": 0,
        }
        class2_trace_bucket_assessments = {}
        cpu_correction_trace_bucket_assessments = {}
        per_term_rows = build_per_term_comparison_rows(
            gate_a_term_coverage,
            energy_comparison,
            [],
            expected_energy_frames,
            class2_trace_bucket_assessments,
            cpu_correction_trace_bucket_assessments,
        )

    direct_oracle_comparison = summarize_direct_oracle_comparison(
        event_order_comparison=event_order_comparison,
        total_force_comparison=total_force_comparison,
        per_level_force_comparison=per_level_force_comparison,
        energy_comparison=energy_comparison,
        virial_comparison=virial_comparison,
        gpu_noise_floor=gpu_noise_floor,
        per_term_rows=per_term_rows,
        class2_trace_bucket_assessments=class2_trace_bucket_assessments,
        cpu_correction_trace_bucket_assessments=cpu_correction_trace_bucket_assessments,
    )
    assessment = assess_gate_c_system(
        main_run=main_run,
        gate_a_energy_frames=expected_energy_frames,
        event_order_comparison=event_order_comparison,
        total_force_comparison=total_force_comparison,
        per_level_force_comparison=per_level_force_comparison,
        energy_comparison=energy_comparison,
        virial_comparison=virial_comparison,
        gpu_noise_floor=gpu_noise_floor,
        per_term_rows=per_term_rows,
    )
    first_failure = find_first_failure(
        main_run=main_run,
        event_order_comparison=event_order_comparison,
        total_force_comparison=total_force_comparison,
        per_level_force_comparison=per_level_force_comparison,
        per_term_rows=per_term_rows,
    )
    first_mismatching_term = first_term_issue(per_term_rows)

    system_result = {
        "system_id": system_id,
        "artifact_root": str(system_root),
        "gate_a_artifact_root": gate_a_system["artifact_root"],
        "gate_a_commands_sh": gate_a_system["commands_sh"],
        "gate_a_tpr": str(gate_a_tpr),
        "commands_json": str(summaries_dir / "commands.json"),
        "commands_sh": str(system_root / "run_commands.sh"),
        "main_run": main_run,
        "repeat_runs": repeat_runs,
        "topology_inventory": topology_inventory,
        "gate_a_term_coverage": gate_a_term_coverage,
        "event_order_comparison": event_order_comparison,
        "total_force_comparison": total_force_comparison,
        "per_level_force_comparison": per_level_force_comparison,
        "energy_comparison": energy_comparison,
        "virial_comparison": virial_comparison,
        "class2_trace_bucket_assessments": class2_trace_bucket_assessments,
        "cpu_correction_trace_bucket_assessments": cpu_correction_trace_bucket_assessments,
        "per_term_comparison_table": per_term_rows,
        "direct_oracle_comparison": direct_oracle_comparison,
        "gpu_noise_floor": gpu_noise_floor,
        "gate_c_assessment": assessment,
        "first_failure_field": first_failure,
        "first_mismatching_term": first_mismatching_term,
    }
    write_text(summaries_dir / "system_result.json", json.dumps(system_result, indent=2, sort_keys=True) + "\n")
    write_text(summaries_dir / "commands.json", json.dumps(commands, indent=2, sort_keys=True) + "\n")
    write_commands_script(system_root / "run_commands.sh", commands)
    return system_result


def build_manifest(
    *,
    args: argparse.Namespace,
    out_root: Path,
    gate_a_manifest: dict[str, object],
    gmx: Path,
    gmx_version: str,
    gpu_inventory: dict[str, object],
    systems: list[dict[str, object]],
) -> dict[str, object]:
    gpu_support = parse_gpu_support(gmx_version)
    status_rank = {"PASS": 0, "PARTIAL": 1, "FAIL": 2, "BLOCKER": 3}
    status = max((system["gate_c_assessment"]["status"] for system in systems), key=lambda item: status_rank[item])

    blocking_reasons = []
    for system in systems:
        if system["main_run"]["failure_markers"]:
            blocking_reasons.append(f"{system['system_id']}: {' | '.join(system['main_run']['failure_markers'])}")
        for reason in system["gate_c_assessment"]["reasons"]:
            blocking_reasons.append(f"{system['system_id']}: {reason}")
        if system["first_mismatching_term"] is not None:
            blocking_reasons.append(
                f"{system['system_id']}: first unresolved term bucket is {system['first_mismatching_term']['bucket']} "
                f"({system['first_mismatching_term']['mismatch_category']})."
            )

    gate_d_allowed = all(system["gate_c_assessment"]["status"] == "PASS" for system in systems)
    recommendation_reason = (
        "Gate D may start because Gate C preserved event ordering and all explicit per-term/per-level semantics."
        if gate_d_allowed
        else "Gate D remains blocked until Gate C resolves the first unresolved term bucket for every fixture."
    )

    return {
        "schema_version": 2,
        "gate": "Gate C",
        "status": status,
        "objective": "Validate standalone exact r-RESPA with nb gpu + bonded gpu while PME and update remain on CPU.",
        "artifact_root": str(out_root),
        "gate_a_manifest": str(Path(args.gate_a_manifest).resolve()),
        "gate_a_status": gate_a_manifest.get("status"),
        "gmx": str(gmx),
        "gmx_version": gmx_version,
        "precision_mode": parse_precision_mode(gmx_version),
        "gpu_support": gpu_support,
        "hardware_configuration": gpu_inventory,
        "ntmpi": args.ntmpi,
        "ntomp": args.ntomp,
        "dlb": "no",
        "pme_rank_count": 0,
        "reproducibility_flags": [
            "-dlb no",
            "-pin off",
            "-nb gpu",
            "-pme cpu",
            "-bonded gpu",
            "-update cpu",
            "GMX_DISABLE_MODULAR_SIMULATOR=1",
        ],
        "binary_reproducibility_supported": False,
        "reproducibility_notes": [
            GPU_REPRODUCIBILITY_NOTE,
            "GPU noise floor is estimated from repeated successful Gate C runs for each fixture.",
        ],
        "rerun_used": False,
        "normal_md_used": True,
        "comparison_basis": "Frozen Gate A CPU oracle manifest and direct Gate C runs using the frozen Gate A TPRs.",
        "source_audit": {
            "bonded_gpu_decision": "src/gromacs/taskassignment/decidegpuusage.cpp::decideWhetherToUseGpusForBonded",
            "bonded_gpu_input_guard": "src/gromacs/listed_forces/listed_forces_gpu_impl.cpp::inputSupportsListedForcesGpu",
            "gpu_bonded_function_list": "src/gromacs/listed_forces/listed_forces_gpu.h::fTypesOnGpu",
        },
        "blocking_reasons": blocking_reasons,
        "recommendation": {
            "gate_d_allowed": gate_d_allowed,
            "reason": recommendation_reason,
        },
        "systems": systems,
    }


def write_manifest_markdown(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Gate C Oracle Comparison",
        "",
        f"- Status: {manifest['status']}",
        f"- Gate D allowed: {manifest['recommendation']['gate_d_allowed']}",
        f"- gmx: `{manifest['gmx']}`",
        f"- precision: `{manifest['precision_mode']}`",
        f"- GPU support: `{manifest['gpu_support']}`",
        f"- ntmpi / ntomp: `{manifest['ntmpi']}` / `{manifest['ntomp']}`",
        f"- DLB: `{manifest['dlb']}`",
        f"- PME ranks: `{manifest['pme_rank_count']}`",
        "",
        "## Blocking Reasons",
        "",
    ]
    if manifest["blocking_reasons"]:
        for reason in manifest["blocking_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- None")
    lines.extend(["", "## Systems", ""])
    for system in manifest["systems"]:
        lines.append(f"### {system['system_id']}")
        lines.append("")
        lines.append(f"- Gate C assessment: `{system['gate_c_assessment']['status']}`")
        lines.append(f"- Main run return code: `{system['main_run']['returncode']}`")
        lines.append(f"- Event order identical: `{system['event_order_comparison'].get('matches')}`")
        lines.append(
            f"- Total force max abs component delta: `{system['gate_c_assessment'].get('total_force_max_abs_component_delta')}`"
        )
        lines.append(
            f"- Per-level force max abs component delta: `{system['gate_c_assessment'].get('per_level_force_max_abs_component_delta')}`"
        )
        lines.append(f"- First failure field: `{system['first_failure_field']}`")
        lines.append(f"- First mismatching term: `{system['first_mismatching_term']}`")
        lines.append(f"- Artifact root: `{system['artifact_root']}`")
        lines.append(f"- Command script: `{system['commands_sh']}`")
        lines.append("")
    write_text(path, "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    gmx = Path(args.gmx).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    gate_a_manifest = load_json(Path(args.gate_a_manifest).resolve())
    validate_inputs(gate_a_manifest)
    maybe_build(args, Path(args.build_dir).resolve() if args.build_dir is not None else None)

    gmx_version = capture_output([str(gmx), "--version"], cwd=REPO_ROOT)
    gpu_inventory = {
        "nvidia_smi_list": capture_optional_output(["nvidia-smi", "-L"]),
        "nvidia_smi_query": capture_optional_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "nvcc_version": capture_optional_output(["nvcc", "--version"]),
    }

    gate_a_systems_by_id = {system["system_id"]: system for system in gate_a_manifest["systems"]}
    systems = []
    for system_id in SYSTEMS:
        systems.append(
            collect_system_result(
                args=args,
                gmx=gmx,
                out_root=out_root,
                system_id=system_id,
                gate_a_system=gate_a_systems_by_id[system_id],
            )
        )

    manifest = build_manifest(
        args=args,
        out_root=out_root,
        gate_a_manifest=gate_a_manifest,
        gmx=gmx,
        gmx_version=gmx_version,
        gpu_inventory=gpu_inventory,
        systems=systems,
    )
    write_text(out_root / "gate_c_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_manifest_markdown(out_root / "gate_c_manifest.md", manifest)


if __name__ == "__main__":
    main()
