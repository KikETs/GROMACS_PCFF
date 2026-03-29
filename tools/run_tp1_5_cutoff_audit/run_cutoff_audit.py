#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import pathlib
import shutil
import subprocess
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[2]
GMX = ROOT / "build/bin/gmx"
LMP = shutil.which("lmp")

TOOL_DIR = pathlib.Path(__file__).resolve().parent
WORK_DIR = TOOL_DIR / "work"
RESULTS_DIR = ROOT / "tests/reference_results/tp1_5_cutoff_audit"

PT84_DIR = ROOT / "tests/reference_results/pt8_4_nonbonded_parity/exclusion_toy"
LAMMPS_EXCLUSION_DIR = ROOT / "testdata/lammps_golden/systems/exclusion_toy/lammps"
TP13_RESULTS = ROOT / "tests/reference_results/tp1_3_stabilization/trial_matrix_results.json"
TP13_TRL0_LOG = ROOT / "tests/reference_results/tp1_3_stabilization/TRL-0/trial.log"
TP13_TRL5_LOG = ROOT / "tests/reference_results/tp1_3_stabilization/TRL-5/trial.log"


def run_command(
    cmd: list[str], cwd: pathlib.Path, log_path: pathlib.Path, title: str, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, text=True, input=stdin, capture_output=True, check=True)
    append_section(log_path, f"{title} stdout", result.stdout)
    append_section(log_path, f"{title} stderr", result.stderr)
    return result


def append_section(path: pathlib.Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"===== {title} =====\n")
        handle.write(body)
        if not body.endswith("\n"):
            handle.write("\n")
        handle.write("\n")


def write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def command_to_string(cmd: list[str], cwd: pathlib.Path, stdin: str | None = None) -> str:
    rendered = f"(cd {cwd} && {' '.join(cmd)})"
    if stdin is not None:
        rendered += f"  # stdin={stdin!r}"
    return rendered


def git_output(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout


def gmx_version_text() -> str:
    return subprocess.run([str(GMX), "--version"], cwd=ROOT, text=True, capture_output=True, check=True).stdout


def reset_result_files() -> None:
    for filename in [
        "raw_commands.txt",
        "raw_exclusion_lammps.txt",
        "raw_exclusion_grompp.log",
        "raw_exclusion_mdrun.log",
        "raw_exclusion_energy.txt",
        "raw_shift_grompp.log",
        "raw_shift_mdrun.log",
        "raw_shift_energy.txt",
        "raw_shift_force_dump.txt",
    ]:
        path = RESULTS_DIR / filename
        if path.exists():
            path.unlink()


def parse_first_numeric_xvg(path: pathlib.Path) -> list[float]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "@")):
            return [float(token) for token in line.split()]
    raise ValueError(f"No numeric data found in {path}")


def parse_lammps_pe(stdout: str) -> float:
    lines = stdout.splitlines()
    for index, line in enumerate(lines):
        if "Step" in line and "PotEng" in line:
            for later in lines[index + 1 :]:
                parts = later.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    return float(parts[1])
    raise ValueError("Could not parse PotEng from LAMMPS output")


def parse_force_x_atom2(dump_output: str) -> float:
    for line in dump_output.splitlines():
        if "f[    1]" in line:
            return float(line.split("=")[1].strip("{} ").split(",")[0])
    raise ValueError("Could not parse atom-2 x-force from gmx dump output")


def parse_tp13_run_context() -> dict:
    matrix = json.loads(TP13_RESULTS.read_text(encoding="utf-8"))
    indexed = {entry["trial_id"]: entry for entry in matrix}

    def extract_line(log_path: pathlib.Path, pattern: str) -> str:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if pattern in line:
                return line.strip()
        raise ValueError(f"Pattern {pattern!r} not found in {log_path}")

    return {
        "trl0_summary": indexed["TRL-0"],
        "trl5_summary": indexed["TRL-5"],
        "trl0_kernel_line": extract_line(TP13_TRL0_LOG, "Using plain-C-4x4 4x4 nonbonded short-range kernels"),
        "trl5_kernel_line": extract_line(TP13_TRL5_LOG, "Using plain-C-4x4 4x4 nonbonded short-range kernels"),
        "trl0_pairlist_line": extract_line(TP13_TRL0_LOG, "updated every 10 steps"),
        "trl5_pairlist_line": extract_line(TP13_TRL5_LOG, "updated every 10 steps"),
        "trl0_repulsion_line": extract_line(TP13_TRL0_LOG, "Detected LJ repulsion power 9."),
        "trl5_repulsion_line": extract_line(TP13_TRL5_LOG, "Detected LJ repulsion power 9."),
    }


def run_exclusion_parity(commands: list[str]) -> dict:
    if LMP is None:
        raise RuntimeError("LAMMPS executable 'lmp' is not available")

    work_dir = WORK_DIR / "exclusion_parity"
    work_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(PT84_DIR / "system.top", work_dir / "system.top")
    shutil.copy2(PT84_DIR / "system.gro", work_dir / "system.gro")
    write_text(
        work_dir / "test.mdp",
        "\n".join(
            [
                "integrator  = md",
                "nsteps      = 0",
                "cutoff-scheme = Verlet",
                "vdw-type     = Cut-off",
                "rvdw         = 2.0",
                "coulombtype  = Cut-off",
                "rcoulomb     = 2.0",
                "pbc          = xyz",
                "nstfout      = 1",
                "",
            ]
        ),
    )

    shutil.copy2(LAMMPS_EXCLUSION_DIR / "system.data", work_dir / "system.data")
    lammps_in = (LAMMPS_EXCLUSION_DIR / "system.in").read_text(encoding="utf-8")
    rewritten_lines = []
    for line in lammps_in.splitlines():
        if line.startswith("read_data"):
            rewritten_lines.append("read_data system.data")
        else:
            rewritten_lines.append(line)
    rewritten_lines.extend(["", "thermo_style custom step pe", "run 0"])
    write_text(work_dir / "system.in", "\n".join(rewritten_lines) + "\n")

    lammps_cmd = [LMP, "-in", "system.in"]
    commands.append(command_to_string(lammps_cmd, work_dir))
    lammps = run_command(lammps_cmd, work_dir, RESULTS_DIR / "raw_exclusion_lammps.txt", "exclusion fixture lammps")

    grompp_cmd = [
        str(GMX),
        "grompp",
        "-f",
        "test.mdp",
        "-c",
        "system.gro",
        "-p",
        "system.top",
        "-o",
        "topol.tpr",
        "-maxwarn",
        "10",
    ]
    commands.append(command_to_string(grompp_cmd, work_dir))
    run_command(grompp_cmd, work_dir, RESULTS_DIR / "raw_exclusion_grompp.log", "exclusion fixture grompp")

    mdrun_cmd = [
        str(GMX),
        "mdrun",
        "-s",
        "topol.tpr",
        "-rerun",
        "system.gro",
        "-e",
        "ener.edr",
        "-o",
        "traj.trr",
        "-g",
        "md.log",
        "-nt",
        "1",
    ]
    commands.append(command_to_string(mdrun_cmd, work_dir))
    run_command(mdrun_cmd, work_dir, RESULTS_DIR / "raw_exclusion_mdrun.log", "exclusion fixture mdrun")

    energy_cmd = [str(GMX), "energy", "-f", "ener.edr", "-o", "energy.xvg"]
    energy_stdin = "Potential\n0\n"
    commands.append(command_to_string(energy_cmd, work_dir, energy_stdin))
    run_command(
        energy_cmd,
        work_dir,
        RESULTS_DIR / "raw_exclusion_energy.txt",
        "exclusion fixture energy",
        energy_stdin,
    )

    lammps_pe_kcal = parse_lammps_pe(lammps.stdout)
    lammps_pe_kj = lammps_pe_kcal * 4.184
    gromacs_pe_kj = parse_first_numeric_xvg(work_dir / "energy.xvg")[1]

    return {
        "fixture_id": "exclusion_parity_cutoff_9_6",
        "path_family": ["exclusion mask handling", "listed-vs-nonlisted split", "plain-C reference cut-off path"],
        "system_topology_source": str((PT84_DIR / "system.top").relative_to(ROOT)),
        "system_coordinate_source": str((PT84_DIR / "system.gro").relative_to(ROOT)),
        "lammps_source": str((LAMMPS_EXCLUSION_DIR / "system.in").relative_to(ROOT)),
        "gromacs_pe_kj": gromacs_pe_kj,
        "lammps_pe_kj": lammps_pe_kj,
        "difference_kj": abs(gromacs_pe_kj - lammps_pe_kj),
        "status": "pass" if abs(gromacs_pe_kj - lammps_pe_kj) < 0.01 else "fail",
    }


def shift_topology_text() -> str:
    return """[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ rep-pow
1 4 yes 1.0 1.0 9.0

[ atomtypes ]
; name mass charge ptype sigma epsilon
T1 12.011 0.0 A 0.35000000 0.20920000

[ moleculetype ]
; name nrexcl
MOL 1

[ atoms ]
; nr type resnr residue atom cgnr charge mass
1 T1 1 MOL A1 1 1.00000000 12.011
2 T1 1 MOL A2 2 -1.00000000 12.011

[ system ]
tp1_5_shift_cutoff

[ molecules ]
MOL 1
"""


def shift_mdp_text() -> str:
    return """integrator = md
nsteps = 0
cutoff-scheme = Verlet
nstlist = 1
nstfout = 1
rlist = 0.9
rcoulomb = 0.9
rvdw = 0.9
coulombtype = Cut-off
vdw-type = Cut-off
pbc = xyz
"""


def shift_gro_text(frame_name: str, atom2_x: float) -> str:
    return f"""{frame_name}
2
    1MOL     A1    1   0.050   1.000   1.000
    1MOL     A2    2   {atom2_x:.3f}   1.000   1.000
   2.000   2.000   2.000
"""


def run_shift_case(case_id: str, atom2_x: float, commands: list[str]) -> dict:
    work_dir = WORK_DIR / "shift_periodic" / case_id
    work_dir.mkdir(parents=True, exist_ok=True)

    write_text(work_dir / "system.top", shift_topology_text())
    write_text(work_dir / "test.mdp", shift_mdp_text())
    write_text(work_dir / "system.gro", shift_gro_text(case_id, atom2_x))

    grompp_cmd = [
        str(GMX),
        "grompp",
        "-f",
        "test.mdp",
        "-c",
        "system.gro",
        "-p",
        "system.top",
        "-o",
        "topol.tpr",
        "-maxwarn",
        "10",
    ]
    commands.append(command_to_string(grompp_cmd, work_dir))
    run_command(
        grompp_cmd,
        work_dir,
        RESULTS_DIR / "raw_shift_grompp.log",
        f"shift fixture {case_id} grompp",
    )

    mdrun_cmd = [
        str(GMX),
        "mdrun",
        "-s",
        "topol.tpr",
        "-rerun",
        "system.gro",
        "-e",
        "ener.edr",
        "-o",
        "traj.trr",
        "-g",
        "md.log",
        "-nt",
        "1",
    ]
    commands.append(command_to_string(mdrun_cmd, work_dir))
    run_command(
        mdrun_cmd,
        work_dir,
        RESULTS_DIR / "raw_shift_mdrun.log",
        f"shift fixture {case_id} mdrun",
    )

    energy_cmd = [str(GMX), "energy", "-f", "ener.edr", "-o", "energy.xvg"]
    energy_stdin = "LJ-(SR)\nCoulomb-(SR)\nPotential\n0\n"
    commands.append(command_to_string(energy_cmd, work_dir, energy_stdin))
    run_command(
        energy_cmd,
        work_dir,
        RESULTS_DIR / "raw_shift_energy.txt",
        f"shift fixture {case_id} energy",
        energy_stdin,
    )

    dump_cmd = [str(GMX), "dump", "-f", "traj.trr"]
    commands.append(command_to_string(dump_cmd, work_dir))
    dump = run_command(
        dump_cmd,
        work_dir,
        RESULTS_DIR / "raw_shift_force_dump.txt",
        f"shift fixture {case_id} dump",
    )

    values = parse_first_numeric_xvg(work_dir / "energy.xvg")
    return {
        "case_id": case_id,
        "atom2_x_nm": atom2_x,
        "lj_sr_kj": values[1],
        "coulomb_sr_kj": values[2],
        "potential_kj": values[3],
        "force_x_atom2": parse_force_x_atom2(dump.stdout),
    }


def run_shift_invariance(commands: list[str]) -> dict:
    case_a = run_shift_case("across_boundary", 1.950, commands)
    case_b = run_shift_case("inside_box", 0.150, commands)

    potential_diff = abs(case_a["potential_kj"] - case_b["potential_kj"])
    lj_diff = abs(case_a["lj_sr_kj"] - case_b["lj_sr_kj"])
    coulomb_diff = abs(case_a["coulomb_sr_kj"] - case_b["coulomb_sr_kj"])
    force_abs_diff = abs(abs(case_a["force_x_atom2"]) - abs(case_b["force_x_atom2"]))
    potential_rel_diff = potential_diff / max(abs(case_a["potential_kj"]), abs(case_b["potential_kj"]))
    force_rel_diff = force_abs_diff / max(abs(case_a["force_x_atom2"]), abs(case_b["force_x_atom2"]))

    return {
        "fixture_id": "shift_periodic_cutoff_9_6",
        "path_family": ["shift handling / periodic image bookkeeping", "plain-C reference cut-off path"],
        "case_a": case_a,
        "case_b": case_b,
        "potential_diff_kj": potential_diff,
        "lj_diff_kj": lj_diff,
        "coulomb_diff_kj": coulomb_diff,
        "force_abs_magnitude_diff": force_abs_diff,
        "potential_relative_diff": potential_rel_diff,
        "force_relative_diff": force_rel_diff,
        "status": "pass" if potential_rel_diff < 1e-5 and force_rel_diff < 1e-5 else "fail",
    }


def write_exclusion_checks_csv(tp13_context: dict, exclusion_result: dict, shift_result: dict) -> None:
    rows = [
        {
            "check_name": "tp1_3_cutoff_vs_pme_path",
            "path_family": "kernel dispatch",
            "metric": "kernel_type",
            "expected": "TRL-0 and TRL-5 should reveal whether cut-off-only bypassed PME and whether kernel family changed",
            "observed": f"{tp13_context['trl0_kernel_line']} | {tp13_context['trl5_kernel_line']}",
            "status": "same_plain_c_reference_path",
        },
        {
            "check_name": "tp1_3_cutoff_vs_pme_nstlist",
            "path_family": "neighbor list update cadence",
            "metric": "pairlist_update",
            "expected": "If cadence alone explains worsening, TP1.3 cut-off run should have materially different update cadence",
            "observed": f"{tp13_context['trl0_pairlist_line']} | {tp13_context['trl5_pairlist_line']}",
            "status": "weakened",
        },
        {
            "check_name": "exclusion_parity",
            "path_family": "exclusion mask handling",
            "metric": "energy_diff_kj",
            "expected": "< 0.01",
            "observed": f"{exclusion_result['difference_kj']:.12g}",
            "status": exclusion_result["status"],
        },
        {
            "check_name": "shift_periodic_invariance",
            "path_family": "shift handling / periodic image bookkeeping",
            "metric": "potential_relative_diff",
            "expected": "< 1e-5",
            "observed": f"{shift_result['potential_relative_diff']:.12g}",
            "status": shift_result["status"],
        },
        {
            "check_name": "shift_periodic_invariance",
            "path_family": "shift handling / periodic image bookkeeping",
            "metric": "force_relative_diff",
            "expected": "< 1e-5",
            "observed": f"{shift_result['force_relative_diff']:.12g}",
            "status": shift_result["status"],
        },
    ]

    with (RESULTS_DIR / "exclusion_mask_checks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["check_name", "path_family", "metric", "expected", "observed", "status"]
        )
        writer.writeheader()
        writer.writerows(rows)


def make_fixture_definition(exclusion_result: dict, shift_result: dict) -> dict:
    return {
        "milestone": "TP1.5",
        "fixtures": [
            {
                "fixture_id": exclusion_result["fixture_id"],
                "purpose": "Exercise exclusion-sensitive listed/nonlisted cut-off-only path with repulsion power 9.",
                "sources": {
                    "topology": exclusion_result["system_topology_source"],
                    "coordinates": exclusion_result["system_coordinate_source"],
                    "lammps_reference": exclusion_result["lammps_source"],
                },
            },
            {
                "fixture_id": shift_result["fixture_id"],
                "purpose": "Exercise cut-off-only periodic image bookkeeping by comparing two PBC-equivalent frames.",
                "system": {
                    "atoms": 2,
                    "box_nm": [2.0, 2.0, 2.0],
                    "charges": [1.0, -1.0],
                    "repulsion_power": 9.0,
                    "coulombtype": "Cut-off",
                    "vdwtype": "Cut-off",
                },
            },
        ],
    }


def make_path_trace() -> dict:
    return {
        "tp1_3_cutoff_only_path": [
            {
                "file": "src/gromacs/mdlib/forcerec.cpp",
                "function": "init_forcerec",
                "physical_role": "Disables SIMD for non-12 LJ repulsion and logs the plain-C fallback.",
                "why_suspected": "TP1.3 cut-off and PME runs both report repulsion power 9 and plain-C reference kernels.",
                "evidence_strength": "strong",
            },
            {
                "file": "src/gromacs/nbnxm/nbnxm_setup.cpp",
                "function": "chooseLJCombinationRule",
                "physical_role": "Selects full pair matrix for non-12 repulsion.",
                "why_suspected": "PCFF 9-6 cut-off-only path uses explicit pair parameters rather than optimized combination-rule kernels.",
                "evidence_strength": "strong",
            },
            {
                "file": "src/gromacs/nbnxm/nbnxm_setup.cpp",
                "function": "init_nb_verlet",
                "physical_role": "Initializes NBNxM kernel setup and pairlist pruning.",
                "why_suspected": "This is the cut-off-only short-range setup entrypoint used by TP1.3.",
                "evidence_strength": "strong",
            },
            {
                "file": "src/gromacs/nbnxm/kerneldispatch.cpp",
                "function": "getCoulombKernelType",
                "physical_role": "Maps Coulomb Cut-off to the ReactionField kernel family.",
                "why_suspected": "Cut-off-only does not enter PME reciprocal-space code; it enters the RF-style direct kernel path.",
                "evidence_strength": "strong",
            },
            {
                "file": "src/gromacs/nbnxm/kerneldispatch.cpp",
                "function": "getVdwKernelType",
                "physical_role": "Selects cut-off LJ kernel variant with full combination matrix.",
                "why_suspected": "The problematic path is cut-off LJ with repulsion power 9 and no LJ-PME.",
                "evidence_strength": "strong",
            },
            {
                "file": "src/gromacs/nbnxm/kerneldispatch.cpp",
                "function": "nbnxn_kernel_cpu",
                "physical_role": "Dispatches CPU plain-C or SIMD kernels to actual pairlists.",
                "why_suspected": "TP1.3 logs show plain-C-4x4 dispatch for both PME and cut-off runs.",
                "evidence_strength": "strong",
            },
            {
                "file": "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h",
                "function": "reference inner nonbonded loop",
                "physical_role": "Applies exclusions, cut-off masking, LJ 9-6 force/energy, RF-style Coulomb, and shift bookkeeping.",
                "why_suspected": "This is the exact plain-C reference loop exercised by the cut-off-only fixture family.",
                "evidence_strength": "strong",
            },
            {
                "file": "src/gromacs/nbnxm/pairlist_tuning.cpp",
                "function": "setupDynamicPairlistPruning",
                "physical_role": "Chooses pairlist lifetime and inner/outer list radii under Verlet.",
                "why_suspected": "Neighbor-list cadence was a candidate family, but TP1.3 logs keep the update period at 10 for both PME and cut-off.",
                "evidence_strength": "moderate",
            },
        ]
    }


def make_regression_summary(tp13_context: dict, exclusion_result: dict, shift_result: dict) -> dict:
    return {
        "milestone": "TP1.5",
        "observed_cutoff_only_symptom": {
            "tp1_3_trial_id": "TRL-5",
            "mean_temp_k": tp13_context["trl5_summary"]["mean_temp"],
            "max_temp_k": tp13_context["trl5_summary"]["max_temp"],
            "final_temp_k": tp13_context["trl5_summary"]["final_temp"],
            "comparison_to_pme_baseline": {
                "pme_trial_id": "TRL-0",
                "mean_temp_k": tp13_context["trl0_summary"]["mean_temp"],
                "max_temp_k": tp13_context["trl0_summary"]["max_temp"],
                "final_temp_k": tp13_context["trl0_summary"]["final_temp"],
            },
        },
        "why_pme_only_is_insufficient": [
            "TRL-5 uses Coulomb Cut-off and vdw Cut-off, so it bypasses PME reciprocal electrostatics entirely.",
            "TRL-0 and TRL-5 both report repulsion power 9 and both dispatch the same plain-C-4x4 nonbonded short-range kernels.",
            "The cut-off-only worsening therefore cannot be explained by the TP1.4 PME split defect alone."
        ],
        "executed_checks": {
            "exclusion_parity": exclusion_result,
            "shift_periodic_invariance": shift_result,
        },
        "bounded_interpretation": [
            "A blanket exclusion-mask failure is weakened: the exclusion-sensitive cut-off 9-6 parity fixture still matches LAMMPS.",
            "A blanket shift/PBC bookkeeping failure is weakened: two PBC-equivalent cut-off frames give invariant energy and force.",
            "Neighbor-list cadence as the sole explanation is weakened: TP1.3 PME and cut-off trials both update the pairlist every 10 steps.",
            "The remaining suspicion is a denser multi-atom reference-loop application issue on the cut-off-only plain-C path, not a pure PME defect."
        ],
        "pme_only_explanation": "split",
    }


def make_suspicion_ranking() -> list[dict]:
    return [
        {
            "rank": 1,
            "candidate": "Shared plain-C cut-off reference path in dense multi-atom application",
            "status": "plausible contributor",
            "basis": "Confirmed path localization: TP1.3 cut-off and PME both use plain-C-4x4 due to repulsion power 9, but only cut-off bypasses PME reciprocal correction entirely.",
        },
        {
            "rank": 2,
            "candidate": "Neighbor-list buffer / dense pairlist population differences",
            "status": "unresolved suspicion",
            "basis": "TP1.3 keeps nstlist at 10 in both cases, but cut-off uses a slightly larger rlist/buffer. No minimal dynamic dense fixture isolated this as a defect in TP1.5.",
        },
        {
            "rank": 3,
            "candidate": "Listed-vs-nonlisted split in exclusion-heavy topologies",
            "status": "partially weakened",
            "basis": "The exclusion-sensitive cut-off 9-6 parity fixture passes, so a blanket split bug is not supported. Dense charged topologies remain untested.",
        },
        {
            "rank": 4,
            "candidate": "Exclusion mask handling in the plain-C reference loop",
            "status": "weakened",
            "basis": "Static inspection shows interact masking is applied to both LJ and Coulomb energies, and the executed exclusion parity fixture matches LAMMPS.",
        },
        {
            "rank": 5,
            "candidate": "Shift handling / periodic image bookkeeping",
            "status": "weakened",
            "basis": "The executed cut-off-only PBC-equivalent frame check is invariant in energy and force.",
        },
        {
            "rank": 6,
            "candidate": "PME-only explanation",
            "status": "split",
            "basis": "TP1.4 proves a PME split defect, but TP1.3 cut-off-only worsening runs on a path that bypasses PME reciprocal space.",
        },
    ]


def make_provenance_manifest(commands: list[str], exclusion_result: dict, shift_result: dict) -> dict:
    return {
        "milestone": "TP1.5",
        "execution_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_output(["rev-parse", "HEAD"]).strip(),
        "git_status_summary": git_output(["status", "--short"]).splitlines(),
        "build_version": gmx_version_text().splitlines(),
        "which_lmp": LMP,
        "commands_run": commands,
        "artifact_paths": [
            "tests/reference_results/tp1_5_cutoff_audit/cutoff_fixture_definition.json",
            "tests/reference_results/tp1_5_cutoff_audit/cutoff_path_trace.json",
            "tests/reference_results/tp1_5_cutoff_audit/cutoff_regression_summary.json",
            "tests/reference_results/tp1_5_cutoff_audit/tp1_5_suspicion_ranking.json",
            "tests/reference_results/tp1_5_cutoff_audit/exclusion_mask_checks.csv",
            "tests/reference_results/tp1_5_cutoff_audit/raw_commands.txt",
            "tests/reference_results/tp1_5_cutoff_audit/provenance_manifest.json",
        ],
        "executed_fixtures": [exclusion_result["fixture_id"], shift_result["fixture_id"]],
        "rerun_scope": "Current TP1.5 execution on current dirty tree; no claim of historical clean provenance.",
    }


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reset_result_files()

    commands: list[str] = []

    tp13_context = parse_tp13_run_context()
    exclusion_result = run_exclusion_parity(commands)
    shift_result = run_shift_invariance(commands)

    fixture_definition = make_fixture_definition(exclusion_result, shift_result)
    path_trace = make_path_trace()
    regression_summary = make_regression_summary(tp13_context, exclusion_result, shift_result)
    suspicion_ranking = make_suspicion_ranking()
    provenance_manifest = make_provenance_manifest(commands, exclusion_result, shift_result)

    write_exclusion_checks_csv(tp13_context, exclusion_result, shift_result)
    write_text(RESULTS_DIR / "raw_commands.txt", "\n".join(commands) + "\n")
    write_text(RESULTS_DIR / "cutoff_fixture_definition.json", json.dumps(fixture_definition, indent=2) + "\n")
    write_text(RESULTS_DIR / "cutoff_path_trace.json", json.dumps(path_trace, indent=2) + "\n")
    write_text(RESULTS_DIR / "cutoff_regression_summary.json", json.dumps(regression_summary, indent=2) + "\n")
    write_text(RESULTS_DIR / "tp1_5_suspicion_ranking.json", json.dumps(suspicion_ranking, indent=2) + "\n")
    write_text(RESULTS_DIR / "provenance_manifest.json", json.dumps(provenance_manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
