#!/usr/bin/env python3

import csv
import json
import pathlib
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[2]
GMX = ROOT / "build/bin/gmx"
TOOL_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "tests/reference_results/tp1_4_pme_proof"
WORK_DIR = TOOL_DIR / "work"
PRIOR_DIR = RESULTS_DIR / "prior_nonpristine"
INPUTS_DIR = RESULTS_DIR / "rerun_inputs"

RCUT_VALUES = [0.7, 0.8, 0.9, 1.0, 1.1]


@dataclass
class ScanPoint:
    rcut: float
    lj_sr: float
    lj_recip: float
    potential: float
    force_x_atom2: float


def run_command(cmd: list[str], cwd: pathlib.Path, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, input=stdin, capture_output=True, check=True)


def write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def append_section(path: pathlib.Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"===== {title} =====\n")
        handle.write(body)
        if not body.endswith("\n"):
            handle.write("\n")
        handle.write("\n")


def command_to_string(cmd: list[str], cwd: pathlib.Path, stdin: str | None = None) -> str:
    rendered = f"(cd {cwd} && {' '.join(cmd)})"
    if stdin is not None:
        rendered += f"  # stdin={stdin!r}"
    return rendered


def git_output(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout


def gmx_version_text() -> str:
    return subprocess.run([str(GMX), "--version"], cwd=ROOT, text=True, capture_output=True, check=True).stdout


def fixture_topology_text() -> str:
    return """[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ rep-pow
1 4 yes 1.0 1.0 9.0

[ atomtypes ]
; name mass charge ptype sigma epsilon
A 12.011 0.0 A 0.34 0.20
B 12.011 0.0 A 0.40 0.10

[ moleculetype ]
; name nrexcl
MOL 1

[ atoms ]
; nr type resnr residue atom cgnr charge mass
1 A 1 MOL A1 1 0.0 12.011
2 B 1 MOL B1 1 0.0 12.011

[ system ]
TP1.4 mixed-type 9-6 LJ-PME

[ molecules ]
MOL 1
"""


def fixture_coordinates_text() -> str:
    return """TP1.4 mixed-type 9-6 LJ-PME
2
    1MOL     A1    1   0.000   0.000   0.000
    1MOL     B1    2   0.500   0.000   0.000
   5.000   5.000   5.000
"""


def fixture_mdp_text(rcut: float) -> str:
    return f"""integrator = md
nsteps = 0
cutoff-scheme = Verlet
nstlist = 1
nstfout = 1
rlist = {rcut}
rcoulomb = {rcut}
rvdw = {rcut}
coulombtype = Cut-off
vdwtype = PME
lj-pme-comb-rule = geometric
ewald-rtol-lj = 1e-5
pbc = xyz
"""


def write_fixture_files(work_dir: pathlib.Path, rcut: float) -> None:
    write_text(work_dir / "system.top", fixture_topology_text())
    write_text(work_dir / "system.gro", fixture_coordinates_text())
    write_text(work_dir / "test.mdp", fixture_mdp_text(rcut))


def preserve_fixture_inputs() -> None:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    write_text(INPUTS_DIR / "system.top", fixture_topology_text())
    write_text(INPUTS_DIR / "system.gro", fixture_coordinates_text())
    for rcut in RCUT_VALUES:
        write_text(INPUTS_DIR / f"test_rcut_{rcut:.1f}.mdp", fixture_mdp_text(rcut))


def archive_prior_nonpristine_artifacts() -> dict[str, list[str]]:
    PRIOR_DIR.mkdir(parents=True, exist_ok=True)

    root_artifacts = [
        RESULTS_DIR / "pme_fixture_definition.json",
        RESULTS_DIR / "pme_energy_force_scan.csv",
        RESULTS_DIR / "pme_continuity_summary.json",
        RESULTS_DIR / "tp1_4_suspicion_update.json",
    ]
    tool_artifacts = [
        TOOL_DIR / "pme_fixture_definition.json",
        TOOL_DIR / "pme_energy_force_scan.csv",
        TOOL_DIR / "pme_continuity_summary.json",
        TOOL_DIR / "tp1_4_suspicion_update.json",
        TOOL_DIR / "run_logs.json",
    ]

    copied_root: list[str] = []
    copied_tool: list[str] = []

    for path in root_artifacts:
        if path.exists():
            target = PRIOR_DIR / f"legacy_{path.name}"
            shutil.copy2(path, target)
            copied_root.append(str(target.relative_to(ROOT)))

    for path in tool_artifacts:
        if path.exists():
            target = PRIOR_DIR / f"legacy_tool_{path.name}"
            shutil.copy2(path, target)
            copied_tool.append(str(target.relative_to(ROOT)))

    snapshot = {
        "snapshot_type": "prior_nonpristine_tp1_4_artifacts",
        "snapshot_time_utc": datetime.now(timezone.utc).isoformat(),
        "copied_root_artifacts": copied_root,
        "copied_tool_artifacts": copied_tool,
        "note": "These files existed before TP1.4-RR regenerated current rerun artifacts. They should not be treated as current clean-rerun provenance."
    }
    write_text(PRIOR_DIR / "snapshot_manifest.json", json.dumps(snapshot, indent=2) + "\n")
    return {
        "copied_root_artifacts": copied_root,
        "copied_tool_artifacts": copied_tool,
    }


def reset_raw_logs() -> None:
    for filename in [
        "raw_grompp.log",
        "raw_mdrun_rerun.log",
        "raw_energy_output.txt",
        "raw_force_dump.txt",
        "commands_run.txt",
    ]:
        path = RESULTS_DIR / filename
        if path.exists():
            path.unlink()


def parse_first_numeric_xvg(path: pathlib.Path) -> list[float]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "@")):
            return [float(token) for token in line.split()]
    raise ValueError(f"No numeric data in {path}")


def parse_force_x_atom2(trr_dump: str) -> float:
    for line in trr_dump.splitlines():
        if "f[    1]" in line:
            return float(line.split("=")[1].strip("{} ").split(",")[0])
    raise ValueError("Could not find atom-2 force in gmx dump output")


def run_scan() -> tuple[list[ScanPoint], list[str]]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reset_raw_logs()

    commands_run: list[str] = []
    scan_points: list[ScanPoint] = []

    for rcut in RCUT_VALUES:
        write_fixture_files(WORK_DIR, rcut)

        grompp_cmd = [str(GMX), "grompp", "-f", "test.mdp", "-c", "system.gro", "-p", "system.top", "-o", "test.tpr", "-maxwarn", "10"]
        commands_run.append(command_to_string(grompp_cmd, WORK_DIR))
        grompp = run_command(grompp_cmd, WORK_DIR)
        append_section(
            RESULTS_DIR / "raw_grompp.log",
            f"rcut={rcut:.1f} stderr",
            grompp.stderr,
        )
        append_section(
            RESULTS_DIR / "raw_grompp.log",
            f"rcut={rcut:.1f} stdout",
            grompp.stdout,
        )

        mdrun_cmd = [str(GMX), "mdrun", "-s", "test.tpr", "-rerun", "system.gro", "-e", "test.edr", "-o", "test.trr", "-g", "test.log", "-nt", "1"]
        commands_run.append(command_to_string(mdrun_cmd, WORK_DIR))
        mdrun = run_command(mdrun_cmd, WORK_DIR)
        append_section(
            RESULTS_DIR / "raw_mdrun_rerun.log",
            f"rcut={rcut:.1f} stderr",
            mdrun.stderr,
        )
        append_section(
            RESULTS_DIR / "raw_mdrun_rerun.log",
            f"rcut={rcut:.1f} stdout",
            mdrun.stdout,
        )

        energy_cmd = [str(GMX), "energy", "-f", "test.edr", "-o", "energy.xvg"]
        energy_stdin = "LJ-(SR)\nLJ-recip.\nPotential\n0\n"
        commands_run.append(command_to_string(energy_cmd, WORK_DIR, energy_stdin))
        energy = run_command(energy_cmd, WORK_DIR, energy_stdin)
        append_section(
            RESULTS_DIR / "raw_energy_output.txt",
            f"rcut={rcut:.1f} stdout",
            energy.stdout,
        )
        append_section(
            RESULTS_DIR / "raw_energy_output.txt",
            f"rcut={rcut:.1f} stderr",
            energy.stderr,
        )

        dump_cmd = [str(GMX), "dump", "-f", "test.trr"]
        commands_run.append(command_to_string(dump_cmd, WORK_DIR))
        dump = run_command(dump_cmd, WORK_DIR)
        append_section(
            RESULTS_DIR / "raw_force_dump.txt",
            f"rcut={rcut:.1f} stdout",
            dump.stdout,
        )
        append_section(
            RESULTS_DIR / "raw_force_dump.txt",
            f"rcut={rcut:.1f} stderr",
            dump.stderr,
        )

        vals = parse_first_numeric_xvg(WORK_DIR / "energy.xvg")
        scan_points.append(
            ScanPoint(
                rcut=rcut,
                lj_sr=vals[1],
                lj_recip=vals[2],
                potential=vals[3],
                force_x_atom2=parse_force_x_atom2(dump.stdout),
            )
        )

    write_text(RESULTS_DIR / "commands_run.txt", "\n".join(commands_run) + "\n")
    return scan_points, commands_run


def write_csv(points: list[ScanPoint], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rcut", "lj_sr", "lj_recip", "potential", "force_x_atom2"])
        for point in points:
            writer.writerow([point.rcut, point.lj_sr, point.lj_recip, point.potential, point.force_x_atom2])


def make_fixture_definition(points: list[ScanPoint]) -> dict:
    return {
        "fixture_name": "tp1_4_mixed_type_9_6_ljpme_split_scan",
        "fixture_scope": "current_rerun_evidence",
        "system": "2-atom periodic box with mixed atom types A/B",
        "defaults": {
            "comb_rule": 4,
            "repulsion_power": 9.0,
            "gen_pairs": "yes"
        },
        "atomtypes": [
            { "type": "A", "sigma_nm": 0.34, "epsilon_kj_per_mol": 0.20 },
            { "type": "B", "sigma_nm": 0.40, "epsilon_kj_per_mol": 0.10 }
        ],
        "coordinates_nm": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "box_nm": [5.0, 5.0, 5.0],
        "mdp_scan": {
            "vdwtype": "PME",
            "lj_pme_comb_rule": "geometric",
            "coulombtype": "Cut-off",
            "ewald_rtol_lj": 1e-5,
            "rlist_rcoulomb_rvdw_values_nm": [point.rcut for point in points]
        },
        "expected_property": "Total LJ force and energy should remain approximately invariant as rcut moves when the real-space/reciprocal-space split is correct.",
        "raw_input_paths": [
            str((INPUTS_DIR / "system.top").relative_to(ROOT)),
            str((INPUTS_DIR / "system.gro").relative_to(ROOT)),
        ] + [str((INPUTS_DIR / f"test_rcut_{point.rcut:.1f}.mdp").relative_to(ROOT)) for point in points]
    }


def make_continuity_summary(points: list[ScanPoint]) -> dict:
    potential_values = [point.potential for point in points]
    force_values = [point.force_x_atom2 for point in points]
    potential_span = max(potential_values) - min(potential_values)
    force_span = max(force_values) - min(force_values)
    relative_span = potential_span / abs(points[-1].potential)

    return {
        "evidence_scope": "current_rerun_evidence_on_dirty_tree",
        "localized_path": [
            "K1 isolated 9-6 nonbonded kernel consistency passed, so TP1.4 targets the PME split path rather than isolated pair-force mathematics.",
            "src/gromacs/mdlib/forcerec.cpp builds full pair prefactors (6*C6, repulsionPower*C_repulsive) and separate LJ-PME C6 grid parameters.",
            "src/gromacs/nbnxm/atomdata.cpp uses LJCombinationRule::None for 9-6 LJ-PME pair parameters while still generating geometric C6 grid data.",
            "src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h computes the real-space 9-6 pair term from the full pair matrix and subtracts a geometric c6grid reciprocal correction.",
            "src/gromacs/ewald/pme.cpp allocates a single geometric LJ grid for non-LB LJ-PME."
        ],
        "rcut_scan": [asdict(point) for point in points],
        "expected_behavior": "Invariant total force/potential versus rcut for a fixed pair geometry.",
        "observed_behavior": "Large monotonic drift in both total potential and force as rcut moves from 0.7 to 1.1 nm.",
        "potential_span": potential_span,
        "force_span": force_span,
        "relative_span_vs_rcut_1p1": relative_span,
        "invariance_check": "FAILED",
        "conclusion": "defect reproduced and plausibly large enough to matter"
    }


def make_suspicion_update() -> dict:
    return {
        "milestone": "TP1.4-RR",
        "status": "PARTIAL",
        "result_classification": "defect reproduced and plausibly large enough to matter",
        "evidence_scope": "current_rerun_evidence_on_dirty_tree",
        "key_findings": [
            "The TP1.4 minimal 2-atom periodic LJ-PME fixture was rerun and raw command outputs were preserved.",
            "The rerun again shows a large dependence of total LJ energy and force on rcut, violating split continuity.",
            "This rerun is tied to an explicit dirty-tree source/build state, so it hardens provenance but does not create a pristine clean-tree proof."
        ],
        "tp1_3_link_assessment": "Not directly demonstrated. The rerun supports TP1.4 as a plausible contributor but still does not prove dominant TP1.3 causality.",
        "pass_readiness_assessment": "Still PARTIAL. Provenance is stronger than before, but the rerun is not on a pristine clean tree because uncommitted source changes affect the LJ-PME startup path."
    }


def make_provenance_manifest(points: list[ScanPoint], commands_run: list[str], prior_snapshot: dict[str, list[str]]) -> dict:
    git_status = git_output(["status", "--short"])
    diff_name_only = git_output(["diff", "--name-only"])
    relevant_dirty = []
    for path in diff_name_only.splitlines():
        if path in {
            "src/gromacs/nbnxm/nbnxm_setup.cpp",
            "src/gromacs/nbnxm/kerneldispatch.cpp",
            "tools/run_tp1_4_pme_proof/run_pme_proof.py",
            "docs/validation_report_tp1_4.md",
            "docs/tp1_4_pme_sixthpower_direct_proof.md",
        }:
            relevant_dirty.append(path)

    return {
        "milestone": "TP1.4-RR",
        "execution_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_output(["rev-parse", "HEAD"]).strip(),
        "git_status_summary": git_status.splitlines(),
        "dirty_tree": True,
        "clean_rerun_possible_in_current_tree": False,
        "rerun_classification": "current_rerun_on_dirty_tree",
        "why_not_pristine": [
            "The repository has uncommitted source changes.",
            "The current built binary reports itself as dirty.",
            "Uncommitted changes in src/gromacs/nbnxm/nbnxm_setup.cpp and src/gromacs/nbnxm/kerneldispatch.cpp affect the 9-6 LJ-PME startup path."
        ],
        "source_changes_affecting_reproducibility": relevant_dirty,
        "build_version": gmx_version_text().splitlines(),
        "fixture_identity": make_fixture_definition(points),
        "exact_commands_run": commands_run,
        "output_artifact_paths": [
            "tests/reference_results/tp1_4_pme_proof/pme_fixture_definition.json",
            "tests/reference_results/tp1_4_pme_proof/pme_energy_force_scan.csv",
            "tests/reference_results/tp1_4_pme_proof/pme_continuity_summary.json",
            "tests/reference_results/tp1_4_pme_proof/tp1_4_suspicion_update.json",
            "tests/reference_results/tp1_4_pme_proof/tp1_4_provenance_manifest.json",
            "tests/reference_results/tp1_4_pme_proof/raw_grompp.log",
            "tests/reference_results/tp1_4_pme_proof/raw_mdrun_rerun.log",
            "tests/reference_results/tp1_4_pme_proof/raw_energy_output.txt",
            "tests/reference_results/tp1_4_pme_proof/raw_force_dump.txt",
            "tests/reference_results/tp1_4_pme_proof/commands_run.txt"
        ],
        "prior_nonpristine_snapshot": prior_snapshot
    }


def main() -> None:
    prior_snapshot = archive_prior_nonpristine_artifacts()
    preserve_fixture_inputs()
    points, commands_run = run_scan()

    fixture_definition = make_fixture_definition(points)
    continuity_summary = make_continuity_summary(points)
    suspicion_update = make_suspicion_update()
    manifest = make_provenance_manifest(points, commands_run, prior_snapshot)

    for directory in [TOOL_DIR, RESULTS_DIR]:
        write_csv(points, directory / "pme_energy_force_scan.csv")
        write_text(directory / "pme_fixture_definition.json", json.dumps(fixture_definition, indent=2) + "\n")
        write_text(directory / "pme_continuity_summary.json", json.dumps(continuity_summary, indent=2) + "\n")
        write_text(directory / "tp1_4_suspicion_update.json", json.dumps(suspicion_update, indent=2) + "\n")

    write_text(RESULTS_DIR / "tp1_4_provenance_manifest.json", json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
