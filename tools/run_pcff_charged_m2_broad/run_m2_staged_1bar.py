#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_pcff_charged_capability_subset import run_m1_m3 as core  # noqa: E402
from tools.run_pcff_charged_m2_broad import run_m2_broad as broad  # noqa: E402


DEFAULT_OUT = REPO_ROOT / "tests/reference_results/pcff_charged_expansion/m2_broad_v4_staged_250bar_to_1bar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M2 staged 250 bar to 1 bar dense-parity campaign.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gmx", type=Path, default=REPO_ROOT / "build/bin/gmx")
    parser.add_argument("--m5-spec", type=Path, default=broad.DEFAULT_M5_SPEC)
    parser.add_argument("--gate-h-fixture", type=Path, default=broad.DEFAULT_GATE_H_FIXTURE)
    parser.add_argument("--precondition-ps", type=float, default=100.0)
    parser.add_argument("--target-ps", type=float, default=100.0)
    parser.add_argument("--analysis-window-ps", type=float, default=50.0)
    parser.add_argument("--density-threshold", type=float, default=0.05)
    parser.add_argument("--volume-threshold", type=float, default=0.05)
    parser.add_argument("--precondition-ref-p-bar", type=float, default=250.0)
    parser.add_argument("--target-ref-p-bar", type=float, default=1.0)
    parser.add_argument("--precondition-warmup-ps", type=float, default=0.0)
    parser.add_argument("--precondition-warmup-scope", choices=["gmx-only", "paired"], default="gmx-only")
    parser.add_argument("--m5-formula-count", type=int, default=18)
    parser.add_argument("--m5-box-nm", type=float, default=2.4)
    parser.add_argument("--seed", type=int, default=20260407)
    parser.add_argument("--gmx-integrator", choices=["md", "md-vv"], default="md-vv")
    parser.add_argument("--gmx-tcoupl", choices=["v-rescale", "nose-hoover"], default="nose-hoover")
    parser.add_argument("--gmx-pcoupl", choices=["berendsen", "c-rescale", "parrinello-rahman", "mttk"], default="mttk")
    parser.add_argument("--thermal-start", choices=["generated", "fixture"], default="generated")
    parser.add_argument("--tau-t-ps", type=float, default=0.1)
    parser.add_argument("--tau-p-ps", type=float, default=1.0)
    parser.add_argument("--compressibility-bar-inv", type=float, default=4.5e-5)
    parser.add_argument("--lmp-target-barostat", choices=["npt", "berendsen"], default="npt")
    parser.add_argument("--lmp-neighbor-skin-angstrom", type=float, default=4.0)
    parser.add_argument("--lmp-neighbor-every", type=int, default=1)
    parser.add_argument("--gmx-threads", type=int, default=8)
    parser.add_argument("--lmp-ranks", type=int, default=1)
    parser.add_argument("--lmp-omp-threads", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed PASS precondition stages and existing terminal target reports.")
    parser.add_argument("--systems", nargs="*", help="Optional subset: gate_h_dense_salt_polymer_2x2x2 and/or monoglyme_ethane_litfsi_1to1_dense18.")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_protocol(args: argparse.Namespace, out_root: Path, m5_manifest: Path) -> Path:
    m5_system_id = load_json(m5_manifest)["derived_system"]
    protocol = {
        "schema_name": "pcff_charged_m2_staged_1bar_protocol",
        "schema_version": 1,
        "milestone": "M2 ambient root-cause follow-up, staged protocol candidate",
        "predeclared_before_target_result_interpretation": True,
        "claim_boundary_if_pass": (
            "pressure-preconditioned 1 bar staged dense charged density/volume parity only; "
            "not ambient 1 bar equilibrium parity and not generic dense charged readiness"
        ),
        "required_systems": [
            {
                "system_id": "gate_h_dense_salt_polymer_2x2x2",
                "strict_fixture_manifest": str(args.gate_h_fixture.resolve()),
                "role": "existing dense scaffold",
            },
            {
                "system_id": m5_system_id,
                "strict_fixture_manifest": str(m5_manifest.resolve()),
                "role": "second strict paired dense chemistry",
            },
        ],
        "strict_qualification_rule": {
            "same_pcff_source_required": True,
            "acpype_gaff2_allowed": False,
            "gromacs_side_must_be_pcff_derived": True,
            "lammps_side_must_be_pcff_class2": True,
        },
        "stages": {
            "precondition": {
                "ensemble": "NPT",
                "ref_p_bar": args.precondition_ref_p_bar,
                "duration_ps": args.precondition_ps,
                "warmup_ps": args.precondition_warmup_ps,
                "warmup_scope": args.precondition_warmup_scope if args.precondition_warmup_ps > 0 else None,
                "purpose": "drive both paired systems into the dense basin before the staged 1 bar target",
            },
            "target": {
                "ensemble": "NPT",
                "ref_p_bar": args.target_ref_p_bar,
                "duration_ps": args.target_ps,
                "analysis_window_ps": args.analysis_window_ps,
                "density_relative_difference_max": args.density_threshold,
                "volume_relative_difference_max": args.volume_threshold,
                "fail_rule": "campaign PASS requires every predeclared system to pass the staged target density and volume thresholds; one failure makes campaign FAIL",
            },
        },
        "barostat_thermostat": {
            "gmx_integrator": args.gmx_integrator,
            "gmx_tcoupl": args.gmx_tcoupl,
            "gmx_pcoupl": args.gmx_pcoupl,
            "thermal_start": args.thermal_start,
            "tau_t_ps": args.tau_t_ps,
            "tau_p_ps": args.tau_p_ps,
            "compressibility_bar_inv": args.compressibility_bar_inv,
            "lmp_target_barostat": args.lmp_target_barostat,
            "lmp_neighbor_skin_angstrom": args.lmp_neighbor_skin_angstrom,
            "lmp_neighbor_every": args.lmp_neighbor_every,
        },
        "execution_resources": {
            "gmx_threads": args.gmx_threads,
            "lmp_ranks": args.lmp_ranks,
            "lmp_omp_threads": args.lmp_omp_threads,
        },
        "anti_cherry_pick_rule": "No best-case reporting: all required systems must pass the staged target gate, otherwise staged 1 bar PASS is forbidden.",
        "non_claims": [
            "ambient 1 bar equilibrium dense parity",
            "fully generic dense charged PCFF readiness",
            "charged transport readiness",
            "TP1 thermal recovery as M2 evidence",
            "M5 smoke/grompp success as M2 dense-parity evidence",
        ],
    }
    path = out_root / "m2_staged_1bar_protocol.json"
    write_json(path, protocol)
    return path


def run_precondition_system(args: argparse.Namespace, out_root: Path, system_id: str, fixture_manifest: Path) -> int:
    target = out_root / "systems" / system_id / "precondition_250bar"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/run_pcff_charged_capability_subset/run_m1_m3.py"),
        "--system",
        system_id,
        "--out",
        str(target),
        "--fixture-manifest",
        str(fixture_manifest),
        "--milestone-subset",
        "M2 staged 250bar precondition",
        "--warmup-ps",
        str(args.precondition_warmup_ps),
        "--warmup-scope",
        args.precondition_warmup_scope,
        "--npt-ps",
        str(args.precondition_ps),
        "--analysis-window-ps",
        str(args.analysis_window_ps),
        "--skip-nvt",
        "--density-threshold",
        str(args.density_threshold),
        "--volume-threshold",
        str(args.volume_threshold),
        "--seed",
        str(args.seed),
        "--gmx-integrator",
        args.gmx_integrator,
        "--gmx-tcoupl",
        args.gmx_tcoupl,
        "--gmx-pcoupl",
        args.gmx_pcoupl,
        "--thermal-start",
        args.thermal_start,
        "--tau-t-ps",
        str(args.tau_t_ps),
        "--tau-p-ps",
        str(args.tau_p_ps),
        "--ref-p-bar",
        str(args.precondition_ref_p_bar),
        "--compressibility-bar-inv",
        str(args.compressibility_bar_inv),
        "--lmp-target-barostat",
        args.lmp_target_barostat,
        "--lmp-neighbor-skin-angstrom",
        str(args.lmp_neighbor_skin_angstrom),
        "--lmp-neighbor-every",
        str(args.lmp_neighbor_every),
        "--gmx-threads",
        str(args.gmx_threads),
        "--lmp-ranks",
        str(args.lmp_ranks),
        "--lmp-omp-threads",
        str(args.lmp_omp_threads),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        errors="replace",
    )
    (out_root / f"{system_id}.precondition.stdout").write_text(result.stdout, encoding="utf-8")
    (out_root / f"{system_id}.precondition.stderr").write_text(result.stderr, encoding="utf-8")
    return result.returncode


def summarize_series(values: list[float]) -> dict[str, float | None]:
    return core.summarize_series(values)


def run_target_stage(args: argparse.Namespace, out_root: Path, system_id: str) -> dict[str, object]:
    system_root = out_root / "systems" / system_id
    pre_root = system_root / "precondition_250bar"
    target_root = system_root / "target_1bar"
    if target_root.exists():
        shutil.rmtree(target_root)
    gmx_dir = target_root / "gromacs"
    lmp_dir = target_root / "lammps"
    gmx_dir.mkdir(parents=True)
    lmp_dir.mkdir(parents=True)

    gmx_pre = pre_root / "paired_npt/gromacs"
    lmp_pre = pre_root / "paired_npt/lammps"
    for name in ["system.top", "npt.gro", "npt.cpt"]:
        shutil.copy(gmx_pre / name, gmx_dir / name)
    shutil.copy(lmp_pre / "system.in", lmp_dir / "system.in")
    shutil.copy(lmp_pre / "final.data", lmp_dir / "precondition_final.data")

    core.write_text(
        gmx_dir / "target_1bar.mdp",
        core.build_gmx_npt_mdp(
            args.gmx_integrator,
            int(args.target_ps / 0.001),
            args.seed,
            args.gmx_tcoupl,
            core.GROMACS_PCOUPL[args.gmx_pcoupl],
            "fixture",
            None,
            None,
            args.tau_t_ps,
            args.tau_p_ps,
            args.target_ref_p_bar,
            args.compressibility_bar_inv,
            continuation=True,
        ),
    )
    core.run_command(
        [
            str(REPO_ROOT / "build/bin/gmx"),
            "grompp",
            "-f",
            "target_1bar.mdp",
            "-c",
            "npt.gro",
            "-t",
            "npt.cpt",
            "-p",
            "system.top",
            "-o",
            "target_1bar.tpr",
            "-po",
            "target_1bar_mdout.mdp",
            "-maxwarn",
            "1",
        ],
        gmx_dir,
        gmx_dir / "grompp_target.stdout",
        gmx_dir / "grompp_target.stderr",
    )
    core.write_lammps_npt_input(
        lmp_dir / "system.in",
        lmp_dir / "target_1bar.in",
        args.seed,
        int(args.target_ps * 1000.0),
        "fixture",
        args.tau_t_ps,
        args.tau_p_ps,
        args.target_ref_p_bar,
        args.lmp_target_barostat,
        args.lmp_neighbor_skin_angstrom,
        args.lmp_neighbor_every,
        read_data_name="precondition_final.data",
    )
    lmp_cmd, lmp_env = core.lammps_command("target_1bar.in", "target_1bar.log", args)
    gmx_proc = core.spawn_command(
        [str(REPO_ROOT / "build/bin/gmx"), "mdrun", "-s", "target_1bar.tpr", "-deffnm", "target_1bar", "-nt", str(args.gmx_threads), "-pin", "off", "-reprod"],
        gmx_dir,
        gmx_dir / "mdrun_target.stdout",
        gmx_dir / "mdrun_target.stderr",
    )
    lmp_proc = core.spawn_command(
        lmp_cmd,
        lmp_dir,
        lmp_dir / "target_1bar.stdout",
        lmp_dir / "target_1bar.stderr",
        extra_env=lmp_env,
    )
    core.wait_for_process_pair(gmx_proc, lmp_proc)
    core.run_command(
        [str(REPO_ROOT / "build/bin/gmx"), "energy", "-f", "target_1bar.edr", "-o", "target_1bar_energy.xvg"],
        gmx_dir,
        gmx_dir / "energy_target.stdout",
        gmx_dir / "energy_target.stderr",
        stdin_text="Potential\nTemperature\nPressure\nVolume\nDensity\n0\n",
    )

    gmx_rows = core.parse_xvg(gmx_dir / "target_1bar_energy.xvg")
    gmx_max_time = max(row[0] for row in gmx_rows)
    gmx_window = [row for row in gmx_rows if row[0] >= gmx_max_time - args.analysis_window_ps]
    lmp_blocks = core.parse_lammps_blocks(lmp_dir / "target_1bar.log")
    require(bool(lmp_blocks), f"LAMMPS target stage did not produce a thermo block for {system_id}")
    lmp_rows = lmp_blocks[-1]["rows"]  # type: ignore[index]
    lmp_window = [row for row in lmp_rows if (row[0] * 0.001) >= args.target_ps - args.analysis_window_ps]
    require(bool(gmx_window), f"GROMACS target stage has no analysis-window rows for {system_id}")
    require(bool(lmp_window), f"LAMMPS target stage has no analysis-window rows for {system_id}")

    gmx_density = [row[5] for row in gmx_window]
    gmx_volume = [row[4] for row in gmx_window]
    lmp_density = [row[7] * 1000.0 for row in lmp_window]
    lmp_volume = [row[6] / 1000.0 for row in lmp_window]
    density_rel_diff = abs(core.statistics.fmean(gmx_density) - core.statistics.fmean(lmp_density)) / core.statistics.fmean(lmp_density)
    volume_rel_diff = abs(core.statistics.fmean(gmx_volume) - core.statistics.fmean(lmp_volume)) / core.statistics.fmean(lmp_volume)
    passed = density_rel_diff <= args.density_threshold and volume_rel_diff <= args.volume_threshold

    report = {
        "schema_name": "pcff_charged_m2_staged_1bar_target_report",
        "schema_version": 1,
        "system_id": system_id,
        "status": "PASS" if passed else "FAIL",
        "claim_boundary_if_pass": "pressure-preconditioned 1 bar staged dense parity only",
        "precondition_report": str((pre_root / "paired_npt/dense_npt_parity_report.json").resolve()),
        "protocol": {
            "precondition_ref_p_bar": args.precondition_ref_p_bar,
            "precondition_duration_ps": args.precondition_ps,
            "target_ref_p_bar": args.target_ref_p_bar,
            "target_duration_ps": args.target_ps,
            "analysis_window_ps": args.analysis_window_ps,
            "density_rel_diff_max": args.density_threshold,
            "volume_rel_diff_max": args.volume_threshold,
            "gmx_time_window_ps": [gmx_max_time - args.analysis_window_ps, gmx_max_time],
            "lammps_time_window_ps": [args.target_ps - args.analysis_window_ps, args.target_ps],
        },
        "gromacs": {
            "density_kg_m3": summarize_series(gmx_density),
            "volume_nm3": summarize_series(gmx_volume),
            "pressure_bar": summarize_series([row[3] for row in gmx_window]),
            "temperature_k": summarize_series([row[2] for row in gmx_window]),
        },
        "lammps": {
            "density_kg_m3": summarize_series(lmp_density),
            "volume_nm3": summarize_series(lmp_volume),
            "pressure_atm": summarize_series([row[5] for row in lmp_window]),
            "temperature_k": summarize_series([row[1] for row in lmp_window]),
        },
        "parity_metrics": {
            "density_rel_diff": density_rel_diff,
            "volume_rel_diff": volume_rel_diff,
        },
        "artifacts": {
            "gromacs_root": str(gmx_dir.resolve()),
            "lammps_root": str(lmp_dir.resolve()),
            "gromacs_energy_xvg": str((gmx_dir / "target_1bar_energy.xvg").resolve()),
            "lammps_log": str((lmp_dir / "target_1bar.log").resolve()),
        },
        "non_claims": [
            "ambient 1 bar equilibrium parity",
            "transport readiness",
            "generic dense charged readiness",
        ],
    }
    write_json(target_root / "staged_1bar_parity_report.json", report)
    return report


def summarize_campaign(args: argparse.Namespace, out_root: Path, protocol_path: Path, manifests: dict[str, Path]) -> Path:
    systems: list[dict[str, object]] = []
    for system_id, manifest in manifests.items():
        pre_report_path = out_root / "systems" / system_id / "precondition_250bar/paired_npt/dense_npt_parity_report.json"
        target_report_path = out_root / "systems" / system_id / "target_1bar/staged_1bar_parity_report.json"
        entry: dict[str, object] = {
            "system_id": system_id,
            "strict_fixture_manifest": str(manifest.resolve()),
            "precondition_report": str(pre_report_path.resolve()) if pre_report_path.exists() else None,
            "target_report": str(target_report_path.resolve()) if target_report_path.exists() else None,
            "precondition_status": None,
            "target_status": "NOT_RUN",
            "density_rel_diff": None,
            "volume_rel_diff": None,
            "analysis_window_ps": None,
            "target_duration_ps": None,
        }
        if pre_report_path.exists():
            pre_report = load_json(pre_report_path)
            entry["precondition_status"] = pre_report.get("status")
        if target_report_path.exists():
            target_report = load_json(target_report_path)
            entry["target_status"] = target_report.get("status")
            entry["density_rel_diff"] = target_report.get("parity_metrics", {}).get("density_rel_diff")
            entry["volume_rel_diff"] = target_report.get("parity_metrics", {}).get("volume_rel_diff")
            entry["analysis_window_ps"] = target_report.get("protocol", {}).get("analysis_window_ps")
            entry["target_duration_ps"] = target_report.get("protocol", {}).get("target_duration_ps")
        systems.append(entry)

    all_run = all(system["target_status"] in {"PASS", "FAIL"} for system in systems)
    all_pass = all(
        system["precondition_status"] == "PASS"
        and system["target_status"] == "PASS"
        and float(system["analysis_window_ps"] or 0.0) >= args.analysis_window_ps
        and float(system["target_duration_ps"] or 0.0) >= args.target_ps
        for system in systems
    )
    summary = {
        "schema_name": "pcff_charged_m2_staged_1bar_campaign_summary",
        "schema_version": 1,
        "status": "PASS" if all_pass else "FAIL" if all_run else "PENDING",
        "protocol": str(protocol_path.resolve()),
        "old_boundary": "ambient 1 bar broader dense parity is unresolved; prior broader M2 pass is high-pressure 250 bar only",
        "new_candidate_boundary": "pressure-preconditioned 1 bar staged dense charged parity only if all required systems pass",
        "systems": systems,
        "anti_cherry_pick_rule_enforced": True,
        "claim_honesty": {
            "ambient_equilibrium_claimed": False,
            "tp1_thermal_recovery_counted_as_m2": False,
            "m5_smoke_counted_as_m2": False,
            "high_pressure_250bar_relabelled_as_1bar": False,
        },
    }
    path = out_root / "m2_staged_1bar_campaign_summary.json"
    write_json(path, summary)
    return path


def write_sha_manifest(root: Path) -> Path:
    manifest_path = root / "sha256_manifest.txt"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest_path:
            rows.append(f"{broad.sha256(path)}  {broad.repo_rel(path)}")
    broad.write_text(manifest_path, "\n".join(rows) + "\n")
    return manifest_path


def main() -> int:
    args = parse_args()
    require(args.precondition_ps >= 100.0, "staged protocol requires precondition_ps >= 100")
    require(args.target_ps >= 100.0, "staged protocol requires target_ps >= 100")
    require(args.analysis_window_ps >= 50.0, "staged protocol requires analysis_window_ps >= 50")
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    m5_manifest = broad.prepare_m5_dense_fixture(args, out_root)
    m5_system_id = load_json(m5_manifest)["derived_system"]
    protocol_path = freeze_protocol(args, out_root, m5_manifest)
    manifests = {
        "gate_h_dense_salt_polymer_2x2x2": args.gate_h_fixture.resolve(),
        m5_system_id: m5_manifest.resolve(),
    }
    requested = set(args.systems or manifests.keys())
    unknown = requested.difference(manifests)
    require(not unknown, f"Unknown --systems entries: {sorted(unknown)}")

    if args.execute:
        for system_id, manifest in manifests.items():
            if system_id not in requested:
                continue
            system_root = out_root / "systems" / system_id
            pre_report_path = system_root / "precondition_250bar/paired_npt/dense_npt_parity_report.json"
            target_report_path = system_root / "target_1bar/staged_1bar_parity_report.json"
            if args.resume and target_report_path.exists() and load_json(target_report_path).get("status") in {"PASS", "FAIL"}:
                continue
            if args.resume and pre_report_path.exists() and load_json(pre_report_path).get("status") == "PASS":
                pre_rc = 0
            else:
                pre_rc = run_precondition_system(args, out_root, system_id, manifest)
            if pre_rc != 0:
                break
            if pre_report_path.exists() and load_json(pre_report_path).get("status") != "PASS":
                break
            run_target_stage(args, out_root, system_id)

    summary_path = summarize_campaign(args, out_root, protocol_path, manifests)
    sha_path = write_sha_manifest(out_root)
    print(json.dumps({"protocol": str(protocol_path), "summary": str(summary_path), "sha256_manifest": str(sha_path)}, indent=2))
    summary = load_json(summary_path)
    return 0 if summary["status"] == "PASS" else 2 if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
