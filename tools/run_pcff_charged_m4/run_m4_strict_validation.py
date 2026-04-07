#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
M11_ROOT = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "pcff_charged_expansion"
    / "probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4"
)
DEFAULT_MANIFEST = M11_ROOT / "qualified_pair_manifest.json"
DEFAULT_NPT_REPORT = M11_ROOT / "paired_npt" / "dense_npt_parity_report.json"
DEFAULT_TRANSPORT_RESULT = (
    M11_ROOT
    / "m4_strict_validation"
    / "transport_facing_rerun"
    / "summaries"
    / "candidate_result.json"
)
DEFAULT_OUT = M11_ROOT / "m4_strict_validation"
KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1
KCAL_PER_A_TO_KJ_PER_NM = KCAL_TO_KJ / ANGSTROM_TO_NM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/audit M4 strict charged validation on the qualified pair.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--npt-report", type=Path, default=DEFAULT_NPT_REPORT)
    parser.add_argument("--transport-result", type=Path, default=DEFAULT_TRANSPORT_RESULT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gmx", type=Path, default=REPO_ROOT / "build" / "bin" / "gmx")
    parser.add_argument("--energy-rel-threshold", type=float, default=1.0e-3)
    parser.add_argument("--force-rms-rel-threshold", type=float, default=0.05)
    parser.add_argument("--force-max-rel-threshold", type=float, default=0.15)
    parser.add_argument("--force-max-abs-threshold", type=float, default=5.0)
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(read_text(path))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_sha256_manifest(root: Path, manifest_path: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {repo_rel(path)}")
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def run_command(
    cmd: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    stdin_text: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        input=stdin_text,
        text=True,
        capture_output=True,
        errors="replace",
        env={**os.environ, "GMX_MAXBACKUP": "-1", **(extra_env or {})},
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def parse_lammps_data_atoms(data_path: Path) -> tuple[list[dict[str, float | int]], list[float]]:
    atoms: list[dict[str, float | int]] = []
    bounds: dict[str, tuple[float, float]] = {}
    section: str | None = None
    for raw in read_text(data_path).splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[2] == "xlo" and parts[3] == "xhi":
            bounds["x"] = (float(parts[0]), float(parts[1]))
            continue
        if len(parts) >= 4 and parts[2] == "ylo" and parts[3] == "yhi":
            bounds["y"] = (float(parts[0]), float(parts[1]))
            continue
        if len(parts) >= 4 and parts[2] == "zlo" and parts[3] == "zhi":
            bounds["z"] = (float(parts[0]), float(parts[1]))
            continue
        if line == "Atoms":
            section = "Atoms"
            continue
        if line in {"Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"}:
            section = line
            continue
        if section == "Atoms" and len(parts) >= 7 and parts[0].lstrip("-").isdigit():
            atoms.append(
                {
                    "id": int(parts[0]),
                    "molecule_id": int(parts[1]),
                    "type": int(parts[2]),
                    "charge": float(parts[3]),
                    "x": float(parts[4]),
                    "y": float(parts[5]),
                    "z": float(parts[6]),
                }
            )
    atoms.sort(key=lambda atom: int(atom["id"]))
    box_nm = [(bounds[axis][1] - bounds[axis][0]) * ANGSTROM_TO_NM for axis in ("x", "y", "z")]
    return atoms, box_nm


def write_high_precision_gro(data_path: Path, gro_path: Path) -> None:
    atoms, box_nm = parse_lammps_data_atoms(data_path)
    lines = ["Generated from M4 strict LAMMPS data", f"{len(atoms):5d}"]
    local_index_by_molecule: dict[int, int] = {}
    for atom in atoms:
        mol = int(atom["molecule_id"])
        local_index = local_index_by_molecule.get(mol, 0) + 1
        local_index_by_molecule[mol] = local_index
        x = float(atom["x"]) * ANGSTROM_TO_NM
        y = float(atom["y"]) * ANGSTROM_TO_NM
        z = float(atom["z"]) * ANGSTROM_TO_NM
        lines.append(f"{mol % 100000:5d}{'MOL':<5s}{f'A{local_index}':>5s}{int(atom['id']) % 100000:5d}{x:15.7f}{y:15.7f}{z:15.7f}")
    lines.append(f"{box_nm[0]:15.7f}{box_nm[1]:15.7f}{box_nm[2]:15.7f}")
    write_text(gro_path, "\n".join(lines) + "\n")


def mechanical_mdp() -> str:
    return "\n".join(
        [
            "integrator = md",
            "nsteps = 0",
            "cutoff-scheme = Verlet",
            "nstlist = 10",
            "verlet-buffer-tolerance = -1",
            "rlist = 0.9",
            "coulombtype = PME",
            "coulomb-modifier = none",
            "rcoulomb = 0.9",
            "pme-order = 4",
            "fourierspacing = 0.12",
            "ewald-rtol = 1e-4",
            "vdw-type = Cut-off",
            "vdw-modifier = none",
            "rvdw = 0.9",
            "DispCorr = no",
            "pbc = xyz",
            "nstfout = 1",
            "",
        ]
    )


def parse_lammps_pe(log_path: Path) -> float:
    lines = read_text(log_path).splitlines()
    for index, line in enumerate(lines):
        if "Step" in line and "PotEng" in line:
            for row in lines[index + 1 :]:
                parts = row.split()
                if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
                    return float(parts[1])
    raise ValueError(f"Unable to parse LAMMPS PotEng from {log_path}")


def parse_lammps_forces(dump_path: Path) -> dict[int, list[float]]:
    forces: dict[int, list[float]] = {}
    reading = False
    for raw in read_text(dump_path).splitlines():
        if raw.startswith("ITEM: ATOMS"):
            reading = True
            continue
        if raw.startswith("ITEM:"):
            reading = False
            continue
        if not reading:
            continue
        parts = raw.split()
        if len(parts) >= 4:
            forces[int(parts[0])] = [float(parts[1]) * KCAL_PER_A_TO_KJ_PER_NM, float(parts[2]) * KCAL_PER_A_TO_KJ_PER_NM, float(parts[3]) * KCAL_PER_A_TO_KJ_PER_NM]
    return forces


def parse_gromacs_energy_xvg(xvg_path: Path) -> float:
    values = []
    for raw in read_text(xvg_path).splitlines():
        if raw.startswith(("#", "@")) or not raw.strip():
            continue
        parts = raw.split()
        if len(parts) >= 2:
            values.append(float(parts[1]))
    if not values:
        raise ValueError(f"Unable to parse GROMACS energy from {xvg_path}")
    return values[-1]


def parse_gromacs_forces_dump(dump_path: Path) -> dict[int, list[float]]:
    pattern = re.compile(r"f\[\s*(\d+)\]=\{([^,]+),\s*([^,]+),\s*([^}]+)\}")
    forces: dict[int, list[float]] = {}
    for line in read_text(dump_path).splitlines():
        match = pattern.search(line)
        if match:
            atom_id = int(match.group(1)) + 1
            forces[atom_id] = [float(match.group(2)), float(match.group(3)), float(match.group(4))]
    return forces


def run_mechanical_parity(args: argparse.Namespace, manifest: dict, out_root: Path) -> dict:
    work = out_root / "mechanical_parity"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    artifacts = manifest["artifacts"]
    lmp_data = Path(artifacts["lammps_data"])
    lmp_input = Path(artifacts["lammps_input"])
    top = Path(artifacts["gromacs_topology"])
    shutil.copy(lmp_data, work / "system.data")
    shutil.copy(lmp_input, work / "system.in")
    shutil.copy(top, work / "system.top")
    write_high_precision_gro(work / "system.data", work / "system.gro")
    write_text(work / "mechanical.mdp", mechanical_mdp())

    rendered_lmp = []
    for line in read_text(work / "system.in").splitlines():
        rendered_lmp.append("read_data system.data" if line.startswith("read_data") else line)
    rendered_lmp.extend(
        [
            "",
            "thermo_style custom step pe",
            "dump 1 all custom 1 lammps_forces.dump id fx fy fz",
            "dump_modify 1 sort id",
            "run 0",
            "",
        ]
    )
    write_text(work / "mechanical.in", "\n".join(rendered_lmp))
    run_command(["lmp", "-log", "lammps_run0.log", "-in", "mechanical.in"], work, work / "lammps.stdout", work / "lammps.stderr")
    lammps_energy_kj = parse_lammps_pe(work / "lammps_run0.log") * KCAL_TO_KJ
    lammps_forces = parse_lammps_forces(work / "lammps_forces.dump")

    run_command(
        [
            str(args.gmx),
            "grompp",
            "-f",
            "mechanical.mdp",
            "-c",
            "system.gro",
            "-p",
            "system.top",
            "-o",
            "mechanical.tpr",
            "-po",
            "mechanical_mdout.mdp",
            "-maxwarn",
            "1",
        ],
        work,
        work / "grompp.stdout",
        work / "grompp.stderr",
    )
    run_command(
        [str(args.gmx), "mdrun", "-s", "mechanical.tpr", "-rerun", "system.gro", "-deffnm", "mechanical", "-nt", "1", "-pin", "off", "-reprod"],
        work,
        work / "mdrun.stdout",
        work / "mdrun.stderr",
    )
    run_command(
        [str(args.gmx), "energy", "-f", "mechanical.edr", "-o", "gromacs_energy.xvg"],
        work,
        work / "energy.stdout",
        work / "energy.stderr",
        stdin_text="Potential\n0\n",
    )
    run_command([str(args.gmx), "dump", "-f", "mechanical.trr"], work, work / "gromacs_dump.stdout", work / "gromacs_dump.stderr")
    gromacs_energy_kj = parse_gromacs_energy_xvg(work / "gromacs_energy.xvg")
    gromacs_forces = parse_gromacs_forces_dump(work / "gromacs_dump.stdout")

    common_atoms = sorted(set(lammps_forces) & set(gromacs_forces))
    if len(common_atoms) != len(lammps_forces):
        raise ValueError("GROMACS/LAMMPS force atom sets differ.")
    sum_sq_diff = 0.0
    sum_sq_ref = 0.0
    max_abs_diff = 0.0
    max_ref_mag = 0.0
    max_atom_id = None
    for atom_id in common_atoms:
        lf = lammps_forces[atom_id]
        gf = gromacs_forces[atom_id]
        ref_mag = math.sqrt(sum(value * value for value in lf))
        max_ref_mag = max(max_ref_mag, ref_mag)
        for component_index in range(3):
            diff = abs(lf[component_index] - gf[component_index])
            if diff > max_abs_diff:
                max_abs_diff = diff
                max_atom_id = atom_id
            sum_sq_diff += diff * diff
            sum_sq_ref += lf[component_index] * lf[component_index]
    rms_abs_diff = math.sqrt(sum_sq_diff / (3 * len(common_atoms)))
    rms_ref = math.sqrt(sum_sq_ref / (3 * len(common_atoms)))
    rms_rel_diff = rms_abs_diff / rms_ref if rms_ref else math.inf
    max_rel_diff = max_abs_diff / max_ref_mag if max_ref_mag else math.inf
    energy_abs_diff = abs(gromacs_energy_kj - lammps_energy_kj)
    energy_rel_diff = energy_abs_diff / max(abs(lammps_energy_kj), 1.0)
    status = (
        energy_rel_diff <= args.energy_rel_threshold
        and rms_rel_diff <= args.force_rms_rel_threshold
        and max_rel_diff <= args.force_max_rel_threshold
        and max_abs_diff <= args.force_max_abs_threshold
    )
    report = {
        "system_id": manifest["system_id"],
        "scope": "GROMACS-vs-LAMMPS run-0 mechanical parity on the strict PCFF-qualified charged pair.",
        "protocol": {
            "gromacs": {
                "binary": str(args.gmx.resolve()),
                "coulombtype": "PME",
                "ewald_rtol": 1.0e-4,
                "pme_order": 4,
                "fourierspacing": 0.12,
                "vdw_type": "Cut-off",
                "cutoffs_nm": 0.9,
            },
            "lammps": manifest["lammps_styles"],
            "thresholds": {
                "energy_rel_diff_max": args.energy_rel_threshold,
                "force_rms_rel_diff_max": args.force_rms_rel_threshold,
                "force_max_rel_diff_max": args.force_max_rel_threshold,
                "force_max_abs_diff_kj_mol_nm_max": args.force_max_abs_threshold,
            },
        },
        "metrics": {
            "lammps_potential_energy_kj_mol": lammps_energy_kj,
            "gromacs_potential_energy_kj_mol": gromacs_energy_kj,
            "energy_abs_diff_kj_mol": energy_abs_diff,
            "energy_rel_diff": energy_rel_diff,
            "force_rms_abs_diff_kj_mol_nm": rms_abs_diff,
            "force_rms_reference_kj_mol_nm": rms_ref,
            "force_rms_rel_diff": rms_rel_diff,
            "force_max_abs_diff_kj_mol_nm": max_abs_diff,
            "force_max_reference_magnitude_kj_mol_nm": max_ref_mag,
            "force_max_rel_diff": max_rel_diff,
            "force_max_diff_atom_id": max_atom_id,
            "atom_count": len(common_atoms),
        },
        "status": "PASS" if status else "FAIL",
        "artifacts": {
            "root": repo_rel(work),
            "lammps_log": repo_rel(work / "lammps_run0.log"),
            "lammps_forces": repo_rel(work / "lammps_forces.dump"),
            "gromacs_mdout": repo_rel(work / "mechanical_mdout.mdp"),
            "gromacs_energy_xvg": repo_rel(work / "gromacs_energy.xvg"),
            "gromacs_force_dump": repo_rel(work / "gromacs_dump.stdout"),
        },
    }
    write_json(work / "mechanical_parity_report.json", report)
    return report


def parse_xvg(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw in read_text(path).splitlines():
        if raw.startswith(("#", "@")) or not raw.strip():
            continue
        rows.append([float(token) for token in raw.split()])
    return rows


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}


def parse_lammps_blocks(log_path: Path) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    header: list[str] | None = None
    rows: list[list[float]] = []
    for raw in read_text(log_path).splitlines():
        stripped = raw.strip()
        if stripped.startswith("Step "):
            if header and rows:
                blocks.append({"header": header, "rows": rows})
            header = stripped.split()
            rows = []
            continue
        if header is None:
            continue
        if stripped.startswith("Loop time"):
            if rows:
                blocks.append({"header": header, "rows": rows})
            header = None
            rows = []
            continue
        parts = stripped.split()
        if parts and parts[0].lstrip("-").isdigit():
            try:
                rows.append([float(token) for token in parts])
            except ValueError:
                pass
    if header and rows:
        blocks.append({"header": header, "rows": rows})
    return blocks


def run_structural_density_audit(npt_report_path: Path, out_root: Path) -> dict:
    source = load_json(npt_report_path)
    work = out_root / "structural_density_parity"
    work.mkdir(parents=True, exist_ok=True)
    gmx_rows = parse_xvg(Path(source["artifacts"]["gromacs_energy_xvg"]))
    lmp_blocks = parse_lammps_blocks(Path(source["artifacts"]["lammps_log"]))
    protocol = source["protocol"]
    analysis_start = max(0.0, float(protocol["duration_ps"]) - float(protocol["analysis_window_ps"]))
    gmx_window = [row for row in gmx_rows if row[0] >= analysis_start]
    lmp_rows = lmp_blocks[-1]["rows"]
    lmp_window = [row for row in lmp_rows if row[0] * 0.001 >= analysis_start]
    gmx_density = [row[5] for row in gmx_window]
    gmx_volume = [row[4] for row in gmx_window]
    gmx_temp = [row[2] for row in gmx_window]
    lmp_density = [row[7] * 1000.0 for row in lmp_window]
    lmp_volume = [row[6] / 1000.0 for row in lmp_window]
    lmp_temp = [row[1] for row in lmp_window]
    density_rel = abs(statistics.fmean(gmx_density) - statistics.fmean(lmp_density)) / statistics.fmean(lmp_density)
    volume_rel = abs(statistics.fmean(gmx_volume) - statistics.fmean(lmp_volume)) / statistics.fmean(lmp_volume)
    status = (
        density_rel <= float(protocol["thresholds"]["density_rel_diff_max"])
        and volume_rel <= float(protocol["thresholds"]["volume_rel_diff_max"])
    )
    report = {
        "system_id": source["system_id"],
        "scope": "M4 reanalysis of structural/density parity from the strict qualified paired NPT raw artifacts.",
        "source_report": repo_rel(npt_report_path),
        "protocol": protocol,
        "gromacs": {
            "density_kg_m3": summarize(gmx_density),
            "volume_nm3": summarize(gmx_volume),
            "temperature_k": summarize(gmx_temp),
        },
        "lammps": {
            "density_kg_m3": summarize(lmp_density),
            "volume_nm3": summarize(lmp_volume),
            "temperature_k": summarize(lmp_temp),
        },
        "parity_metrics": {
            "density_rel_diff": density_rel,
            "volume_rel_diff": volume_rel,
        },
        "status": "PASS" if status else "FAIL",
        "artifacts": {
            "gromacs_energy_xvg": repo_rel(Path(source["artifacts"]["gromacs_energy_xvg"])),
            "lammps_log": repo_rel(Path(source["artifacts"]["lammps_log"])),
        },
    }
    write_json(work / "structural_density_parity_report.json", report)
    return report


def run_transport_facing_audit(args: argparse.Namespace, manifest: dict, out_root: Path) -> dict:
    work = out_root / "transport_facing_parity"
    work.mkdir(parents=True, exist_ok=True)
    result = load_json(args.transport_result)
    manifest_fixture = Path(manifest["fixture_manifest"]).resolve()
    transport_scaffold = Path(result["scaffold_manifest"]).resolve()
    primary = [metric for metric, payload in result["observable_comparisons"].items() if metric in ("cation_diffusivity_cm2_s", "anion_diffusivity_cm2_s", "conductivity_cne_s_cm", "transference_ne")]
    primary_pass = {metric: bool(result["observable_comparisons"][metric].get("passes")) for metric in primary}
    replica_temperatures = [
        float(replica["analysis"]["average_temperature_k"])
        for replicas in result.get("per_layout_replicas", {}).values()
        for replica in replicas
        if "average_temperature_k" in replica.get("analysis", {})
    ]
    status = (
        result["status"] == "PASS"
        and result["preset"] == "charged-large"
        and manifest_fixture == transport_scaffold
        and all(primary_pass.values())
    )
    source_result = args.transport_result.resolve()
    expected_fresh_root = (out_root / "transport_facing_rerun").resolve()
    fresh_m4_rerun = expected_fresh_root in source_result.parents
    report = {
        "system_id": manifest["system_id"],
        "scope": "Fresh M4 transport-facing CPU/GPU observable parity rerun on the strict charged scaffold; this is not LAMMPS transport parity and not production readiness.",
        "source_result": repo_rel(args.transport_result),
        "fresh_m4_rerun": fresh_m4_rerun,
        "scaffold_manifest_matches_qualified_pair": manifest_fixture == transport_scaffold,
        "transport_result_status": result["status"],
        "gmx_binary": result.get("gmx_binary"),
        "gpu_support": result.get("gpu_support"),
        "run_settings": result.get("run_settings", {}),
        "temperature_caveat": {
            "replica_average_temperature_min_k": min(replica_temperatures) if replica_temperatures else None,
            "replica_average_temperature_max_k": max(replica_temperatures) if replica_temperatures else None,
            "interpretation": "Temperature is not used as thermophysical transport evidence for this short-horizon M4 smoke rerun.",
        },
        "primary_observables": primary_pass,
        "observable_comparisons": result["observable_comparisons"],
        "status": "PASS" if status and fresh_m4_rerun else "FAIL",
        "non_claimable_statement": "Do not use this as charged transport readiness; it only shows short-horizon transport-facing observable generation/comparison on the qualified scaffold.",
    }
    write_json(work / "transport_facing_parity_report.json", report)
    return report


def main() -> int:
    args = parse_args()
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = load_json(args.manifest)
    if manifest.get("pair_status") != "strict_pcff_qualified" or manifest.get("acpype_gaff2_dependency") is not False:
        raise ValueError("M4 requires a strict PCFF-qualified pair with no ACPYPE/GAFF2 dependency.")

    mechanical = run_mechanical_parity(args, manifest, out_root)
    structural = run_structural_density_audit(args.npt_report.resolve(), out_root)
    transport = run_transport_facing_audit(args, manifest, out_root)
    overall = mechanical["status"] == "PASS" and structural["status"] == "PASS" and transport["status"] == "PASS"
    inventory = {
        "schema_name": "pcff_charged_m4_strict_validation_inventory",
        "schema_version": 1,
        "system_id": manifest["system_id"],
        "qualified_pair_manifest": repo_rel(args.manifest),
        "status": "PASS" if overall else "FAIL",
        "component_status": {
            "mechanical_parity": mechanical["status"],
            "structural_density_parity": structural["status"],
            "transport_facing_parity": transport["status"],
        },
        "reports": {
            "mechanical_parity": repo_rel(out_root / "mechanical_parity" / "mechanical_parity_report.json"),
            "structural_density_parity": repo_rel(out_root / "structural_density_parity" / "structural_density_parity_report.json"),
            "transport_facing_parity": repo_rel(out_root / "transport_facing_parity" / "transport_facing_parity_report.json"),
        },
        "sha256_manifest": repo_rel(out_root / "sha256_manifest.txt"),
        "claimable_statement": "M4 strict charged validation passes only if all three component reports pass on the strict-PCFF-qualified charged scaffold.",
        "non_claimable_statement": "M4 does not establish broad PCFF chemistry, LAMMPS-vs-GROMACS transport parity, charged transport readiness, or production readiness.",
    }
    write_json(out_root / "m4_strict_validation_inventory.json", inventory)
    write_sha256_manifest(out_root, out_root / "sha256_manifest.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
