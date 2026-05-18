from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
M5_ROOT = REPO_ROOT / "tests" / "reference_results" / "m5"
M6_RESPA_ROOT = REPO_ROOT / "tests" / "reference_results" / "m6_respa"
CORPUS_ROOT = REPO_ROOT / "testdata" / "lammps_golden" / "systems"

sys.path.insert(0, str(REPO_ROOT / "tools" / "generate_lammps_golden"))
from common import dump_json, parse_dump_custom, parse_thermo_table  # noqa: E402


ANGSTROM_TO_NM = 0.1
RESPA_OUTER_TIMESTEP_FS = 2.0
DEFAULT_RESPA_OUTER_STEPS = 5
RESPA_LEVEL_FACTORS = [2, 2]
RESPA_SCHEDULE = {
    "bond_level": 1,
    "angle_level": 1,
    "dihedral_level": 1,
    "improper_level": 1,
    "kspace_level": 3,
    "inner_level": 1,
    "middle_level": 2,
    "outer_level": 3,
    "inner_off_angstrom": 3.0,
    "inner_on_angstrom": 4.5,
    "outer_on_angstrom": 6.0,
    "outer_off_angstrom": 8.0,
}

REFERENCE_TOLERANCES = {
    "step0_potential_kcal_mol": 0.02,
    "initial_total_kcal_mol": 0.02,
    "final_total_kcal_mol": 0.02,
    "total_energy_drift_abs_kcal_mol": 0.005,
    "total_energy_span_kcal_mol": 0.005,
    "polymer_end_to_end_nm": 1e-4,
    "polymer_rg_nm": 1e-4,
    "ion_distance_nm": 1e-4,
    "step0_virial_pressure_xx_atm": 5.0,
    "step0_virial_pressure_yy_atm": 5.0,
    "step0_virial_pressure_zz_atm": 5.0,
    "step0_virial_pressure_xy_atm": 5.0,
    "step0_virial_pressure_xz_atm": 5.0,
    "step0_virial_pressure_yz_atm": 5.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen LAMMPS run_style respa parity references.")
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="System id to regenerate. Default: all respa systems.",
    )
    parser.add_argument(
        "--out",
        default=str(M6_RESPA_ROOT),
        help="Output directory for generated respa reference artifacts.",
    )
    parser.add_argument(
        "--workdir",
        default=str(REPO_ROOT / "output" / "tmp" / "pcff_respa_parity"),
        help="Temporary directory used for LAMMPS runs.",
    )
    parser.add_argument(
        "--lammps-cmd",
        default="lmp",
        help="LAMMPS executable for generating the frozen run_style respa reference.",
    )
    parser.add_argument(
        "--outer-steps",
        type=int,
        default=DEFAULT_RESPA_OUTER_STEPS,
        help="Number of outer r-RESPA steps to run.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def parse_lammps_data(path: Path) -> dict:
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    box = {}

    lines = path.read_text(encoding="utf-8").splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith("xlo xhi"):
            xlo, xhi, *_ = line.split()
            box["x"] = (float(xlo), float(xhi))
            continue
        if line.endswith("ylo yhi"):
            ylo, yhi, *_ = line.split()
            box["y"] = (float(ylo), float(yhi))
            continue
        if line.endswith("zlo zhi"):
            zlo, zhi, *_ = line.split()
            box["z"] = (float(zlo), float(zhi))
            continue
        if line in {"Masses", "Atoms # full", "Bonds", "Angles", "Dihedrals", "Impropers"}:
            current_section = line
            sections[current_section] = []
            continue
        if current_section is not None:
            sections[current_section].append(line)

    atoms = []
    for line in sections.get("Atoms # full", []):
        atom_id, molecule, atom_type, charge, x, y, z = line.split()
        atoms.append(
            {
                "id": int(atom_id),
                "molecule": int(molecule),
                "type": int(atom_type),
                "charge": float(charge),
                "x": float(x),
                "y": float(y),
                "z": float(z),
            }
        )
    atoms.sort(key=lambda atom: atom["id"])
    return {"box": box, "atoms": atoms}


def box_lengths_nm(parsed_data: dict) -> tuple[float, float, float]:
    return tuple(
        (parsed_data["box"][axis][1] - parsed_data["box"][axis][0]) * ANGSTROM_TO_NM for axis in ("x", "y", "z")
    )


def minimum_image(delta: float, box_length: float) -> float:
    if box_length <= 0:
        return delta
    return delta - box_length * round(delta / box_length)


def vector_with_minimum_image(
    a: tuple[float, float, float], b: tuple[float, float, float], box_nm: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(minimum_image(a[d] - b[d], box_nm[d]) for d in range(3))


def norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def structural_metrics(system_id: str, atoms: list[dict], parsed_data: dict) -> dict[str, float]:
    box_nm = box_lengths_nm(parsed_data)
    atom_by_id = {atom["id"]: atom for atom in atoms}
    num_polymer_atoms = 6 if system_id == "small_oligomer" else 8
    polymer_atoms = [atom_by_id[atom_id] for atom_id in range(1, num_polymer_atoms + 1)]
    polymer_coords = [
        (
            atom["x"] * ANGSTROM_TO_NM,
            atom["y"] * ANGSTROM_TO_NM,
            atom["z"] * ANGSTROM_TO_NM,
        )
        for atom in polymer_atoms
    ]

    end_to_end_vector = vector_with_minimum_image(polymer_coords[-1], polymer_coords[0], box_nm)
    metrics = {"polymer_end_to_end_nm": norm(end_to_end_vector)}

    unwrapped = [polymer_coords[0]]
    for coordinate in polymer_coords[1:]:
        delta = vector_with_minimum_image(coordinate, unwrapped[-1], box_nm)
        unwrapped.append(tuple(unwrapped[-1][d] + delta[d] for d in range(3)))
    center_of_mass = tuple(sum(coord[d] for coord in unwrapped) / len(unwrapped) for d in range(3))
    metrics["polymer_rg_nm"] = math.sqrt(
        sum(sum((coord[d] - center_of_mass[d]) ** 2 for d in range(3)) for coord in unwrapped) / len(unwrapped)
    )

    if system_id == "small_salt_polymer_box":
        ion_a = atom_by_id[9]
        ion_b = atom_by_id[10]
        ion_a_coord = (ion_a["x"] * ANGSTROM_TO_NM, ion_a["y"] * ANGSTROM_TO_NM, ion_a["z"] * ANGSTROM_TO_NM)
        ion_b_coord = (ion_b["x"] * ANGSTROM_TO_NM, ion_b["y"] * ANGSTROM_TO_NM, ion_b["z"] * ANGSTROM_TO_NM)
        metrics["ion_distance_nm"] = norm(vector_with_minimum_image(ion_b_coord, ion_a_coord, box_nm))

    return metrics


def render_respa_nve(system_meta: dict, outer_steps: int) -> str:
    fields = "step time pe ke etotal temp c_vir[1] c_vir[2] c_vir[3] c_vir[4] c_vir[5] c_vir[6]"
    config = system_meta["expected_observables"]["nve_drift"]
    return "\n".join(
        [
            "log respa_nve.log",
            "include system.in",
            "reset_timestep 0",
            f"run_style respa 3 {RESPA_LEVEL_FACTORS[0]} {RESPA_LEVEL_FACTORS[1]} "
            f"bond {RESPA_SCHEDULE['bond_level']} "
            f"angle {RESPA_SCHEDULE['angle_level']} "
            f"dihedral {RESPA_SCHEDULE['dihedral_level']} "
            f"improper {RESPA_SCHEDULE['improper_level']} "
            f"inner {RESPA_SCHEDULE['inner_level']} {RESPA_SCHEDULE['inner_off_angstrom']} {RESPA_SCHEDULE['inner_on_angstrom']} "
            f"middle {RESPA_SCHEDULE['middle_level']} {RESPA_SCHEDULE['outer_on_angstrom']} {RESPA_SCHEDULE['outer_off_angstrom']} "
            f"outer {RESPA_SCHEDULE['outer_level']} "
            f"kspace {RESPA_SCHEDULE['kspace_level']}",
            f"timestep {RESPA_OUTER_TIMESTEP_FS}",
            f"velocity all create {config['initial_temperature_K']} {config['velocity_seed']} mom yes rot yes dist gaussian",
            "fix integ all nve",
            "compute vir all pressure NULL virial",
            "thermo 1",
            f"thermo_style custom {fields}",
            "thermo_modify flush yes",
            "dump respa_dump all custom 1 respa_nve.dump id type q x y z vx vy vz",
            "dump_modify respa_dump sort id",
            f"run {outer_steps}",
            "undump respa_dump",
            "unfix integ",
        ]
    )


def reference_tolerances(summary: dict) -> dict[str, float]:
    tolerances = {}
    for key in summary["reference"]["nve"]:
        if key not in REFERENCE_TOLERANCES:
            raise KeyError(f"No frozen M6 tolerance declared for {key}")
        tolerances[key] = REFERENCE_TOLERANCES[key]
    return tolerances


def reference_tsv(summary: dict) -> str:
    lines = ["# schema_version 1", f"system {summary['system_id']}"]
    for key, value in sorted(summary["reference"]["nve"].items()):
        lines.append(f"reference nve {key} {value:.12f}")
    for key, value in sorted(summary["tolerance"]["nve"].items()):
        lines.append(f"tolerance nve {key} {value:.12f}")
    return "\n".join(lines) + "\n"


def generate_system(system_id: str, out_root: Path, workdir: Path, lammps_cmd: str, outer_steps: int) -> None:
    source_root = CORPUS_ROOT / system_id
    system_meta = load_json(source_root / "system.json")
    parsed_data = parse_lammps_data(source_root / "lammps" / "system.data")

    stage_dir = workdir / system_id
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    shutil.copy2(source_root / "lammps" / "system.data", stage_dir / "system.data")
    shutil.copy2(source_root / "lammps" / "system.in", stage_dir / "system.in")
    write_text(stage_dir / "respa_nve.in", render_respa_nve(system_meta, outer_steps) + "\n")

    subprocess.run(
        ["/bin/bash", "-lc", f"OMPI_MCA_plm=isolated {lammps_cmd} -in respa_nve.in"],
        cwd=stage_dir,
        check=True,
    )

    trace = parse_thermo_table(
        stage_dir / "respa_nve.log",
        ["step", "time", "pe", "ke", "etotal", "temp", "c_vir[1]", "c_vir[2]", "c_vir[3]", "c_vir[4]", "c_vir[5]", "c_vir[6]"],
    )
    final_frame = parse_dump_custom(stage_dir / "respa_nve.dump")[-1]
    metrics = structural_metrics(system_id, final_frame["atoms"], parsed_data)

    output_dir = out_root / system_id
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(M5_ROOT / system_id / "topol.top", output_dir / "topol.top")
    shutil.copy2(M5_ROOT / system_id / "initial_nve.gro", output_dir / "initial_nve.gro")

    normalized = {
        "schema_version": 1,
        "system_id": system_id,
        "schedule": {
            "levels": 3,
            "loop_factors": RESPA_LEVEL_FACTORS,
            "outer_timestep_fs": RESPA_OUTER_TIMESTEP_FS,
            "outer_steps": outer_steps,
            **RESPA_SCHEDULE,
        },
        "trace": trace,
        "final_frame": final_frame,
    }
    dump_json(output_dir / "nve_respa.json", normalized)

    virial_metrics = {
        "step0_virial_pressure_xx_atm": trace[0]["c_vir[1]"],
        "step0_virial_pressure_yy_atm": trace[0]["c_vir[2]"],
        "step0_virial_pressure_zz_atm": trace[0]["c_vir[3]"],
        "step0_virial_pressure_xy_atm": trace[0]["c_vir[4]"],
        "step0_virial_pressure_xz_atm": trace[0]["c_vir[5]"],
        "step0_virial_pressure_yz_atm": trace[0]["c_vir[6]"],
    }

    summary = {
        "schema_version": 1,
        "system_id": system_id,
        "mode": "lammps_run_style_respa",
        "sources": {
            "lammps_input": f"testdata/lammps_golden/systems/{system_id}/lammps/system.in",
            "lammps_topology": f"testdata/lammps_golden/systems/{system_id}/lammps/system.data",
            "gromacs_topology": f"tests/reference_results/m6_respa/{system_id}/topol.top",
            "gromacs_initial_state": f"tests/reference_results/m6_respa/{system_id}/initial_nve.gro",
        },
        "schedule": normalized["schedule"],
        "reference": {
            "nve": {
                "step0_potential_kcal_mol": trace[0]["pe"],
                "initial_total_kcal_mol": trace[0]["etotal"],
                "final_total_kcal_mol": trace[-1]["etotal"],
                "total_energy_drift_abs_kcal_mol": abs(trace[-1]["etotal"] - trace[0]["etotal"]),
                "total_energy_span_kcal_mol": max(frame["etotal"] for frame in trace)
                - min(frame["etotal"] for frame in trace),
                **virial_metrics,
                **metrics,
            }
        },
        "tolerance": {
            "nve": {},
        },
        "notes": [
            "Reference schedule is run_style respa 3 2 2 with bonded terms all on level 1, kspace on level 3, and class2 inner-middle-outer pair splitting.",
            "GROMACS comparison input reuses the already-frozen M5 topol.top and initial_nve.gro so that the respa harness isolates scheduler/runtime differences instead of re-opening topology conversion risk.",
            "Step-0 virial pressure components come from LAMMPS compute pressure NULL virial and remain in real-units atmospheres.",
        ],
        "unresolved_items": [
            "LAMMPS special_bonds 1-4 terms are embedded in the pair style, while the GROMACS exact path maps them through listed pair14 interactions. This harness is intended to expose any residual mismatch.",
            "GROMACS exact r-RESPA scheduler/integrator parity remains an open item. The current fork contains both a legacy md-based exact path and a dedicated md-vv prototype path, and the harness is intended to expose whichever one is active.",
        ],
    }
    summary["tolerance"]["nve"] = reference_tolerances(summary)
    dump_json(output_dir / "reference_summary.json", summary)
    write_text(output_dir / "reference_summary.tsv", reference_tsv(summary))


def main() -> None:
    args = parse_args()
    out_root = Path(args.out).resolve()
    workdir = Path(args.workdir).resolve()
    systems = args.systems or ["small_oligomer", "small_salt_polymer_box"]

    out_root.mkdir(parents=True, exist_ok=True)
    for system_id in systems:
        generate_system(system_id, out_root, workdir, args.lammps_cmd, args.outer_steps)


if __name__ == "__main__":
    main()
