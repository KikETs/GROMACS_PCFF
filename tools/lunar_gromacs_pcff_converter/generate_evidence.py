from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_ROOT = REPO_ROOT / "tools" / "pcff_fixture_bridge"
sys.path.insert(0, str(BRIDGE_ROOT))

from common import (  # noqa: E402
    ANGSTROM_TO_NM,
    CORPUS_ROOT,
    build_typed_ir,
    dump_json,
    parse_lammps_data,
    parse_lammps_input,
    render_gromacs_topology,
)


DEFAULT_OUT_ROOT = REPO_ROOT / "tests" / "reference_results" / "lunar_gromacs_pcff_converter"
CASE_ID = "small_oligomer"
CASE_DIR_NAME = "small_oligomer_polymer_only"
CASE_RECORD = {"id": CASE_ID, "path": f"systems/{CASE_ID}"}

SMOKE_MDP = """integrator = md
nsteps = 1
dt = 0.001
cutoff-scheme = Verlet
vdw-type = Cut-off
rvdw = 0.9
coulombtype = PME
rcoulomb = 0.9
pbc = xyz
nstlist = 10
rlist = 0.9
verlet-buffer-tolerance = 0.005
constraints = none
gen-vel = yes
gen-temp = 300
gen-seed = 12345
ld-seed = 12345
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate public evidence for the narrow LUNAR/LAMMPS-data to "
            "GROMACS PCFF converter claim."
        )
    )
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Evidence output root.")
    parser.add_argument("--corpus-root", default=str(CORPUS_ROOT), help="LAMMPS golden corpus root.")
    parser.add_argument(
        "--gmx",
        default=None,
        help="GROMACS executable. Defaults to build/bin/gmx, then PATH gmx.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_gmx(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    build_gmx = REPO_ROOT / "build" / "bin" / "gmx"
    if build_gmx.is_file():
        return build_gmx
    path_gmx = shutil.which("gmx")
    if path_gmx:
        return Path(path_gmx).resolve()
    raise SystemExit("No GROMACS executable found. Pass --gmx explicitly.")


def source_path(corpus_root: Path, relative: str) -> str:
    path = corpus_root / "systems" / CASE_ID / "lammps" / relative
    try:
        return rel(path)
    except ValueError:
        return str(path)


def local_atom_names(parsed_data: dict) -> dict[int, str]:
    atoms_by_molecule: dict[int, list[dict]] = {}
    for atom in parsed_data["atoms"]:
        atoms_by_molecule.setdefault(atom["molecule_id"], []).append(atom)
    names = {}
    for atoms in atoms_by_molecule.values():
        for local_index, atom in enumerate(sorted(atoms, key=lambda item: item["id"]), start=1):
            names[atom["id"]] = f"A{local_index}"
    return names


def render_gro(parsed_data: dict) -> str:
    box_x = (parsed_data["box"]["x"]["hi"] - parsed_data["box"]["x"]["lo"]) * ANGSTROM_TO_NM
    box_y = (parsed_data["box"]["y"]["hi"] - parsed_data["box"]["y"]["lo"]) * ANGSTROM_TO_NM
    box_z = (parsed_data["box"]["z"]["hi"] - parsed_data["box"]["z"]["lo"]) * ANGSTROM_TO_NM
    atom_names = local_atom_names(parsed_data)

    lines = ["Generated from LAMMPS data for converter grompp smoke", f"{len(parsed_data['atoms']):>5d}"]
    for atom in parsed_data["atoms"]:
        x = atom["x_angstrom"] * ANGSTROM_TO_NM
        y = atom["y_angstrom"] * ANGSTROM_TO_NM
        z = atom["z_angstrom"] * ANGSTROM_TO_NM
        residue_id = atom["molecule_id"] % 100000
        lines.append(
            f"{residue_id:>5d}{'OLI':<5s}{atom_names[atom['id']]:>5s}{atom['id']:>5d}"
            f"{x:15.7f}{y:15.7f}{z:15.7f}"
        )
    lines.append(f"{box_x:15.7f}{box_y:15.7f}{box_z:15.7f}")
    return "\n".join(lines) + "\n"


def topology_sections(topology_text: str) -> list[str]:
    sections = []
    for line in topology_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            sections.append(stripped.strip("[] "))
    return sections


def count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


def write_case_readme(case_dir: Path) -> None:
    (case_dir / "README.md").write_text(
        "\n".join(
            [
                "# small_oligomer_polymer_only",
                "",
                "This is the narrow public converter smoke case.",
                "",
                "Boundary:",
                "",
                "- Input: repository-local LAMMPS `data` plus `system.in` fixture.",
                "- Converter: `tools/pcff_fixture_bridge/common.py` parser and topology renderer.",
                "- Claim: non-salt, single-molecule oligomer topology can be parsed, mapped, emitted, and accepted by `grompp`.",
                "- Non-claim: this is not charged salt, ion, dense electrolyte, or general LUNAR auto-conversion support.",
                "",
                "Main artifacts:",
                "",
                "- `typed_system.json`: parser and mapping IR with source line provenance.",
                "- `parser_summary.json`: parser contract and section counts.",
                "- `mapping_summary.json`: Class2 mapping contract and generated 1-4 pair evidence.",
                "- `topol.top`: emitted GROMACS PCFF topology.",
                "- `system.gro`: coordinate file generated only for `grompp` smoke validation.",
                "- `grompp/grompp_smoke_report.json`: `grompp` exit status, warning count, and generated-output hashes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser_summary(corpus_root: Path, parsed_data: dict, parsed_input: dict) -> dict:
    first_atom = parsed_data["atoms"][0]
    first_bond = parsed_data["bonds"][0]
    return {
        "schema_name": "lunar_gromacs_pcff_converter_parser_summary",
        "schema_version": 1,
        "status": "PASS",
        "case_id": CASE_ID,
        "parser_functions": [
            "tools/pcff_fixture_bridge/common.py::parse_lammps_data",
            "tools/pcff_fixture_bridge/common.py::parse_lammps_input",
        ],
        "input_files": {
            "system_data": source_path(corpus_root, "system.data"),
            "system_in": source_path(corpus_root, "system.in"),
        },
        "accepted_lammps_subset": {
            "units": parsed_input["styles"]["units"],
            "atom_style": parsed_input["styles"]["atom_style"],
            "pair_style": parsed_input["styles"]["pair_style"]["kind"],
            "pair_modify": parsed_input["styles"]["pair_modify"],
            "bond_style": parsed_input["styles"]["bond_style"],
            "angle_style": parsed_input["styles"]["angle_style"],
            "dihedral_style": parsed_input["styles"]["dihedral_style"],
            "improper_style": parsed_input["styles"]["improper_style"],
            "special_bonds": parsed_input["styles"]["special_bonds"],
        },
        "parsed_counts": {
            "atoms": parsed_data["header_counts"]["atoms"],
            "bonds": parsed_data["header_counts"]["bonds"],
            "angles": parsed_data["header_counts"]["angles"],
            "dihedrals": parsed_data["header_counts"]["dihedrals"],
            "impropers": parsed_data["header_counts"]["impropers"],
            "atom_types": parsed_data["type_counts"]["atom types"],
            "bond_types": parsed_data["type_counts"]["bond types"],
            "angle_types": parsed_data["type_counts"]["angle types"],
            "dihedral_types": parsed_data["type_counts"]["dihedral types"],
        },
        "source_line_samples": {
            "first_atom": first_atom["source"],
            "first_bond": first_bond["source"],
            "pair_style": parsed_input["style_sources"]["pair_style"],
            "special_bonds": parsed_input["style_sources"]["special_bonds"],
        },
        "unsupported_by_this_parser_claim": [
            "arbitrary LUNAR output not matching the frozen subset",
            "cross pair_coeff in the public polymer-only claim",
            "non-Class2 bonded styles",
            "multi-file include processing",
        ],
    }


def build_mapping_summary(typed_ir: dict) -> dict:
    template = typed_ir["molecule_templates"][0]
    return {
        "schema_name": "lunar_gromacs_pcff_converter_mapping_summary",
        "schema_version": 1,
        "status": "PASS",
        "case_id": CASE_ID,
        "mapping_functions": [
            "tools/pcff_fixture_bridge/common.py::gromacs_bond_params",
            "tools/pcff_fixture_bridge/common.py::gromacs_angle_params",
            "tools/pcff_fixture_bridge/common.py::gromacs_dihedral_params",
            "tools/pcff_fixture_bridge/common.py::render_gromacs_topology",
        ],
        "unit_mapping": {
            "distance": "angstrom to nm",
            "energy": "kcal/mol to kJ/mol",
            "charge": "e preserved",
            "pair_model": "GROMACS [ defaults ] comb-rule 4, rep-pow 9",
        },
        "interaction_mapping": {
            "bond_class2_count": len(template["bonds"]),
            "angle_class2_count": len(template["angles"]),
            "dihedral_class2_count": len(template["dihedrals"]),
            "improper_class2_count": len(template["impropers"]),
            "generated_pair14_count": len(template["generated_pairs"]),
            "generated_pair14_rule": typed_ir["diagnostics"]["generated_pair_rule"],
        },
        "molecule_scope": {
            "template_count": len(typed_ir["molecule_templates"]),
            "instance_count": len(typed_ir["molecule_instances"]),
            "template_name": template["name"],
            "nrexcl": template["nrexcl"],
            "no_separate_ion_or_salt_molecules": True,
            "net_charge_e": round(sum(atom["charge_e"] for atom in template["atoms"]), 8),
        },
        "coefficient_sources": {
            "atom_type_sources": [atom_type["pair_coeff"]["source"] for atom_type in typed_ir["atom_types"]],
            "first_bond_type_source": typed_ir["bond_types"][0]["source"],
            "first_angle_main_source": typed_ir["angle_types"][0]["main"]["source"],
            "first_dihedral_main_source": typed_ir["dihedral_types"][0]["main"]["source"],
        },
        "unsupported_by_this_mapping_claim": [
            "charged salt/ion component mapping",
            "dense electrolyte multi-component assembly",
            "broad PCFF chemistry outside this fixture",
            "production ensemble or transport readiness",
        ],
    }


def build_emission_summary(case_root: Path, topology_text: str, typed_ir: dict) -> dict:
    topol_path = case_root / "topol.top"
    return {
        "schema_name": "lunar_gromacs_pcff_converter_emission_summary",
        "schema_version": 1,
        "status": "PASS",
        "case_id": CASE_ID,
        "emitter_function": "tools/pcff_fixture_bridge/common.py::render_gromacs_topology",
        "emitted_topology": rel(topol_path),
        "topology_sha256": sha256_text(topology_text),
        "topology_line_count": len(topology_text.splitlines()),
        "sections": topology_sections(topology_text),
        "molecule_records": typed_ir["molecule_instances"],
        "grompp_required_for_viability_claim": True,
        "note": "This emission artifact is not treated as proof until paired with grompp/grompp_smoke_report.json.",
    }


def run_grompp(case_root: Path, gmx: Path) -> dict:
    grompp_root = case_root / "grompp"
    grompp_root.mkdir(parents=True, exist_ok=True)
    smoke_mdp = grompp_root / "smoke.mdp"
    stdout_path = grompp_root / "grompp.stdout"
    stderr_path = grompp_root / "grompp.stderr"
    tpr_path = grompp_root / "smoke.tpr"
    mdout_path = grompp_root / "mdout.mdp"

    smoke_mdp.write_text(SMOKE_MDP, encoding="utf-8")
    for path in (stdout_path, stderr_path, tpr_path, mdout_path):
        if path.exists():
            path.unlink()

    command = [
        str(gmx),
        "grompp",
        "-f",
        str(smoke_mdp),
        "-c",
        str(case_root / "system.gro"),
        "-p",
        str(case_root / "topol.top"),
        "-o",
        str(tpr_path),
        "-po",
        str(mdout_path),
    ]
    env = dict(os.environ)
    env["GMX_NO_QUOTES"] = "1"
    result = subprocess.run(command, cwd=case_root, text=True, capture_output=True, env=env, check=False)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")

    warning_count = count_pattern(result.stderr, r"\bWARNING\s+\d+\b")
    note_count = count_pattern(result.stderr, r"\bNOTE\s+\d+\b")
    status = "PASS" if result.returncode == 0 and warning_count == 0 and tpr_path.exists() else "FAIL"
    report = {
        "schema_name": "lunar_gromacs_pcff_converter_grompp_smoke",
        "schema_version": 1,
        "status": status,
        "case_id": CASE_ID,
        "scope": "polymer-only non-salt converter smoke; not a production MD validation",
        "command": command,
        "working_directory": rel(case_root),
        "returncode": result.returncode,
        "warning_count": warning_count,
        "note_count": note_count,
        "stdout_path": rel(stdout_path),
        "stderr_path": rel(stderr_path),
        "generated_outputs": {
            "smoke_tpr": {
                "path": rel(tpr_path),
                "exists": tpr_path.exists(),
                "sha256": sha256_file(tpr_path) if tpr_path.exists() else None,
            },
            "mdout_mdp": {
                "path": rel(mdout_path),
                "exists": mdout_path.exists(),
                "sha256": sha256_file(mdout_path) if mdout_path.exists() else None,
            },
        },
        "success_criteria": {
            "returncode_zero": result.returncode == 0,
            "warning_count_zero": warning_count == 0,
            "tpr_generated": tpr_path.exists(),
        },
    }
    dump_json(grompp_root / "grompp_smoke_report.json", report)
    return report


def build_contract(out_root: Path, smoke_report: dict) -> dict:
    case_root = out_root / "cases" / CASE_DIR_NAME
    out_root_arg = rel(out_root)
    return {
        "schema_name": "lunar_gromacs_pcff_converter_contract",
        "schema_version": 1,
        "claim_status_as_of": date.today().isoformat(),
        "public_claim": (
            "The public converter claim is limited to the repository-local "
            "`small_oligomer` LAMMPS data/input fixture: parser -> Class2 mapping "
            "-> GROMACS PCFF topology emission -> grompp smoke passes."
        ),
        "supported_case": {
            "case_id": CASE_ID,
            "case_label": CASE_DIR_NAME,
            "artifact_root": rel(case_root),
            "input_data": "testdata/lammps_golden/systems/small_oligomer/lammps/system.data",
            "input_settings": "testdata/lammps_golden/systems/small_oligomer/lammps/system.in",
            "single_molecule_non_salt": True,
            "net_charge_e": 0.0,
        },
        "conversion_chain": [
            {
                "stage": "parser",
                "artifact": rel(case_root / "parser_summary.json"),
                "primary_code": "tools/pcff_fixture_bridge/common.py::parse_lammps_data",
            },
            {
                "stage": "mapping",
                "artifact": rel(case_root / "mapping_summary.json"),
                "primary_code": "tools/pcff_fixture_bridge/common.py::build_typed_ir",
            },
            {
                "stage": "emission",
                "artifact": rel(case_root / "emission_summary.json"),
                "primary_code": "tools/pcff_fixture_bridge/common.py::render_gromacs_topology",
            },
            {
                "stage": "grompp_smoke",
                "artifact": rel(case_root / "grompp" / "grompp_smoke_report.json"),
                "status": smoke_report["status"],
            },
        ],
        "unsupported": [
            "general LUNAR output beyond the frozen LAMMPS data/input subset",
            "charged salt or ion conversion as part of this converter claim",
            "dense polymer-electrolyte conversion support",
            "arbitrary PCFF chemistry or external pcff.frc auto-merge",
            "cross pair_coeff coverage in this public smoke case",
            "grompp success as production MD, ensemble parity, or transport readiness",
        ],
        "charged_ion_extension_gate": {
            "status": "closed_to_claim_until_evidenced",
            "minimum_artifacts_required": [
                "charged parser summary with explicit ion/salt component provenance",
                "charged mapping summary covering ion/salt parameters and exclusions",
                "charged emitted topology and coordinate bundle",
                "charged grompp smoke report with returncode 0 and warning policy stated",
                "separate validation report that does not reuse polymer-only smoke as charged evidence",
            ],
        },
        "reproduction_command": (
            "python3 tools/lunar_gromacs_pcff_converter/generate_evidence.py "
            f"--out-root {out_root_arg}"
        ),
    }


def build_support_matrix(out_root: Path, smoke_report: dict) -> dict:
    case_root = out_root / "cases" / CASE_DIR_NAME
    return {
        "schema_name": "lunar_gromacs_pcff_converter_support_matrix",
        "schema_version": 1,
        "claim_status_as_of": date.today().isoformat(),
        "status_definitions": {
            "exact": "Checked-in artifacts directly support the statement.",
            "unsupported": "The current artifacts explicitly do not support the statement.",
            "gated": "The path needs separate public artifacts before any support claim.",
        },
        "strongest_public_boundary": (
            "Repository-local `small_oligomer` LAMMPS data/input fixture, single "
            "non-salt oligomer molecule, Class2 bond/angle/dihedral plus "
            "lj/class2/coul/long sixth-power topology, grompp smoke only."
        ),
        "items": [
            {
                "id": "parser.lammps_data_input_subset",
                "status": "exact",
                "claimable_statement": "The frozen `small_oligomer` LAMMPS data/input subset parses with source-line provenance.",
                "evidence": [
                    rel(case_root / "typed_system.json"),
                    rel(case_root / "parser_summary.json"),
                ],
            },
            {
                "id": "mapping.class2_polymer_only_small_oligomer",
                "status": "exact",
                "claimable_statement": "Class2 bond/angle/dihedral parameters and generated 1-4 pairs are mapped for the supported polymer-only smoke case.",
                "evidence": [
                    rel(case_root / "mapping_summary.json"),
                    rel(case_root / "typed_system.json"),
                ],
            },
            {
                "id": "emission.gromacs_pcff_topology",
                "status": "exact",
                "claimable_statement": "The mapped IR emits a deterministic GROMACS PCFF topology for the supported case.",
                "evidence": [
                    rel(case_root / "topol.top"),
                    rel(case_root / "emission_summary.json"),
                ],
            },
            {
                "id": "viability.grompp_smoke",
                "status": "exact" if smoke_report["status"] == "PASS" else "unsupported",
                "claimable_statement": "The emitted topology is accepted by grompp for the supported polymer-only smoke case.",
                "evidence": [
                    rel(case_root / "system.gro"),
                    rel(case_root / "grompp" / "smoke.mdp"),
                    rel(case_root / "grompp" / "grompp_smoke_report.json"),
                    rel(case_root / "grompp" / "grompp.stderr"),
                    rel(case_root / "grompp" / "grompp.stdout"),
                ],
            },
            {
                "id": "charged_ion_extension",
                "status": "gated",
                "claimable_statement": "No charged/ion converter support is claimed from these polymer-only artifacts.",
                "evidence": [
                    rel(out_root / "conversion_contract.json"),
                ],
            },
            {
                "id": "general_lunar_conversion",
                "status": "unsupported",
                "claimable_statement": "General LUNAR data-file conversion remains unsupported by this evidence set.",
                "evidence": [
                    rel(out_root / "conversion_contract.json"),
                ],
            },
        ],
    }


def write_top_readme(out_root: Path) -> None:
    out_root_arg = rel(out_root)
    (out_root / "README.md").write_text(
        "\n".join(
            [
                "# LUNAR/LAMMPS Data to GROMACS PCFF Converter Evidence",
                "",
                "This directory is the public evidence bundle for the narrow converter claim.",
                "",
                "Supported now:",
                "",
                "- `small_oligomer_polymer_only`: frozen repository-local LAMMPS `system.data` + `system.in` to GROMACS PCFF `topol.top`.",
                "- Evidence chain: parser summary, mapping summary, emitted topology, and `grompp` smoke report.",
                "",
                "Not supported by this bundle:",
                "",
                "- general LUNAR output",
                "- charged salt or ion conversion",
                "- dense polymer-electrolyte support",
                "- production MD, ensemble parity, or transport readiness",
                "- database-wide conversion success for `simulation-trajectory-aggregate_aligned.csv`",
                "",
                "Optional database scope audit:",
                "",
                "- `database_scope/`: records whether the provided aligned CSV database has a public artifact chain.",
                "- `database_lunar_smoke/`: records parser -> mapping -> emission -> `grompp` smoke artifacts for currently available LUNAR PCFF single-chain `.data` inputs.",
                "- Current audited database status remains separate from the single-case converter smoke claim and does not make a database-wide claim.",
                "",
                "Regenerate:",
                "",
                "```bash",
                "python3 tools/lunar_gromacs_pcff_converter/generate_evidence.py \\",
                f"  --out-root {out_root_arg}",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def clean_owned_outputs(out_root: Path) -> None:
    owned_paths = [
        out_root / "cases" / CASE_DIR_NAME,
        out_root / "conversion_contract.json",
        out_root / "support_matrix.json",
        out_root / "README.md",
    ]
    for path in owned_paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    (out_root / "cases").mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root).resolve()
    corpus_root = Path(args.corpus_root).resolve()
    gmx = resolve_gmx(args.gmx)

    clean_owned_outputs(out_root)
    case_root = out_root / "cases" / CASE_DIR_NAME
    case_root.mkdir(parents=True)

    parsed_data = parse_lammps_data(corpus_root / "systems" / CASE_ID / "lammps" / "system.data")
    parsed_input = parse_lammps_input(corpus_root / "systems" / CASE_ID / "lammps" / "system.in")
    typed_ir = build_typed_ir(CASE_RECORD, corpus_root)
    topology_text = render_gromacs_topology(typed_ir)

    write_case_readme(case_root)
    dump_json(case_root / "typed_system.json", typed_ir)
    dump_json(case_root / "parser_summary.json", build_parser_summary(corpus_root, parsed_data, parsed_input))
    dump_json(case_root / "mapping_summary.json", build_mapping_summary(typed_ir))
    (case_root / "topol.top").write_text(topology_text, encoding="utf-8")
    (case_root / "system.gro").write_text(render_gro(parsed_data), encoding="utf-8")
    dump_json(case_root / "emission_summary.json", build_emission_summary(case_root, topology_text, typed_ir))

    smoke_report = run_grompp(case_root, gmx)
    dump_json(out_root / "conversion_contract.json", build_contract(out_root, smoke_report))
    dump_json(out_root / "support_matrix.json", build_support_matrix(out_root, smoke_report))
    write_top_readme(out_root)
    return 0 if smoke_report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
