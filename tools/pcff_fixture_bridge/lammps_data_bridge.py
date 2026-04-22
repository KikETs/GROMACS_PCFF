from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from common import (
    BridgeError,
    build_typed_ir_from_lammps_data,
    dump_json,
    parse_lammps_data,
    render_gromacs_gro_from_lammps_data,
    render_gromacs_topology,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a LUNAR/LAMMPS PCFF Class2 data file with inline coefficients and emit GROMACS files."
    )
    parser.add_argument("--data", required=True, help="LAMMPS data file containing Masses, Coeffs, and topology sections.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--system-id", default=None, help="System id recorded in typed_system.json and [ system ].")
    parser.add_argument("--display-name", default=None, help="Human-readable display name.")
    parser.add_argument(
        "--category",
        default="polymer_box",
        choices=["polymer_box", "oligomer", "toy", "generic"],
        help="Labeling mode for molecule/residue names. polymer_box labels multi-atom molecules as POL.",
    )
    parser.add_argument("--pair-style", default="lj/class2/coul/long", help="Assumed LAMMPS pair_style kind.")
    parser.add_argument(
        "--pair-style-arg",
        action="append",
        default=None,
        help="Assumed pair_style argument. Repeat to preserve cutoffs in provenance.",
    )
    parser.add_argument("--pair-modify", default="mix sixthpower", help="Assumed LAMMPS pair_modify setting.")
    parser.add_argument(
        "--special-bonds",
        default="lj/coul 0.0 0.0 1.0 angle no dihedral no",
        help="Assumed LAMMPS special_bonds setting for GROMACS exclusion/pair generation.",
    )
    parser.add_argument("--kspace-style", default="pppm 1.0e-6", help="Assumed LAMMPS kspace_style setting.")
    parser.add_argument(
        "--no-kspace-style",
        action="store_true",
        help="Record no kspace_style assumption.",
    )
    parser.add_argument(
        "--no-shift-to-origin",
        action="store_true",
        help="Do not shift coordinates by box lower bounds when writing system.gro.",
    )
    return parser.parse_args()


def file_record(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "line_count": len(path.read_text(encoding="utf-8").splitlines()),
    }


def main() -> int:
    args = parse_args()
    data_path = Path(args.data).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        typed_ir = build_typed_ir_from_lammps_data(
            data_path,
            system_id=args.system_id,
            display_name=args.display_name,
            category=args.category,
            pair_style=args.pair_style,
            pair_style_args=args.pair_style_arg,
            pair_modify=args.pair_modify,
            special_bonds=args.special_bonds,
            kspace_style=None if args.no_kspace_style else args.kspace_style,
        )
        parsed_data = parse_lammps_data(data_path)
        topology_text = render_gromacs_topology(typed_ir)
        gro_text = render_gromacs_gro_from_lammps_data(
            parsed_data,
            title=f"Generated from {data_path.name}",
            residue_name="POL" if args.category == "polymer_box" else "MOL",
            shift_to_origin=not args.no_shift_to_origin,
        )
    except BridgeError as error:
        raise SystemExit(f"lammps_data_bridge: {error}") from error

    typed_path = out_root / "typed_system.json"
    top_path = out_root / "topol.top"
    gro_path = out_root / "system.gro"
    dump_json(typed_path, typed_ir)
    top_path.write_text(topology_text, encoding="utf-8")
    gro_path.write_text(gro_text, encoding="utf-8")

    dump_json(
        out_root / "bridge_manifest.json",
        {
            "schema_version": 1,
            "command": "lammps-data-export-gromacs",
            "source_data": str(data_path),
            "assumptions": {
                "pair_style": {"kind": args.pair_style, "args": [] if args.pair_style_arg is None else args.pair_style_arg},
                "pair_modify": args.pair_modify,
                "special_bonds_input": args.special_bonds,
                "special_bonds_effective": typed_ir["styles"]["special_bonds"]["value"],
                "kspace_style": None if args.no_kspace_style else args.kspace_style,
                "coordinates_shifted_to_box_origin": not args.no_shift_to_origin,
            },
            "counts": {
                "atoms": len(parsed_data["atoms"]),
                "bonds": len(parsed_data["bonds"]),
                "angles": len(parsed_data["angles"]),
                "dihedrals": len(parsed_data["dihedrals"]),
                "impropers": len(parsed_data["impropers"]),
                "molecule_templates": len(typed_ir["molecule_templates"]),
                "molecule_instances": len(typed_ir["molecule_instances"]),
            },
            "outputs": {
                "typed_system": file_record(typed_path),
                "gromacs_topology": file_record(top_path),
                "gromacs_coordinates": file_record(gro_path),
            },
            "claim_boundary": (
                "This is a topology/coordinate bridge from a PCFF Class2 LAMMPS data file with inline coefficients. "
                "It does not by itself prove LAMMPS-vs-GROMACS energy, force, density, or transport parity."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
