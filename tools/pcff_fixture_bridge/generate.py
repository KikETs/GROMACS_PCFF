from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    CORPUS_ROOT,
    BridgeError,
    build_typed_ir,
    dump_json,
    iter_system_records,
    render_gromacs_topology,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate typed PCFF fixture IR and deterministic GROMACS topologies from frozen LAMMPS fixtures."
    )
    parser.add_argument(
        "--corpus-root",
        default=str(CORPUS_ROOT),
        help="Root directory containing corpus_manifest.json and system fixture directories.",
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="System id to export. Repeat to select multiple systems. Default: all systems.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("typed-ir")
    subparsers.add_parser("export-gromacs")
    return parser.parse_args()


def export_typed_ir(out_root: Path, corpus_root: Path, systems: list[str] | None) -> list[dict]:
    records = iter_system_records(systems, corpus_root)
    manifest_records = []
    for record in records:
        typed_ir = build_typed_ir(record, corpus_root)
        system_out = out_root / record["id"]
        dump_json(system_out / "typed_system.json", typed_ir)
        manifest_records.append(
            {
                "system_id": record["id"],
                "typed_ir": str((system_out / "typed_system.json").relative_to(out_root)),
            }
        )
    return manifest_records


def export_gromacs(out_root: Path, corpus_root: Path, systems: list[str] | None) -> list[dict]:
    records = iter_system_records(systems, corpus_root)
    manifest_records = []
    for record in records:
        typed_ir = build_typed_ir(record, corpus_root)
        system_out = out_root / record["id"]
        dump_json(system_out / "typed_system.json", typed_ir)
        topology_path = system_out / "topol.top"
        topology_path.parent.mkdir(parents=True, exist_ok=True)
        topology_path.write_text(render_gromacs_topology(typed_ir), encoding="utf-8")
        manifest_records.append(
            {
                "system_id": record["id"],
                "typed_ir": str((system_out / "typed_system.json").relative_to(out_root)),
                "gromacs_topology": str(topology_path.relative_to(out_root)),
            }
        )
    return manifest_records


def main() -> int:
    args = parse_args()
    out_root = Path(args.out).resolve()
    corpus_root = Path(args.corpus_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        if args.command == "typed-ir":
            systems = export_typed_ir(out_root, corpus_root, args.systems)
        else:
            systems = export_gromacs(out_root, corpus_root, args.systems)
    except BridgeError as error:
        raise SystemExit(f"pcff_fixture_bridge: {error}") from error

    dump_json(
        out_root / "bridge_manifest.json",
        {
            "schema_version": 1,
            "command": args.command,
            "corpus_root": str(corpus_root),
            "systems": systems,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
