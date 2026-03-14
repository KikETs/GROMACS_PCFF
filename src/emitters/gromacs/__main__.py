from __future__ import annotations

import argparse
from pathlib import Path

from .engine import dumps_manifest, emit_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit deterministic GROMACS topology files from the PT1-PT5 typed IR chain.")
    parser.add_argument("path", help="Input structure path")
    parser.add_argument("--input-format", default=None, help="Override input format")
    parser.add_argument("--source-id", default=None, help="Override source id")
    parser.add_argument("--out", default=None, help="Output directory for forcefield_pcff.itp, molecule.itp, and topol.top")
    parser.add_argument("--dry-run", action="store_true", help="Render and validate without writing files")
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Compare the rendered bundle against an existing output directory",
    )
    args = parser.parse_args()

    manifest = emit_file(
        args.path,
        input_format=args.input_format,
        source_id=args.source_id,
        out_dir=None if args.out is None else Path(args.out),
        dry_run=args.dry_run,
        validate_existing=args.validate_existing,
    )
    print(dumps_manifest(manifest), end="")


if __name__ == "__main__":
    main()
