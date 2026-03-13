from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .ir import dumps_ir, write_ir
from .parser import parse_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TYPING_GOLDEN = REPO_ROOT / "testdata" / "typing_golden"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse structure files into the PT1 typed IR schema.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse")
    parse_parser.add_argument("input_path")
    parse_parser.add_argument("--format", dest="input_format", default=None)
    parse_parser.add_argument("--source-id", default=None)
    parse_parser.add_argument("--out", default=None)

    export_parser = subparsers.add_parser("export-typing-golden")
    export_parser.add_argument(
        "--corpus-root",
        default=str(DEFAULT_TYPING_GOLDEN),
        help="Path to the typing_golden corpus root.",
    )
    export_parser.add_argument(
        "--out-root",
        default=None,
        help="Optional alternate root for generated examples. Defaults to in-place writes.",
    )
    export_parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Case id to export. Repeat to select multiple cases.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "parse":
        ir = parse_file(
            args.input_path,
            input_format=args.input_format,
            source_id=args.source_id,
        )
        if args.out is None:
            print(dumps_ir(ir), end="")
        else:
            write_ir(Path(args.out), ir)
        return 0

    export_typing_golden(
        corpus_root=Path(args.corpus_root).resolve(),
        out_root=None if args.out_root is None else Path(args.out_root).resolve(),
        case_ids=args.cases,
    )
    return 0


def export_typing_golden(
    *,
    corpus_root: Path,
    out_root: Path | None,
    case_ids: list[str] | None,
) -> list[Path]:
    cases_root = corpus_root / "cases"
    case_dirs = sorted(path for path in cases_root.iterdir() if path.is_dir())
    selected = None if case_ids is None else set(case_ids)

    if out_root is not None:
        if out_root.exists():
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

    written_paths = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        if selected is not None and case_id not in selected:
            continue
        input_path = case_dir / "inputs" / "structure.mol"
        try:
            source_id = input_path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            source_id = input_path.name
        ir = parse_file(input_path, input_format="mol_v2000", source_id=source_id)
        if out_root is None:
            output_path = case_dir / "examples" / "typed_system.json"
        else:
            output_path = out_root / "cases" / case_id / "examples" / "typed_system.json"
        write_ir(output_path, ir)
        written_paths.append(output_path)
    return written_paths


if __name__ == "__main__":
    raise SystemExit(main())
