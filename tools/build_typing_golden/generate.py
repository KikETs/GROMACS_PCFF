from __future__ import annotations

import argparse
from pathlib import Path

from common import CORPUS_ROOT, TypingGoldenError, build_manifest, dump_json, stage_corpus, validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate, stage, or validate the PT0 PCFF typing golden corpus."
    )
    parser.add_argument(
        "--corpus-root",
        default=str(CORPUS_ROOT),
        help="Root directory containing the typing golden corpus.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument(
        "--out",
        default=None,
        help="Output path for the regenerated manifest. Default: <corpus-root>/corpus_manifest.json",
    )

    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--out", required=True, help="Output directory for a staged deterministic bundle.")
    stage_parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Case id to include. Repeat to select multiple cases. Default: all cases.",
    )

    subparsers.add_parser("validate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus_root = Path(args.corpus_root).resolve()

    try:
        if args.command == "manifest":
            out_path = Path(args.out).resolve() if args.out is not None else corpus_root / "corpus_manifest.json"
            dump_json(out_path, build_manifest(corpus_root))
        elif args.command == "stage":
            stage_corpus(Path(args.out).resolve(), corpus_root, args.cases)
        else:
            validate_manifest(corpus_root)
    except TypingGoldenError as error:
        raise SystemExit(f"build_typing_golden: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
