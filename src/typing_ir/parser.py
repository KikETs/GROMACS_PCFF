from __future__ import annotations

from pathlib import Path

from .formats import parse_path
from .ir import build_ir


def parse_file(
    path: str | Path,
    *,
    input_format: str | None = None,
    source_id: str | None = None,
) -> dict:
    input_path = Path(path)
    parsed = parse_path(input_path, input_format=input_format)
    source_label = source_id if source_id is not None else input_path.name
    return build_ir(
        parsed,
        input_format=parsed["input_format"],
        source_id=source_label,
        source_bytes=input_path.read_bytes(),
    )
