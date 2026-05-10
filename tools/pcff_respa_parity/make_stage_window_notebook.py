#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_NOTEBOOK = REPO / "output/jupyter-notebook/polygen_pcff_rrespa_lammps_gromacs_benchmark.ipynb"
DEFAULT_OUT_DIR = REPO / "tmp/jupyter-notebook"


def parse_bool(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {text!r}")


def py_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return repr(value)


def replace_assignment(source: list[str], name: str, value: Any) -> bool:
    replacement = f"{name} = {py_literal(value)}\n"
    for index, line in enumerate(source):
        stripped = line.lstrip()
        if stripped.startswith(f"{name} ="):
            indent = line[: len(line) - len(stripped)]
            source[index] = indent + replacement
            return True
    return False


def find_config_cell(cells: list[dict[str, Any]]) -> dict[str, Any]:
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        if "RUN_GROMACS_CPU" in text and "GMX_STAGE_START_NAME" in text:
            return cell
    raise RuntimeError("could not find benchmark config cell")


def validate_code_cells(nb: dict[str, Any]) -> None:
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell {index}")


def make_notebook(args: argparse.Namespace) -> Path:
    notebook = Path(args.notebook).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lane = args.lane
    if lane not in {"cpu", "gpu"}:
        raise ValueError(f"unsupported lane: {lane}")

    out_name = args.output_name
    if out_name is None:
        stage_slug = args.stage_start.replace("/", "_").replace(" ", "_")
        out_name = f"{lane}_{stage_slug}_stage_window.ipynb"
    out_path = out_dir / out_name

    nb = json.loads(notebook.read_text(encoding="utf-8"))
    config_cell = find_config_cell(nb["cells"])
    source = list(config_cell.get("source", []))
    stop_stage = args.stage_stop if args.stage_stop is not None else args.stage_start
    replacements: dict[str, Any] = {
        "RUN_LAMMPS": False,
        "RUN_GROMACS_CPU": lane == "cpu",
        "RUN_GROMACS_GPU": lane == "gpu",
        "RUN_PRODUCTION": args.run_production,
        "RUN_PRODUCTION_ONLY": args.run_production_only,
        "GMX_STAGE_START_NAME": args.stage_start,
        "GMX_STAGE_STOP_NAME": stop_stage,
        "ENABLE_ANALYSIS_TRAJECTORIES": args.analysis_trajectories,
        "CLEAN_ENABLED_LANE_OUTPUTS_BEFORE_RUN": args.clean_lane,
    }
    missing = [name for name, value in replacements.items() if not replace_assignment(source, name, value)]
    if missing:
        raise RuntimeError(f"missing assignments in config cell: {', '.join(missing)}")
    config_cell["source"] = source
    validate_code_cells(nb)
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def execute_notebook(path: Path, timeout: int) -> Path:
    output_name = path.with_suffix(".out.ipynb").name
    log_path = path.with_suffix(".nbconvert.log")
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(path),
        "--output",
        output_name,
        "--output-dir",
        str(path.parent),
        f"--ExecutePreprocessor.timeout={timeout}",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"nbconvert failed with return code {proc.returncode}; see {log_path}")
    return path.parent / output_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and optionally execute a benchmark notebook for one stage window.")
    parser.add_argument("--notebook", default=str(DEFAULT_NOTEBOOK))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--lane", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--stage-start", required=True)
    parser.add_argument("--stage-stop", default=None)
    parser.add_argument("--run-production", type=parse_bool, default=False)
    parser.add_argument("--run-production-only", type=parse_bool, default=False)
    parser.add_argument("--analysis-trajectories", type=parse_bool, default=False)
    parser.add_argument("--clean-lane", type=parse_bool, default=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = make_notebook(args)
    print(path)
    if args.execute:
        executed = execute_notebook(path, args.timeout)
        print(executed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
