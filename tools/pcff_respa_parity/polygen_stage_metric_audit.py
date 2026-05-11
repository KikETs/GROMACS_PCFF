from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NOTEBOOK = REPO_ROOT / "output/jupyter-notebook/polygen_pcff_rrespa_lammps_gromacs_benchmark.ipynb"
DEFAULT_OUT_ROOT = REPO_ROOT / "output/polygen_pcff_gromacs_initial_em_notebook"
DEFAULT_AUDIT_OUT = DEFAULT_OUT_ROOT / "current_stage_metric_audit"
DEFAULT_GMX = REPO_ROOT / "build_gateb_cuda/bin/gmx"
DEFAULT_LMP = Path("/home/kiket/anaconda3/envs/MD/bin/lmp")

ATM_TO_BAR = 1.01325
KCAL_TO_KJ = 4.184
AMU_PER_NM3_TO_G_CM3 = 0.00166053906660
SYSTEM_MASS_AMU = 62860.6404999947
KJ_MOL_NM3_TO_BAR = 16.605390671738468

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit PolyGen LAMMPS/GROMACS equilibration outputs by stage, including "
            "runtime-signature freshness and available thermodynamic metrics."
        )
    )
    parser.add_argument("--notebook", default=str(DEFAULT_NOTEBOOK), help="Benchmark notebook used as current config source.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Notebook output root.")
    parser.add_argument("--audit-out", default=str(DEFAULT_AUDIT_OUT), help="Directory for CSV/JSON/Markdown reports.")
    parser.add_argument("--gmx", default=str(DEFAULT_GMX), help="GROMACS executable used for gmx energy extraction.")
    parser.add_argument("--lmp", default=str(DEFAULT_LMP), help="LAMMPS executable used for endpoint run-0 probes.")
    parser.add_argument(
        "--run-lammps-endpoint-probes",
        action="store_true",
        help=(
            "Run cheap LAMMPS run-0 probes from existing restart endpoints to collect pressure/virial tensors "
            "and fill scalar endpoint metrics missing from the thermo trace."
        ),
    )
    parser.add_argument(
        "--lanes",
        nargs="*",
        default=["gromacs_cpu_openmp", "gromacs_gpu_hybrid"],
        help="GROMACS lane directories under --out-root.",
    )
    parser.add_argument("--no-edr", action="store_true", help="Skip gmx energy extraction from .edr files.")
    parser.add_argument(
        "--same-start-probe-roots",
        nargs="*",
        default=None,
        help=(
            "Optional eq02 same-start propagation probe directories or summary CSV files. "
            "When omitted, no same-start probe CSVs are loaded; pass explicit current probe roots to avoid "
            "mixing stale diagnostic runs into the audit."
        ),
    )
    parser.add_argument(
        "--include-all-stages",
        action="store_true",
        help=(
            "Audit the full eq+production schedule even when the notebook is currently set to "
            "RUN_PRODUCTION_ONLY=True. This only changes audit stage selection and expected "
            "runtime signatures; it does not edit the notebook or rerun simulations."
        ),
    )
    parser.add_argument(
        "--relax-nonphysics-signature",
        action="store_true",
        help=(
            "Ignore signature fields that only change audit/run selection, and accept the current "
            "pcff_class2_traceguard performance tag suffix as physics-equivalent to the previous tag. "
            "Use only for reading already-completed outputs without rerunning full stages."
        ),
    )
    parser.add_argument(
        "--expected-pme-order",
        type=int,
        choices=(4, 5),
        default=None,
        help=(
            "Override the notebook-derived gmx_pme_order expected in runtime signatures. "
            "Use this to audit strict PME-order-5 CPU parity outputs separately from GPU "
            "speed-mode PME-order-4 outputs."
        ),
    )
    parser.add_argument(
        "--prod-duration-ps",
        type=float,
        default=None,
        help=(
            "Override the production duration used only for expected GROMACS production chunk selection. "
            "Use 20000 for a 20 ns extension audit without editing the notebook config."
        ),
    )
    parser.add_argument(
        "--extra-lammps-log",
        action="append",
        default=[],
        help=(
            "Additional LAMMPS stdout log to parse after equil_from_em.stdout.log and "
            "prod_from_relaxed.stdout.log. May be passed more than once."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def safe_literal_eval(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.JoinedStr):
        return "".join(str(safe_literal_eval(value, env)) for value in node.values)
    if isinstance(node, ast.FormattedValue):
        return safe_literal_eval(node.value, env)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -safe_literal_eval(node.operand, env)
    if isinstance(node, ast.Tuple):
        return tuple(safe_literal_eval(elt, env) for elt in node.elts)
    if isinstance(node, ast.List):
        return [safe_literal_eval(elt, env) for elt in node.elts]
    if isinstance(node, ast.Set):
        return {safe_literal_eval(elt, env) for elt in node.elts}
    if isinstance(node, ast.Dict):
        return {safe_literal_eval(k, env): safe_literal_eval(v, env) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.IfExp):
        branch = node.body if safe_condition_eval(node.test, env) else node.orelse
        return safe_literal_eval(branch, env)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [safe_literal_eval(arg, env) for arg in node.args]
        if node.func.id == "dict" and len(args) == 1 and not node.keywords:
            return dict(args[0])
        if node.func.id == "list" and len(args) == 1 and not node.keywords:
            return list(args[0])
        if node.func.id == "tuple" and len(args) == 1 and not node.keywords:
            return tuple(args[0])
        if node.func.id == "set" and len(args) == 1 and not node.keywords:
            return set(args[0])
    if isinstance(node, (ast.BoolOp, ast.Compare)):
        return safe_condition_eval(node, env)
    if isinstance(node, ast.BinOp):
        left = safe_literal_eval(node.left, env)
        right = safe_literal_eval(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError(f"unsupported expression: {ast.dump(node, include_attributes=False)}")


def safe_condition_eval(node: ast.AST, env: dict[str, Any]) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not safe_condition_eval(node.operand, env)
    if isinstance(node, ast.BoolOp):
        values = [safe_condition_eval(value, env) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
    if isinstance(node, ast.Compare):
        left = safe_literal_eval(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            right = safe_literal_eval(comparator, env)
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.In):
                ok = left in right
            elif isinstance(op, ast.NotIn):
                ok = left not in right
            elif isinstance(op, ast.Is):
                ok = left is right
            elif isinstance(op, ast.IsNot):
                ok = left is not right
            else:
                raise ValueError(f"unsupported comparison: {ast.dump(op, include_attributes=False)}")
            if not ok:
                return False
            left = right
        return True
    return bool(safe_literal_eval(node, env))


def load_notebook_config(notebook: Path) -> dict[str, Any]:
    nb = load_json(notebook)
    source = "".join(nb["cells"][1]["source"])
    tree = ast.parse(source)
    env: dict[str, Any] = {}

    def apply_assignments(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            targets: list[ast.expr] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            elif isinstance(node, ast.If):
                try:
                    branch = node.body if safe_condition_eval(node.test, env) else node.orelse
                except Exception:
                    continue
                apply_assignments(list(branch))
                continue
            else:
                continue
            if value is None:
                continue
            try:
                evaluated = safe_literal_eval(value, env)
            except Exception:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    env[target.id] = evaluated

    apply_assignments(list(tree.body))
    return env


def parse_lammps_pppm_by_input(log_path: Path) -> dict[str, dict[str, Any]]:
    if not log_path.exists():
        return {}
    cmd_re = re.compile(r"\$ .* -in ((?:resume_inputs/)?(?:lammps_equil_[^\s]+|lammps_prod_chunk\d{4})\.in)")
    g_re = re.compile(r"G vector \(1/distance\) =\s*([0-9.eE+-]+)")
    grid_re = re.compile(r"grid =\s*(\d+)\s+(\d+)\s+(\d+)")
    order_re = re.compile(r"stencil order =\s*(\d+)")
    current_input: str | None = None
    current: dict[str, Any] = {}
    values: dict[str, dict[str, Any]] = {}

    def flush() -> None:
        if current_input is not None and current:
            values[current_input] = dict(current)

    for raw in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        cmd = cmd_re.search(raw)
        if cmd:
            flush()
            current_input = cmd.group(1)
            current = {}
            continue
        if current_input is None:
            continue
        if "g_vector_inv_a" not in current:
            g = g_re.search(raw)
            if g:
                current["g_vector_inv_a"] = float(g.group(1))
                continue
        if "grid" not in current:
            grid = grid_re.search(raw)
            if grid:
                current["grid"] = tuple(int(grid.group(i)) for i in range(1, 4))
                continue
        if "order" not in current:
            order = order_re.search(raw)
            if order:
                current["order"] = int(order.group(1))
                continue
    flush()
    return values


def lammps_input_to_gmx_stage_key(input_name: str) -> str | None:
    stem = Path(input_name).stem
    if stem == "lammps_equil_00_initial_minimize":
        return "gromacs_initial_em"
    if stem == "lammps_equil_03_minimize":
        return "eq03_pre_2fs_minimize"
    if stem == "lammps_equil_12_npt_avg_cell_1200ps":
        return "eq12_npt_1200ps"
    if stem == "prod_from_relaxed":
        return "prod01_nvt_10000ps_chunk0001"
    mprod = re.match(r"lammps_prod_chunk(\d{4})$", stem)
    if mprod:
        return f"prod01_nvt_10000ps_chunk{mprod.group(1)}"
    m = re.match(r"lammps_equil_(\d{2})_(.+?)_chunk(\d{4})$", stem)
    if not m:
        return None
    index = int(m.group(1))
    chunk = m.group(3)
    stage_map = {
        1: "eq01_nvt_50ps",
        2: "eq02_npt_compress_100ps",
        4: "eq04_npt_compress_500ps",
        5: "eq05_npt_hold_hi_400ps",
        6: "eq06_npt_decompress_600ps",
        7: "eq07_npt_heat_400ps",
        8: "eq08_npt_cool_400ps",
        9: "eq09_npt_compress_300ps",
        10: "eq10_npt_decompress_300ps",
        11: "eq11_nvt_800ps",
        13: "eq13_nvt_fixed_volume_1000ps",
    }
    base = stage_map.get(index)
    if base is None:
        return None
    if index == 1:
        return base
    return f"{base}_chunk{chunk}"


def gmx_stage_key_to_stem(stage_key: str) -> str:
    if stage_key == "gromacs_initial_em":
        return "00_gromacs_initial_em"
    m = re.match(r"eq(\d{2})_(.+)", stage_key)
    if m:
        return f"{m.group(1)}_{stage_key}"
    mprod = re.match(r"prod01_(.+)", stage_key)
    if mprod:
        return f"14_{stage_key}"
    return stage_key


def gmx_stem_to_stage_key(stem: str) -> str:
    if stem == "00_gromacs_initial_em":
        return "gromacs_initial_em"
    m = re.match(r"^\d{2}_(.+)$", stem)
    return m.group(1) if m else stem


def expected_gmx_stage_stems(config: dict[str, Any]) -> list[str]:
    """Return only stems that belong to the current notebook schedule.

    Old notebook runs left aggregate files such as 04_eq04_npt_compress_500ps
    next to the current chunked schedule.  Counting those files as stale makes
    the audit report a tooling artifact, not a parity signal.
    """

    run_production = bool(config.get("RUN_PRODUCTION"))
    run_production_only = bool(config.get("RUN_PRODUCTION_ONLY"))
    run_initial_em = bool(config.get("RUN_GMX_INITIAL_EM", True))
    match_lammps_chunks = bool(config.get("GMX_MATCH_LAMMPS_RUN_CHUNKS", False))
    unsplit = set(config.get("GMX_UNSPLIT_LAMMPS_CHUNK_STAGE_NAMES", set()))
    chunk_steps = int(config.get("LAMMPS_RESUME_CHUNK_STEPS", 100_000) or 100_000)

    base_stages = [
        (1, "eq01_nvt_50ps", "equil", "md", 50.0, 0.5),
        (2, "eq02_npt_compress_100ps", "equil", "md", 100.0, 0.5),
        (3, "eq03_pre_2fs_minimize", "equil", "em", 0.0, 2.0),
        (4, "eq04_npt_compress_500ps", "equil", "md", 500.0, 2.0),
        (5, "eq05_npt_hold_hi_400ps", "equil", "md", 400.0, 2.0),
        (6, "eq06_npt_decompress_600ps", "equil", "md", 600.0, 2.0),
        (7, "eq07_npt_heat_400ps", "equil", "md", 400.0, 2.0),
        (8, "eq08_npt_cool_400ps", "equil", "md", 400.0, 2.0),
        (9, "eq09_npt_compress_300ps", "equil", "md", 300.0, 2.0),
        (10, "eq10_npt_decompress_300ps", "equil", "md", 300.0, 2.0),
        (11, "eq11_nvt_800ps", "equil", "md", 800.0, 2.0),
        (12, "eq12_npt_1200ps", "equil", "md", 1200.0, 2.0),
        (13, "eq13_nvt_fixed_volume_1000ps", "equil", "md", 1000.0, 2.0),
        (
            14,
            "prod01_nvt_10000ps",
            "prod",
            "md",
            float(config.get("_AUDIT_PROD_DURATION_PS_OVERRIDE", 10000.0) or 10000.0),
            2.0,
        ),
    ]

    stage_items: list[tuple[int, str, str]] = []
    expanded_index = 0
    for base_index, name, phase, kind, duration_ps, timestep_fs in base_stages:
        if run_production_only and phase != "prod":
            continue
        if not run_production and phase == "prod":
            continue
        if match_lammps_chunks and kind == "md" and name not in unsplit:
            total_steps = int(math.floor(duration_ps * 1000.0 / timestep_fs))
            n_chunks = max(1, math.ceil(total_steps / max(1, chunk_steps)))
        else:
            n_chunks = 1
        for chunk_index in range(1, n_chunks + 1):
            expanded_index += 1
            stage_key = name if n_chunks == 1 else f"{name}_chunk{chunk_index:04d}"
            stem = f"{base_index:02d}_{stage_key}"
            stage_items.append((expanded_index, stage_key, stem))

    def selector_matches(item: tuple[int, str, str], selector: Any) -> bool:
        expanded_idx, stage_key, stem = item
        selector_text = str(selector)
        return selector_text in {stage_key, stem, f"{expanded_idx:02d}_{stage_key}"}

    start = config.get("GMX_STAGE_START_NAME")
    stop = config.get("GMX_STAGE_STOP_NAME")
    selected = stage_items
    if start is not None:
        matches = [pos for pos, item in enumerate(selected) if selector_matches(item, start)]
        selected = selected[matches[0] :] if matches else []
    if stop is not None:
        matches = [pos for pos, item in enumerate(selected) if selector_matches(item, stop)]
        selected = selected[: matches[0] + 1] if matches else []

    stems = [stem for _, _, stem in selected]
    if run_initial_em and not run_production_only:
        stems.insert(0, "00_gromacs_initial_em")
    return stems


def metric_mean(rows: list[dict[str, float]], key: str) -> float | None:
    vals = [row[key] for row in rows if key in row and math.isfinite(row[key])]
    return statistics.fmean(vals) if vals else None


def parse_lammps_equil_trace(log_path: Path) -> dict[str, dict[str, Any]]:
    if not log_path.exists():
        return {}
    cmd_re = re.compile(
        r"\$ .* -in ((?:resume_inputs/)?(?:lammps_equil_[^\s]+|lammps_prod_chunk\d{4})\.in)"
    )
    segments: list[dict[str, Any]] = []
    current_input: str | None = None
    header: list[str] | None = None
    rows: list[dict[str, float]] = []
    completed = False
    box_volume_nm3: float | None = None

    def flush() -> None:
        if current_input and header and rows:
            stage_key = lammps_input_to_gmx_stage_key(current_input)
            if stage_key:
                segments.append(
                    {
                        "input": current_input,
                        "stage_key": stage_key,
                        "header": list(header),
                        "rows": list(rows),
                        "completed": completed,
                        "box_volume_nm3": box_volume_nm3,
                    }
                )

    for raw in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        cmd = cmd_re.search(raw)
        if cmd:
            flush()
            current_input = cmd.group(1)
            header = None
            rows = []
            completed = False
            box_volume_nm3 = None
            continue
        if current_input is None:
            continue
        if "orthogonal box =" in raw:
            vals = [float(value) for value in FLOAT_RE.findall(raw)]
            if len(vals) >= 6:
                xlo, ylo, zlo, xhi, yhi, zhi = vals[:6]
                box_volume_nm3 = (xhi - xlo) * (yhi - ylo) * (zhi - zlo) / 1000.0
            continue
        fields = raw.split()
        if fields and fields[0] == "Step" and ("v_time" in fields or "Fmax" in fields):
            header = fields
            rows = []
            continue
        if "Loop time of" in raw or "Minimization stats:" in raw:
            completed = True
        if not header:
            continue
        vals = FLOAT_RE.findall(raw)
        if len(vals) == len(header):
            try:
                rows.append({name: float(value) for name, value in zip(header, vals)})
            except ValueError:
                pass
    flush()

    def metrics_from_rows(
        stage_key: str,
        input_name: str,
        rows: list[dict[str, float]],
        completed: bool,
        box_volume_nm3: float | None = None,
    ) -> dict[str, Any]:
        final = rows[-1]
        metrics: dict[str, Any] = {
            "engine": "lammps",
            "stage_key": stage_key,
            "input": input_name,
            "sample_count": len(rows),
            "completed": completed,
        }
        if "v_time" in final:
            metrics["time_ps_final"] = final["v_time"]
            mean = metric_mean(rows, "v_time")
            if mean is not None:
                metrics["time_ps_mean"] = mean
        if "Press" in final:
            metrics["pressure_bar_final"] = final["Press"] * ATM_TO_BAR
            mean = metric_mean(rows, "Press")
            if mean is not None:
                metrics["pressure_bar_mean"] = mean * ATM_TO_BAR
        add_lammps_pressure_tensor_metrics(metrics, final, rows)
        if "Volume" in final:
            metrics["volume_nm3_final"] = final["Volume"] / 1000.0
            mean = metric_mean(rows, "Volume")
            if mean is not None:
                metrics["volume_nm3_mean"] = mean / 1000.0
        elif box_volume_nm3 is not None and math.isfinite(box_volume_nm3):
            metrics["volume_nm3_final"] = box_volume_nm3
            metrics["volume_nm3_mean"] = box_volume_nm3
        if "v_sysdensity" in final:
            metrics["density_g_cm3_final"] = final["v_sysdensity"]
            mean = metric_mean(rows, "v_sysdensity")
            if mean is not None:
                metrics["density_g_cm3_mean"] = mean
        elif "volume_nm3_final" in metrics and metrics["volume_nm3_final"]:
            metrics["density_g_cm3_final"] = SYSTEM_MASS_AMU * AMU_PER_NM3_TO_G_CM3 / metrics["volume_nm3_final"]
            metrics["density_g_cm3_mean"] = metrics["density_g_cm3_final"]
        add_lammps_raw_virial_metrics(metrics, final, rows)
        if "Temp" in final:
            metrics["temperature_k_final"] = final["Temp"]
            mean = metric_mean(rows, "Temp")
            if mean is not None:
                metrics["temperature_k_mean"] = mean
        if "PotEng" in final:
            metrics["potential_kj_mol_final"] = final["PotEng"] * KCAL_TO_KJ
            mean = metric_mean(rows, "PotEng")
            if mean is not None:
                metrics["potential_kj_mol_mean"] = mean * KCAL_TO_KJ
        if "KinEng" in final:
            metrics["kinetic_energy_kj_mol_final"] = final["KinEng"] * KCAL_TO_KJ
            mean = metric_mean(rows, "KinEng")
            if mean is not None:
                metrics["kinetic_energy_kj_mol_mean"] = mean * KCAL_TO_KJ
        if "TotEng" in final:
            metrics["total_energy_kj_mol_final"] = final["TotEng"] * KCAL_TO_KJ
            mean = metric_mean(rows, "TotEng")
            if mean is not None:
                metrics["total_energy_kj_mol_mean"] = mean * KCAL_TO_KJ
        if "Fmax" in final:
            metrics["fmax_final"] = final["Fmax"]
        if "Fnorm" in final:
            metrics["fnorm_final"] = final["Fnorm"]
        return metrics

    by_stage: dict[str, dict[str, Any]] = {}
    completed_segments: list[dict[str, Any]] = []
    for segment in segments:
        if not segment["completed"]:
            continue
        completed_segments.append(segment)
        stage_key = segment["stage_key"]
        by_stage[stage_key] = metrics_from_rows(
            stage_key, segment["input"], segment["rows"], True, segment.get("box_volume_nm3")
        )

    chunk_groups: dict[str, list[dict[str, Any]]] = {}
    for segment in completed_segments:
        stage_key = segment["stage_key"]
        if "_chunk" not in stage_key:
            continue
        base_key = re.sub(r"_chunk\d{4}$", "", stage_key)
        if base_key == stage_key:
            continue
        chunk_groups.setdefault(base_key, []).append(segment)
    for base_key, group in chunk_groups.items():
        if base_key in by_stage:
            continue
        ordered = sorted(group, key=lambda item: item["stage_key"])
        combined_rows: list[dict[str, float]] = []
        inputs: list[str] = []
        for segment in ordered:
            combined_rows.extend(segment["rows"])
            inputs.append(segment["input"])
        if not combined_rows:
            continue
        by_stage[base_key] = metrics_from_rows(base_key, "+".join(inputs), combined_rows, True)
    return by_stage


def rebuild_lammps_chunk_aggregate(metrics: dict[str, dict[str, Any]], base_key: str) -> None:
    """Rebuild a split-stage aggregate from all currently loaded chunk rows.

    This is needed when a production run is extended by appending a second
    stdout log.  Each parsed log contains its own aggregate row, so a plain
    dict update would make the aggregate represent only the last log.  The
    chunk rows themselves remain authoritative and are combined here.
    """

    chunk_re = re.compile(rf"^{re.escape(base_key)}_chunk(\d{{4}})$")
    chunks = sorted(
        (
            (int(match.group(1)), row)
            for key, row in metrics.items()
            if (match := chunk_re.match(key))
        ),
        key=lambda item: item[0],
    )
    if not chunks:
        return
    rows = [row for _, row in chunks]
    total_samples = sum(int(row.get("sample_count") or 0) for row in rows)
    last = rows[-1]
    aggregate: dict[str, Any] = {
        "engine": "lammps",
        "stage_key": base_key,
        "input": "+".join(str(row.get("input") or "") for row in rows),
        "sample_count": total_samples,
        "completed": all(bool(row.get("completed", True)) for row in rows),
    }
    keys = {key for row in rows for key in row}
    for key in sorted(keys):
        if key in aggregate or key in {"engine", "stage_key", "input", "sample_count", "completed"}:
            continue
        if key.endswith("_final") or key in {"time_ps_final", "fmax_final", "fnorm_final"}:
            aggregate[key] = last.get(key)
            continue
        if key.endswith("_mean") or key == "time_ps_mean":
            weighted_values = [
                (float(row[key]), int(row.get("sample_count") or 0))
                for row in rows
                if row.get(key) is not None and int(row.get("sample_count") or 0) > 0
            ]
            if weighted_values:
                denom = sum(weight for _, weight in weighted_values)
                aggregate[key] = sum(value * weight for value, weight in weighted_values) / denom
            continue
        if key not in aggregate:
            aggregate[key] = last.get(key)
    metrics[base_key] = aggregate


def add_lammps_pressure_tensor_metrics(
    metrics: dict[str, Any], final: dict[str, float], rows: list[dict[str, float]]
) -> None:
    scalar_map = {
        "c_p_full": "pressure_full_bar",
        "c_p_vir": "pressure_virial_bar",
    }
    tensor_map = {
        "c_p_full[1]": "pressure_full_xx_bar",
        "c_p_full[2]": "pressure_full_yy_bar",
        "c_p_full[3]": "pressure_full_zz_bar",
        "c_p_full[4]": "pressure_full_xy_bar",
        "c_p_full[5]": "pressure_full_xz_bar",
        "c_p_full[6]": "pressure_full_yz_bar",
        "c_p_vir[1]": "pressure_virial_xx_bar",
        "c_p_vir[2]": "pressure_virial_yy_bar",
        "c_p_vir[3]": "pressure_virial_zz_bar",
        "c_p_vir[4]": "pressure_virial_xy_bar",
        "c_p_vir[5]": "pressure_virial_xz_bar",
        "c_p_vir[6]": "pressure_virial_yz_bar",
    }
    for source, target in {**scalar_map, **tensor_map}.items():
        if source not in final:
            continue
        metrics[f"{target}_final"] = final[source] * ATM_TO_BAR
        mean = metric_mean(rows, source)
        if mean is not None:
            metrics[f"{target}_mean"] = mean * ATM_TO_BAR


def add_lammps_raw_virial_metrics(metrics: dict[str, Any], final: dict[str, float], rows: list[dict[str, float]]) -> None:
    component_map = {
        "xx": "c_p_vir[1]",
        "yy": "c_p_vir[2]",
        "zz": "c_p_vir[3]",
        "xy": "c_p_vir[4]",
        "xz": "c_p_vir[5]",
        "yz": "c_p_vir[6]",
    }

    def virial_from_pressure_bar(pressure_bar: float, volume_nm3: float) -> float:
        return -0.5 * pressure_bar * volume_nm3 / KJ_MOL_NM3_TO_BAR

    volume_final = metrics.get("volume_nm3_final")
    volume_mean = metrics.get("volume_nm3_mean", volume_final)
    for component, source in component_map.items():
        final_pressure_key = f"pressure_virial_{component}_bar_final"
        mean_pressure_key = f"pressure_virial_{component}_bar_mean"
        if isinstance(metrics.get(final_pressure_key), (int, float)) and isinstance(volume_final, (int, float)):
            metrics[f"virial_{component}_kj_mol_final"] = virial_from_pressure_bar(
                float(metrics[final_pressure_key]), float(volume_final)
            )

        row_values: list[float] = []
        for row in rows:
            if source not in row:
                continue
            row_volume_a3 = row.get("Volume")
            if row_volume_a3 is None:
                continue
            row_values.append(virial_from_pressure_bar(row[source] * ATM_TO_BAR, row_volume_a3 / 1000.0))
        if row_values:
            metrics[f"virial_{component}_kj_mol_mean"] = statistics.fmean(row_values)
        elif isinstance(metrics.get(mean_pressure_key), (int, float)) and isinstance(volume_mean, (int, float)):
            metrics[f"virial_{component}_kj_mol_mean"] = virial_from_pressure_bar(
                float(metrics[mean_pressure_key]), float(volume_mean)
            )


def restart_to_lammps_input_name(restart: Path) -> str | None:
    stem = restart.stem
    if not stem.startswith("equil_"):
        return None
    return f"resume_inputs/lammps_{stem}.in"


def lammps_endpoint_probe_text(restart: Path, stage_key: str) -> str:
    compute_no = stage_key.startswith("eq01_") or stage_key.startswith("eq02_")
    kspace_compute = "no" if compute_no else "yes"
    return f"""
echo both
units           real
boundary        p p p
atom_style      full

pair_style      lj/class2/coul/long 9.500000
kspace_style    pppm 0.0001
pair_modify     mix sixthpower
pair_modify     tail yes
bond_style      class2
angle_style     class2
dihedral_style  class2
improper_style  class2

read_restart    {restart}

neighbor        3.0 bin
neigh_modify    delay 0 every 1 check yes
special_bonds   lj/coul 0.0 0.0 1.0
kspace_modify   compute {kspace_compute}

variable        sysvol      equal vol
variable        sysmass     equal mass(all)/6.0221367e+23
variable        sysdensity  equal v_sysmass/v_sysvol/1.0e-24
compute         t_thermo all temp
compute         p_full all pressure t_thermo
compute         p_vir all pressure NULL virial

thermo_style    custom step press c_p_full c_p_full[1] c_p_full[2] c_p_full[3] c_p_full[4] c_p_full[5] c_p_full[6] c_p_vir c_p_vir[1] c_p_vir[2] c_p_vir[3] c_p_vir[4] c_p_vir[5] c_p_vir[6] vol v_sysdensity temp pe ke etotal
thermo          1
thermo_modify   flush yes
run             0 post no
"""


def run_lammps_endpoint_probes(lmp: Path, lammps_work: Path, probe_dir: Path) -> dict[str, Path]:
    probe_dir.mkdir(parents=True, exist_ok=True)
    restart_paths = sorted((lammps_work / ".resume_state").glob("equil_*.restart"))
    outputs: dict[str, Path] = {}
    for restart in restart_paths:
        input_name = restart_to_lammps_input_name(restart)
        stage_key = lammps_input_to_gmx_stage_key(input_name or "")
        if stage_key is None:
            continue
        script = probe_dir / f"{gmx_stage_key_to_stem(stage_key)}.in"
        stdout = probe_dir / f"{gmx_stage_key_to_stem(stage_key)}.stdout.log"
        script.write_text(lammps_endpoint_probe_text(restart.resolve(), stage_key), encoding="utf-8")
        if (
            stdout.exists()
            and stdout.stat().st_mtime >= restart.stat().st_mtime
            and "Loop time of" in stdout.read_text(encoding="utf-8", errors="ignore")
        ):
            outputs[stage_key] = stdout
            continue
        cmd = [str(lmp), "-sf", "omp", "-pk", "omp", "12", "-in", str(script)]
        run = subprocess.run(cmd, cwd=lammps_work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        stdout.write_text(run.stdout, encoding="utf-8")
        if run.returncode != 0:
            raise RuntimeError(f"LAMMPS endpoint probe failed for {stage_key}: {stdout}")
        outputs[stage_key] = stdout
    return outputs


def parse_lammps_endpoint_probe(path: Path, stage_key: str) -> dict[str, Any] | None:
    header: list[str] | None = None
    rows: list[dict[str, float]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = raw.split()
        if fields and fields[0] == "Step" and "c_p_full" in fields and "c_p_vir" in fields:
            header = fields
            rows = []
            continue
        if not header:
            continue
        vals = FLOAT_RE.findall(raw)
        if len(vals) == len(header):
            rows.append({name: float(value) for name, value in zip(header, vals)})
    if not header or not rows:
        return None
    final = rows[-1]
    metrics: dict[str, Any] = {
        "engine": "lammps",
        "stage_key": stage_key,
        "endpoint_probe_log": str(path),
        "endpoint_probe_sample_count": len(rows),
    }
    if "Press" in final:
        metrics["pressure_bar_final"] = final["Press"] * ATM_TO_BAR
    if "Volume" in final:
        metrics["volume_nm3_final"] = final["Volume"] / 1000.0
    if "v_sysdensity" in final:
        metrics["density_g_cm3_final"] = final["v_sysdensity"]
    if "Temp" in final:
        metrics["temperature_k_final"] = final["Temp"]
    if "PotEng" in final:
        metrics["potential_kj_mol_final"] = final["PotEng"] * KCAL_TO_KJ
    if "KinEng" in final:
        metrics["kinetic_energy_kj_mol_final"] = final["KinEng"] * KCAL_TO_KJ
    if "TotEng" in final:
        metrics["total_energy_kj_mol_final"] = final["TotEng"] * KCAL_TO_KJ
    add_lammps_pressure_tensor_metrics(metrics, final, rows)
    return metrics


def merge_lammps_endpoint_probes(lammps_metrics: dict[str, dict[str, Any]], probe_logs: dict[str, Path]) -> None:
    endpoint_scalar_metrics = {
        "pressure_bar_final",
        "volume_nm3_final",
        "density_g_cm3_final",
        "temperature_k_final",
        "potential_kj_mol_final",
        "kinetic_energy_kj_mol_final",
        "total_energy_kj_mol_final",
    }
    for stage_key, path in probe_logs.items():
        parsed = parse_lammps_endpoint_probe(path, stage_key)
        if not parsed:
            continue
        existing = lammps_metrics.setdefault(stage_key, {"engine": "lammps", "stage_key": stage_key})
        for key, value in parsed.items():
            if key in {"engine", "stage_key", "endpoint_probe_log", "endpoint_probe_sample_count"}:
                existing[key] = value
            elif key.startswith("pressure_full_") or key.startswith("pressure_virial_"):
                existing[key] = value
            elif key in endpoint_scalar_metrics:
                existing.setdefault(key, value)


def expected_signature_fragment(config: dict[str, Any], lammps_log: Path) -> dict[str, Any]:
    beta_by_stage = dict(config.get("GMX_PCFF_EWALD_BETA_INV_A_BY_STAGE", {}))
    for input_name, record in parse_lammps_pppm_by_input(lammps_log).items():
        stage_key = lammps_input_to_gmx_stage_key(input_name)
        if stage_key and "g_vector_inv_a" in record:
            beta_by_stage[stage_key] = float(record["g_vector_inv_a"])
    beta_schedule_by_stage = dict(config.get("GMX_PCFF_EWALD_BETA_INV_A_SCHEDULE_BY_STAGE", {}))
    eq01_beta = beta_by_stage.get("eq01_nvt_50ps")
    eq02_chunk2_beta = beta_by_stage.get("eq02_npt_compress_100ps_chunk0002")
    if eq01_beta is not None and eq02_chunk2_beta is not None:
        beta_schedule_by_stage["eq02_npt_compress_100ps"] = (
            f"0:{float(eq01_beta):.9g},400001:{float(eq02_chunk2_beta):.9g}"
        )

    respa_mode = config.get("GMX_RESPA_MODE")
    respa_layouts = config.get("GMX_RESPA_LAYOUTS", {})
    respa_layout = respa_layouts.get(respa_mode, {}) if isinstance(respa_layouts, dict) else {}
    respa_levels = respa_layout.get("levels")
    exact_respa_factor = None
    if isinstance(respa_levels, int):
        exact_respa_factor = respa_layout.get(f"level{respa_levels}_factor")

    expected = {
        "gmx_code_opt_tag": config.get("GMX_CODE_OPT_TAG"),
        "gmx_kspace_compute_no_mode": config.get("GMX_KSPACE_COMPUTE_NO_MODE"),
        "respa_mode": respa_mode,
        "gmx_respa_layout_description": respa_layout.get("description"),
        "exact_respa_factor": exact_respa_factor,
        "exact_respa_levels": respa_levels,
        "exact_respa_level2_factor": respa_layout.get("level2_factor"),
        "exact_respa_level3_factor": respa_layout.get("level3_factor"),
        "bond_level": respa_layout.get("bond_level"),
        "angle_level": respa_layout.get("angle_level"),
        "dihedral_level": respa_layout.get("dihedral_level"),
        "improper_level": respa_layout.get("improper_level"),
        "pair14_level": respa_layout.get("pair14_level"),
        "pair_level": respa_layout.get("pair_level"),
        "kspace_level": respa_layout.get("kspace_level"),
        "gmx_initial_em_in_gromacs": config.get("RUN_GMX_INITIAL_EM"),
        "gmx_initial_em_nsteps": config.get("GMX_INITIAL_EM_NSTEPS"),
        "gmx_em_integrator": config.get("GMX_EM_INTEGRATOR"),
        "gmx_em_nsteps": config.get("GMX_EM_NSTEPS"),
        "gmx_em_tol_kj_mol_nm": config.get("GMX_EM_TOL_KJ_MOL_NM"),
        "gmx_em_step_nm": config.get("GMX_EM_STEP_NM"),
        "gmx_em_nstcgsteep": config.get("GMX_EM_NSTCGSTEEP"),
        "gmx_em_match_lammps_cg": config.get("GMX_EM_MATCH_LAMMPS_CG"),
        "gmx_dispcorr": config.get("GMX_DISPCORR"),
        "gmx_pme_order": config.get("GMX_PME_ORDER"),
        "gmx_nstlist": config.get("GMX_NSTLIST"),
        "gmx_rlist_nm": config.get("GMX_RLIST_NM"),
        "gmx_cutoff_nm": config.get("GMX_CUTOFF_NM"),
        "gmx_npt_ref_pressure_mode": config.get("GMX_NPT_REF_PRESSURE_MODE"),
        "gmx_pressure_ramp_mode": config.get("GMX_NPT_REF_PRESSURE_MODE"),
        "gmx_match_lammps_run_chunks": config.get("GMX_MATCH_LAMMPS_RUN_CHUNKS"),
        "gmx_unsplit_lammps_chunk_stage_names": sorted(config.get("GMX_UNSPLIT_LAMMPS_CHUNK_STAGE_NAMES", [])),
        "gmx_apply_eq12_avg_cell_to_eq13": config.get("GMX_APPLY_EQ12_AVG_CELL_TO_EQ13"),
        "gmx_eq12_avg_cell_window_ps": list(config.get("GMX_EQ12_AVG_CELL_WINDOW_PS", [])),
        "gmx_use_lammps_reset_velocities": config.get("GMX_USE_LAMMPS_RESET_VELOCITIES"),
        "gmx_match_lammps_temperature_dof": config.get("GMX_MATCH_LAMMPS_TEMPERATURE_DOF"),
        "gmx_lammps_dof_nstcomm": config.get("GMX_LAMMPS_DOF_NSTCOMM"),
        "gmx_mttk_nreset_steps": config.get("GMX_MTTK_NRESET_STEPS"),
        "gmx_pcff_mttk_mass_mode": config.get("GMX_PCFF_MTTK_MASS_MODE"),
        "gmx_pcff_nvt_mass_mode": config.get("GMX_PCFF_NVT_MASS_MODE"),
        "gmx_pcff_mttk_mass_mode_by_stage": dict(
            sorted(config.get("GMX_PCFF_MTTK_MASS_MODE_BY_STAGE", {}).items())
        ),
        "gmx_pcff_mttk_pressure_mass_scale": config.get("GMX_PCFF_MTTK_PRESSURE_MASS_SCALE"),
        "gmx_pcff_mttk_pressure_mass_scale_by_stage": dict(
            sorted(config.get("GMX_PCFF_MTTK_PRESSURE_MASS_SCALE_BY_STAGE", {}).items())
        ),
        "gmx_pcff_mttk_boxv_integrator": config.get("GMX_PCFF_MTTK_BOXV_INTEGRATOR"),
        "gmx_pcff_ewald_beta_inv_a_schedule_by_stage": dict(sorted(beta_schedule_by_stage.items())),
        "gmx_pcff_exact_respa_pre_trotter": config.get("GMX_PCFF_EXACT_RESPA_PRE_TROTTER"),
        "gmx_pcff_exact_respa_post_trotter": config.get("GMX_PCFF_EXACT_RESPA_POST_TROTTER"),
        "gmx_pcff_exact_respa_post_trotter_by_ensemble": dict(
            sorted(config.get("GMX_PCFF_EXACT_RESPA_POST_TROTTER_BY_ENSEMBLE", {}).items())
        ),
        "gmx_pcff_mttk_extended_update_mode": config.get("GMX_PCFF_MTTK_EXTENDED_UPDATE_MODE"),
        "gmx_pcff_mttk_veta_scale": config.get("GMX_PCFF_MTTK_VETA_SCALE"),
        "gmx_pcff_mttk_veta_scale_by_stage": dict(
            sorted(config.get("GMX_PCFF_MTTK_VETA_SCALE_BY_STAGE", {}).items())
        ),
        "gmx_pass_eq01_cpt_to_eq02": config.get("GMX_PASS_EQ01_CPT_TO_EQ02"),
        "gmx_pass_same_base_stage_cpt": config.get("GMX_PASS_SAME_BASE_STAGE_CPT"),
        "gmx_pass_same_base_stage_edr_state": config.get("GMX_PASS_SAME_BASE_STAGE_EDR_STATE"),
        "gmx_print_nose_hoover_chain_variables": config.get("GMX_PRINT_NOSE_HOOVER_CHAIN_VARIABLES"),
        "gmx_reset_nh_mttk_state_on_stage_start": config.get("GMX_RESET_NH_MTTK_STATE_ON_STAGE_START"),
        "gmx_nbnxm_owner_step_scalar_fallback": config.get("GMX_NBNXM_OWNER_STEP_SCALAR_FALLBACK"),
        "gmx_compute_no_forces_bonded_cpu": config.get("GMX_COMPUTE_NO_FORCES_BONDED_CPU"),
        "lammps_eq01_gewald_inv_a": beta_by_stage.get("eq01_nvt_50ps"),
        "lammps_eq02_chunk2_gewald_inv_a": beta_by_stage.get("eq02_npt_compress_100ps_chunk0002"),
        "lammps_eq03_gewald_inv_a": beta_by_stage.get("eq03_pre_2fs_minimize"),
        "gmx_ewald_beta_inv_a_by_stage": beta_by_stage,
    }
    nhc_integrator = config.get("GMX_PCFF_NHC_INTEGRATOR")
    if nhc_integrator:
        expected["gmx_pcff_nhc_integrator"] = nhc_integrator
    return expected


def relax_nonphysics_signature(expected: dict[str, Any]) -> dict[str, Any]:
    relaxed = dict(expected)
    # These flags select which stages the notebook attempts. They do not alter
    # a completed stage's MDP/TPR/runtime physics.
    relaxed.pop("run_production", None)
    relaxed.pop("run_production_only", None)

    # The code tag records performance and trace-guard edits as well as physics
    # edits.  In relaxed audit mode it is not a reliable physics freshness gate;
    # the MDP/signature fields below still carry the actual dynamics settings.
    relaxed.pop("gmx_code_opt_tag", None)
    return relaxed


def value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)
    return actual == expected


def signature_status(runtime_path: Path, expected: dict[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    if not runtime_path.exists():
        return "missing_runtime", ["runtime_json_missing"], {}
    try:
        payload = load_json(runtime_path)
    except Exception as exc:
        return "unreadable_runtime", [f"runtime_json_unreadable:{exc}"], {}
    signature = payload.get("gmx_runtime_signature", {})
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if key == "gmx_ewald_beta_inv_a_by_stage":
            actual_map = signature.get(key)
            if not isinstance(actual_map, dict):
                mismatches.append(f"{key}:missing_map")
                continue
            for stage_key, stage_expected in expected_value.items():
                stage_actual = actual_map.get(stage_key)
                if not value_matches(stage_actual, stage_expected):
                    mismatches.append(f"{key}.{stage_key}:actual={stage_actual!r},expected={stage_expected!r}")
            continue
        actual_value = signature.get(key)
        if not value_matches(actual_value, expected_value):
            mismatches.append(f"{key}:actual={actual_value!r},expected={expected_value!r}")
    return ("current_critical" if not mismatches else "stale_critical"), mismatches, payload


def parse_gmx_energy_terms(gmx: Path, edr: Path, out_dir: Path) -> dict[str, dict[str, float]]:
    probe = subprocess.run(
        [str(gmx), "energy", "-f", str(edr), "-o", str(out_dir / "_discard.xvg")],
        input="0\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    available = {
        match.group(2)
        for match in re.finditer(r"(?:^|\s{2,})(\d+)\s+([A-Za-z0-9#.*()/-]+)", probe.stdout, flags=re.M)
    }
    desired = [
        "Class2-Bond",
        "Class2-Angle",
        "Class2-Dih.",
        "Class2-Impr.",
        "LJ-14",
        "Coulomb-14",
        "LJ-(SR)",
        "Disper.-corr.",
        "Coulomb-(SR)",
        "Coul.-recip.",
        "Potential",
        "Kinetic-En.",
        "Total-Energy",
        "Temperature",
        "Pres.-DC",
        "Pressure",
        "Volume",
        "Density",
        "Vir-XX",
        "Vir-XY",
        "Vir-XZ",
        "Vir-YX",
        "Vir-YY",
        "Vir-YZ",
        "Vir-ZX",
        "Vir-ZY",
        "Vir-ZZ",
        "Pres-XX",
        "Pres-XY",
        "Pres-XZ",
        "Pres-YX",
        "Pres-YY",
        "Pres-YZ",
        "Pres-ZX",
        "Pres-ZY",
        "Pres-ZZ",
    ]
    selected = [term for term in desired if term in available]
    if not selected:
        return {}
    with tempfile.NamedTemporaryFile(prefix="gmx_energy_", suffix=".xvg", dir=out_dir, delete=False) as handle:
        xvg_path = Path(handle.name)
    selection = "\n".join(selected) + "\n0\n"
    run = subprocess.run(
        [str(gmx), "energy", "-f", str(edr), "-o", str(xvg_path)],
        input=selection,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if run.returncode != 0:
        return {"_error": {"returncode": float(run.returncode)}}
    legends: list[str] = []
    rows: list[list[float]] = []
    for raw in xvg_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            m = re.search(r'@\s+s(\d+)\s+legend\s+"(.+)"', line)
            if m:
                idx = int(m.group(1))
                while len(legends) <= idx:
                    legends.append("")
                legends[idx] = m.group(2)
            continue
        rows.append([float(part) for part in line.split()])
    if not rows:
        return {}
    out: dict[str, dict[str, float]] = {}
    for idx, legend in enumerate(legends):
        if not legend:
            continue
        values = [row[idx + 1] for row in rows if len(row) > idx + 1]
        if not values:
            continue
        out[legend] = {
            "first": values[0],
            "last": values[-1],
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
            "sample_count": float(len(values)),
        }
    legend_index = {legend: idx for idx, legend in enumerate(legends) if legend}
    volume_idx = legend_index.get("Volume")
    if volume_idx is not None:
        volume_values = [row[volume_idx + 1] for row in rows if len(row) > volume_idx + 1]
        pres_dc_idx = legend_index.get("Pres. DC")
        for component in ("XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"):
            vir_idx = legend_index.get(f"Vir-{component}")
            if vir_idx is None:
                continue
            converted: list[float] = []
            for row, volume_nm3 in zip(rows, volume_values):
                if len(row) <= vir_idx + 1 or not volume_nm3:
                    continue
                # GROMACS pressure uses calc_pres(): pres = 2 * (ekin - vir) * c_presfac / V.
                # Therefore the virial-only pressure contribution comparable to LAMMPS
                # compute pressure NULL virial is -2 * vir * c_presfac / V.
                virial_pressure = -2.0 * row[vir_idx + 1] / volume_nm3 * KJ_MOL_NM3_TO_BAR
                converted.append(virial_pressure)
            if not converted:
                continue
            out[f"VirialPress-{component}"] = {
                "first": converted[0],
                "last": converted[-1],
                "mean": statistics.fmean(converted),
                "min": min(converted),
                "max": max(converted),
                "sample_count": float(len(converted)),
            }
    pressure_idx = legend_index.get("Pressure")
    pres_dc_idx = legend_index.get("Pres. DC")
    if pressure_idx is not None and pres_dc_idx is not None:
        converted = []
        for row in rows:
            if len(row) <= max(pressure_idx, pres_dc_idx) + 1:
                continue
            converted.append(row[pressure_idx + 1] - row[pres_dc_idx + 1])
        if converted:
            out["PressureNoDispCorr"] = {
                "first": converted[0],
                "last": converted[-1],
                "mean": statistics.fmean(converted),
                "min": min(converted),
                "max": max(converted),
                "sample_count": float(len(converted)),
            }
    return out


def gmx_terms_to_metrics(terms: dict[str, dict[str, float]]) -> dict[str, Any]:
    def term(name: str, stat: str) -> float | None:
        value = terms.get(name, {}).get(stat)
        return value if value is not None and math.isfinite(value) else None

    metrics: dict[str, Any] = {"engine": "gromacs"}
    term_map = {
        "Potential": "potential_kj_mol",
        "Kinetic-En.": "kinetic_energy_kj_mol",
        "Kinetic En.": "kinetic_energy_kj_mol",
        "Total-Energy": "total_energy_kj_mol",
        "Total Energy": "total_energy_kj_mol",
        "Temperature": "temperature_k",
        "Pres. DC": "pressure_disp_corr_bar",
        "Pressure": "pressure_bar",
        "PressureNoDispCorr": "pressure_no_disp_corr_bar",
        "Volume": "volume_nm3",
        "Pres-XX": "pressure_xx_bar",
        "Pres-XY": "pressure_xy_bar",
        "Pres-XZ": "pressure_xz_bar",
        "Pres-YX": "pressure_yx_bar",
        "Pres-YY": "pressure_yy_bar",
        "Pres-YZ": "pressure_yz_bar",
        "Pres-ZX": "pressure_zx_bar",
        "Pres-ZY": "pressure_zy_bar",
        "Pres-ZZ": "pressure_zz_bar",
        "Vir-XX": "virial_xx_kj_mol",
        "Vir-XY": "virial_xy_kj_mol",
        "Vir-XZ": "virial_xz_kj_mol",
        "Vir-YX": "virial_yx_kj_mol",
        "Vir-YY": "virial_yy_kj_mol",
        "Vir-YZ": "virial_yz_kj_mol",
        "Vir-ZX": "virial_zx_kj_mol",
        "Vir-ZY": "virial_zy_kj_mol",
        "Vir-ZZ": "virial_zz_kj_mol",
        "VirialPress-XX": "virial_pressure_xx_bar",
        "VirialPress-XY": "virial_pressure_xy_bar",
        "VirialPress-XZ": "virial_pressure_xz_bar",
        "VirialPress-YX": "virial_pressure_yx_bar",
        "VirialPress-YY": "virial_pressure_yy_bar",
        "VirialPress-YZ": "virial_pressure_yz_bar",
        "VirialPress-ZX": "virial_pressure_zx_bar",
        "VirialPress-ZY": "virial_pressure_zy_bar",
        "VirialPress-ZZ": "virial_pressure_zz_bar",
    }
    for source, target in term_map.items():
        for stat in ("first", "last", "mean", "min", "max"):
            value = term(source, stat)
            if value is not None:
                metrics[f"{target}_{'final' if stat == 'last' else stat}"] = value
    for stat in ("first", "last", "mean", "min", "max"):
        value = term("Density", stat)
        if value is not None:
            suffix = "final" if stat == "last" else stat
            metrics[f"density_g_cm3_{suffix}"] = value / 1000.0
    potential = term("Potential", "last")
    if potential is not None:
        metrics["sample_count"] = int(next(iter(terms.values())).get("sample_count", 0.0))
    return metrics


def gro_volume_nm3(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        vals = [float(value) for value in path.read_text(encoding="utf-8", errors="ignore").splitlines()[-1].split()[:3]]
    except Exception:
        return None
    if len(vals) != 3:
        return None
    return vals[0] * vals[1] * vals[2]


def discover_runtime_stems(work: Path) -> list[str]:
    stems = set()

    def is_schedule_stem(stem: str) -> bool:
        if stem.startswith("probe_") or stem.startswith("sanity_"):
            return False
        if stem.endswith("_startcheck"):
            return False
        return stem == "00_gromacs_initial_em" or re.match(r"^\d{2}_", stem) is not None

    for runtime in (work / ".resume_state").glob("*.runtime.json"):
        try:
            payload = load_json(runtime)
        except Exception:
            continue
        stage = payload.get("stage")
        if isinstance(stage, str) and is_schedule_stem(stage):
            stems.add(stage)
    for edr in work.glob("*.edr"):
        if not edr.name.endswith(".part0001.edr") and is_schedule_stem(edr.stem):
            stems.add(edr.stem)
    return sorted(stems)


def gromacs_stage_freshness_stamp(work: Path, stem: str) -> Path | None:
    """Return a lane-local stamp that must be older than current stage outputs."""
    if re.match(r"^14_prod01_nvt_10000ps_chunk\d{4}$", stem):
        stamp = work / "14_prod01_nvt_10000ps_chunk0001.lammps_relaxed_box.g96"
        if stamp.exists():
            return stamp
    return None


def files_older_than(reference: Path, files: list[Path]) -> list[str]:
    older: list[str] = []
    reference_mtime = reference.stat().st_mtime
    for file in files:
        if file.exists() and file.stat().st_mtime < reference_mtime:
            older.append(file.name)
    return older


def collect_gromacs_lane(
    gmx: Path,
    work: Path,
    lane: str,
    expected: dict[str, Any],
    audit_out: Path,
    extract_edr: bool,
    expected_stems: list[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    status_rows: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    stems = expected_stems if expected_stems is not None else discover_runtime_stems(work)
    for stem in stems:
        runtime = work / ".resume_state" / f"gromacs_{lane}_{stem}.runtime.json"
        done = work / ".resume_state" / f"gromacs_{lane}_{stem}.done.json"
        status, mismatches, payload = signature_status(runtime, expected)
        done_status, done_mismatches, done_payload = signature_status(done, expected)
        if not done.exists():
            status = "incomplete_current" if status == "current_critical" else "incomplete_stale"
            mismatches = ["done_marker_missing", *mismatches]
        elif done_status != "current_critical":
            status = "incomplete_current" if status == "current_critical" else "stale_critical"
            mismatches = [f"done_marker_{done_status}", *done_mismatches, *mismatches]
        edr = work / f"{stem}.edr"
        gro = work / f"{stem}.gro"
        tpr = work / f"{stem}.tpr"
        stale_files: list[str] = []
        if tpr.exists():
            for output in (gro, edr):
                if output.exists() and output.stat().st_mtime < tpr.stat().st_mtime:
                    stale_files.append(output.name)
        if stale_files:
            status = "stale_critical"
            mismatches = [f"output_older_than_tpr:{'|'.join(stale_files)}", *mismatches]
        freshness_stamp = gromacs_stage_freshness_stamp(work, stem)
        if freshness_stamp is not None:
            stamp_stale_files = files_older_than(freshness_stamp, [runtime, done, tpr, gro, edr])
            if stamp_stale_files:
                status = "stale_critical"
                mismatches = [
                    f"output_older_than_stage_stamp:{freshness_stamp.name}:{'|'.join(stamp_stale_files)}",
                    *mismatches,
                ]
        row = {
            "lane": lane,
            "stem": stem,
            "stage_key": gmx_stem_to_stage_key(stem),
            "runtime_status": status,
            "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else "",
            "runtime_json": str(runtime) if runtime.exists() else "",
            "done_json": str(done) if done.exists() else "",
            "edr_exists": edr.exists(),
            "gro_exists": gro.exists(),
            "timestamp": payload.get("timestamp", "") if payload else "",
            "done_timestamp": done_payload.get("timestamp", "") if done_payload else "",
        }
        status_rows.append(row)
        metric_row: dict[str, Any] = {
            "engine": "gromacs",
            "lane": lane,
            "stem": stem,
            "stage_key": row["stage_key"],
            "runtime_status": status,
            "mismatch_count": len(mismatches),
            "expected_current_stage": expected_stems is not None,
        }
        metrics_are_current = status == "current_critical"
        if not metrics_are_current:
            metric_row["metrics_blocked_reason"] = (
                "GROMACS outputs are incomplete or stale for the current runtime; "
                "skip thermodynamic comparison to avoid mixing partial .edr with old final coordinates."
            )
            metrics[f"{lane}:{row['stage_key']}"] = metric_row
            continue
        vol = gro_volume_nm3(gro)
        gro_density = None
        if vol is not None:
            metric_row["volume_nm3_from_gro"] = vol
            gro_density = SYSTEM_MASS_AMU * AMU_PER_NM3_TO_G_CM3 / vol
            metric_row["density_g_cm3_from_gro"] = gro_density
        if extract_edr and edr.exists():
            terms = parse_gmx_energy_terms(gmx, edr, audit_out)
            metric_row.update(gmx_terms_to_metrics(terms))
        if vol is not None:
            metric_row.setdefault("volume_nm3_final", vol)
        if gro_density is not None:
            metric_row.setdefault("density_g_cm3_final", gro_density)
        if "_nvt_" in row["stage_key"] or row["stage_key"].startswith("prod"):
            # Fixed-volume stages can omit Volume/Density from the energy file,
            # but the .gro box is still the constant volume for the whole chunk.
            if vol is not None:
                metric_row.setdefault("volume_nm3_mean", vol)
            if gro_density is not None:
                metric_row.setdefault("density_g_cm3_mean", gro_density)
        metrics[f"{lane}:{row['stage_key']}"] = metric_row
    return status_rows, metrics


def stage_family(stage_key: str) -> str:
    if stage_key == "gromacs_initial_em" or "minimize" in stage_key:
        return "minimize"
    if "_npt_" in stage_key:
        return "npt"
    if "_nvt_" in stage_key:
        return "nvt"
    if stage_key.startswith("prod"):
        return "prod"
    return "unknown"


def metric_stat(metric: str) -> str:
    if "_final" in metric:
        return "final"
    if "_mean" in metric:
        return "mean"
    if metric.endswith("_first"):
        return "first"
    if metric.endswith("_min"):
        return "min"
    if metric.endswith("_max"):
        return "max"
    return "other"


def annotate_comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    family = stage_family(str(row.get("stage_key", "")))
    stat = metric_stat(str(row.get("metric", "")))
    row["stage_family"] = family
    row["metric_stat"] = stat

    if row.get("status") != "measured":
        row["comparison_use"] = "unavailable"
        row["trajectory_sensitive"] = False
        return row

    if family in {"nvt", "npt", "prod"}:
        row["trajectory_sensitive"] = True
        if stat == "final":
            row["comparison_use"] = "independent_dynamic_endpoint_diagnostic"
            row["interpretation_note"] = (
                "Do not use this as a root-cause gate: LAMMPS and GROMACS have already "
                "followed independent Nose-Hoover/MTTK trajectories."
            )
        elif stat == "mean":
            row["comparison_use"] = "independent_dynamic_window_statistic"
            row["interpretation_note"] = (
                "Use as an ensemble/window diagnostic only; it does not prove step-by-step trajectory identity."
            )
        else:
            row["comparison_use"] = "dynamic_diagnostic"
        return row

    if family == "minimize":
        row["trajectory_sensitive"] = False
        row["comparison_use"] = "minimization_endpoint_diagnostic"
        row["interpretation_note"] = (
            "Minimization endpoints depend on minimizer line-search/details; same-state run-0 probes are "
            "the stronger force/virial root-cause evidence."
        )
        return row

    row["trajectory_sensitive"] = False
    row["comparison_use"] = "mechanical_diagnostic"
    return row


def annotate_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [annotate_comparison_row(row) for row in rows]


def resolve_same_start_probe_csvs(entries: list[str] | None) -> list[Path]:
    candidates: list[Path] = []
    if entries is None:
        candidates = []
    else:
        for entry in entries:
            path = Path(entry).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if path.is_dir():
                candidates.append(path / "eq02_short_propagation_summary.csv")
                candidates.append(path / "stage_short_probe_summary.csv")
            else:
                candidates.append(path)
    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def load_same_start_probe_rows(entries: list[str] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in resolve_same_start_probe_csvs(entries):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                row: dict[str, Any] = dict(raw)
                row["probe_summary_csv"] = str(csv_path)
                row["probe_root"] = str(csv_path.parent)
                duration = parse_optional_float(row.get("time_ps"))
                if duration is None:
                    outer_steps = parse_optional_float(row.get("outer_steps"))
                    duration = None if outer_steps is None else outer_steps * 0.0005
                row["probe_duration_ps"] = duration
                for key in (
                    "temperature_delta",
                    "pressure_delta",
                    "pressure_initial_delta",
                    "volume_delta",
                    "potential_delta",
                    "potential_initial_delta",
                    "density_g_cm3_delta",
                    "virial_pressure_delta_bar",
                    "virial_pressure_initial_delta_bar",
                    "lammps_volume_nm3",
                    "lammps_potential_kj_mol",
                    "lammps_potential_initial_kj_mol",
                    "outer_steps",
                    "pressure_mass_scale",
                ):
                    value = parse_optional_float(row.get(key))
                    if value is not None:
                        row[key] = value
                volume_delta = parse_optional_float(row.get("volume_delta"))
                lammps_volume = parse_optional_float(row.get("lammps_volume_nm3"))
                if volume_delta is not None and lammps_volume:
                    row["volume_pct_delta_vs_lammps"] = 100.0 * volume_delta / lammps_volume
                potential_delta = parse_optional_float(row.get("potential_delta"))
                lammps_potential = parse_optional_float(row.get("lammps_potential_kj_mol"))
                if potential_delta is not None and lammps_potential:
                    row["potential_pct_delta_vs_lammps"] = 100.0 * potential_delta / lammps_potential
                is_minimize_probe = str(row.get("stage_kind")) == "minimize" or str(row.get("ensemble")) == "minimize"
                if is_minimize_probe:
                    row["probe_class"] = "minimization_same_start"
                    row["comparison_use"] = "same_start_minimization_diagnostic"
                    row["interpretation_note"] = (
                        "Minimization uses the same starting coordinates/box. Initial PE/virial deltas are the "
                        "mechanical parity gate; final endpoint deltas are CG line-search diagnostics. LAMMPS "
                        "full pressure during minimization can include restart kinetic-state contributions, so "
                        "use virial-pressure deltas for root-cause checks."
                    )
                elif duration is not None and duration <= 0.02:
                    row["probe_class"] = "initial_order_gate"
                    row["comparison_use"] = "same_start_short_step_root_cause_gate"
                elif duration is not None and duration <= 1.0:
                    row["probe_class"] = "short_propagation_gate"
                    row["comparison_use"] = "same_start_short_propagation_diagnostic"
                else:
                    row["probe_class"] = "trajectory_divergence_scan"
                    row["comparison_use"] = "same_start_longer_divergence_diagnostic"
                if not is_minimize_probe:
                    row["interpretation_note"] = (
                        "This starts both engines from the same Eq01 endpoint. It is stronger root-cause evidence "
                        "than comparing independently propagated full-stage endpoints."
                    )
                chunk_match = re.search(r"_chunk(\d{4})$", str(row.get("stage_key", "")))
                if (
                    not row.get("root_cause_gate")
                    and chunk_match
                    and int(chunk_match.group(1)) > 1
                    and str(row.get("ensemble")) in {"nvt", "npt"}
                ):
                    row["root_cause_gate"] = "diagnostic_only"
                    row["root_cause_gate_reason"] = (
                        "LAMMPS read_restart restores FixNH extended state for split chunks, but the same-start "
                        "data/gro bridge cannot carry that state into GROMACS."
                    )
                if str(row.get("root_cause_gate")) == "diagnostic_only":
                    row["comparison_use"] = "same_start_split_chunk_diagnostic_only"
                    row["interpretation_note"] = str(
                        row.get("root_cause_gate_reason")
                        or "Split-chunk same-start probes do not carry LAMMPS FixNH extended state into GROMACS."
                    )
                rows.append(row)
    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("stage_key", "")),
            str(row.get("probe_class", "")),
            str(row.get("ensemble_override", "")),
        )
        current = deduped.get(key)
        if current is None:
            deduped[key] = row
            continue
        current_score = sum(1 for value in current.values() if value not in (None, ""))
        row_score = sum(1 for value in row.values() if value not in (None, ""))
        if row_score >= current_score:
            deduped[key] = row
    rows = list(deduped.values())
    rows.sort(key=lambda row: (str(row.get("probe_class")), float(row.get("probe_duration_ps") or 0.0), str(row.get("probe_root"))))
    return rows


def compare_lammps_gromacs(lammps: dict[str, dict[str, Any]], gmx_metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparable_metrics = [
        "volume_nm3_final",
        "volume_nm3_mean",
        "density_g_cm3_final",
        "density_g_cm3_mean",
        "temperature_k_final",
        "temperature_k_mean",
        "pressure_bar_final",
        "pressure_bar_mean",
        "pressure_disp_corr_bar_final",
        "pressure_disp_corr_bar_mean",
        "pressure_no_disp_corr_bar_final",
        "pressure_no_disp_corr_bar_mean",
        "potential_kj_mol_final",
        "potential_kj_mol_mean",
        "kinetic_energy_kj_mol_final",
        "kinetic_energy_kj_mol_mean",
        "total_energy_kj_mol_final",
        "total_energy_kj_mol_mean",
        "fmax_final",
        "fnorm_final",
    ]
    tensor_comparisons = {
        "pressure_xx_bar_final": "pressure_full_xx_bar_final",
        "pressure_xy_bar_final": "pressure_full_xy_bar_final",
        "pressure_xz_bar_final": "pressure_full_xz_bar_final",
        "pressure_yx_bar_final": "pressure_full_xy_bar_final",
        "pressure_yy_bar_final": "pressure_full_yy_bar_final",
        "pressure_yz_bar_final": "pressure_full_yz_bar_final",
        "pressure_zx_bar_final": "pressure_full_xz_bar_final",
        "pressure_zy_bar_final": "pressure_full_yz_bar_final",
        "pressure_zz_bar_final": "pressure_full_zz_bar_final",
        "pressure_xx_bar_mean": "pressure_full_xx_bar_mean",
        "pressure_xy_bar_mean": "pressure_full_xy_bar_mean",
        "pressure_xz_bar_mean": "pressure_full_xz_bar_mean",
        "pressure_yx_bar_mean": "pressure_full_xy_bar_mean",
        "pressure_yy_bar_mean": "pressure_full_yy_bar_mean",
        "pressure_yz_bar_mean": "pressure_full_yz_bar_mean",
        "pressure_zx_bar_mean": "pressure_full_xz_bar_mean",
        "pressure_zy_bar_mean": "pressure_full_yz_bar_mean",
        "pressure_zz_bar_mean": "pressure_full_zz_bar_mean",
        "virial_pressure_xx_bar_final": "pressure_virial_xx_bar_final",
        "virial_pressure_xy_bar_final": "pressure_virial_xy_bar_final",
        "virial_pressure_xz_bar_final": "pressure_virial_xz_bar_final",
        "virial_pressure_yx_bar_final": "pressure_virial_xy_bar_final",
        "virial_pressure_yy_bar_final": "pressure_virial_yy_bar_final",
        "virial_pressure_yz_bar_final": "pressure_virial_yz_bar_final",
        "virial_pressure_zx_bar_final": "pressure_virial_xz_bar_final",
        "virial_pressure_zy_bar_final": "pressure_virial_yz_bar_final",
        "virial_pressure_zz_bar_final": "pressure_virial_zz_bar_final",
        "virial_pressure_xx_bar_mean": "pressure_virial_xx_bar_mean",
        "virial_pressure_xy_bar_mean": "pressure_virial_xy_bar_mean",
        "virial_pressure_xz_bar_mean": "pressure_virial_xz_bar_mean",
        "virial_pressure_yx_bar_mean": "pressure_virial_xy_bar_mean",
        "virial_pressure_yy_bar_mean": "pressure_virial_yy_bar_mean",
        "virial_pressure_yz_bar_mean": "pressure_virial_yz_bar_mean",
        "virial_pressure_zx_bar_mean": "pressure_virial_xz_bar_mean",
        "virial_pressure_zy_bar_mean": "pressure_virial_yz_bar_mean",
        "virial_pressure_zz_bar_mean": "pressure_virial_zz_bar_mean",
        "virial_xx_kj_mol_final": "virial_xx_kj_mol_final",
        "virial_xy_kj_mol_final": "virial_xy_kj_mol_final",
        "virial_xz_kj_mol_final": "virial_xz_kj_mol_final",
        "virial_yy_kj_mol_final": "virial_yy_kj_mol_final",
        "virial_yz_kj_mol_final": "virial_yz_kj_mol_final",
        "virial_zz_kj_mol_final": "virial_zz_kj_mol_final",
        "virial_xx_kj_mol_mean": "virial_xx_kj_mol_mean",
        "virial_xy_kj_mol_mean": "virial_xy_kj_mol_mean",
        "virial_xz_kj_mol_mean": "virial_xz_kj_mol_mean",
        "virial_yy_kj_mol_mean": "virial_yy_kj_mol_mean",
        "virial_yz_kj_mol_mean": "virial_yz_kj_mol_mean",
        "virial_zz_kj_mol_mean": "virial_zz_kj_mol_mean",
    }
    em_endpoint_virial_comparisons = {
        "pressure_bar_final": "pressure_virial_bar_final",
        "pressure_xx_bar_final": "pressure_virial_xx_bar_final",
        "pressure_xy_bar_final": "pressure_virial_xy_bar_final",
        "pressure_xz_bar_final": "pressure_virial_xz_bar_final",
        "pressure_yx_bar_final": "pressure_virial_xy_bar_final",
        "pressure_yy_bar_final": "pressure_virial_yy_bar_final",
        "pressure_yz_bar_final": "pressure_virial_yz_bar_final",
        "pressure_zx_bar_final": "pressure_virial_xz_bar_final",
        "pressure_zy_bar_final": "pressure_virial_yz_bar_final",
        "pressure_zz_bar_final": "pressure_virial_zz_bar_final",
    }
    for key, gmx in sorted(gmx_metrics.items()):
        lane = str(gmx.get("lane"))
        stage_key = str(gmx.get("stage_key"))
        ref = lammps.get(stage_key)
        is_minimize = stage_key == "gromacs_initial_em" or "minimize" in stage_key
        for metric in comparable_metrics:
            if is_minimize and metric.startswith("pressure_"):
                continue
            gmx_value = gmx.get(metric)
            ref_value = ref.get(metric) if ref else None
            row: dict[str, Any] = {
                "lane": lane,
                "stage_key": stage_key,
                "metric": metric,
                "runtime_status": gmx.get("runtime_status"),
                "lammps_value": ref_value,
                "gromacs_value": gmx_value,
                "status": "missing_lammps" if ref is None or ref_value is None else "missing_gromacs" if gmx_value is None else "measured",
            }
            if isinstance(ref_value, (int, float)) and isinstance(gmx_value, (int, float)):
                delta = float(gmx_value) - float(ref_value)
                row["delta_gromacs_minus_lammps"] = delta
                if ref_value:
                    row["pct_delta_vs_lammps"] = 100.0 * delta / float(ref_value)
            rows.append(row)
        for gmx_metric, lammps_metric in tensor_comparisons.items():
            if gmx_metric not in gmx:
                continue
            if is_minimize and gmx_metric.startswith("pressure_"):
                continue
            ref_value = ref.get(lammps_metric) if ref else None
            gmx_value = gmx.get(gmx_metric)
            row = {
                "lane": lane,
                "stage_key": stage_key,
                "metric": gmx_metric,
                "lammps_metric": lammps_metric,
                "runtime_status": gmx.get("runtime_status"),
                "lammps_value": ref_value,
                "gromacs_value": gmx_value,
                "status": "missing_lammps" if ref is None or ref_value is None else "measured",
            }
            if isinstance(ref_value, (int, float)) and isinstance(gmx_value, (int, float)):
                delta = float(gmx_value) - float(ref_value)
                row["delta_gromacs_minus_lammps"] = delta
                if ref_value:
                    row["pct_delta_vs_lammps"] = 100.0 * delta / float(ref_value)
            rows.append(row)
        if stage_key == "gromacs_initial_em" or "minimize" in stage_key:
            # EM pressure has no meaningful kinetic-pressure counterpart.  The
            # LAMMPS thermo pressure can retain velocity-state contributions
            # from the restart, so compare the GROMACS EM pressure tensor
            # against a cheap LAMMPS endpoint run-0 virial probe instead.
            for gmx_metric, lammps_metric in em_endpoint_virial_comparisons.items():
                if gmx_metric not in gmx:
                    continue
                ref_value = ref.get(lammps_metric) if ref else None
                gmx_value = gmx.get(gmx_metric)
                row = {
                    "lane": lane,
                    "stage_key": stage_key,
                    "metric": f"{gmx_metric}_vs_lammps_endpoint_virial",
                    "lammps_metric": lammps_metric,
                    "runtime_status": gmx.get("runtime_status"),
                    "lammps_value": ref_value,
                    "gromacs_value": gmx_value,
                    "status": "missing_lammps" if ref is None or ref_value is None else "measured",
                }
                if isinstance(ref_value, (int, float)) and isinstance(gmx_value, (int, float)):
                    delta = float(gmx_value) - float(ref_value)
                    row["delta_gromacs_minus_lammps"] = delta
                    if ref_value:
                        row["pct_delta_vs_lammps"] = 100.0 * delta / float(ref_value)
                rows.append(row)
        for tensor_metric in (
            "pressure_xx_bar_final",
            "pressure_xy_bar_final",
            "pressure_xz_bar_final",
            "pressure_yy_bar_final",
            "pressure_yz_bar_final",
            "pressure_zz_bar_final",
            "virial_xx_kj_mol_final",
            "virial_xy_kj_mol_final",
            "virial_xz_kj_mol_final",
            "virial_yy_kj_mol_final",
            "virial_yz_kj_mol_final",
            "virial_zz_kj_mol_final",
        ):
            if tensor_metric in gmx:
                if tensor_metric in tensor_comparisons:
                    continue
                rows.append(
                    {
                        "lane": lane,
                        "stage_key": stage_key,
                        "metric": tensor_metric,
                        "runtime_status": gmx.get("runtime_status"),
                        "lammps_value": None,
                        "gromacs_value": gmx.get(tensor_metric),
                        "status": "missing_lammps_tensor_reference",
                    }
                )
    return annotate_comparison_rows(rows)


def stage_sort_key(stage_key: str) -> tuple[int, str]:
    if stage_key == "gromacs_initial_em":
        return (0, stage_key)
    m = re.match(r"eq(\d{2})_", stage_key)
    if m:
        return (int(m.group(1)), stage_key)
    if stage_key.startswith("prod"):
        return (14, stage_key)
    return (99, stage_key)


def build_stage_parity_rollup(comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_groups: dict[str, list[str]] = {
        "volume_density": [
            "volume_nm3_final",
            "volume_nm3_mean",
            "density_g_cm3_final",
            "density_g_cm3_mean",
        ],
        "thermal": [
            "temperature_k_final",
            "temperature_k_mean",
            "kinetic_energy_kj_mol_final",
            "kinetic_energy_kj_mol_mean",
        ],
        "energy": [
            "potential_kj_mol_final",
            "potential_kj_mol_mean",
            "total_energy_kj_mol_final",
            "total_energy_kj_mol_mean",
        ],
        "pressure": [
            "pressure_bar_final",
            "pressure_bar_final_vs_lammps_endpoint_virial",
            "pressure_bar_mean",
            "pressure_no_disp_corr_bar_final",
            "pressure_no_disp_corr_bar_mean",
        ],
    }
    virial_metrics = sorted(
        {
            str(row.get("metric"))
            for row in comparison_rows
            if str(row.get("metric", "")).startswith("virial_pressure_")
        }
    )
    metric_groups["virial_pressure"] = virial_metrics

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in comparison_rows:
        lane = str(row.get("lane") or "")
        stage_key = str(row.get("stage_key") or "")
        if not lane or not stage_key:
            continue
        grouped.setdefault((lane, stage_key), []).append(row)

    rollup: list[dict[str, Any]] = []
    for (lane, stage_key), rows in sorted(grouped.items(), key=lambda item: (item[0][0], stage_sort_key(item[0][1]))):
        out: dict[str, Any] = {"lane": lane, "stage_key": stage_key}
        for group_name, metrics in metric_groups.items():
            selected = [row for row in rows if row.get("metric") in metrics]
            measured = [row for row in selected if row.get("status") == "measured"]
            out[f"{group_name}_measured_count"] = len(measured)
            out[f"{group_name}_missing_count"] = len(selected) - len(measured)
            scored: list[tuple[float, dict[str, Any]]] = []
            for row in measured:
                if group_name in {"pressure", "virial_pressure", "energy"}:
                    score = parse_optional_float(row.get("delta_gromacs_minus_lammps"))
                else:
                    score = parse_optional_float(row.get("pct_delta_vs_lammps"))
                if score is not None:
                    scored.append((abs(score), row))
            if not scored:
                continue
            score, worst = max(scored, key=lambda item: item[0])
            out[f"{group_name}_score"] = score
            out[f"{group_name}_score_unit"] = "bar" if group_name in {"pressure", "virial_pressure"} else (
                "kJ/mol" if group_name == "energy" else "pct"
            )
            out[f"{group_name}_worst_metric"] = worst.get("metric")
            out[f"{group_name}_worst_delta"] = worst.get("delta_gromacs_minus_lammps")
            out[f"{group_name}_worst_pct"] = worst.get("pct_delta_vs_lammps")
        rollup.append(out)
    return rollup


def write_summary(
    audit_out: Path,
    status_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    same_start_rows: list[dict[str, Any]],
) -> None:
    stale = [row for row in status_rows if row["runtime_status"] != "current_critical"]
    measured = [row for row in comparison_rows if row["status"] == "measured"]
    measured_current = [row for row in measured if row["runtime_status"] == "current_critical"]
    mechanical_current = [
        row
        for row in measured_current
        if row.get("comparison_use")
        not in {"independent_dynamic_endpoint_diagnostic", "independent_dynamic_window_statistic", "dynamic_diagnostic"}
    ]
    dynamic_endpoint_current = [
        row for row in measured_current if row.get("comparison_use") == "independent_dynamic_endpoint_diagnostic"
    ]
    volume_off = [
        row
        for row in mechanical_current
        if row["metric"] == "volume_nm3_final"
        and row.get("pct_delta_vs_lammps") not in (None, "")
        and abs(float(row["pct_delta_vs_lammps"])) > 0.5
    ]
    density_off = [
        row
        for row in mechanical_current
        if row["metric"] == "density_g_cm3_final"
        and row.get("pct_delta_vs_lammps") not in (None, "")
        and abs(float(row["pct_delta_vs_lammps"])) > 0.5
    ]
    potential_off = [
        row
        for row in mechanical_current
        if row["metric"] == "potential_kj_mol_final"
        and row.get("pct_delta_vs_lammps") not in (None, "")
        and abs(float(row["pct_delta_vs_lammps"])) > 1.0
    ]
    dynamic_endpoint_off = [
        row
        for row in dynamic_endpoint_current
        if row["metric"] in {"volume_nm3_final", "density_g_cm3_final", "potential_kj_mol_final"}
        and row.get("pct_delta_vs_lammps") not in (None, "")
        and abs(float(row["pct_delta_vs_lammps"])) > (1.0 if row["metric"] == "potential_kj_mol_final" else 0.5)
    ]
    same_start_initial = [
        row
        for row in same_start_rows
        if row.get("comparison_use") == "same_start_short_step_root_cause_gate"
    ]
    same_start_short = [
        row
        for row in same_start_rows
        if row.get("comparison_use") == "same_start_short_propagation_diagnostic"
    ]
    same_start_minimize = [
        row
        for row in same_start_rows
        if row.get("comparison_use") == "same_start_minimization_diagnostic"
    ]
    first_initial_gate = sorted(
        same_start_initial,
        key=lambda row: abs(float(row.get("volume_delta") or 0.0)),
    )[:1]
    first_short_gate = sorted(
        same_start_short,
        key=lambda row: float(row.get("probe_duration_ps") or 0.0),
    )[:1]

    lines = [
        "# PolyGen Stage Metric Audit",
        "",
        f"- runtime records: {len(status_rows)}",
        f"- non-current runtime records: {len(stale)}",
        f"- measured LAMMPS/GROMACS scalar comparisons: {len(measured)}",
        f"- measured current scalar comparisons: {len(measured_current)}",
        f"- non-dynamic endpoint diagnostic current rows: {len(mechanical_current)}",
        f"- dynamic endpoint diagnostic rows not used as root-cause gates: {len(dynamic_endpoint_current)}",
        f"- non-dynamic endpoint volume >0.5% rows: {len(volume_off)}",
        f"- non-dynamic endpoint density >0.5% rows: {len(density_off)}",
        f"- non-dynamic endpoint potential >1.0% rows: {len(potential_off)}",
        f"- dynamic endpoint rows over those thresholds, reported separately: {len(dynamic_endpoint_off)}",
        f"- same-start propagation probe rows: {len(same_start_rows)}",
        "",
        "## Root-Cause Boundary",
        "- Root-cause gates use same-start probes, not independently propagated NVT/NPT endpoints.",
        "- Independent NVT/NPT endpoint deltas are trajectory diagnostics only; by themselves they do not prove a pressure-mass or r-RESPA schedule bug.",
    ]
    if first_initial_gate:
        row = first_initial_gate[0]
        lines.append(
            "- Same-start initial gate ({stage}, {ensemble}): duration={duration:g} ps, dV={dv:.6g} nm^3, "
            "dP={dp:.6g} bar, dT={dt:.6g} K, dPE={dpe:.6g} kJ/mol.".format(
                stage=str(row.get("stage_key") or Path(str(row.get("probe_root", ""))).name),
                ensemble=str(row.get("ensemble") or "unknown"),
                duration=float(row.get("probe_duration_ps") or 0.0),
                dv=float(row.get("volume_delta") or 0.0),
                dp=float(row.get("pressure_delta") or 0.0),
                dt=float(row.get("temperature_delta") or 0.0),
                dpe=float(row.get("potential_delta") or 0.0),
            )
        )
    else:
        lines.append("- Same-start initial gate: missing. Do not claim a root cause from endpoint comparisons alone.")
    if first_short_gate:
        row = first_short_gate[0]
        lines.append(
            "- Same-start short propagation ({stage}, {ensemble}): duration={duration:g} ps, dV={dv:.6g} nm^3 "
            "({vpct:.6g}%), dP={dp:.6g} bar, dT={dt:.6g} K, dPE={dpe:.6g} kJ/mol.".format(
                stage=str(row.get("stage_key") or Path(str(row.get("probe_root", ""))).name),
                ensemble=str(row.get("ensemble") or "unknown"),
                duration=float(row.get("probe_duration_ps") or 0.0),
                dv=float(row.get("volume_delta") or 0.0),
                vpct=float(row.get("volume_pct_delta_vs_lammps") or 0.0),
                dp=float(row.get("pressure_delta") or 0.0),
                dt=float(row.get("temperature_delta") or 0.0),
                dpe=float(row.get("potential_delta") or 0.0),
            )
        )
    else:
        lines.append("- Same-start short propagation: missing.")
    if same_start_minimize:
        row = sorted(
            same_start_minimize,
            key=lambda item: (
                item.get("virial_pressure_initial_delta_bar") in (None, ""),
                abs(float(item.get("virial_pressure_initial_delta_bar") or 0.0)),
            ),
        )[0]
        lines.append(
            "- Same-start minimization ({stage}): initial dPE={dpe0:.6g} kJ/mol, "
            "initial dPvir={dpvir0:.6g} bar, final dPE={dpe:.6g} kJ/mol, "
            "final dPvir={dpvir:.6g} bar; full-pressure dP={dpfull:.6g} bar is kinetic-pressure-semantics only.".format(
                stage=str(row.get("stage_key") or Path(str(row.get("probe_root", ""))).name),
                dpe0=float(row.get("potential_initial_delta") or 0.0),
                dpvir0=float(row.get("virial_pressure_initial_delta_bar") or 0.0),
                dpe=float(row.get("potential_delta") or 0.0),
                dpvir=float(row.get("virial_pressure_delta_bar") or 0.0),
                dpfull=float(row.get("pressure_delta") or 0.0),
            )
        )
    if dynamic_endpoint_off:
        first = sorted(
            dynamic_endpoint_off,
            key=lambda row: abs(float(row.get("pct_delta_vs_lammps") or 0.0)),
            reverse=True,
        )[0]
        lines.append(
            "- Largest excluded dynamic endpoint deviation: {stage_key} {metric}, "
            "delta={delta:.6g}, pct={pct:.6g}%.".format(
                stage_key=first["stage_key"],
                metric=first["metric"],
                delta=float(first.get("delta_gromacs_minus_lammps") or 0.0),
                pct=float(first.get("pct_delta_vs_lammps") or 0.0),
            )
        )
    lines.extend([
        "- If same-start gates pass but independent endpoints drift, the next root-cause target is upstream trajectory divergence in the preceding dynamic stage.",
        "",
        "## First Runtime Mismatches",
    ])
    for row in stale[:12]:
        lines.append(f"- {row['lane']} {row['stem']}: {row['first_mismatch']}")
    if not stale:
        lines.append("- none")

    lines.extend(["", "## Largest Non-Dynamic Endpoint Volume/Density/Potential Deviations"])
    offenders = sorted(
        [*volume_off, *density_off, *potential_off],
        key=lambda row: abs(float(row.get("pct_delta_vs_lammps") or 0.0)),
        reverse=True,
    )
    for row in offenders[:20]:
        lines.append(
            "- {lane} {stage_key} {metric}: delta={delta:.6g}, pct={pct:.6g}% "
            "(runtime={runtime})".format(
                lane=row["lane"],
                stage_key=row["stage_key"],
                metric=row["metric"],
                delta=float(row.get("delta_gromacs_minus_lammps") or 0.0),
                pct=float(row.get("pct_delta_vs_lammps") or 0.0),
                runtime=row["runtime_status"],
            )
        )
    if not offenders:
        lines.append("- none")
    lines.extend(["", "## Dynamic Endpoint Deviations Not Used For Root Cause"])
    dynamic_offenders = sorted(
        dynamic_endpoint_off,
        key=lambda row: abs(float(row.get("pct_delta_vs_lammps") or 0.0)),
        reverse=True,
    )
    for row in dynamic_offenders[:20]:
        lines.append(
            "- {lane} {stage_key} {metric}: delta={delta:.6g}, pct={pct:.6g}% "
            "(runtime={runtime}, use={use})".format(
                lane=row["lane"],
                stage_key=row["stage_key"],
                metric=row["metric"],
                delta=float(row.get("delta_gromacs_minus_lammps") or 0.0),
                pct=float(row.get("pct_delta_vs_lammps") or 0.0),
                runtime=row["runtime_status"],
                use=row.get("comparison_use", ""),
            )
        )
    if not dynamic_offenders:
        lines.append("- none")
    lines.extend(["", "## Same-Start Propagation Probes"])
    if same_start_rows:
        for row in sorted(
            same_start_rows,
            key=lambda item: (float(item.get("probe_duration_ps") or 0.0), str(item.get("probe_root"))),
        )[:20]:
            duration = parse_optional_float(row.get("probe_duration_ps")) or 0.0
            if row.get("comparison_use") == "same_start_minimization_diagnostic":
                lines.append(
                    "- {duration:g} ps {cls} ({stage}, minimize): dPE0={dpe0:.6g} kJ/mol, "
                    "dPvir0={dpvir0:.6g} bar, dPEfinal={dpe:.6g} kJ/mol, dPvirfinal={dpvir:.6g} bar, "
                    "dPfull={dp:.6g} bar".format(
                        duration=duration,
                        cls=row.get("probe_class", ""),
                        stage=str(row.get("stage_key") or Path(str(row.get("probe_root", ""))).name),
                        dpe0=float(row.get("potential_initial_delta") or 0.0),
                        dpvir0=float(row.get("virial_pressure_initial_delta_bar") or 0.0),
                        dpe=float(row.get("potential_delta") or 0.0),
                        dpvir=float(row.get("virial_pressure_delta_bar") or 0.0),
                        dp=float(row.get("pressure_delta") or 0.0),
                    )
                )
            else:
                lines.append(
                    "- {duration:g} ps {cls} ({stage}, {ensemble}): dV={dv:.6g} nm^3, dP={dp:.6g} bar, "
                    "dT={dt:.6g} K, dPE={dpe:.6g} kJ/mol, scale={scale}".format(
                        duration=duration,
                        cls=row.get("probe_class", ""),
                        stage=str(row.get("stage_key") or Path(str(row.get("probe_root", ""))).name),
                        ensemble=str(row.get("ensemble") or "unknown"),
                        dv=float(row.get("volume_delta") or 0.0),
                        dp=float(row.get("pressure_delta") or 0.0),
                        dt=float(row.get("temperature_delta") or 0.0),
                        dpe=float(row.get("potential_delta") or 0.0),
                        scale=row.get("pressure_mass_scale", ""),
                    )
                )
    else:
        lines.append("- none found")
    lines.extend(
        [
            "",
            "## Virial/Pressure Tensor Note",
            "",
            "GROMACS pressure tensor and virial tensor are extracted when present in `.edr`. "
            "When `--run-lammps-endpoint-probes` is used, LAMMPS full-pressure and virial-pressure "
            "tensors are collected from existing restart endpoints with cheap `run 0` probes. "
            "The probe tensor values are used only for tensor comparison; scalar energy/volume/temperature "
            "metrics stay sourced from the original dynamics thermo stream.",
        ]
    )
    (audit_out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    notebook = Path(args.notebook).resolve()
    out_root = Path(args.out_root).resolve()
    audit_out = Path(args.audit_out).resolve()
    gmx = Path(args.gmx).resolve()
    lmp = Path(args.lmp).resolve()
    audit_out.mkdir(parents=True, exist_ok=True)

    lammps_work = out_root / "lammps_openmp"
    lammps_log = lammps_work / "equil_from_em.stdout.log"
    lammps_prod_log = lammps_work / "prod_from_relaxed.stdout.log"
    config = load_notebook_config(notebook)
    if args.include_all_stages:
        config["RUN_PRODUCTION"] = True
        config["RUN_PRODUCTION_ONLY"] = False
        config["GMX_STAGE_START_NAME"] = None
        config["GMX_STAGE_STOP_NAME"] = None
    if args.prod_duration_ps is not None:
        config["_AUDIT_PROD_DURATION_PS_OVERRIDE"] = float(args.prod_duration_ps)
    expected = expected_signature_fragment(config, lammps_log)
    if args.expected_pme_order is not None:
        expected["gmx_pme_order"] = args.expected_pme_order
    if args.relax_nonphysics_signature:
        expected = relax_nonphysics_signature(expected)
    expected_stems = expected_gmx_stage_stems(config)
    lammps_metrics = parse_lammps_equil_trace(lammps_log)
    lammps_metrics.update(parse_lammps_equil_trace(lammps_prod_log))
    for extra_log in args.extra_lammps_log:
        lammps_metrics.update(parse_lammps_equil_trace(Path(extra_log).resolve()))
    rebuild_lammps_chunk_aggregate(lammps_metrics, "prod01_nvt_10000ps")
    if args.run_lammps_endpoint_probes:
        probe_logs = run_lammps_endpoint_probes(lmp, lammps_work, audit_out / "lammps_endpoint_virial_probe")
        merge_lammps_endpoint_probes(lammps_metrics, probe_logs)

    status_rows: list[dict[str, Any]] = []
    gmx_metrics: dict[str, dict[str, Any]] = {}
    for lane in args.lanes:
        lane_status, lane_metrics = collect_gromacs_lane(
            gmx,
            out_root / lane,
            lane,
            expected,
            audit_out,
            extract_edr=not args.no_edr,
            expected_stems=expected_stems,
        )
        status_rows.extend(lane_status)
        gmx_metrics.update(lane_metrics)

    lammps_rows = [lammps_metrics[key] for key in sorted(lammps_metrics, key=stage_sort_key)]
    gmx_rows = [gmx_metrics[key] for key in sorted(gmx_metrics)]
    comparison_rows = compare_lammps_gromacs(lammps_metrics, gmx_metrics)
    same_start_rows = load_same_start_probe_rows(args.same_start_probe_roots)
    rollup_rows = build_stage_parity_rollup(comparison_rows)

    write_csv(audit_out / "runtime_status.csv", status_rows)
    write_csv(audit_out / "lammps_stage_metrics.csv", lammps_rows)
    write_csv(audit_out / "gromacs_stage_metrics.csv", gmx_rows)
    write_csv(audit_out / "stage_metric_comparison.csv", comparison_rows)
    write_csv(audit_out / "same_start_probe_summary.csv", same_start_rows)
    write_csv(audit_out / "stage_parity_rollup.csv", rollup_rows)
    dump_json(
        audit_out / "stage_metric_audit.json",
        {
            "notebook": str(notebook),
            "out_root": str(out_root),
            "expected_signature_fragment": expected,
            "relaxed_nonphysics_signature": bool(args.relax_nonphysics_signature),
            "expected_gromacs_stage_stems": expected_stems,
            "lammps_stage_metrics": lammps_metrics,
            "gromacs_stage_metrics": gmx_metrics,
            "runtime_status": status_rows,
            "stage_metric_comparison": comparison_rows,
            "same_start_probe_summary": same_start_rows,
            "stage_parity_rollup": rollup_rows,
        },
    )
    write_summary(audit_out, status_rows, comparison_rows, same_start_rows)

    stale_count = sum(1 for row in status_rows if row["runtime_status"] != "current_critical")
    print(f"wrote {audit_out}")
    print(f"runtime_records={len(status_rows)} stale_or_missing={stale_count}")
    print(f"lammps_stage_records={len(lammps_metrics)} gromacs_stage_records={len(gmx_metrics)}")
    print(f"same_start_probe_records={len(same_start_rows)}")


if __name__ == "__main__":
    main()
