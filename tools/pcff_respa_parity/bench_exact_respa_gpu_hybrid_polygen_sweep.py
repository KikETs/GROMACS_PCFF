from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BRIDGE_OUTPUT = REPO_ROOT.parent / "GROMACS_PCFF-lunar-data-bridge" / "output" / "lammps_data_bridge_polygen_system"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_gate_i_charged_long_npt_conditioning import make_gate_i_npt_mdp


PERFORMANCE_RE = re.compile(
    r"Performance:\s+(?P<ns_per_day>[0-9.]+)\s+(?P<hour_per_ns>[0-9.]+)\s+(?P<ms_per_step>[0-9.]+)"
)

WALLCYCLE_LABELS = (
    ("Neighbor search", "neighbor_search_seconds"),
    ("Launch PP GPU ops.", "launch_pp_gpu_ops_seconds"),
    ("Force", "force_seconds"),
    ("PME mesh", "pme_mesh_seconds"),
    ("PME wait for PP", "pme_wait_for_pp_seconds"),
    ("Wait GPU NB local", "wait_gpu_nb_local_seconds"),
    ("NB X/F buffer ops.", "nb_xf_buffer_ops_seconds"),
    ("Update", "update_seconds"),
    ("Total", "total_wallcycle_seconds"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the exact r-RESPA GPU-hybrid path on the imported PolyGen/LAMMPS "
            "5475-atom PCFF system. This is a performance calibration, not an exactness gate."
        )
    )
    parser.add_argument("--gmx", default=str(REPO_ROOT / "build_gateb_cuda" / "bin" / "gmx"))
    parser.add_argument(
        "--gro",
        default=str(BRIDGE_OUTPUT / "system.gro"),
    )
    parser.add_argument(
        "--top",
        default=str(BRIDGE_OUTPUT / "topol.top"),
    )
    parser.add_argument(
        "--bridge-manifest",
        default=str(BRIDGE_OUTPUT / "bridge_manifest.json"),
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "output" / "exact_respa_gpu_hybrid_polygen_5475_ntomp_sweep_20260422"),
    )
    parser.add_argument("--ntomp-list", nargs="+", type=int, default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--ntmpi", type=int, default=1)
    parser.add_argument("--equil-ps", type=float, default=5.0)
    parser.add_argument("--prod-ps", type=float, default=10.0)
    parser.add_argument("--sample-interval", type=int, default=400)
    parser.add_argument("--seed", type=int, default=91001)
    parser.add_argument("--ld-seed-base", type=int, default=92001)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--tau-t-ps", type=float, default=0.5)
    parser.add_argument("--tau-p-ps", type=float, default=5.0)
    parser.add_argument("--compressibility-bar-inv", type=float, default=4.5e-5)
    parser.add_argument("--pin", choices=("on", "off", "auto"), default="on")
    parser.add_argument("--pinstride", type=int, default=1)
    parser.add_argument("--maxwarn", type=int, default=1)
    return parser.parse_args()


def run(argv: list[str], *, cwd: Path, env: dict[str, str], stdout: Path, stderr: Path) -> float:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    elapsed = time.time() - started
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(argv)}\nSee {stderr}")
    return elapsed


def starts_with_wallcycle_label(line: str, label: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(label) and (len(stripped) == len(label) or stripped[len(label)].isspace())


def extract_wallcycle_seconds(log_text: str, label: str) -> float | None:
    for line in log_text.splitlines():
        if not starts_with_wallcycle_label(line, label):
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            return float(fields[-3])
        except ValueError:
            return None
    return None


def parse_mdrun_log(log_path: Path) -> dict[str, float | None]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = PERFORMANCE_RE.search(text)
    metrics: dict[str, float | None] = {
        "ns_per_day": None,
        "hour_per_ns": None,
        "ms_per_step": None,
    }
    if match is not None:
        metrics["ns_per_day"] = float(match.group("ns_per_day"))
        metrics["hour_per_ns"] = float(match.group("hour_per_ns"))
        metrics["ms_per_step"] = float(match.group("ms_per_step"))
    for label, field in WALLCYCLE_LABELS:
        metrics[field] = extract_wallcycle_seconds(text, label)
    return metrics


def combined_ns_day(total_ps: float, elapsed_seconds: float) -> float | None:
    if elapsed_seconds <= 0:
        return None
    return (total_ps / 1000.0) * 86400.0 / elapsed_seconds


def write_tsv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "ntomp",
        "equil_ns_day",
        "prod_ns_day",
        "combined_ns_day",
        "equil_ms_step",
        "prod_ms_step",
        "equil_force_s",
        "prod_force_s",
        "equil_update_s",
        "prod_update_s",
        "equil_total_s",
        "prod_total_s",
        "combined_elapsed_s",
    ]
    lines = ["\t".join(fields)]
    for row in rows:
        equil = row["equil_metrics"]
        prod = row["prod_metrics"]
        values = [
            row["ntomp"],
            equil["ns_per_day"],
            prod["ns_per_day"],
            row["combined_ns_per_day"],
            equil["ms_per_step"],
            prod["ms_per_step"],
            equil["force_seconds"],
            prod["force_seconds"],
            equil["update_seconds"],
            prod["update_seconds"],
            equil["total_wallcycle_seconds"],
            prod["total_wallcycle_seconds"],
            row["combined_elapsed_seconds"],
        ]
        lines.append(
            "\t".join("" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value)) for value in values)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    gmx = Path(args.gmx).resolve()
    gro = Path(args.gro).resolve()
    top = Path(args.top).resolve()
    bridge_manifest = Path(args.bridge_manifest).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.ntmpi != 1:
        raise ValueError("This GPU-hybrid benchmark is frozen to -ntmpi 1.")
    for path in (gmx, gro, top, bridge_manifest):
        if not path.exists():
            raise FileNotFoundError(path)

    gmx_version = subprocess.run(
        [str(gmx), "--version"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    bridge_payload = json.loads(bridge_manifest.read_text(encoding="utf-8"))

    rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "schema_version": 1,
        "kind": "exact_respa_gpu_hybrid_polygen_5475_ntomp_sweep",
        "claim_boundary": (
            "Short host-local performance calibration only. This artifact does not prove "
            "density/volume convergence, transport readiness, or broad GPU production readiness."
        ),
        "inputs": {
            "gmx": str(gmx),
            "gro": str(gro),
            "top": str(top),
            "bridge_manifest": str(bridge_manifest),
            "ntmpi": args.ntmpi,
            "ntomp_list": args.ntomp_list,
            "equil_ps": args.equil_ps,
            "prod_ps": args.prod_ps,
            "sample_interval": args.sample_interval,
            "pin": args.pin,
            "pinstride": args.pinstride,
            "mdrun_flags": "-nb gpu -pme cpu -bonded cpu -update cpu -dlb no -notunepme",
        },
        "bridge": bridge_payload,
        "host": {
            "gmx_version": gmx_version,
            "nvidia_smi_L": subprocess.run(
                ["nvidia-smi", "-L"], text=True, capture_output=True, check=False
            ).stdout.strip(),
        },
        "runs": rows,
    }

    for ntomp in args.ntomp_list:
        case_root = out_root / f"ntomp{ntomp:02d}"
        inputs_dir = case_root / "inputs"
        logs_dir = case_root / "logs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        equil_mdp = inputs_dir / "equil.mdp"
        prod_mdp = inputs_dir / "prod.mdp"
        equil_tpr = inputs_dir / "equil.tpr"
        prod_tpr = inputs_dir / "prod.tpr"
        equil_deffnm = case_root / "equil"
        prod_deffnm = case_root / "prod"

        equil_mdp.write_text(
            make_gate_i_npt_mdp(
                duration_ps=args.equil_ps,
                sample_interval=args.sample_interval,
                phase="equil",
                seed=args.seed,
                args=args,
                ld_seed=args.ld_seed_base,
            ),
            encoding="utf-8",
        )
        prod_mdp.write_text(
            make_gate_i_npt_mdp(
                duration_ps=args.prod_ps,
                sample_interval=args.sample_interval,
                phase="prod",
                seed=args.seed + 1,
                args=args,
                ld_seed=args.ld_seed_base + 100000,
            ),
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(ntomp)
        env["GMX_DISABLE_MODULAR_SIMULATOR"] = "1"
        env["GMX_MAXBACKUP"] = "-1"

        equil_grompp = [
            str(gmx),
            "grompp",
            "-f",
            str(equil_mdp),
            "-c",
            str(gro),
            "-p",
            str(top),
            "-o",
            str(equil_tpr),
            "-po",
            str(inputs_dir / "equil.mdout.mdp"),
            "-maxwarn",
            str(args.maxwarn),
        ]
        run(equil_grompp, cwd=REPO_ROOT, env=env, stdout=logs_dir / "equil_grompp.stdout", stderr=logs_dir / "equil_grompp.stderr")

        common_mdrun = [
            "-ntmpi",
            str(args.ntmpi),
            "-ntomp",
            str(ntomp),
            "-dlb",
            "no",
            "-nb",
            "gpu",
            "-pme",
            "cpu",
            "-bonded",
            "cpu",
            "-update",
            "cpu",
            "-notunepme",
            "-pin",
            args.pin,
        ]
        if args.pin != "off":
            common_mdrun.extend(["-pinstride", str(args.pinstride)])

        equil_elapsed = run(
            [str(gmx), "mdrun", "-s", str(equil_tpr), "-deffnm", str(equil_deffnm), *common_mdrun],
            cwd=REPO_ROOT,
            env=env,
            stdout=logs_dir / "equil_mdrun.stdout",
            stderr=logs_dir / "equil_mdrun.stderr",
        )

        prod_grompp = [
            str(gmx),
            "grompp",
            "-f",
            str(prod_mdp),
            "-c",
            str(equil_deffnm.with_suffix(".gro")),
            "-t",
            str(equil_deffnm.with_suffix(".cpt")),
            "-p",
            str(top),
            "-o",
            str(prod_tpr),
            "-po",
            str(inputs_dir / "prod.mdout.mdp"),
            "-maxwarn",
            str(args.maxwarn),
        ]
        run(prod_grompp, cwd=REPO_ROOT, env=env, stdout=logs_dir / "prod_grompp.stdout", stderr=logs_dir / "prod_grompp.stderr")

        prod_elapsed = run(
            [str(gmx), "mdrun", "-s", str(prod_tpr), "-deffnm", str(prod_deffnm), *common_mdrun],
            cwd=REPO_ROOT,
            env=env,
            stdout=logs_dir / "prod_mdrun.stdout",
            stderr=logs_dir / "prod_mdrun.stderr",
        )

        row = {
            "ntomp": ntomp,
            "env": {
                "OMP_NUM_THREADS": env["OMP_NUM_THREADS"],
                "GMX_DISABLE_MODULAR_SIMULATOR": env["GMX_DISABLE_MODULAR_SIMULATOR"],
                "GMX_MAXBACKUP": env["GMX_MAXBACKUP"],
            },
            "equil_elapsed_seconds": equil_elapsed,
            "prod_elapsed_seconds": prod_elapsed,
            "combined_elapsed_seconds": equil_elapsed + prod_elapsed,
            "combined_ns_per_day": combined_ns_day(args.equil_ps + args.prod_ps, equil_elapsed + prod_elapsed),
            "equil_metrics": parse_mdrun_log(equil_deffnm.with_suffix(".log")),
            "prod_metrics": parse_mdrun_log(prod_deffnm.with_suffix(".log")),
            "paths": {
                "equil_log": str(equil_deffnm.with_suffix(".log")),
                "prod_log": str(prod_deffnm.with_suffix(".log")),
            },
        }
        rows.append(row)
        manifest["runs"] = rows
        (out_root / "summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_tsv(rows, out_root / "summary.tsv")
        print(
            f"ntomp={ntomp} prod={row['prod_metrics']['ns_per_day']} ns/day "
            f"combined={row['combined_ns_per_day']:.3f} ns/day"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
