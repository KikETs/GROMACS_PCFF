#!/usr/bin/env python3

import csv
import json
import pathlib
import subprocess
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
GMX = ROOT / "build/bin/gmx"
OUTDIR = ROOT / "tests/reference_results/tp1_6_regressions"
WORKDIR = pathlib.Path(__file__).resolve().parent / "work"


def run_command(cmd: list[str], cwd: pathlib.Path, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, input=stdin, text=True, capture_output=True, check=False)


def write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_existing_tp14_fixture() -> dict[str, Any]:
    result = run_command(["python3", "tools/run_tp1_4_pme_proof/test_split.py"], ROOT)
    stdout_path = OUTDIR / "post_fix_tp1_4_existing_split_stdout.txt"
    stderr_path = OUTDIR / "post_fix_tp1_4_existing_split_stderr.txt"
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)

    rows: list[dict[str, float]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("rcut"):
            continue
        parts = [token.strip() for token in stripped.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "rcut": float(parts[0]),
                "sr": float(parts[1]),
                "recip": float(parts[2]),
                "potential": float(parts[3]),
                "force": float(parts[4]),
            }
        )

    return {
        "command": "python3 tools/run_tp1_4_pme_proof/test_split.py",
        "returncode": result.returncode,
        "stdout_artifact": str(stdout_path.relative_to(ROOT)),
        "stderr_artifact": str(stderr_path.relative_to(ROOT)),
        "rows": rows,
        "potential_span": max(row["potential"] for row in rows) - min(row["potential"] for row in rows),
        "force_span": max(row["force"] for row in rows) - min(row["force"] for row in rows),
    }


def run_mixed_type_fixture() -> dict[str, Any]:
    fixture_dir = WORKDIR / "mixed_type_9_6_ljpme"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    write_text(
        fixture_dir / "system.top",
        """[ defaults ]
1 4 yes 1.0 1.0 9.0

[ atomtypes ]
A 12.011 0.0 A 0.34 0.20
B 12.011 0.0 A 0.40 0.10

[ moleculetype ]
MOL 1

[ atoms ]
1 A 1 MOL A1 1 0.0 12.011
2 B 1 MOL B1 1 0.0 12.011

[ system ]
TP1.6 mixed-type 9-6 LJ-PME

[ molecules ]
MOL 1
""",
    )
    write_text(
        fixture_dir / "system.gro",
        """TP1.6 mixed-type 9-6 LJ-PME
2
    1MOL     A1    1   0.000   0.000   0.000
    1MOL     B1    2   0.500   0.000   0.000
   5.000   5.000   5.000
""",
    )
    write_text(
        fixture_dir / "test.mdp",
        """integrator = md
nsteps = 0
cutoff-scheme = Verlet
nstlist = 1
rlist = 1.0
rcoulomb = 1.0
rvdw = 1.0
coulombtype = Cut-off
vdwtype = PME
lj-pme-comb-rule = geometric
ewald-rtol-lj = 1e-5
pbc = xyz
""",
    )

    grompp = run_command(
        [str(GMX), "grompp", "-f", "test.mdp", "-c", "system.gro", "-p", "system.top", "-o", "test.tpr", "-maxwarn", "10"],
        fixture_dir,
    )
    mdrun = run_command([str(GMX), "mdrun", "-s", "test.tpr", "-deffnm", "test", "-nt", "1"], fixture_dir)
    energy = run_command([str(GMX), "energy", "-f", "test.edr", "-o", "energy.xvg"], fixture_dir, "LJ-(SR)\nLJ-recip.\nPotential\n0\n")

    write_text(OUTDIR / "post_fix_mixed_type_9_6_grompp_stdout.txt", grompp.stdout)
    write_text(OUTDIR / "post_fix_mixed_type_9_6_grompp_stderr.txt", grompp.stderr)
    write_text(OUTDIR / "post_fix_mixed_type_9_6_mdrun_stdout.txt", mdrun.stdout)
    write_text(OUTDIR / "post_fix_mixed_type_9_6_mdrun_stderr.txt", mdrun.stderr)
    write_text(OUTDIR / "post_fix_mixed_type_9_6_energy_stdout.txt", energy.stdout)
    write_text(OUTDIR / "post_fix_mixed_type_9_6_energy_stderr.txt", energy.stderr)

    energy_xvg = fixture_dir / "energy.xvg"
    if energy_xvg.exists():
        write_text(OUTDIR / "post_fix_mixed_type_9_6_energy.xvg", energy_xvg.read_text(encoding="utf-8"))

    numeric_line = None
    if energy_xvg.exists():
        for line in energy_xvg.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "@")):
                numeric_line = line
                break

    energy_terms = None
    if numeric_line is not None:
        parts = numeric_line.split()
        energy_terms = {
            "time_ps": float(parts[0]),
            "lj_sr": float(parts[1]),
            "lj_recip": float(parts[2]),
            "potential": float(parts[3]),
        }

    return {
        "command": "gmx grompp && gmx mdrun && gmx energy on mixed-type 9-6 LJ-PME fixture",
        "fixture_dir": str(fixture_dir.relative_to(ROOT)),
        "grompp_returncode": grompp.returncode,
        "mdrun_returncode": mdrun.returncode,
        "energy_returncode": energy.returncode,
        "energy_terms": energy_terms,
    }


def load_pre_results() -> dict[str, Any]:
    return json.loads((OUTDIR / "pre_fix_results.json").read_text(encoding="utf-8"))


def write_comparison(pre_results: dict[str, Any], post_results: dict[str, Any]) -> None:
    existing_pre = pre_results["results"]["existing_tp1_4_split_scan"]
    existing_post = post_results["results"]["existing_tp1_4_split_scan"]
    mixed_pre = pre_results["results"]["mixed_type_9_6_ljpme_startup"]
    mixed_post = post_results["results"]["mixed_type_9_6_ljpme_startup"]

    with (OUTDIR / "regression_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fixture", "metric", "pre", "post", "status"])
        writer.writerow(
            [
                "existing_tp1_4_split_scan",
                "returncode",
                existing_pre["returncode"],
                existing_post["returncode"],
                "unchanged",
            ]
        )
        writer.writerow(
            [
                "existing_tp1_4_split_scan",
                "potential_span",
                f"{existing_pre['potential_span']:.6f}",
                f"{existing_post['potential_span']:.6f}",
                "unchanged_unfixed",
            ]
        )
        writer.writerow(
            [
                "existing_tp1_4_split_scan",
                "force_span",
                f"{existing_pre['force_span']:.6f}",
                f"{existing_post['force_span']:.6f}",
                "unchanged_unfixed",
            ]
        )
        writer.writerow(
            [
                "mixed_type_9_6_ljpme_startup",
                "mdrun_returncode",
                mixed_pre["mdrun_returncode"],
                mixed_post["mdrun_returncode"],
                "fixed" if mixed_pre["mdrun_returncode"] != 0 and mixed_post["mdrun_returncode"] == 0 else "unchanged",
            ]
        )
        writer.writerow(
            [
                "mixed_type_9_6_ljpme_startup",
                "potential",
                "",
                f"{mixed_post['energy_terms']['potential']:.6f}",
                "fixed",
            ]
        )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    WORKDIR.mkdir(parents=True, exist_ok=True)

    post_results = {
        "milestone": "TP1.6",
        "results": {
            "existing_tp1_4_split_scan": run_existing_tp14_fixture(),
            "mixed_type_9_6_ljpme_startup": run_mixed_type_fixture(),
        },
    }
    (OUTDIR / "post_fix_results.json").write_text(json.dumps(post_results, indent=2), encoding="utf-8")
    write_comparison(load_pre_results(), post_results)


if __name__ == "__main__":
    main()
