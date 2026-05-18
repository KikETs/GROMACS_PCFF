#!/usr/bin/env python3
"""Deploy the lean PolyGen multi-system workspace to SSH workers.

The deploy path intentionally avoids copying historical outputs/build trees.
For remote middle workers, LAMMPS input projects can be prepared locally and
then rsynced so the remote only needs to run LAMMPS/GROMACS, not pysoftk/LUNAR
system construction.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_OUTDIR = Path("GROMACS_PCFF/output/polygen_multisystem_validation_20260512_m1p50")
REMOTES = {
    "remote_mid_a": ("user@100.110.123.78", None),
    "remote_mid_b": ("user@100.120.161.20", None),
}


def workspace_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "GROMACS_PCFF").is_dir() and (candidate / "MY_PAPER_RELATED").is_dir():
            return candidate
    raise RuntimeError(f"Cannot locate workspace from {start}")


def run(cmd: list[str], *, cwd: Path, dry_run: bool = False, check: bool = True) -> None:
    print("$", " ".join(shlex.quote(str(x)) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=check)


def ssh_transport_args(port: str | None) -> list[str]:
    return ["ssh"] if not port else ["ssh", "-p", str(port)]


def rsync(
    src: Path,
    dst: str,
    *,
    port: str | None,
    cwd: Path,
    excludes: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    cmd = ["rsync", "-az", "-e", " ".join(ssh_transport_args(port))]
    for item in excludes or []:
        cmd += ["--exclude", item]
    src_text = str(src)
    if src.is_dir() and not src_text.endswith("/"):
        src_text += "/"
    cmd += [src_text, dst]
    run(cmd, cwd=cwd, dry_run=dry_run)


def refresh_manifest(workspace: Path, outdir: Path, *, dry_run: bool) -> None:
    script = workspace / "GROMACS_PCFF/tools/pcff_respa_parity/polygen_multisystem_manifest.py"
    run([sys.executable, str(script), "--outdir", str(outdir)], cwd=workspace, dry_run=dry_run)


def prepare_role_inputs(
    workspace: Path,
    outdir: Path,
    *,
    role: str,
    run_group: str,
    nproc: int,
    eqfactor: float,
    lmp_binary: str | None,
    molality_override: float | None,
    dry_run: bool,
) -> None:
    worker = workspace / "GROMACS_PCFF/tools/pcff_respa_parity/polygen_multisystem_worker.py"
    cmd = [
        sys.executable,
        str(worker),
        "prepare-inputs",
        "--workspace",
        str(workspace),
        "--outdir",
        str(outdir),
        "--role",
        role,
        "--run-group",
        run_group,
        "--lane",
        "lammps_cpu",
        "--resume-existing",
        "--nproc",
        str(nproc),
        "--eqfactor",
        str(eqfactor),
    ]
    if lmp_binary:
        cmd += ["--lmp-binary", lmp_binary]
    if molality_override is not None:
        cmd += ["--molality-override", str(molality_override)]
    run(cmd, cwd=workspace, dry_run=dry_run)


def make_selected_csvs(
    workspace: Path,
    outdir: Path,
    *,
    roles: list[str],
    run_group: str,
    molality_override: float | None,
    dry_run: bool,
) -> None:
    worker = workspace / "GROMACS_PCFF/tools/pcff_respa_parity/polygen_multisystem_worker.py"
    for role in roles:
        for lane in ["lammps_cpu", "gmx_cpu", "gmx_gpu"]:
            cmd = [
                sys.executable,
                str(worker),
                "make-selected",
                "--workspace",
                str(workspace),
                "--outdir",
                str(outdir),
                "--role",
                role,
                "--run-group",
                run_group,
                "--lane",
                lane,
            ]
            if molality_override is not None:
                cmd += ["--molality-override", str(molality_override)]
            run(cmd, cwd=workspace, dry_run=dry_run)


def deploy_role(
    workspace: Path,
    outdir: Path,
    *,
    role: str,
    run_group: str,
    remote_root: str,
    dry_run: bool,
) -> None:
    target, port = REMOTES[role]
    remote_base = f"{target}:{remote_root.rstrip('/')}/"
    try:
        outdir_rel = outdir.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"outdir must be inside workspace for deployment: {outdir}") from exc
    remote_outdir = f"{remote_root.rstrip('/')}/{outdir_rel}"
    mkdirs = [
        remote_root,
        f"{remote_root}/GROMACS_PCFF/output/jupyter-notebook",
        f"{remote_outdir}/manifest",
        f"{remote_outdir}/selected",
        f"{remote_outdir}/runs_batch/{run_group}/{role}/lammps_cpu",
        f"{remote_outdir}/runs_batch/smoke_{run_group}/{role}/lammps_cpu",
        f"{remote_root}/MY_PAPER_RELATED/LAMMPS_BATCH",
        f"{remote_root}/MY_PAPER_RELATED/LAMMPS_BATCH/extern",
        f"{remote_root}/MY_PAPER_RELATED/GROMACS_PCFF_BATCH",
        f"{remote_root}/MY_PAPER_RELATED/GROMACS_PCFF_BATCH/extern",
        f"{remote_root}/MY_PAPER_RELATED/Decoder_Only/data",
    ]
    run([*ssh_transport_args(port), target, "mkdir -p " + " ".join(mkdirs)], cwd=workspace, dry_run=dry_run)

    rsync(
        workspace / "GROMACS_PCFF/",
        remote_base + "GROMACS_PCFF/",
        port=port,
        cwd=workspace,
        dry_run=dry_run,
        excludes=[
            ".git",
            "build*/",
            "tmp/",
            "tests/reference_results",
            "output/",
            "__pycache__",
        ],
    )
    # The GROMACS source tree has a real source helper directory named
    # src/external/build-fftw.  The generic build* exclude above is needed to
    # avoid historical build trees, so copy this source directory explicitly.
    rsync(
        workspace / "GROMACS_PCFF/src/external/build-fftw/",
        remote_base + "GROMACS_PCFF/src/external/build-fftw/",
        port=port,
        cwd=workspace,
        dry_run=dry_run,
    )
    rsync(
        workspace / "GROMACS_PCFF-lunar-data-bridge/",
        remote_base + "GROMACS_PCFF-lunar-data-bridge/",
        port=port,
        cwd=workspace,
        dry_run=dry_run,
        excludes=[".git", "build*", "tmp", "tests", "output", "__pycache__"],
    )

    for rel in [
        "MY_PAPER_RELATED/LAMMPS_BATCH/BASE/",
        "MY_PAPER_RELATED/LAMMPS_BATCH/batch_utils/",
        "MY_PAPER_RELATED/LAMMPS_BATCH/extern/LUNAR/",
        "MY_PAPER_RELATED/LAMMPS_BATCH/htpmd/",
        "MY_PAPER_RELATED/GROMACS_PCFF_BATCH/BASE/",
        "MY_PAPER_RELATED/GROMACS_PCFF_BATCH/batch_utils/",
        "MY_PAPER_RELATED/GROMACS_PCFF_BATCH/htpmd/",
        "MY_PAPER_RELATED/GROMACS_PCFF_BATCH/data/",
    ]:
        rsync(
            workspace / rel,
            remote_base + rel,
            port=port,
            cwd=workspace,
            dry_run=dry_run,
            excludes=["__pycache__"],
        )

    csv_src = workspace / "MY_PAPER_RELATED/Decoder_Only/data/simulation-trajectory-aggregate.csv"
    rsync(
        csv_src,
        remote_base + "MY_PAPER_RELATED/Decoder_Only/data/simulation-trajectory-aggregate.csv",
        port=port,
        cwd=workspace,
        dry_run=dry_run,
    )

    for rel in [
        "GROMACS_PCFF/output/jupyter-notebook/polygen_multisystem_validation_remote_mid_a.ipynb",
        "GROMACS_PCFF/output/jupyter-notebook/polygen_multisystem_validation_remote_mid_b.ipynb",
        "GROMACS_PCFF/output/jupyter-notebook/polygen_multisystem_deploy_remotes.ipynb",
        "GROMACS_PCFF/output/jupyter-notebook/polygen_multisystem_remote_setup.ipynb",
        "GROMACS_PCFF/output/jupyter-notebook/polygen_multisystem_remote_setup_remote_mid_a.ipynb",
        "GROMACS_PCFF/output/jupyter-notebook/polygen_multisystem_remote_setup_remote_mid_b.ipynb",
        f"{outdir_rel}/manifest/",
        f"{outdir_rel}/selected/",
        f"{outdir_rel}/runs_batch/{run_group}/{role}/lammps_cpu/",
        f"{outdir_rel}/runs_batch/smoke_{run_group}/{role}/lammps_cpu/",
    ]:
        src = workspace / rel
        if src.exists():
            rsync(
                src,
                remote_base + rel,
                port=port,
                cwd=workspace,
                dry_run=dry_run,
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--remote-root", default="~/polygen_validation_workspace")
    p.add_argument("--role", choices=sorted(REMOTES), action="append", default=None)
    p.add_argument("--run-group", default="main20")
    p.add_argument("--prepare-inputs", action="store_true")
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--eqfactor", type=float, default=0.83)
    p.add_argument("--lmp-binary", default=None)
    p.add_argument("--molality-override", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve() if args.workspace else workspace_root(Path.cwd())
    outdir = (workspace / args.outdir).resolve() if not args.outdir.is_absolute() else args.outdir.resolve()
    roles = args.role or ["remote_mid_a", "remote_mid_b"]

    refresh_manifest(workspace, outdir, dry_run=args.dry_run)
    make_selected_csvs(
        workspace,
        outdir,
        roles=roles,
        run_group=args.run_group,
        molality_override=args.molality_override,
        dry_run=args.dry_run,
    )
    if args.prepare_inputs:
        for role in roles:
            prepare_role_inputs(
                workspace,
                outdir,
                role=role,
                run_group=args.run_group,
                nproc=args.nproc,
                eqfactor=args.eqfactor,
                lmp_binary=args.lmp_binary,
                molality_override=args.molality_override,
                dry_run=args.dry_run,
            )
    for role in roles:
        deploy_role(workspace, outdir, role=role, run_group=args.run_group, remote_root=args.remote_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
