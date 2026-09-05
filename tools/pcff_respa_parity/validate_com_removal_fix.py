#!/usr/bin/env python3
"""Regression-check periodic COM removal against a preserved full-system probe.

Requires numpy/pandas/MDAnalysis (the existing MD conda environment provides them).
The reference directory is the output of the COM-injection diagnosis; all new
TPRs, trajectories, logs and checkpoints are written under --outdir.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
from MDAnalysis.coordinates.TRR import TRRReader


def run(cmd, cwd, env, label, stdin=None):
    proc = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, input=stdin,
                          text=True, capture_output=True)
    (cwd / f'{label}.stdout.log').write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(f'{label} failed in {cwd}: {(proc.stdout + proc.stderr)[-2000:]}')


def read_frames(path):
    with TRRReader(str(path), convert_units=False) as reader:
        return [(ts.time, ts.positions.copy(), ts.velocities.copy()) for ts in reader]


def com_series(gmx, traj, tpr, cwd, env):
    output = cwd / 'com_velocity.xvg'
    run([gmx, 'traj', '-f', traj, '-s', tpr, '-com', '-ov', output,
         '-xvg', 'none', '-fp'], cwd, env, 'com_velocity', '0\n')
    values = np.loadtxt(output)
    return values[:, 0], values[:, 1:4]


def replace(text, key, value):
    return re.sub(r'^' + re.escape(key) + r'\s*=.*$', f'{key} = {value}', text, flags=re.M)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gmx', type=Path, required=True)
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--topology', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu')
    p.add_argument('--update', choices=['cpu', 'gpu'], default='cpu')
    p.add_argument('--ntomp', type=int, default=2)
    p.add_argument('--pme-order', type=int, default=5)
    p.add_argument('--expect-legacy-failure', action='store_true')
    p.add_argument('--check-legacy-checkpoint', action='store_true')
    args = p.parse_args()
    for key in ['gmx', 'reference_dir', 'topology', 'outdir']:
        setattr(args, key, getattr(args, key).resolve())
    args.outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.ntomp), OPENBLAS_NUM_THREADS='1',
               GMX_MAXBACKUP='-1', GMX_PCFF_EXACT_RESPA_ALLOW_LINEAR_COM_REMOVAL='1',
               GMX_DISABLE_MODULAR_SIMULATOR='1')
    env.pop('GMX_PCFF_EWALD_BETA_INV_A', None)
    configuration = dict(vars(args))
    configuration['runtime_environment'] = {
        key: value for key, value in env.items()
        if key.startswith('GMX_') or key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']
    }
    (args.outdir / 'configuration.json').write_text(json.dumps(configuration, default=str, indent=2))
    mdargs = ['-ntmpi', '1', '-ntomp', str(args.ntomp), '-nb', args.backend,
              '-pme', args.backend, '-bonded', args.backend, '-update', args.update,
              '-pin', 'off', '-notunepme']
    rows = []
    for thermostat, suffix in [('NVE', ''), ('NVT', '_nvt')]:
        reference = args.reference_dir / ('com_removal_probe_matched_beta' + suffix)
        for comm in [4, 1000000000]:
            name = f'{thermostat}_nstcomm_{comm}'
            dest = args.outdir / name
            dest.mkdir(exist_ok=True)
            old = reference / f'exact_yes_nstcomm_{comm}'
            mdp = replace((old / 'eval.mdp').read_text(), 'pme-order', args.pme_order)
            (dest / 'eval.mdp').write_text(mdp)
            if not (dest / 'eval.gro').exists():
                run([args.gmx, 'grompp', '-f', 'eval.mdp', '-c', reference / 'state.gro',
                     '-p', args.topology, '-o', 'eval.tpr'], dest, env, 'grompp')
                run([args.gmx, 'mdrun', '-deffnm', 'eval', *mdargs], dest, env, 'mdrun')
            times, com = com_series(args.gmx, dest / 'eval.trr', dest / 'eval.tpr', dest, env)
            residual = float(np.linalg.norm(com[times > 0.0021], axis=1).max())
            final_speed = float(np.linalg.norm(com[-1]))
            periodic_pass = residual < 1e-5
            disabled_pass = final_speed > 1e-3
            passed = (not periodic_pass if args.expect_legacy_failure else periodic_pass) if comm == 4 else disabled_pass
            row = dict(test=name, backend=args.backend, pme_order=args.pme_order,
                       initial_speed=float(np.linalg.norm(com[0])), final_speed=final_speed,
                       residual_speed=residual, passed=bool(passed))
            if comm == 1000000000 and args.backend == 'cpu' and args.pme_order == 5:
                before = read_frames(old / 'eval.trr')[-1]
                after = read_frames(dest / 'eval.trr')[-1]
                row['disabled_position_delta_max_nm'] = float(np.abs(before[1] - after[1]).max())
                row['disabled_velocity_delta_max_nm_ps'] = float(np.abs(before[2] - after[2]).max())
            rows.append(row)
            print(row, flush=True)
        # Compare an uninterrupted 0.2 ps run to a 0.1+0.1 ps checkpoint resume.
        if args.expect_legacy_failure and not args.check_legacy_checkpoint:
            continue
        full = args.outdir / f'{thermostat}_nstcomm_4'
        split = args.outdir / f'{thermostat}_checkpoint'
        split.mkdir(exist_ok=True)
        (split / 'first.mdp').write_text(replace((full / 'eval.mdp').read_text(), 'nsteps', 200))
        if not (split / 'resumed.part0002.gro').exists():
            run([args.gmx, 'grompp', '-f', 'first.mdp', '-c', reference / 'state.gro',
                 '-p', args.topology, '-o', 'first.tpr'], split, env, 'first_grompp')
            run([args.gmx, 'mdrun', '-deffnm', 'first', *mdargs], split, env, 'first_mdrun')
            run([args.gmx, 'convert-tpr', '-s', 'first.tpr', '-nsteps', '400', '-o', 'resumed.tpr'],
                split, env, 'extend_tpr')
            run([args.gmx, 'mdrun', '-s', 'resumed.tpr', '-cpi', 'first.cpt', '-deffnm',
                 'resumed', '-noappend', *mdargs], split, env, 'resumed_mdrun')
        completed = read_frames(full / 'eval.trr')[-1]
        resumed = read_frames(split / 'resumed.part0002.trr')[-1]
        dx = float(np.abs(completed[1] - resumed[1]).max())
        dv = float(np.abs(completed[2] - resumed[2]).max())
        # Exposing TRR through MDAnalysis is float32 even for double files.
        tolerance = 2e-6 if args.backend == 'cpu' else 2e-4
        rows.append(dict(test=thermostat + '_checkpoint', backend=args.backend,
                         position_delta_max_nm=dx, velocity_delta_max_nm_ps=dv,
                         passed=bool(abs(resumed[0] - .2) < 1e-6 and dx < tolerance and dv < tolerance)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(args.outdir / 'checks.csv', index=False)
    if not all(r['passed'] for r in rows):
        raise RuntimeError('One or more COM-removal regression checks failed; see checks.csv')


if __name__ == '__main__':
    main()
