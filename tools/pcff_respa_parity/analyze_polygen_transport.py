#!/usr/bin/env python3
"""Transport analysis for the PolyGen PCFF LAMMPS/GROMACS benchmark outputs.

The script builds a compact ion-only trajectory cache from the production
outputs, then computes MSD, diffusion, NE conductivity, HTP-MD-style cNE0,
cluster-resolved cNE diagnostics, and collective Einstein conductivity.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


E_CHARGE = 1.6021766209e-19
KB = 1.38064852e-23
ANGSTROM = 1.0e-10
CENTIMETER = 1.0e-2
PICOSECOND = 1.0e-12

DEFAULT_ROOT = Path("output/polygen_pcff_gromacs_initial_em_notebook")
NITROGEN_MASS = 14.0067
ANION_TRACE_MASSES = (14.0067, 15.9994, 32.064)
ELEMENT_MASS_TOLERANCE = 0.2


@dataclass(frozen=True)
class LaneSpec:
    key: str
    kind: str
    path: Path
    gmx: Path | None = None


@dataclass
class Topology:
    atom_ids: np.ndarray
    mol_ids: np.ndarray
    types: np.ndarray
    masses: np.ndarray
    charges: np.ndarray
    ion_atom_ids: np.ndarray
    ion_indices0: np.ndarray
    cat_mols: np.ndarray
    anion_mols: np.ndarray
    mol_charge: dict[int, float]
    mol_mass: dict[int, float]
    mol_to_sel_indices: dict[int, np.ndarray]
    cat_atom_sel: np.ndarray
    anion_ref_atom_sel: np.ndarray
    anion_trace_sel_by_mol: dict[int, np.ndarray]


def iter_progress(items: Iterable, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit="chunk")


def natural_key(path: Path) -> tuple:
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name))


def parse_first_lammps_frame(path: Path) -> Topology:
    with path.open() as fh:
        natoms = None
        fields = None
        while True:
            line = fh.readline()
            if not line:
                raise RuntimeError(f"No atom frame found in {path}")
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                natoms = int(fh.readline())
            elif line.startswith("ITEM: ATOMS"):
                fields = line.split()[2:]
                break
        if natoms is None or fields is None:
            raise RuntimeError(f"Malformed LAMMPS dump header in {path}")
        idx = {name: i for i, name in enumerate(fields)}
        required = {"id", "mol", "type", "mass", "q"}
        missing = required - set(idx)
        if missing:
            raise RuntimeError(f"{path} is missing fields: {sorted(missing)}")
        rows = []
        for _ in range(natoms):
            parts = fh.readline().split()
            rows.append(
                (
                    int(parts[idx["id"]]),
                    int(parts[idx["mol"]]),
                    int(parts[idx["type"]]),
                    float(parts[idx["mass"]]),
                    float(parts[idx["q"]]),
                )
            )
    rows.sort(key=lambda x: x[0])
    atom_ids = np.array([r[0] for r in rows], dtype=np.int32)
    if not np.array_equal(atom_ids, np.arange(1, len(atom_ids) + 1, dtype=np.int32)):
        raise RuntimeError("Atom ids are not a contiguous 1-based sequence")
    mol_ids = np.array([r[1] for r in rows], dtype=np.int32)
    types = np.array([r[2] for r in rows], dtype=np.int16)
    masses = np.array([r[3] for r in rows], dtype=np.float64)
    charges = np.array([r[4] for r in rows], dtype=np.float64)

    mol_charge: dict[int, float] = defaultdict(float)
    mol_mass: dict[int, float] = defaultdict(float)
    for mol, mass, charge in zip(mol_ids, masses, charges):
        mol_charge[int(mol)] += float(charge)
        mol_mass[int(mol)] += float(mass)

    cat_mols = np.array(sorted(m for m, q in mol_charge.items() if q > 0.1), dtype=np.int32)
    anion_mols = np.array(sorted(m for m, q in mol_charge.items() if q < -0.1), dtype=np.int32)
    if len(cat_mols) == 0 or len(anion_mols) == 0:
        raise RuntimeError("Could not identify charged molecules from first dump frame")

    # Atom type ids are assigned after the polymer types and therefore vary
    # between generated systems.  Select ions by charged-molecule membership
    # instead of the legacy fixed type ids 90-95.
    cat_mol_set = set(map(int, cat_mols))
    anion_mol_set = set(map(int, anion_mols))
    ion_mask = np.isin(mol_ids, np.concatenate((cat_mols, anion_mols)))
    ion_indices0 = np.nonzero(ion_mask)[0].astype(np.int32)
    ion_atom_ids = atom_ids[ion_indices0]

    mol_to_sel: dict[int, list[int]] = defaultdict(list)
    anion_trace_sel_by_mol: dict[int, list[int]] = defaultdict(list)
    cat_atom_sel = []
    anion_ref_atom_sel = []
    for sel_i, atom_i in enumerate(ion_indices0):
        mol = int(mol_ids[atom_i])
        mass = float(masses[atom_i])
        mol_to_sel[mol].append(sel_i)
        if mol in cat_mol_set:
            cat_atom_sel.append(sel_i)
        if mol in anion_mol_set and abs(mass - NITROGEN_MASS) <= ELEMENT_MASS_TOLERANCE:
            anion_ref_atom_sel.append(sel_i)
        if mol in anion_mol_set and any(
            abs(mass - target) <= ELEMENT_MASS_TOLERANCE for target in ANION_TRACE_MASSES
        ):
            anion_trace_sel_by_mol[mol].append(sel_i)

    bad_cations = [mol for mol in cat_mols if len(mol_to_sel[int(mol)]) != 1]
    bad_anion_refs = [
        mol
        for mol in anion_mols
        if sum(
            abs(float(masses[ion_indices0[sel_i]]) - NITROGEN_MASS) <= ELEMENT_MASS_TOLERANCE
            for sel_i in mol_to_sel[int(mol)]
        )
        != 1
    ]
    bad_anion_traces = [mol for mol in anion_mols if not anion_trace_sel_by_mol[int(mol)]]
    if bad_cations:
        raise RuntimeError(f"Expected monatomic cations, invalid molecule ids: {bad_cations[:10]}")
    if bad_anion_refs:
        raise RuntimeError(f"Expected one TFSI nitrogen reference atom, invalid molecule ids: {bad_anion_refs[:10]}")
    if bad_anion_traces:
        raise RuntimeError(f"Missing TFSI N/O/S trace atoms for molecule ids: {bad_anion_traces[:10]}")

    return Topology(
        atom_ids=atom_ids,
        mol_ids=mol_ids,
        types=types,
        masses=masses,
        charges=charges,
        ion_atom_ids=ion_atom_ids,
        ion_indices0=ion_indices0,
        cat_mols=cat_mols,
        anion_mols=anion_mols,
        mol_charge=dict(mol_charge),
        mol_mass=dict(mol_mass),
        mol_to_sel_indices={m: np.array(v, dtype=np.int32) for m, v in mol_to_sel.items()},
        cat_atom_sel=np.array(cat_atom_sel, dtype=np.int32),
        anion_ref_atom_sel=np.array(anion_ref_atom_sel, dtype=np.int32),
        anion_trace_sel_by_mol={m: np.array(v, dtype=np.int32) for m, v in anion_trace_sel_by_mol.items()},
    )


def read_lammps_selected_frame(fh, topology: Topology):
    line = fh.readline()
    if not line:
        return None
    if not line.startswith("ITEM: TIMESTEP"):
        raise RuntimeError(f"Expected TIMESTEP, got: {line.strip()}")
    local_step = int(fh.readline())
    if not fh.readline().startswith("ITEM: NUMBER OF ATOMS"):
        raise RuntimeError("Malformed LAMMPS dump: missing atom count")
    natoms = int(fh.readline())
    if natoms != len(topology.atom_ids):
        raise RuntimeError(f"LAMMPS atom count changed: {natoms} != {len(topology.atom_ids)}")
    if not fh.readline().startswith("ITEM: BOX BOUNDS"):
        raise RuntimeError("Malformed LAMMPS dump: missing box")
    bounds = np.array([[float(x) for x in fh.readline().split()[:2]] for _ in range(3)], dtype=np.float64)
    box_ang = bounds[:, 1] - bounds[:, 0]
    atom_header = fh.readline()
    if not atom_header.startswith("ITEM: ATOMS"):
        raise RuntimeError("Malformed LAMMPS dump: missing atom fields")
    fields = atom_header.split()[2:]
    idx = {name: i for i, name in enumerate(fields)}
    required = {"id", "x", "y", "z", "ix", "iy", "iz"}
    missing = required - set(idx)
    if missing:
        raise RuntimeError(f"LAMMPS dump missing coordinate/image fields: {sorted(missing)}")

    nsel = len(topology.ion_atom_ids)
    wrapped = np.empty((nsel, 3), dtype=np.float32)
    unwrapped = np.empty((nsel, 3), dtype=np.float32)
    sel_pos_by_atom = {int(atom_id): i for i, atom_id in enumerate(topology.ion_atom_ids)}
    seen = 0
    for _ in range(natoms):
        parts = fh.readline().split()
        atom_id = int(parts[idx["id"]])
        sel_i = sel_pos_by_atom.get(atom_id)
        if sel_i is None:
            continue
        xyz_ang = np.array([float(parts[idx["x"]]), float(parts[idx["y"]]), float(parts[idx["z"]])])
        img = np.array([int(parts[idx["ix"]]), int(parts[idx["iy"]]), int(parts[idx["iz"]])])
        wrapped[sel_i] = xyz_ang / 10.0
        unwrapped[sel_i] = (bounds[:, 0] + xyz_ang + img * box_ang) / 10.0
        seen += 1
    if seen != nsel:
        raise RuntimeError(f"Only read {seen}/{nsel} selected ion atoms from LAMMPS frame")
    return local_step, box_ang / 10.0, wrapped, unwrapped


def load_lammps_cache(lane: LaneSpec, topology: Topology, cache_path: Path, stride_ps: float, max_chunks: int | None):
    files = sorted((lane.path / "prod_traj").glob("prod_chunk*.lammpstrj"), key=natural_key)
    if max_chunks:
        files = files[:max_chunks]
    if not files:
        raise RuntimeError(f"No LAMMPS production dumps found under {lane.path / 'prod_traj'}")
    times = []
    boxes = []
    wrapped_frames = []
    unwrapped_frames = []
    global_frame = 0
    for chunk_i, path in enumerate(iter_progress(files, "read lammps")):
        with path.open() as fh:
            local_frame = 0
            while True:
                frame = read_lammps_selected_frame(fh, topology)
                if frame is None:
                    break
                _, box, wrapped, unwrapped = frame
                if chunk_i > 0 and local_frame == 0:
                    local_frame += 1
                    continue
                times.append(global_frame * stride_ps)
                boxes.append(box)
                wrapped_frames.append(wrapped)
                unwrapped_frames.append(unwrapped)
                global_frame += 1
                local_frame += 1
    save_cache(cache_path, lane, topology, np.array(times), np.stack(boxes), np.stack(wrapped_frames), np.stack(unwrapped_frames))


def write_index_file(path: Path, atom_ids: np.ndarray):
    lines = ["[ ion_atoms ]"]
    row = []
    for atom_id in atom_ids:
        row.append(str(int(atom_id)))
        if len(row) == 15:
            lines.append(" ".join(row))
            row = []
    if row:
        lines.append(" ".join(row))
    path.write_text("\n".join(lines) + "\n")


def read_xvg_matrix(path: Path) -> np.ndarray:
    data = []
    with path.open() as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "@")):
                continue
            data.append([float(x) for x in line.split()])
    if not data:
        raise RuntimeError(f"No data in {path}")
    return np.array(data, dtype=np.float64)


def unwrap_gromacs(wrapped: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    unwrapped = np.empty_like(wrapped, dtype=np.float32)
    unwrapped[0] = wrapped[0]
    prev_wrapped = wrapped[0].astype(np.float64)
    prev_unwrapped = wrapped[0].astype(np.float64)
    for i in range(1, len(wrapped)):
        box = boxes[i].astype(np.float64)
        delta = wrapped[i].astype(np.float64) - prev_wrapped
        delta -= np.rint(delta / box) * box
        current = prev_unwrapped + delta
        unwrapped[i] = current
        prev_wrapped = wrapped[i].astype(np.float64)
        prev_unwrapped = current
    return unwrapped


def load_gromacs_cache(lane: LaneSpec, topology: Topology, cache_path: Path, stride_ps: float, max_chunks: int | None):
    if lane.gmx is None:
        raise RuntimeError(f"No gmx executable configured for lane {lane.key}")
    if not lane.gmx.exists():
        raise RuntimeError(f"gmx executable does not exist: {lane.gmx}")
    xtcs = sorted(lane.path.glob("14_prod01_nvt_10000ps_chunk*.xtc"), key=natural_key)
    tprs = sorted(lane.path.glob("14_prod01_nvt_10000ps_chunk*.tpr"), key=natural_key)
    if max_chunks:
        xtcs = xtcs[:max_chunks]
        tprs = tprs[:max_chunks]
    if len(xtcs) != len(tprs) or not xtcs:
        raise RuntimeError(f"GROMACS production xtc/tpr count mismatch in {lane.path}: {len(xtcs)} vs {len(tprs)}")
    times = []
    boxes = []
    wrapped_frames = []
    with tempfile.TemporaryDirectory(prefix="polygen_transport_gmx_") as tmpdir_s:
        tmpdir = Path(tmpdir_s)
        ndx = tmpdir / "ion_atoms.ndx"
        write_index_file(ndx, topology.ion_atom_ids)
        global_frame = 0
        for chunk_i, (xtc, tpr) in enumerate(iter_progress(list(zip(xtcs, tprs)), f"read {lane.key}")):
            coord_xvg = tmpdir / f"{lane.key}_{chunk_i:04d}_coord.xvg"
            box_xvg = tmpdir / f"{lane.key}_{chunk_i:04d}_box.xvg"
            cmd = [
                str(lane.gmx),
                "traj",
                "-f",
                str(xtc),
                "-s",
                str(tpr),
                "-n",
                str(ndx),
                "-ox",
                str(coord_xvg),
                "-ob",
                str(box_xvg),
                "-xvg",
                "none",
                "-fp",
            ]
            proc = subprocess.run(
                cmd,
                input="0\n",
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                cwd=Path.cwd(),
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"gmx traj failed for {xtc}\n{proc.stderr[-4000:]}")
            coord = read_xvg_matrix(coord_xvg)
            box = read_xvg_matrix(box_xvg)
            if coord.shape[0] != box.shape[0]:
                raise RuntimeError(f"coord/box frame mismatch for {xtc}: {coord.shape[0]} vs {box.shape[0]}")
            if coord.shape[1] != 1 + 3 * len(topology.ion_atom_ids):
                raise RuntimeError(f"Unexpected coord columns for {xtc}: {coord.shape[1]}")
            local_times = coord[:, 0]
            local_coords = coord[:, 1:].reshape(coord.shape[0], len(topology.ion_atom_ids), 3).astype(np.float32)
            local_boxes = box[:, 1:4].astype(np.float32)
            start = 1 if chunk_i > 0 else 0
            for frame_i in range(start, coord.shape[0]):
                times.append(global_frame * stride_ps)
                wrapped_frames.append(local_coords[frame_i])
                boxes.append(local_boxes[frame_i])
                global_frame += 1
            coord_xvg.unlink(missing_ok=True)
            box_xvg.unlink(missing_ok=True)
    wrapped = np.stack(wrapped_frames)
    boxes_a = np.stack(boxes)
    unwrapped = unwrap_gromacs(wrapped, boxes_a)
    save_cache(cache_path, lane, topology, np.array(times), boxes_a, wrapped, unwrapped)


def save_cache(
    cache_path: Path,
    lane: LaneSpec,
    topology: Topology,
    time_ps: np.ndarray,
    box_nm: np.ndarray,
    wrapped_nm: np.ndarray,
    unwrapped_nm: np.ndarray,
):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        lane=lane.key,
        time_ps=time_ps.astype(np.float64),
        box_nm=box_nm.astype(np.float32),
        wrapped_nm=wrapped_nm.astype(np.float32),
        unwrapped_nm=unwrapped_nm.astype(np.float32),
        ion_atom_ids=topology.ion_atom_ids.astype(np.int32),
        ion_types=topology.types[topology.ion_indices0].astype(np.int16),
        ion_mol_ids=topology.mol_ids[topology.ion_indices0].astype(np.int32),
        ion_masses=topology.masses[topology.ion_indices0].astype(np.float64),
        ion_charges=topology.charges[topology.ion_indices0].astype(np.float64),
        cat_mols=topology.cat_mols.astype(np.int32),
        anion_mols=topology.anion_mols.astype(np.int32),
    )


def validate_cache_topology(cache, topology: Topology):
    """Reject old type-ID based selections instead of reusing the wrong ions."""
    expected = {
        "ion_atom_ids": topology.ion_atom_ids,
        "ion_mol_ids": topology.mol_ids[topology.ion_indices0],
        "ion_masses": topology.masses[topology.ion_indices0],
        "ion_charges": topology.charges[topology.ion_indices0],
        "cat_mols": topology.cat_mols,
        "anion_mols": topology.anion_mols,
    }
    for key, values in expected.items():
        if key not in cache or not np.array_equal(cache[key], values):
            raise ValueError(
                f"Trajectory cache {key} does not match the current ion topology; "
                "rebuild the cache with --force-cache before analysis."
            )


def load_or_build_cache(
    lane: LaneSpec,
    topology: Topology,
    outdir: Path,
    stride_ps: float,
    max_chunks: int | None,
    force: bool,
) -> Path:
    suffix = f"_{max_chunks}chunks" if max_chunks else ""
    cache_path = outdir / "cache" / f"{lane.key}{suffix}_ion_trajectory.npz"
    if cache_path.exists() and not force:
        with np.load(cache_path) as cache:
            validate_cache_topology(cache, topology)
        return cache_path
    if lane.kind == "lammps":
        load_lammps_cache(lane, topology, cache_path, stride_ps, max_chunks)
    elif lane.kind == "gromacs":
        load_gromacs_cache(lane, topology, cache_path, stride_ps, max_chunks)
    else:
        raise RuntimeError(f"Unknown lane kind: {lane.kind}")
    return cache_path


def molecule_com_from_cache(cache, topology: Topology, drift_remove: bool = True):
    unwrapped = cache["unwrapped_nm"].astype(np.float64)
    masses = cache["ion_masses"].astype(np.float64)
    mol_ids = cache["ion_mol_ids"].astype(np.int32)
    cat_com = []
    anion_com = []
    for mol in topology.cat_mols:
        idx = np.nonzero(mol_ids == mol)[0]
        cat_com.append(np.average(unwrapped[:, idx, :], weights=masses[idx], axis=1))
    for mol in topology.anion_mols:
        idx = np.nonzero(mol_ids == mol)[0]
        anion_com.append(np.average(unwrapped[:, idx, :], weights=masses[idx], axis=1))
    cat = np.transpose(np.array(cat_com), (1, 0, 2))
    anion = np.transpose(np.array(anion_com), (1, 0, 2))
    if drift_remove:
        total_mass = masses.sum()
        drift = np.einsum("tai,a->ti", unwrapped, masses) / total_mass
        drift_delta = drift - drift[0]
        cat = cat - drift_delta[:, None, :]
        anion = anion - drift_delta[:, None, :]
    return cat, anion


def selected_atom_coords(cache, topology: Topology, sel: np.ndarray, drift_remove: bool = False) -> np.ndarray:
    coords = cache["unwrapped_nm"].astype(np.float64)[:, sel, :]
    if drift_remove:
        masses = cache["ion_masses"].astype(np.float64)
        drift = np.einsum("tai,a->ti", cache["unwrapped_nm"].astype(np.float64), masses) / masses.sum()
        coords = coords - (drift - drift[0])[:, None, :]
    return coords


def msd_curve(coords_nm: np.ndarray, time_ps: np.ndarray, n_lags: int = 220):
    n_frames = coords_nm.shape[0]
    max_lag = n_frames - 1
    if max_lag < 4:
        raise RuntimeError("Not enough frames for MSD")
    raw_lags = np.unique(np.rint(np.geomspace(1, max_lag, min(n_lags, max_lag))).astype(int))
    raw_lags = raw_lags[raw_lags > 0]
    msd = np.empty(len(raw_lags), dtype=np.float64)
    for i, lag in enumerate(raw_lags):
        diff = coords_nm[lag:] - coords_nm[:-lag]
        msd[i] = np.mean(np.sum(diff * diff, axis=-1))
    dt_ps = float(np.median(np.diff(time_ps)))
    return raw_lags * dt_ps, msd


def fit_diffusion_from_msd(time_lag_ps: np.ndarray, msd_nm2: np.ndarray, fit_fraction=(0.2, 0.8)):
    lo = fit_fraction[0] * time_lag_ps[-1]
    hi = fit_fraction[1] * time_lag_ps[-1]
    mask = (time_lag_ps >= lo) & (time_lag_ps <= hi)
    if mask.sum() < 3:
        raise RuntimeError("Not enough MSD points in fit window")
    x = time_lag_ps[mask]
    y = msd_nm2[mask]
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # nm^2/ps = 1e-6 m^2/s = 1e-2 cm^2/s
    diff_cm2_s = slope / 6.0 * 1.0e-2
    return {
        "diffusion_cm2_s": float(diff_cm2_s),
        "slope_nm2_ps": float(slope),
        "intercept_nm2": float(intercept),
        "fit_start_ps": float(lo),
        "fit_end_ps": float(hi),
        "r2": float(r2),
    }


def endpoint_diffusion_htp(coords_nm: np.ndarray, time_ps: np.ndarray):
    disp_nm = coords_nm[-1] - coords_nm[0]
    msd_nm2 = float(np.mean(np.sum(disp_nm * disp_nm, axis=-1)))
    elapsed_ps = float(time_ps[-1] - time_ps[0])
    diff_cm2_s = msd_nm2 / 6.0 / elapsed_ps * 1.0e-2
    return diff_cm2_s, msd_nm2


def conductivity_ne(diff_cat_cm2_s: float, diff_anion_cm2_s: float, n_cat: int, n_anion: int, volume_nm3: float, temperature: float, z: float):
    volume_m3 = volume_nm3 * 1.0e-27
    diff_cat_m2_s = diff_cat_cm2_s * 1.0e-4
    diff_anion_m2_s = diff_anion_cm2_s * 1.0e-4
    sigma_s_m = E_CHARGE**2 / (KB * temperature * volume_m3) * (
        n_cat * z * z * diff_cat_m2_s + n_anion * z * z * diff_anion_m2_s
    )
    denom = n_cat * z * z * diff_cat_m2_s + n_anion * z * z * diff_anion_m2_s
    t_plus = (n_cat * z * z * diff_cat_m2_s / denom) if denom else float("nan")
    return sigma_s_m / 100.0, t_plus


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def min_image_delta(a: np.ndarray, b: np.ndarray, box: np.ndarray) -> np.ndarray:
    delta = b - a
    delta -= np.rint(delta / box) * box
    return delta


def population_matrix_size(topology: Topology, requested: int | None) -> int:
    """Include every possible ion composition by default; size is exclusive."""
    if requested is None:
        return max(len(topology.cat_mols), len(topology.anion_mols)) + 1
    if requested < 1:
        raise ValueError("--max-cluster must be a positive matrix dimension")
    return requested


def htp_atom_population_matrix(cache, topology: Topology, max_cluster: int, cutoff_nm: float, sample_stride: int):
    max_cluster = population_matrix_size(topology, max_cluster)
    if sample_stride < 1:
        raise ValueError("cluster sample stride must be positive")
    wrapped = cache["wrapped_nm"].astype(np.float64)
    boxes = cache["box_nm"].astype(np.float64)
    cat_sel = np.asarray(topology.cat_atom_sel, dtype=np.int32)
    anion_trace_sel = np.concatenate(
        [topology.anion_trace_sel_by_mol[int(mol)] for mol in topology.anion_mols]
    )
    trace_sel = np.concatenate((cat_sel, anion_trace_sel))
    cat_sel_set = set(map(int, cat_sel))
    anion_ref_sel_set = set(map(int, topology.anion_ref_atom_sel))
    pop = np.zeros((max_cluster, max_cluster), dtype=np.float64)
    frame_count = 0
    for frame_i in range(0, wrapped.shape[0], sample_stride):
        coords = np.mod(wrapped[frame_i, trace_sel, :], boxes[frame_i])
        box = boxes[frame_i]
        uf = UnionFind(len(trace_sel))
        tree = cKDTree(coords, boxsize=box)
        for i, j in tree.query_pairs(cutoff_nm):
            uf.union(int(i), int(j))
        counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for i, sel_i in enumerate(trace_sel):
            root = uf.find(i)
            if int(sel_i) in cat_sel_set:
                counts[root][0] += 1
            elif int(sel_i) in anion_ref_sel_set:
                counts[root][1] += 1
        for ncat, nani in counts.values():
            if ncat >= max_cluster or nani >= max_cluster:
                required = max(ncat, nani) + 1
                raise ValueError(
                    f"Cluster ({ncat} cations, {nani} anions) in frame {frame_i} "
                    f"does not fit --max-cluster {max_cluster}; refusing to drop ions. "
                    f"Use automatic sizing or --max-cluster >= {required}."
                )
            pop[ncat, nani] += 1.0
        frame_count += 1
    if not frame_count:
        raise ValueError("No frames available for cluster population analysis")
    pop /= frame_count
    i, j = np.indices(pop.shape)
    retained = np.array([(pop * i).sum(), (pop * j).sum()])
    expected = np.array([len(topology.cat_mols), len(topology.anion_mols)])
    if not np.allclose(retained, expected, rtol=0, atol=1e-8):
        raise RuntimeError(f"Cluster population lost ions: retained {retained}, expected {expected}")
    return pop, frame_count


def molecular_clusters_from_precomputed(coords, box, cat_li_sel, an_trace, cutoff_nm: float):
    n_cat = len(cat_li_sel)
    n_an = len(an_trace)
    uf = UnionFind(n_cat + n_an)
    cutoff2 = cutoff_nm * cutoff_nm
    trace_coords = []
    trace_owner = []
    for ai, tr_sel in enumerate(an_trace):
        trace_coords.append(coords[tr_sel])
        trace_owner.extend([ai] * len(tr_sel))
    trace_coords_a = np.mod(np.vstack(trace_coords), box)
    trace_owner_a = np.array(trace_owner, dtype=np.int32)
    tree = cKDTree(trace_coords_a, boxsize=box)
    for ci, li_sel in enumerate(cat_li_sel):
        li = np.mod(coords[li_sel], box)
        hits = tree.query_ball_point(li, cutoff_nm)
        if hits:
            for ai in np.unique(trace_owner_a[hits]):
                uf.union(ci, n_cat + int(ai))
    groups: dict[int, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for ci in range(n_cat):
        groups[uf.find(ci)][0].append(ci)
    for ai in range(n_an):
        groups[uf.find(n_cat + ai)][1].append(ai)
    return list(groups.values())


def cne0_htp_conductivity(pop_mat, li_diff_cm2_s, anion_diff_cm2_s, volume_nm3, temperature, max_cluster: int):
    volume_cm3 = volume_nm3 * (1.0e-7) ** 3
    cond = 0.0
    tn_num = 0.0
    tn_den = 0.0
    for i in range(max_cluster):
        for j in range(max_cluster):
            if i == j:
                continue
            charge2 = float((i - j) ** 2)
            if i > j:
                diff = li_diff_cm2_s
            else:
                diff = anion_diff_cm2_s
            weight = float(pop_mat[i, j])
            cond += E_CHARGE**2 / volume_cm3 / KB / temperature * charge2 * weight * diff
            tn_num += i * (i - j) * weight * diff
            tn_den += charge2 * weight * diff
    return cond, (tn_num / tn_den if tn_den else float("nan"))


def collective_conductivity(cat_com_nm, anion_com_nm, time_ps, volume_nm3, temperature, z: float):
    lag_ps, _ = msd_curve(cat_com_nm[:, :1, :], time_ps, n_lags=220)
    lag_idx = np.rint(lag_ps / np.median(np.diff(time_ps))).astype(int)
    total_msd = []
    pp_msd = []
    mm_msd = []
    pm_cross = []
    for lag in lag_idx:
        cat_delta = cat_com_nm[lag:] - cat_com_nm[:-lag]
        an_delta = anion_com_nm[lag:] - anion_com_nm[:-lag]
        q_plus = z * np.sum(cat_delta, axis=1)
        q_minus = -z * np.sum(an_delta, axis=1)
        total = q_plus + q_minus
        total_msd.append(float(np.mean(np.sum(total * total, axis=1))))
        pp_msd.append(float(np.mean(np.sum(q_plus * q_plus, axis=1))))
        mm_msd.append(float(np.mean(np.sum(q_minus * q_minus, axis=1))))
        pm_cross.append(float(np.mean(np.sum(q_plus * q_minus, axis=1))))
    time_lag = lag_ps
    total_msd = np.array(total_msd)
    pp_msd = np.array(pp_msd)
    mm_msd = np.array(mm_msd)
    pm_cross = np.array(pm_cross)
    fit_mask = (time_lag >= 0.2 * time_lag[-1]) & (time_lag <= 0.8 * time_lag[-1])
    if fit_mask.sum() < 3:
        raise RuntimeError("Not enough collective frames for conductivity fit")

    def fit_series(y):
        slope, intercept = np.polyfit(time_lag[fit_mask], y[fit_mask], 1)
        return float(slope), float(intercept)

    total_slope, _ = fit_series(total_msd)
    pp_slope, _ = fit_series(pp_msd)
    mm_slope, _ = fit_series(mm_msd)
    pm_slope, _ = fit_series(pm_cross)
    volume_m3 = volume_nm3 * 1.0e-27
    factor = E_CHARGE**2 * 1.0e-6 / (6.0 * KB * temperature * volume_m3) / 100.0
    sigma_s_cm = factor * total_slope
    denom = pp_slope + mm_slope + 2.0 * pm_slope
    t_plus = (pp_slope + pm_slope) / denom if denom else float("nan")
    return {
        "conductivity_s_cm": float(sigma_s_cm),
        "t_plus": float(t_plus),
        "slope_total_z2_nm2_ps": float(total_slope),
        "slope_pp": float(pp_slope),
        "slope_mm": float(mm_slope),
        "slope_pm": float(pm_slope),
    }


def cne_lifetime_diagnostic(cache, topology: Topology, cat_com_nm, anion_com_nm, time_ps, volume_nm3, temperature, z, max_cluster, cutoff_nm, sample_stride):
    wrapped = cache["wrapped_nm"].astype(np.float64)
    boxes = cache["box_nm"].astype(np.float64)
    cat_mols = list(map(int, topology.cat_mols))
    anion_mols = list(map(int, topology.anion_mols))
    cat_li_sel = []
    for mol in cat_mols:
        idx = topology.mol_to_sel_indices[mol]
        if len(idx) != 1:
            raise RuntimeError(f"Cation mol {mol} has {len(idx)} Li atoms")
        cat_li_sel.append(int(idx[0]))
    an_trace = []
    for mol in anion_mols:
        idx = topology.anion_trace_sel_by_mol[mol]
        if len(idx) == 0:
            raise RuntimeError(f"Anion mol {mol} has no trace atoms")
        an_trace.append(idx)

    active: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, np.ndarray, tuple[int, int]]] = {}
    disps: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    pop_counts: Counter[tuple[int, int]] = Counter()
    frame_count = 0
    for frame_i in range(0, len(time_ps), sample_stride):
        clusters = molecular_clusters_from_precomputed(wrapped[frame_i], boxes[frame_i], cat_li_sel, an_trace, cutoff_nm)
        frame_count += 1
        current_keys = set()
        for cats, ans in clusters:
            key = (tuple(sorted(cats)), tuple(sorted(ans)))
            current_keys.add(key)
            comp = (len(cats), len(ans))
            pop_counts[comp] += 1
            coords = []
            weights = []
            for ci in cats:
                coords.append(cat_com_nm[frame_i, ci])
                weights.append(float(topology.mol_mass[int(topology.cat_mols[ci])]))
            for ai in ans:
                coords.append(anion_com_nm[frame_i, ai])
                weights.append(float(topology.mol_mass[int(topology.anion_mols[ai])]))
            coords_a = np.array(coords)
            weights_a = np.array(weights)
            com = np.average(coords_a, weights=weights_a, axis=0)
            if key not in active:
                active[key] = (frame_i, com, comp)
            else:
                start_i, start_com, start_comp = active[key]
                if start_comp != comp:
                    active[key] = (frame_i, com, comp)
                    continue
                dt_ps = float(time_ps[frame_i] - time_ps[start_i])
                if dt_ps > 0:
                    disp2 = float(np.sum((com - start_com) ** 2))
                    disps[comp].append((dt_ps, disp2))
        for key in list(active):
            if key not in current_keys:
                del active[key]

    rows = []
    volume_m3 = volume_nm3 * 1.0e-27
    pref = E_CHARGE**2 / (KB * temperature * volume_m3) / 100.0
    total_sigma = 0.0
    tn_num = 0.0
    tn_den = 0.0
    for comp, vals in sorted(disps.items()):
        if comp[0] >= max_cluster or comp[1] >= max_cluster:
            continue
        if len(vals) < 5:
            continue
        arr = np.array(vals, dtype=np.float64)
        # Endpoint-like segment diffusivity, weighted by segment duration.
        d_cm2_s = float(np.mean(arr[:, 1] / (6.0 * arr[:, 0]) * 1.0e-2))
        pop_avg = float(pop_counts[comp] / frame_count) if frame_count else 0.0
        charge = z * (comp[0] - comp[1])
        rows.append(
            {
                "ncat": comp[0],
                "nanion": comp[1],
                "population_avg": pop_avg,
                "samples": len(vals),
                "diffusion_cm2_s": d_cm2_s,
            }
        )
        sigma = pref * charge * charge * pop_avg * d_cm2_s * 1.0e-4
        total_sigma += sigma
        tn_num += comp[0] * z * charge * pop_avg * d_cm2_s
        tn_den += charge * charge * pop_avg * d_cm2_s
    return total_sigma, (tn_num / tn_den if tn_den else float("nan")), rows


def analyze_cache(cache_path: Path, topology: Topology, args, outdir: Path):
    args = copy.copy(args)
    args.max_cluster = population_matrix_size(topology, args.max_cluster)
    cache = np.load(cache_path)
    validate_cache_topology(cache, topology)
    lane = str(cache["lane"])
    time_ps = cache["time_ps"].astype(np.float64)
    box_nm = cache["box_nm"].astype(np.float64)
    volume_nm3 = float(np.mean(np.prod(box_nm, axis=1)))

    cat_com, anion_com = molecule_com_from_cache(cache, topology, drift_remove=True)
    cat_msd_t, cat_msd = msd_curve(cat_com, time_ps, args.msd_lags)
    an_msd_t, an_msd = msd_curve(anion_com, time_ps, args.msd_lags)
    cat_fit = fit_diffusion_from_msd(cat_msd_t, cat_msd)
    an_fit = fit_diffusion_from_msd(an_msd_t, an_msd)
    ne_sigma, ne_tplus = conductivity_ne(
        cat_fit["diffusion_cm2_s"],
        an_fit["diffusion_cm2_s"],
        len(topology.cat_mols),
        len(topology.anion_mols),
        volume_nm3,
        args.temperature,
        args.z,
    )

    li_raw = selected_atom_coords(cache, topology, topology.cat_atom_sel, drift_remove=False)
    n_raw = selected_atom_coords(cache, topology, topology.anion_ref_atom_sel, drift_remove=False)
    li_diff_htp, li_msd_endpoint = endpoint_diffusion_htp(li_raw, time_ps)
    an_diff_htp, an_msd_endpoint = endpoint_diffusion_htp(n_raw, time_ps)
    pop_mat, pop_frames = htp_atom_population_matrix(
        cache,
        topology,
        max_cluster=args.max_cluster,
        cutoff_nm=args.cluster_cutoff_angstrom / 10.0,
        sample_stride=args.cluster_sample_stride,
    )
    cne0_sigma, cne0_tplus = cne0_htp_conductivity(
        pop_mat,
        li_diff_htp,
        an_diff_htp,
        volume_nm3,
        args.temperature,
        args.max_cluster,
    )
    cne0_msd_sigma, cne0_msd_tplus = cne0_htp_conductivity(
        pop_mat,
        cat_fit["diffusion_cm2_s"],
        an_fit["diffusion_cm2_s"],
        volume_nm3,
        args.temperature,
        args.max_cluster,
    )

    collective = collective_conductivity(cat_com, anion_com, time_ps, volume_nm3, args.temperature, args.z)
    cne_sigma, cne_tplus, cne_rows = cne_lifetime_diagnostic(
        cache,
        topology,
        cat_com,
        anion_com,
        time_ps,
        volume_nm3,
        args.temperature,
        args.z,
        args.max_cluster,
        args.cluster_cutoff_angstrom / 10.0,
        args.cluster_sample_stride,
    )

    lane_dir = outdir / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_lag_ps": cat_msd_t, "msd_nm2": cat_msd}).to_csv(lane_dir / "msd_cation.csv", index=False)
    pd.DataFrame({"time_lag_ps": an_msd_t, "msd_nm2": an_msd}).to_csv(lane_dir / "msd_anion.csv", index=False)
    pd.DataFrame(pop_mat).to_csv(lane_dir / "population_matrix_htpmd.csv", index=False, header=False)
    pd.DataFrame(cne_rows).to_csv(lane_dir / "cne_lifetime_cluster_diffusivities.csv", index=False)

    summary = {
        "lane": lane,
        "frames": int(len(time_ps)),
        "time_start_ps": float(time_ps[0]),
        "time_end_ps": float(time_ps[-1]),
        "stride_ps": float(np.median(np.diff(time_ps))),
        "mean_volume_nm3": volume_nm3,
        "temperature_K": float(args.temperature),
        "formal_z": float(args.z),
        "n_cations": int(len(topology.cat_mols)),
        "n_anions": int(len(topology.anion_mols)),
        "cluster_cutoff_angstrom": float(args.cluster_cutoff_angstrom),
        "cluster_sample_stride_frames": int(args.cluster_sample_stride),
        "cluster_population_frames": int(pop_frames),
        "cluster_population_matrix_dimension": int(pop_mat.shape[0]),
        "cluster_population_retains_all_ions": True,
        "diffusion_msd_fit_drift_removed": {
            "cation": cat_fit,
            "anion": an_fit,
        },
        "diffusion_htp_endpoint_raw": {
            "cation_type90_cm2_s": float(li_diff_htp),
            "anion_type93_cm2_s": float(an_diff_htp),
            "cation_type90_endpoint_msd_nm2": float(li_msd_endpoint),
            "anion_type93_endpoint_msd_nm2": float(an_msd_endpoint),
        },
        "NE_msd_fit": {
            "conductivity_s_cm": float(ne_sigma),
            "t_plus": float(ne_tplus),
        },
        "cNE0_htpmd": {
            "conductivity_s_cm": float(cne0_sigma),
            "t_plus": float(cne0_tplus),
            "basis": "HTP-MD polymer.compute_conductivity style: atom-level 3.4 A population matrix, monatomic-cation/TFSI nitrogen-reference endpoint diffusivity; population retains all ions",
        },
        "cNE0_msd_fit": {
            "conductivity_s_cm": float(cne0_msd_sigma),
            "t_plus": float(cne0_msd_tplus),
            "basis": "Same HTP-MD atom-level population matrix, but cation/anion diffusivities come from drift-removed molecular COM MSD 20-80% lag-time fits.",
        },
        "cNE_lifetime_tracked": {
            "conductivity_s_cm": float(cne_sigma),
            "t_plus": float(cne_tplus),
            "basis": "diagnostic exact-membership molecular cluster lifetimes; noisy if cluster lifetimes are short",
        },
        "Einstein_collective": collective,
    }
    (lane_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def write_combined_report(summaries: list[dict], topology: Topology, outdir: Path, args):
    rows = []
    for s in summaries:
        rows.append(
            {
                "lane": s["lane"],
                "frames": s["frames"],
                "time_ns": s["time_end_ps"] / 1000.0,
                "volume_nm3": s["mean_volume_nm3"],
                "D_cation_fit_cm2_s": s["diffusion_msd_fit_drift_removed"]["cation"]["diffusion_cm2_s"],
                "D_anion_fit_cm2_s": s["diffusion_msd_fit_drift_removed"]["anion"]["diffusion_cm2_s"],
                "NE_sigma_S_cm": s["NE_msd_fit"]["conductivity_s_cm"],
                "NE_t_plus": s["NE_msd_fit"]["t_plus"],
                "D_cation_htp_cm2_s": s["diffusion_htp_endpoint_raw"]["cation_type90_cm2_s"],
                "D_anion_htp_cm2_s": s["diffusion_htp_endpoint_raw"]["anion_type93_cm2_s"],
                "cNE0_htp_sigma_S_cm": s["cNE0_htpmd"]["conductivity_s_cm"],
                "cNE0_htp_t_plus": s["cNE0_htpmd"]["t_plus"],
                "cNE0_msd_fit_sigma_S_cm": s["cNE0_msd_fit"]["conductivity_s_cm"],
                "cNE0_msd_fit_t_plus": s["cNE0_msd_fit"]["t_plus"],
                "cNE_lifetime_sigma_S_cm": s["cNE_lifetime_tracked"]["conductivity_s_cm"],
                "cNE_lifetime_t_plus": s["cNE_lifetime_tracked"]["t_plus"],
                "Einstein_sigma_S_cm": s["Einstein_collective"]["conductivity_s_cm"],
                "Einstein_t_plus": s["Einstein_collective"]["t_plus"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "transport_summary.csv", index=False)
    md = []
    md.append("# PolyGen transport analysis")
    md.append("")
    md.append(f"- Source root: `{args.root}`")
    md.append(f"- Formal charge magnitude z: `{args.z}`")
    md.append(f"- Temperature: `{args.temperature} K`")
    md.append(f"- cNE0 style: HTP-MD `polymer.compute_conductivity`, max_cluster `{args.max_cluster}`, cutoff `{args.cluster_cutoff_angstrom} A`")
    md.append(f"- Ion topology from first LAMMPS prod dump: atoms `{len(topology.atom_ids)}`, cations `{len(topology.cat_mols)}`, anions `{len(topology.anion_mols)}`")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(df.to_markdown(index=False, floatfmt=".6g"))
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- `NE_*` uses drift-removed cation/anion molecular COM MSD with a 20-80% lag-time linear fit.")
    md.append("- `cNE0_htp_*` uses raw type-90 Li and type-93 anion reference atom endpoint diffusivity, matching the local HTP-MD implementation style.")
    md.append("- `cNE0_msd_fit_*` keeps the HTP-MD atom-level population matrix but replaces endpoint diffusivity with the same drift-removed molecular COM MSD-fit diffusivities used by `NE_*`.")
    md.append("- `Einstein_*` is the collective charge-displacement conductivity. It is not the same estimator as cluster NE, but it is the direct correlation-inclusive conductivity estimator from the same trajectory.")
    md.append("- `cNE_lifetime_*` is a diagnostic exact-membership molecular cluster estimator. Treat it as unreliable if cluster lifetimes are short or sample counts are sparse; inspect each lane's `cne_lifetime_cluster_diffusivities.csv`.")
    md.append("")
    md.append("## Reliability Flags")
    md.append("")
    if float(df["time_ns"].min()) < 20.0:
        md.append("- Production length is below 20 ns. Local protocol `docs/transport_protocol_freeze.md` marks charged transport production as >=20 ns, so the 10 ns values are screening-level.")
    for _, row in df.iterrows():
        lane = row["lane"]
        if row["Einstein_sigma_S_cm"] <= 0:
            md.append(f"- `{lane}` has non-positive collective Einstein conductivity slope; do not use its `Einstein_*` value as converged transport.")
        if row["cNE_lifetime_sigma_S_cm"] > 2.0 * row["NE_sigma_S_cm"]:
            md.append(f"- `{lane}` cNE lifetime diagnostic is >2x NE. That indicates short-lifetime cluster displacement bias; do not treat `cNE_lifetime_*` as production cNE without a stricter cluster lifetime estimator.")
        if row["D_cation_htp_cm2_s"] > 2.0 * row["D_cation_fit_cm2_s"] or row["D_anion_htp_cm2_s"] > 2.0 * row["D_anion_fit_cm2_s"]:
            md.append(f"- `{lane}` HTP endpoint diffusivity differs by >2x from MSD-fit diffusivity. This is expected sensitivity of the HTP-MD endpoint style, but it makes `cNE0_htp_*` noisy for this trajectory.")
    (outdir / "transport_report.md").write_text("\n".join(md) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--outdir", type=Path, default=Path("output/polygen_transport_analysis/current"))
    p.add_argument("--lanes", default="lammps,cpu,gpu", help="Comma-separated: lammps,cpu,gpu")
    p.add_argument("--stride-ps", type=float, default=2.0)
    p.add_argument("--temperature", type=float, default=353.0)
    p.add_argument("--z", type=float, default=1.0)
    p.add_argument("--max-cluster", type=int, default=None, help="Population matrix dimension (exclusive); default includes all ions. Too-small values fail if a cluster would be dropped.")
    p.add_argument("--cluster-cutoff-angstrom", type=float, default=3.4)
    p.add_argument("--cluster-sample-stride", type=int, default=1, help="Use every Nth frame for cluster population/lifetime work")
    p.add_argument("--msd-lags", type=int, default=220)
    p.add_argument("--max-chunks", type=int, default=None)
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--gmx-cpu", type=Path, default=Path("build_gateb_double_cpu/bin/gmx_d"))
    p.add_argument("--gmx-gpu", type=Path, default=Path("build_gateb_cuda/bin/gmx"))
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root
    lammps_dir = root / "lammps_openmp"
    first_dump = lammps_dir / "prod_traj" / "prod_chunk0001.lammpstrj"
    if not first_dump.exists():
        raise RuntimeError(f"Missing first LAMMPS prod dump: {first_dump}")
    topology = parse_first_lammps_frame(first_dump)
    args.max_cluster = population_matrix_size(topology, args.max_cluster)
    if len(topology.atom_ids) != 7075:
        raise RuntimeError(f"Unexpected atom count from production dump: {len(topology.atom_ids)}")
    if len(topology.cat_mols) != len(topology.anion_mols):
        raise RuntimeError(f"Non-neutral ion molecule counts: {len(topology.cat_mols)} vs {len(topology.anion_mols)}")

    lane_map = {
        "lammps": LaneSpec("lammps_openmp", "lammps", lammps_dir),
        "cpu": LaneSpec("gromacs_cpu_openmp", "gromacs", root / "gromacs_cpu_openmp", args.gmx_cpu),
        "gpu": LaneSpec("gromacs_gpu_hybrid_strict_pme5", "gromacs", root / "gromacs_gpu_hybrid_strict_pme5", args.gmx_gpu),
    }
    requested = [x.strip() for x in args.lanes.split(",") if x.strip()]
    lanes = [lane_map[x] for x in requested]
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "analysis_config.json").write_text(json.dumps(vars(args), default=str, indent=2, sort_keys=True))

    summaries = []
    for lane in lanes:
        cache_path = load_or_build_cache(lane, topology, args.outdir, args.stride_ps, args.max_chunks, args.force_cache)
        summaries.append(analyze_cache(cache_path, topology, args, args.outdir))
    write_combined_report(summaries, topology, args.outdir, args)
    print(args.outdir / "transport_report.md")


if __name__ == "__main__":
    main()
