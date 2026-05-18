# `generate_lammps_golden`

M1 toolchain for staging and generating the LAMMPS-side golden corpus for the PCFF/Class2 + `run_style respa` project.

## Goals

- Keep M1 reproducible and machine-readable.
- Freeze the observable formats before any GROMACS runtime implementation.
- Avoid any dependency on existing GROMACS runtime behavior.

## Layout

- [generate.py](./generate.py)
  - stages deterministic LAMMPS input bundles
  - optionally runs LAMMPS and normalizes raw outputs to JSON
- [compare.py](./compare.py)
  - compares normalized candidate outputs against normalized golden outputs
- [common.py](./common.py)
  - shared manifest loading, parsing, and JSON helpers

## Supported M1 observables

- `single_point`
- `forces`
- `finite_difference`
- `nve_drift`
- `nvt_snapshot`

## Recommended usage

Stage deterministic bundles:

```bash
python3 tools/generate_lammps_golden/generate.py stage --out output/tmp/lammps_golden_stage
```

Run LAMMPS and normalize outputs:

```bash
python3 tools/generate_lammps_golden/generate.py run --out output/tmp/lammps_golden_run --lammps-cmd lmp
```

Compare candidate normalized outputs against a normalized golden directory:

```bash
python3 tools/generate_lammps_golden/compare.py \
  --golden output/tmp/lammps_golden_run \
  --candidate output/tmp/gromacs_candidate \
  --energy-abs-tol 1e-8 \
  --force-abs-tol 1e-8 \
  --trace-abs-tol 1e-8
```

## M1 limitations

- This tool does not assume LAMMPS is installed. `stage` works without it; `run` requires it.
- No golden physics outputs are committed in M1.
- Tolerance policy is intentionally not frozen in M1. The comparison tool defaults to exact equality unless tolerances are provided explicitly.
- `run_style respa` is part of the frozen reference scope, but M1 generator inputs currently focus on the observable contract and static/dynamic corpus scaffolding. The later M2/M3 work must decide the exact `respa` fixture variants to run.
