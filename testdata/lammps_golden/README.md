# LAMMPS Golden Corpus

This directory contains the M1 reference corpus for the PCFF/Class2 + LAMMPS-style `respa` project.

## Structure

- `corpus_manifest.json`: top-level machine-readable index.
- `systems/<id>/system.json`: per-system metadata.
- `systems/<id>/lammps/system.data`: topology and coordinates.
- `systems/<id>/lammps/system.in`: LAMMPS settings and coefficients.

Golden outputs are not checked in here yet. They are generated into a staging or run directory by:

- [tools/generate_lammps_golden/generate.py](../../tools/generate_lammps_golden/generate.py)

## M1 principles

- Small systems first.
- Deterministic input files.
- Explicit metadata.
- No hidden defaults for exclusions, mixing, or KSpace.
- LAMMPS output normalized to JSON for later GROMACS-vs-LAMMPS comparison.
