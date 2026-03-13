# `pcff_respa_parity`

This toolchain measures the current GROMACS exact `mts-mode = lammps-respa` path against a frozen LAMMPS `run_style respa` reference.

It is intentionally narrower than the M5 short-MD parity workflow:

- scope is `run_style respa`
- ensemble is `NVE` only
- supported schedule is the frozen `3-level` `run_style respa 3 2 2` contract only

## Basis

The schedule is frozen from local LAMMPS primary sources:

- `/home/user/lammps/doc/src/run_style.rst`
- `/home/user/lammps/doc/src/pair_class2.rst`

The current harness uses:

- `run_style respa 3 2 2`
- `bond 1 angle 1 dihedral 1 improper 1`
- `inner 1 3.0 4.5 middle 2 6.0 8.0 outer 3`
- `kspace 3`
- outer timestep `2.0 fs`
- `5` outer steps, matching `20` inner GROMACS base steps at `0.5 fs`

## Files

- `prepare_reference.py`
  - runs LAMMPS and freezes `nve_respa.json` and `reference_summary.json`
- `compare.py`
  - compares actual GROMACS summaries against the frozen reference
- `run.py`
  - optional reference regeneration, GROMACS build/run, and comparison orchestration
- `drift_scan.py`
  - sweeps outer-step counts and GROMACS `mts-respa-pair14-level` settings to localize short-NVE drift
  - can optionally enable the nested CPU prototype with `--nested-prototype`
- `force_compare.py`
  - runs exact `r-RESPA` on CPU, extracts outer-step coordinates at high precision, and compares the exact total force against an unsplit legacy single-point force on the same coordinates

## Output layout

Frozen references live under:

- `tests/reference_results/m6_respa/<system>/`

Actual GROMACS summaries default to:

- `tests/reference_results/m6_respa/last_run_actual/`

Comparison summaries default to:

- `tests/reference_results/m6_respa/last_run_compare/`

Same-coordinate force summaries can be frozen separately, for example:

- `tests/reference_results/m6_respa/force_compare_summary.json`

## Usage

Regenerate the LAMMPS reference:

```bash
python3 tools/pcff_respa_parity/prepare_reference.py --out tests/reference_results/m6_respa
```

Run the full harness:

```bash
python3 tools/pcff_respa_parity/run.py --prepare-reference
```

Run the same harness with the nested CPU prototype enabled:

```bash
python3 tools/pcff_respa_parity/run.py --prepare-reference --nested-prototype
```

This prototype path is not part of the supported M6 gate. It exists only for debugging alternative CPU propagation semantics.

Run the outer-step / pair14 drift scan:

```bash
python3 tools/pcff_respa_parity/drift_scan.py
```

Run the exact-vs-unsplit same-coordinate force comparison:

```bash
python3 tools/pcff_respa_parity/force_compare.py
```

## Important limitations

The frozen `3-level` CPU path now has explicit pass/fail tolerances:

- the in-tree regression in [pcff_short_md.cpp](/home/user/바탕화면/gromacs/src/programs/mdrun/tests/pcff_short_md.cpp) enforces them directly
- `compare.py` also loads the same tolerances from `reference_summary.tsv` and marks per-system `pass` / `measured` / `incomplete`

What is still not covered:

- exact `2-level` mode is not part of the supported contract and should be rejected by validation
- virial parity is frozen only for the step-0 tensor in the two M6 fixtures
- restart/checkpoint parity is covered only as an outer-boundary smoke test in [pcff_short_md.cpp](/home/user/바탕화면/gromacs/src/programs/mdrun/tests/pcff_short_md.cpp)
- arbitrary mid-period exact-mode termination is not a supported restart contract

The force-compare helper is narrower:

- it forces legacy simulator execution for the unsplit probe with `GMX_DISABLE_MODULAR_SIMULATOR=ON`
- it extracts exact frames as `.g96` to avoid `.gro` precision loss
- it is intended to answer one question only: whether the exact `r-RESPA` force calculation matches the unsplit Hamiltonian on identical coordinates
