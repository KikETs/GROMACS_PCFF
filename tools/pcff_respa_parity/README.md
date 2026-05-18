# `pcff_respa_parity`

This toolchain measures the standalone GROMACS exact `r-RESPA` path against a
frozen LAMMPS `run_style respa` reference.

It is intentionally narrower than the short-MD parity workflows:

- scope is `run_style respa`
- ensemble is `NVE` only
- supported schedule is the frozen `3-level` `run_style respa 3 2 2` contract only

## Basis

The schedule is frozen from LAMMPS primary documentation:

- `doc/src/run_style.rst`
- `doc/src/pair_class2.rst`

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
  - current default workflow entrypoint for the checked-in M6 exact `r-RESPA` comparison path
- `drift_scan.py`
  - sweeps outer-step counts and GROMACS `exact-respa-pair14-level` settings to localize short-NVE drift
  - can optionally enable the nested CPU prototype with `--nested-prototype`
- `force_compare.py`
  - runs exact `r-RESPA` on CPU, extracts outer-step coordinates at high precision, and compares the exact total force against an unsplit legacy single-point force on the same coordinates
- `lammps_force_trace_compare.py`
  - runs standalone exact `r-RESPA`, records exact outer-step force frames in the exact `.trr`, extracts matching outer-step coordinates as `.g96`, and compares that exact force trace directly against LAMMPS `run 0`
  - current fallback acceptance path for M9 when frozen thermo parity remains narrower than direct force-trace parity

No separate offline oracle comparator is currently checked into this tree. Do
not cite an offline/plain-facing comparator as an available authoritative
workflow unless the script and its output artifacts are restored together.

## Output Layout

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

The nested prototype path is not part of the supported M6 gate. It exists only
for debugging alternative CPU propagation semantics.

Run the outer-step / pair14 drift scan:

```bash
python3 tools/pcff_respa_parity/drift_scan.py
```

Run the exact-vs-unsplit same-coordinate force comparison:

```bash
python3 tools/pcff_respa_parity/force_compare.py
```

Run the direct exact-frame-vs-LAMMPS force-trace comparison:

```bash
python3 tools/pcff_respa_parity/lammps_force_trace_compare.py
```

Run the M6 parity compare helper directly:

```bash
python3 tools/pcff_respa_parity/compare.py
```

## Important Limitations

The frozen `3-level` CPU path has explicit pass/fail tolerances:

- the in-tree regression in [pcff_short_md.cpp](../../src/programs/mdrun/tests/pcff_short_md.cpp) enforces them directly
- `compare.py` also loads the same tolerances from `reference_summary.tsv` and marks per-system `pass` / `measured` / `incomplete`

What is still not covered:

- exact `2-level` mode is not part of the supported contract and should be rejected by validation
- virial parity is frozen only for the step-0 tensor in the two M6 fixtures
- restart/checkpoint parity is covered only as an outer-boundary smoke test in [pcff_short_md.cpp](../../src/programs/mdrun/tests/pcff_short_md.cpp)
- arbitrary mid-period exact-mode termination is not a supported restart contract

The force-compare helper is narrower:

- it forces legacy simulator execution for the unsplit probe with `GMX_DISABLE_MODULAR_SIMULATOR=ON`
- it extracts exact frames as `.g96` to avoid `.gro` precision loss
- it is intended to answer one question only: whether the exact `r-RESPA` force calculation matches the unsplit Hamiltonian on identical coordinates

The direct LAMMPS force-trace helper is also narrow:

- it uses exact `.trr` force frames from the standalone exact run as the GROMACS force source
- it extracts later outer-step coordinates as `.g96` rather than `.gro` to avoid coordinate truncation before the LAMMPS `run 0` probe
- it compares those exact outer-step total forces against LAMMPS `run 0` at the same extracted outer-step coordinates
- it is intended to answer one question only: whether the standalone exact path itself produces total forces that directly match LAMMPS
