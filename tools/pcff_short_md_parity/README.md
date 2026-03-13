# PCFF Short-MD Parity Tools

`prepare_reference.py` regenerates the frozen M5 reference inputs and summaries:

- `tests/reference_results/m5/<system>/topol.top`
- `tests/reference_results/m5/<system>/initial_nve.gro`
- `tests/reference_results/m5/<system>/initial_nvt.gro`
- `tests/reference_results/m5/<system>/reference_summary.json`
- `tests/reference_results/m5/<system>/reference_summary.tsv`

The script reuses the M1/M4 frozen LAMMPS systems and performs one additional deterministic
LAMMPS setup run to materialize the exact `velocity create` state needed for NVE parity.

Example:

```bash
python3 tools/pcff_short_md_parity/prepare_reference.py
```

`run.py` builds and executes the M5 parity test binary, collects per-case summaries emitted by the
tests, and writes an aggregate JSON report:

```bash
python3 tools/pcff_short_md_parity/run.py
```

The aggregate report is written to `tests/reference_results/m5/last_run/comparison_summary.json`
unless `--summary-dir` is specified.

Notes:

- The default GTest filter is the working `PcffShortMdParity*` pattern.
- The script deletes stale `*.json` files in the summary directory before each run.
- If no per-case summaries are produced, `run.py` exits with an error instead of writing a false-green empty aggregate.
- Per-case JSON payloads now expose:
  - `supported_failure_categories`
  - `observed_failure_categories`
  - `harness_notes`
