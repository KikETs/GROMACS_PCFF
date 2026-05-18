# PCFF M10 Method Validation

This toolchain freezes the current M10 evidence into machine-readable summaries.

It does not rerun heavy production MD. It ingests already generated analysis artifacts from:

- `../DL/gromacs/eval_top10_bottom10_stratified100/results`
- `../DL/gromacs`
- `../DL/LAMMPS_NEW`

The workflow separates:

- `strict parity`
  - direct GROMACS-vs-LAMMPS comparisons on representative paired systems, but only after the paired GROMACS systems pass the PCFF provenance gate
- `screening usefulness`
  - cohort-level ranking/error summaries on completed GROMACS screening runs

## Files

- `generate.py`
  - builds `tests/reference_results/m10/*.json` and `*.csv`

## Usage

From the repository root:

```bash
python3 tools/pcff_method_validation_m10/generate.py
```

The default output directory is:

```text
tests/reference_results/m10
```

## Output

- `comparison_summary.json`
  - top-level M10 summary
- `strict_parity_summary.json`
  - representative-system direct comparison summary after applying the PCFF provenance gate
- `screening_usefulness_summary.json`
  - cohort-level ranking/error summary
- `method_readiness_summary.json`
  - readiness judgment and blocking gaps
- `pcff_paired_provenance_gate.csv`
  - paired-system provenance classification that decides whether a candidate system is allowed into strict PCFF parity
- `strict_parity_metrics.csv`
  - per-system strict metrics; this may be empty when no paired system passes the provenance gate
- `screening_metric_rows.csv`
  - per-trajectory cohort comparison rows
- `density_local_subset.csv`
  - local density comparison subset
- `paired_density_provenance.csv`
  - why strict paired density is unavailable or inconsistent
- `paired_topology_recovery.csv`
  - donor-topology candidates for missing paired GROMACS topologies and dry-run `grompp` recovery status
- `paired_artifact_registry_audit.csv`
  - whether `run_results.csv` claims completed analysis even though the referenced analysis CSV and raw production artifacts are missing
- `chain_size_artifact_status.csv`
  - paired LAMMPS `Rg` availability, GROMACS production prerequisite status, and whether `Rg` generation is even possible
- `transport_decomposition.csv`
  - paired conductivity mismatch decomposition into cNE/NE, population, charge, pair-parameter, and ion-LJ terms

## Important limitations

- the current paired candidates are not automatically valid PCFF evidence; the workflow now rejects paired systems that are ACPYPE/GAFF2-prepared or otherwise not PCFF-qualified
- strict parity can therefore be empty even when paired IDs exist in the sampled results
- paired density is marked unavailable because the sampled structural CSV counts do not match full `topol.top` molecule counts
- `27670` topology can be partially recovered in principle because `14768` has the same `SMILES`, degree of polymerization, density, and molality, and donor-topology `grompp` dry-run succeeds
- paired chain-size parity is marked unavailable because matched GROMACS production `Rg` artifacts were not found
- paired systems currently retain derived sampled metrics in `results/*.csv`, but the raw GROMACS production artifacts and referenced per-run analysis CSVs are missing; the registry audit freezes that mismatch explicitly
- for `27670`, the intended molecule counts can still be recovered from `packmol.inp`, but the final paired GROMACS topology is missing
- the local GROMACS screening pipeline is ACPYPE/GAFF2-based, so screening outputs are not PCFF-qualified for the current M10 claim
- conductivity usefulness is limited by the availability of GROMACS conductivity outputs in the existing cohort
- transport mismatch diagnostics are explanatory, not causal proof; they narrow the likely sources of disagreement but do not by themselves prove the dominant mechanism
