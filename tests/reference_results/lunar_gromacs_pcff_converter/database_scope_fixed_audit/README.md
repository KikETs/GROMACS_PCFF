# Aligned Database Scope Audit

This directory records whether the provided aligned polymer database can support a database-wide converter claim.

Snapshot:

- path: `MY_PAPER_RELATED/MODELS/data/simulation-trajectory-aggregate_aligned.csv`
- sha256: `d4a804b3322463a2edf45be6949f3e85478668dc8bce631023f94c2673d71223`
- rows: `6270`
- unique SMILES: `6042`

Verdict:

- database-wide converter success: `claimable`
- LAMMPS batch trajectory directories found: `716`
- GROMACS batch trajectory directories found: `5`
- grompp smoke reports found: `0`
- repository database_lunar_smoke pass count: `6270`
- repository database_lunar_smoke missing LUNAR PCFF data count: `0`

This audit does not run conversion. It prevents overclaiming by checking whether the public artifact chain exists.
