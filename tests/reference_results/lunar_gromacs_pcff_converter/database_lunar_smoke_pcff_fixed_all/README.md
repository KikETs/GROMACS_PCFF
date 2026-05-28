# Database LUNAR PCFF Smoke Evidence

This directory contains parser -> mapping -> emission -> grompp smoke artifacts for LUNAR PCFF data files.

Boundary:

- Input artifacts are existing LUNAR `all2lmp` PCFF single-chain `.data` files.
- Each passing case includes the inspected input copy as `source_lunar_pcff.data`.
- This is polymer-only single-chain topology smoke evidence.
- This is not charged Li/TFSI or dense polymer-electrolyte conversion support.
- LUNAR generation warnings, when present in `lunar_generation_status.jsonl`, are not closed by `grompp` smoke.

Summary:

- selected scope: `csv`
- selected rows: `6270`
- pass: `6270`
- failure: `0`
- missing LUNAR PCFF data: `0`
- database-wide claim: `claimable`

Claim boundary:

- Strongest surviving claim: all `6270 / 6270` rows in the frozen CSV snapshot have parser -> mapping -> emission -> zero-warning `grompp` PASS evidence with the custom GROMACS_PCFF build.
- This is still a polymer-only single-chain preprocessing smoke claim.
- It is not charged Li/TFSI, dense electrolyte, production `mdrun`, physical validation, or transport readiness evidence.
- Stock upstream GROMACS is not supported for this PCFF topology path.

Important caveats:

- `870` passing rows used explicit zero-parameter fallback records for source LUNAR labels `c4o`, `s_m`, or `S-type-yourself`.
- `S-type-yourself` is a LUNAR atom-typing failure fallback and must not be described as physical parameter completion.
- Excluded-distance failures were unblocked by using a smoke-validation cutoff and expanded smoke `.gro` box derived from topological exclusion distances and coordinate extents.
- The smoke `.gro` box is not a production density or equilibration box.

Primary artifacts:

- `conversion_contract.json`
- `support_matrix.json`
- `database_lunar_smoke_summary.json`
- `parameter_fallback_summary.json`
- `build/pcff_gromacs_build_report.json`
- `cases/Traj_14764/grompp/grompp_smoke_report.json`

Representative full cases:

- `Traj_14764`: no-fallback PASS case.
- `Traj_14261`: `c4o` fallback case.
- `Traj_13651`: `s_m` fallback case.
- `Traj_14746`: `S-type-yourself` fallback case.
- `Traj_13262`: previous excluded-distance error now PASS.
- `Traj_27612`: previous excluded-distance warning now PASS.
- `Traj_26960`: previous excluded-distance warning now PASS.
