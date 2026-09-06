# LUNAR GROMACS PCFF Converter Claim Boundary

This note freezes the current public boundary for the LUNAR/all2lmp PCFF `.data` to GROMACS PCFF topology converter.

It is intentionally narrower than charged polymer-electrolyte support and narrower than any production MD or transport claim.

## Source revision note (2026-09-06)

The 6270-row reports below are historical evidence imported from commit `1e0e40be45a80fc15ef83e9797f5a5f9e678473c`. The consolidated `main` subsequently corrects 1–4 pair generation to use shortest bond paths and verifies that correction with a focused ring regression test. The full database smoke campaign has not been rerun on this corrected converter. Do not attribute the old 6270-row PASS count to a fresh run of current `main`. See [the consolidation record](pcff_main_consolidation_20260906.md).

## Narrow Claim

Use this sentence for the current state:

> Current public evidence supports polymer-only LUNAR PCFF single-chain topology preprocessing for the frozen CSV snapshot `MY_PAPER_RELATED/MODELS/data/simulation-trajectory-aggregate_aligned.csv` with SHA256 `d4a804b3322463a2edf45be6949f3e85478668dc8bce631023f94c2673d71223`: `6270 / 6270` rows have parser -> mapping -> emission -> zero-warning `grompp` PASS evidence using the custom GROMACS_PCFF build. This is a preprocessing smoke claim only. Charged Li/TFSI, dense electrolyte, production `mdrun`, physical parameter validation, and transport-production support remain unsupported.

## What Is Closed

- The custom GROMACS_PCFF CPU build exists on `lab@100.121.61.51` and was used for the database smoke run.
- A full public parser -> mapping -> emission -> `grompp` artifact chain is present for representative PASS case `Traj_14764`.
- Lightweight per-row parser, mapping, emission, and `grompp` reports are present for all `6270` selected rows.
- The frozen CSV smoke summary reports `6270` PASS, `0` FAIL, and `0` missing LUNAR PCFF data artifacts.
- Previous zero-mass and excluded-distance failures are closed for the smoke-preprocessing gate by explicit fallback records and smoke geometry rules.

Primary machine-readable artifacts:

- [Conversion contract](../tests/reference_results/lunar_gromacs_pcff_converter/database_lunar_smoke_pcff_fixed_all/conversion_contract.json)
- [Support matrix](../tests/reference_results/lunar_gromacs_pcff_converter/database_lunar_smoke_pcff_fixed_all/support_matrix.json)
- [Smoke summary](../tests/reference_results/lunar_gromacs_pcff_converter/database_lunar_smoke_pcff_fixed_all/database_lunar_smoke_summary.json)
- [Parameter fallback summary](../tests/reference_results/lunar_gromacs_pcff_converter/database_lunar_smoke_pcff_fixed_all/parameter_fallback_summary.json)
- [Database scope audit](../tests/reference_results/lunar_gromacs_pcff_converter/database_scope_fixed_audit/database_claim_audit.json)
- [GROMACS_PCFF build report](../tests/reference_results/lunar_gromacs_pcff_converter/database_lunar_smoke_pcff_fixed_all/build/pcff_gromacs_build_report.json)
- [Representative PASS grompp report](../tests/reference_results/lunar_gromacs_pcff_converter/database_lunar_smoke_pcff_fixed_all/cases/Traj_14764/grompp/grompp_smoke_report.json)

## What Is Not Closed

- Do not claim charged Li/TFSI, separate ion, or dense electrolyte conversion support.
- Do not count emitted text files alone as success when the row lacks a clean `grompp` PASS.
- Do not describe the zero-parameter fallback rows as physical parameter-completion evidence.
- Do not treat `S-type-yourself` as a supported PCFF type; it is a smoke-only LUNAR atom-typing failure fallback.
- Do not treat the smoke-expanded `.gro` box as a production density or equilibration box.
- Do not claim physical validation, `mdrun` stability, ensemble validity, conductivity, or transport readiness from this smoke run.
- Do not claim stock upstream GROMACS support for these PCFF topologies; the smoke run requires the custom GROMACS_PCFF build.

## Caveat Boundary

The previous `929` failures are closed only inside the smoke-preprocessing boundary:

- `870` passing rows use explicit fallback records for source LUNAR atom labels `c4o`, `s_m`, or `S-type-yourself`.
- `c4o` and `s_m` fallbacks use `pcff_interface_v1_6mBN.frc` mass/nonbond values.
- `S-type-yourself` is mapped to a smoke-only sulfur fallback and remains unsupported for physical parameter-completion wording.
- Excluded-distance errors/warnings are closed by deriving the smoke cutoff and smoke `.gro` box from topological exclusion distances and coordinate extents.

Representative caveat artifacts:

- [c4o fallback example](../tests/reference_results/lunar_gromacs_pcff_converter/database_lunar_smoke_pcff_fixed_all/cases/Traj_14261/grompp/grompp_smoke_report.json)
- [s_m fallback example](../tests/reference_results/lunar_gromacs_pcff_converter/database_lunar_smoke_pcff_fixed_all/cases/Traj_13651/grompp/grompp_smoke_report.json)
- [S-type-yourself fallback example](../tests/reference_results/lunar_gromacs_pcff_converter/database_lunar_smoke_pcff_fixed_all/cases/Traj_14746/grompp/grompp_smoke_report.json)
- [previous excluded-distance error now PASS](../tests/reference_results/lunar_gromacs_pcff_converter/database_lunar_smoke_pcff_fixed_all/cases/Traj_13262/grompp/grompp_smoke_report.json)
- [previous excluded-distance warning now PASS](../tests/reference_results/lunar_gromacs_pcff_converter/database_lunar_smoke_pcff_fixed_all/cases/Traj_27612/grompp/grompp_smoke_report.json)

## Charged Extension Gate

Charged support is closed to claim until separate public artifacts exist for:

- charged parser summaries
- charged mapping summaries with ion/salt molecules and net-charge accounting
- emitted charged topology and coordinate artifacts
- custom GROMACS_PCFF zero-warning `grompp` PASS for at least one charged case
- unsupported charged-term list

Passing polymer-only rows do not weaken this gate.
