# R-RESPA M2b Validation Report

## Scope

M2b does not reopen M2 exact-path reconnection. It keeps the same validated microfixture harness and does only two things:

- correct the overstated M1 continuity wording
- isolate the `dense_oligomer` exact 3-level bookkeeping defect, starting from the `Coulomb-(SR)` mismatch

Out of scope:

- full-system TRL-5
- production readiness
- TP1.xx blocker work
- performance or transport claims

## Starting Point

- worktree: `..`
- branch: `respa-m2-exact-three-level`
- build: `../ab_builds/respa_m2_exact_three_level/bin/gmx`
- M2 boundary carried forward:
  - exact 3-level runtime activation is real
  - `dense_oligomer` bookkeeping fails badly
  - the old "legacy comparator" wording overstated continuity to archived M1

## Files Changed

- updated `../tools/run_respa_m2_microfixtures/run_respa_m2.py`
- added `./validation_report_respa_m2b.md`

No engine C++ behavior was changed for M2b. The only new runtime evidence comes from the same harness plus a narrow debug rerun of the dense exact path.

## M1 Continuity Correction

Direct archived-M1 continuity was not claimed for M2b because it is not exact.

Archived M1 used:

- `integrator = md`
- `coulombtype = Cut-off`
- `mts-mode = legacy`
- `mts-level2-forces = nonbonded`

Files:

- `../tools/run_respa_m1_microfixtures/run_respa_m1.py`
- `../tests/reference_results/r_respa_m1_microfixtures/summary.json`

The simpler split in M2b is therefore renamed and bounded as:

- `pme_legacy_side_reference`

That label means:

- same deterministic harness family
- same PME-side fixture settings as the exact 3-level run
- not direct archived-M1 continuity

## Commands Run

Main M2b harness:

```bash
python3 ../tools/run_respa_m2_microfixtures/run_respa_m2.py \
  --gmx-bin ../ab_builds/respa_m2_exact_three_level/bin/gmx \
  --fixture dense_oligomer \
  --dense-bookkeeping-isolation \
  --milestone-name 'R-RESPA M2b' \
  --out ../tests/reference_results/r_respa_m2b_dense_bookkeeping_isolation
```

Exact per-case commands, including the dense term extraction, are stored in:

- `../tests/reference_results/r_respa_m2b_dense_bookkeeping_isolation/raw_commands.txt`

## Strongest Confirmed Finding

The dense exact-path step-0 bookkeeping defect narrows to the excluded-pair Coulomb correction contribution inside `Coulomb-(SR)` accounting.

Direct evidence:

- plain step-0 `Coulomb-(SR) = -7160.835449`
- exact step-0 `Coulomb-(SR) = 1422.623291`
- exact minus plain = `8583.458740`
- exact debug `excludedPairs coul = 8583.447727`

Files:

- `../tests/reference_results/r_respa_m2b_dense_bookkeeping_isolation/dense_oligomer/fixture_summary.json`
- `../tests/reference_results/r_respa_m2b_dense_bookkeeping_isolation/dense_oligomer/dt_0p0005/exact_three_level/exact_mdrun.stderr.txt`

This is strong evidence for a mis-owned excluded-pair Coulomb correction contribution in the exact step-0 bookkeeping path. It is not evidence for generic physics failure.

## Strongest Unresolved Uncertainty

The force-side defect is still not cleanly closed.

The dense exact run still shows:

- step-0 force L2 difference vs plain Verlet = `742.348184`
- step-0 force max-abs difference vs plain Verlet = `35.9212`

M2b narrows the energy bookkeeping defect, but it does not yet prove whether that same excluded-pair ownership bug fully explains the force mismatch or whether a separate force-buffer ownership issue remains.

## Verdict

`DENSE BOOKKEEPING STILL PARTIAL BUT NARROWED`

## Next Step

Keep the same dense microfixture and inspect only the exact 3-level force-buffer ownership for the excluded-pair Coulomb correction path.
