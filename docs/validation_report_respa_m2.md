# R-RESPA M2 Validation Report

## Scope

M2 does not reopen the M1 two-level bring-up and does not claim full-system readiness.

It reconnects the frozen exact 3-level `mts-mode = lammps-respa` path onto the same deterministic microfixture harness style used for M1, then checks only:

- scheduler execution
- bookkeeping coherence
- tight-limit behavior against plain Verlet
- relationship to the already-working legacy two-level split on the same harness

Out of scope:

- TP1.xx blocker work
- `vcoul` reopening
- full-system TRL-5
- transport or performance claims

## Starting Point

- worktree: `/home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2`
- branch: `respa-m2-exact-three-level`
- build: `/home/kiket/바탕화면/test/ab_builds/respa_m2_exact_three_level/bin/gmx`
- exact 3-level design basis:
  - `src/gromacs/mdtypes/multipletimestepping.cpp`
  - `src/gromacs/gmxpreprocess/readir.cpp`
  - `src/gromacs/mdlib/sim_util.cpp`
  - `src/programs/mdrun/tests/pcff_short_md.cpp`
  - `docs/respa_design_m6.md`

M1 had already shown that the harness style works for the current legacy two-level split on validated microfixtures. M2 therefore only reconnects the frozen exact 3-level path instead of reopening the 2-level design.

## Files Changed

- added `/home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tools/run_respa_m2_microfixtures/run_respa_m2.py`
- added `/home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/docs/validation_report_respa_m2.md`

No engine C++ files were changed for M2. The exact 3-level runtime path was already present; the reconnection work was in the microfixture harness and validation logic.

## Exact 3-Level Schedule Reconnected

Active exact schedule used by the harness:

- mode: `lammps-respa`
- integrator: `md-vv`
- levels: `3`
- level factors:
  - level 2 factor = `2`
  - level 3 factor = `4`
- ownership:
  - inner: `bond`, `angle`, `dihedral`, `improper`, `pair14`, `nonbonded_inner`
  - middle: `nonbonded_middle`
  - outer: `pair`, `nonbonded_outer`, `kspace`
- switching:
  - `inner-off = 0.30 nm`
  - `inner-on = 0.45 nm`
  - `outer-on = 0.60 nm`
  - `outer-off = 0.80 nm`

The exact path stayed exact 3-level. It was not silently downgraded back to 2 levels.

## Fixtures Used

- `coulomb_toy`
  - smallest validated charged Cut-off microfixture
  - useful for confirming exact 3-level scheduler activation and tight-limit behavior without listed-term complexity
- `dense_oligomer`
  - same richer validated microfixture family used in M1
  - useful for checking whether the reconnected exact path remains coherent once listed terms are present

## Commands Run

Main M2 harness:

```bash
python3 /home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tools/run_respa_m2_microfixtures/run_respa_m2.py \
  --gmx-bin /home/kiket/바탕화면/test/ab_builds/respa_m2_exact_three_level/bin/gmx \
  --out /home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tests/reference_results/r_respa_m2_microfixtures
```

The exact per-case `grompp`, `mdrun`, `dump`, and `energy` invocations are stored in:

- `/home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tests/reference_results/r_respa_m2_microfixtures/raw_commands.txt`

Focused dense bookkeeping diagnosis:

```bash
cd /home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tests/reference_results/r_respa_m2_microfixtures/dense_oligomer/dt_0p0005/plain_verlet
/home/kiket/바탕화면/test/ab_builds/respa_m2_exact_three_level/bin/gmx energy -f plain.edr -o plain_terms.xvg -xvg none

cd /home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tests/reference_results/r_respa_m2_microfixtures/dense_oligomer/dt_0p0005/exact_three_level
/home/kiket/바탕화면/test/ab_builds/respa_m2_exact_three_level/bin/gmx energy -f exact.edr -o exact_terms.xvg -xvg none
```

## Strongest Confirmed Finding

The exact 3-level path is reconnected and genuinely active on the M2 microfixture harness.

Evidence:

- both fixtures complete `grompp` and `mdrun` without fatal errors
- the dumped exact schedule reports:
  - `mts = true`
  - `mts-mode = lammps-respa`
  - `mts-level2-factor = 2`
  - `mts-level3-factor = 4`
  - exact inner/middle/outer ownership keys
- `coulomb_toy` is clean:
  - exact schedule active = `true`
  - bookkeeping ok = `true`
  - convergence ok = `true`
  - exact-vs-plain final differences are near machine noise and tighter `dt` improves them further

Primary evidence file:

- `/home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tests/reference_results/r_respa_m2_microfixtures/summary.json`

## Strongest Unresolved Uncertainty

`dense_oligomer` shows that execution alone is not enough; exact 3-level bookkeeping is still not clean once listed terms are present.

The failure is not subtle:

- step-0 exact-vs-plain force difference: `742.348184 kJ/mol/nm`
- step-0 exact-vs-plain potential difference: `8583.465088 kJ/mol`
- coarse exact-vs-plain final potential difference: `8583.863281 kJ/mol`
- fine exact-vs-plain final potential difference: `8583.873779 kJ/mol`

The most concrete localized signal in the current artifacts is the `Coulomb-(SR)` energy term at `t = 0`:

- plain Verlet: `-7160.835449 kJ/mol`
- exact 3-level: `1422.623291 kJ/mol`

Files:

- `/home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tests/reference_results/r_respa_m2_microfixtures/dense_oligomer/dt_0p0005/plain_verlet/plain_terms.xvg`
- `/home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tests/reference_results/r_respa_m2_microfixtures/dense_oligomer/dt_0p0005/exact_three_level/exact_terms.xvg`

This is strong evidence of a real bookkeeping defect in the exact path on this richer fixture, not just another harness false-negative.

## Fixture-by-Fixture Outcome

### `coulomb_toy`

- execution: pass
- exact 3-level schedule active: yes
- bookkeeping: pass
- tight-limit behavior vs plain Verlet: pass
- relationship to legacy two-level:
  - exact is far closer to plain Verlet than legacy at both timesteps

### `dense_oligomer`

- execution: pass
- exact 3-level schedule active: yes
- bookkeeping: fail
- tight-limit behavior vs plain Verlet: fail
- relationship to legacy two-level:
  - legacy remains much closer to plain Verlet than exact on force and potential bookkeeping
  - exact coordinate drift remains small in absolute terms, but the bookkeeping defect dominates the milestone result

## Verdict

`EXACT 3-LEVEL PATH RUNS BUT BOOKKEEPING/CONVERGENCE IS STILL PARTIAL`

## Next Step

Keep the same M2 harness and isolate the dense exact-path bookkeeping defect before any broader scheduler work.

Priority:

1. trace exact 3-level dense `Coulomb-(SR)` bookkeeping against plain Verlet on the same `dense_oligomer` fixture
2. verify whether the bad term is only energy/reporting, or the underlying exact force buffer ownership is also wrong for listed-rich fixtures
3. do not broaden to full-system TRL-5 until this fixture-level exact bookkeeping defect is resolved
