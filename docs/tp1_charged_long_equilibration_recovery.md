# TP1 - Charged Long-Equilibration Recovery Plan and Result

## 1. System Selection

- Target system: `dense_salt_polymer`
- Identity: 270 atoms, Na/Cl salt in generic polymer matrix
- Source: `testdata/lammps_golden/systems/dense_salt_polymer/`

## 2. Historical TP1.2 Failure

The historical 5 ns TP1.2 attempt failed at `3.017 ns` with thermal runaway.

That run remains a failed artifact, but it is no longer the current exact-system verdict because the processed MDP did not match the intended protocol:

- intended thermostat was not applied: historical `mdout.mdp` has `tcoupl = No`
- intended barostat was not applied: historical `mdout.mdp` has `pcoupl = No`
- runner defect: `tcouple` / `pcouple` / `gen_vel` were used instead of `tcoupl` / `pcoupl` / `gen-vel`

Historical artifacts remain under:

- `tools/run_tp1_2_charged_recovery/run_tp1.py`
- `tests/reference_results/tp1_charged_recovery/dense_salt_polymer/`

## 3. Corrected Exact Recovery

The corrected recovery rerun uses:

- runner: `tools/run_tp1_exact_recovery/run_tp1_exact.py`
- audit script: `tools/run_tp1_exact_recovery/audit_tp1_exact.py`
- execution target: `dense_salt_polymer`
- duration: `5000 ps`
- time step: `0.001 ps`
- thermostat: `tcoupl = v-rescale`
- barostat: `pcoupl = Berendsen`
- velocity generation: `gen-vel = yes`

Corrected artifacts are stored under:

- `tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/`

## 4. Corrected Recovery Result

Final-window analysis over `4000-5000 ps`:

- completed duration: `5000.0 ps`
- mean temperature: `299.8209499100899 K`
- max temperature: `360.481812 K`
- mean density: `1571.6190931828173 kg/m^3`
- mean volume: `4.636349732267733 nm^3`
- corrected protocol contract: `PASS`
- raw artifact bundle: `PASS`
- exact TP1 thermal-runaway blocker: `PASS`

Endpoint caveat:

- final box: `1.66790 1.66790 1.66790 nm`
- cutoffs: `0.9 nm`
- half-box margin: `-0.06605000000000005 nm`
- endpoint continuation safety: `FAIL`

## 5. Current Verdict

TP1 thermal-runaway recovery is `PASS` only for the corrected 5 ns exact-system NPT rerun.

TP1 does not establish:

- dense GROMACS-vs-LAMMPS density or volume parity
- endpoint continuation safety from the final coordinates
- charged transport readiness
- generic charged dense-box production readiness
