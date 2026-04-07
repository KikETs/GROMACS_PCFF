# TP1 - Charged Long-Equilibration Recovery Validation Report

## 1. Executive Summary

The historical TP1 `dense_salt_polymer` thermal-runaway blocker is now superseded for the exact corrected protocol.

This is not a charged transport-readiness claim.

The earlier TP1.2 artifact remains a real failed historical run: it stopped at `3.017 ns` with thermal runaway. That failure is no longer treated as the current exact-system verdict because the processed `mdout.mdp` shows the intended thermostat and barostat were not applied:

- historical `mdout.mdp`: `tcoupl = No`
- historical `mdout.mdp`: `pcoupl = No`
- historical runner defect: `tcouple` / `pcouple` / `gen_vel` were used instead of GROMACS keys `tcoupl` / `pcoupl` / `gen-vel`

The corrected direct rerun completes `5.000 ns` on the same authoritative 270-atom `dense_salt_polymer` system with the intended `tcoupl = v-rescale`, `pcoupl = Berendsen`, and `gen-vel = yes` contract applied.

## 2. System Details

- System ID: `dense_salt_polymer`
- Composition: Na/Cl salt in polymer electrolyte matrix
- Size: 270 atoms
- Source: `testdata/lammps_golden/systems/dense_salt_polymer/`
- Corrected runner: `tools/run_tp1_exact_recovery/run_tp1_exact.py`
- Audit script: `tools/run_tp1_exact_recovery/audit_tp1_exact.py`

## 3. Corrected Protocol

- Duration: `5000 ps`
- Time step: `0.001 ps`
- Steps: `5,000,000`
- Thermostat: `tcoupl = v-rescale`
- Barostat: `pcoupl = Berendsen`
- Velocity generation: `gen-vel = yes`
- Electrostatics: `PME`, `rcoulomb = 0.9 nm`
- VdW: `Cut-off`, `rvdw = 0.9 nm`
- Analysis window: final `1000 ps`
- Thermal thresholds: mean temperature `300 +/- 20 K`, max temperature `<= 400 K`

The Berendsen warning was explicitly allowed with `-maxwarn 1` because this rerun is a historical-protocol recovery, not a production NPT protocol recommendation.

## 4. Results

Corrected 5 ns recovery:

- Status: `PASS` for the exact TP1 thermal-runaway blocker
- Completed duration: `5000.0 ps`
- Final-window mean temperature: `299.8209499100899 K`
- Final-window max temperature: `360.481812 K`
- Final-window mean density: `1571.6190931828173 kg/m^3`
- Final-window mean volume: `4.636349732267733 nm^3`
- Corrected `mdout` contract: `PASS`
- Raw artifact bundle: `PASS`

Important caveat:

- Final box: `1.66790 1.66790 1.66790 nm`
- Active cutoffs: `rlist = 0.9 nm`, `rcoulomb = 0.9 nm`, `rvdw = 0.9 nm`
- Half-box margin: `-0.06605000000000005 nm`
- Endpoint cutoff-margin verdict: `FAIL`

This means the corrected 5 ns run resolves the historical thermal-runaway blocker, but the final coordinates are not continuation-safe or transport-entry-ready as-is.

## 5. Artifacts

Historical failed run:

- Summary: `tests/reference_results/tp1_charged_recovery/dense_salt_polymer/recovery_summary.json`
- Engine log: `tests/reference_results/tp1_charged_recovery/dense_salt_polymer/tp1_equil.log`
- Historical `mdout`: `tests/reference_results/tp1_charged_recovery/dense_salt_polymer/mdout.mdp`
- Drift data: `tests/reference_results/tp1_charged_recovery/dense_salt_polymer/drift_analysis.csv`
- Energy trace: `tests/reference_results/tp1_charged_recovery/dense_salt_polymer/energy_raw.xvg`

Corrected 5 ns run:

- Audit: `tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/tp1_exact_recovery_audit.json`
- Report: `tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/tp1_exact_recovery_report.json`
- Protocol: `tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/tp1_exact_protocol.json`
- Engine log: `tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/tp1_equil.log`
- Energy file: `tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/tp1_equil.edr`
- Energy trace: `tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/tp1_energy.xvg`
- Checkpoint: `tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/tp1_equil.cpt`
- Final coordinates: `tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/tp1_equil.gro`
- Processed MDP: `tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/tp1_equil_mdout.mdp`
- SHA-256 manifest: `tests/reference_results/tp1_exact_recovery/dense_salt_polymer_corrected_npt_5ns/sha256_manifest.txt`

## 6. Verdict

- Exact TP1 thermal-runaway blocker: `PASS`
- Corrected protocol contract: `PASS`
- Raw artifact bundle: `PASS`
- Endpoint continuation safety: `FAIL`
- Charged transport readiness: `FAIL`

The correct present-tense claim is narrow: the exact TP1 thermal-runaway blocker is resolved for the corrected 5 ns `dense_salt_polymer` NPT rerun only. It does not establish dense GROMACS-vs-LAMMPS parity, endpoint continuation safety, production readiness, or charged transport readiness.
