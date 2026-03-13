# M5 Validation Report

## Scope

This report captures the current short-MD CPU parity status for the frozen PCFF/Class2 M5 fixtures after closing the last NVT harness mismatch.

The relevant harness decisions are now:

- `fourierspacing = 0.08` to match the reciprocal-space accuracy required by the frozen LAMMPS PPPM reference.
- LAMMPS-style Nose-Hoover mapping in the NVT harness:
  - `Tdamp = 50 fs` is mapped to `tau-t = 2*pi*Tdamp`.
  - `nh-chain-length = 3`, matching the default LAMMPS `fix nvt` chain length.
- Final NVT temperature is reconstructed from kinetic energy using the default LAMMPS `compute temp` convention, i.e. translational `3` degrees of freedom removed.
- For NVT only, the GROMACS test mdp uses `comm-mode = Linear` with `nstcomm = 1000`.
  - This is deliberate.
  - It gives the thermostat the same `3N-3` effective degrees of freedom that LAMMPS uses by default.
  - The removal period is far longer than the 20-step fixture, so no actual COM zeroing occurs during the tested trajectory.

## Result Summary

All M5 representative cases now pass:

- `small_oligomer` `nve`
- `small_oligomer` `nvt`
- `small_salt_polymer_box` `nve`
- `small_salt_polymer_box` `nvt`

## Machine-Readable Evidence

Latest run summaries:

- `/tmp/pcff_m5_closed/small_oligomer_nve.json`
- `/tmp/pcff_m5_closed/small_oligomer_nvt.json`
- `/tmp/pcff_m5_closed/small_salt_polymer_box_nve.json`
- `/tmp/pcff_m5_closed/small_salt_polymer_box_nvt.json`

Validation command:

- `python3 tools/pcff_short_md_parity/run.py`

Current workflow guarantees:

- the default GTest filter matches the instantiated `PcffShortMdParity` cases
- stale summary JSON files are removed before each run
- the workflow exits with an error if no per-case summaries are produced
- per-case JSON now emits explicit failure taxonomy fields:
  - `supported_failure_categories`
  - `observed_failure_categories`
  - `harness_notes`

## Numerical Agreement

### small_oligomer nve

- `step0_potential_kcal_mol`: `25.319391` vs `25.320359`
- `initial_total_kcal_mol`: `29.790605` vs `29.791574`
- `final_total_kcal_mol`: `29.763583` vs `29.764563`
- `total_energy_drift_abs_kcal_mol`: `0.027022` vs `0.027011`

### small_oligomer nvt

- `final_potential_kcal_mol`: `6.112342` vs `6.112775`
- `final_total_kcal_mol`: `24.987449` vs `24.988366`
- `final_temperature_K`: `1266.443038` vs `1266.473900`
- `final_pressure_atm`: `-297.203412` vs `-297.223300`

### small_salt_polymer_box nve

- `step0_potential_kcal_mol`: `92.114173` vs `92.114120`
- `initial_total_kcal_mol`: `100.162360` vs `100.162310`
- `final_total_kcal_mol`: `99.992340` vs `99.991653`
- `total_energy_drift_abs_kcal_mol`: `0.170020` vs `0.170657`

### small_salt_polymer_box nvt

- `final_potential_kcal_mol`: `-18.220745` vs `-18.220878`
- `final_total_kcal_mol`: `78.980229` vs `78.979269`
- `final_temperature_K`: `3623.216930` vs `3623.181700`
- `final_pressure_atm`: `-205.044136` vs `-205.154370`
- `ion_distance_nm`: `1.069488319` vs `1.069488023`
- `polymer_end_to_end_nm`: `0.968031525` vs `0.968031826`
- `polymer_rg_nm`: `0.318014224` vs `0.318014238`

## Interpretation

The remaining M5 blocker was not a force-field bug.

It was a thermostat-semantics mismatch in the harness:

- GROMACS NVT with `comm-mode = none` kept the thermostat at `3N` effective degrees of freedom.
- The frozen LAMMPS reference uses the default `compute temp`, which removes translational `3` DOF.
- On the salt box this difference was enough to shift the short 20-step heating trajectory and push final kinetic observables outside tolerance.

After aligning the effective thermostat DOF without introducing actual COM removal during the fixture, the last NVT case converged to the frozen reference within tolerance.

## Current Assessment

M5 is closed.

What is now established:

- End-to-end single-point parity for the representative PCFF fixtures.
- CPU short-MD parity for both representative `nve` fixtures.
- CPU short-MD parity for both representative `nvt` fixtures.
- An automated M5 workflow that separates force-field parity from thermostat-observable semantics.

Failure classification is now explicit at two levels:

- `physics` and `numerics` are emitted per metric in the machine-readable case summaries.
- `harness` is emitted explicitly through `harness_notes`, so thermostat/DOF mapping assumptions are no longer prose-only.

What M5 does not claim:

- Exact stepwise trajectory identity across engines under Nose-Hoover dynamics.
- General thermostat equivalence beyond the frozen 20-step fixtures.

## Readiness

The CPU PCFF path is now behaviorally aligned enough on the frozen M5 fixtures to proceed to the next blocked milestone.
