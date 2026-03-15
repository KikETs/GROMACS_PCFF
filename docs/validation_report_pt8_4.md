# PT8.4 — Nonbonded & 1-4 / Exclusions Parity Validation Report

## Overview
This report documents the validation of nonbonded interactions (LJ 9-6), exclusions, and 1-4 scaling for milestone PT8.4. Direct numeric parity between LAMMPS and GROMACS was established using specialized toy fixtures.

## Validated Outcomes
1.  **LJ 9-6 Nonbonded Parity:**
    - System: `lj96_toy` (2 atoms, no bonds)
    - Status: PASS
    - LAMMPS PE: -0.183572 kJ/mol
    - GROMACS PE: -0.183554 kJ/mol
    - Difference: 1.84e-05 kJ/mol
2.  **1-4 / Exclusions Parity:**
    - System: `exclusion_toy` (4 atoms in a chain, 1-4 interaction)
    - Status: PASS
    - LAMMPS PE: -0.095356 kJ/mol
    - GROMACS PE: -0.095356 kJ/mol
    - Difference: 1.04e-10 kJ/mol

## Technical Details
- **LJ 9-6 Implementation:** Verified that the GROMACS fork correctly uses `rep-pow 9.0` in `[ defaults ]` and `nbfunc 1` to implement the 9-6 potential.
- **Combination Rules:** Verified that `CombinationRule::SixthPower` (comb-rule 4) correctly implements PCFF sigma/epsilon mixing.
- **Exclusions:** Verified that `nrexcl 3` correctly excludes 1-2 and 1-3 interactions in GROMACS, matching LAMMPS `special_bonds lj/coul 0.0 0.0 1.0`.
- **1-4 Scaling:** Verified that 1-4 pairs are correctly generated from dihedrals and included at full strength (`fudgeLJ 1.0`).

## Remaining Gaps / Not Validated
- **Coulomb Long-Range Parity:** Minimal Coulomb sanity was explored but not fully automated in this milestone due to engine-specific reciprocal space differences (PPPM vs PME) on extremely small systems.
- **Pair Overrides:** Explicit `[ pairtypes ]` or `pair_coeff` overrides for specific atom pairs were not tested.
- **Combined System Interplay:** Interplay between complex bonded topologies and nonbonded interactions is deferred to the next milestone.

## Artifacts Produced
- `tools/run_pt8_4_nonbonded_parity/run_nonbonded_parity.py`: Nonbonded parity runner.
- `testdata/lammps_golden/systems/lj96_toy/`: LJ 9-6 fixture.
- `testdata/lammps_golden/systems/exclusion_toy/`: Exclusions/1-4 fixture.
- `tests/reference_results/pt8_4_nonbonded_parity/`: Directory containing parity reports.
