# PT8.4.1 — Cross-Type Mixing Parity Validation Report

## Overview
This report documents the validation of off-diagonal LJ 9-6 mixing rules for milestone PT8.4.1. Direct numeric parity between LAMMPS and GROMACS was established using a two-type toy fixture.

## Validated Outcomes
1.  **Off-Diagonal LJ 9-6 Mixing:**
    - System: `mixing_toy` (2 atoms of different types)
    - Status: PASS
    - LAMMPS PE: -0.257216 kJ/mol
    - GROMACS PE: -0.257178 kJ/mol
    - Difference: 3.81e-05 kJ/mol

## Technical Details
- **Mixing Rule:** Established that `CombinationRule::SixthPower` (comb-rule 4) correctly implements the PCFF mixing rules for both diagonal and off-diagonal terms.
- **Verification Path:** The test uses real engine execution for both LAMMPS and GROMACS, verifying that the GROMACS fork's kernel correctly handles the mixed pair parameters derived from `rep-pow 9.0`.

## Remaining Gaps / Not Validated
- **Combined-System Complexity:** The interplay between mixed nonbonded terms and complex bonded topologies is deferred to PT8.5.
- **Large Type Sets:** Only a 2-type system was tested; however, the underlying mixing rule logic is generic.

## Artifacts Produced
- `tools/run_pt8_4_1_mixing_parity/run_mixing_parity.py`: Mixing parity runner.
- `testdata/lammps_golden/systems/mixing_toy/`: Two-type LJ 9-6 fixture.
- `tests/reference_results/pt8_4_1_mixing_parity/`: Directory containing parity reports.
