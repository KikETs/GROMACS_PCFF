# PT8.3 — Force Vector Parity Bridge Validation Report

## Overview
This report documents the validation of atom-wise force-vector parity for Class2 bonded interactions (bond, angle, dihedral, improper). Milestone PT8.3.1 provided final verification of the improper mapping logic.

## Validated Outcomes
1.  **Force Parity Matrix:**
    - `bond_toy`: PASS (Max Diff: 0.0336 kJ/mol/nm)
    - `angle_toy`: PASS (Max Diff: 0.0173 kJ/mol/nm)
    - `dihedral_toy`: PASS (Max Diff: 0.0052 kJ/mol/nm)
    - `improper_toy`: PASS (Max Diff: 7.65 kJ/mol/nm / 0.06% relative)
2.  **Mapping Reliability:**
    - Improper mapping was empirically tested by swapping $K_1/K_2$ coefficients. The increased error confirmed the original bridge mapping is correct for the GROMACS fork kernel.
3.  **High-Precision Path:**
    - Established the use of 7-decimal coordinate inputs for force verification to bypass standard `.gro` rounding errors.

## Remaining Gaps / Not Validated
- **Non-bonded forces:** Deferred to PT8.4.
- **Combined system interplay:** Realistic polymer topologies with overlapping forces are not yet validated.

## Conclusion
Physical force parity for all Class2 bonded interaction types is now confirmed. PT8.3 is considered PASS.
