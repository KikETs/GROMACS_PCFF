# PT8.1 — Force/Energy Sanity Verification Findings

## Overview
This document summarizes the numeric sanity verification between GROMACS and LAMMPS for the Class2 bond potential implemented in the GROMACS-PCFF fork.

## Calculation Details
- **System:** `bond_toy` (2 atoms, 1 bond)
- **LAMMPS Settings:** `units real`, `bond_style class2`, `pair_style lj/class2 9.0`
- **GROMACS Settings:** `[ defaults ]` comb-rule 4, `[ bonds ]` type 11 (Class2), `rvdw = 0.9`

## Results Summary
Potential Energy (kJ/mol):
- **LAMMPS (Reference):** 5.075975
- **GROMACS (Emitter + Fork):** 5.076100
- **Absolute Error:** 0.000125 (0.0025%)

## Interpretation
The error is within the expected range for single-point comparisons given differences in:
1.  **Coordinate Precision:** `system.gro` uses 3 decimal places (standard for .gro files).
2.  **Output Rounding:** `gmx energy` rounds the potential energy output.
3.  **Fundamental Constants:** Slight variations in `kcal` to `kJ` conversions in different packages.

## Conclusion
The GROMACS-PCFF fork correctly interprets and calculates the Class2 bond potential as defined in the LAMMPS-based reference. The bridge successfully converts LAMMPS coefficients and coordinates into a valid GROMACS topology and structure.
