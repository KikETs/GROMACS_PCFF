# PT8.1 — Independent Physical Sanity Bridge Validation Report

## Overview
This report documents the validation of the GROMACS-PCFF bridge for milestone PT8.1. The goal was to ensure independent physical sanity of the GROMACS topology emitter and numeric parity with LAMMPS on a minimal supported system.

## Validated Outcomes
1.  **Independent Unit-Conversion Validation:**
    All conversion factors used in `common.py` (kcal to kJ, Angstrom to nm, etc.) have been tested against reference values.
2.  **GROMACS Smoke Workflow:**
    A complete workflow from LAMMPS fixture to GROMACS `.tpr` (via `grompp`) and single-point energy (via `mdrun -rerun`) was successfully executed.
3.  **LAMMPS Smoke Workflow:**
    A LAMMPS `run 0` was executed on the same fixture to provide a reference potential energy.
4.  **Direct Numeric Sanity Comparison:**
    A comparison of potential energy for the `bond_toy` system showed excellent agreement between LAMMPS and GROMACS.

## System: `bond_toy`
- **Description:** 2 atoms, 1 Class2 bond.
- **LAMMPS Potential Energy:** 5.075975 kJ/mol (1.213187 kcal/mol)
- **GROMACS Potential Energy:** 5.076100 kJ/mol
- **Difference:** 1.25e-04 kJ/mol (within 1e-3 kJ/mol tolerance)

## Remaining Gaps / Not Validated
- **Trajectory-level parity:** Only a single frame was compared.
- **Force parity:** Numeric comparison of forces was not automated in this milestone, although the `mdrun` execution implies forces were calculated.
- **Advanced chemistry:** Only the `bond_toy` system (Class2 bonds) was tested. Angles, dihedrals, and impropers were not explicitly compared in this sanity check, although the emitter logic for them is present.
- **Non-bonded interactions:** Tested with a simple LJ cutoff; long-range electrostatics (KSpace) were not part of this minimal sanity check.

## Artifacts Produced
- `tools/run_pt8_1_sanity/run_sanity.py`: Main sanity check script.
- `tests/test_unit_conversions.py`: Unit conversion test suite.
- `tests/reference_results/pt8_1_sanity/bond_toy/`: Directory containing LAMMPS and GROMACS inputs/outputs and the final report.
