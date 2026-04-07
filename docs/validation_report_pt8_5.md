# PT8.5 — Combined Small-System Parity Validation Report

## Overview
This report documents combined bonded and nonbonded parity on frozen small fixtures for milestone PT8.5. It supports a small-fixture mechanics claim only; it does not by itself justify broad chemistry readiness, dense charged ensemble readiness, or charged transport readiness.

## Validated Outcomes
1.  **Neutral Combined System Parity:**
    - System: `small_oligomer` (6 atoms, multiple bond/angle/dihedral terms, LJ 9-6, 1-4 interactions)
    - Status: PASS
    - Energy Difference: 0.0184 kJ/mol
    - Max Force Diff: 0.1714 kJ/mol/nm (Relative: 0.0042%)
2.  **Charged/Salt Combined System Parity (Sanity):**
    - System: `small_salt_polymer_box` (10 atoms, polymer chain + Na/Cl, long-range Coulomb)
    - Status: PASS
    - Energy Difference: 0.3455 kJ/mol
    - Max Force Diff: 3.6875 kJ/mol/nm (Relative: 0.0368%)

## Technical Implementation
- **Full Interaction Interplay:** Verified that the bridge correctly combines Class2 bonded terms, LJ 9-6 mixing rules, and 1-4 scaling in a single topology.
- **Coulomb Path Sanity:** Successfully established parity for a charged system using PME in GROMACS and PPPM in LAMMPS. The slightly higher error (0.35 kJ/mol) is expected due to reciprocal-space solver differences on small periodic boxes.
- **Force Vector Parity:** Confirmed that force distribution remains physically consistent (sub-0.1% relative error) even when multiple interaction types overlap.

## Remaining Gaps / Not Validated
- **Large-Scale Dynamics:** Trajectory-level parity over long timescales was not part of this milestone.
- **Thermodynamic Integration:** Free energy and transport property validation are deferred to later stages.
- **Extreme Density/Charge:** Only dilute salt/small box scenarios were validated.
- **Broad Chemistry Readiness:** The charged fixture is a frozen mechanics fixture, not a chemistry-complete PCFF readiness claim.

## Artifacts Produced
- `tools/run_pt8_5_combined_parity/run_combined_parity.py`: Combined parity runner.
- `tests/reference_results/pt8_5_combined_parity/`: Directory containing parity reports and logs.
