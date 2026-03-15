# PT8.2 — Multi-Term Bonded Parity Bridge Validation Report

## Overview
This report documents the validation of Class2 multi-term bonded interactions (angles, dihedrals, impropers) for milestone PT8.2. Numeric parity between LAMMPS and GROMACS was verified using specialized toy fixtures.

## Validated Outcomes
1.  **Angle-Focused Toy Parity:**
    - System: `angle_toy` (3 atoms, 1 Class2 angle)
    - Exercises: `angle_class2` (quartic angle, bond-bond, bond-angle)
    - Status: PASS (Diff: 0.097 kJ/mol)
2.  **Dihedral-Focused Toy Parity:**
    - System: `dihedral_toy` (4 atoms, 1 Class2 dihedral)
    - Exercises: `dihedral_class2` (Fourier-like torsion, middle-bond torsion, end-bond torsion, angle-torsion, angle-angle-torsion, bond-bond-13)
    - Status: PASS (Diff: 3.36e-05 kJ/mol)
3.  **Improper-Focused Toy Parity:**
    - System: `improper_toy` (4 atoms, 1 Class2 improper)
    - Exercises: `improper_class2` (harmonic improper chi, angle-angle)
    - Status: PASS (Diff: 0.0038 kJ/mol)

## Multi-Term Coverage
| Engine Term | Exercise System | Coverage Notes |
| :--- | :--- | :--- |
| `angle_class2` | `angle_toy` | $k_2, k_3, k_4, bb, ba$ validated. |
| `dihedral_class2` | `dihedral_toy` | $k_1, k_2, k_3, mbt, ebt, at, aat, bb13$ validated. |
| `improper_class2` | `improper_toy` | $k_0, aa$ validated. |

## Remaining Gaps / Not Validated
- **Coordinate Precision:** Parity for `angle_toy` is affected by the low precision of `.gro` files (3 decimal places), leading to a higher but acceptable error (0.1 kJ/mol).
- **Trajectory validation:** Verification was limited to single-point energies.
- **Force parity:** Numeric comparison of force vectors was not automated.
- **Non-bonded interactions:** Cross-term interactions between bonded and non-bonded forces are not yet fully explored.

## Artifacts Produced
- `tools/run_pt8_2_sanity/run_parity.py`: Extended sanity runner.
- `tests/reference_results/pt8_2_sanity/`: Directory containing inputs, outputs, and report JSONs.
