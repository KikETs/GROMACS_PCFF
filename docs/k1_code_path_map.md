# K1 — Code Path Map

This document maps the primary source files and functions involved in the custom GROMACS-PCFF implementation.

## 1. Listed Force Kernels
| Function | File | Physical Role |
| :--- | :--- | :--- |
| `bond_class2` | `src/gromacs/listed_forces/bonded.cpp` | Quartic bond potential. |
| `angle_class2` | `src/gromacs/listed_forces/bonded.cpp` | Quartic angle and Angle-Angle/Bond-Angle cross terms. |
| `dihedral_class2` | `src/gromacs/listed_forces/bonded.cpp` | Torsion and multiple cross terms (BT, AT, AAT, BB13T). |
| `improper_class2` | `src/gromacs/listed_forces/bonded.cpp` | Improper torsion and Angle-Angle cross terms. |

## 2. Nonbonded Kernels
| Component | File | Physical Role |
| :--- | :--- | :--- |
| `repulsionPower` | `src/gromacs/mdtypes/ffparams.h` | Stores global $n$ for $1/r^n$. |
| `Plain-C loop` | `src/gromacs/nbnxm/kernels_reference/kernel_ref_inner.h` | Reference nonbonded evaluator used for $n \neq 12$. |
| `LJ function` | `src/gromacs/nbnxm/simd_lennardjones_functions.h` | SIMD LJ implementation (disabled for 9-6). |

## 3. Parameter Mapping
| Task | File | Function |
| :--- | :--- | :--- |
| Topology conversion | `src/gromacs/gmxpreprocess/convparm.cpp` | `enter_function` / `assign_param`. |
| Table scaling | `src/gromacs/tables/forcetable.cpp` | `make_tables` / `fill_table`. |
| Derivative prefactors | `src/gromacs/mdlib/forcerec.cpp` | `makeNonBondedParameterLists`. |
