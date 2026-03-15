# GROMACS-PCFF Bridge

This project provides an auditable, high-fidelity bridge between the PCFF (Polymer Consistent Force Field) and a specialized GROMACS fork. It ensures that complex polymer topologies and Class2 potentials are mapped correctly between GROMACS and LAMMPS.

## Project Status: v1.0.0-rc1
The bridge has completed its foundational validation phase. It is currently in **Release Candidate 1**.

### Readiness Matrix
| System Type | Readiness | Key Constraint |
| :--- | :--- | :--- |
| **Neutral Polymers** | **Production-Ready** | None (sub-1% density parity) |
| **Charged/Salt Systems** | **Qualified-Ready** | Mandatory density drift review |
| **Transport Properties** | **Out-of-Scope** | Not validated for rc1 |

For details, see the [Release Readiness Matrix](docs/release_readiness_matrix.md).

## Quick Start
1.  **Topology:** Use the PCFF emitter to generate `.top` and `.gro` files.
2.  **Workflow:** Follow the [Neutral Workflow Template](docs/m10_5_neutral_workflow_template.md) or [Charged Workflow Template](docs/m10_5_charged_workflow_template.md).
3.  **Validation:** Review [Validation Report M10.5](docs/validation_report_m10_5.md) for the latest audit results.

## Key Documentation
- [Release Notes (v1.0.0-rc1)](docs/releases/v1.0.0-rc1.md)
- [Known Limitations](docs/known_limitations.md)
- [Troubleshooting Guide](docs/m10_5_troubleshooting.md)

## License
Distributed under the LGPL v2.1. See `COPYING` for details.
