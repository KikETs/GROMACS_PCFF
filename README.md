# GROMACS-PCFF Bridge

This project provides an auditable, high-fidelity bridge between the PCFF (Polymer Consistent Force Field) and a specialized GROMACS fork. It ensures that complex polymer topologies and Class2 potentials are mapped correctly between GROMACS and LAMMPS.

## Current Status
The current implementation status is documented in [Current Status Note](docs/current_status_note.md).

Current evidence supports only a narrow locked-scope closure for the recent `r-RESPA` debugging work:
- engine-side `LJ-(SR)` event-669 geometry fix is in place
- locked-scope `LJ-(SR)` comparator branch is closed
- locked-scope `Coulomb-(SR)` residual is still open
- `Potential` remains a secondary aggregate term and is not treated as a simple mirror of `Coulomb-(SR)`

## Quick Start
1.  **Topology:** Use the PCFF emitter to generate `.top` and `.gro` files.
2.  **Workflow References:** Use the [Neutral Workflow Template](docs/m10_5_neutral_workflow_template.md) or [Charged Workflow Template](docs/m10_5_charged_workflow_template.md) as operational references, not as blanket readiness guarantees.
3.  **Status / Validation:** Start with the [Current Status Note](docs/current_status_note.md) and then review the supporting validation reports for the specific scope you care about.

## Key Documentation
- [Current Status Note](docs/current_status_note.md)
- [Known Limitations](docs/known_limitations.md)
- [Troubleshooting Guide](docs/m10_5_troubleshooting.md)

## License
Distributed under the LGPL v2.1. See `COPYING` for details.
