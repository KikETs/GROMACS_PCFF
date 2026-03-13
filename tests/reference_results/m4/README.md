PCFF/Class2 M4 reference outputs derived from the frozen M1 LAMMPS golden corpus.

Contents:
- `small_oligomer/` stores normalized LAMMPS outputs for the charged six-atom oligomer fixture.
- `small_salt_polymer_box/` stores normalized LAMMPS outputs for the periodic polymer-plus-salt fixture.

Each system directory contains:
- `topol.top`
- `initial_nve.gro`
- `single_point.json`
- `forces.json`
- `finite_difference.json`
- `nve_drift.json`
- `nvt_snapshot.json`

Units follow LAMMPS `units real`:
- energies: `kcal/mol`
- distances: `angstrom`
- forces: `kcal/mol/angstrom`

These files freeze the M4 nonbonded reference data for later whole-system CPU parity and short-MD regression work.

`topol.top` and `initial_nve.gro` are the committed GROMACS-side fixtures used by the automated
single-point regression against the frozen LAMMPS outputs.
