# `pcff_fixture_bridge`

Deterministic bridge foundation for the frozen LAMMPS PCFF/Class2 fixtures.

## Scope

- Reads the repository-local LAMMPS fixture subset under `testdata/lammps_golden/`.
- Emits a typed intermediate representation first.
- Optionally renders a deterministic GROMACS `topol.top` from that IR.
- Fails explicitly on unsupported styles, cross-pair coefficients, or missing Class2 coefficient families.

This tool does not do general chemistry auto-typing. The current milestone is narrower: preserve PCFF/Class2 provenance for the frozen fixtures and stop relying on ACPYPE/GAFF2-prepared GROMACS inputs.

## Supported fixture subset

- `units real`
- `atom_style full`
- `pair_style lj/class2` or `lj/class2/coul/long`
- `pair_modify mix sixthpower`
- `bond_style class2`
- `angle_style class2` or `none`
- `dihedral_style class2` or `none`
- `improper_style class2` or `none`
- `special_bonds lj/coul 0.0 0.0 1.0 angle no dihedral no`
- explicit self-only `pair_coeff`

## Outputs

- `typed_system.json`
  - engine-neutral typed IR in LAMMPS real units
  - every typed record carries a `source` object with file, line, and original text
- `topol.top`
  - deterministic GROMACS topology rendered from the IR
- `bridge_manifest.json`
  - machine-readable list of generated artifacts

## Usage

Emit typed IR only:

```bash
python3 tools/pcff_fixture_bridge/generate.py \
  --out /tmp/pcff_bridge_ir \
  typed-ir
```

Emit typed IR and GROMACS topologies:

```bash
python3 tools/pcff_fixture_bridge/generate.py \
  --out /tmp/pcff_bridge_top \
  export-gromacs
```

Limit export to selected systems:

```bash
python3 tools/pcff_fixture_bridge/generate.py \
  --out /tmp/pcff_bridge_top \
  --system small_oligomer \
  --system small_salt_polymer_box \
  export-gromacs
```

## Determinism and traceability

- JSON is emitted with sorted keys and stable ordering.
- Molecule templates are derived from the fixture graph and emitted in first-seen order.
- Generated `1-4` pairs are derived directly from dihedral records and recorded in the IR.
- Missing Class2 cross-term families abort export with a diagnostic instead of silently degrading the topology.
