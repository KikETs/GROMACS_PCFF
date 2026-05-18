# LUNAR/LAMMPS Data Bridge

## Scope

This bridge imports a LUNAR/LAMMPS `.data` file that already contains PCFF/Class2
`Masses`, `Pair Coeffs`, bonded coefficients, Class2 cross-term coefficients,
and topology sections. It emits:

- `typed_system.json`
- `topol.top`
- `system.gro`
- `bridge_manifest.json`

The bridge does not run atom typing from `chain_fixed.mol2`. It preserves the
LUNAR/LAMMPS type ids, charges, coefficients, and topology records from the
`.data` file.

## Supported Input Subset

- Orthorhombic LAMMPS data box bounds.
- `Atoms # full` records with id, molecule id, type id, charge, and coordinates.
- PCFF/Class2 coefficient sections:
  - `Pair Coeffs`
  - `Bond Coeffs`
  - `Angle Coeffs`
  - `BondBond Coeffs`
  - `BondAngle Coeffs`
  - `Dihedral Coeffs`
  - `MiddleBondTorsion Coeffs`
  - `EndBondTorsion Coeffs`
  - `AngleTorsion Coeffs`
  - `AngleAngleTorsion Coeffs`
  - `BondBond13 Coeffs`
  - `Improper Coeffs`
  - `AngleAngle Coeffs`
- Topology sections:
  - `Atoms`
  - `Bonds`
  - `Angles`
  - `Dihedrals`
  - `Impropers`

Missing Class2 cross-term families abort export. Unsupported chemistry is not
silently downgraded.

## Style Assumptions

LAMMPS data files do not encode the full input script. The bridge records these
assumptions explicitly in `bridge_manifest.json` and `typed_system.json`:

- `units real`
- `atom_style full`
- `pair_modify mix sixthpower`
- `special_bonds lj/coul 0.0 0.0 1.0 angle no dihedral no`
- `kspace_style pppm 1.0e-6` unless disabled by CLI

The short form `special_bonds lj/coul 0.0 0.0 1.0` is normalized to the effective
GROMACS export contract above.

## Usage

```bash
python3 tools/pcff_fixture_bridge/lammps_data_bridge.py \
  --data /path/to/chain_fixed_typed_nodup_IFF.data \
  --out /tmp/lunar_gromacs_pcff \
  --system-id Traj_14764
```

## Validation Evidence

Validated in the branch `lunar-lammps-data-gromacs-bridge`:

- Unit/reference tests: `PYTHONPATH=. pytest -q tests/test_unit_conversions.py tests/reference_schema/test_pcff_fixture_bridge.py`
- Result: `13 passed`
- Actual LUNAR output imported:
  - source: `../batch_runs/Traj_14764/build/lunar_pcff/chain_fixed_typed_nodup_IFF.data`
  - atoms: 359
  - bonds: 358
  - angles: 663
  - dihedrals: 762
  - impropers: 408
  - `grompp` smoke: PASS
  - 0-step `mdrun` smoke: PASS
- PolyGen example data imported:
  - source: `../PolyGen/Example-simulation-files/equilibration/system.lmp`
  - atoms: 5475
  - bonds: 5460
  - angles: 10125
  - dihedrals: 11610
  - impropers: 6240
  - `grompp` smoke: PASS
  - 0-step `mdrun` smoke: PASS

## Claim Boundary

This bridge proves that the relevant LUNAR/LAMMPS `.data` topology and PCFF/Class2
coefficient payload can be converted into GROMACS-readable `topol.top` and
`system.gro` artifacts through the GROMACS_PCFF Class2 topology path.

It does not prove:

- LAMMPS-vs-GROMACS energy parity.
- LAMMPS-vs-GROMACS force parity.
- NPT density or volume convergence.
- r-RESPA production readiness.
- Conductivity or transport readiness.
- Support for LAMMPS input-script operations such as `deposit`, `include`,
  `molecule`, or dynamic ion insertion.

Those require separate parity and ensemble gates.
