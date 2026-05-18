# PCFF/Class2 + LAMMPS-style r-RESPA Reference Specification

## Purpose

This document freezes the M1 reference target for the long-term GROMACS fork work:

1. Full LAMMPS-compatible Class2/PCFF bonded semantics.
2. Full LAMMPS-style `run_style respa` semantics.
3. Later GPU mapping, but not in M1.

M1 does **not** change runtime behavior in this repository. It only freezes the reference semantics and the golden-data contract used for later CPU-first validation.

## Repository baseline

The current repository already contains a limited multiple-time-stepping implementation, but it is not the target behavior for this project:

- [src/gromacs/mdtypes/multipletimestepping.h](../src/gromacs/mdtypes/multipletimestepping.h#L50) defines a fixed set of force groups and asserts that only 0 or 2 MTS levels are supported.
- [src/gromacs/mdtypes/multipletimestepping.cpp](../src/gromacs/mdtypes/multipletimestepping.cpp#L51) hard-codes two-level handling and rejects `mts-levels != 2`.
- A repository search under `src/` and `share/` for `class2`, `pcff`, `lj/class2`, and `coul/long` returns no current implementation hits.

Therefore M1 treats current GROMACS runtime physics as **non-reference** for PCFF/Class2 and full LAMMPS `respa`.

## Primary reference sources

The target semantics are frozen against official LAMMPS documentation, not informal secondary summaries.

- `bond_style class2`: <https://docs.lammps.org/bond_class2.html>
- `angle_style class2`: <https://docs.lammps.org/angle_class2.html>
- `dihedral_style class2`: <https://docs.lammps.org/dihedral_class2.html>
- `improper_style class2`: <https://docs.lammps.org/improper_class2.html>
- `pair_style lj/class2` and `lj/class2/coul/long`: <https://docs.lammps.org/pair_class2.html>
- `special_bonds`: <https://docs.lammps.org/special_bonds.html>
- `run_style respa`: <https://docs.lammps.org/run_style.html>
- `units`: <https://docs.lammps.org/units.html>
- `kspace_style`: <https://docs.lammps.org/kspace_style.html>
- `read_restart`: <https://docs.lammps.org/read_restart.html>
- `read_data`: <https://docs.lammps.org/read_data.html>

## Frozen target scope

### Target force-field terms

The reference scope for later implementation is the exact LAMMPS meaning of these styles and their documented coefficient sub-terms:

- `bond_style class2`
  - Quartic bond term only.
- `angle_style class2`
  - Angle term.
  - Bond-bond cross term.
  - Bond-angle cross term.
- `dihedral_style class2`
  - Dihedral term.
  - Middle-bond-torsion term.
  - End-bond-torsion term.
  - Angle-torsion term.
  - Angle-angle-torsion term.
  - Bond-bond-1,3 term.
- `improper_style class2`
  - Improper term.
  - Angle-angle cross term.
- `pair_style lj/class2`
  - 6/9 Lennard-Jones form.
- `pair_style lj/class2/coul/long`
  - Same 6/9 Lennard-Jones form plus long-range Coulomb coupling through a KSpace solver.

### Explicitly out of M1 scope

- Any GROMACS runtime force calculation changes.
- Any CUDA kernel work.
- Any attempt to map reference semantics into current GROMACS MTS internals.
- Any assumption that current GROMACS MTS behavior is already numerically compatible with LAMMPS `respa`.
- Any undocumented LAMMPS behavior inferred only from examples or folklore.

## Bonded functional forms to preserve

### Bonds

LAMMPS documents `bond_style class2` as:

`E = K2 (r-r0)^2 + K3 (r-r0)^3 + K4 (r-r0)^4`

The coefficient order is `r0 K2 K3 K4`.

### Angles

LAMMPS documents `angle_style class2` as a sum of:

- `Ea = K2 (theta-theta0)^2 + K3 (theta-theta0)^3 + K4 (theta-theta0)^4`
- `Ebb = M (r_ij-r1) (r_jk-r2)`
- `Eba = N1 (r_ij-r1) (theta-theta0) + N2 (r_jk-r2) (theta-theta0)`

The reference implementation must preserve:

- The documented coefficient ordering for `angle_coeff`.
- The degree/radian handling exactly as LAMMPS interprets it in `units real`.
- The fact that bond-angle cross terms depend on the bond lengths associated with the angle ordering.

### Dihedrals

LAMMPS documents `dihedral_style class2` as a sum of:

- `Ed`
- `Embt`
- `Eebt`
- `Eat`
- `Eaat`
- `Ebb13`

The reference implementation must preserve:

- The documented three-term Fourier-like main torsion form.
- All five cross terms and their coefficient orderings.
- The LAMMPS atom-order dependence of the dihedral definition.

M1 does not yet freeze a GROMACS internal decomposition because later implementation must reproduce the total LAMMPS `edihed` behavior first.

### Impropers

LAMMPS documents `improper_style class2` as:

- `Ei = K (chi-chi0)^2`
- `Eaa = M1 (theta_1-theta_10) (theta_2-theta_20) + M2 (theta_1-theta_10) (theta_3-theta_30) + M3 (theta_2-theta_20) (theta_3-theta_30)`

The documented symmetry convention matters:

- The second atom in the improper quadruplet is the central atom.
- The first and third atoms are treated symmetrically.

That ordering is part of the frozen reference.

## Nonbonded target semantics

### Lennard-Jones form

LAMMPS documents the `lj/class2` family as the 6/9 form:

`E = epsilon [ 2 (sigma/r)^9 - 3 (sigma/r)^6 ]`

The reference target is **not** 12/6 Lennard-Jones and not a compatibility approximation.

### Mixing rule semantics

LAMMPS documents the `lj/class2` family as using sixth-power mixing:

- `epsilon_ij = 2 * sqrt(epsilon_i * epsilon_j) * sigma_i^3 * sigma_j^3 / (sigma_i^6 + sigma_j^6)`
- `sigma_ij = ((sigma_i^6 + sigma_j^6) / 2)^(1/6)`

Implications for M1:

- If a golden fixture omits explicit `pair_coeff i j`, later comparison must assume LAMMPS sixth-power mixing, not Lorentz-Berthelot.
- The golden metadata records whether a system relies on mixing or explicit cross coefficients.

### 1-2 / 1-3 / 1-4 and exclusions semantics

LAMMPS does **not** infer a universal PCFF rule from the style name alone. `special_bonds` is an explicit input command.

Frozen M1 rule:

- Each golden system stores its exact `special_bonds` command in metadata and in the LAMMPS input file.
- Later GROMACS parity must match the per-system explicit exclusion/scaling behavior, not a guessed global "PCFF default."

LAMMPS documents special pair weights as independent weights for:

- 1-2 pairs
- 1-3 pairs
- 1-4 pairs

for both:

- `lj`
- `coul`

The M1 corpus intentionally uses explicit `special_bonds` commands and treats them as authoritative fixture data.

### Long-range electrostatics semantics

For charged systems in this corpus, the target is:

- `pair_style lj/class2/coul/long`
- `kspace_style pppm <accuracy>`

Frozen M1 rules:

- Charged golden systems must declare their KSpace style and requested accuracy explicitly.
- The long-range electrostatics contribution to compare later is the LAMMPS decomposition `ecoul` plus `elong` as reported by LAMMPS thermo output.
- CPU parity comes before any future GPU parity.

Open implementation note:

- M1 does not yet freeze PME/PPPM algorithmic internals beyond the LAMMPS observable contract described above.

## Units and conversion rules

The frozen unit system is LAMMPS `units real`.

LAMMPS documents for `real` units:

- distance: Angstrom
- time: femtoseconds
- energy: kcal/mol
- temperature: Kelvin
- pressure: atmospheres
- charge: multiple of electron charge
- velocity: Angstrom/femtosecond
- force: `(kcal/mol)/Angstrom`

Later GROMACS-side comparison must explicitly convert to and from native GROMACS units. The conversion contract is:

- `1 Angstrom = 0.1 nm`
- `1 kcal/mol = 4.184 kJ/mol`
- `1 (kcal/mol)/Angstrom = 41.84 (kJ/mol)/nm`
- `1 fs = 0.001 ps`
- `1 atm = 1.01325 bar`

M1 rule:

- Golden outputs are stored in native LAMMPS `real` units.
- Any later comparison tool must declare conversions explicitly and must never silently compare mixed unit systems.

## Topology semantics

### Atom ordering

LAMMPS data-file atom ordering is part of the reference because:

- bonded tuple order affects angle, dihedral, and improper semantics;
- per-atom force dumps are compared by atom id.

M1 freezes:

- atom ids are stable and 1-based;
- all normalized outputs are sorted by atom id;
- any future GROMACS export used for comparison must preserve or explicitly remap ids.

### Molecular topology source of truth

For M1 fixtures, the source of truth is the checked-in LAMMPS input pair:

- `lammps/system.data`
- `lammps/system.in`

No topology translation layer is considered authoritative in M1.

### Coefficient source of truth

For M1 fixtures, the source of truth is whatever is explicitly written in:

- LAMMPS `*_coeff` commands inside `lammps/system.in`

No coefficients are inferred from force-field names.

## Restart and checkpoint expectations

LAMMPS documents that restart files do not restart "exactly" in all cases and that some settings are not stored and must be specified again after `read_restart`.

Frozen M1 expectation:

- We do **not** require bitwise restart identity as an M1 acceptance criterion.
- We do require that future parity work document which quantities are restored from restart and which input commands must be replayed after `read_restart`.
- Golden generation writes restart files for observable-producing runs only as diagnostic artifacts, not as the primary truth source.

Later M2/M3 restart validation should compare:

- fresh run vs restart-continued run in LAMMPS first;
- then GROMACS vs the corresponding LAMMPS restart observable contract.

## r-RESPA target semantics for later implementation

The target is LAMMPS `run_style respa`, not current GROMACS two-level MTS behavior.

Frozen M1 requirements for later implementation:

- Reference behavior is whatever LAMMPS documents and the golden corpus emits for the chosen `run_style respa` settings.
- Multiple nesting levels must be treated as part of the target design space.
- Level assignment of bonded, pair, and long-range terms must follow explicit LAMMPS input, not GROMACS internal convenience groupings.
- The observable contract must include at least:
  - single-point energy and forces at step 0,
  - short NVE drift traces,
  - short NVT snapshots,
  - restart diagnostics where applicable.

M1 does **not** freeze a final GROMACS internal API for r-RESPA.

## Golden corpus observables

The normalized outputs expected from the corpus are:

- `single_point.json`
  - total energy
  - thermo decomposition where available
- `forces.json`
  - per-atom forces at step 0
- `finite_difference.json`
  - central-difference validation for selected atoms/components
- `nve_drift.json`
  - short deterministic NVE trace
- `nvt_snapshot.json`
  - short deterministic NVT observable trace

For toy systems, some dynamic observables are explicitly disabled in metadata when they add little value.

## Open questions and unresolved items

The following items are intentionally unresolved in M1 and must remain explicit.

1. The project objective says "PCFF(class2)", but LAMMPS style names freeze functional forms, not a unique published PCFF parameter set. M1 fixtures therefore use explicit repository-local coefficients; no global parameter provenance is silently assumed.
2. The exact future GROMACS representation of Class2 cross terms is unresolved.
3. The exact future GROMACS representation of multi-level `run_style respa` scheduling is unresolved.
4. The acceptable numerical tolerance policy for CPU parity vs GPU parity is unresolved. M1 only defines formats and observable classes.
5. Whether additional golden systems are needed for `special_bonds angle yes` / `dihedral yes` edge cases is unresolved.
6. Whether restart parity should be stepwise, ensemble-statistical, or both is unresolved.
7. Whether PPPM-specific tuning knobs beyond requested accuracy must be frozen in metadata is unresolved.

## M1 decisions that are frozen now

- Reference engine: LAMMPS.
- Reference unit system: `units real`.
- Reference bonded styles: `bond/angle/dihedral/improper style class2`.
- Reference nonbonded styles: `lj/class2` and `lj/class2/coul/long`.
- Reference long-range electrostatics interface: explicit `kspace_style`.
- Reference exclusion semantics: explicit per-system `special_bonds`.
- Reference validation artifacts: normalized JSON outputs generated from LAMMPS runs.
- Reference corpus philosophy: small, deterministic, machine-readable fixtures first.
