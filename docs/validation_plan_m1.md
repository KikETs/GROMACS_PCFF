# Validation Plan M1

## What is frozen

- The semantic target is frozen against official LAMMPS documentation for Class2 bonded terms, `lj/class2`, `lj/class2/coul/long`, `special_bonds`, `units real`, and `run_style respa`.
- The M1 golden corpus contract is frozen under `testdata/lammps_golden/`.
- The normalized observable formats are frozen as JSON output files produced by the LAMMPS-side generator.
- Per-system `special_bonds`, KSpace settings, and fixture-local coefficients are frozen as explicit input, not inferred defaults.
- The repository baseline is frozen as "no runtime physics changes yet."

## What remains unresolved

- A global published PCFF parameter provenance is not frozen yet. Current fixtures use repository-local deterministic Class2 coefficients.
- Full LAMMPS `run_style respa` nesting and force-assignment semantics are not yet mapped into a GROMACS design.
- CPU parity tolerances and later GPU tolerances are not frozen yet.
- Restart parity criteria are not frozen beyond diagnostic expectations.
- The future GROMACS topology and parameter serialization format for Class2 terms is not frozen yet.
- Additional edge-case corpus systems for `special_bonds` variants and hybrid long-range settings may still be needed.

## M1 deliverables in this repository

- [docs/pcff_respa_reference_spec.md](./pcff_respa_reference_spec.md)
- [testdata/lammps_golden/corpus_manifest.json](../testdata/lammps_golden/corpus_manifest.json)
- [tools/generate_lammps_golden/README.md](../tools/generate_lammps_golden/README.md)
- [tools/generate_lammps_golden/generate.py](../tools/generate_lammps_golden/generate.py)
- [tools/generate_lammps_golden/compare.py](../tools/generate_lammps_golden/compare.py)
- [tests/reference_schema/test_corpus_schema.py](../tests/reference_schema/test_corpus_schema.py)

## Risks that will affect M2 and M3

1. The current repository MTS code path is explicitly two-level only, while LAMMPS `respa` supports a broader scheduling model. M2 will need a real design decision, not an incremental patch on the current assumption set.
2. Class2 bonded cross terms do not fit cleanly into existing GROMACS bonded abstractions without deciding where coefficient ownership, serialization, and energy accounting live.
3. `lj/class2/coul/long` parity is not just a new functional form; long-range decomposition, exclusions, and mixing semantics must all line up together.
4. GPU work will be fragile if CPU-side data ownership and reference decomposition are not settled first.
5. Restart validation can become misleading if M2 starts coding before the restart contract is explicitly chosen.

## Recommended M2 entry sequence

1. Freeze the GROMACS-side topology/parameter data model for Class2 terms.
2. Implement CPU single-point energy and force parity for `bond_style class2`, then `angle_style class2`, then `dihedral_style class2`, then `improper_style class2`.
3. Implement CPU `lj/class2` and then `lj/class2/coul/long` parity using the golden single-point and force corpus.
4. Only after single-point CPU parity is stable, design the full `run_style respa` scheduling model.
5. Defer CUDA work until the CPU observable contract passes against the golden corpus.
