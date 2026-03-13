# PCFF Auto-Typing Scope (PT0)

## Status

This document freezes the **PT0** scope only.

PT0 is a specification-and-corpus milestone. It does **not** implement a runtime typing engine, a rule executor, or final topology export from arbitrary chemistry.

That cut is intentional. Freezing broad chemistry support before freezing input constraints, failure modes, and a reproducible golden corpus would be weak engineering.

## PT0 Objective

Freeze all of the following before any runtime atom-typing implementation:

- supported chemistry
- supported input formats
- supported output contracts
- unsupported chemistry and explicit failure modes
- a deterministic golden typing corpus with machine-readable metadata

## Non-Negotiable Project Constraints

- deterministic rule-based typing only
- no ML
- emit a typed intermediate representation before any final topology export
- every assigned atom type and parameter source must be explainable
- unknown or unsupported chemistry must fail explicitly with diagnostics
- outputs must be reproducible and diff-friendly

## Supported Input Formats

PT0 freezes the intended v1 typing input contract to the following:

### Primary supported format

- `mol_v2000`
  - MDL Molfile V2000
  - one molecular component per file
  - explicit bond orders required
  - formal charges required where chemically relevant
  - explicit hydrogens required for hydrogen-bearing organic species
  - hydrogen-free ions are allowed

### Allowed Molfile subset

- atom symbols from the PT0 supported element set only
- bond orders `1` and `2` only
- `M  CHG` charge records allowed
- 2D or 3D coordinates allowed

### Explicitly unsupported input formats in PT0

- SMILES
- PDB
- XYZ
- MOL2
- SDF multi-record input
- Molfile V3000
- CIF
- LAMMPS data/input files as typing-engine input

These are excluded because they either lose critical typing information, add parser ambiguity, or would widen scope before the rulebook is frozen.

## Supported Output Contracts

PT0 freezes the intended output contracts but does not implement the runtime engine that emits them yet.

### Primary typing output

- `typed_system.json`
  - machine-readable typed intermediate representation
  - one record per atom and per typed interaction-relevant feature
  - explicit provenance for every decision
  - deterministic ordering

### Required diagnostic output

- `typing_diagnostics.json`
  - status: `supported`, `unsupported`, or `invalid_input`
  - failure code
  - human-readable diagnostics
  - references to the rule or validation gate that failed

### Out of scope for PT0

- direct GROMACS topology export from arbitrary input chemistry
- direct LAMMPS topology export from arbitrary input chemistry
- force-field parameter lookup beyond the frozen rulebook outline

## Supported Chemistry Coverage

PT0 freezes a narrow chemistry subset that is directly motivated by the polymer-electrolyte target domain.

### Supported elements

- `H`
- `C`
- `O`
- `Li`
- `N`
- `S`
- `F`

### Supported charge states

- neutral organic fragments
- monatomic `Li+`
- monovalent anions represented with explicit formal charge

### Supported molecular families

1. Acyclic saturated hydrocarbons
   - only single-bond `sp3` carbon/hydrogen environments

2. Acyclic ether fragments
   - `C-O-C` polyether-style environments
   - intended to cover PEO-like local environments

3. Monatomic lithium cation
   - standalone `Li+`

4. Explicit TFSI-like sulfonimide anion encoding
   - `N(-)(SO2CF3)2` style graph
   - explicit `S=O`, `S-N`, `S-C`, and `C-F` bond orders
   - explicit formal charge encoding required

### Supported graph constraints

- acyclic organic fragments only
- one molecular component per input file
- no aromatic bond model
- no bond-order guessing
- no protonation-state guessing

## Explicitly Unsupported Chemistry

The following are out of scope in PT0 and must fail explicitly if presented to a future engine:

- aromatic or conjugated ring systems
- generic `sp2`/`sp` carbon chemistry
- carbonyls, carboxylates, esters, aldehydes, ketones, amides
- cyclic ethers and carbonate solvents
- phosphate, borate, perchlorate, or `PF6`/`BF4` salt chemistry
- radicals
- isotopically labeled chemistry
- transition metals or coordination complexes
- hypervalent sulfur encodings outside the frozen TFSI-style pattern
- multi-component assembly in one input file
- implicit-hydrogen inputs for hydrogen-bearing organics

## Unresolved Ambiguities Frozen As Deferred

These are intentionally **not** resolved in PT0:

1. Final published PCFF parameter provenance beyond the local type-family scope.
2. Exact final PCFF atom-type labels for each supported family.
3. Canonical handling of resonance-equivalent encodings outside the explicit PT0 TFSI corpus form.
4. System-level assembly semantics for polymer + ion mixtures from separate typed components.
5. Any rule that would require geometry-based inference rather than graph/charge/bond-order evidence.

If a future implementation needs one of these decisions, that is a later milestone, not PT0.

## Golden Corpus Contract

The PT0 golden corpus lives under:

- `testdata/typing_golden/`

Its role is to freeze:

- supported positive cases
- unsupported negative cases
- exact input files
- exact metadata
- expected support/failure classification
- deterministic file hashes

PT0 does **not** freeze a complete executable typing rule table yet. It freezes the corpus that later rule implementations must satisfy.
