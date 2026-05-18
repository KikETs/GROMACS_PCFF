# Typing IR Specification (PT1)

## Status

This document freezes the **PT1** parser-layer contract only.

PT1 implements:

- a deterministic JSON intermediate representation
- parser support for `mol_v2000`, `sdf`, `mol2`, and `pdb`
- canonical atom indexing
- JSON serialization and deserialization

PT1 does **not** implement:

- atom typing rules
- parameter assignment
- topology export
- chemistry support beyond the parser layer

That separation matters. A file being parsable into IR does **not** mean its chemistry is supported for deterministic PCFF typing.

## Scope Boundary

PT1 sits strictly before typing.

- parser input support:
  - `mol_v2000`
  - `sdf`
  - `mol2`
  - `pdb`
- typing support:
  - unchanged from PT0
  - still defined by [docs/pcff_typing_scope.md](./pcff_typing_scope.md)

The parser layer normalizes structure data into IR and preserves missing information explicitly. It does not guess bond orders, atom types, protonation states, or formal charges when the source format does not encode them.

## Deterministic IR Contract

Top-level JSON object:

- `schema_name`
  - fixed string: `typed_system`
- `schema_version`
  - fixed integer: `1`
- `ir_stage`
  - fixed string: `parsed_only`
- `canonicalization`
  - `algorithm`: `graph_refine_distance_v1`
  - `atom_index_base`: `1`
- `source`
  - `source_id`
  - `input_format`
  - `sha256`
- `typing`
  - `status`: `not_run`
  - `ruleset_id`: `null`
- `components`
  - PT1 requires exactly one connected component per input

Component object:

- `component_id`
- `name`
- `atom_count`
- `bond_count`
- `element_counts`
- `bond_order_histogram`
- `bond_code_histogram`
- `net_formal_charge`
  - integer if every atom carries explicit formal charge information
  - `null` otherwise
- `partial_charge_sum`
  - float if every atom carries a parsed partial charge
  - `null` otherwise
- `atoms`
- `bonds`

Atom object:

- `canonical_index`
- `source_index`
- `source_atom_id`
- `element`
- `atom_name`
- `formal_charge`
- `partial_charge`
- `coordinates`
- `annotations`
- `provenance`

Bond object:

- `canonical_index`
- `source_index`
- `source_bond_id`
- `source_atom_indices`
- `atom_indices`
- `order`
  - integer if the source format encoded a numeric bond order
  - `null` if the source format did not encode a deterministic bond order
- `bond_code`
  - raw normalized bond token from the source format
- `annotations`
- `provenance`

## Canonical Atom Indexing

PT1 canonical atom ordering uses the following precedence:

1. local graph invariants
   - element
   - explicit formal charge if available
   - graph degree
   - incident bond labels
2. iterative neighbor-label refinement
3. geometry-derived distance signature
   - sorted pairwise distances from the atom to the full component
4. coordinate tuple
5. source atom id and source index as final deterministic tie-breaks

This is deterministic and diff-friendly. It is deliberately explainable, and it avoids silent chemistry inference.

The tie-break policy is the weak point. Symmetry-equivalent atoms can still require a final deterministic fallback. PT1 freezes that behavior instead of pretending the problem disappeared.

## Supported Parser Subsets

### `mol_v2000`

- exactly one V2000 structure
- explicit atom table
- explicit bond table
- `M  CHG` supported
- Molfile atom charge codes supported where valid
- non-`M  CHG` property lines are rejected

### `sdf`

- exactly one structure record
- underlying structure must be Molfile V2000
- record data after `M  END` is ignored for IR construction
- multi-record SDF fails explicitly

### `mol2`

- exactly one `@<TRIPOS>MOLECULE` block
- `ATOM` section required
- `BOND` section required for multi-atom inputs
- numeric bond orders are preserved as integers
- non-numeric bond codes are preserved as `bond_code` and emitted with `order: null`
- atom charges are preserved as:
  - `formal_charge` if the parsed value is integral
  - `partial_charge` otherwise

### `pdb`

- single model only
- `ATOM`/`HETATM` records only
- explicit element columns required
- alternate locations rejected
- explicit `CONECT` required for multi-atom inputs
- bond order is not guessed from `CONECT`
  - `bond_code` is emitted as `conect`
  - `order` is emitted as `null`
- formal charge is preserved only if the PDB charge column is populated

## Malformed Input Behavior

Malformed or unsupported parser input must raise a structured error with a stable code.

Frozen parser-layer codes used in PT1:

- `unsupported_input_format`
- `invalid_input`
- `malformed_mol_v2000`
- `malformed_sdf`
- `unsupported_multirecord_sdf`
- `malformed_mol2`
- `malformed_pdb`
- `unsupported_pdb_multiple_models`
- `unsupported_pdb_altloc`
- `unsupported_pdb_missing_element`
- `unsupported_pdb_missing_connectivity`
- `unsupported_multicomponent_input`

Behavior rules:

- missing required sections fail explicitly
- duplicate bonds fail explicitly
- out-of-range atom references fail explicitly
- missing explicit connectivity in PDB fails explicitly
- disconnected multi-atom structures fail explicitly
- unsupported or ambiguous source constructs are preserved only if they can be represented honestly in IR; otherwise they fail

## Serialization Contract

- serialization is JSON with `indent=2` and `sort_keys=true`
- field order is stable through deterministic key sorting
- deserialization validates the PT1 schema before returning the payload
- PT1 tests require exact JSON round-trip equality

## Golden Example Outputs

PT1 adds parse-only example outputs for the PT0 golden inputs under:

- `testdata/typing_golden/cases/<id>/examples/typed_system.json`

These examples are generated with:

```bash
PYTHONPATH=src python3 -m typing_ir export-typing-golden
```

That command intentionally emits parse-only IR:

- `typing.status = not_run`
- no atom typing rules executed
- no parameter assignment executed

## Out of Scope

- any chemistry acceptance beyond parser normalization
- automatic bond inference for PDB
- automatic formal-charge inference for PDB or MOL2
- resonance normalization
- aromaticity perception
- atom typing and parameter lookup
