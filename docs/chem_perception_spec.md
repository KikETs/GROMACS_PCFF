# Chemical Perception Specification (PT2)

## Status

This document freezes the **PT2** chemical-perception layer only.

PT2 implements:

- deterministic local feature extraction on top of PT1 `typed_system` IR
- valence inference from explicit local bonding evidence
- ring membership and ring-system detection
- aromaticity handling with explicit `aromatic`, `non_aromatic`, and `indeterminate` states
- coordination-environment summaries
- neighbor-shell query support
- improper-center candidacy
- polymer connection point tags from explicit placeholders or annotations

PT2 does **not** implement:

- final atom typing assignment
- parameter lookup
- topology export
- chemistry guessing from missing bond orders or missing formal charges

That boundary is intentional. PT2 computes the feature surface that later deterministic typing rules may consume, but it does not assign PCFF atom types yet.

## Module Layout

Source lives under:

- [src/chem_perception](/home/kiket/바탕화면/test/GROMACS_PCFF/src/chem_perception)

Main entry points:

- `chem_perception.perceive_ir(ir)`
- `chem_perception.perceive_file(path, input_format=..., source_id=...)`
- `chem_perception.query_neighbor_shell(report, atom_index, depth, heavy_only=False)`

## Input Contract

PT2 consumes the PT1 parse-only IR:

- `schema_name = "typed_system"`
- `schema_version = 1`
- `ir_stage = "parsed_only"`
- one connected component only

The perception layer validates that contract before computing features.

## Output Contract

Top-level report:

- `schema_name`
  - fixed: `chem_perception`
- `schema_version`
  - fixed: `1`
- `source`
  - typed IR provenance and hash
- `perception`
  - model/version identifiers for ring, aromaticity, valence, and polymer-tag logic
- `components`
  - PT2 requires exactly one component

Per-component fields:

- `component_id`
- `name`
- `atom_count`
- `bond_count`
- `atoms`
- `bonds`
- `rings`
- `ring_systems`
- `aromatic_systems`
- `polymer_connection_points`

## Computed Atom Features

Every atom record contains:

- `canonical_index`
- `element`
- `formal_charge`
- `neighbor_indices`
- `neighbor_links`
  - neighbor atom index
  - bond index
  - bond code
  - numeric order if known
- `heavy_neighbor_indices`
- `hydrogen_neighbor_indices`
- `neighbor_element_counts`

### `valence`

Fields:

- `known`
- `allowed_valences`
- `explicit_bond_order_sum`
- `inferred_valence`
- `pi_bond_count`
- `status`

Meaning:

- `explicit_bond_order_sum`
  - sum of explicit numeric bond orders on incident bonds
  - `null` if any incident bond has unknown numeric order
- `allowed_valences`
  - deterministic local table keyed by `(element, formal_charge)`
- `inferred_valence`
  - populated only when `explicit_bond_order_sum` exactly matches an allowed value
- `pi_bond_count`
  - count of incident multiple bonds plus explicit aromatic bond codes

Frozen valence statuses:

- `exact`
- `underfilled`
- `overfilled`
- `ambiguous`
- `indeterminate_missing_bond_orders`
- `indeterminate_missing_formal_charge`
- `not_modeled`

Important constraint:

- PT2 does not infer missing formal charges.
- PT2 does not infer bond orders from geometry or atom labels.

### `ring`

Fields:

- `is_ring_atom`
- `ring_ids`
- `ring_system_ids`
- `ring_sizes`
- `smallest_ring_size`

Meaning:

- ring membership is derived from shortest-cycle detection over the explicit graph
- ring systems are connected components of ring bonds
- `smallest_ring_size` is the minimum detected cycle size containing the atom

### `aromaticity`

Fields:

- `status`
- `reason`
- `aromatic_ring_ids`

Frozen aromaticity states:

- `aromatic`
- `non_aromatic`
- `indeterminate`

PT2 never guesses aromaticity when bond-order evidence is missing. `indeterminate` is a first-class output, not a soft failure.

### `coordination`

Fields:

- `coordination_number`
- `heavy_coordination_number`
- `hydrogen_coordination_number`
- `geometry_hint`

Frozen geometry hints:

- `monatomic`
- `terminal`
- `linear_or_sp_candidate`
- `bent_or_chain_candidate`
- `trigonal_planar_candidate`
- `trigonal_pyramidal_candidate`
- `tetrahedral_candidate`
- `tetrahedral_hypervalent_candidate`
- `trigonal_bipyramidal_candidate`
- `octahedral_candidate`
- `hypercoordinate_candidate`

These are coordination hints for later rules, not final hybridization labels.

### `improper_center_candidate`

Fields:

- `is_candidate`
- `kinds`
- `ordered_neighbor_indices`

Frozen candidate kinds:

- `planar_trigonal`
- `tetrahedral_distinct_substituents`

Behavior:

- `planar_trigonal`
  - degree 3
  - aromatic membership or at least one explicit local pi bond
- `tetrahedral_distinct_substituents`
  - degree 4
  - four substituent signatures are locally distinct

The ordered neighbor list is deterministic and intended for later improper-term construction rules.

### `polymer_connection`

Fields:

- `is_placeholder`
- `tags`
- `tag_sources`

PT2 does **not** guess polymer attachment points from generic terminal atoms.

Connection tags are emitted only from explicit evidence:

- atom annotation `polymer_connection_label`
- placeholder atom elements matching `Du`, `R<number>`, or `*`

If a placeholder is bonded to a real atom, the tag is propagated to that bonded atom and recorded in `polymer_connection_points`.

## Computed Bond Features

Every bond record contains:

- `canonical_index`
- `atom_indices`
- `order`
- `bond_code`
- `ring`
  - `is_ring_bond`
  - `ring_ids`
  - `ring_sizes`
  - `smallest_ring_size`
- `aromaticity`
  - `status`
  - `reason`
  - `aromatic_ring_ids`

## Computed Ring Features

Every ring record contains:

- `ring_id`
- `ring_system_id`
- `atom_indices`
- `bond_indices`
- `size`
- `aromaticity`
  - `status`
  - `electron_count`
  - `reason`

### Ring model

PT2 uses `edge_shortest_cycle_v1`:

- for each bond, remove that bond
- find the shortest remaining path between its endpoints
- if one exists, that bond belongs to a cycle
- deduplicate cycles canonically

This produces deterministic local ring features without importing an opaque chemistry toolkit.

## Aromaticity Model

PT2 uses `explicit_and_kekule_huckel_v1`.

Precedence:

1. If all bonds in a cycle have explicit `bond_code = "ar"`, the cycle is `aromatic`.
2. If aromatic bond codes are mixed with localized numeric orders, the cycle is `indeterminate`.
3. If any in-ring bond order is unknown, the cycle is `indeterminate`.
4. Otherwise PT2 evaluates a localized Hückel-style model on the explicit cycle.

Localized Hückel contributions:

- atom with exactly one in-ring multiple bond: contributes `1`
- hetero atom in `{N, O, S, P}` with only in-ring single bonds and no exocyclic multiple bond: contributes `2`
- carbanion with only in-ring single bonds and no exocyclic multiple bond: contributes `2`

The cycle is aromatic only if:

- every atom contributes to a continuous pi system
- total electron count satisfies `4n + 2`

If evidence is insufficient, the result is `indeterminate`, not guessed.

## Neighbor-Shell Queries

PT2 exposes:

```python
query_neighbor_shell(report, atom_index, depth, heavy_only=False)
```

Return fields:

- `atom_index`
- `depth`
- `atom_indices`
- `element_counts`

Definition:

- shell depth is shortest-path distance in the explicit graph
- returned atoms are exactly at that distance
- traversal is deterministic because neighbor order follows canonical indices

This API exists because later typing rules often depend on 1-shell and 2-shell context, but the required shell depth is rule-specific.

## Determinism Rules

PT2 outputs are deterministic because:

- input atoms already have canonical PT1 indices
- cycles are canonically deduplicated
- neighbor ordering is canonical-index ordered
- JSON serialization uses `indent=2` and sorted keys

PT2 tests explicitly check deterministic repeated perception and exact JSON round-trip behavior.

## What PT2 Refuses To Guess

PT2 intentionally does **not** guess:

- missing bond orders in PDB-derived graphs
- missing formal charges
- aromaticity from geometry alone
- polymer attachment points from uncapped terminal atoms
- final atom types

When evidence is incomplete, PT2 preserves `indeterminate` or `not_modeled` states instead of inventing chemistry.
