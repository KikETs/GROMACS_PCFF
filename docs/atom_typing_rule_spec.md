# Atom Typing Rule Specification (PT3)

## Status

This document freezes the **PT3** PCFF atom typing rule engine only.

PT3 implements:

- a deterministic rule representation format
- deterministic component classification and atom-rule matching
- precedence handling
- explanation records for every assigned atom type family
- explicit diagnostics for unsupported, unresolved, and ambiguous cases

PT3 does **not** implement:

- bonded parameter assignment
- nonbonded parameter assignment
- final topology export
- chemistry fallback beyond explicit diagnostics

## Deliverables

Rules live at:

- [rules/pcff_atom_types.json](/home/kiket/바탕화면/test/GROMACS_PCFF/rules/pcff_atom_types.json)

Engine code lives at:

- [src/atom_typing](/home/kiket/바탕화면/test/GROMACS_PCFF/src/atom_typing)

## Why JSON

PT3 uses JSON instead of YAML or TOML.

Reason:

- no extra parser dependency
- deterministic parsing behavior in the standard library
- rule arrays preserve explicit file order
- diff-friendly formatting

The weak alternative would have been YAML with custom parsing quirks. That would add complexity without helping the rule engine.

## Ruleset Schema

Top-level fields:

- `schema_name`
  - fixed: `pcff_atom_type_rules`
- `schema_version`
  - fixed: `1`
- `ruleset_id`
- `description`
- `supported_elements`
- `component_rules`
- `atom_type_rules`

## Component Rule Format

Each component rule contains:

- `rule_id`
- `kind`
  - `support` or `reject`
- `precedence`
  - lower value wins
- `predicate`
  - PT3 uses `predicate.builtin`
- `family`
  - required for `support`
- `failure_code`
  - required for `reject`

Example:

```json
{
  "rule_id": "classify_acyclic_alkane",
  "kind": "support",
  "precedence": 130,
  "family": "acyclic_alkane",
  "predicate": {
    "builtin": "component_is_acyclic_alkane"
  }
}
```

PT3 uses builtin component predicates because the supported PT0 chemistry classes are graph-pattern decisions, not just flat field equality.

## Atom Rule Format

Each atom rule contains:

- `rule_id`
- `component_family`
- `family`
- `precedence`
- `match.conditions`

Each condition contains:

- `path`
- exactly one operator:
  - `equals`
  - `in`
  - `contains`
  - `contains_all`

Example:

```json
{
  "rule_id": "atom_ether_oxygen_sp3",
  "component_family": "acyclic_ether",
  "family": "ether_oxygen_sp3",
  "precedence": 110,
  "match": {
    "conditions": [
      { "path": "element", "equals": "O" },
      { "path": "valence.inferred_valence", "equals": 2 },
      { "path": "neighbor_element_counts", "equals": { "C": 2 } }
    ]
  }
}
```

## Atom Match Context

Atom rules match against a deterministic context assembled from PT1 IR plus PT2 perception.

Available paths include:

- `canonical_index`
- `source_index`
- `source_atom_id`
- `element`
- `formal_charge`
- `neighbor_indices`
- `neighbor_element_counts`
- `valence.*`
- `ring.*`
- `aromaticity.*`
- `coordination.*`
- `improper_center_candidate.*`
- `polymer_connection.*`
- `attached_atom.*`
  - available only for atoms with exactly one bonded neighbor
- `attached_bond.*`
  - available only for atoms with exactly one bonded neighbor
- `component_family`

This split matters:

- `canonical_index` is the PT1 deterministic index used internally by the engine
- `source_index` is the original source atom order from the parsed structure

PT3 keeps both because the frozen PT0 golden corpus expectations are source-order keyed.

## Precedence and Ambiguity Policy

### Component classification

1. Evaluate every component rule.
2. Keep all matching rules.
3. Choose the lowest `precedence`.
4. If exactly one rule matches at that precedence, use it.
5. If multiple rules match at the same best precedence, emit `ambiguous_component_classification`.

### Atom typing

1. Restrict atom rules to the chosen `component_family`.
2. Evaluate every matching rule candidate.
3. Choose the lowest `precedence`.
4. If exactly one rule matches at that precedence, assign the family.
5. If multiple rules match at the same best precedence, emit `ambiguous_atom_type_match`.
6. If no rules match, emit `unresolved_atom_type`.

This is strict on purpose. PT3 does not silently use file order to hide overlapping rules.

## Supported PT3 Component Families

The current ruleset supports exactly the PT0 positive chemistry families:

- `acyclic_alkane`
- `acyclic_ether`
- `lithium_cation`
- `tfsi_like_sulfonimide`

The current ruleset explicitly rejects at least:

- `unsupported_element`
- `unsupported_transition_metal_or_coordination`
- `unsupported_aromatic_sp2_ring`
- `unsupported_resonance_encoding`
- `unsupported_carbonyl_chemistry`
- `unsupported_component_family`

No silent fallback is allowed.

## Typing Report Schema

Top-level fields:

- `schema_name`
  - fixed: `pcff_atom_typing_report`
- `schema_version`
  - fixed: `1`
- `source`
  - hashes for typed IR, chem perception report, and ruleset
  - source id, input format, rules path, ruleset id
- `typing`
  - overall status
- `components`

Per-component fields:

- `component_id`
- `name`
- `atom_count`
- `classification`
- `atoms`
- `atom_type_explanations`
- `diagnostics`

## Per-Atom Output

Each atom record contains:

- `canonical_index`
- `source_index`
- `source_atom_id`
- `element`
- `status`
  - `assigned`
  - `unresolved`
  - `ambiguous`
  - `skipped_unsupported`
  - `skipped_ambiguous`
- `assigned_family`
- `matched_rule_id`
- `precedence`
- `explanation_id`
  - present only when assigned

## Explanation Record Contract

Each assigned atom emits one explanation record.

Fields:

- `explanation_id`
- `atom_index`
  - canonical index
- `source_index`
- `source_atom_id`
- `element`
- `assigned_family`
- `rule_id`
- `precedence`
- `component_family`
- `component_rule_id`
- `canonical_atom_indices`
  - atom plus local neighbors involved in the matched context
- `source_atom_indices`
  - source-order trace for the same local neighborhood
- `evidence`
  - matched conditions with:
    - `path`
    - `operator`
    - `expected`
    - `actual`

This is the PT3 traceability contract. If a user asks why an atom got a family, the answer is in this record, not in hidden control flow.

## Diagnostics Contract

Diagnostics are explicit records, not log text.

Per-diagnostic fields:

- `scope`
  - `component` or `atom`
- `code`
- `message`
- `rule_id`
  - when relevant
- `candidate_rule_ids`
  - for ambiguity
- `atom_index`
  - for atom diagnostics
- `source_index`
  - for atom diagnostics

Frozen PT3 engine-level diagnostic codes:

- `ambiguous_component_classification`
- `ambiguous_atom_type_match`
- `unresolved_atom_type`

PT0 failure codes remain in use for unsupported chemistry rejection.

## Golden Regression Contract

PT3 regression tests compare:

- supported cases:
  - expected family by `source_index`
  - explanation record existence
  - matched rule id presence
- unsupported cases:
  - expected failure code
  - expected diagnostic substrings

This source-index choice is deliberate. The PT0 corpus was authored against source-order atom expectations, and PT3 preserves that mapping explicitly instead of rewriting history.

## Out of Scope

- bond, angle, dihedral, improper, or nonbonded parameter assignment
- final PCFF published atom labels beyond the frozen family names
- topology export
- chemistry guessing for ambiguous structures
