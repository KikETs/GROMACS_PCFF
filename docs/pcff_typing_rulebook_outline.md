# PCFF Typing Rulebook Outline (PT0)

## Purpose

This document freezes the structure of the future deterministic rulebook without implementing the rules yet.

The rulebook is intentionally an outline, not an executable specification, because PT0 is only meant to close scope and corpus ambiguity.

## Rulebook Architecture

The future typing engine should execute rules in the order below.

1. Input gate
   - validate file format
   - validate supported element set
   - validate explicit formal charge representation
   - validate explicit hydrogens where required
   - reject unsupported Molfile features before chemistry reasoning starts

2. Graph normalization
   - build an atom/bond graph from the accepted input
   - preserve atom ordering from input
   - preserve formal charges exactly
   - do not infer aromaticity
   - do not infer missing bond orders

3. Component classification
   - determine whether the component is:
     - acyclic alkane fragment
     - acyclic ether fragment
     - monatomic lithium cation
     - explicit TFSI-like sulfonimide anion
   - reject everything else with an explicit failure code

4. Local environment extraction
   - element
   - degree
   - bond-order neighborhood
   - formal charge
   - first-neighbor element multiset
   - component family tag

5. Type-family assignment
   - assign a frozen **type family** to each atom
   - PT0 freezes the family concept, not the final PCFF label string
   - families visible in the corpus include:
     - `alkane_carbon_sp3`
     - `hydrogen_on_alkane_sp3`
     - `ether_alpha_carbon_sp3`
     - `ether_oxygen_sp3`
     - `hydrogen_on_ether_alpha_carbon`
     - `lithium_monoatomic_cation`
     - `sulfonimide_n_anion`
     - `sulfonyl_sulfur`
     - `sulfonyl_oxygen`
     - `trifluoromethyl_carbon`
     - `fluorine_on_trifluoromethyl`

6. Provenance attachment
   - every assigned family must carry:
     - the matched rule id
     - the evidence used
     - the source atom indices involved

7. Diagnostic emission
   - unsupported chemistry must return a stable failure code
   - ambiguous or invalid inputs must not silently downgrade to a nearby rule

## Rule Precedence

The future implementation should use strict precedence:

1. input invalidity
2. unsupported chemistry rejection
3. exact family-specific positive matches
4. component-level consistency checks
5. final typed IR emission

This ordering matters. A broad positive match must never hide an invalid or unsupported input condition.

## Required Rulebook Tables

A later implementation must provide, at minimum:

- supported element whitelist
- supported formal charge table
- supported bond-order patterns
- supported component-family signatures
- per-family atom-environment predicates
- stable failure code table
- typed IR field definitions

PT0 does not fill those tables with complete production content yet.

## Failure Taxonomy To Freeze Now

The following failure codes are frozen for PT0-level corpus and documentation:

- `unsupported_input_format`
- `unsupported_molfile_feature`
- `input_missing_explicit_hydrogens`
- `unsupported_element`
- `unsupported_multicomponent_input`
- `unsupported_aromatic_sp2_ring`
- `unsupported_carbonyl_chemistry`
- `unsupported_pf6_bf4_phosphate_borate_family`
- `unsupported_transition_metal_or_coordination`
- `unsupported_resonance_encoding`
- `unsupported_component_family`

## Ambiguity Policy

The future engine must not resolve these by guess:

- aromatic vs localized bond representations
- protonation state if charge is absent
- resonance redistribution if the encoding differs from the frozen supported pattern
- missing hydrogens on hydrogen-bearing organics
- coordination bonding to lithium or other metals

The correct response is an explicit diagnostic, not a guessed type.

## What PT0 Does Not Freeze

- final PCFF atom-type names
- parameter table lookups
- bond/angle/dihedral/improper parameter assignment rules
- direct topology export rules
- mixed-component system assembly rules

Those are later milestones.
