# Typing Golden Corpus

PT0 golden corpus for the deterministic PCFF auto-typing project.

## Scope

This corpus freezes:

- supported positive chemistry cases
- unsupported negative chemistry cases
- supported input format subset
- expected support/failure classification
- deterministic metadata and file hashes

This corpus does **not** include a runtime typing engine.

## Layout

- `corpus_manifest.json`
  - generated machine-readable index
- `cases/<id>/case.json`
  - case metadata
- `cases/<id>/inputs/structure.mol`
  - Molfile V2000 input
- `cases/<id>/expected/outcome.json`
  - expected support/failure outcome for a future deterministic engine
- `cases/<id>/examples/typed_system.json`
  - PT1 parse-only IR example for the checked-in input

Use:

- [tools/build_typing_golden/generate.py](../../tools/build_typing_golden/generate.py)

to regenerate or validate the manifest.

Use:

- `PYTHONPATH=src python3 -m typing_ir export-typing-golden`

to regenerate the parse-only IR examples.
