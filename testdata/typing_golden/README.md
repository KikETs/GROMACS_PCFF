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

Use:

- [tools/build_typing_golden/generate.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/build_typing_golden/generate.py)

to regenerate or validate the manifest.
