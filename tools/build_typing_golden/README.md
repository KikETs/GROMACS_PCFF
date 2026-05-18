# `build_typing_golden`

PT0 tooling for the PCFF auto-typing golden corpus.

## Purpose

- regenerate the machine-readable corpus manifest
- stage a deterministic bundle of corpus cases
- validate that the checked-in corpus metadata and file hashes are current

This tool does **not** run any typing rules. It validates corpus structure only.

## Commands

Regenerate the checked-in manifest:

```bash
python3 tools/build_typing_golden/generate.py manifest
```

Stage a deterministic copy of the corpus:

```bash
python3 tools/build_typing_golden/generate.py stage --out output/tmp/typing_golden_stage
```

Validate that the checked-in corpus matches a freshly generated manifest:

```bash
python3 tools/build_typing_golden/generate.py validate
```
