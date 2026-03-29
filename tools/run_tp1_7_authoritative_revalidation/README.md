TP1.7 reruns the authoritative `dense_salt_polymer` charged-system tier under explicit safe short-range settings and compares those runs against the trusted TP1.3 historical reference artifact.

This milestone does not patch production code. It only:

- re-extracts authoritative TP1.3 observables from the existing `TRL-0` artifact
- reruns the same system under the preferred TP1.6 safe baseline candidate
- reruns one secondary manual-safe candidate for interpretation
- preserves raw logs plus machine-readable summaries under `tests/reference_results/tp1_7_authoritative_revalidation`

Run:

```bash
python3 tools/run_tp1_7_authoritative_revalidation/run_authoritative_revalidation.py
```
