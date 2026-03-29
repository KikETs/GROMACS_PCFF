TP1.6 reruns the TP1.5b `dense_nonlisted` cut-off fixture under:

- one unsafe reference (`n10_r0909`)
- one tight reference (`tight_ref_n1_r1200`)
- three safer regimes (`n1_r0909`, `n10_r0911`, `auto_buffer_n10_vbt0005`)

Goal:
- verify whether the TP1.5b worsening disappears, weakens, or persists once the fixture is moved to safer pairlist/buffer settings
- keep the comparison fair by fixing topology, coordinates, charges, box, cutoff family, seed, and repulsion power

Run:

```bash
python3 tools/run_tp1_6_rebaseline_audit/run_rebaseline_audit.py
```
