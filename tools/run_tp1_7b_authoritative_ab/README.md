TP1.7b performs a same-build A/B comparison on the authoritative `dense_salt_polymer` charged system.

It uses:

- a same-build unsafe manual pairlist regime: `nstlist=10, rlist=0.909, verlet-buffer-tolerance=-1`
- a same-build safe manual pairlist regime: `nstlist=10, rlist=0.911, verlet-buffer-tolerance=-1`

Both runs keep the current authoritative TP1.3 executed baseline otherwise fixed.

Run:

```bash
python3 tools/run_tp1_7b_authoritative_ab/run_authoritative_ab.py
```
