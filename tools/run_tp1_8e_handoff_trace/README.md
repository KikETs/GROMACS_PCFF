TP1.8e reruns the authoritative safe baseline and the narrowed Ewald variant with one-level-deeper handoff tracing enabled.

The runner writes one CSV row per traced stage:
- `after_do_force_return`
- `before_update_coords`
- `after_update_coords`
- `after_compute_globals`

The trace is aggregate, not per-pair. It is intended to answer whether the PME-vs-Ewald force-side difference that survived TP1.8d still survives into:
- the force buffer handed into `Update::update_coords`
- the immediate post-update state
- the later virial/pressure handoff after `compute_globals`
