TP1.8d reruns the authoritative safe baseline and the narrowed Ewald variant with downstream consumer tracing enabled.

The runner writes one CSV row per MD step and per traced stage:
- `after_longrange`
- `before_postprocess`
- `after_postprocess`
- `after_accumulate_energy`

The trace is aggregate, not per-pair. It is intended to answer whether the PME-vs-Ewald difference observed at `CpuPpLongRangeNonbondeds::calculate` survives into:
- final force-buffer accumulation
- direct virial transfer into `vir_force`
- final potential-energy bookkeeping
