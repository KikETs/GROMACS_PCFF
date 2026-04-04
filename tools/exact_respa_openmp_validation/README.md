# `exact_respa_openmp_validation`

This toolchain exists for one job only:

- turn host-local exact `r-RESPA` CPU OpenMP evidence into structured reports
- refuse a broader desktop-class CPU OpenMP claim until enough topology-diverse reports exist

It does not replace the in-tree exact OpenMP regression tests.
It orchestrates them, adds host-topology evidence, and aggregates multiple host reports into a claim decision.

## Why this exists

The exact standalone `mts-mode = lammps-respa` path already has strong in-tree mechanics checks:

- exact parity against the `ntomp=1` oracle
- per-level force-dump parity
- restart continuity
- affinity-on exact parity for `-pin auto`, `-pin on`, and `-pin inherit`
- TSAN exact subset support where a TSAN build exists

That is still not enough for a broader desktop-class CPU OpenMP claim.

The remaining gaps are:

- host diversity
- topology-aware production rules

## Files

- `collect_host_report.py`
  - runs the exact release suite
  - optionally runs the exact TSAN suite
  - collects CPU topology metadata
  - runs a locality benchmark under `-pin inherit`
  - emits one host-local JSON report
- `aggregate_reports.py`
  - loads multiple host reports
  - checks the required topology classes
  - checks exact mechanical evidence on every host
  - separates a broader desktop/workstation mechanics claim from a shared production-envelope claim
  - refuses a shared production-envelope claim unless the reports share a common locality-based production rule

## Required host classes

`aggregate_reports.py` will not pass unless the input reports cover all three classes:

- `low-core-workstation`
- `mid-core-hybrid-desktop`
- `numa-or-chiplet`

Server CPU support is out of scope for this aggregate. One more host is not enough.

## Example: collect one host report

```bash
python3 tools/exact_respa_openmp_validation/collect_host_report.py \
  --out /tmp/exact-openmp-9900x.json \
  --topology-class numa-or-chiplet \
  --release-binary build-worktree/bin/mdrun-non-integrator-test \
  --tsan-binary build-clang-tsan-o2/bin/mdrun-non-integrator-test \
  --gmx-bin build-worktree/bin/gmx \
  --tsan-env LD_LIBRARY_PATH=/path/to/clang/lib \
  --tsan-env TSAN_OPTIONS='halt_on_error=1 history_size=7 second_deadlock_stack=1 ignore_noninstrumented_modules=1 external_symbolizer_path=/path/to/llvm-symbolizer'
```

Important:

- `--topology-class` is a human-audited label. The tool records raw topology evidence, but it does not guess your claim class for you.
- the host-local rule that `collect_host_report.py` emits is only a candidate
- it is not a broader desktop/workstation support claim by itself

## Example: aggregate multiple host reports

```bash
python3 tools/exact_respa_openmp_validation/aggregate_reports.py \
  --out /tmp/exact-openmp-aggregate.json \
  /tmp/exact-openmp-lowcore.json \
  /tmp/exact-openmp-midcore.json \
  /tmp/exact-openmp-numa.json
```

The aggregator exits nonzero when the shared production-envelope claim is not earned.

This is stricter than the mechanics claim. A summary can still report that a broader
desktop/workstation mechanics claim is allowed while the process exits nonzero because
the production envelope remains host-local.

The aggregator fails the production-envelope step when:

- any required topology class is missing
- any host lacks release exact-suite evidence
- any host lacks TSAN exact evidence unless `--allow-missing-tsan` is set
- any host lacks locality benchmark evidence
- the collected hosts do not share one common locality-based production rule

## Output semantics

Each host report separates:

- exact mechanical evidence
- locality benchmark evidence
- a host-local rule candidate

The aggregate summary separates:

- broader desktop/workstation mechanics claim
- production rule
- correctness-only rule
- unsupported or unproven region

If the aggregate step fails, the broader desktop/workstation production-envelope claim is not earned.
That does not automatically mean the broader mechanics claim failed; check the summary fields.
