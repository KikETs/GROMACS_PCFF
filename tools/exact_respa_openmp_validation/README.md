# `exact_respa_openmp_validation`

This toolchain exists for one job only:

- turn host-local exact `r-RESPA` CPU OpenMP evidence into structured reports
- refuse a broader desktop-class CPU OpenMP claim until enough topology-diverse reports exist

It does not replace the in-tree exact OpenMP regression tests.
It orchestrates them, adds host-topology evidence, and aggregates multiple host reports into a claim decision.
It now also maintains a manifest-driven host-report backend so stale JSONs and mixed-era filenames cannot silently re-enter the active evidence set.

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
- multi-host TSAN-backed race evidence
- recurring automation beyond one-off local replay

## Files

- `collect_host_report.py`
  - runs the exact release suite
  - optionally runs the exact TSAN suite
  - collects CPU topology metadata
  - runs a locality benchmark under `-pin inherit`
  - emits one host-local JSON report, including a host-local production-envelope candidate
  - enforces canonical CPU-distinguishing report filenames
- `aggregate_reports.py`
  - loads multiple host reports
  - rejects stale schema versions and non-canonical report filenames
  - checks the required topology classes
  - checks exact mechanical evidence on every host
  - re-derives the host-local production candidate from raw benchmark data for every report
  - separates a broader desktop/workstation mechanics claim from a shared production-envelope claim
  - refuses a shared production-envelope claim unless the reports share a common locality-based production rule
  - tracks whether multi-host TSAN-backed race evidence and recurring automation are actually present
- `run_host_profile.py`
  - loads one host profile from `report_set_manifest.json`
  - checks that the current CPU model matches the selected profile
  - runs `collect_host_report.py` with the manifest-declared paths and TSAN policy
- `run_backend_cycle.py`
  - runs one backend recollection cycle for selected manifest profiles
  - then runs `validate_report_set.py`
  - is the preferred recurring entrypoint for self-hosted runners or local schedulers
- `validate_report_set.py`
  - validates the active/pending host profile set from `report_set_manifest.json`
  - writes `validation-summary.json`, `aggregate-allow-missing-tsan.json`, and `stale-check.json`
  - distinguishes active evidence from pending external recollection work
- `validate_checked_in_reports.py`
  - validates the checked-in host-report inventory
  - is used by the GitHub Actions report-guard workflow
  - does not pretend that CI has recollected every host report

## Required host classes

`aggregate_reports.py` will not pass unless the input reports cover all three classes:

- `low-core-workstation`
- `mid-core-hybrid-desktop`
- `numa-or-chiplet`

Server CPU support is out of scope for this aggregate. One more host is not enough.

## Backend files

- `tests/reference_results/exact_respa_openmp_validation/report_set_manifest.json`
  - declares active and pending host profiles
  - records target filenames, host labels, topology classes, and TSAN expectations
  - records the intended recurring backend mode and runner labels where they exist
- `tests/reference_results/exact_respa_openmp_validation/host_reports/`
  - active fresh reports only
- `tests/reference_results/exact_respa_openmp_validation/stale_host_reports/`
  - archived or removed mixed-era reports only

## Canonical host report filenames

Checked-in host reports must use auditable CPU-distinguishing names:

- `amd_ryzen_7_5800x_low_core_workstation.json`
- `intel_i9_12900k_mid_core_hybrid_desktop.json`
- `amd_ryzen_9_9900x_numa_or_chiplet.json`

If the same CPU model is used on multiple hosts and host-level distinction matters, append a host suffix:

- `amd_ryzen_7_5800x_low_core_workstation_lab.json`

`collect_host_report.py` now rejects non-canonical output filenames.

## Example: collect one manifest-declared host profile

```bash
python3 tools/exact_respa_openmp_validation/run_host_profile.py \
  --profile-id amd_ryzen_9_9900x_numa_or_chiplet
```

That entrypoint is the preferred backend path. It keeps filename policy, host identity, and TSAN policy aligned with the manifest.
It refuses to stamp `ci` or `scheduled` collection modes unless the run is attested by
`EXACT_OPENMP_RECURRING_BACKEND=<mode>` or a real GitHub Actions CI context.

## Example: run one recurring backend cycle

```bash
python3 tools/exact_respa_openmp_validation/run_backend_cycle.py \
  --profile-id amd_ryzen_9_9900x_numa_or_chiplet \
  --collection-mode-override scheduled \
  --out-dir /tmp/exact-openmp-backend-9900x
```

Use `scheduled` for local cron/systemd timers and `ci` for self-hosted CI runners.
Do not mark a report as `ci` or `scheduled` if you only ran it by hand once.
The recurring entrypoints validate in strict mode by default. Use `--no-strict` only for
diagnostics when you explicitly do not want the broader-claim gate to fail the run.

## Example: collect one host report directly

```bash
python3 tools/exact_respa_openmp_validation/collect_host_report.py \
  --out tests/reference_results/exact_respa_openmp_validation/host_reports/amd_ryzen_9_9900x_numa_or_chiplet.json \
  --topology-class numa-or-chiplet \
  --release-binary build-worktree/bin/mdrun-non-integrator-test \
  --tsan-binary build-clang-tsan-o2/bin/mdrun-non-integrator-test \
  --gmx-bin build-worktree/bin/gmx \
  --tsan-env LD_LIBRARY_PATH=/path/to/clang/lib \
  --tsan-env TSAN_OPTIONS='halt_on_error=1 history_size=7 second_deadlock_stack=1 ignore_noninstrumented_modules=1 external_symbolizer_path=/path/to/llvm-symbolizer'
```

Important:

- `--topology-class` is a human-audited label. The tool records raw topology evidence, but it does not guess your claim class for you.
- `schema_version >= 3` reports are required for aggregation.
- the host-local rule that `collect_host_report.py` emits is only a candidate
- it is not a broader desktop/workstation support claim by itself
- the current production candidate is plateau-based: it tracks the highest tested thread count within one L3/CCD-equivalent locality group that remains within 95% of the best exact rate observed in that locality group
- if infra semantics, TSAN workflow semantics, or report naming policy changes, previously collected reports are stale and must be regenerated
- direct collection is allowed, but the final active inventory should still be managed through `report_set_manifest.json`
- recurring `ci` or `scheduled` reports now record an attestation block; reports without that attestation are stale for strict aggregation

## Example: validate the active/pending report backend

```bash
python3 tools/exact_respa_openmp_validation/validate_report_set.py \
  --manifest tests/reference_results/exact_respa_openmp_validation/report_set_manifest.json \
  --out-dir /tmp/exact-openmp-validation-v2
```

This produces:

- `validation-summary.json`
- `aggregate-allow-missing-tsan.json`
- `aggregate-strict.json`
- `stale-check.json`

Use `--strict` when you need the full broader desktop/workstation claim gate:

```bash
python3 tools/exact_respa_openmp_validation/validate_report_set.py \
  --manifest tests/reference_results/exact_respa_openmp_validation/report_set_manifest.json \
  --out-dir /tmp/exact-openmp-validation-v2 \
  --strict
```

In strict mode, the command exits nonzero unless the active inventory still defends:

- multi-host TSAN-backed evidence
- recurring automation evidence
- the broader desktop/workstation mechanics claim
- the shared plateau-knee production-envelope rule

## Example: aggregate multiple host reports

```bash
python3 tools/exact_respa_openmp_validation/aggregate_reports.py \
  --out /tmp/exact-openmp-aggregate.json \
  tests/reference_results/exact_respa_openmp_validation/host_reports/amd_ryzen_7_5800x_low_core_workstation.json \
  tests/reference_results/exact_respa_openmp_validation/host_reports/intel_i9_12900k_mid_core_hybrid_desktop.json \
  tests/reference_results/exact_respa_openmp_validation/host_reports/amd_ryzen_9_9900x_numa_or_chiplet.json
```

The aggregator exits nonzero when the full broader claim is not earned.

This is stricter than the mechanics and production-envelope summaries. A summary can still
report that a broader desktop/workstation mechanics claim is allowed, or even that a
shared production-envelope rule exists across the tested hosts, while the process exits
nonzero because multi-host TSAN-backed race evidence or recurring automation is still incomplete.

The aggregator fails the production-envelope step when:

- any required topology class is missing
- any checked-in report is stale or uses a non-canonical filename
- any host lacks release exact-suite evidence
- any host lacks TSAN exact evidence unless `--allow-missing-tsan` is set
- any host lacks locality benchmark evidence
- the collected hosts do not share one common locality-based production rule
- any host report lacks the recurring infra metadata needed for G1/G4 accounting

## Checked-in report guard

The repo includes `.github/workflows/exact_openmp_report_guard.yml`.

That workflow:

- validates the schema and filename policy of checked-in host reports
- regenerates the strict aggregate summary from checked-in reports
- fails if stale or mixed-era report files remain in the tracked inventory

It is wired to `validate_report_set.py --strict`, so a green report guard means the
checked-in active inventory still passes the broader desktop/workstation claim gate.
It does **not** prove that every external host has been recollected under GitHub-hosted CI.
It is a repo-side strict gate against stale data, inventory drift, and broadened claims
that outrun the checked-in evidence.

## Recurring backend

The repo now also includes `.github/workflows/exact_openmp_collect_9900x.yml`.

That workflow is intentionally narrow:

- it targets only the active 9900X manifest profile
- it expects a self-hosted runner with labels `self-hosted`, `linux`, `x64`, `exact-openmp-9900x`
- it requires prebuilt local binaries at the manifest paths
- it runs `run_recurring_backend.py` with `--collection-mode ci --strict`

This is the recurring backend for the current 9900X host class.
It does **not** automatically solve recurring backend ownership for 5800X or 12900K.
Those hosts still need either:

- their own self-hosted runner integration, or
- a local scheduled backend that runs `run_backend_cycle.py` with `--collection-mode-override scheduled`

The recurring backend entrypoints now attest the collection mode. A hand-run
`run_host_profile.py --collection-mode-override scheduled` is rejected unless
`EXACT_OPENMP_RECURRING_BACKEND=scheduled` is present. The same rule applies to `ci`.

## Output semantics

Each host report separates:

- exact mechanical evidence
- locality benchmark evidence
- a host-local rule candidate

The aggregate summary separates:

- broader desktop/workstation mechanics claim
- production rule
- infrastructure readiness for G1/G4
- correctness-only rule
- unsupported or unproven region

If the aggregate step fails, the broader desktop/workstation production-envelope claim is not earned.
That does not automatically mean the broader mechanics claim failed; check the summary fields.
If `validate_report_set.py` reports only pending external host recollection work, that is a backend-state issue, not automatically a stale-data failure.
If `validate_report_set.py --strict` fails, the repo should be treated as not currently
defending the broader desktop/workstation claim, even if relaxed summaries still contain
useful host-local evidence.

Any broader wording must stay inside these boundaries:

- single-rank only
- CPU-only only
- standalone exact `r-RESPA` only
- desktop/workstation CPUs only
- server CPUs unvalidated
- MPI coexistence out of scope
- GPU coexistence out of scope
- multi-host TSAN-backed race evidence may still be incomplete when `--allow-missing-tsan` is used
- recurring automation may still be incomplete when reports are collected in `manual` mode
