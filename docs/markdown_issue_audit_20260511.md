# Markdown Issue Audit, 2026-05-11

## Scope

This audit reviewed Markdown files that can affect repository-facing claims and run guidance.

Counts from `rg --files -g '*.md'` / `Path.rglob('*.md')`:

- Total Markdown files in the repository: 510
- Project-owned Markdown files reviewed for claim consistency: 458
- Excluded from claim editing: vendored or generated documentation under `src/external/`, `docs/doxygen/`, `python_packaging/`, and `tests/physicalvalidation/`

The excluded files are not PCFF claim sources. Editing them would damage vendor/generated documentation without improving the project issue state.

## Resolved Issues Removed From Active Wording

The following were found as stale current-tense issues and corrected in active project docs:

- Gate I charged long-NPT density/volume conditioning was still described as pending in `docs/exact_respa_cpu_claim_boundary.md`.
  - Correct state: Gate I has a dated CPU-only exact-rRESPA PASS artifact.
  - Remaining boundary: Gate I does not establish charged transport readiness or LAMMPS-vs-GROMACS transport parity.
- `docs/gate_i_charged_long_npt_conditioning.md` still introduced Gate I as the "next gate".
  - Correct state: it is now a completed conditioning gate with PASS evidence.
- `docs/exact_respa_gpu_hybrid_track.md` still described GPU bonded PCFF/Class2 as completely outside the admitted path.
  - Correct state: the current PolyGen strict GPU lane uses `-nb gpu -pme cpu -bonded gpu -update cpu` and has full-run screening evidence, but this does not imply broad GPU bonded readiness.
- `docs/known_limitations.md` still read like a frozen `v1.0.0-rc1` limitation list.
  - Correct state: current active limitations are transport duration/statistics, cNE0 endpoint stability, LAMMPS-vs-GROMACS transport parity, broad chemistry scope, and strict GPU performance ceiling.
- `docs/release_checklist_rc1.md` still looked like an active release checklist.
  - Correct state: it is retained as historical withdrawn rc1 process context only.

## Current Active Issues

Current active issues are now centralized in [Current Active Issues](current_active_issues.md).

Short version:

1. PolyGen transport analysis is still 10 ns; charged TP0 requires at least 20 ns.
2. HTP-MD-style cNE0 is not stable over the current 10 ns CPU/GPU screening trajectory.
3. LAMMPS-vs-GROMACS charged transport parity is not closed.
4. GROMACS strict GPU production speed is below the 200 ns/day target under the PolyGen-equivalent r-RESPA settings.
5. Broad PCFF chemistry remains unsupported outside the explicitly validated subsets.

## Files Updated

- `README.md`
- `docs/current_active_issues.md`
- `docs/current_status_note.md`
- `docs/exact_respa_cpu_claim_boundary.md`
- `docs/exact_respa_gpu_hybrid_track.md`
- `docs/gate_i_charged_long_npt_conditioning.md`
- `docs/known_limitations.md`
- `docs/release_checklist_rc1.md`
- `docs/release_readiness_matrix.md`
- `docs/transport_scope_matrix.md`

Historical validation reports were not deleted. Their old failures remain useful evidence as long as they are not presented as current blockers.
