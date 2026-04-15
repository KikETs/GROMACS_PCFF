# CPU Exact r-RESPA Claim Boundary

This note freezes the public CPU-only exact `r-RESPA` boundary.

It is intentionally narrower than the broader bridge status in [Current Status Note](current_status_note.md).
It is also intentionally narrower than any transport-facing discussion.

## Narrow Claim

Use this sentence for the current CPU exact `r-RESPA` state:

> Current evidence supports a narrow CPU exact-r-RESPA claim: for single-rank, CPU-only, standalone exact r-RESPA, exact event order, restart continuity, and small-fixture mechanical behavior are frozen on the Gate A oracle, and a bounded exact CPU OpenMP mechanics claim is allowed across the tested low-core, hybrid-desktop, and chiplet workstation classes only for the audited ntomp>1 buckets `ntompSmall` and `ntompCeiling` under `-pin auto`, `-pin on`, and `-pin inherit`. That OpenMP claim is discrete, not a continuous ntomp envelope: ntomp=1 remains the oracle baseline, host-local throughput benchmarks do not broaden support, and intermediate or larger ntomp counts remain unsupported. Exact long-run ensemble evidence is still narrow: Gate G passes a 40 ps NVT `small_oligomer` check and a 40 ps NPT `small_salt_polymer_box` check, but those are small-fixture gates only. Gate H reuses the exact NVT path on larger transport scaffolds, yet transport production remains NO-GO, and charged medium-scale long-NPT density conditioning is still missing. This claim does not imply conductivity-production readiness, LAMMPS-vs-GROMACS transport parity, generic medium-scale NPT convergence, server-CPU coverage, MPI support, or GPU coexistence support.

## What Is Closed

- Gate A freezes CPU-only exact event order, per-level force totals, total-force ledgers, and restart continuity on `small_oligomer` and `small_salt_polymer_box`.
- The checked-in exact OpenMP host inventory supports a bounded desktop/workstation CPU mechanics claim on the tested low-core, hybrid-desktop, and chiplet classes.
  That claim is limited to the audited ntomp>1 buckets `ntompSmall` and `ntompCeiling` under `-pin auto`, `-pin on`, and `-pin inherit`.
- No correctness-only OpenMP envelope is currently closed beyond those discrete buckets.
- Gate G closes a narrow exact ensemble boundary on small fixtures:
  `small_oligomer` NVT `PASS`
  `small_salt_polymer_box` NPT `PASS`

## CPU OpenMP Envelope

- Supported:
  On the tested low-core-workstation, mid-core-hybrid-desktop, and numa-or-chiplet desktop/workstation hosts, single-rank CPU-only standalone exact `r-RESPA` passes ntomp>1 oracle parity and restart parity for the audited `ntompSmall` and `ntompCeiling` buckets under `-pin auto`, `-pin on`, and `-pin inherit`.
- Correctness-only:
  None. No checked-in artifact extends support from those audited buckets to intermediate ntomp values, larger ntomp values, or benchmark-only runs.
- Weak or unsupported:
  `ntomp=1` is the oracle baseline only; benchmark-only `-pin inherit` throughput scans remain host-local observations only; intermediate counts, counts above the audited ceiling, server CPUs, MPI, GPU coexistence, and non-audited affinity/runtime shapes remain unsupported or unproven.

Primary machine-readable artifacts:

- [CPU exact claim summary](../tests/reference_results/cpu_exact_respa_claim/cpu_exact_claim_summary.json)
- [CPU exact support matrix](../tests/reference_results/cpu_exact_respa_claim/support_matrix.json)
- [CPU exact boundary and blockers](../tests/reference_results/cpu_exact_respa_claim/boundary_and_blockers.json)
- [CPU exact mechanical evidence index](../tests/reference_results/cpu_exact_respa_claim/mechanical_evidence_index.json)
- [Exact OpenMP validation summary](../tests/reference_results/cpu_exact_respa_claim/openmp_validation_summary.json)
- [Gate I charged long-NPT contract](../tests/reference_results/gate_i_charged_long_npt_conditioning/gate_i_contract.json)
- [Gate I charged long-NPT manifest](../tests/reference_results/gate_i_charged_long_npt_conditioning/gate_i_manifest.json)

## What Is Not Closed

- Gate G small-fixture NPT is not a medium-scale density/volume convergence claim.
- Gate H large/medium is still `FAIL` / `NO-GO`.
- Gate H reuses the exact NVT path, but that does not create transport readiness.
- No current public artifact proves exact-r-RESPA charged medium-scale long-NPT density/volume convergence.

## Mechanical vs Transport

Mechanical parity and transport readiness must stay separate.

- Mechanical exactness:
  Gate A oracle plus the checked-in CPU OpenMP inventory show that the exact CPU path is mechanically controlled on the frozen scope, but only for the audited discrete OpenMP buckets.
- Ensemble exactness:
  Gate G adds only small-fixture long-run ensemble evidence.
- Transport readiness:
  Gate H explicitly remains `NO-GO`, and charged transport still lacks long-NPT density conditioning on the intended scaffold.

Do not promote short-window mechanical closure, small-fixture NPT success, or NVT-only scaffold reuse into conductivity-production wording.

## M10 Ambiguity

The legacy M10 medium-scale NVT/NPT diagnostics are still useful, but they are not exact `r-RESPA` evidence.

Why:

- [run_m10_2.py](/home/kiket/Desktop/test/GROMACS_PCFF/tools/run_m10_2_ensemble_gate/run_m10_2.py)
  `get_mdp_nvt()` and `get_mdp_npt()` write plain `integrator = md` inputs.
- [run_m10_2_1.py](/home/kiket/Desktop/test/GROMACS_PCFF/tools/run_m10_2_1_convergence_gate/run_m10_2_1.py)
  `get_mdp_npt()` also writes plain `integrator = md`.

Those files can still support the blocker narrative:

- [M10.2 gate decision](../tests/reference_results/m10_2_ensemble_gate/m10_2_gate_decision.json): medium-scale fixed-volume NVT `pass`, short NPT `partial`
- [M10.2.1 gate decision](../tests/reference_results/m10_2_1_convergence_gate/m10_2_1_gate_decision.json): longer NPT convergence still not sufficient

They must not be cited as exact `r-RESPA` proof.

## Sole Immediate Blocker

The sole immediate blocker for moving from the current exact CPU closure toward charged transport-valid use is missing exact-r-RESPA charged medium-scale long-NPT density/volume convergence evidence.

That blocker is explicit in:

- [Gate H large/medium manifest](../tests/reference_results/gate_h_transport_validation_large_medium/gate_h_manifest.json)
- [Gate H small manifest](../tests/reference_results/gate_h_transport_validation/gate_h_manifest.json)
- [CPU exact boundary and blockers](../tests/reference_results/cpu_exact_respa_claim/boundary_and_blockers.json)

Temperature agreement, short NPT stability, or NVT-only transport-facing observables do not close that blocker.

The repository now freezes the blocker as a concrete Gate I contract, but the checked-in Gate I manifest is still pending execution.
That means the blocker is explicit and machine-readable, not resolved.

## Next Gate

Run Gate I: a predeclared single-rank CPU exact-r-RESPA long-NPT conditioning gate on the charged large/medium scaffold.

Primary Gate I public artifacts:

- [Gate I charged long-NPT note](gate_i_charged_long_npt_conditioning.md)
- [Gate I contract JSON](../tests/reference_results/gate_i_charged_long_npt_conditioning/gate_i_contract.json)
- [Gate I manifest JSON](../tests/reference_results/gate_i_charged_long_npt_conditioning/gate_i_manifest.json)

That gate must freeze:

- density block-drift thresholds
- volume block-drift thresholds
- cross-replica consistency thresholds
- the conditioned-state handoff that would feed any later TP0-scale production campaign

Even if that passes, it still does not automatically grant conductivity-production readiness.
It only closes the density/volume conditioning blocker that currently prevents an honest transport-valid handoff.
