# M10.5 — Charged Diagnostic Workflow Template

This file is retained as a diagnostic example only.

It is **not** a present-tense charged-readiness or production-readiness claim.

Current source of truth:

- [Current Status Note](current_status_note.md)
- [Machine-Readable Support Matrix](../tests/reference_results/pcff_ion_narrow_claim/support_matrix.json)

## 1. System Requirements
- Frozen charged diagnostics only.
- Use this template only when your system already falls inside the narrow charged scope described in `Current Status Note`.
- Do **not** treat this template as approval for arbitrary charged or salt-containing PCFF chemistry.
- **Warning:** PME/PPPM virial differences may lead to different equilibrium densities than LAMMPS.

## 2. Mandatory Monitoring Rules

### 2.1 Extended Equilibration
- **Duration:** > 500 ps recommended.
- **Rule:** Do not interpret a run as ensemble-ready just because it stays numerically stable for a short window.

### 2.2 Electrostatics Accuracy
- **Setting:** `ewald-rtol = 1e-5` (GROMACS) / `kspace_style pppm 1e-5` (LAMMPS).
- **Note:** Higher accuracy is required to minimize reciprocal-space stress artifacts.

## 3. Recommended MDP Settings

### 3.1 Equilibration (`equil_charged.mdp`)
```ini
integrator  = md
dt          = 0.001
tcoupl      = v-rescale
tc-grps     = System
tau-t       = 0.5 ; Increased for charged stability
ref-t       = 300
pcoupl      = Berendsen
pcoupltype  = isotropic
tau-p       = 5.0 ; Slower coupling to manage ionic stress
compressibility = 4.5e-5
ref-p       = 1.0
```

## 4. Diagnostic Use Only

These checks are diagnostic triage checks, not production-entry criteria:

- [ ] Potential energy is reviewed against the frozen reference with explicit caveats.
- [ ] Density trace is inspected manually for drift rather than assumed converged.
- [ ] Reciprocal-space warnings in `grompp` are reviewed and recorded.
- [ ] Result is compared against [TP1 recovery summary](../tests/reference_results/tp1_charged_recovery/dense_salt_polymer/recovery_summary.json) before making any stronger claim.

## 5. User Responsibility
Users of charged diagnostics are responsible for performing a **manual sanity check** on density, drift, and long-horizon stability.

If density deviates materially from LAMMPS or experiment, or if stability is not demonstrated over the intended horizon, the result must remain diagnostic only.
