# Current Active Issues

This file is the current issue boundary for the PolyGen exact r-RESPA PCFF work.

It separates unresolved issues from historical failures that have already been superseded.

## Resolved And No Longer Active Blockers

The following should not be described as current blockers:

- TP1 exact `dense_salt_polymer` thermal runaway.
  - Current state: resolved only for the corrected 5 ns exact-system NPT rerun.
  - Remaining caveat: the final box fails endpoint continuation safety.
- Gate I CPU-only exact-rRESPA charged long-NPT density/volume conditioning.
  - Current state: dated Gate I PASS evidence exists for `gate_h_dense_salt_polymer_2x2x2`.
  - Remaining caveat: this is conditioning evidence, not charged transport readiness.
- Old locked-scope LJ-SR / LJ-(SR) issue wording from the M10 handoff.
  - Current state: not an active blocker for the current PolyGen exact-rRESPA path.

## Active Issues

### 1. PolyGen Transport Duration Is Still 10 ns

Current evidence:

- `docs/polygen_cpu_gpu_transport_screening_20260510.md`
- CPU/GPU GROMACS production chunks: 50 chunks, 10 ns total.
- Local charged transport protocol in `docs/transport_protocol_freeze.md` requires at least 20 ns.

Required next action:

- Extend LAMMPS, GROMACS CPU, and GROMACS GPU production from 10 ns to 20 ns without overwriting chunks 1-50.
- Re-run transport analysis on the 20 ns trajectories.

### 2. HTP-MD-Style cNE0 Is Not Stable Over 10 ns

Current evidence:

- CPU `cNE0_htp_sigma_S_cm`: `0.00215901 S/cm`
- GPU `cNE0_htp_sigma_S_cm`: `0.00610284 S/cm`
- CPU/GPU relative delta: `182.67%`

Required next action:

- Recompute cNE0 after the 20 ns extension.
- Add block analysis before treating cNE0 as more than diagnostic.

### 3. LAMMPS-vs-GROMACS Charged Transport Parity Is Not Closed

Current evidence:

- Latest transport analysis is GROMACS CPU/GPU only.
- The LAMMPS production dump was used as topology metadata for the analyzer, not as a transport reference in the 2026-05-10 report.

Required next action:

- Run or reuse a transport-ready 20 ns LAMMPS reference trajectory.
- Compare LAMMPS vs GROMACS CPU and GPU for NE, cNE0, MSD-derived diffusion, conductivity, and transference.

### 4. Strict GPU Production Speed Is Below The 200 ns/day Target

Current evidence:

- Final strict GPU full-run mean production speed: `147.315 ns/day`.
- Strict mode: `-nb gpu -pme cpu -bonded gpu -update cpu`, `ntomp=12`.

Required next action:

- Do not change physics to chase speed.
- Profile remaining strict-rRESPA synchronization, PME-on-CPU wait, listed-force residency, and force-copy overhead.

### 5. Broad PCFF Chemistry Remains Unsupported

Current evidence:

- Current status remains bounded to explicitly validated PT8/M11 subsets.
- CSV-scope coverage is still not a broad PCFF chemistry pass.

Required next action:

- Treat new chemistry as out-of-scope until it has typing/export, mechanics, dense ensemble, and transport-facing evidence.

## Not An Issue By Itself

- CPU/GPU pressure mean differences in NVT production are diagnostics, not a standalone failure, because instantaneous pressure fluctuations are large and the fixed-volume density/volume match is the stronger transport-entry sanity check.
- CPU double vs GPU mixed trajectories are not expected to be bitwise identical.
