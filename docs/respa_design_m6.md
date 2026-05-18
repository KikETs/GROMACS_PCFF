# M6 Exact LAMMPS-style r-RESPA CPU Design

## Scope

M6 now contains two CPU-only exact-schedule paths for `mts-mode = lammps-respa`:

- the older `integrator = md` path
- a new dedicated `integrator = md-vv` path that does not reuse the leap-frog `md` update

The new `md-vv` route exists because the remaining M6 risk was no longer parser or nonbonded force ownership. It was kick/decomposition semantics versus LAMMPS `run_style respa` + `fix nve`.

Closed in this design:

- exact MDP schema for force-class ownership
- exact CPU scheduler ownership for bonded, pair14, k-space, and split real-space nonbonded work
- explicit inner / middle / outer real-space switching path
- CPU-only runtime dispatch and force-buffer plumbing for that schedule
- a dedicated `md-vv` exact base-step propagation path in `src/gromacs/mdrun/md.cpp`

Not in scope:

- GPU execution
- free-energy perturbation
- LJ-PME or shifted real-space modifiers
- a claim of full scientific parity without direct comparison against LAMMPS golden outputs
- any claim that the new `md-vv` path has already closed LAMMPS parity

## Architectural Entry Points

The exact path is split across five subsystems.

1. MDP parsing and input record state
- `src/gromacs/gmxpreprocess/readir.cpp`
- `src/gromacs/mdtypes/multipletimestepping.h`
- `src/gromacs/mdtypes/multipletimestepping.cpp`
- `src/gromacs/mdtypes/inputrec.cpp`
- `src/gromacs/fileio/tpxio.cpp`

2. Force-class ownership
- `src/gromacs/listed_forces/listed_forces.h`
- `src/gromacs/listed_forces/listed_forces.cpp`

3. Runtime workload selection
- `src/gromacs/taskassignment/decidesimulationworkload.cpp`
- `src/gromacs/mdtypes/simulation_workload.h`
- `src/gromacs/mdrun/runner.cpp`

4. Force buffers and virial ownership
- `src/gromacs/mdtypes/forcebuffers.h`
- `src/gromacs/mdtypes/forcebuffers.cpp`
- `src/gromacs/mdlib/forcerec.cpp`

5. Exact CPU nonbonded split execution
- `src/gromacs/mdlib/sim_util.cpp`

## Schedule Semantics

`mts-mode = lammps-respa` adds explicit ownership keys instead of reusing the legacy `mts-levelN-forces` lists.

Supported ownership controls:

- `mts-respa-bond-level`
- `mts-respa-angle-level`
- `mts-respa-dihedral-level`
- `mts-respa-improper-level`
- `mts-respa-pair14-level`
- `mts-respa-pair-level`
- `mts-respa-kspace-level`
- `mts-respa-inner-level`
- `mts-respa-middle-level`
- `mts-respa-outer-level`

Supported switching controls:

- `mts-respa-inner-off`
- `mts-respa-inner-on`
- `mts-respa-outer-on`
- `mts-respa-outer-off`

Current level count:

- `2 <= mts-levels <= 8` in the generic legacy MTS container
- the exact CPU validation and strict regression freeze only the intended 3-level `inner / middle / outer` schedule
- exact `mts-mode = lammps-respa` is now rejected unless `mts-levels = 3`

Legacy `mts-levelN-forces` input is rejected in exact mode on purpose. Mixing the two ownership models would make the schedule ambiguous.

As of the exact-runtime hardening follow-up, `mts-mode = lammps-respa` is treated as a parser alias only. After the input is converted to standalone exact-r-RESPA metadata, the runtime `t_inputrec` is canonicalized so that `useMts = false`, `mtsLevels` is empty, `mtsMode = legacy`, and `lammpsRespa` is cleared. Any exact-r-RESPA runtime path that still sees legacy GROMACS MTS state now fails a release assertion instead of silently reusing the old path.

## Runtime Model

### Base-step trace model

The current code now exposes an explicit `LammpsRespaBaseStepTrace` in `src/gromacs/mdtypes/multipletimestepping.*`.

This is not cosmetic. It freezes the base-step mapping that both CPU prototypes are supposed to follow when emulating LAMMPS recursive `run_style respa` inside the GROMACS per-base-step MD loop.

For the frozen 3-level `2 x 2` schedule, one outer period is represented as:

- base step 0: initial kicks `[2, 1, 0]`, force refresh `[0]`, final kicks `[0]`
- base step 1: initial kicks `[0]`, force refresh `[0, 1]`, final kicks `[0, 1]`
- base step 2: initial kicks `[1, 0]`, force refresh `[0]`, final kicks `[0]`
- base step 3: initial kicks `[0]`, force refresh `[0, 1, 2]`, final kicks `[0, 1, 2]`

This ordering matches the event sequence implied by LAMMPS `Respa::recurse()` plus `FixNVE::{initial,final}_integrate_respa()` when projected onto four fastest substeps per outer step.

The important consequence is negative as well as positive: once this trace is fixed and unit-tested, remaining drift is no longer plausibly explained by a simple start/end kick ordering bug.

### Dedicated `md-vv` propagation path

The new `integrator = md-vv` exact path does not call the standard leap-frog `md` update. Instead it:

- computes the current-step forces with the normal exact force ownership path
- computes current-step kinetic observables explicitly for output
- applies the LAMMPS base-step initial half-kicks from the active per-level force buffers
- drifts positions by the fastest base step
- performs a look-ahead force evaluation for `step + 1`
- applies the final half-kicks implied by the next-step refresh set

This is intentionally narrow. The current implementation only supports CPU-only NVE with:

- `tcoupl = no`
- `pcoupl = no`
- `comm-mode = none`
- no constraints
- no DD
- no GPU
- no virtual sites
- no replica exchange
- no special-force modules

That restriction is deliberate. Accepting broader settings here would hide semantics the code does not yet implement.

### Force ownership

The exact mode uses dedicated force groups for:

- `Bond`
- `Angle`
- `Dihedral`
- `Improper`
- `Pair`
- `LongrangeNonbonded`
- `NonbondedInner`
- `NonbondedMiddle`
- `NonbondedOuter`

Level 0 keeps the fast-force complement, but explicit level-0 exact assignments are preserved. This matters for LAMMPS-style schedules where bonds and inner nonbonded work often live at the innermost level.

### Force buffers

The force-buffer layout was generalized from a single slow-force buffer to:

- base force buffer
- one buffer per explicit slow level
- one combined MTS force buffer

This is required because exact `r-RESPA` needs separate ownership and later weighted combination for more than one slow level.

One subtle point turned out to matter more than the pair-splitting details: on slow steps, `force()` is the physical total force after MTS combination, not the fast level-0 force. The exact kick path in [md.cpp](../src/gromacs/mdrun/md.cpp) therefore now reconstructs the level-0 kick force as `F_total - sum(F_slow_levels)` before applying the explicit slow-level kicks. Without that reconstruction, slow-level impulses are counted twice.

### Exact split nonbonded execution

`computeLammpsRespaNonbondedCpu(...)` in `src/gromacs/mdlib/sim_util.cpp` is the exact CPU helper for split real-space work.

It:

- forces use of the plain pairlist
- evaluates direct-space interactions on CPU only
- distributes each pair contribution into `inner`, `middle`, or `outer`
- adds the long-range Coulomb correction only to the outer contribution
- accumulates outer direct virial using the full physical direct contribution needed for virial bookkeeping

The switching logic was modeled against the local LAMMPS reference source:

- `../lammps/src/CLASS2/pair_lj_class2_coul_long.cpp`

Basis:

- `compute_inner()`
- `compute_middle()`
- `compute_outer()`

The exact parity harness also needs a real Verlet buffer. LAMMPS `units real` defaults the neighbor skin to `2.0 Angstrom` in `../lammps/src/update.cpp`, but the frozen `small_oligomer` box is only `2.0 nm` wide, so a literal `rlist = 1.1 nm` would exceed half the shortest box vector and be rejected by GROMACS. The M6 fixtures therefore now use the largest valid buffered setting that still fits the frozen box, `rlist = 0.99 nm` for `rcoulomb = rvdw = 0.9 nm`. The earlier `rlist = cutoff` setup with `nstlist > 1` was not a meaningful scheduler comparison, because it let the plain pairlist go stale over the outer-step interval.

## Explicit Constraints

The exact CPU path currently rejects:

- `vdw-type` other than `cut-off`
- `vdw-modifier` other than `none`
- `coulombtype` other than `PME` or `Ewald`
- `coulomb-modifier` other than `none`
- `mts-respa-outer-off > min(rcoulomb, rvdw)` when pair splitting is active
- GPU nonbonded / PME / bonded / update use
- free-energy perturbation

These are not arbitrary restrictions. The exact helper only implements the semantics above. Accepting more options would silently lie about the physics.

## What Is Exact vs What Is Still Unresolved

Exact in the current codebase:

- exact ownership parsing
- exact per-level scheduler activation
- exact CPU force buffering and combination
- exact CPU direct-space split path for inner / middle / outer real-space work
- exact-vs-unsplit same-coordinate force parity for the frozen `small_oligomer` and `small_salt_polymer_box` fixtures, measured with `tools/pcff_respa_parity/force_compare.py`

Not yet closed:

- broader virial coverage beyond the frozen step-0 tensor in the two M6 fixtures
- restart / checkpoint parity away from outer-force boundaries
- any GPU path

The unresolved point is important. GROMACS unsplit PME energy bookkeeping is not the same validation target as LAMMPS-style split bookkeeping. Internal GROMACS force/trajectory agreement is useful, but it is not sufficient evidence of full LAMMPS parity.

The current M6 measurement now confirms a narrower and more useful point. The dedicated `md-vv` path plus level-0 force reconstruction keeps same-coordinate force parity at about `1e-3 kJ/mol/nm` and reduces the frozen 3-level LAMMPS NVE deltas to the `1e-3` to `1e-2 kcal/mol` range. That is strong enough to serve as the CPU reference for the next GPU stage, but it is not evidence that every exact-mode schedule is equally validated. That is why exact `2-level` mode is rejected instead of being left half-supported.
