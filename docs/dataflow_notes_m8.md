# M8 Dataflow Notes

## Objective

M8 does not introduce new physics. It freezes the supported GPU-resident-compatible dataflow for the current PCFF path and documents where host-device transfers are still unavoidable.

## Key Runtime Rules

### 1. Buffer ops are tied to nonbonded GPU offload

The runtime only allows GPU X/F buffer ops when:

- nonbonded is on the GPU
- MTS is off
- buffer ops are not explicitly disabled

Basis:

- [decidesimulationworkload.cpp](../src/gromacs/taskassignment/decidesimulationworkload.cpp#L160)

This matters for PCFF because the new `9-6` path participates in the same nonbonded buffer-op machinery as the standard GPU short-range path. There is no separate PCFF-specific residency path.

### 2. GPU update requires GPU buffer ops

If GPU update or direct GPU communication is enabled, the runtime asserts that X/F buffer ops are available.

Basis:

- [decidesimulationworkload.cpp](../src/gromacs/taskassignment/decidesimulationworkload.cpp#L166)

This is why M8 validation had to include `-update gpu` explicitly.

### 3. Virial steps break the resident force-reduction path

Per-step force buffer ops are disabled when virial is computed.

Basis:

- [decidesimulationworkload.cpp](../src/gromacs/taskassignment/decidesimulationworkload.cpp#L301)

And GPU PME force reduction depends on GPU F buffer ops being active:

- [decidesimulationworkload.cpp](../src/gromacs/taskassignment/decidesimulationworkload.cpp#L304)

Consequence:

- if `nstcalcenergy = 1`, the test does not exercise the intended GPU-resident force path
- M8 therefore uses sparse energy/virial cadence

### 4. CPU bonded work still forces coordinate staging to host

When GPU update is enabled and there is CPU local force work, coordinates are copied from GPU to CPU on non-search steps.

Basis:

- [sim_util.cpp](../src/gromacs/mdlib/sim_util.cpp#L2174)

This is the main reason M8 does **not** claim full no-copy execution.

### 5. Coordinates otherwise stay on device across non-search steps

When GPU update is active and the step is not a search step, the runtime uses the existing device-resident update completion event instead of scheduling a fresh H2D coordinate copy.

Basis:

- [sim_util.cpp](../src/gromacs/mdlib/sim_util.cpp#L2187)

### 6. GPU X buffer ops consume device coordinates directly

On non-search steps with GPU X buffer ops enabled, nonbonded coordinate conversion happens from the device coordinate buffer.

Basis:

- [sim_util.cpp](../src/gromacs/mdlib/sim_util.cpp#L2267)

## M8 Validation Configuration

The M8 resident-style test uses:

- `integrator = md`
- `-nb gpu`
- `-pme gpu`
- `-pmefft gpu`
- `-bonded cpu`
- `-update gpu`
- single rank
- no MTS

The `md` integrator is not optional here. GPU update currently rejects anything else.

Basis:

- [decidegpuusage.cpp](../src/gromacs/taskassignment/decidegpuusage.cpp#L779)

## Transfer Decisions

### Transfers intentionally preserved

- GPU -> CPU coordinate copies needed for CPU bonded work
- final output writes to host-visible files
- virial-step force reductions that legitimately fall back to the CPU path

### Transfers intentionally avoided in the M8 validation path

- per-step virial/energy output that would disable GPU F buffer ops every step
- per-step trajectory force output
- CPU PME fallback

## Implication For Future Work

M8 means the current PCFF nonbonded CUDA path is now safe to use under the existing GPU update/PME/buffer-op model.

It does **not** mean:

- bonded CPU work has stopped forcing host interaction
- full GPU-resident execution is solved

That remaining gap belongs to optional bonded GPU work later, not to M8.
