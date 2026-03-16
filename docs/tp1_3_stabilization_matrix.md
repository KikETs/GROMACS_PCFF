# TP1.3 Stabilization Trial Matrix

The following matrix defines the diagnostic trials for identifying the root cause of the `dense_salt_polymer` (270-atom Na/Cl) thermal runaway.

## Target System
- **ID:** `dense_salt_polymer`
- **Identity:** 270 atoms, Na/Cl salt in polymer matrix.

## Trial Matrix Results

| Trial ID | Timestep (fs) | Ensemble | Thermostat | Barostat | Coulomb | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRL-0** | 1.0 | NPT | V-rescale (0.5) | Berendsen | PME | **RUNAWAY** (738K) |
| **TRL-1** | 0.5 | NPT | V-rescale (0.5) | Berendsen | PME | **RUNAWAY** (732K) |
| **TRL-2** | 1.0 | NVT | V-rescale (0.5) | None | PME | **RUNAWAY** (718K) |
| **TRL-3** | 0.5 | NVT | V-rescale (0.5) | None | PME | **RUNAWAY** (717K) |
| **TRL-4** | 1.0 | NPT | V-rescale (0.5) | Berendsen | PME | **RUNAWAY** (715K) |
| **TRL-5** | 1.0 | NPT | V-rescale (0.5) | Berendsen | Cut-off | **RUNAWAY** (826K) |
| **TRL-6** | 1.0 | NPT | V-rescale (0.01)| Berendsen | PME | **RUNAWAY** (705K) |

## Conclusion
Stability could not be recovered by protocol adjustments. The issue is implementation-level.
