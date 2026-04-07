# TP1.3 Stabilization Trial Matrix

The following matrix defines the diagnostic trials for identifying the root cause of the `dense_salt_polymer` (270-atom Na/Cl) thermal runaway.

Supersession note: this matrix is historical. The TP1.3 runner used wrong GROMACS key names for coupling controls, so these trials must not be cited as the current exact-system stability verdict. The corrected TP1 exact rerun is documented in [TP1 Charged Long-Equilibration Recovery](validation_report_tp1.md).

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
This historical matrix is superseded for current TP1 stability claims. The corrected exact TP1 rerun resolves the thermal-runaway blocker only for the exact 5 ns `dense_salt_polymer` NPT protocol; it does not establish endpoint continuation safety or charged transport readiness.
