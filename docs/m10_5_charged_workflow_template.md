# M10.5 — Charged/Salt Qualified Workflow Template

## 1. System Requirements
- Charged or salt-containing system.
- Supported PCFF atom types and ionic species (Na+, Cl-, etc.).
- **Warning:** PME/PPPM virial differences may lead to different equilibrium densities than LAMMPS.

## 2. Mandatory Monitoring Rules

### 2.1 Extended Equilibration
- **Duration:** > 500 ps recommended.
- **Rule:** Do not start production until density drift is < 0.1% per 100 ps.

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

## 4. Acceptance Criteria for Production
- [ ] Potential energy matches LAMMPS static reference within 0.1% (M10.4 baseline).
- [ ] Density trace has reached a plateau.
- [ ] No reciprocal-space grid warnings in `grompp`.

## 5. User Responsibility
Users of charged systems are responsible for performing a **manual sanity check** on the final density. If GROMACS density deviates significantly (> 10%) from known experimental or LAMMPS benchmarks, the run should be treated as "Qualified but Unvalidated".
