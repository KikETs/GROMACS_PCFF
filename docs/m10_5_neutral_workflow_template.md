# M10.5 — Neutral Production Workflow Template

## 1. System Requirements
- Neutral system (net charge = 0.0).
- Supported PCFF atom types.
- Minimum box size > 2x Cutoff (typically > 2.0 nm).

## 2. Recommended MDP Settings

### 2.1 Minimization (`min.mdp`)
```ini
integrator  = steep
nsteps      = 1000
emtol       = 100.0
cutoff-scheme = Verlet
vdw-type     = Cut-off
rvdw         = 0.9
coulombtype  = PME
rcoulomb     = 0.9
```

### 2.2 Equilibration (`equil.mdp`)
- **Duration:** 100 ps - 500 ps (depending on density).
- **Settings:**
```ini
integrator  = md
dt          = 0.001
tcoupl      = v-rescale
tc-grps     = System
tau-t       = 0.1
ref-t       = 300
pcoupl      = Berendsen
pcoupltype  = isotropic
tau-p       = 1.0
compressibility = 4.5e-5
ref-p       = 1.0
```

### 2.3 Production (`prod.mdp`)
- **Duration:** > 1 ns recommended.
- **Settings:**
```ini
integrator  = md
dt          = 0.001
tcoupl      = v-rescale
tc-grps     = System
tau-t       = 0.1
ref-t       = 300
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau-p       = 2.0
```

## 3. Validation Checklist
- [ ] Potential energy vs time is stable (no monotonic drift).
- [ ] Temperature is within 2 K of target.
- [ ] Density is within 1% of LAMMPS reference (if available).
- [ ] Volume fluctuations are stable.
