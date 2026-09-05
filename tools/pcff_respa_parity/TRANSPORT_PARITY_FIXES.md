# Transport parity fixes and remaining validation

**Scope freeze, 2026-09-05:** further transport expansion is stopped by the user's
decision to target an implementation paper. See
`../../docs/implementation_paper_scope_20260905.md` for the numerical claim and
the final validation artifact directory. The unresolved transport statements
below remain evidence limitations, not an instruction to launch more production.

The comparison reference is LAMMPS. Agreement between GROMACS CPU and GPU is an
implementation diagnostic, not the transport acceptance criterion. Long-time
LAMMPS NE/cNE0 agreement remains unverified as of 2026-09-05.

## Exact-rRESPA COM removal

The engine change executes scheduled global COM removal in the exact-rRESPA
integration path. It does not change an existing TPR's removal interval.

The existing MDP generator reads `GROMACS_BATCH_NSTCOMM` when preparing inputs:

```bash
export GROMACS_BATCH_NSTCOMM=400
```

For the tested production base timestep `dt=0.0005` ps this requests removal
every 0.2 ps. The generated MDP/TPR must contain `comm-mode=Linear`,
`comm-grps=System`, and `nstcomm=400`. Setting this environment variable only
when executing an already prepared TPR does not change that TPR. The runtime
also uses the existing `GMX_PCFF_EXACT_RESPA_ALLOW_LINEAR_COM_REMOVAL=1` gate.

The default remains `1000000000` to preserve the existing experimental protocol
unless a control explicitly requests removal. That default effectively disables
periodic removal and does not promise to clean imported initial velocities.
Changing only the executable is therefore insufficient to reproduce the COM
intervention. Initial/restart boundary handling is retained by the engine patch;
do not expect the saved initial frame to have zero COM velocity merely because
periodic removal is enabled.

The original LAMMPS production has no `fix momentum`. Its measured global COM
drift was approximately 1e-7 nm in the three 27068 trajectories. The GPU control
uses periodic removal to suppress numerical drift. These are distinct numerical
policies and should be reported as such.

## Complete cluster populations

The analyzer identifies ions by charged-molecule membership instead of fixed
atom-type numbers. Automatic population-matrix sizing includes all ions, and
an explicit undersized matrix raises an error instead of silently omitting large
clusters. Metadata checks reject caches with incompatible ion IDs, molecules,
masses, or charges.

For the current 100 Li / 100 TFSI system the reviewed matrix size is 101.
Keep the original atom-contact definition and 0.34 nm cutoff when comparing the
current experiment. The reported majority-ion-diffusion approximation is
**cNE0**, not full cluster-tracked cNE. COM removal in MD is also distinct from
subtracting ion COM displacement during trajectory analysis.

## Optional LAMMPS tail weighting

The additional engine mode is selected at runtime:

```bash
export GMX_PCFF_LAMMPS_DISPERSION_CORRECTION=1
```

It uses LAMMPS Ni*Nj population weighting for PCFF 9-6 cutoff interactions and
does not subtract bonded exclusions when averaging tail coefficients. It is
restricted to non-FEP, non-TPI, non-Buckingham 9-6 cutoff interactions with
`vdw-modifier=None`. For the complete LAMMPS energy and pressure tail, retain
`DispCorr=AllEnerPres`; the environment variable does not change the DispCorr
selection. The mode is off by default.

The mode was tested separately from the COM intervention. In the tested fixed
volume 0.2 ps NVT control, off/off-repeat/on trajectories were byte-identical.
It changes pressure and energy and can affect NPT dynamics. It has not been
shown to resolve the long-time NE/cNE0 discrepancy. Do not retroactively mix it
into the running COM experiment.

## Verified controls and limitations

- Injected COM velocity is removed by corrected CPU and GPU runs; the preserved
  original executable retains it. COM-off CPU trajectory regression passed.
- CPU checkpoint regression passed. Strict GPU trajectory-byte/restart tests
  reported differences, including comparable repeated/unpatched GPU runs;
  these failures remain recorded and were not converted into passes.
- Population regression tests passed, and automatic population matrices matched
  reviewed 101-by-101 results over all 5001 frames in 18 original trajectories.
- Same-state forces, including ionic molecular net forces, were compared directly
  with LAMMPS on initial and production snapshots. Fine-grid agreement does not
  imply agreement of the original production electrostatic approximations.
- All three COM GPU repeats have completed. Their matched mean NE differs from
  LAMMPS by +0.42%, while matched mean cNE0 differs by -34.43%; the individual
  NE differences range from -34.85% to +25.56%. Mean proximity with three noisy
  repeats does not establish equivalence. A common-LAMMPS-initial-state
  production also completed with NE -7.68% and cNE0 -34.09% relative to its
  matched LAMMPS trajectory. The identical-TPR GPU repeat completed with NE
  +25.98% and cNE0 +60.93% relative to that same LAMMPS trajectory; cNE0 changed
  by +144.17% between identical-input GPU runs. This demonstrates substantial
  observed run variability and does not establish unbiased ensemble means.
  COM removal alone is not a validated transport fix.
- Sparse-energy short controls actually exercised force reuse, including GPU
  neighbor-search boundaries. Disabling reuse and the specialized force-only
  bonded kernel did not eliminate GPU repeat variability. Diagnostic force
  dumps disable live force reuse, so dump agreement alone does not cover it.
- Mixed CPU repeated short runs became byte-identical with `-reprod` at both
  1 and 8 threads. GPU nonbonded plus `-reprod` is explicitly rejected by the
  current engine. Do not advertise that option as a GPU reproducibility fix.
- A diagnostic intervention changing only CPU FFT planning to FFTW_ESTIMATE
  reproduced the byte-identical `-reprod` result; the original MEASURE runs
  selected different plans. With CPU FFT planning fixed, moving only short-range
  nonbonded work to GPU was sufficient to produce repeat differences. GPU PME
  and bonded kernels are not necessary for that observation. This isolates
  numerical repeat variability, not long-time ensemble bias against LAMMPS.
- LAMMPS reference controls now show that identical-input 12-thread runs also
  diverge over 10 ps, while 1-thread baseline dumps are byte-identical. A
  controlled 1.84e-8 nm/ps initial velocity perturbation in the reproducible
  1-thread layout grows to 0.104 nm coordinate RMS at 10 ps. This demonstrates
  sensitivity in the reference engine; it does not measure transport bias.
- The small double single-point energy residual after tail matching has been
  isolated to LAMMPS Coulomb table/polynomial approximation and Coulomb-constant
  conventions. Independent exact-erfc/constant accounting reduces the residual
  to about 1e-6 to 3e-6 kJ/mol in two states. These are diagnostic calculations,
  not a new production binary or a demonstrated conductivity correction.

The detailed experiment evidence is under
`GROMACS_PCFF/output/polygen_multisystem_validation_20260512_m1p50/analysis/parity_root_cause_20260905`
in the shared workspace. `root_cause_progress.md` records the current completion
state; do not infer completion from this implementation note or a single test.
The frozen worker used for existing runs is preserved separately; this note
does not authorize replacing it or rewriting existing trajectories.

## Forced scalar diagnostic setup repair

The explicit `GMX_PCFF_EXACT_RESPA_DISABLE_NBNXM_NARROW=1` route previously
aborted on an unsplit two-level production input because runner setup did not
request the complete pairlist required by the scalar evaluator. Runner setup
now requests it when that CPU route is explicitly selected; ordinary GPU and
CPU settings are unchanged. The failed original run is retained.

Four 0.2 ps same-LAMMPS-state double CPU controls now complete: forced scalar
and default CPU, each with dense (4) and sparse production (40000) energy
cadence. Scalar/default position RMS differs by about 3.6e-15 nm; all four
differ from fine-grid LAMMPS by about 2.0e-8 nm. Default CPU trajectories are
byte-identical before/after the patch at both cadences. A separate diagnostic
installation preserves existing production binaries. See
`scalar_energy_cadence_pairlist_fix/report.md` in the evidence directory.
This fixes a diagnostic startup bug, not the outstanding transport discrepancy.
