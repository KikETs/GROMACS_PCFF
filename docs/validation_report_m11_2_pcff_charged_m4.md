# M11.2 - Strict Charged M4 Validation

## Scope

This report closes M4 only for the explicit strict-PCFF-qualified `gate_h_dense_salt_polymer_2x2x2` subset.

It does not claim:

- broad PCFF chemistry coverage
- generic dense charged ensemble readiness
- LAMMPS-vs-GROMACS charged transport parity
- charged transport readiness
- production readiness

## Artifact Inventory

Primary M4 artifacts:

- [M4 strict validation inventory](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation/m4_strict_validation_inventory.json)
- [Mechanical parity report](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation/mechanical_parity/mechanical_parity_report.json)
- [Structural / density parity report](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation/structural_density_parity/structural_density_parity_report.json)
- [Transport-facing parity report](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation/transport_facing_parity/transport_facing_parity_report.json)
- [Fresh transport-facing rerun candidate result](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation/transport_facing_rerun/summaries/candidate_result.json)
- [M4 SHA256 manifest](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation/sha256_manifest.txt)

Runner:

- [M4 runner](../tools/run_pcff_charged_m4/run_m4_strict_validation.py)

Raw bundle roots:

- [Mechanical parity raw bundle](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation/mechanical_parity)
- [Fresh transport-facing rerun raw bundle](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/m4_strict_validation/transport_facing_rerun)

## Protocol

M4 uses the strict pair from [qualified_pair_manifest.json](../tests/reference_results/pcff_charged_expansion/probe_gate_h_dense_salt_polymer_2x2x2_gmxwarm5_target10_mttk_r4/qualified_pair_manifest.json).

Separated validation branches:

- mechanical parity: fresh GROMACS-vs-LAMMPS run-0 energy/force comparison on the qualified pair
- structural / density parity: reanalysis of the strict paired NPT raw artifacts from M11.1
- transport-facing parity: fresh short-horizon CPU/GPU observable rerun on the qualified scaffold

Runner provenance:

- mechanical parity GROMACS binary: `/home/kiket/Desktop/test/GROMACS_PCFF/build/bin/gmx`
- transport-facing rerun GROMACS binary: `/home/kiket/Desktop/test/GROMACS_PCFF/build_gateb_cuda/bin/gmx`
- transport-facing GPU support: `CUDA`

Transport-facing rerun settings:

- preset: `charged-large`
- replicas: `2`
- equilibration: `0.5 ps`
- production: `5.0 ps`
- coordinate stride: `0.1 ps`
- energy stride: `0.5 ps`
- maxtau: `5.0 ps`
- CPU shape: `nb cpu / bonded cpu / pme cpu / update cpu`
- GPU shape: `nb gpu / bonded gpu / pme gpu / update gpu`

## Results

M4 inventory:

- overall status: `PASS`
- mechanical parity: `PASS`
- structural / density parity: `PASS`
- transport-facing parity: `PASS`

Mechanical parity:

- atom count: `2160`
- energy relative difference: `0.00027196319022261913`
- force RMS relative difference: `0.03722166875460191`
- force max relative difference: `0.07663604945418892`
- force max absolute difference: `3.6297686880000004 kJ/mol/nm`

Structural / density parity:

- density relative difference: `0.04539570154423199`
- volume relative difference: `0.0480657063244706`
- threshold: `0.05`

Transport-facing CPU/GPU observable parity:

- source result is a fresh M4 rerun: `true`
- scaffold manifest matches the qualified pair: `true`
- cation diffusivity comparison: `PASS`
- anion diffusivity comparison: `PASS`
- conductivity comparison: `PASS`
- transference comparison: `PASS`

Temperature caveat for the fresh transport-facing rerun:

- average production-window temperatures across the four CPU/GPU replicas were approximately `372.01 K` to `375.77 K`
- this is acceptable only for the limited CPU/GPU observable-consistency smoke purpose
- it is not acceptable as charged transport thermophysical evidence

## Capability Delta

Previous M4 state:

- no separated strict M4 rerun on the qualified pair
- no fresh M4 transport-facing source artifact

New M4 boundary:

- one strict-PCFF-qualified charged pair now has separated M4 outputs for mechanical parity, structural / density parity, and short-horizon transport-facing CPU/GPU observable parity

## Verdicts

Claim honesty verdict: `PASS`.

Capability expansion verdict: `PASS` for M4 on the explicit `gate_h_dense_salt_polymer_2x2x2` subset.

Full charged readiness verdict: `FAIL`.

Remaining blockers:

- no broad chemistry-family expansion beyond the explicit subset
- no LAMMPS-vs-GROMACS charged transport parity
- no publication-grade charged diffusion / conductivity / transference validation
- corrected TP1 endpoint continuation remains blocked by the cutoff / box margin caveat
