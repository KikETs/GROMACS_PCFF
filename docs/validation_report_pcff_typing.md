# PT8 PCFF Typing Validation Report

## Status

이 문서는 **PT8** 범위만 고정한다.

현재 readout:

- supported SPE scope validation: **pass**
- M10 dependency handoff status: **ready for supported scope only**
- direct LAMMPS force/energy parity: **not claimed**

핵심 이유:

- PT7 golden SPE systems 세 개가 end-to-end로 모두 통과한다.
- typing failure / parameter failure / emitter failure가 machine-readable probe로 분리되어 있다.
- LAMMPS 쪽은 trajectory-matched SPE reference가 없어서 contract-level smoke parity까지만 주장한다.

## Primary Outputs

- [validation_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/pt8_typing_validation/validation_summary.json)
- [per_case_results.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/pt8_typing_validation/per_case_results.json)
- [failure_probe_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/pt8_typing_validation/failure_probe_summary.json)
- [lammps_smoke_parity_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/pt8_typing_validation/lammps_smoke_parity_summary.json)
- validation runner: [tools/run_typing_validation](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_typing_validation)

## End-to-End SPE Result

Validated systems:

- `monoglyme_litfsi_1to1`
- `diglyme_litfsi_1to1`
- `triglyme_litfsi_2to2`

Observed result:

- `3 / 3` cases pass end-to-end
- all cases are charge neutral
- all cases pass salt-balance checks
- all cases regenerate identical GROMACS outputs under `validate_existing`

Evidence basis:

- per-case bundle hashes and source-chain hashes are frozen in [per_case_results.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/pt8_typing_validation/per_case_results.json)
- PT7 workflow contract remains the execution path: [src/polymer_workflow/engine.py](/home/kiket/바탕화면/test/GROMACS_PCFF/src/polymer_workflow/engine.py)

## Golden Reference Validation

PT8 freezes the machine-readable validation summaries themselves as the golden reference set under:

- [tests/reference_results/pt8_typing_validation](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/pt8_typing_validation)

Regeneration contract:

- `PYTHONPATH=src python3 tools/run_typing_validation/generate.py generate`
- `PYTHONPATH=src python3 tools/run_typing_validation/generate.py validate`

`validate`는 regenerated JSON을 체크인된 reference와 byte-for-byte 비교한다. 이건 output drift를 감추지 않기 위한 의도적 설계다.

## Failure-Class Separation

PT8는 downstream M10이 실패 지점을 혼동하지 않도록 세 failure class를 분리한다.

- `typing_failure`
- `parameter_failure`
- `emitter_failure`

Frozen probe evidence:

- `typing_failure`
  - probe: `typing_missing_polyether_atom_rules`
  - observed code: `unresolved_atom_type`
- `parameter_failure`
  - probe: `bonded_missing_polyether_backbone_c_o_rule`
  - observed code: `missing_parameter`
  - probe: `nonbonded_missing_polyether_backbone_rule`
  - observed code: `missing_nonbonded_atom_type`
- `emitter_failure`
  - probe: `emitter_pair14_scaling_rejection`
  - observed code: `unsupported_pair14_scaling`

Direct evidence:

- [failure_probe_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/pt8_typing_validation/failure_probe_summary.json)

## LAMMPS Smoke Parity

현재 저장소에는 PT7 SPE case와 직접 짝지을 수 있는 LAMMPS trajectory reference가 없다. 그래서 PT8는 parity를 과장하지 않고 **contract-only smoke parity**만 확인한다.

Reference used:

- [testdata/lammps_golden/systems/small_salt_polymer_box/system.json](/home/kiket/바탕화면/test/GROMACS_PCFF/testdata/lammps_golden/systems/small_salt_polymer_box/system.json)

Checked contract items:

- `special_bonds = "lj/coul 0.0 0.0 1.0 angle no dihedral no"`
- `pair_modify = "mix sixthpower"`
- charged pair style contract is `lj/class2/coul/long`
- salt-containing mixed systems require kspace semantics
- explicit self-only pair coefficient policy remains intact

Observed result:

- all three SPE cases pass this contract-level smoke parity

Direct evidence:

- [lammps_smoke_parity_summary.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/pt8_typing_validation/lammps_smoke_parity_summary.json)

## Honest Readout

What PT8 now supports:

- deterministic SPE typing/export validation for the PT7 supported chemistry subset
- machine-readable separation of typing / parameter / emitter failures
- reproducible GROMACS topology regeneration suitable for M10 dependency handoff

What PT8 does **not** prove:

- general polymer electrolyte readiness outside the PT7 supported subset
- direct LAMMPS-vs-GROMACS energy/force parity for these SPE systems
- scientific transport validity by itself

That second point matters. If you treat contract-level LAMMPS smoke parity as force parity, you are overstating the evidence.
