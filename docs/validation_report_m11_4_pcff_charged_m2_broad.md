# M11.4 - M2 Broader Dense Charged Parity

## Scope

This report closes M2.1-M2.5 only for one explicitly predeclared high-pressure broader dense-parity campaign:

- campaign: `m2_broad_v3_250bar_mttk_skin4_lmp1`
- ensemble: NPT
- target pressure: `250 bar`
- target horizon: `100 ps`
- final analysis window: `50 ps`
- required systems: `gate_h_dense_salt_polymer_2x2x2` and `monoglyme_ethane_litfsi_1to1_dense18`

This is not an ambient 1 bar dense-parity claim, not generic dense charged ensemble readiness, and not charged transport readiness.

## Artifact Inventory

Primary campaign artifacts:

- [M2 broad protocol](../tests/reference_results/pcff_charged_expansion/m2_broad_v3_250bar_mttk_skin4_lmp1/m2_broad_protocol.json)
- [M2 broad campaign summary](../tests/reference_results/pcff_charged_expansion/m2_broad_v3_250bar_mttk_skin4_lmp1/m2_broad_campaign_summary.json)
- [M2 broad SHA256 manifest](../tests/reference_results/pcff_charged_expansion/m2_broad_v3_250bar_mttk_skin4_lmp1/sha256_manifest.txt)

Strict pair manifests:

- [gate_h qualified pair manifest](../tests/reference_results/pcff_charged_expansion/m2_broad_v3_250bar_mttk_skin4_lmp1/systems/gate_h_dense_salt_polymer_2x2x2/qualified_pair_manifest.json)
- [M5 dense18 qualified pair manifest](../tests/reference_results/pcff_charged_expansion/m2_broad_v3_250bar_mttk_skin4_lmp1/systems/monoglyme_ethane_litfsi_1to1_dense18/qualified_pair_manifest.json)

Dense NPT reports:

- [gate_h dense NPT parity report](../tests/reference_results/pcff_charged_expansion/m2_broad_v3_250bar_mttk_skin4_lmp1/systems/gate_h_dense_salt_polymer_2x2x2/paired_npt/dense_npt_parity_report.json)
- [M5 dense18 dense NPT parity report](../tests/reference_results/pcff_charged_expansion/m2_broad_v3_250bar_mttk_skin4_lmp1/systems/monoglyme_ethane_litfsi_1to1_dense18/paired_npt/dense_npt_parity_report.json)

Runner:

- [M2 broad runner](../tools/run_pcff_charged_m2_broad/run_m2_broad.py)
- [M1-M3 paired runner used by the M2 campaign](../tools/run_pcff_charged_capability_subset/run_m1_m3.py)

## Protocol

The protocol was frozen in `m2_broad_protocol.json` before interpreting the v3 campaign results.

Required gate:

- density relative difference `<= 0.05`
- volume relative difference `<= 0.05`
- target horizon `100 ps`
- final analysis window `50 ps`
- all predeclared systems must pass
- TP1 thermal recovery is not counted as M2 evidence
- M5 workflow smoke is not counted as M2 evidence
- the old 10 ps / 5 ps single-system result is not counted as broader M2 evidence

Execution details:

- GROMACS: `md-vv`, `nose-hoover`, `MTTK`
- LAMMPS: `fix npt`
- pressure: `250 bar`
- thermal start: generated velocities
- warmup: `5 ps` GROMACS-only warmup, explicitly predeclared
- LAMMPS neighbor rule: `neighbor 4.0 bin`, `neigh_modify delay 0 every 1 check yes`

Because the warmup is GROMACS-only and the target pressure is 250 bar, this is not a fully symmetric equilibration claim and not an ambient-density claim.

## Results

Campaign summary status: `PASS`.

`gate_h_dense_salt_polymer_2x2x2`:

- status: `PASS`
- density relative difference: `0.01838659601828917`
- volume relative difference: `0.018277706713654057`
- GROMACS density mean: `1534.4095746566866 kg/m^3`
- LAMMPS density mean: `1506.706373253493 kg/m^3`
- GROMACS volume mean: `37.991190409181634 nm^3`
- LAMMPS volume mean: `38.698510433133734 nm^3`

`monoglyme_ethane_litfsi_1to1_dense18`:

- status: `PASS`
- density relative difference: `0.03978330099184161`
- volume relative difference: `0.04393341199598194`
- GROMACS density mean: `1405.2603501556887 kg/m^3`
- LAMMPS density mean: `1463.4825155688623 kg/m^3`
- GROMACS volume mean: `8.691774115768464 nm^3`
- LAMMPS volume mean: `8.325985178642714 nm^3`

## Capability Delta

Old M2 boundary:

- one strict paired dense system: `gate_h_dense_salt_polymer_2x2x2`
- target horizon: `10 ps`
- final analysis window: `5 ps`
- near-threshold density and volume parity

New M2 boundary:

- two strict-PCFF-qualified dense charged paired systems
- target horizon: `100 ps`
- final analysis window: `50 ps`
- both systems pass density and volume gates in the same predeclared campaign
- M5 workflow chemistry is promoted to dense paired evidence as `monoglyme_ethane_litfsi_1to1_dense18`

## Verdicts

Claim honesty verdict: `PASS`.

M2 capability verdict: `PASS` for the explicit high-pressure broader dense-parity campaign.

Ambient 1 bar broader M2 verdict: `FAIL`.

Generic dense charged readiness verdict: `FAIL`.

Charged transport readiness verdict: `FAIL`.

Remaining blockers:

- ambient 1 bar broader dense parity remains unresolved
- GROMACS-only warmup prevents a fully symmetric equilibration claim
- transport-facing and transport-property validation are outside this M2 evidence
- broad PCFF chemistry coverage remains unsupported
