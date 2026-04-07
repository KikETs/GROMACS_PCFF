# Current Support Matrix

The older `Production-Ready` / `Qualified-Ready` wording overstated the current charged path and is withdrawn.

Source of truth:

- [Current Status Note](current_status_note.md)
- [Machine-Readable Support Matrix](../tests/reference_results/pcff_ion_narrow_claim/support_matrix.json)

| Scope Item | Current Status | Evidence | Present-Tense Boundary |
| :--- | :--- | :--- | :--- |
| PT8 supported SPE typing/export | `exact` | PT8 typing validation | Three frozen glyme + Li/TFSI cases only |
| Frozen charged topology semantics | `exact` | PT8 smoke parity, PT8.4, PT8.4.1 | `lj/class2/coul/long`, sixth-power mixing, `special_bonds`, k-space requirement |
| Frozen small charged combined mechanics | `exact` | PT8.5 combined parity | Small charged fixture only |
| Dense charged-box short-horizon mean PE / temperature | `approximate` | M10.4 summary | Partial diagnostic only |
| Dense charged-box density / volume parity | `exact` for explicit M11.1 subset; `unsupported` generally | M11.1/M11.2 subset reports + M10.4 summary | Only `gate_h_dense_salt_polymer_2x2x2` is density/volume-parity-qualified |
| Long-horizon charged stability | `exact` for corrected TP1 thermal blocker only | TP1 exact recovery audit | Endpoint continuation safety still fails cutoff/box audit |
| M4 strict charged validation | `exact` for explicit M11.2 subset | M4 strict validation inventory | Mechanical, structural/density, and short-horizon transport-facing CPU/GPU observable parity only |
| M5 chemistry-scope expansion | `exact` for explicit M11.3 workflow subset | M5 chemistry expansion report | One acyclic alkane neutral additive in `monoglyme_ethane_litfsi_1to1` only |
| Charged transport observables | `unsupported` for readiness | M10 method readiness + M11.2 transport-facing report | No LAMMPS-vs-GROMACS charged transport parity or publication-grade transport claim |
| Broad chemistry outside PT8 subset | `unsupported` | CSV scope audit | No chemistry-complete PCFF claim |

Status definitions come from the support matrix JSON:

- `exact`: checked-in artifact directly supports the statement inside a frozen scope
- `approximate`: diagnostic evidence exists, but it is too caveated to justify readiness language
- `unsupported`: present-tense support claim is blocked by direct evidence or an explicit scope boundary
