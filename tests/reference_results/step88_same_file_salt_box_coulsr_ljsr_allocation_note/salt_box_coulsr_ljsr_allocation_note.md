# Salt-Box Allocation Note

System: `small_salt_polymer_box`

TSV companion: [salt_box_coulsr_ljsr_allocation_note.tsv](./salt_box_coulsr_ljsr_allocation_note.tsv)

This is a same-file numeric allocation note at the serialized JSON boundary. It is not a writer-localization result.

## Salt-Box-Only Allocation Table

| system | coulsr signed delta | ljsr signed delta | two-term signed delta | coulsr signed share | ljsr signed share | coulsr absolute-value share | ljsr absolute-value share | interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `small_salt_polymer_box` | `0.000000000752446000` | `0.000000000000148002` | `0.000000000752594002` | `99.980334430034%` | `0.019665569966%` | `99.980334430034%` | `0.019665569966%` | `tiny but nonzero total delta with numeric allocation` |

## Denominator Handling

- Signed share denominator: signed `coulsr+ljsr` two-term identity delta.
- Absolute-value share denominator: `abs(coulsr delta) + abs(ljsr delta)`.

## Small-Oligomer Note

- `small_oligomer` had an exact-zero two-term delta in the earlier artifact, so shares were previously `N.A.` and are not repeated here.
