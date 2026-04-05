# Frozen-Reference Parity Mismatch Evidence Summary

This is an artifact-only summary of existing frozen-reference parity mismatch evidence. No fresh runtime rerun was performed for this commit.

- same-file arithmetic/identity family is closed enough for comparison purposes
- remaining live mismatch is the frozen-reference parity family
- first justified broken family is `step0_potential_state`
- `initial_total` and `final_total` remain downstream

Machine-readable index: `frozen_reference_parity_mismatch_index.tsv`

## Cross-Linked Evidence

- First-family ranking and downstream classification come from `step36_runtime_parity_gate_relocalization_summary/dense_oligomer/dt_0p0005/runtime_parity_relocalization_summary.json`.
- Frozen-reference contract deltas come from `step58_exact_respa_initial_total_contract_compare`, `step59_exact_respa_final_total_contract_compare`, and `step60_exact_respa_step0_potential_contract_compare`.
- Same-file closure context comes from `step81_same_file_raw_sum_identity_compare`, `step86_same_file_complementary_slice_identity_compare`, and `step87_same_file_coulsr_ljsr_contribution_share`.

## Notes

- This summary does not identify a writer, serialization bug, or root cause.
- This summary does not add a new arithmetic identity proof.
