# R-RESPA M2h Validation Report

- Milestone: `R-RESPA M2h — Pair-Specific Rule-Derivation Trace Inside gen_nnb() / do_gen() / nnb2excl()`
- Worktree: `/home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2`
- Branch: `respa-m2-exact-three-level`
- Head commit at run start: `1d28ea15277b82394bd34c99e74d972fcf1525be`

## Scope

- Fixture: `dense_oligomer` only
- Integrator path relevance: exact 3-level downstream context only
- Preprocessing focus: `grompp` / `generate_excl` rule derivation only
- Timestep: coarse `dt = 0.0005 ps` only
- Step relevance: `0` only
- Target pair: `(0,1)` only
- Narrow control: `(0,4)` only
- Out of scope: merge-stage re-analysis, outer write re-proof, full-system TRL-5, production claims

## Starting Boundary

- M2g proved pair `(0,1)` flows from bonded topology through `generate_excl_output`, runtime exclusions input, bit clear, append branch, and `excludedPairs ordinal=0`.
- M2g did not prove whether the `generate_excl` rule derivation itself was already wrong.
- M2h therefore traces the rule internals inside `gen_nnb()`, `do_gen()`, and `nnb2excl()`.

## Files Changed

- `src/gromacs/gmxpreprocess/gpp_nextnb.cpp`
- `tools/run_respa_m2_microfixtures/run_respa_m2.py`
- `docs/validation_report_respa_m2h.md`

## Commands Run

- Build:
  - `cmake --build /home/kiket/바탕화면/test/ab_builds/respa_m2_exact_three_level --target gmx -j4`
- M2h harness:
  - `python3 /home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tools/run_respa_m2_microfixtures/run_respa_m2.py --gmx-bin /home/kiket/바탕화면/test/ab_builds/respa_m2_exact_three_level/bin/gmx --fixture dense_oligomer --pair-rule-derivation-trace --milestone-name 'R-RESPA M2h' --out /home/kiket/바탕화면/test/ab_worktrees/GROMACS_PCFF_respa_m2/tests/reference_results/r_respa_m2h_pair_rule_derivation_trace`
- Full command log:
  - `tests/reference_results/r_respa_m2h_pair_rule_derivation_trace/raw_commands.txt`

## Pair (0,1) Internal Rule Trace

The pair-specific rule lineage is now explicit inside `generate_excl`:

1. `gen_nnb_bond_membership`
   - target `(0,1)` is present exactly once in the bidirectional bond list
   - control `(0,4)` is absent
2. `do_gen_direct_bond_rule_fire`
   - target fires `source=exclude_all_bonded_atoms`
   - `condition_nrex_positive=true`
   - `nre_bucket=1`
   - `add_nnb_called=true`
3. `do_gen_summary`
   - `target_direct_bond_rule_fired=true`
   - `target_higher_order_rule_fired=false`
   - no control rule fires
4. `sort_and_purge_output`
   - atom `0` buckets become `level0=0 level1=1 level2=2 level3=3`
   - target is in `level1`, not a higher-order artifact
5. `nnb2excl_emit`
   - emitted exclusions are `0,1,2,3`
   - target `(0,1)` is present
   - control `(0,4)` is absent

## Exact Rule-Fire Proof

The exact rule-fire point is:

- stage: `do_gen_direct_bond_rule_fire`
- predicate basis:
  - pair `(0,1)` appears in the `gen_nnb()` bond list
  - `nrexcl > 0`
  - the direct bonded-neighbor exclusion rule is active
- action:
  - `add_nnb(nnb, 1, 0, 1)` is taken under `exclude_all_bonded_atoms`

This is not inferred from final output alone. The proof chain is:

- internal rule fire:
  - `step0_generate_excl_internal_trace.txt`
- emitted exclusions:
  - `step0_generate_excl_internal_trace.txt`
- materialized `generate_excl_output`:
  - `step0_grompp_generate_excl_trace.txt`
- runtime continuity:
  - `step0_runtime_exclusions_input.txt`

## Policy Interpretation Verdict

- `BASELINE-INTENDED`

Reason:

- topology has `[ bonds ] 1 2`
- topology has `nrexcl = 3`
- the traced internal rule is exactly `exclude_all_bonded_atoms`
- the target enters exclusions as a direct bonded neighbor at level 1, not through an anomalous higher-order propagation path

Therefore, for pair `(0,1)`, bonded-neighbor exclusion generation inside `generate_excl` is baseline-intended behavior under the visible topology/exclusion policy.

## Earliest Bad Handoff Verdict

- `NOT-HERE`

Reason:

- M2h closes the generate-exclusion-rule question in the negative: the rule derivation is behaving as intended for the traced pair
- this means the earliest semantically bad handoff is not inside `gen_nnb()` / `do_gen()` / `nnb2excl()` for pair `(0,1)`
- the remaining defect must be later, when the downstream exact 3-level ownership path consumes this valid exclusion set incorrectly

## Control Pair

Control pair `(0,4)` stays clean under the same instrumentation depth:

- absent from `gen_nnb_bond_membership`
- no direct-bond or higher-order rule fire
- absent from `sort_and_purge_output` exclusion buckets
- absent from `nnb2excl_emit`
- absent from `generate_excl_output`
- absent from runtime exclusions input

This shows the tracing method can distinguish intended bonded-neighbor exclusion generation from a nearby non-bonded pair that should remain outside the exclusion path.

## Continuity With M2g

Internal rule-fire and M2g output artifacts now form one continuous chain:

1. `do_gen_direct_bond_rule_fire`
2. `sort_and_purge_output`
3. `nnb2excl_emit`
4. `generate_excl_output`
5. `runtime_exclusions_input`

The emitted exclusion set `{0,1,2,3}` matches the later materialized output and runtime input modulo ordering.

## Verdict

- `BASELINE-INTENDED GENERATE_EXCL RULE FIRE PROVEN; EARLIEST BAD HANDOFF NOT HERE`

## Narrow Conclusion

Within the locked scope `dense_oligomer`, exact 3-level context, coarse `dt=0.0005`, step `0`, pair `(0,1)`:

- the bond-to-exclusion transition inside `generate_excl` is baseline-intended
- the first bad handoff is not inside `gen_nnb()` / `do_gen()` / `nnb2excl()`
- the remaining defect must be sought later than exclusion generation, not earlier
