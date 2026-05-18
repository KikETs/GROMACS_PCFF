# Validation Report — TP1.18 TRL-5 Physics Validation Rebaseline

## Verdict

- milestone: `TP1.18`
- policy persisted: `YES`
- new experiments required: `NO`
- overall verdict: `REBASELINE POLICY VALIDATED`

## Outcome

TP1.18 does not add a new physics claim. It persists the already-supported comparator policy for authoritative TRL-5 Cut-off validation.

The load-bearing basis is:

- the RF `vcoul` accounting bug is real, fixed, and closed
- TP1.14 showed that the blocker still persists under the archived intended TRL-5 auto-buffered full-system Cut-off regime
- TP1.15b and TP1.16 showed that, within the intended positive-`verlet-buffer-tolerance` regime, tighter cadence materially weakens the blocker while the smaller-margin explanation alone does not
- TP1.17 concluded that intended positive-VBT `nstlist=10` reuse is better interpreted as contract-consistent but accuracy-limited, not as proven scheduler-bug evidence

That means future authoritative TRL-5 Cut-off validation should no longer treat intended positive-VBT `nstlist=10` as the best available comparator. The tighter intended comparator is now intended positive-VBT `nstlist=1`.

This does not solve the remaining blocker. The blocker still exists under the tighter comparator: on the same authoritative full-system path, `T >= 700 K` is delayed from `28 ps` to `73 ps`, but it still happens, and the max temperature is still `774.447 K`.

## Rebaseline Policy

- primary comparator:
  - intended positive-VBT `nstlist=1`
- secondary production-like path:
  - intended positive-VBT `nstlist=10`

Allowed interpretations:

- `nstlist=1` may be used as the tighter comparator for future authoritative TRL-5 Cut-off physics and blocker validation
- `nstlist=10` may be used to quantify production-like sensitivity or degradation relative to the tighter comparator
- differences between the two may be used to discuss intended-regime cadence sensitivity

Forbidden overclaims:

- do not treat intended positive-VBT `nstlist=10` as a proof-quality physics reference
- do not treat intended positive-VBT `nstlist=1` as absolute ground truth
- do not infer source-level scheduler-bug proof from TP1.17 or TP1.18
- do not claim the remaining blocker is solved
- do not claim remaining blocker causation is fully resolved

## Evidence Basis

Repository-local evidence used:

- intended-regime vcoul A/B: [comparison_summary.json](../ab_runs/tp1_14_archived_auto_buffered_trl5_rebaseline/20260321_trl5_archived_auto/comparison_summary.json)
- intended-regime dynamic comparator A/B: [comparison_summary.json](../ab_runs/tp1_15b_intended_regime_dynamic_reuse_probe/20260321_trl5_fullsystem_auto_vbt/comparison_summary.json)
- cadence-vs-margin separation: [tp1_16_three_way_summary.json](../ab_runs/tp1_16_intended_regime_cadence_vs_margin_separation/20260321_trl5_fullsystem_auto_vbt_sep/tp1_16_three_way_summary.json)
- manual-unsafe contract downgrade basis: [tp1_5e_contract_verdict.json](../tests/reference_results/tp1_5e_pairlist_contract_audit/tp1_5e_contract_verdict.json)
- intended-regime contract interpretation basis: [20260321_tp1_13_contract_audit_summary.json](../ab_runs/tp1_13_contract_audit/20260321_tp1_13_contract_audit_summary.json)

## Reporting

- files changed
  - [docs/validation_report_tp1_18.md](./validation_report_tp1_18.md)
  - [docs/tp1_18_trl5_rebaseline_policy.md](./tp1_18_trl5_rebaseline_policy.md)
  - [tests/reference_results/tp1_18_rebaseline_policy/tp1_18_policy.json](../tests/reference_results/tp1_18_rebaseline_policy/tp1_18_policy.json)
- strongest confirmed finding
  - intended positive-VBT `nstlist=1` is the tighter authoritative comparator currently supported by repository evidence for future TRL-5 Cut-off validation
- strongest unresolved uncertainty
  - the remaining blocker still exists under that tighter comparator, and its exact root cause is still unresolved
- exact next step recommendation
  - frame future authoritative TRL-5 Cut-off validation relative to the TP1.18 comparator policy, with `nstlist=1` primary and `nstlist=10` secondary only
- verdict
  - `REBASELINE POLICY VALIDATED`
