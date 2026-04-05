# TP1.18 TRL-5 Rebaseline Policy

## A. Re-anchored Basis

- vcoul fix validation:
  - the RF `vcoul` accounting bug is fixed and closed
  - under the intended TRL-5 regime, the fix removes the constant offset in `Coulomb-(SR)`, `Potential`, and `Total-Energy`, but does not change blocker observables
- TP1.14 intended-regime persistence:
  - the archived intended TRL-5 positive-VBT full-system Cut-off path still shows the blocker under `nstlist=10`
- TP1.15b / TP1.16 intended-regime cadence-sensitive effect:
  - tightening cadence to `nstlist=1` materially weakens the blocker on the same intended path
  - changing the runtime margin alone in the opposite direction does not explain that improvement
- TP1.17 contract conclusion:
  - intended positive-VBT `nstlist=10` reuse is better interpreted as contract-consistent but accuracy-limited
  - this is not source-level scheduler-bug proof

## B. Comparator Policy

- primary comparator:
  - intended positive-VBT `nstlist=1`
- secondary production-like path:
  - intended positive-VBT `nstlist=10`

Rationale:

- `nstlist=1` stays inside the intended positive-VBT contract while removing the reuse window
- `nstlist=10` is the archived production-like path, but repository evidence now bounds it as an accuracy-limited comparator rather than the best available physics reference

## C. Allowed Interpretations

- `nstlist=1` may be used as the tighter comparator for future authoritative TRL-5 Cut-off physics and blocker validation
- `nstlist=10` may be used to assess production-like sensitivity or degradation relative to the tighter comparator
- the `nstlist=10` versus `nstlist=1` gap may be used to describe intended-regime cadence-sensitive behavior
- future blocker reports may state that the blocker still persists even under the tighter intended comparator

## D. Forbidden Overclaims

- do not treat intended positive-VBT `nstlist=10` as a proof-quality physics reference
- do not treat intended positive-VBT `nstlist=1` as absolute ground truth
- do not infer source-level scheduler-bug proof from TP1.17 alone
- do not reopen manual-unsafe reuse as the main proof path
- do not claim the remaining blocker is solved
- do not claim the exact remaining blocker cause is resolved

## E. Next-step Consequence

Future authoritative TRL-5 Cut-off validation should be framed relative to this policy:

- primary reference:
  - intended positive-VBT `nstlist=1`
- secondary sensitivity path:
  - intended positive-VBT `nstlist=10`

Any later source-bug claim must be supported by new intended-regime contract evidence, not by reusing the old `nstlist=10` production-like path as if it were the best available physics comparator.
