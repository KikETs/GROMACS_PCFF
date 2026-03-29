# Validation Report — TP1.5e Pairlist Contract / Buffer-Policy Compliance Audit

## Verdict

- milestone: `TP1.5e`
- classification: `ALLOWED-UNSAFE`
- patching now justified: `NO`
- overall verdict: `PASS`

## Outcome

TP1.5e does not support calling the `(1,4, shift 21)` omission a confirmed implementation defect. On the TP1.5b `dense_nonlisted` fixture, the `n10_r0909` run rebuilt at step `170` with the pair still outside the manually requested outer list by `0.000534773 nm`. The pair only entered `rlist` at step `171` and only entered the actual interaction cutoff at step `172`, while the next rebuild did not occur until step `180`.

That behavior is consistent with the current manual-buffer contract:

- `verlet-buffer-tolerance = -1` means `rlist` is user-controlled rather than auto-sized from the Verlet-buffer tolerance contract
- the audited run does not have dynamic pruning enabled
- therefore a pair that is outside `rlist` at the last rebuild is not guaranteed to be captured before the next rebuild

The same fixture family already showed in TP1.5b that `auto_buffer_n10_vbt0005` chose `rlist ≈ 0.911 nm` and removed the worsening. TP1.5e therefore classifies the reproduced omission as contract-compliant but unsafe under the chosen manual `rlist = 0.909 nm` / `nstlist = 10` regime.

## Evidence Boundaries

- confirmed:
  - `n10_r0909` last rebuild before the divergence is step `170`
  - step `170` pair distance is `0.9095347524 nm`, larger than `rlist = 0.9089999795 nm`
  - step `171` is the first step below `rlist`
  - step `172` is the first step below the actual cutoff `0.9 nm`
  - dynamic pruning is inactive in both `n1_r0909` and `n10_r0909`
- not confirmed:
  - that manual `rlist = 0.909` is a good production choice for the larger TP1.3 system
  - that GROMACS should promise stronger capture semantics in manual-`rlist` mode

## Sources

Checked on `2026-03-19`:

- GROMACS Manual 2024.4, `mdp-options`: <https://manual.gromacs.org/documentation/2024.4/user-guide/mdp-options.html>
- GROMACS Manual 2024.4, release notes: <https://manual.gromacs.org/2024.4/release-notes/2024/2024.4.html>

Repository basis:

- [pairlist_tuning.cpp](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlist_tuning.cpp)
- [pairlistsets.h](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/nbnxm/pairlistsets.h)
- [calc_verletbuf.h](/home/kiket/바탕화면/test/GROMACS_PCFF/src/gromacs/mdlib/calc_verletbuf.h)
- [tp1_5e_contract_verdict.json](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5e_pairlist_contract_audit/tp1_5e_contract_verdict.json)

## Reporting

- files changed
  - [tools/run_tp1_5e_pairlist_contract_audit/run_pairlist_contract_audit.py](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_5e_pairlist_contract_audit/run_pairlist_contract_audit.py)
  - [tools/run_tp1_5e_pairlist_contract_audit/README.md](/home/kiket/바탕화면/test/GROMACS_PCFF/tools/run_tp1_5e_pairlist_contract_audit/README.md)
  - [docs/validation_report_tp1_5e.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/validation_report_tp1_5e.md)
  - [docs/tp1_5e_pairlist_contract_audit.md](/home/kiket/바탕화면/test/GROMACS_PCFF/docs/tp1_5e_pairlist_contract_audit.md)
  - [tests/reference_results/tp1_5e_pairlist_contract_audit](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5e_pairlist_contract_audit)
- commands run
  - `git status --short`
  - `git rev-parse HEAD`
  - `build/bin/gmx --version`
  - multiple `sed -n ...` and `rg -n ...` inspections over TP1.5b/TP1.5c/TP1.5d artifacts and pairlist source paths
  - `python3 -m py_compile tools/run_tp1_5e_pairlist_contract_audit/run_pairlist_contract_audit.py`
  - `python3 tools/run_tp1_5e_pairlist_contract_audit/run_pairlist_contract_audit.py`
  - exact `gmx grompp` and `gmx mdrun` commands: [raw_commands.txt](/home/kiket/바탕화면/test/GROMACS_PCFF/tests/reference_results/tp1_5e_pairlist_contract_audit/raw_commands.txt)
- fixtures executed
  - TP1.5b `dense_nonlisted` under `n1_r0909`
  - TP1.5b `dense_nonlisted` under `n10_r0909`
- strongest confirmed finding
  - at the last `n10_r0909` rebuild before the divergence, the pair was still outside the manually requested outer list; omission after that rebuild is therefore contract-compliant in manual-buffer mode
- strongest unresolved uncertainty
  - whether manual `rlist = 0.909` is an acceptable production operating point for the larger dense charged TP1.3 setup
- exact next step recommendation
  - keep production code unchanged and treat this fixture as a parameter-contract issue unless a later milestone shows that manual-`rlist` mode must provide stronger guarantees
- verdict
  - `PASS`
