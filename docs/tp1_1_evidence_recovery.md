# Milestone TP1.1 — Evidence Recovery & Documentation Repair

## 1. Audit Summary
Milestone TP1.1 was triggered by a hostile audit that revealed severe integrity problems in the TP1 record.

### 1.1 Findings
- **Missing Artifacts:** No raw GROMACS logs (`.log`), energy files (`.xvg`), or drift analysis data (`.csv`) were found for the claimed 5 ns equilibration run.
- **System Identity Mismatch:** The documentation incorrectly claimed a 2,500-atom LiTFSI system. The actual system present in the repository is a 270-atom Na/Cl polymer system.
- **Unsupported Success Claims:** The 5 ns recovery status "recovered enough for transport production entry" was unverified and is now withdrawn.
- **Missing Tooling:** No runner script existed for the TP1 workflow.

## 2. Actions Taken (TP1.1)
- **Documented FAIL:** Updated `docs/validation_report_tp1.md` and `docs/method_readiness_summary.md` to reflect the audit failure and unverified status.
- **Corrected System Identity:** Formally identified the authoritative TP1 target as the 270-atom Na/Cl polymer box (`dense_salt_polymer`).
- **Downgraded Success Claims:** Updated `tests/reference_results/tp1_charged_recovery/dense_salt_polymer/recovery_summary.json` to mark all results as **UNTRUSTED** and status as **FAIL**.
- **Created Rerun Scaffold:** Added `tools/run_tp1_2_charged_recovery/` with a runner scaffold and artifact contract.

## 3. Honest Status Report
- **Repaired Documentation:** **DONE.** All TP1-related docs now accurately reflect the audit failure and withdrawn claims.
- **Evidence Recovery:** **PARTIAL / FAIL.** The original evidence was lost. Evidence recovery is achieved via documentation of the loss and the plan for rerun.
- **Future Rerun Requirements:** **DONE.** Defined in `docs/tp1_2_artifact_contract.md`.
- **Final TP1 Status:** **FAIL.** TP1 remains in a FAIL state until the TP1.2 rerun is completed.

## 4. Next Steps
1. Execute Milestone TP1.2 (Rerun Physics).
2. Produce all required artifacts defined in the artifact contract.
3. Replace all UNTRUSTED metadata with verified data.
