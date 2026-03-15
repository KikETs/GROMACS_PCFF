# Release Checklist: v1.0.0-rc1

This checklist must be completed by the maintainer before tagging the `v1.0.0-rc1` release.

## 1. Documentation Review
- [ ] `README.md` points to the correct release notes.
- [ ] `docs/releases/v1.0.0-rc1.md` matches the final validation outcomes.
- [ ] Readiness classes (Ready vs Qualified) are consistent across all files.
- [ ] Transport properties are explicitly marked as unvalidated.

## 2. Evidence Verification
- [ ] `m10_3_summary.json` confirms sub-1% neutral density parity.
- [ ] `m10_4_summary.json` confirms sub-0.1% charged energy parity.
- [ ] All validation reports (M10.1 - M10.5) are present in `docs/`.

## 3. Artifact Packaging
- [ ] `readiness.json` metadata is updated with the current version and date.
- [ ] Workflow templates (`neutral`, `charged`) are verified for syntax correctness.
- [ ] No temporary or debug files are left in the repository.

## 4. Final Sanity Check
- [ ] Run a single-point energy check on one neutral fixture.
- [ ] Run a 10-step NPT test on one charged fixture to ensure no immediate blow-up.
