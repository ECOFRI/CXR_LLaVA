# G2-C — Evaluator Adapter Local Result

**Scope:** Local code + synthetic integration verification. **Not** the research-server runtime check.

- `PASS_LOCAL_SYNTHETIC`: 12/12 unit tests passed.
- `PASS_END_TO_END_SYNTHETIC`: wide-style labeler results joined through explicit report keys, converted to long labels, and evaluated on exactly 20 synthetic images × 4 fixed arms × 5 generic findings.
- Expected grids: 80 report rows, 100 reference-label rows, 400 prediction-label rows.
- Controlled test: C0/F01 joint-definite count 16/20 with TP=0 TN=14 FP=1 FN=1; one separate reference-positive/prediction-blank case remained unresolved rather than becoming FN. C2a/F01 matched definite count 18/20 in synthetic fixture.
- Rejection tests: duplicate prediction, duplicate image, missing prediction, missing report, unsupported label, unmatched labeler source all reject.
- Missing-label policy: uncertain and blank remain distinct and excluded from main joint-definite confusion matrix. Undefined metrics are null.
- Local smoke-test wall duration: ~0.77 s, peak RSS 96,776 KiB (from `/usr/bin/time`); local unit-test wall duration: ~0.75 s, peak RSS 97,036 KiB. No GPU execution.

**Outstanding:** Pinned original CheXpert-Labeler runtime and actual output schema were not accessible in this conversation's container. The server repository path `/storage/student4/hanhpm/CXR_LLaVA_Improvement` is not mounted here. Four-K80 permission has been given by the user; resource reservation and allocation were not performed. Therefore the overall project gate remains `HOLD_G2C_MORE_EVIDENCE`, specifically `HOLD_SERVER_LABELER_RUNTIME_VERIFICATION` and any unresolved host quota checks.

**Recommended operation:** Give the bundle plus `docs/SOL_LIGHT_SERVER_HANDOFF.md` to the authorized server agent. Do not run E4 before the server reports `PASS_G2C_EVALUATOR_READY` and compute readiness evidence.
