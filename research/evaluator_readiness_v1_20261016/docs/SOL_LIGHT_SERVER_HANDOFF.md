# GPT-6 Sol Light — G2-C Evaluator Runtime Completion

Current state: G2-C is HOLD because the pinned evaluator's runtime and a four-arm adapter remain unverified on the research server. G2-B2 PASS (20 images, 100 CLEAR scores). The user authorizes use of four K80 GPUs for a future E4 run, but E4 generation remains unapproved and must not start.

## Objective

Integrate and verify the attached standalone `G2C_evaluator_readiness` package **without modifying the original evaluator**. Complete only the evaluator blocker and inspect actual GPU allocation/quota/output allowance; do not run E4.

## Tasks

1. Locate verified project root `/storage/student4/hanhpm/CXR_LLaVA_Improvement` and the G2-C/G2-B2 manifests. Verify their hashes and exact five finding names. Do not rerun previous gates.
2. Check the existing pinned CheXpert-Labeler source, entrypoint, dependencies and real CSV/JSON output shape. Document observed source signatures; never guess column names or silently coerce labels.
3. Place the standalone tools under a **new versioned directory in `research/`**. Check that files do not already exist; never overwrite original code or previous run artifacts.
4. Run `python -m unittest discover -s tests -v` and `python tests/smoke_end_to_end.py <versioned_synthetic_dir>` using the current server Python environment, without GPU.
5. Exercise the **actual original labeler runtime on safe synthetic reports** if allowed and technically supported, then use an explicit report-key mapping into the new wide-to-long bridge. Confirm label semantics with controlled positive/negative/uncertain/blank examples; compare expected labels where labeler semantics support them. Do not claim full semantic correctness from synthetic language templates alone. If the original labeler cannot run without additional dependencies, STOP and record exactly what is missing; do not auto-install packages.
6. Validate report input/output mapping and structural coverage of 20 cohort IDs × C0/C1/C2a/C3 × five pinned findings (expected 80 reports, 100 references, 400 prediction-label records) using synthetic fixtures only. Check deliberate missing and duplicate cases fail.
7. Inspect GPU quota/allocation policy and output allowance through read-only cluster commands, using the user's authorization for up to four K80 GPUs **later**. Do not reserve or launch the model. Distinguish user permission from scheduler/host availability.
8. Save `G2C_evaluator_runtime_report.md`, `G2C_adapter_mapping.md`, `G2C_compute_allocation.md`, `G2C_decision.md` and a run manifest with verified hashes under a new versioned `research/runs/` directory.

## Decision

- `PASS_G2C_EVALUATOR_READY`: the pinned original labeler executed successfully on synthetic texts and its real outputs were joined to the four-arm adapter without missing keys; the 80/100/400 coverage and missing-label policies were validated. Separately report compute allocation PASS or HOLD.
- `HOLD_G2C_EVALUATOR`: dependency/source/schema mismatch or labeler runtime not confirmed; report bounded remedy.
- `BLOCKED_G2C_EVALUATOR`: primary input integrity or unresolvable interface blocker.

Do not claim E4 readiness if either evaluator or compute quota still fails. Do not start E4. Stop and report in Vietnamese with evidence paths, measured CPU/RAM, and decision.
