# Experiment Analysis — E4-R4

## 1. Experiment Information

E4-R4; 2026-10-17T00:46:13.450814+07:00; CXR_LLaVA_Improvement; branchmaster; commitf6535c6d9d7eb23c2e9a2ac24f48771a1d276efc; statusPASS_E4_TECHNICAL_PILOT; repository https://github.com/hanhpm/CXR_LLaVA_Improvement. Technical exploratory pilot; not clinical validation.

## 2. Objective

Reuse2 verified reports and execute at most78 missing frozen configurations; evaluate only a complete80-key grid.

## 3. Survey / Technical Motivation

Reviewed original E4/R1/R2/D1/R3 and frozen G2 protocols. D1 passed one fresh C1; sequential memory stability and image-specific CLEAR benefit remained unproven.

## 4. Environment

Linux4.19.0-14-amd64; four dynamically eligible K80s['GPU-27c3b5ab-b82d-579b-14da-ceece5dfaa16', 'GPU-64ba6275-3071-8d64-069d-3c23552c6764', 'GPU-7c841195-e5a5-796c-f50e-06b4028639ed', 'GPU-9bdf39c7-917e-ba7f-73e5-ee97a9e87083']; python pinned cxr-llava-k80; torch1.12.1+cu102, transformers4.36.2, CUDA10.2. Full config/preflight archived.

## 5. Dependency Setup

Reused offline pinned environment. No installs/downloads or model/checkpoint edits.

## 6. Code / Notebook Changes

New R4-only supervisor/atomic store/worker/merge-aware evaluator. R2 guard copied byte-for-byte. CPU fault tests precede model loading. Read-only sparse phase hooks do not mutate model math, tensors, RNG or prompts. No original files edited.

## 7. Experiment Configuration

Frozen20 frontal Indiana images; four arms C0/C1/C2a/C3; FP32 original four-device map; batch1; same per-case seed, temperature0.2/top_p0.8/top_k50/max512/use_cache/native stop. 90min/64MiB, no retries, used guard10624MiB/free768MiB plus approved pressure trend/budget stops.

## 8. Code Executed

`PYTHONDONTWRITEBYTECODE=1 python3 research/runs/20261017_E4_R4_controlled_resume_v1/execute.py --approval EXPLICIT_E4_R4_MAX78` (one invocation only).

## 9. Output

Attempts new78; success new78; reused2; unique80/80. Stop. Raw records/telemetry/logs retained.

## 10. Output Analysis

Technical completion does not show CLEAR improved generation. Inspect frozen paired contrasts, support and coverage; automated labels reflect report disagreement, not clinical ground truth.

## 11. Quantitative Results



| UUID | Peak whole-device used MiB | Minimum free MiB | Peak worker-process MiB (CLI rounded) | Max observed torch allocated MiB | Max observed torch reserved MiB |
|---|---:|---:|---:|---:|---:|
| `GPU-27c3b5ab-b82d-579b-14da-ceece5dfaa16` | 10228.5000 | 1212.6875 | 10215 | 9435.3056640625 | 9828.0 |
| `GPU-64ba6275-3071-8d64-069d-3c23552c6764` | 7992.5000 | 3448.6875 | 7979 | 7235.318359375 | 7602.0 |
| `GPU-7c841195-e5a5-796c-f50e-06b4028639ed` | 7992.5000 | 3448.6875 | 7979 | 7235.318359375 | 7602.0 |
| `GPU-9bdf39c7-917e-ba7f-73e5-ee97a9e87083` | 7992.5000 | 3448.6875 | 7979 | 7235.333984375 | 7602.0 |

[
  {
    "arm_id": "C0",
    "macro_precision": "0.08333333333333333",
    "defined_precision_findings": "3",
    "macro_recall": "0.5",
    "defined_recall_findings": "1",
    "macro_f1": "0.1111111111111111",
    "defined_f1_findings": "3",
    "macro_coverage": "0.2",
    "defined_coverage_findings": "5",
    "total_image_finding_pairs": "100",
    "joint_definite_pairs": "20",
    "missing_or_uncertain_pairs": "80"
  },
  {
    "arm_id": "C1",
    "macro_precision": "1.0",
    "defined_precision_findings": "1",
    "macro_recall": "0.3333333333333333",
    "defined_recall_findings": "1",
    "macro_f1": "0.5",
    "defined_f1_findings": "1",
    "macro_coverage": "0.11000000000000001",
    "defined_coverage_findings": "5",
    "total_image_finding_pairs": "100",
    "joint_definite_pairs": "11",
    "missing_or_uncertain_pairs": "89"
  },
  {
    "arm_id": "C2a",
    "macro_precision": "0.25",
    "defined_precision_findings": "2",
    "macro_recall": "1.0",
    "defined_recall_findings": "1",
    "macro_f1": "0.3333333333333333",
    "defined_f1_findings": "2",
    "macro_coverage": "0.04",
    "defined_coverage_findings": "5",
    "total_image_finding_pairs": "100",
    "joint_definite_pairs": "4",
    "missing_or_uncertain_pairs": "96"
  },
  {
    "arm_id": "C3",
    "macro_precision": "0.5",
    "defined_precision_findings": "2",
    "macro_recall": "1.0",
    "defined_recall_findings": "1",
    "macro_f1": "0.5",
    "defined_f1_findings": "2",
    "macro_coverage": "0.02",
    "defined_coverage_findings": "5",
    "total_image_finding_pairs": "100",
    "joint_definite_pairs": "2",
    "missing_or_uncertain_pairs": "98"
  }
]

## 12. Qualitative Results

See fixed-hash E4_R4_qualitative_review.md. Experimental model text only; no radiologist adjudication or assessment of clinical correctness.

## 13. Comparison with Baseline

C0/C1 original text preserved. Frozen four-arm contrasts remain unchanged; technical success alone is not evidence of CLEAR benefit. Technical completion does not show CLEAR improved generation. Inspect frozen paired contrasts, support and coverage; automated labels reflect report disagreement, not clinical ground truth.

## 14. Problems Encountered

No execution safety interruption observed; exact logs and current remaining keys retained; no automatic fix/retry.

## 15. Conclusion

PASS_E4_TECHNICAL_PILOT; 80 unique reports retained. Original artifacts unchanged. Technical completion does not show CLEAR improved generation. Inspect frozen paired contrasts, support and coverage; automated labels reflect report disagreement, not clinical ground truth. Scientific and complex-fusion holds remain.

## 16. Next Experiment

None executed or automatically scheduled. Any further inference requires separate scoped authorization; review saved evidence first.

## 17. Reproducibility Checklist

Recorded git branch/commit, pinned source and output hashes, model/environment, frozen cohort/prompt/evidence/seed, raw events/telemetry, allocator measurements, ledger/grid separation, errors, resource bounds and decision. Undefined labels/metrics remain undefined.

## 18. Short Lab Update

Objective: bounded78-call missing-grid resume. Method: immutable C0/C1 references and atomic new commits under R2 guard. Result:PASS_E4_TECHNICAL_PILOT, new78/78, unique80/80. Evaluator:PASS. Limitation:20 pre-exposed images, unverified patient independence, no clinical claims. No next gate launched.

### Full-grid interpretation

## Observed interpretation after full-grid freeze

C2a macro-F1 was0.3333 versusC0=0.1111, C1=0.5000 andC3=0.5000. However, joint-definite coverage was only4% forC2a versus20%,11%,2% respectively. Defined-F1 finding counts differ (C0=3,C1=1,C2a=2,C3=2), so the macro ranks are not directly comparable on a common supported class set. This does not establish an improvement.

There were only4 positive reference-label records, all Cardiomegaly; the other four findings had zero definite reference positives. C2a's Cardiomegaly F1=0.6667 versusC0=0.3333 is an apparently favorable conditional observation, but only2/20 Cardiomegaly pairs are jointly definite forC2a versus7/20 forC0. C2a's observed FN=0 does not establish reduced omissions:3/4 reference positives have unknown predictions, and the omission upper bound remains75% for all four arms.

C2a had93/100 unknown predictions (92blank,1uncertain), compared with52/100 forC0,77/100 forC1 and95/100 forC3. These states are retained, not mapped to negative. C2a's discordant-positive rate is2/(1+2)=66.7%, based on three jointly definite predicted-positive pairs; C0=83.3%, C1=0%, C3=50%, all with sparse/unequal support. This is automated report disagreement, not a clinically verified hallucination rate.

Most importantly for image specificity, C2a and shuffled-C3 report text was exactly identical in17/20 images. Both had median33 characters, versus225 forC0 and143 forC1; C1/C2a text matched in3/20. These whole-cohort text controls are descriptive and use exact equality, without outcome-based case selection. The fixed four qualitative groups also contain generic short reports and differing disease statements, which have not been clinically adjudicated. Useful image-specific CLEAR benefit is **not established**. No prompt/threshold/evaluator change or follow-up experiment was made from this finding.
