# P5 execution progress

## P5_qformer_20261025 launch — 2026-10-25

Status: **P5.1 RUNNING** under detached `tmux` session `vlm_p5_20261025`; supervisor PID 1223796 at launch, initial P5.1 child PID 1223830. Frozen P5 contract SHA-256 `841753a63743c62fe15be5cc2f9f2308a40d593668a891050da18f8e213475ce`. Preflight verified P4 config/index/adapter/metrics hashes, all three manifest hashes, pinned 107,130,880 trainable-parameter shape metadata, and an idle single K80. Six P5 CPU tests, three P4 loss-parity tests and four stage dry-runs passed. No P5 result is claimed from launch; consult `runs/P5_qformer_20261025/status.json`, stage status/budget JSON and logs for live evidence. P5.2/P5.3 remain conditional on machine-verified prior gates.

## P5_qformer_20261025 final BLOCKED

Run: `P5_qformer_20261025`; scientific configuration SHA-256 `841753a63743c62fe15be5cc2f9f2308a40d593668a891050da18f8e213475ce`.
Status: **BLOCKED**. P4 artifacts remain unchanged.
- P5.1: PASS; updates/generations=20; peak reserved GiB=9.939453125
- P5.2: PASS; updates/generations=2000; peak reserved GiB=9.943359375
- P5.3: PASS; updates/generations=1000; peak reserved GiB=7.6953125
Stop reason: RuntimeError: CPU_ANALYSIS process exit 1; inspect /storage/student4/hanhpm/CXR_LLaVA_Improvement/research/blip2_cxr_v4/runs/P5_qformer_20261025/CPU_ANALYSIS.stderr.log


## P5_qformer_20261025 final COMPLETE

Run: `P5_qformer_20261025`; scientific configuration SHA-256 `841753a63743c62fe15be5cc2f9f2308a40d593668a891050da18f8e213475ce`.
Status: **COMPLETE**. P4 artifacts remain unchanged.
- P5.1: PASS; updates/generations=20; peak reserved GiB=9.939453125
- P5.2: PASS; updates/generations=2000; peak reserved GiB=9.943359375
- P5.3: PASS; updates/generations=1000; peak reserved GiB=7.6953125
indiana: paired M3−M1 BLEU-4 -0.05151685011825502, 95% CI [-0.0574076372468186, -0.04620050574398297]; clinical proxies require caution.
mimic: paired M3−M1 BLEU-4 -0.0326666025429177, 95% CI [-0.04494628927985667, -0.026138784306282362]; clinical proxies require caution.


## P5 scientific closeout — 2026-10-26

Execution **COMPLETE**: P5.1 20/20 updates, 193.16 s, peak reserved 9.939 GiB; P5.2 2,000/2,000 updates, one epoch, 9,849.69 s cumulative, peak 9.943 GiB, verified checkpoint; P5.3 1,000/1,000 paired new generations, 9,546.18 s, peak 7.695 GiB. The earlier storage interruption was charged conservatively and resumed from the last verified step. The CPU analysis used saved CheXpert output and the pinned P4 METEOR scorer/WordNet; no P4 predictions were regenerated.

| Cohort (500 each) | P4 M1 BLEU-4 | P5 M3 BLEU-4 | Paired Δ M3−M1; 95% CI |
|---|---:|---:|---:|
| MIMIC validation | 0.034638 | 0.001971 | −0.032667 [−0.044946, −0.026139] |
| Indiana | 0.051517 | 0.000000 | −0.051517 [−0.057408, −0.046201] |

P5 M3 produced only **two distinct report texts per cohort**; all 500 outputs per cohort reached the 128-token generation limit. The two repetitive patterns are nonclinical letter lists. Historical joint-definite clinical micro-F1 is **N/A** for M3 because its labeler outputs have zero definite paired labels; it must not be reported as zero. Under the GT-definite/prediction-missing-negative sensitivity policy, micro-F1 fell from 0.1372 to 0 on MIMIC and from 0.0488 to 0 on Indiana. Text/labeler omission candidates rose from 606 to 658 on MIMIC and 202 to 209 on Indiana (class-case counts); these are not radiologist-confirmed errors. Zero unsupported-positive candidates for M3 reflect report collapse, not clinical improvement.

**Decision: HOLD further Q-Former training/inference with this configuration.** The execution gates passed, but the matched scientific comparison strongly disfavors P5 M3 versus P4 M1. Investigate collapse using saved artifacts and CPU-only diagnostics before proposing any new GPU run; any changed optimization or architecture requires a new protocol and approval. Full per-class counts, coverage, lexical metrics, bootstrap and error proxies: `runs/P5_qformer_20261025/P5.3_cpu_analysis/metrics.json`, `error_pairing.json`; machine-readable decision: `P5_final_decision.json`. No P4 artifact was changed.


Google Docs sync VERIFIED: VLM Medical → Report 2/11 → P4 progress 19/10–02/11, mục “P5 — Q-Former: kết quả cuối (26/10/2026)”; đã đọc lại bảng BLEU-4 và hình. Tab ID `t.og7ix2n9u1zd`; biểu đồ cục bộ `runs/P5_qformer_20261025/P5.3_cpu_analysis/bleu4_comparison.png`.
