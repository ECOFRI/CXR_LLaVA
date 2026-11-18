# F1 — CLEAR tokens + Qwen3, 30k training and fixed-dev evaluation

## 1. Experiment Information

Created UTC: 2026-11-18T14:58:39.413547+00:00. Project: CXR_LLaVA_Improvement. Parent commit: `ecb7fddbe709dc60432baed8b0f24b15da93422d`.
Work order: `plan/work_orders/WO_F1_2026-11-13.md`.
Execution status: **COMPLETE**; reason: **STOP_STAGE_A3**.
Classification: **metric-complete evaluation on fixed exposed dev cohorts; NOT test benchmark**.

## 2. Objective

Label Findings for 30k official-train studies, train the matched A2/A3/A4 arms once,
and evaluate on the original P4 500 MIMIC validation + 500 Indiana cases. No official
test or locked CheXpert test; no clinical validation, expert adjudication or diagnosis.

## 3. Survey / Technical Motivation

F0 established feasibility only. F1 reuses F0 pooling, CLEAR forward, bridge and LM
loss masking. [Logit adjustment](https://arxiv.org/abs/2007.07314) motivates the declared
binary prior-log-odds extension. Labeler and NegBio revisions are the pinned P4
versions. P4 text/clinical scoring and the vendored NLTK3.6.7 METEOR source are reused,
with the restored NLTK3.3 runtime and unchanged WordNet ZIP.

## 4. Environment

ict11, Quadro RTX4000, 8192MiB, driver550.163.01/CUDA12.4. Qwen env remains isolated:
Python3.11, torch2.6+cu124, Transformers4.51.3, PEFT0.15.2, Accelerate1.6.0.
CPU labeler: Python3.6.7, NumPy1.15.4, pandas0.23.4, NLTK3.3. Public missing assets
and pinned MKL2019.1/OpenMP/libffi libraries were restored to local model cache;
no existing package was upgraded. `labeler_assets_summary.json` records hashes.
Legacy library search paths are passed only to CPU children. Labeling uses16
processes on a16-CPU affinity mask, below the32-worker limit. Cache/output stays
on the shared storage volume; filesystem free space is not proof of personal quota.

## 5. Dependency Setup

The legacy env was restored independently; the task supplied missing public parser,
CoreNLP3.5.2, punkt/tagset and pinned dynamic libraries. Synthetic-labeler import and
one artificial sentence passed before any Findings labeling. No patient data was
sent to a model API, cloud notebook, package host or other external service.

## 6. Code / Notebook Changes

New `research/rrg_f1/` cohort/resume, labeler, model/data, extractor, train, inference,
evaluation, interventions, supervisor, status, tests and aggregate reporting.
F0 source helpers were factored for reuse while preserving default interfaces;
old raw F0/P4/P5 runs, checkpoints and reports were not edited. Private F1 outputs
and local path configuration are ignored. Only reviewed public source/config/docs
and aggregate evidence may be committed; never push from this task.

## 7. Experiment Configuration

Cohort: 30000 official-train studies, seed42, nonempty
Findings, one first available AP/PA image/study, patient overlap with P4 val=0.
Selection is uniform over eligible studies via a seeded permutation, not tuned on
validation. Exact official split/metadata linkage is checked. Clinical manifests,
raw Findings and labels remain private; public manifest summary contains counts/hashes.
MIMIC image content was treated as read-only; the F1 manifest does not constitute a
full pixel-content rehash. P4 Indiana expected image hashes are checked by evaluation.

BCE retains4 raw states. Positive=1; Negative/Blank=0; Uncertain excluded. Label columns
are name-aligned because Lung Opacity/Lung Lesion orders differ between F0 and labeler.
Train-only prior: (positives+1)/(non-uncertain+2); add log(p/(1-p)), tau1, to BCE logits.
BCE head reads mean projected tokens so its gradient reaches the projector.

A2: image projector + rank16 LoRA, no finding tokens/BCE. A3: A2+BCE. A4: A3+14 tokens
from image-only CLEAR zero-shot scores. Same seed/projector initialization/order and
hyperparameters, frozen FP16 Qwen/FP32 trainables, non-reentrant checkpointing.
Microbatch4, accumulation up to8 (max effective32), AdamW1e-4/eps1e-6/weight_decay0.01,
48-update warmup then cosine over960 planned updates. Flush at1000 samples and epoch end.
Findings target cap256 including EOS, text padding320. Greedy generation cap256, thinking off.
CLEAR448 input -> native1024 patches -> disk256 FP16 -> load64. F0 MIMIC val features are
reused; new train/Indiana extraction batch4 with8 image-loader workers.

Every1000 samples: first8 MIMIC + first8 Indiana dev reports, diversity/cap/repeated4gram
logged. Two consecutive bad checks (unique<4 OR cap>=12) stop the arm. Checkpoint saves
include every2000 samples and dev gates; optimizer/scaler/RNG/position and code hashes
are verified at resume. No additional epoch/arm or automatic OOM batch fallback.

## 8. Code Executed

Exact commands and resume instructions: `research/rrg_f1/README.md`.
Detached `nohup`/`setsid` supervisor uses a process lock, heartbeat and independent
GPU watchdogs: cumulative50400s (14h), each training arm cumulatively14400s (4h).
Loading/I/O/retries inside GPU children count toward the cap. Reservations survive
an interrupted guard conservatively; the ledger must never be reset to gain time.

## 9. Output

### Stage wall times

| Stage | Seconds | Status |
|---|---:|---|
| Cohort (checkpointed cumulative) | 639.4 | COMPLETE |
| Labeler 200-report gate | 18.7 | PASS |
| labeler_readiness | 10.1 | PASS |
| target_cache | 10.1 | PASS |
| train_labels | 1444.6 | PASS |
| encodings | 50.2 | PASS |
| features | 391.9 | STOP |
| labeler_readiness | 10.1 | PASS |
| target_cache | 10.1 | PASS |
| features | 2394.6 | PASS |
| A2 | 9348.8 | PASS |
| A3 | N/A | RUNNING |
| labeler_readiness | 255.1 | STOP |
| labeler_readiness | 10.0 | PASS |
| target_cache | 10.0 | PASS |
| A3 | 10.0 | STOP |
| labeler_readiness | 10.0 | PASS |
| target_cache | 10.0 | PASS |
| A3 | 9151.5 | PASS |
| A4 | 9871.8 | PASS |
| eval_A2 | 1220.2 | PASS |
| eval_A3 | 1670.3 | PASS |
| eval_A4 | 1370.2 | PASS |
| interventions | 260.0 | PASS |
| metrics | 590.1 | PASS |
| intervention_analysis | 30.0 | PASS |

200-report labeler projected time for30k: 2803.5s.
GPU charged total: **35634.1s /50400s = 9.898h**.
Authoritative budget is the private GPU ledger; supervisor wall times include heartbeat
polling overhead. Raw clinical outputs, case labels/features/checkpoints and worker
logs stay in `research/rrg_f1/private/`.

## 10. Output Analysis

| Arm | Status | Samples seen | Optimizer updates | Last LM / BCE loss |
|---|---|---:|---:|---|
| A2 | COMPLETE | 29992 | 959 | 1.3258 / 0.0000 |
| A3 | COMPLETE | 29992 | 959 | 1.3280 / 0.2407 |
| A4 | COMPLETE | 29992 | 959 | 1.3245 / 0.2405 |

Training loss is not clinical quality. Last-step values can be absent after a terminal
checkpoint recovery; raw optimizer logs remain private. Official full-report label
audit covers1000 studies, overall four-state agreement=0.7399.
This compares different text sections and can be dominated by blank agreement;
it is an audit only and never replaces Findings-specific labels.

## 11. Quantitative Results

### mimic

| Model | BLEU1 | BLEU2 | BLEU3 | BLEU4 | ROUGE-L | METEOR | Hist micro-F1 | Blank-neg micro-F1 | Hist paired/possible | F1 denominator |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.0893 | 0.0202 | 0.0076 | 0.0000 | 0.0899 | 0.0688 | 0.2167 | 0.0296 | 107/3500 | 120 |
| M1 | 0.1756 | 0.0935 | 0.0554 | 0.0346 | 0.1663 | 0.1399 | 0.4351 | 0.1284 | 353/3500 | 239 |
| A2 | 0.2273 | 0.1294 | 0.0817 | 0.0559 | 0.2047 | 0.1942 | 0.7238 | 0.3359 | 737/3500 | 420 |
| A3 | 0.2123 | 0.1224 | 0.0778 | 0.0533 | 0.2048 | 0.1958 | 0.7547 | 0.3237 | 689/3500 | 371 |
| A4 | 0.2498 | 0.1438 | 0.0923 | 0.0636 | 0.2116 | 0.2042 | 0.7409 | 0.3638 | 769/3500 | 467 |

### indiana

| Model | BLEU1 | BLEU2 | BLEU3 | BLEU4 | ROUGE-L | METEOR | Hist micro-F1 | Blank-neg micro-F1 | Hist paired/possible | F1 denominator |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.0643 | 0.0222 | 0.0101 | 0.0031 | 0.0852 | 0.0885 | 0.1839 | 0.0498 | 79/3500 | 87 |
| M1 | 0.2288 | 0.1291 | 0.0820 | 0.0515 | 0.2095 | 0.2124 | 0.1045 | 0.0424 | 463/3500 | 134 |
| M2 | 0.2954 | 0.1488 | 0.0824 | 0.0448 | 0.2084 | 0.2365 | 0.5448 | 0.2938 | 867/3500 | 268 |
| A2 | 0.2380 | 0.1379 | 0.0913 | 0.0615 | 0.2446 | 0.2369 | 0.5714 | 0.2072 | 891/3500 | 91 |
| A3 | 0.2284 | 0.1357 | 0.0900 | 0.0601 | 0.2439 | 0.2324 | 0.5833 | 0.2162 | 767/3500 | 96 |
| A4 | 0.2525 | 0.1454 | 0.0959 | 0.0644 | 0.2451 | 0.2415 | 0.5510 | 0.2069 | 882/3500 | 98 |


M0/M1 are the saved P4 BLIP-2 baselines; M2 is historical CXR-LLaVA on Indiana only.
`metrics.json` contains all per-class confusion counts, both-policy coverage/support,
F1 denominators, macro/micro F1, bootstrap intervals and paired differences. Undefined
F1 remains null/N/A. Clinical evaluation follows P4's7-class ontology, distinct from
14-label training. All500 outputs/dataset are retained; empty outputs score zero text
and blank clinical labels rather than being favorably dropped.

## 12. Qualitative Results

No patient images, reference text or generated examples are published here. Outputs
remain experimental. No expert image/report review or clinical correctness assessment.

### Collapse monitoring

| Arm | Samples | Unique /16 | Hit256 /16 | Repeated4gram fraction |
|---|---:|---:|---:|---:|
| A2 | 1000 | 13 | 9 | 0.5016 |
| A2 | 2000 | 8 | 0 | 0.0000 |
| A2 | 3000 | 9 | 0 | 0.0000 |
| A2 | 4000 | 11 | 0 | 0.0000 |
| A2 | 5000 | 11 | 0 | 0.0125 |
| A2 | 6000 | 11 | 6 | 0.2832 |
| A2 | 7000 | 9 | 0 | 0.0000 |
| A2 | 8000 | 15 | 3 | 0.1557 |
| A2 | 9000 | 10 | 0 | 0.0032 |
| A2 | 10000 | 9 | 0 | 0.0107 |
| A2 | 11000 | 8 | 3 | 0.1734 |
| A2 | 12000 | 11 | 5 | 0.2554 |
| A2 | 13000 | 11 | 0 | 0.0022 |
| A2 | 14000 | 14 | 0 | 0.0030 |
| A2 | 15000 | 12 | 0 | 0.0012 |
| A2 | 16000 | 12 | 0 | 0.0030 |
| A2 | 17000 | 9 | 0 | 0.0000 |
| A2 | 18000 | 12 | 0 | 0.0025 |
| A2 | 19000 | 11 | 2 | 0.1220 |
| A2 | 20000 | 8 | 0 | 0.0000 |
| A2 | 21000 | 10 | 0 | 0.0269 |
| A2 | 22000 | 10 | 0 | 0.0000 |
| A2 | 23000 | 11 | 0 | 0.0050 |
| A2 | 24000 | 9 | 0 | 0.0000 |
| A2 | 25000 | 10 | 0 | 0.0047 |
| A2 | 26000 | 9 | 0 | 0.0038 |
| A2 | 27000 | 10 | 0 | 0.0000 |
| A2 | 28000 | 9 | 0 | 0.0010 |
| A2 | 29000 | 9 | 0 | 0.0010 |
| A2 | 29992 | 9 | 0 | 0.0010 |
| A3 | 1000 | 9 | 9 | 0.5190 |
| A3 | 2000 | 5 | 0 | 0.0000 |
| A3 | 3000 | 9 | 0 | 0.0000 |
| A3 | 4000 | 4 | 0 | 0.0000 |
| A3 | 5000 | 9 | 0 | 0.0000 |
| A3 | 6000 | 13 | 2 | 0.1232 |
| A3 | 7000 | 10 | 0 | 0.0031 |
| A3 | 8000 | 13 | 2 | 0.1490 |
| A3 | 9000 | 14 | 0 | 0.0000 |
| A3 | 10000 | 12 | 1 | 0.0607 |
| A3 | 11000 | 10 | 0 | 0.0071 |
| A3 | 12000 | 8 | 4 | 0.2093 |
| A3 | 13000 | 12 | 0 | 0.0112 |
| A3 | 14000 | 9 | 0 | 0.0035 |
| A3 | 15000 | 13 | 0 | 0.0102 |
| A3 | 16000 | 11 | 0 | 0.0069 |
| A3 | 17000 | 10 | 0 | 0.0404 |
| A3 | 18000 | 14 | 0 | 0.0080 |
| A3 | 19000 | 11 | 2 | 0.1014 |
| A3 | 20000 | 13 | 0 | 0.0000 |
| A3 | 21000 | 13 | 0 | 0.0092 |
| A3 | 22000 | 13 | 0 | 0.0000 |
| A3 | 23000 | 12 | 0 | 0.0000 |
| A3 | 24000 | 10 | 2 | 0.0773 |
| A3 | 25000 | 13 | 0 | 0.0037 |
| A3 | 26000 | 13 | 2 | 0.0917 |
| A3 | 27000 | 13 | 2 | 0.0919 |
| A3 | 28000 | 14 | 2 | 0.0911 |
| A3 | 29000 | 14 | 1 | 0.0425 |
| A3 | 29992 | 14 | 1 | 0.0425 |
| A4 | 1000 | 13 | 1 | 0.1375 |
| A4 | 2000 | 7 | 0 | 0.0000 |
| A4 | 3000 | 7 | 0 | 0.0016 |
| A4 | 4000 | 6 | 0 | 0.0000 |
| A4 | 5000 | 13 | 0 | 0.0114 |
| A4 | 6000 | 11 | 0 | 0.0145 |
| A4 | 7000 | 9 | 1 | 0.0655 |
| A4 | 8000 | 13 | 5 | 0.3109 |
| A4 | 9000 | 12 | 0 | 0.0000 |
| A4 | 10000 | 12 | 3 | 0.1711 |
| A4 | 11000 | 11 | 0 | 0.0065 |
| A4 | 12000 | 7 | 2 | 0.1042 |
| A4 | 13000 | 11 | 0 | 0.0000 |
| A4 | 14000 | 11 | 0 | 0.0034 |
| A4 | 15000 | 15 | 0 | 0.0199 |
| A4 | 16000 | 12 | 0 | 0.0000 |
| A4 | 17000 | 11 | 0 | 0.0000 |
| A4 | 18000 | 14 | 0 | 0.0015 |
| A4 | 19000 | 12 | 0 | 0.0359 |
| A4 | 20000 | 11 | 0 | 0.0049 |
| A4 | 21000 | 8 | 0 | 0.0622 |
| A4 | 22000 | 11 | 0 | 0.0570 |
| A4 | 23000 | 11 | 0 | 0.0065 |
| A4 | 24000 | 14 | 1 | 0.0553 |
| A4 | 25000 | 11 | 1 | 0.0920 |
| A4 | 26000 | 13 | 0 | 0.0750 |
| A4 | 27000 | 12 | 1 | 0.0868 |
| A4 | 28000 | 11 | 1 | 0.0868 |
| A4 | 29000 | 14 | 1 | 0.0868 |
| A4 | 29992 | 14 | 1 | 0.0868 |

## 13. Comparison with Baseline

A2/A3/A4 share training cohort/order and hyperparameters. Comparisons to P4 or CXR-LLaVA
also change model, training scale, prompt and generation cap. MIMIC evaluation references
remain P4 Findings+Impression for direct scoring, while F1 trains Findings only. These
are exposed dev cohorts; Indiana baseline exposure and pretraining overlap remain.
The1000 patient-cluster bootstrap draws are paired across models, with unadjusted
percentile95% intervals; secondary contrasts are exploratory, not confirmatory claims.

A4 controls: first50 MIMIC + first50 Indiana, scores all0 and a seed42 shuffle across100.
No GT was used to define or prompt these controls. Direction concordance uses only
nonzero changes in CLEAR scores and definite generated-label changes; zero-shot scores
are uncalibrated proxies and this is not clinical causal validation.

- negative: changed reports 53/100; directional label changes 11 concordant / 2 opposite; paired definite class-cases 420.
- shuffled: changed reports 10/100; directional label changes 0 concordant / 2 opposite; paired definite class-cases 464.

## 14. Problems Encountered

Initial restored env lacked parser and dynamic libraries; repaired with matching public
assets before actual labeling. Original reports/run outputs were preserved. Any stop
reason, incomplete stage or collapsed arm is explicit above; no synthetic numbers or
post-hoc threshold/epoch selection fill missing results. Storage quota remains unknown.

## 15. Conclusion

COMPLETE: metric-complete evaluation on fixed exposed dev cohorts; NOT test benchmark. The experiment does not establish
clinical usefulness, accuracy on an independent test set or a benchmark improvement.

## 16. Next Experiment

Any additional arm, epoch, test evaluation or changed collapse rule needs a new scoped
work order and budget. Do not infer permission from this dev report or checkpoint.

## 17. Reproducibility Checklist

Private protocol lock pins scientific code/config; public config/aggregate hashes are
retained. CPU suite14/14 and root catalog2/2 passed before launch, as did syntax/import,
synthetic labeler and pinned WordNet checks. Runtime data/case outputs never enter Git.
Commit staging must be inspected and scanned for case identifiers and report text;
only public artifacts are committed on the current branch, with the requested coauthor
footer. No push.

## 18. Short Lab Update

Cohort30k and labeler gate prepared under WO_F1. Execution status=COMPLETE;
GPU=9.898h of14h. Read the arm table and fixed-dev metrics above, with
collapse/coverage/protocol limits preserved. Research only; no clinical claim.

## 19. Execution Addendum (manual, 2026-11-18)

- **Stale reason field:** `reason: STOP_STAGE_A3` above is a leftover from an earlier stop; the final supervisor status is COMPLETE with every stage PASS.
- **Infrastructure incident:** the first A3 attempt (14/11) died after ~3.570s when the GlusterFS `/storage` volume returned ENOSPC while writing the GPU ledger; later restarts hung on directory reads. This was a storage failure, not a model/protocol failure.
- **Ledger correction (user-approved, 18/11):** that crashed attempt's charge was set from 3.570s to 0 (original value kept in the ledger as `original_charged_seconds`). Without the refund the reported total would be 39.204s (10,89h), still under the 14h cap. A3 was retrained from scratch; no partial state was reused.
- **Relocation:** A3, A4, eval, interventions and metrics ran on a local-disk copy at `/home/student4/hanhpm/CXR_LLaVA_Improvement`. The protocol lock and work-order sha256 matched before launch; features, A2 checkpoint and encodings were copied unchanged (29.992 train + 500 + 500 features).
- **Evaluation input mirror:** `p4_evaluate.py` now resolves `/storage/...` inputs (500 MIMIC references, 500 Indiana images, 2 X00 CSVs) to a local mirror when one exists. Every file is still verified against the manifest sha256, so scored inputs are identical.

### Paired bootstrap Δ (95% CI, 1.000 patient-cluster draws)

| Contrast | Dataset | BLEU-4 | ROUGE-L | METEOR | Hist micro-F1 |
|---|---|---|---|---|---|
| A4 − A2 | MIMIC | [0,0015; 0,0124] | [0,0002; 0,0149] | [0,0038; 0,0187] | [−0,031; 0,074] |
| A4 − A2 | Indiana | [0,0008; 0,0051] | [−0,0024; 0,0034] | [0,0009; 0,0081] | [−0,131; 0,093] |
| A3 − A2 | MIMIC | [−0,0091; 0,0088] | [−0,0067; 0,0078] | [−0,0066; 0,0103] | [−0,023; 0,102] |
| A3 − A2 | Indiana | [−0,0043; 0,0015] | [−0,0044; 0,0031] | [−0,0097; 0,0006] | [−0,093; 0,111] |
| A4 − A3 | MIMIC | [−0,0034; 0,0145] | [0,0008; 0,0118] | [0,0013; 0,0166] | [−0,064; 0,033] |
| A2 − M1 | MIMIC | [0,0113; 0,0298] | [0,0272; 0,0543] | [0,0379; 0,0672] | [0,198; 0,386] |
| A2 − M1 | Indiana | [0,0037; 0,0166] | [0,0255; 0,0443] | [0,0134; 0,0351] | [0,322; 0,597] |

**Reading (exploratory, unadjusted, exposed dev cohorts):**
- The CLEAR finding tokens (A4) give a small but consistent text-metric gain over A2 on both cohorts; MIMIC BLEU-4 rises from 0,0559 to 0,0636.
- The auxiliary BCE loss alone (A3) shows no measurable text effect.
- None of the A-arm differences in clinical micro-F1 are resolved: the CIs are wide and the F1 denominators differ.
- All three Qwen arms clearly beat the P4 BLIP-2 M1 baseline.
- The A4 interventions are only weakly informative. Setting scores to zero changed 53/100 reports, with 11 concordant vs 2 opposite label flips. Shuffling changed only 10/100 reports, so A4 uses the token content weakly.
