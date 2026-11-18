# CheXpert Table 3 K80 evaluation — matched 518-image cohort

## 1. Experiment Information

- Experiment ID: `inference`; date: 2026-10-06; researcher: student4 / project team.
- Repository: https://github.com/hanhpm/CXR_LLaVA_Improvement; commit: `f6535c6d9d7eb23c2e9a2ac24f48771a1d276efc`.
- Script: `scripts/chexpert_table3_k80.py`; status: completed local evaluation, research only.

## 2. Objective

Evaluate the released CXR-LLaVA v2 checkpoint on the official CheXpert frontal test images with 14 direct yes/no questions per image.

## 3. Survey / Technical Motivation

Paper Table 3: https://arxiv.org/pdf/2310.18341v3. CheXlocalize test images: https://github.com/rajpurkarlab/cheXlocalize. Public consensus labels: https://github.com/rajpurkarlab/cheXpert-test-set-labels. The paper says it directly generated binary labels, but does not expose the exact prompt or decoding code; this evaluation freezes a local prompt in `run_config.json`.

## 4. Environment

Four Tesla K80 GPUs, FP32, torch 1.12.1+cu102, transformers 4.36.2, model revision `b2224786bb90d54b1e1291171866706cfbb44e2b`. Exact package freeze and source snapshot are under `/storage/student4/hanhpm/CXR_LLaVA_Improvement/result/chexpert_table3_k80_20261005_full01/inference/snapshots/`.

## 5. Dependency Setup

Reused `cxr-llava-k80`; no model weights were changed. K80 adaptation changes BF16 casts to FP32 and dispatches the model across four devices.

## 6. Code / Notebook Changes

Added a direct binary-question runner with per-question checkpointing, strict image/study label alignment, transparent parse coverage, bootstrap metrics and example-image archive.

## 7. Experiment Configuration

668 downloaded CheXpert test images comprise 518 frontal and 150 lateral. Only 518 frontal images were used. Their 14 labels agree with `groundtruth.csv` per study, and their positive/negative support matches paper supplementary Table 3 distribution. Fourteen locally chosen yes/no questions per image, helper defaults temperature 0.2/top_p 0.8, per-image-and-label SHA256-derived seed, 1000 image-level bootstrap replicates.

## 8. Code Executed

`python -m scripts.chexpert_table3_k80 --run --output /storage/student4/hanhpm/CXR_LLaVA_Improvement/result/chexpert_table3_k80_20261005_full01/inference` then `python -m scripts.chexpert_table3_k80 --evaluate --output /storage/student4/hanhpm/CXR_LLaVA_Improvement/result/chexpert_table3_k80_20261005_full01/inference`.

## 9. Output

Raw responses/checkpoints: `/storage/student4/hanhpm/CXR_LLaVA_Improvement/result/chexpert_table3_k80_20261005_full01/inference/cases/`; metrics: `metrics.csv`; parse/error review: `review_queue.csv`. Example images: `/storage/student4/hanhpm/CXR_LLaVA_Improvement/reports/chexpert_table3_example_images_inference`; ZIP: `/storage/student4/hanhpm/CXR_LLaVA_Improvement/reports/chexpert_table3_example_images_inference.zip`.

## 10. Output Analysis

Completed 518/518 images; 6950/7252 explicit yes/no answers parsed. Unparsed answers are missing predictions, never converted to negatives. 1113 parsed disagreements across seven shown pathologies; inspect review queue and images before clinical interpretation.

## 11. Quantitative Results

Macro F1 over 5/5 defined paper-primary findings: 0.249; paper reported 0.57. Mean latency: 188.495 seconds/image; peak allocated VRAM on a single device: 8.861 GiB.

| Pathology | GT +/− | TP/FP/TN/FN | Precision | Recall | F1 (95% CI) | Paper F1 |
|---|---:|---:|---:|---:|---:|---:|
| Cardiomegaly | 154/364 | 27/44/320/127 | 0.380 | 0.175 | 0.240 (0.168, 0.314) | 0.620 |
| Lung Opacity | 272/246 | 37/10/236/231 | 0.787 | 0.138 | 0.235 (0.171, 0.298) | 0.840 |
| Edema | 79/439 | 4/30/404/75 | 0.118 | 0.051 | 0.071 (0.017, 0.145) | 0.500 |
| Consolidation | 29/489 | 6/51/438/23 | 0.105 | 0.207 | 0.140 (0.048, 0.240) | 0.240 |
| Atelectasis | 158/360 | 91/96/263/67 | 0.487 | 0.576 | 0.528 (0.459, 0.590) | 0.690 |
| Pleural Effusion | 105/413 | 19/21/383/84 | 0.475 | 0.184 | 0.266 (0.168, 0.359) | 0.630 |
| Support Devices | 266/252 | 29/23/226/231 | 0.558 | 0.112 | 0.186 (0.124, 0.245) | 0.780 |

All 14 findings, accuracy/specificity, support, coverage and bootstrap CI are in `metrics.csv`. The paper's average-F1 aggregation is not independently verified.

## 12. Qualitative Results

`chexpert_table3_example_images_inference/index.csv` pairs selected FP/FN cases with original images and model text. This is an error-review bundle, not radiologist adjudication.

## 13. Comparison with Baseline

The paper used 518 CheXpert images and reported F1 of 0.57 over five principal pathologies. Local cohort identity and GT support match. Prompt, decoding, dtype and exact metric aggregation are not verified as identical, so F1 differences are descriptive.

## 14. Problems Encountered

AzCopy created an extra `test/` directory in the local target; path resolution accounted for this without altering source files. Exact Table 3 prompt and decoding source were unavailable.

## 15. Conclusion

The released model was evaluated on the same-count, matching-label 518 frontal CheXpert test images, with direct binary-question responses and explicit parse coverage. This is a matched-cohort local Table 3 approximation, not an exact paper reproduction.

## 16. Next Experiment

- [ ] Review FP/FN examples with a qualified radiologist before making image-level claims.
- [ ] Obtain the authors' exact prompt/decoding and aggregation code if available.
- [ ] Separately review Indiana FP/FN/omissions and the eight NLM view ambiguities; do not alter its frozen cohort.

## 17. Reproducibility Checklist

- [x] 518 frontal image paths, study labels, source and model hashes, seed, prompt and raw outputs captured.
- [x] Counts and GT support match paper supplementary distribution; metrics and example ZIP saved.
- [ ] Exact Table 3 generation prompt/decoding and independent radiologist review verified.

## 18. Short Lab Update

> Matched the 518-image CheXpert frontal cohort, generated direct yes/no predictions for 14 findings on four K80s, measured 6950/7252 parse coverage and five-finding macro F1 0.249; exact paper prompt remains unresolved.
