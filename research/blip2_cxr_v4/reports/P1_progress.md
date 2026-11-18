# P1 Part A — Indiana X00 progress

## 1. Experiment Information

- **Experiment ID:** X00 Indiana saved-output re-score
- **Date:** 2026-10-22
- **Researcher:** Codex; Claude QA
- **Project:** CXR_LLaVA_Improvement
- **Status:** PARTIAL (METEOR unavailable offline); CPU scoring completed
- **Classification:** local metric re-score, not a paper reproduction or clinical validation

## 2. Objective

Re-score all 3667 valid saved Indiana predictions without model inference.

## 3. Survey / Technical Motivation

Historical local macro-F1 0.4457087416797642 used official CheXpert labeler revision `44ddeb363149aa657296237f18b5472a73c1756f` with NegBio revision `073199e2792824740e89844a59c13d3d40ce4d23`. The seven classes are Cardiomegaly, Consolidation, Edema, Lung Opacity, Pleural Effusion, Pneumonia, and Pneumothorax. The `historical` policy excludes blank and uncertain values from each class joint-definite mask; the three sensitivity policies map uncertainty or blank explicitly. A consistent corpus lexical score and explicit sensitivity policies were needed.

## 4. Environment

| Component | Value |
|---|---|
| Host | ictserver6, Linux 4.19.0-14-amd64 |
| Interpreter | `research/envs/cxr-clear-cpu/bin/python` (Python 3.10.22) |
| Device | CPU only; no model loaded |
| Dependencies | Python standard library; saved CheXpert labels; NLTK/WordNet unavailable |

## 5. Dependency Setup

No installation or download. Cache variables pointed into `research/blip2_cxr_v4/.cache/`.

## 6. Code / Notebook Changes

Added `scripts/eval/evaluate_reports.py`, `src/vlm_medical/eval/{text_metrics,clinical_metrics,bootstrap}.py`, config and three focused tests. Only V4 files changed.

## 7. Experiment Configuration

| Parameter | Value |
|---|---|
| Source | `result/indiana_full_k80_20261004_031940_1252388/inference/predictions.csv` |
| N | 3667 source; 3667 used; 0 excluded |
| Bootstrap | 1,000 paired report/study resamples, seed 42, percentile 95% CI |
| Tokenization | lowercase ASCII alphanumeric regex; punctuation stripped; no stemming |
| Clinical policies | historical, uncertain positive, uncertain negative, blank negative |

## 8. Code Executed

```bash
cd /storage/student4/hanhpm/CXR_LLaVA_Improvement
export TMPDIR=$PWD/research/blip2_cxr_v4/.cache/tmp HF_HOME=$PWD/research/blip2_cxr_v4/.cache/hf TORCH_HOME=$PWD/research/blip2_cxr_v4/.cache/torch PIP_CACHE_DIR=$PWD/research/blip2_cxr_v4/.cache/pip XDG_CACHE_HOME=$PWD/research/blip2_cxr_v4/.cache NLTK_DATA=$PWD/research/blip2_cxr_v4/.cache/nltk
research/envs/cxr-clear-cpu/bin/python research/blip2_cxr_v4/scripts/eval/evaluate_reports.py --config research/blip2_cxr_v4/configs/eval/indiana_x00.yaml
```

## 9. Output

Raw aggregate output: `COMPLETE`, N used 3,667, N excluded 0, historical macro-F1 0.4457087416797642. Full metrics, per-case hash-keyed data and run manifest are in research/blip2_cxr_v4/runs/X00_indiana_rescore_20261022T062016Z. No report text or raw identifiers are printed here.

## 10. Output Analysis

- Data integrity PASS: all 3,667 prediction reference texts equal the frozen selection cohort and saved GT/generated label text; SHA-256 of cohort and labels matches historical summary.
- All status values are success; no empty reference or generated text; report and image keys unique. Generated text has 2,002 unique values; reference text has 2,931.
- All selected views are frontal_unspecified. Reference composition: 3,193 findings+impression, 468 impression only, 6 findings only.
- No patient identifier is present in this IU cohort. `report_id` identifies the report/study and is the bootstrap unit.
- Auxiliary `data/indiana_all_candidates_seed42.csv` overlaps 3,273 selected report/image pairs, with 3,271 equal texts; it is not the frozen reference source and needs P1 follow-up.

## 11. Quantitative Results

| Metric | Value | 95% CI |
|---|---:|---:|
| bleu1 | 0.2879 | [0.2829, 0.2927] |
| bleu2 | 0.1468 | [0.1437, 0.1496] |
| bleu3 | 0.0818 | [0.0796, 0.0840] |
| bleu4 | 0.0449 | [0.0432, 0.0465] |
| rouge_l | 0.2079 | [0.2052, 0.2104] |
| METEOR | N/A | N/A |

| Policy | Micro-F1 (95% CI) | Macro-F1 (95% CI) |
|---|---:|---:|
| historical | 0.5342 [0.5029, 0.5618] | 0.4457 [0.3921, 0.4853] |
| uncertain_as_positive | 0.5403 [0.5118, 0.5659] | 0.4570 [0.4210, 0.4899] |
| uncertain_as_negative | 0.5111 [0.4813, 0.5380] | 0.3959 [0.3475, 0.4341] |
| blank_as_negative | 0.2903 [0.2703, 0.3096] | 0.1931 [0.1712, 0.2143] |


Historical per-pathology counts and intervals:

| Pathology | TP | FP | FN | TN | F1 (95% CI) | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Cardiomegaly | 295 | 343 | 128 | 674 | 0.5561 [0.5182, 0.5914] | 0.393 |
| Consolidation | 5 | 73 | 6 | 254 | 0.1124 [0.0247, 0.2062] | 0.092 |
| Edema | 11 | 9 | 6 | 69 | 0.5946 [0.3783, 0.7541] | 0.026 |
| Lung Opacity | 123 | 55 | 9 | 34 | 0.7935 [0.7400, 0.8409] | 0.060 |
| Pleural Effusion | 67 | 151 | 44 | 2119 | 0.4073 [0.3382, 0.4742] | 0.649 |
| Pneumonia | 16 | 7 | 15 | 38 | 0.5926 [0.4072, 0.7420] | 0.021 |
| Pneumothorax | 2 | 38 | 21 | 1631 | 0.0635 [0.0000, 0.1563] | 0.461 |

Historical macro-F1 reproduced exactly to displayed precision; all seven pathology TP/FP/TN/FN and F1 values match saved `metrics/pathology_metrics.csv`. AUROC is N/A because no continuous scores were evaluated.

## 12. Qualitative Results

Not included: this CPU re-score performs no case review, and reports contain no patient text.

## 13. Comparison with Baseline

The historical 0.4457 macro-F1 was reproduced under the same joint-definite policy. New BLEU/ROUGE values use the recorded tokenizer and are not directly comparable with a paper protocol unless tokenizer, split and reference rules match.

## 14. Problems Encountered

- METEOR blocked: NLTK WordNet unavailable offline; no download was made.
- Patient-level CI blocked: source lacks patient key; report/study bootstrap used and labeled.
- The selected 3,667-case cohort differs from the paper cohort; no paper reproduction claim.

## 15. Conclusion

Part A CPU scoring completed on 3,667 saved reports. Historical macro-F1 was reproduced; corpus BLEU and ROUGE-L were calculated with explicit normalization and study-level confidence intervals. METEOR remains N/A, and these scores are research metrics, not clinical findings.

## 16. Next Experiment

Stop after Part A. Claude review precedes Part B MIMIC linkage work.

## 17. Reproducibility Checklist

- [x] Historical commit/model revision, config and source SHA-256 recorded in run manifest
- [x] Exact command, interpreter, seed and bootstrap unit recorded
- [x] Raw aggregate outputs and per-case hash-keyed CSV saved
- [x] No model inference, GPU work, install or download
- [ ] METEOR available offline
- [ ] Patient-level grouping verified

## P1 Part B checkpoint

Status PARTIAL. `timeout 60 research/envs/cxr-clear-cpu/bin/python research/blip2_cxr_v4/scripts/data/audit_mimic.py --config research/blip2_cxr_v4/configs/eval/mimic_p1.yaml --output-dir research/blip2_cxr_v4/manifests` completed on CPU with the cache environment above. Metadata N=377,110 images; sampled linkage 50/50 reports and 48/50 JPGs. The repository 3,000-path testset is mostly official train, so it is excluded from test evaluation. Source, method, limitations and next step are in `reports/P1_mimic_audit.md`. Next: Part C VinDr.

## P1 Part C checkpoint

Status PARTIAL for evaluation readiness; inventory PASS. `research/envs/cxr-clear-cpu/bin/python research/blip2_cxr_v4/scripts/eval/evaluate_classification.py --config research/blip2_cxr_v4/configs/eval/vindr_classification.yaml --dry-run` reported the 3,000-row test labels present and scores/locked thresholds absent. The 15,000-image training set has three reader rows per image; train/test image overlap is zero. BLEU N/A. Exact source paths, methods, license and limitations: `reports/P1_vindr_audit.md`; next Part D.

## P1 Part D checkpoint

Status PASS for location, PARTIAL for full data availability. Targeted `timeout 20 ls` checks of existing repo-specified paths and Python CSV counts verified NIH local N=4,999 (3,783/344/872) and CheXpert test N=668 images (518 frontal, 150 lateral), 500 study references. No images, model, or test labels were scored. Dataset provenance, split locks, path checks and exact search scope: `reports/P1_nih_chexpert_locations.md`; next Part E.

## P1 Part E checkpoint

Status PASS for candidate metadata and dry-run; download NOT_STARTED. Exact command: `timeout 20 research/blip2_cxr_v4/scripts/download/download_checkpoint.sh --dry-run`. Environment: direct `/home/student4/anaconda3/envs/cxr-llava-k80/bin/python` with VR-local TMPDIR/HF_HOME/TORCH_HOME/PIP_CACHE_DIR/XDG_CACHE_HOME/NLTK_DATA, torch 1.12.1+cu102, transformers 4.36.2; config `configs/download/blip2_opt_2_7b.yaml`. Dataset path/N N/A (checkpoint metadata only); HF model ID Salesforce/blip2-opt-2.7b, selected 13.9551 GiB, SHA pinned. Result CI N/A (no metric run). Limitations: actual K80 fit, inference quality and external paper checkpoint access UNKNOWN. Next: Claude reviews candidate; no download until separate approval. See `reports/P1_blip2_candidates.md`, `runs/P1_blip2_candidate_dry_run/download_manifest.json`, `reports/03_paper_evidence_matrix.md`.
