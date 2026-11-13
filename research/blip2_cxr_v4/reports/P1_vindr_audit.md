# P1 Part C — VinDr-CXR audit (2026-10-22)

Status: PASS for label/split inventory and evaluator preparation; metric run BLOCKED by missing continuous scores and locked validation thresholds. CPU only; no model inference. Local dataset release version UNKNOWN. `LICENSE.txt` is the PhysioNet Credentialed Health Data License 1.5.0; this is a license version, not a dataset-version claim.

## Verified data

| Item | Count / format |
|---|---:|
| Training image labels | 45,000 reader rows for 15,000 unique images |
| Reader multiplicity | Exactly 3 rows per training image; 17 reader IDs observed |
| Test image labels | 3,000 consolidated rows for 3,000 images |
| Image-label classes | 28, binary 0/1 |
| Train/test image overlap | 0 |
| Bounding-box annotations | 69,052 train rows; 4,748 test rows |

The annotation schema is `image_id,rad_id,class_name,x_min,y_min,x_max,y_max` for train and omits `rad_id` on test. Train has 23 box classes; test has 22. Image-label train uses `Other diseases`, while test uses `Other disease`; the evaluator config maps both to the latter. Other ontology details and positive support appear in `manifests/vindr_ontology.csv`.

No paired free-text reports were found in the targeted local inventory. BLEU/ROUGE/METEOR are N/A for this label-only scope. No patient key was verified, so any later confidence interval is image-level and must be labeled accordingly.

## Evaluation contract

`configs/eval/vindr_classification.yaml` and `scripts/eval/evaluate_classification.py` accept long continuous scores (`image_id,finding,score`) and a threshold JSON whose `selection_split` is `validate` and `locked_before_test` is true. The evaluator rejects binary-only scores, duplicate reader labels, incomplete image/class score grids, ontology mismatch, and test-derived thresholds. It calculates per-class AUROC (tie-averaged ranks), AUPRC (non-interpolated average precision), precision/recall/F1 at locked thresholds, support and 1,000 image-level bootstrap intervals. It has not been run on VinDr because score and threshold artifacts are NOT_FOUND.

## Commands and checks

```bash
cd /storage/student4/hanhpm/CXR_LLaVA_Improvement
export TMPDIR=$PWD/research/blip2_cxr_v4/.cache/tmp HF_HOME=$PWD/research/blip2_cxr_v4/.cache/hf TORCH_HOME=$PWD/research/blip2_cxr_v4/.cache/torch PIP_CACHE_DIR=$PWD/research/blip2_cxr_v4/.cache/pip XDG_CACHE_HOME=$PWD/research/blip2_cxr_v4/.cache NLTK_DATA=$PWD/research/blip2_cxr_v4/.cache/nltk
research/envs/cxr-clear-cpu/bin/python research/blip2_cxr_v4/scripts/eval/evaluate_classification.py --config research/blip2_cxr_v4/configs/eval/vindr_classification.yaml --dry-run
research/envs/cxr-clear-cpu/bin/python research/blip2_cxr_v4/tests/test_vindr_classification.py
```

Dry run reported labels present, scores absent, thresholds absent; tests and syntax checks passed. Next: P1 Part D NIH/CheXpert location audit.
