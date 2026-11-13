# E1.3 TRAIN similarity and ridge analysis

## 1. Experiment Information

2026-10-15; existing G0 continued under explicit independent-resource-gate authorization. New bundle `research/runs/20261015T005545_V3_E13_RESOURCE_REFINEMENT`; CPU-only foreground supervised commands. Active V3 plus user clarification, no automatic E5 benchmark.

## 2. Objective

Assess TRAIN feature geometry using E1.3, and refine E5 compute feasibility without evaluating classification performance.

## 3. Survey / Technical Motivation

V3 active; V2 historical/standing safety. Original E1/E2 test outputs never guide modeling choices. Low or high CKA cannot establish predictive complementarity.

## 4. Environment

Existing cxr-clear-cpu; package versions/source hashes in run manifests. NumPy1.26.4, SciPy1.13.1, sklearn1.5.2, matplotlib3.9.2. No GPU frameworks loaded; CUDA_VISIBLE_DEVICES empty.

## 5. Dependency Setup

No dependency installation or environment mutation. No full system inventory repeated. Fresh resource availability and64MiB allocation probe passed; numeric account quota unavailable.

## 6. Code / Notebook Changes

New research/v3_followup.py and v3_followup_guard.py; reused old loader/fold/scaling helpers unchanged. There is no E5 benchmark entrypoint in the new runner. Syntax checked; old component checks and in-run R² sufficient-statistic algebra checks passed.

## 7. Experiment Configuration

NIH exploratory; potential CXR encoder pretraining overlap; not independent external test. TRAIN3783images/1222patients only in calculations. Three original patient folds:1261images each,407/407/408patients. Original train/validation identities/support verified by G0; validation/test not evaluated here. Cached1024-D CXR and768-D CLEAR, native preprocessing differences retained. NPZ hashing/decompression transports full immutable members; only selected TRAIN rows enter numeric analyses.

## 8. Code Executed

Exact command arrays in each run_manifest.json; replay commands saved in reproduce_commands.md. Preflight -> E1.3 -> E5 profiling ran sequentially; no detached jobs or recurring scheduler.

## 9. Output

E1.3 COMPLETE. Fresh gate estimate200.901s(<60min); actualE1.3wall 14.089645s, peakRSS 0.483589GiB,0GPUhours. 12ridge fits; all outputs finite. Source/protocol unchanged throughout.

## 10. Output Analysis

CKA uses sample-centered raw native coordinates and the dimension-independent feature-space formula. Ridgealpha100 fixed; both source/target scalers fit each outer-training fold only, predictions restored to native target units. Targets are feature coordinates, not pathology labels. R² variance-weighted in each target feature space; directional asymmetry is not encoder superiority.

## 11. Quantitative Results

NIH exploratory; potential CXR encoder pretraining overlap; not independent external test.

FullTRAIN CKA: CLS 0.401838;patchmean 0.233407. Oneimage/patient(n1222): CLS 0.326174;patchmean 0.180874. Fold CKA/ridge scores saved in CSV. No inferential CKA CI estimated.

| Representation / direction | OOF R² | Conditional patient95%CI |
|---|---:|---:|
| cls / cxr_to_clear | 0.332632 | 0.314902–0.347025 |
| cls / clear_to_cxr | 0.471733 | 0.460336–0.482203 |
| patchmean / cxr_to_clear | 0.375362 | 0.358715–0.391154 |
| patchmean / clear_to_cxr | 0.592522 | 0.581292–0.602801 |

500sharedpatient-cluster resamples,seed20261015; CIs conditional on fixed OOF fits, no refitting or training-set uncertainty. R² scalar SSE/target-variance identity checked to1e-10. All500draws valid.

## 12. Qualitative Results

e13_similarity.png visually inspected: TRAIN geometry and conditional OOF R², clear scope/limitations. e13_sample_alignment.csv ande13_oof_residuals.csv link de-identified IDs/source SHA to features and errors. No source pixels opened; no clinical image review.

## 13. Comparison with Baseline

No new E1-test or CheXpert metrics; no comparison against old test scores. Compare matching CLS↔CLS orpatchmean↔patchmean only. Shared32×32grids do not imply spatial alignment, and none was assumed.

## 14. Problems Encountered

Initial G0 joint-budget blocker remains historical and immutable. User clarification permits independentE1.3gate. Geometry/pretraining/native preprocessing/width differences and image-weighting sensitivity limit attribution.

## 15. Conclusion

E1.3 executed successfully. Partial linear recovery and nonidentical geometry are descriptive observations. COMPLEMENTARITY_NOT_YET_PROVEN; no claim of better pathology recognition or reportgeneration.

## 16. Next Experiment

Review E1.3 evidence; E5 benchmark requires new explicit authorization even after a resource-eligible profile. Do not launch adaptive/spatial fusion or downstream training.

## 17. Reproducibility Checklist

Original protocol hash, exact IDs/source hashes/cache identities, original folds, fixedalpha100, scalers withinfold, scripts/seeds/resource logs, rawCSVs/figure and CI policy preserved. ConditionalCIs only; no model/checkpoint writes, downloads or newtestevaluation.

## 18. Short Lab Update

E1.3 completed TRAIN3783/1222patients: CKA CLS0.4018,patchmean0.2334; asymmetric OOF R²0.333–0.593.14.09s/0.484GiB CPU; no scientific complementarity conclusion.
