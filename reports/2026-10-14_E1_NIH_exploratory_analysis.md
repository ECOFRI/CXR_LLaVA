# E1 NIH exploratory analysis

## 1. Experiment Information

E1 frozen-feature matched probes; COMPLETE, exploratory metric-complete evaluation. Written 2026-10-14T02:21:57.825783+07:00. Repository commit f6535c6d9d7eb23c2e9a2ac24f48771a1d276efc. Run directories: `research/runs/20261013T220300_E1_CXR`, `research/runs/20261014_E1_CLEAR_CPU`, `research/runs/20261014_E1_NIH_probes`. Scripts: research/e1_features.py, e1_job.py, e1_probe.py, e1_review.py, e1_analysis_report.py.

## 2. Objective

Compare native CLS and spatial patch-mean representations with matched linear probes. No encoder gradients or LLM training. NIH exploratory; potential CXR encoder pretraining overlap; not independent external test.

## 3. Survey / Technical Motivation

Saved paper crosswalk: `research/inventory/paper_dataset_crosswalk_20261013.md`. CXR-LLaVA Table 1 includes NIH in encoder pretraining. Individual local-image membership remains unknown. Native representations have different dimensions and native preprocessing; the result tests this complete representation/probe setup, not isolated architecture superiority.

## 4. Environment

cxr: `/home/student4/anaconda3/envs/cxr-llava-k80`, PyTorch 1.12.1+cu102, FP32, batch 1; K80 GPU0.

clear: `/storage/student4/hanhpm/CXR_LLaVA_Improvement/research/envs/cxr-clear-cpu`, PyTorch 2.3.1+cpu, FP32, batch 1; CPU only.

CPU process versions are saved in environment manifests; the completed system inventory is reused. Hardware differs, so efficiency ranking is unsupported.

## 5. Dependency Setup

Existing stable cxr-llava-k80 and isolated cxr-clear-cpu were reused. No installation, base-environment change or in-place dependency update was performed for E1. CLEAR emits expected CPU xFormers-unavailable warnings; official PyTorch path executes.

## 6. Code / Notebook Changes

Added project research orchestration, provenance checks, compact caches, matched probes and local evidence/report generation. Original CXR-LLaVA, CLEAR implementation and checkpoint weights were preserved. Source/checkpoint hashes are pinned in feature manifests.

## 7. Experiment Configuration

4999 local NIH images: train 3783 / 1222 patients, validation 344 / 57 patients, test 872 / 56 patients. Original NIH14 labels matched source metadata in all 69986 binary comparisons. Hernia excluded by frozen support rule; 13 pathologies retained. Five supplemental labels and No Finding excluded from primary ontology. No cross-split image/patient ID overlap. All 4999 encoded source-file hashes are distinct (saved-hash audit); this does not exclude differently encoded duplicates. Bounded 300-image pixel screen is not exhaustive duplicate validation. CXR CLS/patchmean 1024-D; CLEAR 768-D; register tokens excluded from spatial mean. Native inputs 512 grayscale / 448 RGB. Three warm-up forwards. Seed42; protocol SHA256 `0a188f033d6e31bb05617e5257695f7b0fc2a9b43ebf27b5d074af883a951450`.

## 8. Code Executed

Exact commands: feature/process manifests and `reproduce.sh` in the probe directory. StandardScaler fit on train only. L2 liblinear logistic regression, C in [0.001,0.01,0.1,1], highest validation AP; ties lowest C. Threshold selected on validation F1 over 0.01–0.99; ties closest 0.5 then largest. All models and selections hashed before test scores were computed once. No CheXpert test tuning.

## 9. Output

Feature caches and test predictions saved; all native shapes and numeric checks passed. Deterministic endpoint repeats have maximum absolute difference <=1e-6. Candidate convergence warnings/iteration counts: `convergence_log.csv`. Raw process logs and measured CPU resource records retained.

## 10. Output Analysis

Finite native features and checkpoint integrity passed. Only CLS and patchmean were cached; no full patch sequence or 4096-D LLM-projector cache. Test metrics use the same eligible findings and source cohort for both encoders. Patient bootstrap resamples preserve repeated images and pairing across encoders.

## 11. Quantitative Results

**NIH exploratory; potential CXR encoder pretraining overlap; not independent external test.**

| Encoder / representation | Macro AUROC | Macro AP | Macro F1 |
|---|---:|---:|---:|
| cxr_cls | 0.630225 | 0.148528 | 0.189721 |
| cxr_patchmean | 0.631182 | 0.154900 | 0.173991 |
| clear_cls | 0.761800 | 0.250440 | 0.290319 |
| clear_patchmean | 0.767117 | 0.264146 | 0.305094 |

Per-class 95% patient-cluster intervals and paired CLEAR−CXR differences: `e1_metrics.csv`, `paired_cluster_differences.csv`; 1000 shared bootstrap draws. One-class ranking draws excluded; valid zero F1 preserved. Undefined denominators produce NaN, with valid-draw counts saved.

**NIH exploratory; potential CXR encoder pretraining overlap; not independent external test.**

| Resource | CXR K80 | CLEAR CPU |
|---|---:|---:|
| Runner wall seconds (includes model load) | 3005.933766 | 5336.711332 |
| Mean native forward seconds | 0.518201 | 1.010960 |
| p95 forward seconds | 0.532362 | 1.035009 |
| Peak RSS GiB | 3.196617 | 1.413033 |
| GPU-hours | 0.835017 | 0.000000 |

CLEAR full process wall (including imports): 5347.462718 s; CPU user 20145.299033 s, system 288.263674 s. CXR final CPU-seconds/full-parent-process accounting was not recorded; the saved live observation is not final accounting. Its runner wall and GPU/RSS counters are measured.

## 12. Qualitative Results

`e1_disagreements_contact_sheet.png` displays distinct CLS prediction disagreements selected by a declared score-gap rule; source SHA256 and dataset labels are linked in `image_evidence_manifest.csv`. These are dataset-label disagreements, not verified clinical findings. Expert review NOT PERFORMED. Native preprocessing sheets are saved separately; shared 32×32 grids do not establish spatial alignment.

## 13. Comparison with Baseline

The matched NIH experiment compares four representation conditions under one probe protocol. It is not directly comparable to locked CheXpert report-generation F1. Differences can reflect pretraining, preprocessing, dimension and probe interaction. K80 versus CPU timings cannot establish which encoder is faster.

## 14. Problems Encountered

Potential CXR pretraining exposure, convenience-subset selection, AP/PA distribution shift and only 56 test patients limit generalization. Spatial fusion remains NOT_YET_VALIDATED. Full CLEAR concept E2 remains BLOCKED_CONCEPT_ASSETS_OR_RESOURCE; direct E2 was completed separately. Storage account quota is unverified; observed resource bounds are not a quota guarantee.

## 15. Conclusion

E1 completed frozen extraction and matched probe evaluation on the audited local NIH cohort. Measured metrics are exploratory representation evidence. They do not establish independent external performance, clinical validity or better report generation. H1 general superiority remains inconclusive; H2 concept utility is untested by this experiment.

## 16. Next Experiment

Review per-class uncertainty and image evidence. Identify a pretraining-independent development/evaluation cohort before stronger claims. Validate source-coordinate alignment before spatial fusion. Any expensive adapter/LoRA work requires a compute estimate and explicit approval.

## 17. Reproducibility Checklist

Saved protocol, source cohort, source/checkpoint hashes, feature caches, environment versions, seeds, validation selections, fitted probes, test unlock manifest, test predictions, convergence logs, patient bootstrap intervals, resource records, plots and source-linked image evidence. `reproduce.sh` describes fresh-output commands; current test predictions must be reused for offline report revisions.

## 18. Short Lab Update

Completed E1 frozen native CLS/patchmean extraction and matched linear probes with strict train/validation/test separation. E2 direct contrastive CheXpert evaluation is also complete. GO for reviewing these bounded exploratory results; HOLD independent superiority claims, full concept pipeline and spatial fusion until their missing evidence is resolved.
