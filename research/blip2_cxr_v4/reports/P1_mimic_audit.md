# P1 Part B — MIMIC linkage audit (2026-10-22)

Status: PARTIAL. CPU-only aggregate audit; no report text or identifiers recorded. Data remain read-only.

## Sources and version

- JPG metadata root: `/storage/shared/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.1.0`; its README and core CSV filenames say v2.0.0, while the directory and labeled-test filename say v2.1.0. Exact mixed-version provenance: UNKNOWN.
- JPG files are split across that root's `files/` and `/storage/shared/MIMIC-CXR-JPG/split-4/files/`; the latter supplies most sampled images.
- Free-text report root: `/storage/shared/MIMIC-CXR-Report/download-mimic-cxr-txt/files`; credentialed PhysioNet Health Data License v1.5.0 observed. Report release version: UNKNOWN.

## Metadata and linkage

| Item | Verified count |
|---|---:|
| JPG metadata image rows | 377,110 |
| Unique study IDs | 227,835 |
| Unique subject IDs | 65,379 |
| Official split: train / validate / test images | 368,960 / 2,991 / 5,159 |
| CheXpert-derived label rows | 227,827 |
| NegBio-derived label rows | 227,827 |
| v2.1.0 labeled test subset rows | 687 |

Metadata exposes `dicom_id, subject_id, study_id`; the split file has the same keys plus `split`. Both label files are study-level, so their row count is lower than image metadata. All four official CSVs were readable. The saved manifest contains columns and counts only.

A deterministic reservoir sample selected 5 unique studies per subject prefix p10–p19 (50 studies, seed 42). Expected relative report paths use `pXX/p{subject_id}/s{study_id}.txt`; JPG paths use `files/pXX/p{subject_id}/s{study_id}/{dicom_id}.jpg`. Existence results: 50/50 reports; 48/50 JPGs across both local image roots (7/50 under the first root alone). Prefix-weighted sampled report coverage is 100%; this is an estimate, not a full-coverage claim. Among 50 sampled reports, 35 had a Findings heading, 39 an Impression heading, 28 both, and 4 neither. No text was retained.

## Repository testset provenance

`MIMIC_CXR_DATASET_TESTSET.csv` has 3,000 path-only rows, committed 2024-06-02 (Git `3b0d945`). All 3,000 path stems match official split metadata, but 2,943 map to official train, 23 to validate, and only 34 to test. Its name does not make it an official test split. Do not use it as locked MIMIC test data.

## Contamination and feasibility

The repository model card says CXR-LLaVA was trained on MIMIC. CLEAR upstream preprocessing has MIMIC-specific image/report paths and concepts; exact pretraining membership for this local checkpoint is UNKNOWN. Both pose pretraining-exposure risk on MIMIC; within-experiment split leakage must also be prevented. A BLEU/clinical-F1 cohort is **conditionally feasible** because paired reports and image metadata exist, but full report/image coverage, version coherence, split grouping, and checkpoint exposure must be resolved first. Use official `split` and patient/study grouping, not the repository's path-only testset.

## Reproduction and limits

```bash
cd /storage/student4/hanhpm/CXR_LLaVA_Improvement
export TMPDIR=$PWD/research/blip2_cxr_v4/.cache/tmp HF_HOME=$PWD/research/blip2_cxr_v4/.cache/hf TORCH_HOME=$PWD/research/blip2_cxr_v4/.cache/torch PIP_CACHE_DIR=$PWD/research/blip2_cxr_v4/.cache/pip XDG_CACHE_HOME=$PWD/research/blip2_cxr_v4/.cache NLTK_DATA=$PWD/research/blip2_cxr_v4/.cache/nltk
timeout 60 research/envs/cxr-clear-cpu/bin/python research/blip2_cxr_v4/scripts/data/audit_mimic.py --config research/blip2_cxr_v4/configs/eval/mimic_p1.yaml --output-dir research/blip2_cxr_v4/manifests
```

The script supports `--dry-run` and `--resume`. The first 20-per-prefix attempt timed out; the recorded 5-per-prefix run completed. No data were downloaded. Next: P1 Part C only.
