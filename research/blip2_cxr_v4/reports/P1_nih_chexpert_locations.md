# P1 Part D — NIH and CheXpert local paths (2026-10-22)

Status: PASS for targeted location and metadata; full dataset coverage and pretraining independence remain UNKNOWN. CPU/read-only. No locked test metrics or tuning were performed.

## Search order and sources

Repository evidence was checked first: `research/configs`, `research/inventory/dataset_catalog.csv`, `research/inventory/e1_nih/`, `research/research_state.json`, `research/experiment_registry.csv`, E1/E2 scripts and existing run manifests. Targeted `timeout 20 ls` checks then confirmed only `/storage/student4/hanhpm/datasets/CXR8`, `/storage/student4/hanhpm/datasets/CheXpert_test_labels`, `/storage/shared/sontg/NIH-Chest-X-rays`, and `/storage/shared/sontg/CheXpert`. No broad `/storage/student4` scan was run.

## NIH / CXR8

- Exact local root: `/storage/student4/hanhpm/datasets/CXR8`; original metadata `Data_Entry_2017_v2020.csv` has 112,120 rows. The PruneCXR label CSVs under `PruneCXR/` have 78,506 train, 12,533 validation, and 21,081 test rows. These are label-file counts, not local image counts.
- Existing audited local image cohort: `research/inventory/e1_nih/cohort_full.csv`, 4,999 rows, with 3,783 train / 344 validation / 872 test and 1,222 / 57 / 56 patients respectively. Three bounded image paths from this manifest were readable. The prior `audit_manifest.json` reports no cross-split patient/image collisions; it did only a bounded 300-image pixel-overlap screen.
- `research/runs/20261014_E1_NIH_probes/test_unlock_manifest.json` says choices were frozen before a once-only test evaluation. Preserve that test lock. NIH is exploratory because released CXR encoder pretraining may include NIH; independent external-test status is not established.
- `/storage/shared/sontg/NIH-Chest-X-rays` contains 12 image tar archives, but extracted image counts were not inferred from them.

## CheXpert

- Exact local root: `/storage/student4/hanhpm/datasets/CheXpert_test_labels`; `CheXpert/test_labels.csv` has 668 image rows (518 frontal, 150 lateral by path tokens); `groundtruth.csv` has 500 study rows. The local README describes a 500-study/500-patient radiologist-consensus test set. Three bounded paths resolved under `CheXpert/test/`.
- This is a locked test set. Existing E2 work used 518 frontal images. Train/validation images or labels were NOT_FOUND under the targeted authorized roots; `/storage/shared/sontg/CheXpert` contains only a pointer file. No thresholds, prompts or probes should be tuned on these test labels.

## Reproduction and limits

Commands used: targeted `timeout 20 ls` on the four candidate directories; Python `csv.DictReader` counts of the six named metadata/label CSVs and the existing local cohort; three `Path.is_file()` checks per cohort. No directory-wide image count or image read. Aggregate source paths/counts are in `manifests/nih_chexpert_locations.json`. Source provenance and split limits are in `research/inventory/e1_nih/provenance_gate.md` and prior manifests. Next: P1 Part E candidate comparison only.
