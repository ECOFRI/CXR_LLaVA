# P0 dataset inventory

MIMIC-CXR-JPG metadata and split CSVs each have 377110 rows; columns include `dicom_id,subject_id,study_id` and split `split`. The separate MIMIC-CXR-Report tree has a sampled study text file under a subject folder, and the JPG tree has the matching study directory. This establishes a sample path linkage, not coverage. Free text was not opened. Report completeness, unique studies/patients, and permissions across all partitions remain UNKNOWN.

VinDr label files: train image-reader rows 45000, test image rows 3000; columns include `image_id`, `rad_id` and finding labels. Annotation boxes exist. Version is UNKNOWN; free-text reports NOT_FOUND in shallow inventory. Do not use report metrics.

Indiana local prediction and cohort CSVs each have 3667 rows with `report_id,image_id` and reference/generated report columns. Values were not read. Study matching remains to audit. E4 B1 report JSONL has 264 rows; 24-image uniqueness and valid statuses UNKNOWN pending P1.

See `manifests/datasets.csv` for paths and remaining unknowns. Dataset roots were read only.

Shallow `/storage/shared` scan also found `/storage/shared/sontg/NIH-Chest-X-rays` (12 image tar archives visible at root), `/storage/shared/sontg/CheXpert` (pointer file only), and `/storage/shared/sontg/PadChest` (`links.txt` only). Local metadata/splits/reports for these candidates were NOT_FOUND at root; dataset identity, counts, and availability remain UNKNOWN beyond those observations. No broad `/storage/student4` scan completed; only this repository was inspected there.

Existing output matchability: Indiana 3,667-row prediction file `result/indiana_full_k80_20261004_031940_1252388/inference/predictions.csv` and cohort reference file in the same directory both expose `report_id,image_id`; schema permits a join, but one-to-one completeness is UNKNOWN. E4 B1 264-row artifact `research/runs/20261017_E4_B1_evidence_format_gate_v1/E4_B1_reports.jsonl` exposes `image_id,arm` and has separate sealed reference material under `research/runs/20261017_E4_A2_blinded_paired_report_evaluation_v1/`; exact matching and status validity remain UNKNOWN. No report text or identifiers were copied into this inventory.
