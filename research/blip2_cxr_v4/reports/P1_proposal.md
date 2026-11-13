# Proposed P1 work order

1. CPU audit Indiana 3667 prediction/cohort rows: unique report/image keys, nulls, exact one-to-one linkage, reference source and view policy; inspect E4 B1 264 JSONL statuses, arms, unique image keys and references. Do not score yet.
2. CPU audit MIMIC report/JPG coverage by subject and study path and `subject_id/study_id` metadata, plus patient/study grouped splits. Use aggregate counts only in reports.
3. CPU audit VinDr version/license, label ontology and split overlap; retain classification scope unless free-text reports are verified.
4. Test split overlap, duplicate image/study IDs, and known pretraining exposure; emit relative-path manifests with controlled identifiers stored only in protected local artifacts.

Estimated CPU cost: Indiana/E4 seconds to minutes; MIMIC metadata scans and report coverage minutes to tens of minutes depending on network filesystem. No GPU expected. Blockers: /home space, full MIMIC report coverage UNKNOWN, checkpoint/license and paper provenance UNKNOWN.
