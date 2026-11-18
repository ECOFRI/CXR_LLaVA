"""Aggregate official-split/metadata audit; never print case identifiers."""
import csv
import gzip
import json
from pathlib import Path
from research.rrg_f0.core import sha256
from research.rrg_f1.cohort import BASE, PRIVATE, dev_rows, atomic_json


def main():
    rows = [json.loads(line) for line in (PRIVATE / 'train_manifest.jsonl').read_text().splitlines()]
    if len(rows) != 30000 or len({(r['subject_id'], r['study_id']) for r in rows}) != 30000:
        raise ValueError('STOP_COHORT_COUNT_OR_UNIQUENESS')
    by_image = {r['dicom_id']: r for r in rows}
    if len(by_image) != 30000:
        raise ValueError('STOP_DUPLICATE_IMAGE')
    local = json.loads((BASE / 'paths.local.json').read_text())
    root = Path(local['jpg_root'])
    matches = 0
    with gzip.open(root / 'mimic-cxr-2.0.0-split.csv.gz', 'rt', newline='') as stream:
        for source in csv.DictReader(stream):
            row = by_image.get(source['dicom_id'])
            if row is None:
                continue
            if source['split'] != 'train' or any(source[k] != row[k] for k in ('subject_id', 'study_id')):
                raise ValueError('STOP_OFFICIAL_SPLIT_LINKAGE_MISMATCH')
            matches += 1
    metadata_matches = 0
    with gzip.open(root / 'mimic-cxr-2.0.0-metadata.csv.gz', 'rt', newline='') as stream:
        for source in csv.DictReader(stream):
            row = by_image.get(source['dicom_id'])
            if row is None:
                continue
            if source['ViewPosition'].upper() not in ('AP', 'PA') or any(source[k] != row[k] for k in ('subject_id', 'study_id')):
                raise ValueError('STOP_FRONTAL_METADATA_LINKAGE_MISMATCH')
            metadata_matches += 1
    overlap = {r['subject_id'] for r in rows} & {r['subject_id'] for r in dev_rows('mimic')}
    if matches != 30000 or metadata_matches != 30000 or overlap:
        raise ValueError('STOP_COHORT_COVERAGE_OR_PATIENT_OVERLAP')
    atomic_json(BASE / 'cohort_audit_summary.json', {'status': 'PASS', 'train_studies': 30000,
                 'unique_train_patients': len({r['subject_id'] for r in rows}), 'official_train_matches': matches,
                 'frontal_metadata_matches': metadata_matches, 'p4_val_patient_overlap': 0,
                 'manifest_sha256': sha256(PRIVATE / 'train_manifest.jsonl')})
    print('COHORT_OFFICIAL_AUDIT_PASS: 30000 train/frontal matches, zero P4-val patient overlap', flush=True)


if __name__ == '__main__':
    main()
