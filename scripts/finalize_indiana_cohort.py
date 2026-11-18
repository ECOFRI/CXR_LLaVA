"""Validate human view/pairing annotations and freeze an Indiana cohort."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FRONTAL_LABELS = {'frontal_PA', 'frontal_AP', 'frontal_unspecified'}
VIEW_LABELS = FRONTAL_LABELS | {'lateral', 'other', 'unknown'}
PAIRING_LABELS = {'matches', 'mismatch', 'uncertain'}
SELECTION_LABELS = {'selected', 'not_selected'}


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True,
                        help='Path to inventory/images.csv')
    parser.add_argument('--annotations', type=Path, required=True,
                        help='Exported review/view_annotations.csv')
    parser.add_argument('--output', type=Path, required=True,
                        help='New output directory for frozen cohort')
    args = parser.parse_args()
    inventory_path, annotation_path, output = (p.resolve() for p in
                                                (args.inventory, args.annotations, args.output))
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    inventory = read_csv(inventory_path)
    annotations = read_csv(annotation_path)
    required = {'report_id', 'image_id', 'view_label', 'view_evidence', 'reviewer',
                'reviewed_at', 'selection_status', 'pairing_status', 'review_notes'}
    missing = required - set(annotations[0] if annotations else [])
    if missing:
        raise ValueError(f'Annotations are missing required columns: {sorted(missing)}')

    expected = {(row['report_id'], row['image_id']): row for row in inventory
                if row['image_exists'] == 'True' and row['image_decodes'] == 'True'}
    actual = {}
    for row in annotations:
        key = (row['report_id'].strip(), row['image_id'].strip())
        if key in actual:
            raise ValueError(f'Duplicate annotation row: {key}')
        if key not in expected:
            raise ValueError(f'Annotation outside inventory: {key}')
        actual[key] = row
    missing_annotations = set(expected) - set(actual)
    if missing_annotations:
        raise ValueError(f'{len(missing_annotations)} decodable images lack review; first: {sorted(missing_annotations)[:5]}')

    chosen_by_report = defaultdict(list)
    image_exclusions = []
    for key, image in expected.items():
        annotation = actual[key]
        view = annotation['view_label'].strip()
        pairing = annotation['pairing_status'].strip()
        selection = annotation['selection_status'].strip()
        missing_review = [field for field in ('view_evidence', 'reviewer', 'reviewed_at')
                          if not annotation[field].strip()]
        if view not in VIEW_LABELS or pairing not in PAIRING_LABELS or selection not in SELECTION_LABELS:
            raise ValueError(f'Incomplete or invalid categorical annotation for {key}')
        if missing_review:
            raise ValueError(f'Missing {missing_review} for annotation {key}')
        if selection == 'selected':
            if view not in FRONTAL_LABELS:
                raise ValueError(f'Only reviewed frontal images can be selected: {key} ({view})')
            if pairing != 'matches':
                raise ValueError(f'Selected image must have a confirmed matching report: {key} ({pairing})')
            if not image['ground_truth_report'].strip():
                raise ValueError(f'Selected image has no reference report: {key}')
            chosen_by_report[key[0]].append(image | annotation)
        elif not annotation['review_notes'].strip():
            raise ValueError(f'Add a reason in review_notes for nonselected image: {key}')

    multiple = {report_id: rows for report_id, rows in chosen_by_report.items() if len(rows) > 1}
    if multiple:
        report_id, rows = next(iter(multiple.items()))
        raise ValueError(f'Multiple selected images for report {report_id}: {[r["image_id"] for r in rows]}')
    if not chosen_by_report:
        raise ValueError('No reviewed frontal report/image pairs selected')

    cohort = []
    for report_id, rows in sorted(chosen_by_report.items()):
        row = rows[0]
        cohort.append({
            'report_id': report_id, 'image_id': row['image_id'], 'image_path': row['image_path'],
            'ground_truth_report': row['ground_truth_report'], 'view_label': row['view_label'],
            'view_evidence': row['view_evidence'], 'reviewer': row['reviewer'],
            'reviewed_at': row['reviewed_at'], 'caption': row['caption'],
            'reference_policy': row['reference_policy'], 'image_sha256': row['image_sha256'],
            'xml_sha256': row['xml_sha256'], 'selection_status': row['selection_status'],
            'pairing_status': row['pairing_status'],
        })
    selected_keys = {(row['report_id'], row['image_id']) for row in cohort}
    for key, image in expected.items():
        if key not in selected_keys:
            annotation = actual[key]
            reason = annotation['review_notes'].strip()
            if annotation['selection_status'].strip() == 'selected':
                reason = 'selected image not retained due to invalid cohort rules'
            image_exclusions.append({
                'report_id': image['report_id'], 'image_id': image['image_id'],
                'view_label': annotation['view_label'], 'pairing_status': annotation['pairing_status'],
                'reason': reason,
            })

    output.mkdir(parents=True, exist_ok=False)
    fields = list(cohort[0])
    write_csv(output / 'cohort.csv', cohort, fields)
    write_csv(output / 'image_exclusions.csv', image_exclusions,
              ['report_id', 'image_id', 'view_label', 'pairing_status', 'reason'])
    report_ids = {row['report_id'] for row in cohort}
    report_exclusions = [{'report_id': row['report_id'], 'reason': 'no selected reviewed frontal image'}
                         for row in read_csv(inventory_path.parent / 'reports.csv')
                         if row['report_id'] not in report_ids]
    write_csv(output / 'report_exclusions.csv', report_exclusions, ['report_id', 'reason'])
    counts = Counter(row['view_label'] for row in cohort)
    config = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'Indiana/OpenI NLMCXR', 'n_selected': len(cohort),
        'view_counts': dict(counts), 'reference_policy': 'original FINDINGS + IMPRESSION; whichever sections are present',
        'selection_policy': 'one human-reviewed frontal image per report; report pairing must be marked matches',
        'annotation_sha256': sha256(annotation_path), 'inventory_images_sha256': sha256(inventory_path),
        'cohort_sha256': sha256(output / 'cohort.csv'),
        'exact_paper_split': 'unresolved',
        'benchmark_claim': 'reconstructed-cohort evaluation; not exact reproduction',
    }
    (output / 'cohort_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
