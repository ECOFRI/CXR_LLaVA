"""Select an Indiana frontal cohort using NLM's image-level human view labels."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


NLM_PAGE = 'https://lhncbc.nlm.nih.gov/CHRB/CHRB-resources.html'
FRONTAL_URL = 'https://data.lhncbc.nlm.nih.gov/public/chest-xray/frontal_final.csv'
LATERAL_URL = 'https://data.lhncbc.nlm.nih.gov/public/chest-xray/lateral_final.csv'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_nlm_ids(path):
    names = [line.strip() for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    normalized = [name.removesuffix('.png') if name.startswith('CXR')
                  else 'CXR' + name.removesuffix('.png') for name in names]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f'Duplicate image IDs in official view list: {path}')
    return set(normalized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--frontal', type=Path, required=True)
    parser.add_argument('--lateral', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory for review and frozen cohort artifacts')
    args = parser.parse_args()
    inventory_path = args.inventory.resolve()
    reports_path = inventory_path.parent / 'reports.csv'
    frontal_path, lateral_path, output = (p.resolve() for p in
                                         (args.frontal, args.lateral, args.output))
    if output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {output}')

    images = read_csv(inventory_path)
    reports = read_csv(reports_path)
    frontal, lateral = read_nlm_ids(frontal_path), read_nlm_ids(lateral_path)
    by_key = {}
    image_ids = set()
    for row in images:
        key = (row['report_id'], row['image_id'])
        if key in by_key or row['image_id'] in image_ids:
            raise ValueError(f'Duplicate local report/image ID: {key}')
        by_key[key] = row
        image_ids.add(row['image_id'])
    report_ids = [row['report_id'] for row in reports]
    if len(report_ids) != len(set(report_ids)):
        raise ValueError('Duplicate report UID in inventory')

    all_annotations, frontal_candidates, pairing_review = [], [], []
    frontal_by_report = defaultdict(list)
    hash_reports = defaultdict(set)
    hash_images = defaultdict(list)
    for row in images:
        report_id, image_id = row['report_id'], row['image_id']
        if image_id in frontal and image_id in lateral:
            view, evidence, exclusion = 'unknown', 'conflict: listed as both NLM frontal and lateral', 'official_label_conflict'
        elif image_id in frontal:
            view, evidence, exclusion = 'frontal_unspecified', f'NLM manually classified frontal; {FRONTAL_URL}; image_id={image_id}', ''
        elif image_id in lateral:
            view, evidence, exclusion = 'lateral', f'NLM manually classified lateral; {LATERAL_URL}; image_id={image_id}', 'lateral'
        else:
            view, evidence, exclusion = 'unknown', 'image ID absent from both NLM view lists', 'official_label_missing'

        xml_link = Path(row['xml_path']).is_file() and image_id.startswith(report_id + '_')
        image_ready = row['image_exists'] == 'True' and row['image_decodes'] == 'True'
        if not xml_link:
            pairing_review.append({'report_id': report_id, 'image_id': image_id,
                                   'issue': 'XML UID/image ID disagreement or missing XML',
                                   'evidence': row['xml_path']})
            exclusion = 'pairing_uncertain'
        if not image_ready:
            exclusion = row['exclusion_reason'] or 'image_missing_or_undecodable'
        if not row['ground_truth_report'].strip():
            exclusion = 'empty_reference'
        if row['image_sha256']:
            hash_reports[row['image_sha256']].add(report_id)
            hash_images[row['image_sha256']].append((report_id, image_id))

        annotation = {
            'report_id': report_id, 'image_id': image_id, 'figure_id': row['figure_id'],
            'view_label': view, 'view_evidence': evidence,
            'view_source': 'NLM image-level manually classified labels',
            'view_annotation_method': 'external_human_labels',
            'pairing_status': 'xml_linked' if xml_link else 'needs_review',
            'pairing_evidence': f'parentImage in {row["xml_path"]}; image_id starts with report UID' if xml_link else '',
            'selection_status': 'not_selected', 'exclusion_reason': exclusion,
        }
        all_annotations.append(annotation)
        if view == 'frontal_unspecified' and not exclusion:
            frontal_by_report[report_id].append(row)
            frontal_candidates.append(row | annotation)

    cross_report_duplicate_hashes = {digest: sorted(values) for digest, values in hash_reports.items()
                                     if len(values) > 1}
    if cross_report_duplicate_hashes:
        for digest, values in cross_report_duplicate_hashes.items():
            pairing_review.append({'report_id': ';'.join(values), 'image_id': '',
                                   'issue': 'same image hash across reports', 'evidence': digest})
        raise ValueError('Cross-report duplicate images require manual pairing review')
    for digest, values in hash_images.items():
        if len(values) > 1:
            pairing_review.append({
                'report_id': values[0][0], 'image_id': ';'.join(image_id for _, image_id in values),
                'issue': 'duplicate image content within one report', 'evidence': digest,
            })

    selected_ids = {}
    for report_id, rows in frontal_by_report.items():
        selected_ids[report_id] = min(row['image_id'] for row in rows)
    for annotation in all_annotations:
        if selected_ids.get(annotation['report_id']) == annotation['image_id']:
            annotation['selection_status'] = 'selected'
            annotation['exclusion_reason'] = ''
        elif annotation['view_label'] == 'frontal_unspecified' and not annotation['exclusion_reason']:
            annotation['exclusion_reason'] = 'other_verified_frontal_selected_by_lexicographic_image_id'

    by_image_id = {row['image_id']: row for row in images}
    cohort = []
    for report_id, image_id in sorted(selected_ids.items()):
        row = by_image_id[image_id]
        annotation = next(a for a in all_annotations if a['image_id'] == image_id)
        cohort.append({
            'report_id': report_id, 'image_id': image_id, 'image_path': row['image_path'],
            'ground_truth_report': row['ground_truth_report'],
            'report_xml_path': row['xml_path'], 'figure_id': row['figure_id'],
            'caption': row['caption'], 'reference_policy': row['reference_policy'],
            'image_sha256': row['image_sha256'], 'report_xml_sha256': row['xml_sha256'],
            'view_label': annotation['view_label'], 'view_evidence': annotation['view_evidence'],
            'view_source': annotation['view_source'], 'view_annotation_method': annotation['view_annotation_method'],
            'pairing_status': annotation['pairing_status'], 'pairing_evidence': annotation['pairing_evidence'],
        })
    if len(cohort) != len(selected_ids) or not cohort:
        raise ValueError('Cohort selection failed report uniqueness or has no rows')

    selected_set = {(row['report_id'], row['image_id']) for row in cohort}
    excluded_images = [a for a in all_annotations if (a['report_id'], a['image_id']) not in selected_set]
    excluded_reports = [{'report_id': row['report_id'], 'reason':
                         'empty_reference' if not row['ground_truth_report'].strip() else
                         'no_decodable_image' if row['image_count_exists'] == '0' else
                         'no_unambiguous_NLM_frontal_image'}
                        for row in reports if row['report_id'] not in selected_ids]
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'review' / 'view_annotations.csv', all_annotations, list(all_annotations[0]))
    write_csv(output / 'review' / 'pairing_review.csv', pairing_review,
              ['report_id', 'image_id', 'issue', 'evidence'])
    write_csv(output / 'selection' / 'frontal_candidates.csv', frontal_candidates,
              list(frontal_candidates[0]))
    write_csv(output / 'selection' / 'cohort.csv', cohort, list(cohort[0]))
    write_csv(output / 'selection' / 'image_exclusions.csv', excluded_images,
              list(all_annotations[0]))
    write_csv(output / 'selection' / 'report_exclusions.csv', excluded_reports,
              ['report_id', 'reason'])

    source = {
        'source_page': NLM_PAGE, 'frontal_url': FRONTAL_URL, 'lateral_url': LATERAL_URL,
        'frontal_sha256': sha256(frontal_path), 'lateral_sha256': sha256(lateral_path),
        'nlm_frontal_count': len(frontal), 'nlm_lateral_count': len(lateral),
        'nlm_overlap_count': len(frontal & lateral), 'nlm_only_not_local_count': len((frontal | lateral) - image_ids),
        'local_unlisted_count': len(image_ids - (frontal | lateral)),
        'local_conflict_ids': sorted(image_ids & frontal & lateral),
        'local_unlisted_ids': sorted(image_ids - (frontal | lateral)),
    }
    (output / 'review' / 'source_provenance.json').write_text(json.dumps(source, indent=2), encoding='utf-8')
    config = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'Indiana/OpenI NLMCXR', 'classification': 'reconstructed cohort using NLM image-level view labels',
        'n_xml_reports': len(reports), 'n_images': len(images), 'n_cohort': len(cohort),
        'n_frontal_candidates': len(frontal_candidates), 'n_multiple_frontal_reports':
            sum(len(rows) > 1 for rows in frontal_by_report.values()),
        'n_excluded_reports': len(excluded_reports), 'n_excluded_images': len(excluded_images),
        'view_policy': 'NLM manual frontal/lateral lists; conflicts/unlisted excluded; PA/AP not inferred',
        'pairing_policy': 'XML parentImage linked to report UID; matching image ID prefix; anomalies excluded',
        'selection_policy': 'one eligible frontal per report; lexicographically smallest image_id',
        'reference_policy': 'original FINDINGS + IMPRESSION, whichever nonempty sections exist',
        'manual_pairing_visual_review': False, 'exact_paper_split': 'unresolved',
        'inventory_images_sha256': sha256(inventory_path),
        'inventory_reports_sha256': sha256(reports_path),
        'nlm_frontal_sha256': source['frontal_sha256'], 'nlm_lateral_sha256': source['lateral_sha256'],
        'cohort_sha256': sha256(output / 'selection' / 'cohort.csv'),
        'view_counts': dict(Counter(a['view_label'] for a in all_annotations)),
        'exclusion_counts': dict(Counter(a['exclusion_reason'] for a in excluded_images)),
    }
    (output / 'selection' / 'cohort_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in config.items() if not key.endswith('_sha256')}, indent=2))


if __name__ == '__main__':
    main()
