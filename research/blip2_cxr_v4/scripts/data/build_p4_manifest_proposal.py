"""Prepare unapproved P4 cohort candidates; never authorize GPU execution."""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

VR = Path(__file__).resolve().parents[2]
ROOT = VR.parents[1]
sys.path.insert(0, str(VR / 'src'))

from vlm_medical.data.manifests import eligible_records


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def mimic_candidates(config, split, count):
    records = []
    seen = set()
    for row in eligible_records(config, split, limit=count):
        key = (row['subject_id'], row['study_id'])
        if key in seen:
            raise ValueError('Duplicate MIMIC study')
        seen.add(key)
        records.append({
            'dataset': 'mimic', 'split': split, 'subject_id': row['subject_id'],
            'study_id': row['study_id'], 'image_id': row['image_id'],
            'image_path': str(row['image'].resolve()), 'view': row['view_position'],
            'reference_path': str(row['report'].resolve()),
            'reference_file_sha256': sha256(row['report']),
            'target_text_sha256': text_sha256(row['text']),
            'reference_policy': 'findings_then_impression_newline_nonempty',
        })
    if len(records) != count:
        raise ValueError(f'Insufficient eligible MIMIC {split}: {len(records)}/{count}')
    return records


def indiana_candidates(cohort_path, predictions_path, count):
    with cohort_path.open(newline='', encoding='utf-8-sig') as stream:
        cohort = list(csv.DictReader(stream))
    with predictions_path.open(newline='', encoding='utf-8-sig') as stream:
        predictions = list(csv.DictReader(stream))
    by_id = {}
    for prediction in predictions:
        key = prediction['report_id']
        if key in by_id:
            raise ValueError('Duplicate X00 prediction study')
        by_id[key] = prediction
    seen = set()
    records = []
    for index, row in enumerate(cohort):
        if not row['view_label'].startswith('frontal'):
            continue
        prediction = by_id.get(row['report_id'])
        if prediction is None or prediction.get('status') != 'success':
            continue
        if not prediction['generated_report'].strip() or not row['ground_truth_report'].strip():
            continue
        for key in ('image_id', 'ground_truth_report'):
            if prediction[key] != row[key]:
                raise ValueError('Indiana prediction/reference identity mismatch')
        image = Path(row['image_path'])
        if not image.is_absolute():
            image = ROOT / image
        if not image.is_file():
            continue
        if row['report_id'] in seen:
            raise ValueError('Duplicate Indiana cohort study')
        seen.add(row['report_id'])
        records.append({
            'dataset': 'indiana', 'split': 'external_x00', 'subject_id': None,
            'study_id': row['report_id'], 'image_id': row['image_id'],
            'image_path': str(image.resolve()), 'view': row['view_label'],
            'expected_image_sha256': row['image_sha256'],
            'reference_source': str(cohort_path.resolve()), 'source_row': index,
            'target_text_sha256': text_sha256(row['ground_truth_report']),
            'reference_policy': row['reference_policy'],
            'x00_predictions_source': str(predictions_path.resolve()),
            'x00_prediction_sha256': text_sha256(prediction['generated_report']),
        })
        if len(records) == count:
            return records
    raise ValueError(f'Insufficient eligible Indiana studies: {len(records)}/{count}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--proposal', action='store_true', required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to((VR / 'runs').resolve()):
        raise ValueError('Proposal must be under the existing V4 runs directory')
    if output.exists():
        raise FileExistsError('Never overwrite a prior proposal/run: ' + str(output))
    config_path = VR / 'configs/eval/mimic_p1.yaml'
    config = json.loads(config_path.read_text())
    cohort_path = ROOT / 'result/indiana_full_k80_20261004_031940_1252388/inference/cohort.csv'
    predictions_path = cohort_path.with_name('predictions.csv')
    output.mkdir(parents=True, mode=0o700)
    groups = {}
    for name, split, count in (('mimic_train', 'train', 2000), ('mimic_validate', 'validate', 500)):
        print(f'Preparing {name} candidate ({count} studies)', flush=True)
        groups[name] = mimic_candidates(config, split, count)
    groups['indiana_x00'] = indiana_candidates(cohort_path, predictions_path, 500)
    train_subjects = {r['subject_id'] for r in groups['mimic_train']}
    validate_subjects = {r['subject_id'] for r in groups['mimic_validate']}
    if train_subjects & validate_subjects:
        raise ValueError('MIMIC train/validation patient overlap')
    manifests = {}
    for name, rows in groups.items():
        path = output / (name + '.draft.jsonl')
        with path.open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
        path.chmod(0o600)
        manifests[name] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
                           'studies': len(rows), 'unique_images': len({r['image_id'] for r in rows})}
    metadata_root = Path(config['jpg_root'])
    inputs = (config_path, cohort_path, predictions_path,
              metadata_root / 'mimic-cxr-2.0.0-split.csv.gz',
              metadata_root / 'mimic-cxr-2.0.0-metadata.csv.gz')
    index = {
        'status': 'PROPOSED_NOT_APPROVED', 'approved': False,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'selection_proposal': 'first eligible unique studies in existing source order; first available AP/PA image per MIMIC study',
        'sampling_seed': None, 'proposed_training_shuffle_seed': 42,
        'train_validation_patient_overlap': 0,
        'image_content_rehash': 'MIMIC not performed; Indiana uses existing expected SHA; input integrity checks required before execution',
        'manifests': manifests, 'inputs_sha256': {str(p): sha256(p) for p in inputs},
    }
    (output / 'manifest_index.proposed.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'status': index['status'], 'manifests': manifests}, indent=2), flush=True)


if __name__ == '__main__':
    main()
