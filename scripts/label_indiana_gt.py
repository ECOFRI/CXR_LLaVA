"""Label frozen Indiana cohort references with pinned official CheXpert."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.exp02_reevaluate import TARGET_PATHOLOGIES
from scripts.label_cohort import LABELER_REVISION, NEGBIO_REVISION


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, columns, header=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        if header:
            writer.writeheader()
        writer.writerows(rows)


def run_labeler(python, labeler, negbio, input_path, output_path, log_path):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(negbio)
    command = [str(python), str(labeler / 'label.py'), '--reports_path', str(input_path),
               '--output_path', str(output_path), '--verbose']
    with log_path.open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=labeler, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not output_path.is_file():
        raise FileNotFoundError(f'CheXpert did not produce {output_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labeler-dir', type=Path, required=True)
    parser.add_argument('--negbio-dir', type=Path, required=True)
    parser.add_argument('--labeler-python', type=Path, required=True)
    args = parser.parse_args()
    cohort_path, output, labeler, negbio, python = (p.resolve() for p in
                                                   (args.cohort, args.output, args.labeler_dir,
                                                    args.negbio_dir, args.labeler_python))
    if output.exists():
        raise FileExistsError(f'Choose a fresh GT labeling output: {output}')
    config_path = cohort_path.parent / 'cohort_config.json'
    config = json.loads(config_path.read_text())
    if config['cohort_sha256'] != sha256(cohort_path):
        raise ValueError('Frozen cohort SHA-256 does not match cohort_config.json')
    cohort = read_csv(cohort_path)
    if len(cohort) != config['n_cohort'] or not cohort:
        raise ValueError('Frozen cohort row count mismatch')
    ids = [row['report_id'] for row in cohort]
    if len(ids) != len(set(ids)):
        raise ValueError('More than one selected image per report')
    if any(not row['ground_truth_report'].strip() for row in cohort):
        raise ValueError('Empty reference in frozen cohort')
    for folder, revision in ((labeler, LABELER_REVISION), (negbio, NEGBIO_REVISION)):
        actual = subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise ValueError(f'Expected pinned revision {revision} in {folder}; found {actual}')
    if not python.is_file():
        raise FileNotFoundError(python)

    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'inputs'
    logs = output / 'logs'
    inputs.mkdir()
    logs.mkdir()
    gt_input = inputs / 'gt_reports.csv'
    row_map_path = output / 'row_map.csv'
    write_csv(gt_input, [{'report': row['ground_truth_report']} for row in cohort], ['report'], header=False)
    row_map = [
        {'row_index': index, 'report_id': row['report_id'], 'image_id': row['image_id'],
         'text_sha256': hashlib.sha256(row['ground_truth_report'].encode('utf-8')).hexdigest()}
        for index, row in enumerate(cohort)
    ]
    write_csv(row_map_path, row_map, ['row_index', 'report_id', 'image_id', 'text_sha256'])
    sample_output = logs / 'official_sample_labels.csv'
    print('Running pinned official CheXpert sample gate', flush=True)
    run_labeler(python, labeler, negbio, labeler / 'sample_reports.csv', sample_output,
                logs / 'official_sample.log')
    expected_sample = pd.read_csv(labeler / 'labeled_reports.csv')
    observed_sample = pd.read_csv(sample_output)
    pd.testing.assert_frame_equal(observed_sample, expected_sample, check_dtype=False)
    print('Official sample gate: PASS', flush=True)

    labels_path = output / 'raw_labels.csv'
    print(f'Labeling {len(cohort)} unique reference reports', flush=True)
    run_labeler(python, labeler, negbio, gt_input, labels_path, logs / 'gt.log')
    labeled = pd.read_csv(labels_path, keep_default_na=False)
    texts = [row['ground_truth_report'] for row in cohort]
    if len(labeled) != len(cohort) or 'Reports' not in labeled:
        raise ValueError('GT label row count or Reports column mismatch')
    if labeled['Reports'].astype(str).tolist() != texts:
        raise ValueError('GT label Reports text differs from frozen input')
    with gt_input.open(newline='', encoding='utf-8') as stream:
        input_texts = [row[0] for row in csv.reader(stream)]
    if input_texts != texts:
        raise ValueError('Headerless GT input differs from frozen cohort')
    support_rows = []
    for pathology in TARGET_PATHOLOGIES:
        if pathology not in labeled:
            raise ValueError(f'Missing CheXpert target: {pathology}')
        values = [str(value).strip() for value in labeled[pathology]]
        counts = Counter(values)
        allowed = {'', '1.0', '0.0', '-1.0', '1', '0', '-1'}
        unexpected = set(counts) - allowed
        if unexpected:
            raise ValueError(f'Unexpected CheXpert values for {pathology}: {unexpected}')
        support_rows.append({
            'pathology': pathology, 'positive': counts['1.0'] + counts['1'],
            'negative': counts['0.0'] + counts['0'],
            'uncertain': counts['-1.0'] + counts['-1'],
            'unmentioned': counts[''], 'reports': len(cohort),
        })
    write_csv(output / 'support.csv', support_rows,
              ['pathology', 'positive', 'negative', 'uncertain', 'unmentioned', 'reports'])
    package_freeze = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
    (logs / 'pip_freeze.txt').write_text(package_freeze)
    provenance = {
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'sample_gate': 'PASS', 'alignment': 'PASS', 'cases': len(cohort),
        'labeler_revision': LABELER_REVISION, 'negbio_revision': NEGBIO_REVISION,
        'labeler_python': str(python), 'cohort_sha256': sha256(cohort_path),
        'input_sha256': sha256(gt_input), 'row_map_sha256': sha256(row_map_path),
        'raw_labels_sha256': sha256(labels_path), 'official_sample_sha256': sha256(sample_output),
    }
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print('GT labeling complete; row/text alignment: PASS', flush=True)
    print(f'Saved: {output}', flush=True)


if __name__ == '__main__':
    main()
