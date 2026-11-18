"""Run pinned official CheXpert in its separate environment, after a sample gate."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from scripts.exp02_reevaluate import source_paths_for_size, sha256_file

LABELER_REVISION = '44ddeb363149aa657296237f18b5472a73c1756f'
NEGBIO_REVISION = '073199e2792824740e89844a59c13d3d40ce4d23'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--labeler-dir', type=Path, required=True)
    parser.add_argument('--negbio-dir', type=Path, required=True)
    parser.add_argument('--labeler-python', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    labeler, negbio = args.labeler_dir.resolve(), args.negbio_dir.resolve()
    python = args.labeler_python.resolve()
    config = json.loads((source / 'run_config.json').read_text())
    dataset, size = config['dataset'], config['cases']
    if dataset not in ('indiana', 'mimic'):
        raise ValueError('A completed cohort run is required')
    paths = source_paths_for_size(source, size, dataset)
    predictions = pd.read_csv(paths['predictions'])
    if len(predictions) != size or not predictions['status'].eq('success').all():
        raise ValueError('Cohort inference is incomplete; no labeling launched')
    for folder, revision in [(labeler, LABELER_REVISION), (negbio, NEGBIO_REVISION)]:
        actual = subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise ValueError(f'Expected pinned revision {revision} in {folder}')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(negbio)
    logs = source / 'labeler_logs'
    logs.mkdir(exist_ok=False)

    def run(input_path, output_path, name):
        command = [str(python), str(labeler / 'label.py'), '--reports_path', str(input_path),
                   '--output_path', str(output_path), '--verbose']
        with (logs / f'{name}.log').open('w') as handle:
            subprocess.run(command, cwd=labeler, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        if not output_path.is_file():
            raise FileNotFoundError(output_path)

    sample_output = logs / 'sample_labels.csv'
    run(labeler / 'sample_reports.csv', sample_output, 'official_sample')
    expected = pd.read_csv(labeler / 'labeled_reports.csv')
    actual = pd.read_csv(sample_output)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    print('Official CheXpert sample labels match reference: PASS', flush=True)
    for kind, input_key, output_key in [('gt', 'gt_input', 'gt_labels'),
                                       ('generated', 'generated_input', 'generated_labels')]:
        if paths[output_key].exists():
            raise FileExistsError(paths[output_key])
        run(paths[input_key], paths[output_key], kind)
        print(f'{kind}: labeling complete', flush=True)
    freeze = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
    (logs / 'pip_freeze.txt').write_text(freeze)
    (logs / 'provenance.json').write_text(json.dumps(dict(
        sample_gate='PASS', labeler_revision=LABELER_REVISION, negbio_revision=NEGBIO_REVISION,
        input_sha256={key: sha256_file(paths[key]) for key in ('gt_input', 'generated_input')},
        sample_sha256=sha256_file(sample_output), labeler_python=str(python)), indent=2))
    print('Run exp02_reevaluate to verify row/text alignment and compute metrics.', flush=True)


if __name__ == '__main__':
    main()
