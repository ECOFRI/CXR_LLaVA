"""CPU VinDr classification evaluation from saved continuous scores."""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

VR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VR / 'src'))

from vlm_medical.eval.bootstrap import interval, resamples
from vlm_medical.eval.classification_metrics import auroc, average_precision, threshold_metrics


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def load_thresholds(path):
    data = json.loads(path.read_text())
    if data.get('selection_split') != 'validate' or data.get('locked_before_test') is not True:
        raise ValueError('Thresholds must be fixed on validation before test')
    if not isinstance(data.get('thresholds'), dict):
        raise ValueError('Missing per-class thresholds')
    return data['thresholds']


def evaluate(labels, scores, thresholds, aliases, reps, seed):
    label_by_image = {}
    columns = [name for name in labels[0] if name != 'image_id']
    canonical = {name: aliases.get(name, name) for name in columns}
    if len(set(canonical.values())) != len(columns):
        raise ValueError('Ontology alias collision')
    for row in labels:
        key = row['image_id']
        if key in label_by_image:
            raise ValueError('Duplicate image label; reader consensus required before evaluation')
        label_by_image[key] = {canonical[name]: int(row[name]) for name in columns}
    score_by_image = {}
    for row in scores:
        image = row['image_id']
        finding = aliases.get(row['finding'], row['finding'])
        if image not in label_by_image or finding not in canonical.values():
            raise ValueError('Unmatched image or finding in scores')
        key = (image, finding)
        if key in score_by_image:
            raise ValueError('Duplicate image/finding score')
        score_by_image[key] = float(row['score'])
    images = list(label_by_image)
    classes = list(canonical.values())
    if len(score_by_image) != len(images) * len(classes):
        raise ValueError('Scores are incomplete for the test cohort/classes')
    if set(thresholds) != set(classes):
        raise ValueError('Threshold class coverage differs from labels')
    output = {}
    for finding in classes:
        truth = [label_by_image[image][finding] for image in images]
        if any(value not in (0, 1) for value in truth):
            raise ValueError('VinDr labels must be binary')
        values = [score_by_image[(image, finding)] for image in images]
        threshold = float(thresholds[finding])
        base = threshold_metrics(truth, values, threshold)
        base.update(auroc=auroc(truth, values), auprc=average_precision(truth, values), threshold=threshold)
        bootstrap = {key: [] for key in ('auroc', 'auprc', 'precision', 'recall', 'f1')}
        for sample in resamples(len(images), reps, seed):
            ys = [truth[i] for i in sample]
            xs = [values[i] for i in sample]
            scores_for_sample = threshold_metrics(ys, xs, threshold)
            scores_for_sample.update(auroc=auroc(ys, xs), auprc=average_precision(ys, xs))
            for key in bootstrap:
                bootstrap[key].append(scores_for_sample[key])
        base['ci95'] = {key: interval(values) for key, values in bootstrap.items()}
        output[finding] = base
    return {'n_images': len(images), 'per_class': output,
            'bootstrap': {'unit': 'image; patient identifier unavailable', 'reps': reps, 'seed': seed}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=VR / 'configs/eval/vindr_classification.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--scores', type=Path)
    parser.add_argument('--thresholds', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    labels_path = args.labels or Path(config['labels'])
    scores_path = args.scores or Path(config['scores'])
    thresholds_path = args.thresholds or Path(config['thresholds'])
    if args.dry_run:
        print(json.dumps({'status': 'DRY_RUN', 'labels_present': labels_path.is_file(),
                          'scores_present': scores_path.is_file(),
                          'thresholds_present': thresholds_path.is_file(),
                          'bootstrap_reps': config['bootstrap_reps']}))
        return
    for path in (labels_path, scores_path, thresholds_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    out = args.output_dir or VR / 'runs' / f'vindr_classification_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
    if out.exists():
        if args.resume and (out / 'status.json').exists():
            status = json.loads((out / 'status.json').read_text())
            if status.get('status') == 'COMPLETE':
                print(json.dumps({'status': 'COMPLETE', 'resumed': True}))
                return
        raise FileExistsError(out)
    out.mkdir(parents=True)
    result = evaluate(read_csv(labels_path), read_csv(scores_path), load_thresholds(thresholds_path),
                      config['ontology_aliases'], config['bootstrap_reps'], config['seed'])
    result.update(classification='metric-complete only for supplied scores; no model inference',
                  auroc_protocol='Mann-Whitney with tie-averaged ranks',
                  auprc_protocol='non-interpolated average precision; grouped tied scores')
    (out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    rows = []
    for finding, data in result['per_class'].items():
        for metric in ('auroc', 'auprc', 'precision', 'recall', 'f1'):
            ci = data['ci95'][metric]
            rows.append({'finding': finding, 'metric': metric, 'value': data[metric],
                         'ci95_low': ci[0], 'ci95_high': ci[1], 'threshold': data['threshold'],
                         'positive_support': data['positive_support'], 'negative_support': data['negative_support']})
    with (out / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (out / 'run_manifest.json').write_text(json.dumps({'config': str(args.config), 'labels': str(labels_path),
        'scores': str(scores_path), 'thresholds': str(thresholds_path), 'n_images': result['n_images'],
        'environment': sys.executable, 'gpu_used': False}, indent=2) + '\n')
    (out / 'status.json').write_text(json.dumps({'status': 'COMPLETE'}) + '\n')
    print(json.dumps({'status': 'COMPLETE', 'n_images': result['n_images'], 'output_dir': str(out)}))


if __name__ == '__main__':
    main()
