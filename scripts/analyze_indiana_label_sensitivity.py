"""Audit frozen Indiana counts and compute an explicitly alternative missing policy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def label(value):
    return None if value == '' else int(float(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gt-labeling', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.gt_labeling / 'raw_labels.csv',
             args.run / 'metrics/generated_raw_labels.csv',
             args.run / 'inference/predictions.csv',
             args.run / 'metrics/pathology_metrics.csv']
    gt, pred, records, metrics = [read(p) for p in paths]
    assert len(gt) == len(pred) == len(records) == 3667
    for i, row in enumerate(records):
        assert int(row['cohort_index']) == i
        assert row['status'] == 'success'
        assert gt[i]['Reports'] == row['ground_truth_report']
        assert pred[i]['Reports'] == row['generated_report']
    results = []
    for metric in metrics:
        name = metric['pathology']
        pairs = [(label(g[name]), label(p[name])) for g, p in zip(gt, pred)]
        counts = {key: sum(g == a and p == b for g, p in pairs)
                  for key, a, b in [('tp', 1, 1), ('fp', 0, 1),
                                    ('tn', 0, 0), ('fn', 1, 0)]}
        assert all(value == int(metric[key]) for key, value in counts.items())
        missed = sum(g == 1 and p not in (0, 1) for g, p in pairs)
        assert missed == int(metric['positive_pred_uncertain']) + int(metric['positive_pred_unmentioned'])
        tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
        results.append({'pathology': name, 'joint_f1': float(metric['f1']),
                        'joint_n': sum(counts.values()), 'gt_positive': int(metric['gt_positive_all']),
                        'positive_unknown': missed, 'tp': tp, 'fp': fp, 'fn': fn,
                        'alternative_f1': 2 * tp / (2 * tp + fp + fn + missed),
                        'positive_recovery': tp / int(metric['gt_positive_all'])})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'label_policy_sensitivity.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        'classification': 'post-hoc policy sensitivity, not replacement benchmark',
        'alternative_policy': 'GT must be definite; prediction uncertain or missing maps to 0',
        'joint_macro_f1': sum(r['joint_f1'] for r in results) / len(results),
        'alternative_macro_f1': sum(r['alternative_f1'] for r in results) / len(results),
        'alignment_and_counts': 'PASS',
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    (args.output / 'sensitivity_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
