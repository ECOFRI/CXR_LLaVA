"""Official CheXpert labeling and seven-pathology metrics for frozen Indiana run."""

import argparse
import csv
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.exp02_reevaluate import TARGET_PATHOLOGIES
from scripts.exp02_provenance import MODEL_REVISION
from scripts.label_cohort import LABELER_REVISION, NEGBIO_REVISION
from scripts.label_indiana_gt import run_labeler, sha256, write_csv


PAPER_F1 = {
    'Cardiomegaly': 0.62, 'Consolidation': 0.31, 'Edema': 0.67,
    'Lung Opacity': 0.85, 'Pleural Effusion': 0.55, 'Pneumonia': 0.63,
    'Pneumothorax': 0.05,
}
PAPER_CI95 = {
    'Cardiomegaly': (0.57, 0.65), 'Consolidation': (0.09, 0.50),
    'Edema': (0.33, 0.86), 'Lung Opacity': (0.80, 0.89),
    'Pleural Effusion': (0.48, 0.62), 'Pneumonia': (0.42, 0.79),
    'Pneumothorax': (0.00, 0.13),
}


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def headerless(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.reader(stream))
    if any(len(row) != 1 for row in rows):
        raise ValueError(f'Expected exactly one report column in {path}')
    return [row[0] for row in rows]


def parse_label(value):
    raw = str(value).strip()
    if raw == '':
        return None
    number = float(raw)
    if number not in (-1, 0, 1):
        raise ValueError(f'Unexpected CheXpert label {raw}')
    return int(number)


def f1_counts(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else float('nan')


def safe_number(value):
    return value if math.isfinite(value) else None


def verify_pins(labeler, negbio):
    for path, revision in ((labeler, LABELER_REVISION), (negbio, NEGBIO_REVISION)):
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise ValueError(f'Wrong official labeler revision: {path} = {actual}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--gt-labeling', type=Path, required=True)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labeler-dir', type=Path, required=True)
    parser.add_argument('--negbio-dir', type=Path, required=True)
    parser.add_argument('--labeler-python', type=Path, required=True)
    parser.add_argument('--bootstrap-iterations', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    paths = [args.cohort, args.gt_labeling, args.inference, args.output,
             args.labeler_dir, args.negbio_dir, args.labeler_python]
    cohort_path, gt_dir, inference, output, labeler, negbio, python = [p.resolve() for p in paths]
    if output.exists():
        raise FileExistsError(f'Choose a new metrics output directory: {output}')
    if args.bootstrap_iterations < 1:
        raise ValueError('--bootstrap-iterations must be positive')
    verify_pins(labeler, negbio)
    cohort_config = json.loads((cohort_path.parent / 'cohort_config.json').read_text())
    if cohort_config['cohort_sha256'] != sha256(cohort_path):
        raise ValueError('Cohort SHA-256 mismatch')
    cohort = read_csv(cohort_path)
    if len(cohort) != cohort_config['n_cohort']:
        raise ValueError('Cohort row count mismatch')
    gt_provenance = json.loads((gt_dir / 'provenance.json').read_text())
    if (gt_provenance['cohort_sha256'] != sha256(cohort_path)
            or gt_provenance['sample_gate'] != 'PASS' or gt_provenance['alignment'] != 'PASS'
            or gt_provenance['raw_labels_sha256'] != sha256(gt_dir / 'raw_labels.csv')):
        raise ValueError('GT labeling provenance/alignment is invalid')
    gt_map = read_csv(gt_dir / 'row_map.csv')
    gt_labels = pd.read_csv(gt_dir / 'raw_labels.csv', keep_default_na=False)
    if len(gt_map) != len(cohort) or len(gt_labels) != len(cohort):
        raise ValueError('GT label/report row count mismatch')
    for index, row in enumerate(cohort):
        mapped = gt_map[index]
        if (int(mapped['row_index']) != index or mapped['report_id'] != row['report_id']
                or mapped['image_id'] != row['image_id']
                or gt_labels.iloc[index]['Reports'] != row['ground_truth_report']):
            raise ValueError(f'GT label/ID/text alignment mismatch at row {index}')

    run_config = json.loads((inference / 'run_config.json').read_text())
    if (run_config['cohort_sha256'] != sha256(cohort_path)
            or run_config['cases_requested'] != len(cohort)
            or run_config['model_revision'] != MODEL_REVISION):
        raise ValueError('Inference config differs from frozen cohort or pinned model')
    status = (inference / 'status.txt').read_text().strip()
    if status not in ('COMPLETED', 'PARTIAL'):
        raise ValueError(f'Inference is not complete: {status}')
    predictions = read_csv(inference / 'predictions.csv')
    pred_by_index = {}
    for record in predictions:
        index = int(record['cohort_index'])
        if index in pred_by_index or not 0 <= index < len(cohort):
            raise ValueError(f'Duplicate or out-of-range prediction index: {index}')
        row = cohort[index]
        if (record['report_id'] != row['report_id'] or record['image_id'] != row['image_id']
                or record['ground_truth_report'] != row['ground_truth_report']):
            raise ValueError(f'Prediction/GT mismatch at row {index}')
        pred_by_index[index] = record
    successful = [(i, pred_by_index[i]) for i in sorted(pred_by_index)
                  if pred_by_index[i]['status'] == 'success' and pred_by_index[i]['generated_report'].strip()]
    if not successful:
        raise ValueError('No successful generated reports to label')
    generated_text = [record['generated_report'] for _, record in successful]
    generated_input = inference / 'chexpert_inputs/generated_reports.csv'
    generated_map = read_csv(inference / 'chexpert_inputs/row_map.csv')
    if headerless(generated_input) != generated_text or len(generated_map) != len(successful):
        raise ValueError('Generated label input/order mismatch')
    for position, (cohort_index, record) in enumerate(successful):
        mapped = generated_map[position]
        if (int(mapped['chexpert_row']) != position
                or int(mapped['cohort_index']) != cohort_index
                or mapped['report_id'] != record['report_id']
                or mapped['image_id'] != record['image_id']):
            raise ValueError(f'Generated label row map mismatch at position {position}')

    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    sample_output = output / 'logs/official_sample_labels.csv'
    run_labeler(python, labeler, negbio, labeler / 'sample_reports.csv', sample_output,
                output / 'logs/official_sample.log')
    pd.testing.assert_frame_equal(pd.read_csv(sample_output), pd.read_csv(labeler / 'labeled_reports.csv'),
                                  check_dtype=False)
    print('Official CheXpert sample gate: PASS', flush=True)
    gen_labels_path = output / 'generated_raw_labels.csv'
    run_labeler(python, labeler, negbio, generated_input, gen_labels_path, output / 'logs/generated.log')
    gen_labels = pd.read_csv(gen_labels_path, keep_default_na=False)
    if len(gen_labels) != len(successful) or gen_labels['Reports'].astype(str).tolist() != generated_text:
        raise ValueError('Generated CheXpert text alignment failed')
    print('Generated CheXpert labeling and alignment: PASS', flush=True)
    if any(pathology not in gt_labels or pathology not in gen_labels for pathology in TARGET_PATHOLOGIES):
        raise ValueError('Missing target pathology columns')

    generated_by_index = {index: position for position, (index, _) in enumerate(successful)}
    metric_rows, review_rows = [], []
    rng = np.random.default_rng(args.seed)
    resamples = rng.integers(0, len(cohort), size=(args.bootstrap_iterations, len(cohort)), dtype=np.int32)
    for pathology in TARGET_PATHOLOGIES:
        gt = [parse_label(gt_labels.iloc[i][pathology]) for i in range(len(cohort))]
        pred = [parse_label(gen_labels.iloc[generated_by_index[i]][pathology])
                if i in generated_by_index else None for i in range(len(cohort))]
        paired = [(g, p) for g, p in zip(gt, pred) if g in (0, 1) and p in (0, 1)]
        tp = sum(g == 1 and p == 1 for g, p in paired)
        fp = sum(g == 0 and p == 1 for g, p in paired)
        tn = sum(g == 0 and p == 0 for g, p in paired)
        fn = sum(g == 1 and p == 0 for g, p in paired)
        precision = tp / (tp + fp) if tp + fp else float('nan')
        recall = tp / (tp + fn) if tp + fn else float('nan')
        f1 = f1_counts(tp, fp, fn)
        ci_values = []
        for sample in resamples:
            sample_pairs = [(gt[i], pred[i]) for i in sample if gt[i] in (0, 1) and pred[i] in (0, 1)]
            a = sum(g == 1 and p == 1 for g, p in sample_pairs)
            b = sum(g == 0 and p == 1 for g, p in sample_pairs)
            c = sum(g == 1 and p == 0 for g, p in sample_pairs)
            value = f1_counts(a, b, c)
            if math.isfinite(value):
                ci_values.append(value)
        ci_low = float(np.quantile(ci_values, 0.025)) if ci_values else float('nan')
        ci_high = float(np.quantile(ci_values, 0.975)) if ci_values else float('nan')
        gt_definite = sum(g in (0, 1) for g in gt)
        gt_positive = sum(g == 1 for g in gt)
        positive_failure = sum(g == 1 and i in pred_by_index and i not in generated_by_index
                               for i, g in enumerate(gt))
        positive_not_attempted = sum(g == 1 and i not in pred_by_index for i, g in enumerate(gt))
        positive_uncertain = sum(g == 1 and p == -1 for g, p in zip(gt, pred))
        positive_unmentioned = sum(g == 1 and p is None and i in generated_by_index
                                   for i, (g, p) in enumerate(zip(gt, pred)))
        negative_unknown = sum(g == 0 and p not in (0, 1) and i in generated_by_index
                               for i, (g, p) in enumerate(zip(gt, pred)))
        metric_rows.append({
            'pathology': pathology, 'n_cohort': len(cohort), 'n_generated_success': len(successful),
            'gt_definite': gt_definite, 'gt_positive_all': gt_positive,
            'joint_definite': len(paired), 'coverage_all': len(paired) / len(cohort),
            'coverage_success': len(paired) / len(successful),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': safe_number(precision), 'recall': safe_number(recall),
            'f1': safe_number(f1), 'ci95_low': safe_number(ci_low), 'ci95_high': safe_number(ci_high),
            'bootstrap_defined': len(ci_values),
            'bootstrap_undefined': args.bootstrap_iterations - len(ci_values),
            'positive_generation_failed': positive_failure,
            'positive_generation_not_attempted': positive_not_attempted,
            'positive_pred_uncertain': positive_uncertain,
            'positive_pred_unmentioned': positive_unmentioned,
            'negative_pred_unknown': negative_unknown,
            'paper_table4_f1': PAPER_F1[pathology],
            'paper_table4_ci95_low': PAPER_CI95[pathology][0],
            'paper_table4_ci95_high': PAPER_CI95[pathology][1],
            'delta_local_minus_paper': safe_number(f1 - PAPER_F1[pathology]),
        })
        for i, (g, p) in enumerate(zip(gt, pred)):
            if g in (0, 1) and (p not in (0, 1) or g != p):
                review_rows.append({
                    'cohort_index': i, 'report_id': cohort[i]['report_id'],
                    'image_id': cohort[i]['image_id'], 'pathology': pathology,
                    'gt_label': g, 'generated_label': p, 'generation_status':
                        pred_by_index[i]['status'] if i in pred_by_index else 'not_attempted',
                    'ground_truth_report': cohort[i]['ground_truth_report'],
                    'generated_report': pred_by_index[i]['generated_report'] if i in pred_by_index else '',
                })

    write_csv(output / 'pathology_metrics.csv', metric_rows, list(metric_rows[0]))
    write_csv(output / 'review_queue.csv', review_rows,
              ['cohort_index', 'report_id', 'image_id', 'pathology', 'gt_label', 'generated_label',
               'generation_status', 'ground_truth_report', 'generated_report'])
    defined_f1 = [row['f1'] for row in metric_rows if row['f1'] is not None]
    completion = Counter(record['status'] for record in predictions)
    latency = [float(record['latency_sec']) for _, record in successful if record['latency_sec']]
    peak = [float(record['peak_vram_gb']) for _, record in successful if record['peak_vram_gb']]
    summary = {
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'classification': 'reconstructed Indiana cohort evaluation; not exact paper reproduction',
        'cohort_n': len(cohort), 'generated_success': len(successful),
        'generated_failed': completion['failed'], 'generated_not_attempted': len(cohort) - len(predictions),
        'macro_f1_defined_only': float(np.mean(defined_f1)) if defined_f1 else None,
        'f1_pathologies_defined': len(defined_f1), 'pathologies_total': len(TARGET_PATHOLOGIES),
        'mean_latency_sec_per_success': float(np.mean(latency)) if latency else None,
        'peak_vram_gb_max_device': max(peak) if peak else None,
        'bootstrap_iterations': args.bootstrap_iterations, 'bootstrap_seed': args.seed,
        'bootstrap_unit': 'report/study; resample full cohort, recompute per-pathology joint-definite mask',
        'gt_uncertain_unmentioned_policy': 'exclude from joint-definite F1; count in support/omission metrics',
        'paper_average_f1': 'unresolved method; no paper average reproduced',
        'paper_table4_source': 'https://arxiv.org/pdf/2310.18341v3',
        'paper_indiana_pairs': 3689,
        'paper_cohort_ids': 'not available; local selected cohort has different N and source policy',
        'cohort_sha256': sha256(cohort_path), 'gt_labels_sha256': sha256(gt_dir / 'raw_labels.csv'),
        'generated_labels_sha256': sha256(gen_labels_path),
        'generated_input_sha256': sha256(generated_input),
        'inference_config_sha256': sha256(inference / 'run_config.json'),
        'labeler_revision': LABELER_REVISION, 'negbio_revision': NEGBIO_REVISION,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output / 'logs/pip_freeze.txt').write_text(
        subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True))
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith('_sha256')}, indent=2))


if __name__ == '__main__':
    main()
