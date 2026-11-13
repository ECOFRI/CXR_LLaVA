"""Reproduce saved CheXpert metrics; never regenerate or tune on locked test."""
import argparse
import csv
import json
import re
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from research.common import BASELINE, RESEARCH, ROOT, finish_run, new_run, pseudonym, sha, write_csv, write_json

FIVE = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
POLICIES = ['joint_definite_parsed_only', 'sensitivity_missing_as_negative']


def parse(text):
    match = re.match(r'^\s*(yes|no)\b', text, re.I)
    return int(match.group(1).lower() == 'yes') if match else None


def pattern(text):
    if not text.strip():
        return 'empty'
    if re.search(r'\b(yes|no)\b', text, re.I):
        return 'yes_no_not_at_start'
    if re.search(r'not provide|no specific|does not|cannot|unable', text, re.I):
        return 'no_explicit_binary_information_or_refusal'
    return 'narrative_without_explicit_yes_no'


def statistics(gt, pred):
    mask = np.isfinite(gt) & np.isin(gt, [0, 1]) & np.isfinite(pred)
    y, p = gt[mask], pred[mask]
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fn = int(np.sum((y == 1) & (p == 0)))
    divide = lambda a, b: a / b if b else float('nan')
    return dict(covered=int(mask.sum()), gt_positive_total=int(np.sum(gt == 1)),
                gt_negative_total=int(np.sum(gt == 0)), positive_evaluated=int(np.sum(y == 1)),
                negative_evaluated=int(np.sum(y == 0)), tp=tp, fp=fp, tn=tn, fn=fn,
                precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                specificity=divide(tn, tn + fp), accuracy=divide(tp + tn, len(y)),
                f1=divide(2 * tp, 2 * tp + fp + fn))


def audit(bootstrap=1000):
    run, manifest = new_run('E_AUDIT')
    manifest.update(command=sys.argv, env=sys.prefix, dataset='CheXpert', split='locked_test',
                    checkpoint='CXR-LLaVA released revision from baseline run_config',
                    metric_policy=';'.join(POLICIES), bootstrap_replicates=bootstrap,
                    statistical_unit='patient path key, cluster bootstrap; image-weighted point estimate',
                    source_sha256={n: sha(BASELINE / n) for n in ['cohort.json', 'metrics.csv', 'summary.json', 'run_config.json']},
                    runner_sha256=sha(__file__))
    cohort = json.loads((BASELINE / 'cohort.json').read_text())
    config = json.loads((BASELINE / 'run_config.json').read_text())
    labels = list(cohort[0]['labels'])
    expected_count = len(cohort)
    source = ROOT.parent / 'datasets/CheXpert_test_labels'
    # Recheck source CSV hashes and frozen per-image/study labels independently.
    assert sha(source / 'CheXpert/test_labels.csv') == config['cohort_source']['image_csv_sha256']
    assert sha(source / 'groundtruth.csv') == config['cohort_source']['study_csv_sha256']
    with (source / 'CheXpert/test_labels.csv').open() as stream:
        image_gt = {r['Path']: r for r in csv.DictReader(stream)}
    with (source / 'groundtruth.csv').open() as stream:
        study_gt = {r['Study'].removeprefix('CheXpert-v1.0/'): r for r in csv.DictReader(stream)}
    gt = np.array([[r['labels'][name] for name in labels] for r in cohort], float)
    pred = np.full(gt.shape, np.nan)
    records, reviews, patterns = [], [], Counter()
    source_hashes = {}
    parser_disagreements = 0
    for i, row in enumerate(cohort):
        path = BASELINE / 'cases' / f'{row["index"]:05d}.json'
        item = json.loads(path.read_text())
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
        assert item['status'] == 'success'
        assert item['relative_path'] == row['relative_path'] and item['image_sha256'] == row['image_sha256']
        assert item['study'] == row['study']
        for name in labels:
            assert row['labels'][name] == int(float(image_gt[row['relative_path']][name]))
            assert row['labels'][name] == int(float(study_gt[row['study']][name]))
        records.append(item)
        for j, name in enumerate(labels):
            response = item['responses'][name]
            result = parse(response['text'])
            parser_disagreements += int(result != response['parsed'] or result != item['parsed_labels'].get(name))
            if result is not None:
                pred[i, j] = result
            kind = 'unparsed' if result is None else 'TP' if result == 1 and gt[i, j] == 1 else 'FP' if result == 1 else 'FN' if gt[i, j] == 1 else 'TN'
            category = pattern(response['text']) if result is None else 'explicit_leading_yes_no'
            if result is None:
                patterns[(name, category)] += 1
            reviews.append({'image_id': pseudonym(row['relative_path']), 'study_id': pseudonym(row['study']),
                            'patient_id': pseudonym(str(Path(row['study']).parent)), 'finding': name,
                            'case_type': kind, 'gt': int(gt[i, j]), 'prediction': result,
                            'output_pattern': category, 'response': response['text'],
                            'source_image_path': row['image_path'], 'model_output_path': str(path),
                            'reference_path': str(source / 'groundtruth.csv'),
                            'review_note': 'technical comparison only; no expert clinical adjudication'})
    assert parser_disagreements == 0, 'Saved parser/cache discrepancy requires diagnosis'
    groups = defaultdict(list)
    for i, row in enumerate(cohort):
        groups[str(Path(row['study']).parent)].append(i)
    cluster_indices = list(groups.values())
    rng = np.random.default_rng(42)
    samples = [np.concatenate([cluster_indices[k] for k in rng.integers(0, len(cluster_indices), len(cluster_indices))])
               for _ in range(bootstrap)]
    metric_rows = []
    for policy in POLICIES:
        predictions = pred if policy == POLICIES[0] else np.nan_to_num(pred, nan=0)
        for j, label in enumerate(labels):
            metrics = statistics(gt[:, j], predictions[:, j])
            values = np.array([statistics(gt[idx, j], predictions[idx, j])['f1'] for idx in samples])
            valid = values[np.isfinite(values)]
            ci = np.quantile(valid, [.025, .975]) if len(valid) else [float('nan')] * 2
            metric_rows.append(dict(policy=policy, finding=label, n=expected_count,
                                    prediction_coverage=float(np.isfinite(pred[:, j]).mean()),
                                    missing_predictions=int(np.isnan(pred[:, j]).sum()),
                                    **metrics, ci95_low=float(ci[0]), ci95_high=float(ci[1]),
                                    bootstrap_defined=len(valid)))
    with (BASELINE / 'metrics.csv').open() as stream:
        saved = {r['pathology']: r for r in csv.DictReader(stream)}
    for row in metric_rows[:len(labels)]:
        old = saved[row['finding']]
        for key in ['covered', 'tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'specificity', 'accuracy', 'f1']:
            assert np.isclose(row[key], float(old[key]), atol=1e-12, equal_nan=True), (row['finding'], key)
    summaries = {policy: float(np.mean([r['f1'] for r in metric_rows if r['policy'] == policy and r['finding'] in FIVE])) for policy in POLICIES}
    saved_summary = json.loads((BASELINE / 'summary.json').read_text())
    assert np.isclose(summaries[POLICIES[0]], saved_summary['macro_f1_five_defined'], atol=1e-12)
    write_csv(run / 'metrics_policies.csv', metric_rows)
    write_csv(run / 'unparsed_patterns.csv', [dict(finding=k[0], pattern=k[1], count=v) for k, v in sorted(patterns.items())])
    write_csv(run / 'review_queue.csv', [r for r in reviews if r['case_type'] != 'TN'])
    write_json(run / 'input_case_hashes.json', source_hashes)
    summary = {'status': 'VERIFIED_FROM_LOCAL_ARTIFACTS', 'n_images': expected_count, 'n_studies': len({r['study'] for r in cohort}),
               'n_patient_clusters': len(groups), 'parsed': int(np.isfinite(pred).sum()),
               'unparsed': int(np.isnan(pred).sum()), 'parser_disagreements': parser_disagreements,
               'gt_disagreements': 0, 'macro_f1_five': summaries,
               'statistical_note': 'Patient-cluster bootstrap replaces historical independent-image CIs; point estimates remain image-weighted.',
               'missing_policy_note': 'Sensitivity is not the official baseline and not a parser fix.',
               'taxonomy_note': 'FP/FN are label disagreements; cannot uniquely attribute to encoder, generation or clinical recognition.'}
    write_json(run / 'metrics.json', summary)
    # Select 3 examples in each group without changing prompts or thresholds.
    selected = []
    for kind in ['TP', 'FP', 'FN', 'unparsed']:
        candidates = [r for r in reviews if r['case_type'] == kind and (r['finding'] in FIVE or kind == 'unparsed')]
        candidates.sort(key=lambda r: (r['finding'], r['image_id']))
        used = set()
        for r in candidates:
            if r['image_id'] not in used:
                selected.append(r)
                used.add(r['image_id'])
            if len(used) == 3:
                break
    figdir = RESEARCH / 'figures' / manifest['start_time'][:10] / manifest['run_id']
    figdir.mkdir(parents=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    sheet = Image.new('RGB', (1500, 590 * 4), 'white')
    figure_manifest = []
    for i, r in enumerate(selected):
        x, y = (i % 3) * 500, (i // 3) * 590
        with Image.open(r['source_image_path']) as image:
            image = image.convert('RGB')
            image.thumbnail((480, 350))
            sheet.paste(image, (x + (500 - image.width) // 2, y))
        draw = ImageDraw.Draw(sheet)
        text = (f"{r['case_type']} | ID {r['image_id']}\n{r['finding']}: GT={r['gt']}, pred={r['prediction']}\n"
                + '\n'.join(textwrap.wrap(r['response'], 49))[:650]
                + '\nCXR-LLaVA v2; CLEAR: not run\nSingle frontal input; no reference report.\nTechnical label comparison; not clinical review.')
        draw.multiline_text((x + 10, y + 355), text, font=font, fill='black', spacing=3)
        figure_manifest.append(dict(figure_path=str(figdir / 'case_comparison_sheet.png'), run_id=manifest['run_id'],
                                    image_id=r['image_id'], study_id=r['study_id'], split='locked_test', case_type=r['case_type'],
                                    source_image_path=r['source_image_path'], reference_path=r['reference_path'],
                                    model_output_path=r['model_output_path'], redacted='metadata IDs pseudonymized; original pixels not clinically screened',
                                    review_note='Local technical review only; no boxes or clinical localization'))
    sheet.save(figdir / 'case_comparison_sheet.png')
    write_csv(run / 'selected_cases.csv', selected)
    write_csv(run / 'figure_manifest.csv', figure_manifest)
    manifest.update(metric_summary=summaries, outputs=[str(figdir / 'case_comparison_sheet.png')],
                    next_action='Plot metrics with existing plotting env; inspect compatibility and single-GPU encoder gate')
    finish_run(run, manifest)
    print(json.dumps({'run': str(run), **summary}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', type=int, default=1000)
    audit(parser.parse_args().bootstrap)
