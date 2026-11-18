"""CPU-only Indiana X00 re-score from saved predictions and CheXpert labels."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VR = Path(__file__).resolve().parents[2]
ROOT = VR.parents[1]
sys.path.insert(0, str(VR / 'src'))

from vlm_medical.eval.bootstrap import interval, resamples
from vlm_medical.eval.clinical_metrics import PATHOLOGIES, POLICIES, confusion, enrich, label, summarize
from vlm_medical.eval.text_metrics import case_statistics, corpus_scores


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_config_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def audit(predictions, cohort, selected, gt_labels, generated_labels):
    n = len(cohort)
    if not all(len(rows) == n for rows in (predictions, selected, gt_labels, generated_labels)):
        raise ValueError('Prediction, cohort, selection and label row counts differ')
    if len({row['report_id'] for row in cohort}) != n or len({row['image_id'] for row in cohort}) != n:
        raise ValueError('Duplicate report or image key in cohort')
    if len({int(row['cohort_index']) for row in predictions}) != n:
        raise ValueError('Duplicate or missing prediction cohort index')
    prediction_by_index = {int(row['cohort_index']): row for row in predictions}
    if set(prediction_by_index) != set(range(n)):
        raise ValueError('Prediction indices do not cover cohort')
    ordered = [prediction_by_index[i] for i in range(n)]
    for i, (pred, ref, source, gt, gen) in enumerate(zip(ordered, cohort, selected, gt_labels, generated_labels)):
        for key in ('report_id', 'image_id', 'ground_truth_report'):
            if pred[key] != ref[key] or source[key] != ref[key]:
                raise ValueError(f'Reference source mismatch at cohort index {i}')
        if gt['Reports'] != ref['ground_truth_report'] or gen['Reports'] != pred['generated_report']:
            raise ValueError(f'CheXpert label text alignment mismatch at cohort index {i}')
    status = Counter(row['status'] for row in ordered)
    valid = [i for i, row in enumerate(ordered) if row['status'] == 'success'
             and row['generated_report'].strip() and row['ground_truth_report'].strip()]
    reference_policy = Counter(row['reference_policy'] for row in cohort)
    return ordered, valid, {'cohort_n': n, 'status_counts': dict(status),
                            'empty_generated': sum(not row['generated_report'].strip() for row in ordered),
                            'empty_reference': sum(not row['ground_truth_report'].strip() for row in ordered),
                            'unique_generated_texts': len({row['generated_report'] for row in ordered}),
                            'unique_reference_texts': len({row['ground_truth_report'] for row in ordered}),
                            'unique_report_keys': n, 'unique_image_keys': n,
                            'reference_matches_selection': n, 'reference_matches_prediction': n,
                            'view_counts': dict(Counter(row['view_label'] for row in cohort)),
                            'reference_section_policies': dict(reference_policy),
                            'patient_key': 'NOT_FOUND; report_id used as study-level bootstrap key'}


def clinical_rows(indices, gt_labels, generated_labels, policy):
    output = {}
    for pathology in PATHOLOGIES:
        pairs = [(label(gt_labels[i][pathology], policy), label(generated_labels[i][pathology], policy))
                 for i in indices]
        output[pathology] = enrich(confusion(pairs), len(indices))
    return output


def clinical_summary_or_na(rows):
    paired = sum(row['paired'] for row in rows.values())
    positive_support = sum(row['support'] for row in rows.values())
    if paired == 0 or positive_support == 0:
        return {'micro_f1': None, 'macro_f1': None,
                'reason': 'no paired reference-positive labels; clinical F1 undefined',
                'paired_labels': paired, 'paired_positive_support': positive_support}
    result = summarize(rows)
    return {'micro_f1': result['micro_f1'], 'macro_f1': result['macro_f1'],
            'reason': None, 'paired_labels': paired, 'paired_positive_support': positive_support}


def bootstrap_metrics(stats, valid, gt_labels, generated_labels, config, policies):
    text_values = {key: [] for key in ('bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge_l')}
    clinical_values = {p: {'micro_f1': [], 'macro_f1': [],
                           'per_pathology': {pathology: [] for pathology in PATHOLOGIES}}
                       for p in policies}
    for sample in resamples(len(valid), config['bootstrap_reps'], config['seed']):
        sample_stats = [stats[j] for j in sample]
        scores = corpus_scores(sample_stats)
        for key in text_values:
            text_values[key].append(scores[key])
        sample_indices = [valid[j] for j in sample]
        for policy in policies:
            rows = clinical_rows(sample_indices, gt_labels, generated_labels, policy)
            summary = summarize(rows)
            for key in ('micro_f1', 'macro_f1'):
                clinical_values[policy][key].append(summary[key])
            for pathology in PATHOLOGIES:
                clinical_values[policy]['per_pathology'][pathology].append(rows[pathology]['f1'])
    return ({key: interval(values) for key, values in text_values.items()},
            {p: {'micro_f1': interval(metrics['micro_f1']),
                 'macro_f1': interval(metrics['macro_f1']),
                 'per_pathology': {name: interval(values) for name, values in metrics['per_pathology'].items()}}
             for p, metrics in clinical_values.items()})


def complete_pathology_intervals(out, paths, config, policies):
    metrics_path = out / 'metrics.json'
    metrics = json.loads(metrics_path.read_text())
    if all('ci95' in metrics['clinical'][policy]['per_pathology'][pathology]
           for policy in policies for pathology in PATHOLOGIES):
        return
    gt_labels = read_csv(paths['gt_labels'])
    generated_labels = read_csv(paths['generated_labels'])
    valid = [int(row['cohort_index']) for row in read_csv(out / 'per_case.csv')]
    values = {policy: {pathology: [] for pathology in PATHOLOGIES} for policy in policies}
    for sample in resamples(len(valid), config['bootstrap_reps'], config['seed']):
        sample_indices = [valid[j] for j in sample]
        for policy in policies:
            rows = clinical_rows(sample_indices, gt_labels, generated_labels, policy)
            for pathology in PATHOLOGIES:
                values[policy][pathology].append(rows[pathology]['f1'])
    for policy in policies:
        intervals = {name: interval(samples) for name, samples in values[policy].items()}
        metrics['clinical'][policy]['ci95']['per_pathology'] = intervals
        for pathology in PATHOLOGIES:
            metrics['clinical'][policy]['per_pathology'][pathology]['ci95'] = intervals[pathology]
    metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')
    benchmark = VR / 'results/benchmarks/indiana/baseline_metrics.json'
    if benchmark.exists():
        summary = json.loads(benchmark.read_text())
        if summary.get('run_id') == out.name:
            benchmark.write_text(json.dumps({'run_id': out.name, **metrics}, indent=2) + '\n')


def evaluate_x02(config, args):
    experiment = json.loads(resolve_config_path(config['experiment_config']).read_text())
    sys.path.insert(0, str(VR / 'scripts/experiments'))
    from infer_blip2 import sample_cases
    cases = sample_cases(experiment)
    if args.dry_run:
        print(json.dumps({'status': 'DRY_RUN_PASS', 'cohort_n': len(cases),
                          'predictions_required': True, 'gpu_job_started': False}))
        return
    if args.predictions is None or not args.predictions.is_file():
        raise FileNotFoundError('X02 predictions.jsonl required')
    from vlm_medical.eval.meteor_metric import score_pairs, sha256 as meteor_sha
    import nltk
    if config['tokenizer'] != 'lowercase_ascii_alnum_regex_v1' or config['clinical_policy'] != 'historical':
        raise ValueError('X02 evaluation protocol mismatch')
    if config['seed'] != 42 or config['bootstrap_reps'] < 1000:
        raise ValueError('X02 requires study bootstrap seed 42 and at least 1000 reps')
    saved = [json.loads(line) for line in args.predictions.open(encoding='utf-8')]
    by_index = {row['cohort_index']: row for row in saved}
    expected = {index for index, _ in cases}
    if len(saved) != len(cases) or len(by_index) != len(saved) or set(by_index) != expected:
        raise ValueError('X02 prediction cohort mismatch')
    predictions = []
    references = []
    for index, row in cases:
        prediction = by_index[index]
        if (prediction['case_key_sha256'] != hashlib.sha256(row['report_id'].encode()).hexdigest()
                or prediction['prompt_version'] != experiment['prompt_version']
                or prediction['status'] != 'success' or not prediction['generated_report'].strip()):
            raise ValueError('X02 prediction validity or prompt mismatch')
        predictions.append(prediction['generated_report'])
        references.append(row['ground_truth_report'])
    all_reference_labels = read_csv(resolve_config_path(config['gt_labels']))
    selected_reference_labels = [all_reference_labels[index] for index, _ in cases]
    if len(all_reference_labels) <= max(expected) or any(
            label_row['Reports'] != reference for label_row, reference in zip(selected_reference_labels, references)):
        raise ValueError('X02 reference-label alignment mismatch')
    out = args.output_dir.resolve() if args.output_dir is not None else None
    if out is None or VR not in out.parents:
        raise ValueError('X02 evaluation output must be inside V4 root')
    if out.exists() and (out / 'metrics.json').exists() and not args.resume:
        raise FileExistsError(out / 'metrics.json')
    out.mkdir(parents=True, exist_ok=True)
    labeler_input = out / 'labeler_input.csv'
    with labeler_input.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerows([[text] for text in predictions])
    labeler_output = out / 'generated_raw_labels.csv'
    labeler = ROOT / 'model_cache/chexpert_tools/chexpert-labeler'
    negbio = ROOT / 'model_cache/chexpert_tools/NegBio'
    historical = json.loads(resolve_config_path(config['historical_labeler_summary']).read_text())
    labeler_revision = subprocess.check_output(['git', '-C', str(labeler), 'rev-parse', 'HEAD'],
                                               universal_newlines=True, timeout=20).strip()
    negbio_revision = subprocess.check_output(['git', '-C', str(negbio), 'rev-parse', 'HEAD'],
                                              universal_newlines=True, timeout=20).strip()
    if (labeler_revision != historical['labeler_revision']
            or negbio_revision != historical['negbio_revision']):
        raise ValueError('CheXpert/NegBio revision differs from X00')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(negbio)
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    with (out / 'labeler.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, str(labeler / 'label.py'), '--reports_path', str(labeler_input),
                        '--output_path', str(labeler_output)], cwd=labeler, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    generated_labels = read_csv(labeler_output)
    if len(generated_labels) != len(cases) or any(
            row['Reports'] != text for row, text in zip(generated_labels, predictions)):
        raise ValueError('X02 generated-label alignment mismatch')
    stats = [case_statistics(ref, hyp) for ref, hyp in zip(references, predictions)]
    meteor = score_pairs(zip(references, predictions))
    text_point = corpus_scores(stats)
    text_point['meteor'] = sum(meteor) / len(meteor)
    clinical_rows_point = clinical_rows(range(len(cases)), selected_reference_labels,
                                        generated_labels, 'historical')
    clinical_point = clinical_summary_or_na(clinical_rows_point)
    text_samples = {name: [] for name in text_point}
    clinical_samples = {name: [] for name in ('micro_f1', 'macro_f1')}
    for sample in resamples(len(cases), config['bootstrap_reps'], config['seed']):
        score = corpus_scores([stats[index] for index in sample])
        score['meteor'] = sum(meteor[index] for index in sample) / len(sample)
        for name in text_samples:
            text_samples[name].append(score[name])
        summary = clinical_summary_or_na(clinical_rows(sample, selected_reference_labels,
                                                       generated_labels, 'historical'))
        for name in clinical_samples:
            clinical_samples[name].append(summary[name])
    result = {'status': 'PASS', 'classification': 'exploratory smoke evaluation; n=10, no benchmark claim',
              'n_studies': len(cases), 'n_valid': len(cases), 'prompt_version': experiment['prompt_version'],
              'text': {name: {'value': value, 'ci95': interval(text_samples[name])}
                       for name, value in text_point.items()},
              'clinical_historical': {name: {'value': clinical_point[name], 'ci95': interval(clinical_samples[name]),
                                            'reason': clinical_point['reason'] if clinical_point[name] is None else None}
                                      for name in clinical_samples},
              'clinical_coverage': {'paired_labels': clinical_point['paired_labels'],
                                    'paired_positive_support': clinical_point['paired_positive_support'],
                                    'generated_definite_labels': sum(label(row[pathology], 'historical') is not None
                                                                     for row in generated_labels for pathology in PATHOLOGIES),
                                    'reference_definite_positives': sum(label(row[pathology], 'historical') == 1
                                                                        for row in selected_reference_labels
                                                                        for pathology in PATHOLOGIES)},
              'clinical_per_pathology': clinical_rows_point,
              'protocol': {'tokenizer': config['tokenizer'], 'clinical_policy': 'historical',
                           'bootstrap_unit': 'report/study', 'seed': config['seed'],
                           'bootstrap_reps': config['bootstrap_reps'], 'nltk_version': nltk.__version__,
                           'wordnet_sha256': meteor_sha(VR / '.cache/nltk/corpora/wordnet.zip'),
                           'labeler': 'existing offline CheXpert CPU labeler',
                           'labeler_revision': labeler_revision, 'negbio_revision': negbio_revision},
              'input_sha256': {'predictions': sha256(args.predictions),
                               'reference_labels': sha256(resolve_config_path(config['gt_labels']))},
              'gpu_used_for_evaluation': False}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'status.json').write_text(json.dumps({'status': 'PASS', 'n_valid': len(cases)}) + '\n')
    print(json.dumps({'status': 'PASS', 'n_valid': len(cases), 'classification': 'exploratory'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=VR / 'configs/eval/indiana_x00.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--meteor-only', action='store_true', help='New derived METEOR run; needs NLTK WordNet')
    parser.add_argument('--predictions', type=Path, help='Saved X02 predictions.jsonl')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get('source_type') == 'x02_smoke':
        evaluate_x02(config, args)
        return
    if config['tokenizer'] != 'lowercase_ascii_alnum_regex_v1':
        raise ValueError('Unsupported tokenizer')
    if config['bootstrap_reps'] != 1000 or config['seed'] != 42:
        raise ValueError('X00 requires 1000 bootstrap repetitions and seed 42')
    policies = config['policies']
    if set(policies) != set(POLICIES):
        raise ValueError('X00 requires historical and all three sensitivity policies')
    source = resolve_config_path(config['source_dir'])
    paths = {'predictions': source / 'inference/predictions.csv',
             'cohort': source / 'inference/cohort.csv',
             'selection': resolve_config_path(config['selection_cohort']),
             'gt_labels': resolve_config_path(config['gt_labels']),
             'generated_labels': resolve_config_path(config['generated_labels'])}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'{key}: {path}')
    if args.meteor_only and not args.dry_run:
        import nltk
        from vlm_medical.eval.meteor_metric import score_pairs, sha256 as resource_sha
        predictions, cohort, selected, gt_labels, generated_labels = [read_csv(paths[key]) for key in paths]
        ordered, valid, audit_result = audit(predictions, cohort, selected, gt_labels, generated_labels)
        scores = score_pairs([(ordered[i]['ground_truth_report'], ordered[i]['generated_report']) for i in valid])
        ci = interval([sum(scores[j] for j in sample) / len(sample)
                       for sample in resamples(len(valid), config['bootstrap_reps'], config['seed'])])
        out = args.output_dir or (VR / 'runs' / ('X00_indiana_meteor_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
        if out.exists():
            raise FileExistsError(out)
        out.mkdir(parents=True)
        resource = VR / '.cache/nltk/corpora/wordnet.zip'
        result = {'status': 'PASS', 'metric': 'meteor', 'value': sum(scores) / len(scores), 'ci95': ci,
                  'n_used': len(valid), 'n_excluded': len(ordered)-len(valid),
                  'tokenizer': config['tokenizer'], 'normalization': 'lowercase ASCII alphanumeric runs',
                  'scorer': 'NLTK 3.6.7 meteor_score.py vendored source; exact/stem/WordNet synonym',
                  'installed_nltk_version': nltk.__version__, 'wordnet_sha256': resource_sha(resource),
                  'scorer_sha256': resource_sha(VR / 'src/vlm_medical/eval/vendor/nltk367_meteor_score.py'),
                  'bootstrap': {'unit': 'report/study', 'seed': 42, 'reps': 1000},
                  'source_sha256': {k: sha256(v) for k,v in paths.items()},
                  'model_loaded': False, 'gpu_used': False}
        (out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        (out / 'status.json').write_text(json.dumps({'status': 'PASS'}) + '\n')
        print(json.dumps({'status': 'PASS', 'meteor': result['value'], 'n_used': len(valid), 'output_dir': str(out)}))
        return
    if args.dry_run:
        print(json.dumps({'status': 'DRY_RUN_PASS', 'source_files': list(paths),
                          'bootstrap_reps': config['bootstrap_reps'], 'seed': config['seed']}))
        return
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = args.output_dir or (VR / 'runs' / f'X00_indiana_rescore_{timestamp}')
    if out.exists():
        if args.resume and (out / 'status.json').exists():
            state = json.loads((out / 'status.json').read_text())
            if state.get('status') == 'COMPLETE':
                complete_pathology_intervals(out, paths, config, policies)
                print(json.dumps({'status': 'COMPLETE', 'output_dir': str(out), 'resumed': True}))
                return
        raise FileExistsError(f'Choose a new output directory: {out}')
    out.mkdir(parents=True)
    (out / 'status.json').write_text(json.dumps({'status': 'RUNNING'}) + '\n')
    predictions, cohort, selected, gt_labels, generated_labels = [read_csv(paths[key]) for key in paths]
    ordered, valid, audit_result = audit(predictions, cohort, selected, gt_labels, generated_labels)
    stats = [case_statistics(ordered[i]['ground_truth_report'], ordered[i]['generated_report']) for i in valid]
    text_scores = corpus_scores(stats)
    text_ci, clinical_ci = bootstrap_metrics(stats, valid, gt_labels, generated_labels, config, policies)
    clinical = {}
    for policy in policies:
        by_class = clinical_rows(valid, gt_labels, generated_labels, policy)
        for pathology in PATHOLOGIES:
            by_class[pathology]['ci95'] = clinical_ci[policy]['per_pathology'][pathology]
        clinical[policy] = {**summarize(by_class), 'ci95': clinical_ci[policy], 'per_pathology': by_class}
    per_case = []
    for i, stat in zip(valid, stats):
        per_case.append({'cohort_index': i, 'report_key_sha256': hashlib.sha256(cohort[i]['report_id'].encode()).hexdigest(),
                         'reference_tokens': stat['ref_len'], 'generated_tokens': stat['hyp_len'],
                         'rouge_l': stat['rouge_l'], 'status': 'included'})
    write_csv(out / 'per_case.csv', per_case, list(per_case[0]))
    metrics = {'classification': 'metric-complete local rescore; not paper reproduction or clinical validation',
               'n_source': len(ordered), 'n_used': len(valid), 'n_excluded': len(ordered) - len(valid),
               'excluded_reason': 'failed status or empty reference/generated text',
               'text': {key: {'value': value, 'ci95': text_ci[key]} for key, value in text_scores.items()},
               'meteor': {'value': None, 'reason': 'NLTK WordNet unavailable offline; no download'},
               'text_protocol': {'tokenizer': config['tokenizer'], 'normalization': 'lowercase; ASCII alphanumeric runs; punctuation stripped',
                                 'bleu': 'corpus modified 1-4 gram precision; clipped counts; no smoothing; closest single reference brevity penalty',
                                 'rouge_l': 'mean per-report LCS F1; no stemming'},
               'clinical': clinical, 'bootstrap': {'unit': 'report/study; patient key unavailable', 'seed': 42, 'reps': 1000,
                                                   'ci': 'percentile 2.5/97.5; paired full-cohort resampling'},
               'audit': audit_result, 'auroc': 'N/A; no continuous scores'}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    count_fields = {'tp': '', 'fp': '', 'fn': '', 'tn': '', 'support': '', 'coverage': ''}
    flat = [{'metric': key, 'policy': 'text', 'value': value['value'], 'ci95_low': value['ci95'][0],
             'ci95_high': value['ci95'][1], **count_fields} for key, value in metrics['text'].items()]
    flat.append({'metric': 'meteor', 'policy': 'text', 'value': 'N/A', 'ci95_low': '', 'ci95_high': '', **count_fields})
    for policy, summary in clinical.items():
        for metric in ('micro_f1', 'macro_f1'):
            ci = summary['ci95'][metric]
            flat.append({'metric': metric, 'policy': policy, 'value': summary[metric],
                         'ci95_low': ci[0], 'ci95_high': ci[1], **count_fields})
        for pathology, row in summary['per_pathology'].items():
            ci = row['ci95']
            flat.append({'metric': f'f1:{pathology}', 'policy': policy, 'value': row['f1'],
                         'ci95_low': ci[0], 'ci95_high': ci[1],
                         **{key: row[key] for key in count_fields}})
    write_csv(out / 'metrics.csv', flat, list(flat[0]))
    manifest = {'run_id': out.name, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'config': str(args.config), 'config_sha256': sha256(args.config),
                'config_snapshot': str(out / 'config.yaml'),
                'inputs': {key: {'path': str(path), 'sha256': sha256(path)} for key, path in paths.items()},
                'environment': {'python': sys.version.split()[0], 'executable': sys.executable},
                'seed': 42, 'bootstrap_reps': 1000, 'model_loaded': False, 'gpu_used': False,
                'labeler_rerun': False, 'git_commit_historical': (source / 'snapshots/git_commit.txt').read_text().strip(),
                'historical_run_config_sha256': sha256(source / 'inference/run_config.json'),
                'historical_model_revision': json.loads((source / 'inference/run_config.json').read_text())['model_revision'],
                'historical_labeler_revision': json.loads((source / 'metrics/summary.json').read_text())['labeler_revision'],
                'historical_negbio_revision': json.loads((source / 'metrics/summary.json').read_text())['negbio_revision']}
    (out / 'config.yaml').write_text(args.config.read_text())
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    benchmark = VR / 'results/benchmarks/indiana'
    benchmark.mkdir(parents=True, exist_ok=True)
    summary_json = benchmark / 'baseline_metrics.json'
    summary_csv = benchmark / 'baseline_metrics.csv'
    if summary_json.exists() or summary_csv.exists():
        if not (summary_json.exists() and summary_csv.exists()):
            raise FileExistsError('Incomplete existing benchmark summary')
        existing = json.loads(summary_json.read_text())
        existing.pop('run_id', None)
        if existing != metrics:
            raise FileExistsError('Existing benchmark summary differs; choose a new benchmark name')
    else:
        summary_json.write_text(json.dumps({'run_id': out.name, **metrics}, indent=2) + '\n')
        write_csv(summary_csv, flat, list(flat[0]))
    (out / 'status.json').write_text(json.dumps({'status': 'COMPLETE', 'n_used': len(valid),
                                                'n_excluded': len(ordered) - len(valid)}, indent=2) + '\n')
    print(json.dumps({'status': 'COMPLETE', 'n_used': len(valid), 'n_excluded': len(ordered) - len(valid),
                      'historical_macro_f1': clinical['historical']['macro_f1'], 'output_dir': str(out)}))


if __name__ == '__main__':
    main()
