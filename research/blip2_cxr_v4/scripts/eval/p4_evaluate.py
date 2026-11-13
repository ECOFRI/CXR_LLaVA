"""CPU evaluation of frozen P4.3 paired outputs and saved Indiana M2."""
import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

VR = Path(__file__).resolve().parents[2]
ROOT = VR.parents[1]
sys.path.insert(0, str(VR / 'src'))
from vlm_medical.data.manifests import report_sections
from vlm_medical.eval.bootstrap import interval
from vlm_medical.eval.clinical_metrics import PATHOLOGIES, confusion, enrich, label, summarize
from vlm_medical.eval.meteor_metric import score_pairs
from vlm_medical.eval.text_metrics import case_statistics, corpus_scores



def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_contract(path):
    contract = json.loads(Path(path).read_text())
    if contract.get('approval_status') != 'APPROVED_P4_20261023':
        raise ValueError('P4 scientific contract is not approved')
    return contract


def stable_key(row):
    return (row['dataset'], row['split'], row['subject_id'], row['study_id'], row['image_id'])


def paired_cohort(index):
    groups = []
    for name, expected, split in [('mimic_validate', 500, 'validate'), ('indiana_x00', 500, 'external_x00')]:
        item = index['manifests'][name]
        path = ROOT / item['path']
        if digest(path) != item['sha256']:
            raise ValueError('Evaluation manifest hash changed')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != expected or len({stable_key(row) for row in rows}) != expected or any(row['split'] != split for row in rows):
            raise ValueError('Evaluation manifest count/split/ID mismatch')
        groups.append(rows)
    return groups[0][:5] + groups[1][:5] + groups[0][5:] + groups[1][5:]


def reference_for(row, x00_cohort):
    if row['dataset'] == 'mimic':
        path = Path(row['reference_path'])
        if digest(path) != row['reference_file_sha256']:
            raise ValueError('MIMIC reference hash mismatch')
        sections = report_sections(path.read_text(errors='replace'))
        text = '\n'.join(sections[k] for k in ('findings', 'impression') if k in sections)
        if hashlib.sha256(text.encode()).hexdigest() != row['target_text_sha256']:
            raise ValueError('MIMIC reference section linkage mismatch')
        return text
    source = x00_cohort[row['source_row']]
    if (source['report_id'] != row['study_id'] or source['image_id'] != row['image_id']
            or hashlib.sha256(source['ground_truth_report'].encode()).hexdigest() != row['target_text_sha256']):
        raise ValueError('Indiana reference stable-ID mismatch')
    if digest(row['image_path']) != row['expected_image_sha256']:
        raise ValueError('Indiana selected image hash mismatch')
    return source['ground_truth_report']

TEXT_KEYS = ('bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge_l', 'meteor')


def atomic_json(path, value):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    os.replace(str(temporary), str(path))



def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def read_inputs(config, index, run):
    contract = read_contract(config)
    rows = paired_cohort(index)
    saved = [json.loads(line) for line in (run / 'P4.3_pairs.jsonl').read_text().splitlines()]
    if len(saved) > len(rows):
        raise ValueError('Prediction count exceeds frozen cohort')
    adapter_hash = digest(run / 'P4.2_last.pt')
    config_hash = digest(config)
    x00_source = read_csv(rows[5]['reference_source'])
    x00_predictions = {r['report_id']: r for r in read_csv(rows[5]['x00_predictions_source'])}
    records = []
    for i, pair in enumerate(saved):
        row = rows[i]
        if (tuple(pair['stable_key']) != stable_key(row) or pair['config_sha256'] != config_hash
                or pair['adapter_sha256'] != adapter_hash or pair['run_id'] != contract['run_id']
                or pair['study_id'] != row['study_id'] or pair['image_id'] != row['image_id']
                or set(pair['outputs']) != {'M0', 'M1'}
                or any(pair['outputs'][arm]['model_id'] != arm for arm in ('M0', 'M1'))):
            raise ValueError('P4.3 prediction stable-ID/adapter mismatch')
        reference = reference_for(row, x00_source)
        result = {'cohort': row['dataset'], 'split': row['split'], 'subject_id': row['subject_id'],
                  'study_id': row['study_id'], 'image_id': row['image_id'], 'reference': reference,
                  'M0': pair['outputs']['M0'], 'M1': pair['outputs']['M1']}
        if row['dataset'] == 'indiana':
            m2 = x00_predictions.get(row['study_id'])
            if (m2 is None or m2['image_id'] != row['image_id']
                    or m2['ground_truth_report'] != reference
                    or hashlib.sha256(m2['generated_report'].encode()).hexdigest() != row['x00_prediction_sha256']
                    or m2['status'] != 'success'):
                raise ValueError('Saved X00 M2 image/study/reference/prediction mismatch')
            result['M2'] = {'generated_report': m2['generated_report'], 'valid_nonempty': bool(m2['generated_report'].strip())}
        records.append(result)
    return records


def label_texts(records, out, config, timeout_seconds=3600):
    labeler = ROOT / 'model_cache/chexpert_tools/chexpert-labeler'
    negbio = ROOT / 'model_cache/chexpert_tools/NegBio'
    labeler_revision = subprocess.check_output(['git', '-C', str(labeler), 'rev-parse', 'HEAD'], universal_newlines=True).strip()
    negbio_revision = subprocess.check_output(['git', '-C', str(negbio), 'rev-parse', 'HEAD'], universal_newlines=True).strip()
    expected = config['evaluation']
    if labeler_revision != expected['chexpert_labeler_revision'] or negbio_revision != expected['negbio_revision']:
        raise ValueError('CheXpert/NegBio revisions differ from approved contract')
    keys = []
    texts = []
    for i, row in enumerate(records):
        for model in ('reference', 'M0', 'M1', 'M2'):
            if model == 'M2' and 'M2' not in row:
                continue
            text = row['reference'] if model == 'reference' else row[model]['generated_report']
            keys.append((i, model))
            texts.append(text)
    label_input = out / 'labeler_input.csv'
    with os.fdopen(os.open(str(label_input), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerows([[text] for text in texts])
    label_input.chmod(0o600)
    label_output = out / 'labeler_output.csv'
    label_output.touch(mode=0o600, exist_ok=False)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(negbio)
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    with (out / 'labeler.log').open('w') as log:
        subprocess.run(['/home/student4/anaconda3/envs/chexpert-label/bin/python', str(labeler / 'label.py'),
                        '--reports_path', str(label_input), '--output_path', str(label_output)], cwd=labeler,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout_seconds)
    label_output.chmod(0o600)
    labeled = read_csv(label_output)
    if len(labeled) != len(keys) or any(row['Reports'] != text for row, text in zip(labeled, texts)):
        raise ValueError('CheXpert output text/order alignment mismatch')
    lookup = {key: row for key, row in zip(keys, labeled)}
    return lookup, {'chexpert_labeler_revision': labeler_revision, 'negbio_revision': negbio_revision,
                    'label_input_sha256': digest(label_input), 'label_output_sha256': digest(label_output)}


def point_text(records, model, indices, meteor_scores, text_stats):
    stats = [text_stats[model][i] for i in indices]
    scores = corpus_scores(stats)
    scores['meteor'] = sum(meteor_scores[model][i] for i in indices) / len(indices)
    return scores


def point_clinical(indices, model, labeled, policy):
    per_class = {}
    for pathology in PATHOLOGIES:
        pairs = [(label(labeled[(i, 'reference')][pathology], policy),
                  label(labeled[(i, model)][pathology], policy)) for i in indices]
        per_class[pathology] = enrich(confusion(pairs), len(indices))
        per_class[pathology]['reference_blank'] = sum(str(labeled[(i, 'reference')][pathology]).strip() == '' for i in indices)
        per_class[pathology]['reference_uncertain'] = sum(str(labeled[(i, 'reference')][pathology]).strip() == '-1' for i in indices)
        per_class[pathology]['generated_blank'] = sum(str(labeled[(i, model)][pathology]).strip() == '' for i in indices)
        per_class[pathology]['generated_uncertain'] = sum(str(labeled[(i, model)][pathology]).strip() == '-1' for i in indices)
        if per_class[pathology]['f1'] is None:
            per_class[pathology]['f1_reason'] = 'no jointly definite TP/FP/FN denominator'
    summary = summarize(per_class)
    return {'per_class': per_class, **summary,
            'paired_label_count': sum(item['paired'] for item in per_class.values()),
            'positive_support': sum(item['support'] for item in per_class.values()),
            'undefined_reason': 'no jointly definite positive/error denominator' if summary['micro_f1'] is None else None}


def bootstrap_dataset(records, indices, models, labeled, meteor_scores, text_stats, config):
    by_cluster = defaultdict(list)
    for i in indices:
        row = records[i]
        key = row['subject_id'] if row['cohort'] == 'mimic' else row['study_id']
        by_cluster[key].append(i)
    clusters = list(by_cluster)
    rng = random.Random(config['evaluation']['bootstrap_seed'])
    text_values = {model: {name: [] for name in TEXT_KEYS} for model in models}
    clinical_values = {model: {'micro_f1': [], 'macro_f1': []} for model in models}
    deltas = {name: [] for name in TEXT_KEYS + ('micro_f1', 'macro_f1')}
    for _ in range(config['evaluation']['bootstrap_reps']):
        sample = [i for _ in clusters for i in by_cluster[rng.choice(clusters)]]
        points = {}
        for model in models:
            text = point_text(records, model, sample, meteor_scores, text_stats)
            clinical = point_clinical(sample, model, labeled, 'historical')
            points[model] = {**text, 'micro_f1': clinical['micro_f1'], 'macro_f1': clinical['macro_f1']}
            for key in TEXT_KEYS:
                text_values[model][key].append(text[key])
            for key in ('micro_f1', 'macro_f1'):
                clinical_values[model][key].append(clinical[key])
        for key in deltas:
            a, b = points['M0'][key], points['M1'][key]
            deltas[key].append(None if a is None or b is None else b - a)
    return ({model: {key: interval(values) for key, values in {**text_values[model], **clinical_values[model]}.items()}
             for model in models}, {key: interval(values) for key, values in deltas.items()}, len(clusters))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest-index', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--evaluation-dir-name', choices=('P4.3_evaluation', 'P4.3_evaluation_full'),
                        default='P4.3_evaluation')
    parser.add_argument('--labeler-timeout-seconds', type=int, default=3600)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.labeler_timeout_seconds <= 0:
        parser.error('--labeler-timeout-seconds must be positive')
    config = read_contract(args.config)
    index = json.loads(args.manifest_index.read_text())
    if not index['approved']:
        raise ValueError('Evaluation index not approved')
    run = args.run_dir.resolve()
    if (VR / 'runs').resolve() not in run.parents:
        raise ValueError('Evaluation run dir outside V4')
    if args.dry_run:
        print(json.dumps({'status': 'DRY_RUN_PASS', 'bootstrap_reps': config['evaluation']['bootstrap_reps'],
                          'primary': config['evaluation']['primary_lexical_metric'], 'gpu_used': False}))
        return
    stage = json.loads((run / 'P4.3_status.json').read_text())
    if stage.get('status') not in ('PASS', 'PARTIAL') or stage.get('validity_gate') != 'PASS':
        raise RuntimeError('P4.3 validity gate and completed paired results required')
    records = read_inputs(args.config, index, run)
    if not records or len(records) != stage['completed_studies']:
        raise ValueError('P4.3 status/prediction count mismatch')
    out = run / args.evaluation_dir_name
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    out.chmod(0o700)
    if (out / 'metrics.json').exists():
        raise FileExistsError('Existing P4.3 evaluation must be preserved')
    labeled, label_provenance = label_texts(records, out, config, args.labeler_timeout_seconds)
    meteor_scores = {}
    text_stats = {}
    for model in ('M0', 'M1', 'M2'):
        meteor_scores[model] = {}
        selected = [i for i, row in enumerate(records) if model in row and row[model]['valid_nonempty']]
        values = score_pairs((records[i]['reference'], records[i][model]['generated_report']) for i in selected)
        meteor_scores[model] = dict(zip(selected, values))
        text_stats[model] = {i: case_statistics(records[i]['reference'], records[i][model]['generated_report']) for i in selected}
    result = {'status': 'PASS' if stage['status'] == 'PASS' and len(records) == 1000 else 'PARTIAL',
              'classification': 'metric-complete research evaluation' if len(records) == 1000 else 'paired-intersection partial research evaluation',
              'planned_studies_by_dataset': {'mimic': 500, 'indiana': 500},
              'completed_pairs': len(records), 'paired_valid_counts': {}, 'datasets': {},
              'protocol': {'primary': 'corpus BLEU-4 M1 minus M0 separately by dataset',
                           'clinical_policy': 'historical jointly definite; blank/uncertain excluded',
                           'secondary': 'exploratory unadjusted', 'bootstrap_reps': 1000, 'bootstrap_seed': 42,
                           'ci': 'paired patient/study cluster percentile 95%', 'm2': 'historical X00 output, different model protocol'},
              'labeler': label_provenance, 'input_sha256': {'pairs': digest(run / 'P4.3_pairs.jsonl'),
                                                           'config': digest(args.config), 'adapter': digest(run / 'P4.2_last.pt')}}
    for dataset in ('mimic', 'indiana'):
        dataset_indices = [i for i, row in enumerate(records) if row['cohort'] == dataset]
        valid = [i for i in dataset_indices if records[i]['M0']['valid_nonempty'] and records[i]['M1']['valid_nonempty']]
        result['paired_valid_counts'][dataset] = len(valid)
        if not valid:
            result['datasets'][dataset] = {'status': 'UNDEFINED', 'reason': 'No valid paired generated reports'}
            continue
        models = ['M0', 'M1'] + (['M2'] if dataset == 'indiana' else [])
        points = {}
        for model in models:
            selected = [i for i in valid if model in records[i] and records[i][model]['valid_nonempty']]
            if not selected:
                continue
            points[model] = {'n': len(selected), 'text': point_text(records, model, selected, meteor_scores, text_stats),
                             'clinical_historical': point_clinical(selected, model, labeled, 'historical'),
                             'clinical_sensitivity': {policy: point_clinical(selected, model, labeled, policy)
                                                      for policy in ('uncertain_as_positive', 'uncertain_as_negative', 'blank_as_negative')},
                             'empty_rate_all_completed': sum(not records[i][model]['valid_nonempty'] for i in dataset_indices) / len(dataset_indices),
                             'mean_generated_tokens': sum(records[i][model].get('generated_tokens', 0) for i in selected) / len(selected) if model != 'M2' else None,
                             'mean_latency_seconds': sum(records[i][model].get('latency_seconds', 0) for i in selected) / len(selected) if model != 'M2' else None,
                             'hit_max_length_count': sum(records[i][model].get('hit_max_length', False) for i in selected) if model != 'M2' else None}
        ci, delta_ci, clusters = bootstrap_dataset(records, valid, ['M0', 'M1'], labeled, meteor_scores, text_stats, config)
        delta = {key: points['M1']['text'][key] - points['M0']['text'][key] for key in TEXT_KEYS}
        for key in ('micro_f1', 'macro_f1'):
            a = points['M0']['clinical_historical'][key]
            b = points['M1']['clinical_historical'][key]
            delta[key] = None if a is None or b is None else b - a
        result['datasets'][dataset] = {'status': 'PASS' if len(valid) == 500 else 'PARTIAL',
                                       'completed': len(dataset_indices), 'valid_paired': len(valid),
                                       'bootstrap_clusters': clusters, 'models': points,
                                       'm1_minus_m0': {key: {'value': delta[key], 'ci95': delta_ci[key]}
                                                       for key in delta}, 'model_ci95': ci}
    train_updates = run / 'P4.2_updates.jsonl'
    if train_updates.is_file():
        updates = [json.loads(line) for line in train_updates.read_text().splitlines()]
        result['training_target_truncated_count'] = sum(row['target_tokens_before_truncation'] > row['target_tokens_used'] for row in updates)
    atomic_json(out / 'metrics.json', result)
    atomic_json(out / 'status.json', {'status': result['status'], 'completed_pairs': len(records),
                                      'valid_pairs': result['paired_valid_counts']})
    print(json.dumps({'status': result['status'], 'completed_pairs': len(records), 'valid_pairs': result['paired_valid_counts']}))


if __name__ == '__main__':
    main()
