"""CPU-only paired M3 versus saved P4 M1 evaluation."""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from p5_common import ROOT, VR, contract_and_index, digest
from vlm_medical.eval.bootstrap import interval
from vlm_medical.eval.clinical_metrics import PATHOLOGIES, confusion, enrich, label, summarize
from vlm_medical.eval.meteor_metric import score_pairs
from vlm_medical.eval.text_metrics import case_statistics, corpus_scores, tokenize
from vlm_medical.train.p4_runtime import atomic_json

sys.path.insert(0, str(VR / 'scripts/eval'))
from p4_evaluate import label_texts, read_csv, read_inputs


def existing_labels(records, out, p4_config):
    """Reuse a complete, aligned CheXpert result after a later analysis failure."""
    input_path = out / 'labeler_input.csv'
    output_path = out / 'labeler_output.csv'
    if not input_path.exists() and not output_path.exists():
        return None
    if not input_path.exists() or not output_path.exists():
        raise ValueError('Incomplete saved CheXpert input/output pair')
    keys = []
    texts = []
    for i, row in enumerate(records):
        for model in ('reference', 'M0', 'M1', 'M2'):
            if model == 'M2' and 'M2' not in row:
                continue
            keys.append((i, model))
            texts.append(row['reference'] if model == 'reference' else row[model]['generated_report'])
    with input_path.open(newline='', encoding='utf-8') as stream:
        saved_input = [line[0] for line in csv.reader(stream)]
    labeled = read_csv(output_path)
    if saved_input != texts or len(labeled) != len(texts) or any(
            row['Reports'] != text for row, text in zip(labeled, texts)):
        raise ValueError('Saved CheXpert labels do not align with current paired reports')
    labeler = ROOT / 'model_cache/chexpert_tools/chexpert-labeler'
    negbio = ROOT / 'model_cache/chexpert_tools/NegBio'
    revisions = {
        'chexpert_labeler_revision': subprocess.check_output(
            ['git', '-C', str(labeler), 'rev-parse', 'HEAD'], text=True).strip(),
        'negbio_revision': subprocess.check_output(
            ['git', '-C', str(negbio), 'rev-parse', 'HEAD'], text=True).strip(),
    }
    if any(revisions[key] != p4_config['evaluation'][key] for key in revisions):
        raise ValueError('Saved CheXpert/NegBio revisions differ from approved contract')
    return dict(zip(keys, labeled)), {**revisions, 'label_input_sha256': digest(input_path),
                                        'label_output_sha256': digest(output_path), 'reused_saved_output': True}


def _value(value):
    value = str(value).strip()
    return None if not value else int(float(value))


def clinical(indices, arm, labels, policy):
    per = {}
    for disease in PATHOLOGIES:
        pairs = []
        gt_missing = Counter()
        pred_missing = Counter()
        for i in indices:
            gt = _value(labels[(i, 'reference')][disease])
            pred = _value(labels[(i, arm)][disease])
            gt_missing['blank' if gt is None else 'uncertain' if gt == -1 else 'definite'] += 1
            pred_missing['blank' if pred is None else 'uncertain' if pred == -1 else 'definite'] += 1
            if policy == 'historical':
                pairs.append((label(labels[(i, 'reference')][disease], 'historical'),
                              label(labels[(i, arm)][disease], 'historical')))
            else:
                pairs.append((gt if gt in (0, 1) else None, pred if pred in (0, 1) else 0))
        per[disease] = {**enrich(confusion(pairs), len(indices)),
                        'gt_blank': gt_missing['blank'], 'gt_uncertain': gt_missing['uncertain'],
                        'pred_blank': pred_missing['blank'], 'pred_uncertain': pred_missing['uncertain']}
    return {'per_class': per, **summarize(per), 'positive_support': sum(x['support'] for x in per.values()),
            'definite_label_coverage': sum(x['paired'] for x in per.values()),
            'total_class_cases': len(indices) * len(PATHOLOGIES)}


def error_proxies(records, labels, indices):
    neg = re.compile(r'\b(?:no|without|absence of|not seen|not identified|no evidence of)\b', re.I)
    words = {'Cardiomegaly': r'cardiomegal\w*|enlarged heart', 'Consolidation': r'consolidat\w*',
             'Edema': r'\bedema\b', 'Lung Opacity': r'\bopacit\w*\b|\binfiltrat\w*\b',
             'Pleural Effusion': r'pleural effusion', 'Pneumonia': r'pneumonia',
             'Pneumothorax': r'pneumothorax'}
    output = {}
    for arm in ('M0', 'M1'):
        counts = Counter()
        for i in indices:
            report = records[i][arm]['generated_report']
            reference = records[i]['reference']
            for disease in PATHOLOGIES:
                gt = _value(labels[(i, 'reference')][disease])
                pred = _value(labels[(i, arm)][disease])
                if gt == 0 and pred == 1:
                    counts['unsupported_positive_labeler_candidate'] += 1
                if gt == 1 and pred != 1:
                    counts['omission_labeler_candidate'] += 1
                terms = list(re.finditer(words[disease], report, re.I))
                if gt == 1 and terms and any(neg.search(report[max(0, m.start()-25):m.start()]) for m in terms):
                    counts['incorrect_negation_cue_candidate'] += 1
                if len(terms) >= 2:
                    polarities = [bool(neg.search(report[max(0, m.start()-25):m.start()])) for m in terms]
                    if len(set(polarities)) > 1:
                        counts['within_report_polarity_conflict_candidate'] += 1
            if len(report.split()) <= 12 and len(reference.split()) >= 25:
                counts['short_generic_candidate'] += 1
        output['P4_M1' if arm == 'M0' else 'P5_M3'] = dict(counts)
    return {'classification': 'text/labeler-derived candidates; no image or radiologist adjudication',
            'counts_by_arm': output, 'denominator_studies': len(indices)}


def repetition_fraction(report):
    words = tokenize(report)
    grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--manifest-index', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    cfg, index = contract_and_index(a.config, a.manifest_index)
    if a.dry_run:
        print(json.dumps({'status': 'DRY_RUN_PASS', 'primary': 'M3 minus P4 M1 paired corpus BLEU-4',
                          'bootstrap': 1000, 'seed': 42, 'gpu_used': False}))
        return
    stage = json.loads((a.run_dir / 'P5.3_status.json').read_text())
    if stage['status'] not in ('PASS', 'PARTIAL') or stage['validity_gate'] != 'PASS':
        raise RuntimeError('P5.3 output validity gate not passed')
    new_path = a.run_dir / 'P5.3_predictions.jsonl'
    new = [json.loads(x) for x in new_path.read_text().splitlines()]
    if not new or len(new) != stage['completed_generations'] or len(new) > 1000:
        raise ValueError('P5 output count/status mismatch')
    p4run = VR / 'runs/P4_projection_pilot_20261023'
    p4cfg = VR / 'configs/experiments/P4_projection_contract.approved.json'
    p4index = p4run / 'manifest_index.approved.json'
    old = read_inputs(p4cfg, json.loads(p4index.read_text()), p4run)
    old_pairs = [json.loads(x) for x in (p4run / 'P4.3_pairs.jsonl').read_text().splitlines()]
    records = []
    for i, entry in enumerate(new):
        row = old[i]
        if (entry['model_id'] != 'M3' or entry['study_id'] != row['study_id'] or entry['image_id'] != row['image_id']
                or tuple(entry['stable_key']) != tuple(old_pairs[i]['stable_key'])
                or entry['config_sha256'] != digest(a.config) or not entry['generated_report'].strip()):
            raise ValueError('P5/P4 study-image-configuration match failed')
        records.append({**row, 'M0': row['M1'], 'M1': entry})  # internal evaluator alias: M0=P4 M1, M1=P5 M3
    out = a.run_dir / 'P5.3_cpu_analysis'
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (out / 'metrics.json').exists():
        raise FileExistsError('P5 analysis result already exists')
    p4_config = json.loads(p4cfg.read_text())
    cached = existing_labels(records, out, p4_config)
    labeled, provenance = cached if cached is not None else label_texts(records, out, p4_config)
    legacy_site = Path('/home/student4/anaconda3/envs/chexpert-label/lib/python3.6/site-packages')
    wordnet = VR / '.cache/nltk/corpora/wordnet.zip'
    if not (legacy_site / 'nltk').is_dir() or digest(wordnet) != 'cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59':
        raise RuntimeError('Pinned P4 NLTK/WordNet runtime unavailable')
    sys.path.append(str(legacy_site))
    os.environ['NLTK_DATA'] = str(VR / '.cache/nltk')
    import nltk
    if nltk.__version__ != '3.3':
        raise RuntimeError('P4 NLTK version mismatch')
    stats = {arm: {i: case_statistics(row['reference'], row[arm]['generated_report'])
                   for i, row in enumerate(records)} for arm in ('M0', 'M1')}
    meteor = {arm: dict(enumerate(score_pairs((row['reference'], row[arm]['generated_report'])
                          for row in records))) for arm in ('M0', 'M1')}
    result = {'status': 'PASS' if len(records) == 1000 and stage['status'] == 'PASS' else 'PARTIAL',
              'classification': 'paired research evaluation; no clinical validation',
              'comparison': 'P5 M3 minus P4 M1', 'completed_pairs': len(records),
              'input_sha256': {'p5_predictions': digest(new_path), 'p4_pairs': digest(p4run / 'P4.3_pairs.jsonl'),
                               'p5_config': digest(a.config)}, 'labeler': provenance,
              'meteor_runtime': {'nltk_version': nltk.__version__, 'wordnet_sha256': digest(wordnet),
                                 'scorer_sha256': digest(VR / 'src/vlm_medical/eval/vendor/nltk367_meteor_score.py')},
              'bootstrap': {'resamples': 1000, 'seed': 42, 'interval': 'paired percentile 95%'}, 'datasets': {}}
    errors = {'status': result['status'], 'datasets': {}}
    for dataset in ('mimic', 'indiana'):
        ids = [i for i, row in enumerate(records) if row['cohort'] == dataset]
        if not ids:
            result['datasets'][dataset] = {'status': 'NO_PAIRED_CASES'}
            continue
        clusters = defaultdict(list)
        for i in ids:
            key = records[i]['subject_id'] if dataset == 'mimic' else records[i]['study_id']
            clusters[key].append(i)
        def point(sample, arm):
            text = corpus_scores([stats[arm][i] for i in sample])
            text['meteor'] = sum(meteor[arm][i] for i in sample) / len(sample)
            return text
        rng = random.Random(42)
        keys = list(clusters)
        draws = []
        for _ in range(1000):
            sample = [i for _ in keys for i in clusters[rng.choice(keys)]]
            draws.append(point(sample, 'M1')['bleu4'] - point(sample, 'M0')['bleu4'])
        base = point(ids, 'M0'); future = point(ids, 'M1')
        out_models = {}
        for arm, name in [('M0', 'P4_M1'), ('M1', 'P5_M3')]:
            out_models[name] = {'text': base if arm == 'M0' else future,
                                'clinical_historical': clinical(ids, arm, labeled, 'historical'),
                                'clinical_gt_definite_pred_missing_negative': clinical(ids, arm, labeled, 'alternative'),
                                'mean_length_tokens': sum(len(records[i][arm]['generated_report'].split()) for i in ids) / len(ids),
                                'mean_repeated_fourgram_fraction': sum(repetition_fraction(records[i][arm]['generated_report']) for i in ids) / len(ids),
                                'unique_report_texts': len({records[i][arm]['generated_report'] for i in ids}),
                                'hit_max_length_count': sum(records[i][arm].get('hit_max_length', False) for i in ids),
                                'empty_count': sum(not records[i][arm]['generated_report'].strip() for i in ids),
                                'mean_latency_seconds': sum(records[i][arm].get('latency_seconds', 0) for i in ids) / len(ids)}
        f1_delta = {}
        for policy in ('clinical_historical', 'clinical_gt_definite_pred_missing_negative'):
            f1_delta[policy] = {}
            for averaging in ('micro_f1', 'macro_f1'):
                previous = out_models['P4_M1'][policy][averaging]
                current = out_models['P5_M3'][policy][averaging]
                f1_delta[policy][averaging] = None if previous is None or current is None else current - previous
        result['datasets'][dataset] = {'status': 'PASS' if len(ids) == 500 else 'PARTIAL',
                                       'paired_studies': len(ids), 'bootstrap_clusters': len(keys),
                                       'models': out_models,
                                       'clinical_f1_delta_m3_minus_m1': f1_delta,
                                       'm3_minus_m1_bleu4': {'value': future['bleu4'] - base['bleu4'], 'ci95': interval(draws)}}
        errors['datasets'][dataset] = error_proxies(records, labeled, ids)
    atomic_json(out / 'metrics.json', result)
    atomic_json(out / 'error_pairing.json', errors)
    atomic_json(out / 'status.json', {'status': result['status'], 'paired_studies': len(records)})
    print(json.dumps({'status': result['status'], 'paired_studies': len(records), 'metrics': str(out / 'metrics.json')}))


if __name__ == '__main__':
    main()
