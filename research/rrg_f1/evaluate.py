"""P4-compatible fixed-dev evaluation; execute in restored legacy CPU labeler env."""
import json
import multiprocessing
import os
import random
import time
from collections import defaultdict

from research.rrg_f0.core import ROOT, sha256
from research.rrg_f1.cohort import BASE, PRIVATE, dev_rows, atomic_json
from research.rrg_f1.labeler import LABELS, label_texts
from research.blip2_cxr_v4.scripts.eval import p4_evaluate as p4

G = None
POLICIES = ('historical', 'blank_as_negative')


def initialize(payload):
    global G
    G = payload


def bootstrap_one(job):
    dataset, model, indices = job
    records, stats, meteor, labeled = G
    by_cluster = defaultdict(list)
    for i in indices:
        key = records[i]['subject_id'] if dataset == 'mimic' else records[i]['study_id']
        by_cluster[key].append(i)
    clusters = list(by_cluster)
    rng = random.Random(42)
    values = {key: [] for key in p4.TEXT_KEYS}
    for policy in POLICIES:
        for metric in ('micro_f1', 'macro_f1'):
            values[policy + '_' + metric] = []
    for _ in range(1000):
        selected = [i for _ in clusters for i in by_cluster[rng.choice(clusters)]]
        text = p4.point_text(records, model, selected, meteor, stats)
        for key in p4.TEXT_KEYS:
            values[key].append(text[key])
        for policy in POLICIES:
            clinical = p4.point_clinical(selected, model, labeled, policy)
            for metric in ('micro_f1', 'macro_f1'):
                values[policy + '_' + metric].append(clinical[metric])
    return dataset, model, values, len(clusters)


def main():
    os.umask(0o077)
    started = time.monotonic()
    path = BASE / 'metrics.json'
    if path.exists():
        print('METRICS_ALREADY_COMPLETE', flush=True)
        return
    config_path = ROOT / 'research/blip2_cxr_v4/configs/experiments/P4_projection_contract.approved.json'
    config = json.loads(config_path.read_text())
    index = json.loads((ROOT / 'research/blip2_cxr_v4/runs/P4_projection_pilot_20261023/manifest_index.approved.json').read_text())
    records = p4.read_inputs(config_path, index, ROOT / 'research/blip2_cxr_v4/runs/P4_projection_pilot_20261023')
    if len(records) != 1000:
        raise ValueError('P4 matched cohort is incomplete')
    map_rows = {(family, i): (str(row['subject_id']), str(row['study_id']), str(row['image_id']))
                for family in ('mimic', 'indiana') for i, row in enumerate(dev_rows(family))}
    by_key = {(r['cohort'], str(r['subject_id']), str(r['study_id']), str(r['image_id'])): i for i, r in enumerate(records)}
    for arm in ('A2', 'A3', 'A4'):
        directory = PRIVATE / arm / 'predictions'
        for position in range(1000):
            prediction = json.loads((directory / ('%04d.json' % position)).read_text())
            family, number = prediction['family'], prediction['index']
            key = (family,) + map_rows[(family, number)]
            records[by_key[key]][arm] = prediction
    entries, texts = [], []
    labeled = {}
    models_for = {'mimic': ['M0', 'M1', 'A2', 'A3', 'A4'], 'indiana': ['M0', 'M1', 'M2', 'A2', 'A3', 'A4']}
    for i, row in enumerate(records):
        for model in ['reference'] + models_for[row['cohort']]:
            text = row['reference'] if model == 'reference' else row[model]['generated_report']
            labeled[(i, model)] = {name: '' for name in LABELS}
            if text.strip():
                entries.append((i, model))
                texts.append(text)
    labels = label_texts(texts, PRIVATE / 'evaluation_label_chunks')
    for entry, values in zip(entries, labels):
        labeled[entry] = {name: {-2: '', -1: '-1', 0: '0', 1: '1'}[value] for name, value in zip(LABELS, values)}
    atomic_json(PRIVATE / 'evaluation_labels.json', {'rows': [{'record_index': i, 'model': m, 'labels': labeled[(i, m)]} for i, m in labeled]})
    stats, meteor = {}, {}
    for model in ('M0', 'M1', 'M2', 'A2', 'A3', 'A4'):
        indices = [i for i, row in enumerate(records) if model in row]
        stats[model] = {i: p4.case_statistics(records[i]['reference'], records[i][model]['generated_report']) for i in indices}
        nonempty = [i for i in indices if records[i][model]['generated_report'].strip()]
        scores = p4.score_pairs((records[i]['reference'], records[i][model]['generated_report']) for i in nonempty)
        meteor[model] = {i: 0. for i in indices}
        meteor[model].update(dict(zip(nonempty, scores)))
    result = {'status': 'PASS', 'classification': 'metric-complete evaluation on fixed exposed dev cohorts; not test benchmark',
              'bootstrap_reps': 1000, 'bootstrap_seed': 42, 'bootstrap': 'paired patient clusters MIMIC; study clusters Indiana; unadjusted percentile95%',
              'text_cohort': 'all500 per dataset; empty generations get zero text scores and blank clinical labels; no favorable dropping',
              'clinical_ontology': list(p4.PATHOLOGIES), 'clinical_policies': list(POLICIES), 'datasets': {}}
    jobs = []
    for family in ('mimic', 'indiana'):
        indices = [i for i, row in enumerate(records) if row['cohort'] == family]
        points = {}
        for model in models_for[family]:
            text = p4.point_text(records, model, indices, meteor, stats)
            clinical = {}
            for policy in POLICIES:
                value = p4.point_clinical(indices, model, labeled, policy)
                for counts in value['per_class'].values():
                    counts['f1_denominator'] = 2 * counts['tp'] + counts['fp'] + counts['fn']
                value['micro_f1_denominator'] = sum(c['f1_denominator'] for c in value['per_class'].values())
                value['possible_class_cases'] = len(indices) * len(p4.PATHOLOGIES)
                clinical[policy] = value
            points[model] = {'n': len(indices), 'text': text, 'clinical': clinical,
                             'empty_count': sum(not records[i][model]['generated_report'].strip() for i in indices),
                             'cap_count': sum(records[i][model].get('hit_max_length', False) for i in indices)}
            jobs.append((family, model, indices))
        result['datasets'][family] = {'models': points}
    with multiprocessing.Pool(8, initializer=initialize, initargs=((records, stats, meteor, labeled),)) as pool:
        boot = pool.map(bootstrap_one, jobs)
    distributions = {}
    for family, model, values, clusters in boot:
        result['datasets'][family].setdefault('ci95', {})[model] = {key: p4.interval(v) for key, v in values.items()}
        result['datasets'][family]['bootstrap_clusters'] = clusters
        distributions[(family, model)] = values
    contrasts = [('A4', 'A2'), ('A3', 'A2'), ('A4', 'A3')] + [(arm, baseline) for arm in ('A2', 'A3', 'A4') for baseline in ('M0', 'M1')]
    for family in ('mimic', 'indiana'):
        deltas = {}
        for left, right in contrasts:
            a, b = distributions[(family, left)], distributions[(family, right)]
            deltas[left + '_minus_' + right] = {key: p4.interval([None if x is None or y is None else x - y for x, y in zip(a[key], b[key])]) for key in a}
        result['datasets'][family]['paired_delta_ci95'] = deltas
    result['wall_seconds'] = time.monotonic() - started
    result['inputs_sha256'] = {'p4_contract': sha256(config_path), 'f1_config': sha256(BASE / 'train_config.json')}
    atomic_json(path, result)
    print(json.dumps({'status': 'PASS', 'datasets': 2, 'cases_each': 500, 'wall_seconds': result['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
