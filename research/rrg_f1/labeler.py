"""Pinned P4 labeler on local Findings; at most 16 CPU subprocesses."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import resource
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from research.rrg_f0.core import ROOT, sha256
from research.rrg_f1.cohort import BASE, PRIVATE, atomic_json
from research.blip2_cxr_v4.src.vlm_medical.data.manifests import report_sections

LABELS = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
          'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
WORKERS = 16



def runtime_env():
    library = str(ROOT / 'model_cache/rrg_f1/legacy_libs/lib')
    temporary = PRIVATE / 'tmp'
    temporary.mkdir(parents=True, exist_ok=True, mode=0o700)
    return {**os.environ, 'TMPDIR': str(temporary), 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(ROOT / 'model_cache/chexpert_tools/NegBio'),
            'NLTK_DATA': str(ROOT / 'research/blip2_cxr_v4/.cache/nltk'), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1',
            'LD_LIBRARY_PATH': library + (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')}


def four_state(value):
    if value is None or str(value).strip() == '':
        return -2  # blank retained, never silently merged in the raw labels
    number = float(value)
    if number not in (-1, 0, 1):
        raise ValueError('Invalid four-state label')
    return int(number)


def load_text(row):
    raw = Path(row['reference_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['reference_file_sha256']:
        raise ValueError('Training reference changed')
    text = raw.decode(errors='replace').replace('\r\n', '\n').replace('\r', '\n')
    target = report_sections(text).get('findings', '')
    if not target or hashlib.sha256(target.encode()).hexdigest() != row['findings_sha256']:
        raise ValueError('Training Findings target changed')
    return target


def label_chunk(item):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    texts, directory = item
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    input_file, output_file = directory / 'input.csv', directory / 'output.csv'
    digest = hashlib.sha256(json.dumps(texts).encode()).hexdigest()
    done = directory / 'done.json'
    if done.exists():
        info = json.loads(done.read_text())
        if info['input_sha256'] != digest or sha256(output_file) != info['output_sha256']:
            raise ValueError('Labeler resume output changed')
    else:
        with input_file.open('w', newline='') as stream:
            csv.writer(stream).writerows([[t] for t in texts])
        local = json.loads((BASE / 'paths.local.json').read_text())
        tool = ROOT / 'model_cache/chexpert_tools/chexpert-labeler'
        env = runtime_env()
        start = time.monotonic()
        with (directory / 'worker.log').open('w') as log:
            affinity = ','.join(str(c) for c in sorted(os.sched_getaffinity(0))[:WORKERS])
            subprocess.run(['taskset', '-c', affinity, local['labeler_python'], str(tool / 'label.py'), '--reports_path', str(input_file.resolve()),
                            '--output_path', str(output_file.resolve())], cwd=tool, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=7200)
        atomic_json(done, {'input_sha256': digest, 'output_sha256': sha256(output_file), 'rows': len(texts),
                           'wall_seconds': time.monotonic() - start})
    with output_file.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(texts) or any(row['Reports'] != t for row, t in zip(rows, texts)):
        raise ValueError('Labeler text/order alignment mismatch')
    return [[four_state(row[name]) for name in LABELS] for row in rows]


def label_texts(texts, directory, chunk_size=500):
    jobs = [(texts[i:i + chunk_size], directory / f'chunk_{i:06d}') for i in range(0, len(texts), chunk_size)]
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        result = []
        for chunk in executor.map(label_chunk, jobs):
            result.extend(chunk)
        return result


def train_targets():
    state = json.loads((PRIVATE / 'cohort_state.json').read_text())
    if state['status'] != 'COMPLETE' or sha256(PRIVATE / 'train_manifest.jsonl') != state['manifest_sha256']:
        raise ValueError('Cohort gate is incomplete')
    rows = [json.loads(line) for line in (PRIVATE / 'train_manifest.jsonl').read_text().splitlines()]
    path = PRIVATE / 'train_texts.json'
    if path.exists():
        texts = json.loads(path.read_text())
        if len(texts) != len(rows) or any(hashlib.sha256(t.encode()).hexdigest() != r['findings_sha256'] for t, r in zip(texts, rows)):
            raise ValueError('Target cache linkage changed')
    else:
        with ThreadPoolExecutor(max_workers=WORKERS) as workers:
            texts = list(workers.map(load_text, rows))
        atomic_json(path, texts)
    return rows, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['benchmark', 'train'], required=True)
    args = parser.parse_args()
    os.umask(0o077)
    config = json.loads((BASE / 'config.json').read_text())
    rows, texts = train_targets()
    start = time.monotonic()
    if args.stage == 'benchmark':
        labels = label_texts(texts[:200], PRIVATE / 'label_benchmark', chunk_size=math.ceil(200 / WORKERS))
        elapsed = time.monotonic() - start
        projected = elapsed * len(texts) / 200
        summary = {'status': 'PASS' if projected <= config['labeler_estimate_seconds_limit'] else 'STOP_LABELER_ESTIMATE_GT_8H',
                   'reports': len(labels), 'workers': WORKERS, 'wall_seconds': elapsed, 'projected_30000_seconds': projected,
                   'projection': 'conservative linear scaling of 200-report wall including concurrent parser startup', 'gpu_seconds': 0}
        atomic_json(BASE / 'label_benchmark_summary.json', summary)
        if summary['status'] != 'PASS':
            raise RuntimeError('STOP_LABELER_ESTIMATE_GT_8H')
        print(json.dumps(summary), flush=True)
        return
    benchmark = json.loads((BASE / 'label_benchmark_summary.json').read_text())
    if benchmark['status'] != 'PASS':
        raise ValueError('Labeler 200-report budget gate did not PASS')
    labels = label_texts(texts, PRIVATE / 'train_label_chunks')
    atomic_json(PRIVATE / 'train_labels.json', labels)
    counts = {name: {str(s): sum(row[j] == s for row in labels) for s in (-2, -1, 0, 1)} for j, name in enumerate(LABELS)}
    prior = [(counts[n]['1'] + 1) / (len(labels) - counts[n]['-1'] + 2) for n in LABELS]
    local = json.loads((BASE / 'paths.local.json').read_text())
    expected = {(r['subject_id'], r['study_id']): i for i, r in enumerate(rows[:1000])}
    paired = []
    official_path = Path(local['jpg_root']) / 'mimic-cxr-2.0.0-chexpert.csv.gz'
    with gzip.open(official_path, 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['subject_id'], row['study_id'])
            if key in expected:
                paired.append((labels[expected[key]], [four_state(row[n]) for n in LABELS]))
    if len(paired) != 1000:
        raise ValueError('STOP_OFFICIAL_LABEL_AUDIT_COHORT_MISSING')
    agreement = {n: sum(a[j] == b[j] for a, b in paired) / len(paired) for j, n in enumerate(LABELS)}
    summary = {'status': 'PASS', 'rows': len(labels), 'workers': WORKERS, 'wall_seconds': time.monotonic() - start,
               'labels': LABELS, 'four_state_counts': counts, 'smoothed_positive_prior': prior,
               'bce_mapping': 'positive1; negative/blank0; uncertain masked; name-aligned columns',
               'official_full_report_audit': {'studies': 1000, 'per_class_four_state_agreement': agreement,
                   'overall_four_state_agreement': sum(agreement.values()) / 14,
                   'interpretation': 'Findings-only vs official full-report labels; audit only, not replacement'},
               'labeler_revision': config['labeler_revision'], 'negbio_revision': config['negbio_revision'],
               'labels_sha256': sha256(PRIVATE / 'train_labels.json'), 'gpu_seconds': 0}
    atomic_json(BASE / 'train_label_summary.json', summary)
    print(json.dumps({'status': 'PASS', 'train_labels': len(labels), 'wall_seconds': summary['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
