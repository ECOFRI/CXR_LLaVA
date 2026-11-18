"""Local-only uniform seeded F1 train cohort, with crash-safe resume."""
import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from research.rrg_f0.core import ROOT, sha256
from research.blip2_cxr_v4.src.vlm_medical.data.manifests import report_sections

BASE = ROOT / 'research/rrg_f1'
PRIVATE = BASE / 'private'


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def dev_rows(family):
    if family not in ('mimic', 'indiana'):
        raise ValueError('Unknown dev family')
    config = json.loads((BASE / 'config.json').read_text())
    path = ROOT / config['p4_index']
    if sha256(path) != config['p4_index_sha256']:
        raise ValueError('P4 index changed')
    index = json.loads(path.read_text())
    name = 'mimic_validate' if family == 'mimic' else 'indiana_x00'
    item = index['manifests'][name]
    if sha256(ROOT / item['path']) != item['sha256']:
        raise ValueError('P4 dev manifest changed')
    rows = [json.loads(s) for s in (ROOT / item['path']).read_text().splitlines()]
    if len(rows) != 500:
        raise ValueError('P4 dev count mismatch')
    return rows


def seeded_order(keys, seed):
    order = list(keys)
    random.Random(seed).shuffle(order)
    return order


def restore_stream(stream, offset):
    """Drop uncheckpointed tail bytes; never replay a committed selection."""
    stream.seek(0, os.SEEK_END)
    if stream.tell() < offset:
        raise ValueError('Manifest is shorter than checkpoint')
    stream.truncate(offset)
    stream.seek(offset)



def publish_summary(state, config, source_hashes, candidates):
    summary = {'status': 'COMPLETE', 'train_studies': state['accepted'], 'seed': config['seed'],
               'selection': 'first eligible entries of uniformly seeded permutation of unique frontal official-train studies',
               'candidate_studies': len(candidates), 'visited_candidates': state['cursor'], 'skips': state['skips'],
               'p4_dev_counts': {'mimic': 500, 'indiana': 500}, 'train_validation_patient_overlap': 0,
               'manifest_sha256': state['manifest_sha256'], 'source_sha256': source_hashes,
               'config_sha256': state['contract']['config_sha256'], 'wall_seconds_cumulative': state['elapsed_seconds'], 'gpu_seconds': 0}
    atomic_json(BASE / 'cohort_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(exist_ok=True, mode=0o700)
    os.umask(0o077)
    with (PRIVATE / 'cohort.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((BASE / 'config.json').read_text())
        local = json.loads((BASE / 'paths.local.json').read_text())
        config_hash = sha256(BASE / 'config.json')
        if sha256(ROOT / config['p4_index']) != config['p4_index_sha256']:
            raise ValueError('P4 index changed')
        index = json.loads((ROOT / config['p4_index']).read_text())
        p4 = {}
        for name, count in [('mimic_validate', 500), ('indiana_x00', 500)]:
            entry = index['manifests'][name]
            p = ROOT / entry['path']
            if sha256(p) != entry['sha256']:
                raise ValueError('P4 dev manifest changed')
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            if len(rows) != count:
                raise ValueError('P4 dev cohort count mismatch')
            p4[name] = rows
        validation_people = {str(r['subject_id']) for r in p4['mimic_validate']}
        started = time.monotonic()
        stage = {'status': 'RUNNING', 'stage': 'cohort_sources', 'pid': os.getpid(), 'accepted': 0, 'visited': 0}
        stop_heartbeat = threading.Event()

        def heartbeat():
            while not stop_heartbeat.wait(10):
                atomic_json(PRIVATE / 'heartbeat.json', {**stage, 'unix_time': time.time(), 'current_process_seconds': time.monotonic() - started})

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            jpg_root = Path(local['jpg_root'])
            sources = {name: jpg_root / name for name in ('mimic-cxr-2.0.0-split.csv.gz', 'mimic-cxr-2.0.0-metadata.csv.gz')}
            source_hashes = {name: sha256(p) for name, p in sources.items()}
            cache = PRIVATE / 'candidates.json'
            if cache.exists():
                payload = json.loads(cache.read_text())
                if payload['source_sha256'] != source_hashes:
                    raise ValueError('Source metadata changed')
                candidates = payload['candidates']
            else:
                frontal = set()
                with gzip.open(sources['mimic-cxr-2.0.0-metadata.csv.gz'], 'rt', newline='') as stream:
                    for row in csv.DictReader(stream):
                        if row['ViewPosition'].upper() in config['view_positions']:
                            frontal.add(row['dicom_id'])
                candidates = {}
                with gzip.open(sources['mimic-cxr-2.0.0-split.csv.gz'], 'rt', newline='') as stream:
                    for row in csv.DictReader(stream):
                        if row['split'] != 'train' or row['dicom_id'] not in frontal:
                            continue
                        key = row['subject_id'] + '/' + row['study_id']
                        candidates.setdefault(key, []).append(row['dicom_id'])
                atomic_json(cache, {'source_sha256': source_hashes, 'candidates': candidates})
            order = seeded_order(candidates, config['seed'])
            contract = {'config_sha256': config_hash, 'source_sha256': source_hashes, 'candidate_sha256': sha256(cache)}
            state_path = PRIVATE / 'cohort_state.json'
            output = PRIVATE / 'train_manifest.jsonl'
            partial = PRIVATE / 'train_manifest.partial.jsonl'
            if state_path.exists():
                if not args.resume:
                    raise FileExistsError('Cohort checkpoint exists; pass --resume')
                state = json.loads(state_path.read_text())
                if state['contract'] != contract:
                    raise ValueError('Resume contract mismatch')
                if state['status'] == 'COMPLETE':
                    if sha256(output) != state['manifest_sha256']:
                        raise ValueError('Completed manifest changed')
                    publish_summary(state, config, source_hashes, candidates)
                    stage['status'] = 'COMPLETE'
                    print('COHORT_ALREADY_COMPLETE', flush=True)
                    return
            else:
                state = {'status': 'RUNNING', 'contract': contract, 'cursor': 0, 'accepted': 0, 'byte_offset': 0,
                         'skips': {}, 'elapsed_seconds': 0}
                if output.exists() or partial.exists():
                    raise FileExistsError('Orphan cohort output; preserve and inspect')
                partial.touch(mode=0o600)
            if output.exists() and not partial.exists():
                if state['accepted'] != config['train_studies'] or sha256(output) != state.get('manifest_sha256'):
                    raise ValueError('Rename recovery manifest mismatch')
                state['status'] = 'COMPLETE'
                atomic_json(state_path, state)
                publish_summary(state, config, source_hashes, candidates)
                stage['status'] = 'COMPLETE'
                print('COHORT_RENAME_RECOVERY_COMPLETE', flush=True)
                return
            skips = Counter(state['skips'])
            stage['stage'] = 'select_train_findings'
            resume_elapsed = state['elapsed_seconds']
            with partial.open('r+b') as stream:
                restore_stream(stream, state['byte_offset'])
                def inspect_candidate(cursor):
                    subject, study = order[cursor].split('/')
                    report = Path(local['report_files']) / ('p' + subject[:2]) / ('p' + subject) / ('s' + study + '.txt')
                    if not report.is_file():
                        return None, 'missing_report'
                    raw = report.read_bytes()
                    text = raw.decode(errors='replace').replace('\r\n', '\n').replace('\r', '\n')
                    findings = report_sections(text).get('findings', '')
                    if not findings.strip():
                        return None, 'empty_findings'
                    for dicom in candidates[order[cursor]]:
                        rel = Path('p' + subject[:2]) / ('p' + subject) / ('s' + study) / (dicom + '.jpg')
                        for image_root in (jpg_root / 'files', Path(local['jpg_alt_files'])):
                            image = image_root / rel
                            if image.is_file():
                                if subject in validation_people:
                                    raise ValueError('STOP_TRAIN_VALIDATION_PATIENT_OVERLAP')
                                row = {'subject_id': subject, 'study_id': study, 'dicom_id': dicom, 'split': 'train',
                                       'image_path': str(image), 'reference_path': str(report),
                                       'reference_file_sha256': hashlib.sha256(raw).hexdigest(),
                                       'findings_sha256': hashlib.sha256(findings.encode()).hexdigest()}
                                return row, None
                    return None, 'missing_image'

                # Bound lookahead to one 250-candidate block; consume in original seeded order.
                with ThreadPoolExecutor(max_workers=16) as workers:
                    for block_start in range(state['cursor'], len(order), 250):
                        cursors = range(block_start, min(block_start + 250, len(order)))
                        for cursor, (row, reason) in zip(cursors, workers.map(inspect_candidate, cursors)):
                            if row is None:
                                skips[reason] += 1
                            else:
                                stream.write((json.dumps(row) + '\n').encode())
                                state['accepted'] += 1
                            state['cursor'] = cursor + 1
                            stage.update(accepted=state['accepted'], visited=state['cursor'])
                            if state['accepted'] == config['train_studies']:
                                break
                        stream.flush()
                        os.fsync(stream.fileno())
                        state.update(byte_offset=stream.tell(), skips=dict(skips), elapsed_seconds=resume_elapsed + time.monotonic() - started)
                        if state['accepted'] == config['train_studies']:
                            state['manifest_sha256'] = sha256(partial)
                        atomic_json(state_path, state)
                        print(json.dumps({'accepted': state['accepted'], 'visited': state['cursor']}), flush=True)
                        if state['accepted'] == config['train_studies']:
                            break
                if state['accepted'] != config['train_studies']:
                    raise RuntimeError('STOP_INSUFFICIENT_ELIGIBLE_COHORT')
            partial.replace(output)
            state.update(status='COMPLETE', manifest_sha256=sha256(output), elapsed_seconds=resume_elapsed + time.monotonic() - started)
            atomic_json(state_path, state)
            summary = publish_summary(state, config, source_hashes, candidates)
            stage['status'] = 'COMPLETE'
            print(json.dumps(summary), flush=True)
        except (OSError, ValueError, RuntimeError) as error:
            stage.update(status='STOP', error_type=type(error).__name__)
            # Detailed traceback stays in the ignored detached runner log.
            raise
        finally:
            stop_heartbeat.set()
            thread.join(timeout=2)
            atomic_json(PRIVATE / 'heartbeat.json', {**stage, 'unix_time': time.time()})


if __name__ == '__main__':
    main()
