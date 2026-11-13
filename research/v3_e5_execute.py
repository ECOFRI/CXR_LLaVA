"""Explicitly authorized E5: integrity preflight then unchanged original implementation."""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from research.common import ROOT, RESEARCH, now, sha, write_json, write_csv
from research.v3_cached import load_dev, folds_for, LABELS, e5, inside
from research.v3_e5_fit_audit import FitAudit


def preflight(output, original, profile):
    assert json.loads((profile / 'results.json').read_text())['gate'] == 'E5_RESOURCE_GATE_ELIGIBLE'
    estimate = json.loads((profile / 'results.json').read_text())
    assert estimate['estimated_E5_wall_sec'] <= 7200 and not estimate['E5_modeling_protocol_changed']
    source_expected = json.loads((profile / 'run_manifest.json').read_text())['source_sha256']['reused_v3_cached.py']
    assert sha(RESEARCH / 'v3_cached.py') == source_expected, 'Original E5 implementation changed'
    original_inputs = json.loads((original / 'input_manifest.json').read_text())
    rows, features, groups, y, nt, alignment, inputs = load_dev()
    assert inputs == original_inputs
    outer, inner = folds_for(groups[:nt])
    saved = {r['image_id']: r for r in csv.DictReader((original / 'fold_manifest.csv').open())}
    support = []
    for k, (fit, held) in enumerate(outer):
        assert not set(groups[fit]) & set(groups[held])
        for i in held: assert int(saved[rows[i]['image_id']]['outer_fold']) == k
        assert set(inner[k][0]) | set(inner[k][1]) == set(fit)
    for k, (fit, select) in enumerate(inner):
        assert not set(groups[fit]) & set(groups[select])
        for role, ix in [('fit', fit), ('select', select)]:
            for i in ix: assert saved[rows[i]['image_id']][f'inner{k}_role'] == role
    partitions = [(f'outer{k}_{role}', ix) for k, pair in enumerate(outer) for role, ix in zip(('fit', 'held'), pair)] + [(f'inner{k}_{role}', ix) for k, pair in enumerate(inner) for role, ix in zip(('fit', 'select'), pair)] + [('train', np.arange(nt)), ('validation_DEV', np.arange(nt, len(rows)))]
    for partition, ix in partitions:
        for j, finding in enumerate(LABELS):
            positive = y[ix, j] == 1
            assert positive.any() and (~positive).any(), f'Undefined support {partition}/{finding}'
            support.append(dict(partition=partition, finding=finding, images=len(ix), patients=len(set(groups[ix])), positive=int(positive.sum()), negative=int((~positive).sum()), positive_patients=len(set(groups[ix][positive]))))
    memory = next(int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')) / 1024**2
    cpus = len(os.sched_getaffinity(0)); assert cpus >= 4 and memory >= 2
    reserve = output / 'allocation_probe.bin'
    with reserve.open('wb') as stream: os.posix_fallocate(stream.fileno(), 0, 128 * 1024**2)
    reserve.unlink()
    for name in ['e5_protocol.yaml', 'fold_manifest.csv', 'dev_sample_alignment.csv']:
        shutil.copyfile(original / name, output / name)
    assert sha(output / 'e5_protocol.yaml') == json.loads((profile / 'profiling_protocol.json').read_text())['original_E5_protocol_sha256']
    write_csv(output / 'class_support_recheck.csv', support)
    write_json(output / 'input_manifest.json', inputs)
    write_json(output / 'resource_estimates.json', {'GO': True, 'E5_estimated_wall_sec': estimate['estimated_E5_wall_sec'],
               'approved_wall_cap_sec': 7200, 'RSS_cap_GiB': 8, 'expected_RSS_GiB': 2,
               'expected_output_MiB': 80, 'available_RAM_GiB': memory,
               'CPU_affinity_count': cpus, 'load_average': os.getloadavg(),
               'free_disk_GiB': shutil.disk_usage(ROOT).free / 1024**3,
               'allocation_probe_MiB': 128, 'quota_note': 'Numeric quota unverified; bounded allocation PASS',
               'profile_path': str(profile), 'profile_results_sha256': sha(profile / 'results.json'),
               'fit_count': 780, 'source_sha256': source_expected,
               'train_images': nt, 'train_patients': 1222, 'validation_DEV_images': 344,
               'protocols': {'e5': sha(output / 'e5_protocol.yaml')},
               'original_test_evaluation': False})
    print('E5_INTEGRITY_AND_RESOURCE_GATE_PASS', flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['preflight', 'run'], required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--gate', required=True); parser.add_argument('--profile')
    args = parser.parse_args(); output, gate = inside(args.output), inside(args.gate)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output.mkdir(parents=True, exist_ok=True); assert not (output / 'results.json').exists()
    with threadpool_limits(limits=4):
        if args.phase == 'preflight': preflight(output, gate, inside(args.profile))
        else:
            m = json.loads((gate / 'run_manifest.json').read_text())
            assert m['status'] == 'COMPLETE' and m['phase'] == 'preflight'
            data = json.loads((gate / 'resource_estimates.json').read_text())
            assert data['GO'] and sha(RESEARCH / 'v3_cached.py') == data['source_sha256']
            assert sha(gate / 'e5_protocol.yaml') == m['protocols']['e5']
            # Calls original E5 unchanged: no C/grid/fold/arm/metric policy overrides.
            with FitAudit(output) as audit:
                e5(output, gate)
            assert audit.completed == audit.started == 780, 'Actual fit count differs from protocol'
    assert not any(k in sys.modules for k in ('torch', 'cupy', 'transformers', 'jax'))
    print('CPU_ONLY_E5_PHASE_COMPLETE', args.phase, flush=True)


if __name__ == '__main__': main()
