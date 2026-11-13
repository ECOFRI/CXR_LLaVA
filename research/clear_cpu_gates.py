"""Approved nested 10 -> 50 CPU gates on the exact prior CXR image identities."""
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research.common import BASELINE, RESEARCH, ROOT, finish_run, new_run, pseudonym, sha, write_csv, write_json

RAM_BUDGET = 8 * 1024**3


def budget_check():
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if rss_peak >= RAM_BUDGET:
        raise RuntimeError('CPU RSS resource budget exceeded (8 GiB)')
    return rss_peak


def run():
    wall = time.perf_counter()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert torch.__version__ == '2.3.1+cpu' and torch.version.cuda is None
    assert Path(sys.prefix).resolve() == (RESEARCH / 'envs/cxr-clear-cpu').resolve()
    state = json.loads((RESEARCH / 'research_state.json').read_text())
    cxr_run = Path(state['runs']['E0A_CXR'])
    prior = json.loads((Path(state['runs']['E0A_CLEAR_CPU']) / 'run_manifest.json').read_text())
    cxr = json.loads((cxr_run / 'run_manifest.json').read_text())
    import csv
    with (cxr_run / 'feature_manifest.csv').open() as stream:
        cxr_features = list(csv.DictReader(stream))
    assert len(cxr_features) == len(cxr['inputs']) == 50
    assert sha(BASELINE / 'cohort.json') == cxr['input_cohort_sha256'] == prior['input_cohort_sha256']
    lookup = {r['index']: r for r in json.loads((BASELINE / 'cohort.json').read_text())}
    cohort = []
    for source, feature in zip(cxr['inputs'], cxr_features):
        row = lookup[source['index']]
        assert row['image_path'] == source['image_path'] and row['image_sha256'] == source['image_sha256']
        assert pseudonym(row['relative_path']) == feature['image_id'] and pseudonym(row['study']) == feature['study_id']
        cohort.append(row)
    assert len({r['image_sha256'] for r in cohort}) == 50
    run, manifest = new_run('E0A_CLEAR_CPU_10_50')
    manifest.update(env=sys.prefix, command=[sys.executable, '-m', 'research.clear_cpu_gates'],
                    dataset='CheXpert', split='locked_test_technical_only', precision='FP32',
                    estimated_gpu_hours=0, actual_gpu_hours=0, device_map={'CLEAR': 'cpu'},
                    approval='Reuse cxr-clear-cpu; 10 first then 50 if clean; no installs/training',
                    input_cohort_sha256=sha(BASELINE / 'cohort.json'), cxr_run=str(cxr_run),
                    cxr_manifest_sha256=sha(cxr_run / 'feature_manifest.csv'), runner_sha256=sha(__file__),
                    warmup_policy='3 untimed forwards of first paired image before all measured samples',
                    timing_scope='native backbone forward_features plus visual.projection; preprocessing, validation, repeated/API forwards and cache IO excluded',
                    cpu_threads=4, ram_budget_bytes=RAM_BUDGET, consistency_tolerance=1e-6,
                    cache_fields=['native_cls', 'projected_global', 'native_patches'])
    write_json(run / 'paired_input_manifest.json', [{'image_id': pseudonym(r['relative_path']),
               'study_id': pseudonym(r['study']), 'image_path': r['image_path'], 'image_sha256': r['image_sha256'],
               'cohort_index': r['index']} for r in cohort])
    try:
        # Bound address space as well as measuring RSS, so runaway allocations fail early.
        resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET, RAM_BUDGET))
        budget_check()
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        clear_repo = ROOT.parent / 'CLEAR'
        dino = RESEARCH / 'upstream/dinov2'
        assert sha(clear_repo / 'src/clear/hub.py') == prior['clear_source_sha256']
        assert sha(dino / 'dinov2/layers/attention.py') == prior['dinov2_attention_sha256']
        sys.path.insert(0, str(clear_repo / 'src'))
        torch.hub.set_dir(str(RESEARCH / 'cache/torch_hub'))
        from clear import load_pretrained
        checkpoint = Path('/home/student4/.cache/huggingface/hub/models--peterhan91--CLEAR/snapshots') / prior['checkpoint'] / 'best_model.pt'
        assert sha(checkpoint) == prior['checkpoint_sha256']
        t = time.perf_counter()
        model, preprocess = load_pretrained(checkpoint_path=checkpoint, device='cpu', local_files_only=True, strict=True)
        model = model.float().eval().requires_grad_(False)
        load_sec = time.perf_counter() - t
        assert all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters())
        budget_check()
        config = {'checkpoint': prior['checkpoint'], 'checkpoint_sha256': prior['checkpoint_sha256'],
                  'source': prior['clear_source_sha256'], 'dino_source': prior['dinov2_attention_sha256'],
                  'preprocess': repr(preprocess), 'precision': 'FP32',
                  'image_sha256': [r['image_sha256'] for r in cohort], 'fields': manifest['cache_fields']}
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
        cache = RESEARCH / 'cache' / ('clear_paired_' + key) / run.name
        cache.mkdir(parents=True, exist_ok=False)
        manifest.update(**{k: config[k] for k in ['checkpoint', 'checkpoint_sha256', 'preprocess']},
                        clear_source_sha256=prior['clear_source_sha256'],
                        dinov2_attention_sha256=prior['dinov2_attention_sha256'], cache_key=key, outputs=[str(cache)])
        write_json(run / 'run_manifest.json', manifest)

        def pixels(row):
            assert sha(row['image_path']) == row['image_sha256']
            with Image.open(row['image_path']) as im:
                x = preprocess(im).unsqueeze(0).float()
            assert x.shape == (1, 3, 448, 448) and torch.isfinite(x).all()
            return x

        def extract(x):
            f = model.visual.backbone.forward_features(x)
            assert f['x_norm_clstoken'].shape == (1, 768)
            assert f['x_norm_regtokens'].shape == (1, 4, 768)
            assert f['x_norm_patchtokens'].shape == (1, 1024, 768)
            assert model.visual.backbone.num_register_tokens == 4
            assert all(torch.isfinite(f[k]).all() for k in ['x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens'])
            return {'native_cls': f['x_norm_clstoken'], 'native_patches': f['x_norm_patchtokens'],
                    'projected_global': model.visual.projection(f['x_norm_clstoken'])}

        x = pixels(cohort[0])
        warmup_start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(3):
                warm = extract(x)
                budget_check()
        warmup_sec = time.perf_counter() - warmup_start
        del warm, x
        rows, gates = [], []
        gate_start = time.perf_counter()
        for i, source in enumerate(cohort):
            x = pixels(source)
            with torch.inference_mode():
                # Check shapes/finiteness outside the timed forward path.
                start = time.perf_counter()
                f = model.visual.backbone.forward_features(x)
                projected = model.visual.projection(f['x_norm_clstoken'])
                latency = time.perf_counter() - start
                out = {'native_cls': f['x_norm_clstoken'], 'native_patches': f['x_norm_patchtokens'], 'projected_global': projected}
                assert f['x_norm_regtokens'].shape == (1, 4, 768)
                assert {k: list(v.shape) for k, v in out.items()} == {'native_cls': [1, 768], 'native_patches': [1, 1024, 768], 'projected_global': [1, 768]}
                assert all(torch.isfinite(v).all() for v in out.values()) and torch.isfinite(f['x_norm_regtokens']).all()
                repeat = extract(x)
                repeat_delta = max(float((out[k] - repeat[k]).abs().max()) for k in out)
                api_delta = float((projected - model.encode_image(x)).abs().max())
                assert repeat_delta <= 1e-6 and api_delta <= 1e-6
            rss = budget_check()
            path = cache / f'{i:05d}.pt'
            item = {'image_id': pseudonym(source['relative_path']), 'study_id': pseudonym(source['study']), 'features': out}
            torch.save(item, path)
            rows.append({'paired_index': i, 'image_id': item['image_id'], 'study_id': item['study_id'],
                         'latency_after_warmup_sec': latency, 'peak_rss_bytes': rss,
                         'repeat_max_abs_diff': repeat_delta, 'official_api_max_abs_diff': api_delta,
                         'feature_shapes': json.dumps({k: list(v.shape) for k, v in out.items()}),
                         'register_shape_verified': '[1,4,768]', 'cache_path': str(path),
                         'cache_sha256': sha(path), 'cache_bytes': path.stat().st_size})
            write_csv(run / 'feature_manifest.csv', rows)
            del f, out, repeat, projected, item, x
            if i + 1 in (10, 50):
                assert sha(checkpoint) == prior['checkpoint_sha256']
                assert sha(clear_repo / 'src/clear/hub.py') == prior['clear_source_sha256']
                budget_check()
                latencies = [r['latency_after_warmup_sec'] for r in rows]
                gates.append({'n': i + 1, 'status': 'PASS', 'mean_latency_after_warmup_sec': float(np.mean(latencies)),
                              'median_latency_sec': float(np.median(latencies)), 'p95_latency_sec': float(np.percentile(latencies, 95)),
                              'wall_since_first_measured_image_sec': time.perf_counter() - gate_start,
                              'peak_rss_bytes': budget_check(), 'checkpoint_integrity': 'PASS',
                              'max_repeat_abs_diff': max(r['repeat_max_abs_diff'] for r in rows),
                              'max_official_api_abs_diff': max(r['official_api_max_abs_diff'] for r in rows)})
                write_json(run / 'gates.json', gates)
                print(json.dumps(gates[-1]), flush=True)
        assert not torch.cuda.is_initialized()
        metrics = {'n': 50, 'gates': gates, 'mean_latency_after_warmup_sec': gates[-1]['mean_latency_after_warmup_sec'],
                   'median_latency_sec': gates[-1]['median_latency_sec'], 'p95_latency_sec': gates[-1]['p95_latency_sec'],
                   'warmup_forwards': 3, 'warmup_sec': warmup_sec, 'load_wall_sec': load_sec,
                   'wall_sec': time.perf_counter() - wall, 'peak_ram_bytes': budget_check(), 'ram_budget_bytes': RAM_BUDGET,
                   'feature_shapes': json.loads(rows[0]['feature_shapes']), 'register_tokens': 4,
                   'max_repeat_abs_diff': gates[-1]['max_repeat_abs_diff'], 'max_official_api_abs_diff': gates[-1]['max_official_api_abs_diff'],
                   'cache_bytes': sum(r['cache_bytes'] for r in rows), 'checkpoint_integrity': 'PASS',
                   'paired_ids_verified': 50, 'cuda_initialized': False, 'actual_gpu_hours': 0,
                   'timing_scope': manifest['timing_scope']}
        write_json(run / 'metrics.json', metrics)
        manifest.update(next_action='Review paired representation harness; independent data gate before probes',
                        metric_summary=metrics)
        finish_run(run, manifest)
        print(str(run), flush=True)
    except (RuntimeError, MemoryError, OSError, AssertionError, ImportError, ValueError) as exc:
        (run / 'stderr.log').write_text(traceback.format_exc())
        manifest['blocker'] = str(exc)
        finish_run(run, manifest, 'FAILED')
        raise


if __name__ == '__main__':
    run()
